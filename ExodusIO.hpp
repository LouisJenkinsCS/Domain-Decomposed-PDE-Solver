// TPetra data structures
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Teuchos_VerboseObject.hpp>

#include "exodusII.h"
#include <string>
#include <iostream>
#include <metis.h>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <cassert>
#include <algorithm>
#include <parmetis.h>
#include <sstream>



/*
    An ExodusII wrapper that provides some C++ bindings
*/

namespace ExodusIO {

    class IO {
        public:
            IO() {}
            
            // Opens the read in exodus file
            bool open(std::string fname, bool read_only = false) {
                int cpuWS = sizeof(real_t);
                int ioWS = sizeof(real_t);
                float version = 0.;

                int tmpFid = ex_open(fname.c_str(), read_only ? EX_READ : EX_WRITE, &cpuWS, &ioWS, &version);
                if (tmpFid <= 0) {
                    perror("ex_open");
                    return false;
                }
                readFID = tmpFid;
                return true;
            }

            // Opens the writen out exodus file (modification of read in exodus file)
            bool create(std::string fname) {
                int cpuWS = sizeof(real_t);
                int ioWS = sizeof(real_t);

                int tmpFid = ex_create(fname.c_str(), EX_CLOBBER, &cpuWS, &ioWS);
                if (tmpFid <= 0) {
                    perror("ex_create");
                    return false;
                }
                writeFID = tmpFid;
                return true;
            }

            // Reads in the partitioning of the Exodus file specified in `open` and calls
            // ParMETIS to construct a dual graph; this dual graph is then partitioned
            // and redistributed/balanced across the appropriate number of processes.
            bool getMatrix(Teuchos::RCP<Tpetra::CrsMatrix<>> *ret) {
                auto comm = Tpetra::getDefaultComm();
                auto rank = Teuchos::rank(*comm);
                auto ranks = Teuchos::size(*comm);
                if (readFID == -1) return false;

                /////////////////////////////////////////////////////////////////////
                // 1. Read in Mesh from Exodus File
                /////////////////////////////////////////////////////////////////////

                // Gather all data we need to pass to ParMETIS - Each MPI Process is doing this...
                ex_init_params params;
                if (ex_get_init_ext(readFID,&params)) {
                    return false;
                }
                if (rank == 0) {
                    std::cout << "Title: " << params.title << "\n# of Dimensions: " << params.num_dim  << "\n# of Blobs: " << params.num_blob << "\n# of Assembly: " << params.num_assembly
                        << "\n# of Nodes: " << params.num_nodes << "\n# of Elements: " << params.num_elem << "\n# of Faces: " << params.num_face
                        << "\n# of Element Blocks: " << params.num_elem_blk << "\n# of Face Blocks: " << params.num_face_blk << "\n# of Node Sets: " << params.num_node_sets 
                        << "\n# of Side Sets: " << params.num_side_sets << "\n# of Face Sets: " << params.num_face_sets << "\n# of Node Maps: " << params.num_node_maps
                        << "\n# of Element Maps: " << params.num_elem_maps << "\n# of Face Maps: " << params.num_face_maps 
                        << "\n# of Bytes in idx_t: " << sizeof(idx_t) << "\n# of Bytes in real_t: " << sizeof(real_t) << std::endl;
                }
                
                int *ids = new int[params.num_elem_blk];
                for (int i = 0; i < params.num_elem_blk; i++) ids[i] = 0;
                idx_t num_elem_in_block = 0;
                idx_t num_nodes_per_elem = 0;
                idx_t num_edges_per_elem = 0;
                idx_t num_faces_per_elem = 0;
                idx_t num_attr = 0;
                char elemtype[MAX_STR_LENGTH+1];

                if (ex_get_ids(readFID, EX_ELEM_BLOCK, ids)) {
                    std::cerr << "Rank #" << Teuchos::rank(*comm) << ": " << "Failed to call `ex_get_ids`" << std::endl;
                    return false;
                }
                for (int i = 0; i < params.num_elem_blk; i++) {
                    std::cout << "Rank #" << Teuchos::rank(*comm) << ": " << "Element Block Id: " << (int) ids[i] << std::endl;
                }
                idx_t elementsIdx[params.num_elem + 1];
                for (int i = 0; i < params.num_elem + 1; i++) elementsIdx[i] = 0;
                std::vector<idx_t> nodesInElements;
                idx_t elemIdx = 0;
                idx_t nodeIdx = 0;

                // Compute current process' start index, and ignore everything before this...
                idx_t startIdx = (params.num_elem / ranks) * rank;
                idx_t endIdx = (params.num_elem / ranks) * (rank + 1);
                idx_t passedIdx = 0;
                // Handle edge case where we have odd number of elements; give the remainder
                // to the last process.
                if (rank == ranks - 1) {
                    endIdx = params.num_elem;
                }

                // using the element block parameters read the element block info
                for (idx_t i=0; i<params.num_elem_blk; i++) {
                    if (ex_get_block(readFID, EX_ELEM_BLOCK, ids[i], elemtype, &num_elem_in_block, &num_nodes_per_elem, &num_edges_per_elem, &num_faces_per_elem, &num_attr)) {
                        std::cerr << "Rank #" << rank << ": " << "Failed to `ex_get_block` element block " << i+1 << " of " << params.num_elem_blk << " with id " << ids[i] << std::endl;
                        return false;
                    }
                    if (rank == 0) {
                        std::cout << "Block #" << i << " has the following..."
                            << "\n\t# of Elements: " << num_elem_in_block
                            << "\n\t# of Nodes per Element: " << num_nodes_per_elem
                            << "\n\t# of Edges per Element: " << num_edges_per_elem
                            << "\n\t# of Faces per Element: " << num_faces_per_elem
                            << "\n\t# of Attributes: " << num_attr
                            << "\n\tElement Type: " << elemtype << std::endl;
                    }

                    idx_t *connect = new idx_t[num_elem_in_block * num_nodes_per_elem];
                    for (int i = 0; i < num_elem_in_block * num_nodes_per_elem; i++) connect[i] = 0;
                    ex_get_elem_conn(readFID, ids[i], connect);
                    // std::cout << "Block #" << i << ": {";
                    for (idx_t j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
                        // if (j) std::cout << ",";
                        if (j % num_nodes_per_elem == 0) { 
                            if (passedIdx++ < startIdx) {
                                j += num_nodes_per_elem - 1;
                                continue;
                            }
                            // std::cout << "[";
                            elementsIdx[elemIdx++] = nodeIdx;
                        }
                        // std::cout << connect[j];
                        nodesInElements.push_back(connect[j]);
                        nodeIdx++;
                        
                        // End of Element Block
                        if ((j+1) % num_nodes_per_elem == 0) {
                            if (passedIdx == endIdx) break;
                            // std::cout << "]";
                        }
                    }
                    // std::cout << "}" << std::endl;
                    delete[] connect;
                }
                elementsIdx[elemIdx] = nodeIdx;

                // TODO: Need to implement critical sections like OpenMP using MPI constructs
                std::stringstream ss;
                ss << "Process #" << rank << std::endl; 
                ss << "Indexing: {";
                for (idx_t i = 0; i < params.num_elem + 1; i++) {
                    if (i) ss << ",";
                    ss << elementsIdx[i];
                }
                ss << "}" << std::endl;
                ss << "Nodes: {";
                for (idx_t i = 0; i < nodesInElements.size(); i++) {
                    if (i) ss << ",";
                    ss << nodesInElements[i];
                }
                ss << "}";

                Teuchos::barrier(*comm);
                for (int i = 0; i < ranks; i++) {
                    if (i == rank) {
                        std::cout << ss.str() << std::endl;
                    }
                    Teuchos::barrier(*comm);
                }


                /////////////////////////////////////////////////////////////////////
                // 2. Construct Dual Graph from Mesh
                /////////////////////////////////////////////////////////////////////

                // Distributed CSR Format - Split up the element list (eind) into nparts contiguous chunks.
                // That is, it is equivalent to sequential CSR but the indexing starts at 0 (local indexing)
                // and the nodes in each element is 1/P starting at some index after the last process and before
                // the next process. Each process is reading in the Exodus file, and hence can just ignore the parts
                // of the process they do not care about.
                // TODO

                // Distribution of elements; scheme used is a simple block distribution, where
                // indices [startIdx, endIdx) contains the elements distributed over this process.
                // Each process must have the same elemdist, and so must also compute the indices for
                // all other MPI processes... TODO
                idx_t *elemdist = new idx_t[ranks+1];
                elemdist[0] = 0;
                for (int i = 1; i <= ranks; i++) {
                    elemdist[i] = elemdist[i-1] + params.num_elem / ranks;
                }
                elemdist[ranks] = params.num_elem;
                idx_t numVertices = elemdist[rank+1] - elemdist[rank];

                if (rank == 0) {
                    std::cout << "Element Distribution: {" << std::endl;
                    int pid = 0;
                    for (int i = 1; i <= ranks; i++) {
                        std::cout << "\tProcess #" << pid++ << " owns " << elemdist[i-1] << " to " << elemdist[i]-1 << std::endl;
                    }
                    std::cout << "}" << std::endl;
                }

                idx_t *eptr = elementsIdx;
                idx_t *eind = nodesInElements.data();
                idx_t numflag = 0; // 0-based indexing (C-style)
                idx_t ncommonnodes = 1;
                idx_t *xadj, *adjncy;
                MPI_Comm mpicomm = MPI_COMM_WORLD;

                // Note: We are assuming that there is only one Element Type in this mesh...
                if (strncmp(elemtype, "TETRA", 5) == 0) {
                    ncommonnodes = 3;
                } else if (strncmp(elemtype, "TRI", 3) == 0) {
                    ncommonnodes = 2;
                } else if (strncmp(elemtype, "HEX", 3) == 0) {
                    ncommonnodes = 4;
                } else {
                    std::cerr << "Currently unsupported element type for mesh: " << elemtype << std::endl;
                    return false;
                }
                // std::cout << "Calling METIS_PartMeshNodal with " << nparts << " partitions." << std::endl;
                int retval = ParMETIS_V3_Mesh2Dual(elemdist, eptr, eind, &numflag, &ncommonnodes, &xadj, &adjncy, &mpicomm);
                if (retval != METIS_OK) {
                    std::cout << "Error Code: " << (retval == METIS_ERROR_INPUT ? "METIS_ERROR_INPUT" : (retval == METIS_ERROR_MEMORY ? "METIS_ERROR_MEMORY" : "METIS_ERROR")) << std::endl;
                    return false;
                }

                ss.str("");
                ss << "Process #" << rank << std::endl;
                ss << "xadj: {";
                for (int i = 0; i <= numVertices; i++) {
                    if (i) ss << ",";
                    ss << xadj[i];
                }
                ss << "}" << std::endl;
                ss << "adjncy: {";
                for (int i = 0; i < xadj[numVertices - 1]; i++) {
                    if (i) ss << ",";
                    ss << adjncy[i];
                }
                ss << "}" << std::endl;

                Teuchos::barrier(*comm);
                for (int i = 0; i < ranks; i++) {
                    if (i == rank) {
                        std::cout << ss.str() << std::endl;
                    }
                    Teuchos::barrier(*comm);
                }

                /////////////////////////////////////////////////////////////////////
                // 3. Construct Tpetra Map of current distribution of dual graph...
                /////////////////////////////////////////////////////////////////////

                Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> indices(numVertices);
                int idx = 0;
                for (int i = elemdist[rank]; i < elemdist[rank+1]; i++) {
                    indices[idx++] = i;
                }
                auto currMap = Teuchos::rcp(new Tpetra::Map<>(params.num_elem, indices, 0, comm));
                auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                currMap->describe(*ostr, Teuchos::EVerbosityLevel::VERB_EXTREME);
                assert(currMap->isOneToOne());

                /////////////////////////////////////////////////////////////////////
                // 4. Construct Tpetra CrsMatrix over Tpetra Map for dual graph
                /////////////////////////////////////////////////////////////////////
                
                size_t maxNumEntriesPerRow = 0;
                for (int i = 0; i < numVertices; i++) {
                    maxNumEntriesPerRow = std::max(maxNumEntriesPerRow, (size_t) (xadj[i+1] - xadj[i]));
                }
                auto origMatrix = Teuchos::rcp(new Tpetra::CrsMatrix<>(currMap, maxNumEntriesPerRow));
                
                // Add elements from adjncy to origMatrix
                for (int i = 0; i < numVertices; i++) {
                    const int numEntries = xadj[i+1] - xadj[i];
                    Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> cols(numEntries);
                    Teuchos::Array<Tpetra::CrsMatrix<>::scalar_type> vals(numEntries);
                    for (int j = 0; j < numEntries; j++) {
                        cols[j] = adjncy[xadj[i] + j];
                        vals[j] = 1;
                    }
                    origMatrix->insertGlobalValues(elemdist[rank] + i, cols, vals);
                }

                origMatrix->fillComplete(currMap, currMap);
                origMatrix->describe(*ostr, Teuchos::EVerbosityLevel::VERB_MEDIUM);
                Teuchos::barrier(*comm);

                for (int i = 0; i < ranks; i++) {
                    if (i == rank) {
                        std::cout << "Process #" << rank << ": isGloballyIndexed() = " << origMatrix->isGloballyIndexed() << ", isDistributed() = " << origMatrix->isDistributed() << std::endl;
                        auto lclmtx = origMatrix->getLocalMatrix();
                        std::cout << "NNZ: " << lclmtx.nnz() << std::endl;
                    }
                    Teuchos::barrier(*comm);
                }

                return false;

                // Move data into matrix...
                // TODO

                /*
                /////////////////////////////////////////////////////////////////////
                // 5. Partition Dual Graph to obtain new partition
                /////////////////////////////////////////////////////////////////////

                idx_t *vtxdist = new idx_t[params.num_elem];
                idx_t *vwgt, *adjwgt = nullptr; // Unweighted
                idx_t wgtflag = 0; // Unweighted
                idx_t numflag = 0; // 0-based indexing (C-style)
                // Note: If Segfault occurs, try setting ncon=1 and just make uniform
                idx_t ncon = 0; // # of weights per vertex is 0
                real_t *tpwgts = nullptr;
                real_t *ubvec = nullptr;
                idx_t *options = nullptr; // May segfault?
                idx_t edgecut = 0;
                idx_t *part = new idx_t[num_vertices];


                retval = ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
                if (retval != METIS_OK) {
                    std::cout << "Error Code: " << (retval == METIS_ERROR_INPUT ? "METIS_ERROR_INPUT" : (retval == METIS_ERROR_MEMORY ? "METIS_ERROR_MEMORY" : "METIS_ERROR")) << std::endl;
                    return false;
                }
                // TODO: Use MPI to communicate the vtxdist

                /////////////////////////////////////////////////////////////////////
                // 6. Use Tpetra::Export to redistribute the data to the new distribution
                //    provided by ParMETIS
                /////////////////////////////////////////////////////////////////////
                
                Tpetra::Export<> exporter(currMap, newMap);
                auto retmatrix = rcp(new Tpetra::CrsMatrix<>(newMap));
                retmatrix->doExport(origMatrix, exporter, Tpetra::INSERT);
                return retmatrix;*/
            }

            // Performs partitioning of the Exodus file specified in `open` and writes
            // out the resulting partitioning scheme as an Exodus file specified in `create`.
            // Uses the sequential METIS since writing an Exodus file is a sequential operation
            // requiring a single node to hold all of the data, meaning this may not work well
            // for extremely large meshes.
            bool decompose(int partitions) {
                if (readFID == -1) return false;

                // Gather all data we need to pass to Metis
                ex_init_params params;
                if (ex_get_init_ext(readFID,&params)) {
                    return false;
                }
                std::cout << "Title: " << params.title << "\n# of Dimensions: " << params.num_dim  << "\n# of Blobs: " << params.num_blob << "\n# of Assembly: " << params.num_assembly
                    << "\n# of Nodes: " << params.num_nodes << "\n# of Elements: " << params.num_elem << "\n# of Faces: " << params.num_face
                    << "\n# of Element Blocks: " << params.num_elem_blk << "\n# of Face Blocks: " << params.num_face_blk << "\n# of Node Sets: " << params.num_node_sets 
                    << "\n# of Side Sets: " << params.num_side_sets << "\n# of Face Sets: " << params.num_face_sets << "\n# of Node Maps: " << params.num_node_maps
                    << "\n# of Element Maps: " << params.num_elem_maps << "\n# of Face Maps: " << params.num_face_maps 
                    << "\n# of Bytes in idx_t: " << sizeof(idx_t) << "\n# of Bytes in real_t: " << sizeof(real_t) << std::endl;

                int *ids = new int[params.num_elem_blk];
                for (int i = 0; i < params.num_elem_blk; i++) ids[i] = 0;
                idx_t num_elem_in_block = 0;
                idx_t num_nodes_per_elem = 0;
                idx_t num_edges_per_elem = 0;
                idx_t num_faces_per_elem = 0;
                idx_t num_attr = 0;
                char elemtype[MAX_STR_LENGTH+1];

                if (ex_get_ids(readFID, EX_ELEM_BLOCK, ids)) {
                    std::cerr << "Failed to call `ex_get_ids`" << std::endl;
                    return false;
                }
                for (int i = 0; i < params.num_elem_blk; i++) {
                    std::cout << "Element Block Id: " << (int) ids[i] << std::endl;
                }
                idx_t elementsIdx[params.num_elem + 1];
                for (int i = 0; i < params.num_elem + 1; i++) elementsIdx[i] = 0;
                std::vector<idx_t> nodesInElements;
                idx_t elemIdx = 0;
                idx_t nodeIdx = 0;

                // using the element block parameters read the element block info
                for (idx_t i=0; i<params.num_elem_blk; i++) {
                    if (ex_get_block(readFID, EX_ELEM_BLOCK, ids[i], elemtype, &num_elem_in_block, &num_nodes_per_elem, &num_edges_per_elem, &num_faces_per_elem, &num_attr)) {
                        std::cerr << "Failed to `ex_get_block` element block " << i+1 << " of " << params.num_elem_blk << " with id " << ids[i] << std::endl;
                        return false;
                    }
                    std::cout << "Block #" << i << " has the following..."
                        << "\n\t# of Elements: " << num_elem_in_block
                        << "\n\t# of Nodes per Element: " << num_nodes_per_elem
                        << "\n\t# of Edges per Element: " << num_edges_per_elem
                        << "\n\t# of Faces per Element: " << num_faces_per_elem
                        << "\n\t# of Attributes: " << num_attr
                        << "\n\tElement Type: " << elemtype << std::endl;

                        idx_t *connect = new idx_t[num_elem_in_block * num_nodes_per_elem];
                        for (int i = 0; i < num_elem_in_block * num_nodes_per_elem; i++) connect[i] = 0;
                        ex_get_elem_conn(readFID, ids[i], connect);
                        for (int i = 0; i < num_elem_in_block * num_nodes_per_elem; i++) connect[i]--;
                        std::cout << "Block #" << i << ": {";
                        for (idx_t j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
                            if (j) std::cout << ",";
                            if (j % num_nodes_per_elem == 0) { 
                                std::cout << "[";
                                elementsIdx[elemIdx++] = nodeIdx;
                            }
                            std::cout << connect[j];
                            nodesInElements.push_back(connect[j]);
                            nodeIdx++;
                            if ((j+1) % num_nodes_per_elem == 0) std::cout << "]";
                        }
                        std::cout << "}" << std::endl;
                        delete[] connect;
                }
                elementsIdx[elemIdx] = nodeIdx;
                std::cout << "Indexing: {";
                for (idx_t i = 0; i < params.num_elem + 1; i++) {
                    if (i) std::cout << ",";
                    std::cout << elementsIdx[i];
                }
                std::cout << "}" << std::endl;
                std::cout << "Nodes: {";
                for (idx_t i = 0; i < nodesInElements.size(); i++) {
                    if (i) std::cout << ",";
                    std::cout << nodesInElements[i];
                }
                std::cout << "}" << std::endl;

                idx_t ne = params.num_elem;
                idx_t nn = params.num_nodes;
                idx_t *eptr = elementsIdx;
                idx_t *eind = nodesInElements.data();
                idx_t *vwgt = nullptr;
                idx_t *vsize = nullptr;
                idx_t ncommon = 1;
                idx_t nparts = partitions;
                real_t *tpwgts = nullptr;
                /* TODO: Set Options METIS_OPTION_NUMBERING */
                idx_t *options = nullptr;
                idx_t objval = 0;
                idx_t *epart = new idx_t[ne];
                idx_t *npart = new idx_t[nn];
                for (idx_t i = 0; i < ne; i++) epart[i] = 0;
                for (idx_t i = 0; i < nn; i++) npart[i] = 0;
                // Note: We are assuming that there is only one Element Type in this mesh...
                if (strncmp(elemtype, "TETRA", 5) == 0) {
                    ncommon = 3;
                } else if (strncmp(elemtype, "TRI", 3) == 0) {
                    ncommon = 2;
                } else if (strncmp(elemtype, "HEX", 3) == 0) {
                    ncommon = 4;
                } else {
                    std::cerr << "Currently unsupported element type for mesh: " << elemtype << std::endl;
                    return false;
                }
                std::cout << "Calling METIS_PartMeshNodal with " << nparts << " partitions." << std::endl;
                int retval = METIS_PartMeshDual(&ne, &nn, eptr, eind, vwgt, vsize, &ncommon, &nparts, tpwgts, options, &objval, epart, npart);
                if (retval != METIS_OK) {
                    std::cout << "Error Code: " << (retval == METIS_ERROR_INPUT ? "METIS_ERROR_INPUT" : (retval == METIS_ERROR_MEMORY ? "METIS_ERROR_MEMORY" : "METIS_ERROR")) << std::endl;
                    return false;
                }

                std::cout << "ObjVal = " << objval << std::endl;
                std::cout << "Element Partition: {";
                for (idx_t i = 0; i < ne; i++) {
                    if (i) std::cout << ",";
                    std::cout << epart[i];
                }
                std::cout << "}" << std::endl;
                std::cout << "Node Partition: {";
                for (idx_t i = 0; i < nn; i++) {
                    if (i) std::cout << ",";
                    std::cout << npart[i];
                }
                std::cout << "}" << std::endl;
                
                if (npart[0] == -2) npart[0] = 0;

                std::vector<idx_t> elembin[nparts + 1];
                for (idx_t i=0; i<params.num_elem_blk; i++) {
                    if (ex_get_block(readFID, EX_ELEM_BLOCK, ids[i], elemtype,&num_elem_in_block,&num_nodes_per_elem, &num_edges_per_elem, &num_faces_per_elem, &num_attr)) return false;
    
                    idx_t *connect = new idx_t[num_elem_in_block * num_nodes_per_elem];
                    ex_get_elem_conn(readFID, ids[i], connect);
                    idx_t idx = 0;
                    idx_t part = epart[idx];
                    for (idx_t j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
                        // Check if every node in this block is in the same partition...
                        if (npart[connect[j]-1] != part && false) {
                            j += num_nodes_per_elem - (j % num_nodes_per_elem) - 1;
                            elembin[nparts].push_back(idx);
                            // Update current partition with next element; check for out-of-bounds
                            if (idx + 1 < num_elem_in_block) part = epart[++idx];
                            continue;
                        }
                        if ((j+1) % num_nodes_per_elem == 0) {
                            elembin[part].push_back(idx);
                            // Update current partition with next element; check for out-of-bounds
                            if (idx + 1 < num_elem_in_block) part = epart[++idx];
                        }
                    }
                    delete[] connect;
                }
                    
                for (idx_t i = 0; i < nparts + 1; i++) {
                    if (i == nparts) std::cout << "Any Partition(" << elembin[i].size() << "): [";
                    else std::cout << "Partition #" << i << "("<< elembin[i].size() <<"): [";
                    for (idx_t j = 0; j < elembin[i].size(); j++) {
                        if (j) std::cout << ",";
                        std::cout << elembin[i][j];
                    }
                    std::cout << "]" << std::endl;
                }

                idx_t numparts = 0;
                for (idx_t i = 0; i < nparts + 1; i++) {
                    idx_t num_nodes_per_elem = -1;
                    for (idx_t elemIdx : elembin[i]) {
                        num_nodes_per_elem = elementsIdx[elemIdx + 1] - elementsIdx[elemIdx];
                        break;
                    }
                    if (num_nodes_per_elem > 0) {
                        numparts++;
                    }
                }

                std::vector<idx_t> nodebin[nparts];
                // Add new node set parameters
                for (int i = 0; i < nn; i++) {
                    if (npart[i] == -2) {
                        nodebin[0].push_back(i);
                    } else {
                        nodebin[npart[i]].push_back(i);
                    }
                }
                // Remove duplicate nodes
                for (int i = 0; i < nparts; i++) {
                    std::vector<idx_t>& vec = nodebin[i];
                    sort(vec.begin(), vec.end());
                    vec.erase(unique(vec.begin(), vec.end()), vec.end());
                }

                // Write out new header
                ex_put_init(writeFID, params.title, params.num_dim, params.num_nodes, params.num_elem, numparts, params.num_node_sets /* + nparts*/, params.num_side_sets);

                // Writes out node coordinations
                std::cout << "Sizeof(real_t) = " << sizeof(real_t) << std::endl;
                real_t *xs = new real_t[params.num_nodes];
                real_t *ys = new real_t[params.num_nodes];
                real_t *zs = NULL;
                if (params.num_dim >= 3) zs = new real_t[params.num_nodes];
                ex_get_coord(readFID, xs, ys, zs);
                std::cout << "Node Coordinates: [";
                for (idx_t i = 0; i < params.num_nodes; i++) {
                    if (i) std::cout << ",";
                    std::cout << "(" << xs[i] << "," << ys[i] << "," << (zs ? zs[i] : 0) << ")";
                }
                std::cout << "]" << std::endl;
                ex_put_coord(writeFID, xs, ys, zs);
                // NOTE: Bad Free when trying to delete[] the ys and zs, but not xs... Debug Later!
                delete[] xs;
                delete[] ys;
                if (params.num_dim >= 3) delete[] zs;

                // Write out coordinate names
                char *coord_names[params.num_dim];
                for (idx_t i = 0; i < params.num_dim; i++) {
                    coord_names[i] = new char[MAX_STR_LENGTH + 1];
                }
                ex_get_coord_names(readFID, coord_names);
                ex_put_coord_names(writeFID, coord_names);
                for (idx_t i = 0; i < params.num_dim; i++) {
                    delete[] coord_names[i];
                }

                // Write element map
                idx_t *elem_map = new idx_t[params.num_elem];
                ex_get_map(readFID, elem_map);
                ex_put_map(writeFID, elem_map);
                delete[] elem_map;

                for (idx_t i = 0; i < nparts + 1; i++) {
                    if (i == nparts) std::cout << "Any Partition(" << elembin[i].size() << "): [";
                    else std::cout << "Partition #" << i << "("<< elembin[i].size() <<"): [";
                    for (idx_t j = 0; j < elembin[i].size(); j++) {
                        if (j) std::cout << ",";
                        std::cout << elembin[i][j];
                    }
                    std::cout << "]" << std::endl;
                }

                // Write new element blocks
                idx_t currPart = 0;
                for (idx_t i = 0; i < nparts + 1; i++) {
                    size_t num_elems_per_block = elembin[i].size();
                    idx_t num_nodes_per_elem = -1;
                    std::cout << "num_elems_per_block=" << num_elems_per_block << std::endl;
                    for (size_t j = 0; j < num_elems_per_block; j++) {
                        std::cout << "Index:" << j << ", Size: " << elembin[i].size() << std::endl;
                        idx_t elemIdx = elembin[i][j];
                        std::cout << "elemIdx=" << elemIdx << std::endl;
                        num_nodes_per_elem = abs(elementsIdx[elemIdx + 1] - elementsIdx[elemIdx]);
                        std::cout << "num_nodes_per_elem=" << num_nodes_per_elem << std::endl;
                        break;
                    }
                    if (num_nodes_per_elem == -1) {
                        std::cerr << "Was not able to deduce the # of nodes per elem for block #" << i << "!" << std::endl;
                        continue;
                    }

                    // Note: There could be faces and sides per entry!!! Need a more general solution!
                    ex_put_block(writeFID, EX_ELEM_BLOCK, currPart, elemtype, num_elems_per_block, num_nodes_per_elem, 0, 0, 0);

                    idx_t *connect = new idx_t[num_elems_per_block * num_nodes_per_elem];
                    idx_t idx = 0;
                    std::cout << "Connectivity for Block #" << currPart << ": [ ";
                    for (idx_t elemIdx : elembin[i]) {
                        std::cout << "[ ";
                        for (idx_t j = elementsIdx[elemIdx]; j < elementsIdx[elemIdx+1]; j++) {
                            connect[idx++] = nodesInElements[j]+1;
                            std::cout << nodesInElements[j]+1 << " ";
                        }
                        std::cout << "]";
                    }
                    std::cout << "]" << std::endl;
                    if (idx == 0) {
                        std::cerr << "Connectivity is bad!" << std::endl;
                    }
                    ex_put_conn(writeFID, EX_ELEM_BLOCK, currPart++, connect, NULL, NULL);
                    delete[] connect;
                }

                ids = new int[params.num_node_sets];
                ex_get_ids(readFID, EX_NODE_SET, ids);
                
                for (idx_t i = 0; i < params.num_node_sets; i++) {
                    idx_t num_nodes_in_set = -1, num_df_in_set = -1;
                    assert(!ex_get_set_param(readFID, EX_NODE_SET, ids[i], &num_nodes_in_set, &num_df_in_set));
                
                    assert(!ex_put_set_param(writeFID, EX_NODE_SET, ids[i], num_nodes_in_set, num_df_in_set));
                    
                    idx_t *node_list = new idx_t[num_nodes_in_set];
                    real_t *dist_fact = new real_t[num_nodes_in_set];\

                
                    assert(!ex_get_set(readFID, EX_NODE_SET, ids[i], node_list, NULL));
                    
                    std::cout << "Nodeset #" << ids[i] << ": [";
                    for (int j = 0; j < num_nodes_in_set; j++) {
                        if (j) std::cout << ",";
                        std::cout << node_list[j];
                    }
                    std::cout << "]" << std::endl;
                
                    assert(!ex_put_set(writeFID, EX_NODE_SET, ids[i], node_list, NULL));
                
                    if (num_df_in_set > 0) {
                        assert(!ex_get_set_dist_fact(readFID, EX_NODE_SET, ids[i], dist_fact));
                        assert(!ex_put_set_dist_fact(writeFID, EX_NODE_SET, ids[i], dist_fact));
                    }
                
                    delete[] node_list;
                    delete[] dist_fact;
                }
                delete[] ids;
                
                /* Commented out as Cubit crashes when you have nodesets that encompass the entire mesh */
                // idx_t nid = params.num_node_sets + 1;
                // for (int i = 0; i < nparts; i++) {
                //     std::cout << "Nodeset #" << nid << ": [";
                //     for (int j = 0; j < nodebin[i].size(); j++) {
                //         if (j) std::cout << ",";
                //         std::cout << nodebin[i][j];
                //     }
                //     std::cout << "]" << std::endl;
                //     int error;
                //     if ((error = ex_put_set_param(writeFID, EX_NODE_SET, nid, nodebin[i].size(), 0))) {
                //         std::cerr << "Failed to put set parameters for nodeset #" << nid << std::endl;
                //         std::cerr << "Parameters: (writeFID=" << writeFID << ", EX_NODE_SET, nid=" << nid << ", num_nodes=" << nodebin[i].size() << ", num_df=0)" << std::endl; 
                //         std::cerr << "Error: " << error << ", status = " << ex_strerror(error) << std::endl;
                //         return false;
                //     }
                //     if (ex_put_set(writeFID, EX_NODE_SET, nid, nodebin[i].data(), NULL)) {
                //         std::cerr << "Failed to put set for nodeset #" << nid << std::endl;
                //         return false;
                //     }
                //     std::cout << "Writing nodeset #" << nid << std::endl;
                //     nid++;
                // }

                /* read node set properties */
                idx_t num_props;
                float fdum;
                char *cdum = NULL;
                ex_inquire(readFID, EX_INQ_NS_PROP, &num_props, &fdum, cdum);
                
                char *prop_names[3];
                for (idx_t i = 0; i < num_props; i++) {
                    prop_names[i] = new char[MAX_STR_LENGTH + 1];
                }
                idx_t *prop_values = new idx_t[params.num_node_sets];
                
                ex_get_prop_names(readFID, EX_NODE_SET, prop_names);
                ex_put_prop_names(writeFID, EX_NODE_SET, num_props, prop_names);
                
                for (idx_t i = 0; i < num_props; i++) {
                    std::cout << "prop_names[" << i << "] = " << prop_names[i] << std::endl;
                    ex_get_prop_array(readFID, EX_NODE_SET, prop_names[i], prop_values);
                    ex_put_prop_array(writeFID, EX_NODE_SET, prop_names[i], prop_values);
                }
                for (idx_t i = 0; i < num_props; i++) {
                    delete[] prop_names[i];
                }
                delete[] prop_values;
                
                /* read and write individual side sets */
                
                ids = new int[params.num_side_sets];
                
                ex_get_ids(readFID, EX_SIDE_SET, ids);
                
                for (idx_t i = 0; i < params.num_side_sets; i++) {
                    idx_t num_sides_in_set = -1, num_df_in_set = -1;
                    ex_get_set_param(readFID, EX_SIDE_SET, ids[i], &num_sides_in_set, &num_df_in_set);                
                    ex_put_set_param(writeFID, EX_SIDE_SET, ids[i], num_sides_in_set, num_df_in_set);
                
                    /* Note: The # of elements is same as # of sides!  */
                    idx_t num_elem_in_set = num_sides_in_set;
                    idx_t *elem_list       = new idx_t[num_elem_in_set];
                    idx_t *side_list       = new idx_t[num_sides_in_set];
                    idx_t *node_ctr_list   = new idx_t[num_elem_in_set];
                    idx_t *node_list       = new idx_t[num_elem_in_set * 21];
                    real_t *dist_fact       = new real_t[num_df_in_set];
                
                    ex_get_set(readFID, EX_SIDE_SET, ids[i], elem_list, side_list);
                    ex_put_set(writeFID, EX_SIDE_SET, ids[i], elem_list, side_list);
                    ex_get_side_set_node_list(readFID, ids[i], node_ctr_list, node_list);

                    if (num_df_in_set > 0) {
                        ex_get_set_dist_fact(readFID, EX_SIDE_SET, ids[i], dist_fact);
                        ex_put_set_dist_fact(writeFID, EX_SIDE_SET, ids[i], dist_fact);
                    }
                
                    delete[] elem_list;
                    delete[] side_list;
                    delete[] node_ctr_list;
                    delete[] node_list;
                    delete[] dist_fact;
                }
                
                /* read side set properties */
                ex_inquire(readFID, EX_INQ_SS_PROP, &num_props, &fdum, cdum);
                
                for (idx_t i = 0; i < num_props; i++) {
                    prop_names[i] = new char[MAX_STR_LENGTH + 1];
                }
                
                ex_get_prop_names(readFID, EX_SIDE_SET, prop_names);
                
                for (idx_t i = 0; i < num_props; i++) {
                    for (idx_t j = 0; j < params.num_side_sets; j++) {
                        idx_t prop_value;
                        ex_get_prop(readFID, EX_SIDE_SET, ids[j], prop_names[i], &prop_value);
                    
                        if (i > 0) { /* first property is ID so it is already stored */
                            ex_put_prop(writeFID, EX_SIDE_SET, ids[j], prop_names[i], prop_value);
                        }
                    }
                }
                for (idx_t i = 0; i < num_props; i++) {
                    delete[] prop_names[i];
                }
                delete[] ids;
                
                /* read and write QA records */
                idx_t num_qa_rec = -1;
                ex_inquire(readFID, EX_INQ_QA, &num_qa_rec, &fdum, cdum);
                char *qa_record[2][4];
                for (idx_t i = 0; i < num_qa_rec; i++) {
                    for (idx_t j = 0; j < 4; j++) {
                        qa_record[i][j] = new char[(MAX_STR_LENGTH + 1)];
                    }
                }
                
                ex_get_qa(readFID, qa_record);
                ex_put_qa(writeFID, num_qa_rec, qa_record);
                
                for (idx_t i = 0; i < num_qa_rec; i++) {
                    for (idx_t j = 0; j < 4; j++) {
                        delete[] qa_record[i][j];
                    }
                }
                /* read and write information records */
                idx_t num_info = -1;
                ex_inquire(readFID, EX_INQ_INFO, &num_info, &fdum, cdum);
                char *info[num_info];
                for (idx_t i = 0; i < num_info; i++) {
                    info[i] = new char[MAX_LINE_LENGTH + 1];
                }
                
                ex_get_info(readFID, info);
                ex_put_info(writeFID, num_info, info);
                
                for (idx_t i = 0; i < num_info; i++) {
                    delete[] info[i];
                }

                // constructTPetra(npart, nn, epart, eptr, eind, ne);

                return true;
            }


            ~IO() {
                if (readFID != -1) {
                    ex_close(readFID);
                }
                if (writeFID != -1) {
                    ex_close(writeFID);
                }
            }

        private:
            int readFID = -1;
            int writeFID = -1;

            // void constructTPetra(idx_t *npart, idx_t nparts, idx_t *epart, idx_t *eptr, idx_t *eind, idx_t num_elems) {
            //     typedef Tpetra::Map<> map_type;
            //     typedef Tpetra::Vector<>::scalar_type scalar_type;
            //     typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
            //     typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
            //     typedef Tpetra::Vector<>::mag_type magnitude_type;
            //     typedef Tpetra::CrsMatrix<> crs_matrix_type;

            //     using Teuchos::RCP;
            //     using Teuchos::rcp;
            //     using Teuchos::Array;
            //     using Teuchos::ArrayView;

            //     auto comm = Tpetra::getDefaultComm();
            //     int rank = comm->getRank();
            //     // Gather # of local indices
            //     idx_t num_local_elems = 0;
            //     for (idx_t i = 0; i < num_elems; i++) {
            //         if (epart[i] == rank) {
            //             num_local_elems++;
            //         }
            //     }
            //     Array<global_ordinal_type> elementList (num_local_elems);
            //     idx_t idx = 0;
            //     for (idx_t i = 0; i < num_elems; i++) {
            //         if (epart[i] == rank) {
            //             elementList[idx++] = i;
            //         }
            //     }
            //     for (Array<global_ordinal_type>::size_type i = 0; i < num_elems; i++) {
            //        elementList[i] = epart[i];
            //     }

            //     // Row map consists of Elements; Column map consists of Nodes that comprise the element
            //     RCP<const map_type> rowmap = rcp(new map_type(num_elems, elementList, 0, Tpetra::getDefaultComm()));
            //     RCP<const map_type> colmap = Tpetra::Details::makeColMap();
            //     // auto matrix = crs_matrix_type();
            // }
    };
};