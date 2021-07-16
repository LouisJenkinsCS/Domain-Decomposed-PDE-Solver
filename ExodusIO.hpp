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
#include <set>
#include <utility>



/*
    An ExodusII wrapper that provides some C++ bindings
*/

namespace ExodusIO {

    namespace TpetraUtilities {
        template <typename LocalOrdinal = Tpetra::CrsMatrix<>::local_ordinal_type, typename GlobalOrdinal = Tpetra::CrsMatrix<>::global_ordinal_type>
        class GhostIDHandler : public Tpetra::Details::TieBreak<LocalOrdinal, GlobalOrdinal> {
        public:
            bool mayHaveSideEffects() const {
                return false;
            }

            std::size_t selectedIndex (GlobalOrdinal GID, const std::vector<std::pair<int, LocalOrdinal> >& pid_and_lid) const {
                auto comm = Tpetra::getDefaultComm();
                int rank = comm->getRank();
                std::stringstream ss;
                ss << "Process #" << rank << ": GID=" << GID << ", pid_and_lid={";
                for (int i = 0; i < pid_and_lid.size(); i++) {
                    if (i) ss << ",";
                    ss << "(" << pid_and_lid[i].first << "," << pid_and_lid[i].second << ")";
                }
                ss << "}";
                std::cout << ss.str() << std::endl;
                return 0;
            }

            ~GhostIDHandler() {}
        };
    };

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

            // Reads in the Exodus file specified in the 'open' function, partitions it 
            // according to ParMETIS, and then returns the Compressed Sparse Row Matrix.
            // The returned matrix is a Node x Node matrix, not one based on elements.
            // To obtain a matrix consisting purely of elements, see `getDual`.
            // TODO: Crashes when run sequentially...
            bool getMatrix(Teuchos::RCP<Tpetra::CrsMatrix<>> *ret, bool verbose=false) {
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
                if (rank == 0 && verbose) {
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
                if (verbose) {
                    for (int i = 0; i < params.num_elem_blk; i++) {
                        std::cout << "Rank #" << Teuchos::rank(*comm) << ": " << "Element Block Id: " << (int) ids[i] << std::endl;
                    }
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
                    if (rank == 0 && verbose) {
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
                    for (idx_t j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
                        if (j % num_nodes_per_elem == 0) { 
                            if (passedIdx++ < startIdx) {
                                j += num_nodes_per_elem - 1;
                                continue;
                            }
                            elementsIdx[elemIdx++] = nodeIdx;
                        }
                        nodesInElements.push_back(connect[j]);
                        nodeIdx++;
                        
                        // End of Element Block
                        if ((j+1) % num_nodes_per_elem == 0) {
                            if (passedIdx == endIdx) break;
                        }
                    }
                    delete[] connect;
                }
                elementsIdx[elemIdx] = nodeIdx;
                size_t numLocalElems = elemIdx + 1;

                if (verbose) {
                    Teuchos::barrier(*comm);
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << std::endl; 
                            std::cout << "Indexing: {";
                            for (idx_t i = 0; i < numLocalElems; i++) {
                                if (i) std::cout << ",";
                                std::cout << elementsIdx[i];
                            }
                            std::cout << "}" << std::endl;
                            std::cout << "Nodes: {";
                            for (idx_t i = 0; i < nodesInElements.size(); i++) {
                                if (i) std::cout << ",";
                                if (i % num_nodes_per_elem == 0) std::cout << "[";
                                std::cout << nodesInElements[i];
                                if ((i+1) % num_nodes_per_elem == 0) std::cout << "]";
                            }
                            std::cout << "}" << std::endl;
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                /////////////////////////////////////////////////////////////////////
                // 2. Partition the Mesh via ParMETIS
                /////////////////////////////////////////////////////////////////////

                // Distributed CSR Format - Split up the element list (eind) into nparts contiguous chunks.
                // That is, it is equivalent to sequential CSR but the indexing starts at 0 (local indexing)
                // and the nodes in each element is 1/P starting at some index after the last process and before
                // the next process. Each process is reading in the Exodus file, and hence can just ignore the parts
                // of the process they do not care about.

                // Distribution of elements; scheme used is a simple block distribution, where
                // indices [startIdx, endIdx) contains the elements distributed over this process.
                // Each process must have the same elemdist, and so must also compute the indices for
                // all other MPI processes...

                // When partitioning the mesh, it will return the partitioning scheme for the elements; the elements
                // should be sent to their respective processes in an all-to-all exchange.
                idx_t *elemdist = new idx_t[ranks+1];
                elemdist[0] = 0;
                for (int i = 1; i <= ranks; i++) {
                    elemdist[i] = elemdist[i-1] + params.num_elem / ranks;
                }
                elemdist[ranks] = params.num_elem;
                idx_t numVertices = elemdist[rank+1] - elemdist[rank];

                Teuchos::barrier(*comm);
                if (rank == 0 && verbose) {
                    std::cout << "Element Distribution: {" << std::endl;
                    int pid = 0;
                    for (int i = 1; i <= ranks; i++) {
                        std::cout << "\tProcess #" << pid++ << " owns " << elemdist[i-1] << " to " << elemdist[i]-1 << std::endl;
                    }
                    std::cout << "}" << std::endl;
                }

                idx_t *eptr = elementsIdx;
                idx_t *eind = nodesInElements.data();
                idx_t *elmwgt = nullptr;
                idx_t wgtflag = 0;
                idx_t numflag = 0; // 0-based indexing (C-style)
                idx_t ncon = 1; // # of weights per vertex is 0
                idx_t ncommonnodes = 1;
                idx_t nparts = ranks;
                real_t tpwgts[ranks];
                for (int i = 0; i < ranks; i++) tpwgts[i] = 1.0 / ranks;
                real_t ubvec[1];
                ubvec[0] = 1.05;
                idx_t options[3]; // May segfault?
                options[0] = 0;
                idx_t edgecut = 0;
                idx_t *part = new idx_t[numVertices];
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
                int retval = ParMETIS_V3_PartMeshKway(elemdist, eptr, eind, elmwgt, &wgtflag, &numflag, &ncon,  &ncommonnodes, &nparts, tpwgts, ubvec, options, &edgecut, part, &mpicomm);
                if (retval != METIS_OK) {
                    std::cout << "Error Code: " << (retval == METIS_ERROR_INPUT ? "METIS_ERROR_INPUT" : (retval == METIS_ERROR_MEMORY ? "METIS_ERROR_MEMORY" : "METIS_ERROR")) << std::endl;
                    return false;
                }
                
                if (verbose) {
                    Teuchos::barrier(*comm);
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << std::endl;
                            std::cout << "Partition: {";
                            for (int i = 0; i < numVertices; i++) {
                                if (i) std::cout << ",";
                                std::cout << part[i];
                            }
                            std::cout << "}" << std::endl;
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                /////////////////////////////////////////////////////////////////////
                // 3. Send the new partitioned indices to each rank along with their
                //    nodes to their respective target processors.
                /////////////////////////////////////////////////////////////////////

                std::vector<idx_t> redistribute[ranks];
                std::vector<idx_t> redistributeNodes[ranks];
                for (int i = 0; i < ranks; i++) {
                    if (rank == i) {                        
                        // Collect vertices by global index to be redistributed to other processes...
                        for (int j = 0; j < numVertices; j++) {
                            redistribute[part[j]].push_back(elemdist[rank] + j);
                            for (int k = eptr[j]; k < eptr[j+1]; k++) {
                                redistributeNodes[part[j]].push_back(eind[k]);
                            }
                        }
                        if (verbose) {
                            for (int j = 0; j < ranks; j++) {
                                if (j == rank) {
                                    std::cout << "Process #" << j << "(" << redistribute[j].size() << "): {";
                                } else {
                                    std::cout << "Process #" << rank << " -> Process #" << j << "(" << redistribute[j].size() << "): {";
                                }
                                for (int k = 0; k < redistribute[j].size(); k++) {
                                    if (k) std::cout << ",";
                                    std::cout << redistribute[j][k];
                                }
                                std::cout << "}" << std::endl;
                            }
                            for (int j = 0; j < ranks; j++) {
                                if (j == rank) {
                                    std::cout << "Process #" << j << "(" << redistributeNodes[j].size() << "): {";
                                } else {
                                    std::cout << "Process #" << rank << " -> Process #" << j << "(" << redistributeNodes[j].size() << "): {";
                                }
                                for (int k = 0; k < redistributeNodes[j].size(); k++) {
                                    if (k) std::cout << ",";
                                    if (k % num_nodes_per_elem == 0) std::cout << "[";
                                    std::cout << redistributeNodes[j][k];
                                    if ((k+1) % num_nodes_per_elem == 0) std::cout << "]";
                                }
                                std::cout << "}" << std::endl;
                            }
                        }
                    }
                    if (verbose) Teuchos::barrier(*comm);
                }

                // Perform an All-To-All Exchange for each MPI Process
                // Each process needs to get its new indices to construct the new map.
                // Fetch each other process' sizes
                size_t asyncOps = 2 * ranks - 2; // (Isend + Irecv per rank) - 2 (no communication to ourselves)
                MPI_Request request[2 * asyncOps];
                MPI_Status status[2 * asyncOps];
                int64_t buflen[ranks][2];
                buflen[rank][0] = redistribute[rank].size();
                buflen[rank][1] = redistributeNodes[rank].size();
                size_t idx = 0;
                for (int i = 0; i < ranks; i++) {
                    if (rank == i) {
                        continue;
                    }
                    size_t len[2] = {redistribute[i].size(), redistributeNodes[i].size()};
                    MPI_Isend(len, 2, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                    MPI_Irecv(&buflen[i], 2, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                }
                if (MPI_Waitall(asyncOps, request, status) != MPI_SUCCESS) {
                    std::cerr << "Unable to pass sizes between processes!!! MPI_Waitall failure!" << std::endl;
                    return false;
                }

                size_t totalElements = 0;
                size_t totalNodes = 0;
                for (int i = 0; i < ranks; i++) {
                    totalElements += buflen[i][0];
                    totalNodes += buflen[i][1];
                }
                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (rank == i) {
                            std::cout << "Process #" << rank << " has " << totalElements << " elements and " << totalNodes << " nodes!" << std::endl;
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                // Gather all entries to be sent to other MPI processes
                // TODO!!! Got a bit confused with how to index this array and so I decided to just 
                // correct the indices (i.e. right now it is 2 x ranks matrix rather than a ranks x 2 matrix)
                // but I was too exhausted at the time to make the change. This makes indexing in the code rather
                // confusing... fix this up! Cutting down on Isend and Irecv isn't worth losing readability and
                // maintainability of the code...
                asyncOps = 4 * ranks - 4; // (2 Isend + 2 Irecv) - 4 (No communication to self)
                idx_t ***buf = new idx_t**[2]; // 2 x ranks matrix -> ranks x 2 matrix
                buf[0] = new idx_t*[ranks];
                buf[1] = new idx_t*[ranks];
                idx = 0;
                for (int i = 0; i < ranks; i++) {
                    if (rank == i) {
                        buf[0][i] = redistribute[rank].data();
                        buf[1][i] = redistributeNodes[rank].data();
                    } else {
                        buf[0][i] = new idx_t[buflen[i][0]];
                        buf[1][i] = new idx_t[buflen[i][1]];
                        MPI_Isend(redistribute[i].data(), redistribute[i].size(), sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                        MPI_Isend(redistributeNodes[i].data(), redistributeNodes[i].size(), sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                        MPI_Irecv(buf[0][i], buflen[i][0], sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                        MPI_Irecv(buf[1][i], buflen[i][1], sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                    }
                }
                if (MPI_Waitall(asyncOps, request, status) != MPI_SUCCESS) {
                    std::cerr << "Unable to pass buffers between processes!!! MPI_Waitall failure!" << std::endl;
                    return false;
                }

                std::vector<idx_t> ourElements(totalElements);
                std::vector<idx_t> ourNodes(totalNodes);
                elemIdx = 0;
                nodeIdx = 0;
                for (int i = 0; i < ranks; i++) {
                    for (int j = 0; j < buflen[i][0]; j++) {
                        ourElements[elemIdx++] = buf[0][i][j];
                    }
                    for (int j = 0; j < buflen[i][1]; j++) {
                        ourNodes[nodeIdx++] = buf[1][i][j];
                    }
                }
                assert(elemIdx == totalElements);
                assert(nodeIdx == totalNodes);

                std::vector<idx_t> nodeIndices(ourNodes);
                sort(nodeIndices.begin(), nodeIndices.end());
                nodeIndices.erase(unique(nodeIndices.begin(), nodeIndices.end()), nodeIndices.end());

                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (rank == i) {
                            std::cout << "Process #" << rank << " owns the elements: {";
                            for (int j = 0; j < totalElements; j++) {
                                if (j) std::cout << ",";
                                std::cout << ourElements[j];
                            }
                            std::cout << "}" << std::endl;
                            std::cout << "Process #" << rank << " owns the nodes: {";
                            for (int j = 0; j < totalNodes; j++) {
                                if (j) std::cout << ",";
                                if (j % num_nodes_per_elem == 0) std::cout << "[";
                                std::cout << ourNodes[j];
                                if ((j+1) % num_nodes_per_elem == 0) std::cout << "]";
                            }
                            std::cout << "}" << std::endl;
                            std::cout << "Process #" << rank << " owns the indices: {";
                            for (int j = 0; j < nodeIndices.size(); j++) {
                                if (j) std::cout << ",";
                                std::cout << nodeIndices[j];
                            }
                            std::cout << "}" << std::endl;
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                /////////////////////////////////////////////////////////////////////
                // 4. Construct the Map consisting of Map indices owned by this process
                /////////////////////////////////////////////////////////////////////

                Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> indices(nodeIndices.size());
                idx = 0;
                for (auto x : nodeIndices) {
                    indices[idx++] = x;
                }
                auto currMap = Teuchos::rcp(new Tpetra::Map<>(params.num_nodes, indices, 0, comm));
                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                    Teuchos::barrier(*comm);
                    currMap->describe(*ostr, Teuchos::EVerbosityLevel::VERB_EXTREME);
                }
                TpetraUtilities::GhostIDHandler<> tieBreaker;
                auto map = Tpetra::createOneToOne(currMap.getConst(), tieBreaker);
                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                    Teuchos::barrier(*comm);
                    map->describe(*ostr, Teuchos::EVerbosityLevel::VERB_EXTREME);
                }
                Teuchos::barrier(*comm);
                assert(map->isOneToOne());
    
                /////////////////////////////////////////////////////////////////////
                // 5. Construct the nodal matrix from the constructed map.
                /////////////////////////////////////////////////////////////////////

                
                // A node is adjacent to another node if both belong to the same element
                // To create the matrix, we need to know the maximum number of columns in
                // a given row, and so we need to group nodes with their adjacent nodes
                // ahead of time.
                std::map<idx_t, std::set<idx_t>> adjacents;
                for (int i = 0; i < totalNodes; i += num_nodes_per_elem) {
                    for (int j = i; j < i + num_nodes_per_elem; j++) {
                        for (int k = i; k < i + num_nodes_per_elem; k++) {
                            if (j == k) continue;
                            adjacents[ourNodes[j]].insert(ourNodes[k]);
                        }
                    }
                }
                size_t maxColumnsPerRow = 0;
                for (auto &row : adjacents) {
                    maxColumnsPerRow = std::max(maxColumnsPerRow, row.second.size());
                }
                
                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (rank == i) {
                            std::cout << "Process #" << rank << " has " << maxColumnsPerRow << " maximum row length!" << std::endl;
                            // std::cout << "Rows: {" << std::endl;
                            // for (idx_t node : nodeIndices) {
                            //     std::cout << "\tRow #" << node << ": {";
                            //     size_t useComma = 0;
                            //     for (idx_t cell : adjacents[node]) {
                            //         if (useComma++) std::cout << ",";
                            //         std::cout << cell;
                            //     }
                            //     std::cout << "}" << std::endl;
                            // }
                            // std::cout << "}" << std::endl;
                            
                            // Check for % of rows that are entirely local vs have remote memory access (non-local adjacent node)
                            size_t remoteRows = 0;
                            size_t remoteCells = 0;
                            size_t totalCells = 0;
                            size_t totalRows = 0;
                            for (idx_t node : nodeIndices) {
                                totalRows++;
                                size_t remoteRow = 0;
                                for (idx_t cell : adjacents[node]) {
                                    if (!map->isNodeGlobalElement(cell)) {
                                        remoteRow++;
                                        remoteCells++;
                                    }
                                    totalCells++;
                                }
                                if (remoteRow) remoteRows++;
                            }
                            std::cout << "Remote Rows: " << (double) remoteRows / totalRows * 100 << "%, Remote Cells: " << (double) remoteCells / totalCells * 100 << "%" << std::endl;
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                return false;
            }

            // Reads in the partitioning of the Exodus file specified in `open` and calls
            // ParMETIS to construct a dual graph; this dual graph is then partitioned
            // and redistributed/balanced across the appropriate number of processes.
            bool getDual(Teuchos::RCP<Tpetra::CrsMatrix<>> *ret, bool verbose=false) {
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
                if (rank == 0 && verbose) {
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
                if (verbose) {
                    for (int i = 0; i < params.num_elem_blk; i++) {
                        std::cout << "Rank #" << Teuchos::rank(*comm) << ": " << "Element Block Id: " << (int) ids[i] << std::endl;
                    }
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
                    if (rank == 0 && verbose) {
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
                    for (idx_t j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
                        if (j % num_nodes_per_elem == 0) { 
                            if (passedIdx++ < startIdx) {
                                j += num_nodes_per_elem - 1;
                                continue;
                            }
                            elementsIdx[elemIdx++] = nodeIdx;
                        }
                        nodesInElements.push_back(connect[j]);
                        nodeIdx++;
                        
                        // End of Element Block
                        if ((j+1) % num_nodes_per_elem == 0) {
                            if (passedIdx == endIdx) break;
                        }
                    }
                    delete[] connect;
                }
                elementsIdx[elemIdx] = nodeIdx;

                if (verbose) {
                    Teuchos::barrier(*comm);
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << std::endl; 
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
                        }
                        Teuchos::barrier(*comm);
                    }
                }


                /////////////////////////////////////////////////////////////////////
                // 2. Construct Dual Graph from Mesh
                /////////////////////////////////////////////////////////////////////

                // Distributed CSR Format - Split up the element list (eind) into nparts contiguous chunks.
                // That is, it is equivalent to sequential CSR but the indexing starts at 0 (local indexing)
                // and the nodes in each element is 1/P starting at some index after the last process and before
                // the next process. Each process is reading in the Exodus file, and hence can just ignore the parts
                // of the process they do not care about.

                // Distribution of elements; scheme used is a simple block distribution, where
                // indices [startIdx, endIdx) contains the elements distributed over this process.
                // Each process must have the same elemdist, and so must also compute the indices for
                // all other MPI processes...
                idx_t *elemdist = new idx_t[ranks+1];
                elemdist[0] = 0;
                for (int i = 1; i <= ranks; i++) {
                    elemdist[i] = elemdist[i-1] + params.num_elem / ranks;
                }
                elemdist[ranks] = params.num_elem;
                idx_t numVertices = elemdist[rank+1] - elemdist[rank];

                if (rank == 0 && verbose) {
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
                int retval = ParMETIS_V3_Mesh2Dual(elemdist, eptr, eind, &numflag, &ncommonnodes, &xadj, &adjncy, &mpicomm);
                if (retval != METIS_OK) {
                    std::cout << "Error Code: " << (retval == METIS_ERROR_INPUT ? "METIS_ERROR_INPUT" : (retval == METIS_ERROR_MEMORY ? "METIS_ERROR_MEMORY" : "METIS_ERROR")) << std::endl;
                    return false;
                }
                
                if (verbose) {
                    Teuchos::barrier(*comm);
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << std::endl;
                            std::cout << "xadj: {";
                            for (int i = 0; i <= numVertices; i++) {
                                if (i) std::cout << ",";
                                std::cout << xadj[i];
                            }
                            std::cout << "}" << std::endl;
                            std::cout << "adjncy: {";
                            for (int i = 0; i < xadj[numVertices - 1]; i++) {
                                if (i) std::cout << ",";
                                std::cout << adjncy[i];
                            }
                            std::cout << "}" << std::endl;
                        }
                        Teuchos::barrier(*comm);
                    }
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
                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                    Teuchos::barrier(*comm);
                    currMap->describe(*ostr, Teuchos::EVerbosityLevel::VERB_EXTREME);
                }
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
                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
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
                }

                /////////////////////////////////////////////////////////////////////
                // 5. Partition Dual Graph to obtain new partition
                /////////////////////////////////////////////////////////////////////

                idx_t *vtxdist = elemdist;
                idx_t *vwgt = nullptr, *adjwgt = nullptr; // Unweighted
                idx_t wgtflag = 0; // Unweighted
                numflag = 0; // 0-based indexing (C-style)
                // Note: If Segfault occurs, try setting ncon=1 and just make uniform
                idx_t ncon = 1; // # of weights per vertex is 0
                real_t tpwgts[ranks];
                for (int i = 0; i < ranks; i++) tpwgts[i] = 1.0 / ranks;
                real_t ubvec[1];
                ubvec[0] = 1.05;
                idx_t options[3]; // May segfault?
                options[0] = 0;
                idx_t edgecut = 0;
                idx_t *part = new idx_t[numVertices];

                retval = ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, (idx_t *) &ranks, tpwgts, ubvec, options, &edgecut, part, &mpicomm);
                if (retval != METIS_OK) {
                    std::cout << "Error Code: " << (retval == METIS_ERROR_INPUT ? "METIS_ERROR_INPUT" : (retval == METIS_ERROR_MEMORY ? "METIS_ERROR_MEMORY" : "METIS_ERROR")) << std::endl;
                    return false;
                }

                if (rank == 0 && verbose) std::cout << "Edgecut = " << edgecut << std::endl;

                std::vector<idx_t> redistribute[ranks];
                for (int i = 0; i < ranks; i++) {
                    if (rank == i) {                        
                        // Collect vertices by global index to be redistributed to other processes...
                        for (int j = 0; j < numVertices; j++) {
                            redistribute[part[j]].push_back(elemdist[rank] + j);
                        }
                        if (verbose) {
                            for (int j = 0; j < ranks; j++) {
                                if (j == rank) {
                                    std::cout << "Process #" << j << "(" << redistribute[j].size() << "): {";
                                } else {
                                    std::cout << "Process #" << rank << " -> Process #" << j << "(" << redistribute[j].size() << "): {";
                                }
                                for (int k = 0; k < redistribute[j].size(); k++) {
                                    if (k) std::cout << ",";
                                    std::cout << redistribute[j][k];
                                }
                                std::cout << "}" << std::endl;
                            }
                        }
                    }
                    if (verbose) Teuchos::barrier(*comm);
                }
                
                // Perform an All-To-All Exchange for each MPI Process
                // Each process needs to get its new indices to construct the new map.
                // Fetch each other process' sizes
                MPI_Request request[2 * ranks - 2];
                MPI_Status status[2 * ranks - 2];
                int64_t buflen[ranks];
                buflen[rank] = redistribute[rank].size();
                idx = 0;
                for (int i = 0; i < ranks; i++) {
                    if (rank == i) {
                        continue;
                    }
                    size_t len = redistribute[i].size();
                    MPI_Isend(&len, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                    MPI_Irecv(&buflen[i], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                }
                if (MPI_Waitall(2 * ranks - 2, request, status) != MPI_SUCCESS) {
                    std::cerr << "Unable to pass sizes between processes!!! MPI_Waitall failure!" << std::endl;
                    return false;
                }

                int64_t total = 0;
                for (int i = 0; i < ranks; i++) total += buflen[i];
                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (rank == i) {
                            std::cout << "Process #" << rank << " has " << total << " entries!" << std::endl;
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                // Gather all entries to be sent to other MPI processes
                idx_t **buf = new idx_t*[ranks];
                idx = 0;
                for (int i = 0; i < ranks; i++) {
                    if (rank == i) {
                        buf[i] = redistribute[rank].data();
                    } else {
                        buf[i] = new idx_t[buflen[i]];
                        MPI_Isend(redistribute[i].data(), redistribute[i].size(), sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                        MPI_Irecv(buf[i], buflen[i], sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[idx++]);
                    }
                }
                if (MPI_Waitall(2 * ranks - 2, request, status) != MPI_SUCCESS) {
                    std::cerr << "Unable to pass buffers between processes!!! MPI_Waitall failure!" << std::endl;
                    return false;
                }
                
                /////////////////////////////////////////////////////////////////////
                // 6. Use Tpetra::Export to redistribute the data to the new distribution
                //    provided by ParMETIS
                /////////////////////////////////////////////////////////////////////
                
                Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> finalIndices(total);
                idx = 0;
                for (int i = 0; i < ranks; i++) {
                    for (int j = 0; j < buflen[i]; j++) {
                        finalIndices[idx++] = buf[i][j];
                    }
                }
                auto newMap = Teuchos::rcp(new Tpetra::Map<>(params.num_elem, finalIndices, 0, comm));
                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                    Teuchos::barrier(*comm);
                    currMap->describe(*ostr, Teuchos::EVerbosityLevel::VERB_EXTREME);
                }
                assert(currMap->isOneToOne());

                // Compute the global maximum number of entries per row for construction of new map...
                size_t globalMaxNumEntriesPerRow = 0;
                MPI_Allreduce(&maxNumEntriesPerRow, &globalMaxNumEntriesPerRow, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

                Tpetra::Export<> exporter(currMap, newMap);
                auto retmatrix = rcp(new Tpetra::CrsMatrix<>(newMap, globalMaxNumEntriesPerRow));
                retmatrix->doExport(*origMatrix, exporter, Tpetra::INSERT);
                retmatrix->fillComplete(newMap, newMap);
                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                    retmatrix->describe(*ostr, Teuchos::EVerbosityLevel::VERB_MEDIUM);
                    Teuchos::barrier(*comm);
                }
                *ret = retmatrix;
                return true;
            }

            // Performs partitioning of the Exodus file specified in `open` and writes
            // out the resulting partitioning scheme as an Exodus file specified in `create`.
            // Uses the sequential METIS since writing an Exodus file is a sequential operation
            // requiring a single node to hold all of the data, meaning this may not work well
            // for extremely large meshes.
            bool decompose(int partitions, bool verbose=false) {
                if (readFID == -1) return false;

                // Gather all data we need to pass to Metis
                ex_init_params params;
                if (ex_get_init_ext(readFID,&params)) {
                    return false;
                }
                if (verbose) {
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
                    std::cerr << "Failed to call `ex_get_ids`" << std::endl;
                    return false;
                }

                if (verbose) {
                    for (int i = 0; i < params.num_elem_blk; i++) {
                        std::cout << "Element Block Id: " << (int) ids[i] << std::endl;
                    }
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
                    if (verbose) {
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
                    for (int i = 0; i < num_elem_in_block * num_nodes_per_elem; i++) connect[i]--;
                    if (verbose) std::cout << "Block #" << i << ": {";
                    for (idx_t j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
                        if (j && verbose) std::cout << ",";
                        if (j % num_nodes_per_elem == 0) { 
                            if (verbose) std::cout << "[";
                            elementsIdx[elemIdx++] = nodeIdx;
                        }
                        if (verbose) std::cout << connect[j];
                        nodesInElements.push_back(connect[j]);
                        nodeIdx++;
                        if ((j+1) % num_nodes_per_elem == 0 && verbose) std::cout << "]";
                    }
                    if (verbose) std::cout << "}" << std::endl;
                    delete[] connect;
                }
                elementsIdx[elemIdx] = nodeIdx;
                if (verbose) {
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
                }

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
                if (verbose) std::cout << "Calling METIS_PartMeshNodal with " << nparts << " partitions." << std::endl;
                int retval = METIS_PartMeshDual(&ne, &nn, eptr, eind, vwgt, vsize, &ncommon, &nparts, tpwgts, options, &objval, epart, npart);
                if (retval != METIS_OK) {
                    std::cerr << "Error Code: " << (retval == METIS_ERROR_INPUT ? "METIS_ERROR_INPUT" : (retval == METIS_ERROR_MEMORY ? "METIS_ERROR_MEMORY" : "METIS_ERROR")) << std::endl;
                    return false;
                }

                if (verbose) {
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
                }

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

                if (verbose) {    
                    for (idx_t i = 0; i < nparts + 1; i++) {
                        if (i == nparts) std::cout << "Any Partition(" << elembin[i].size() << "): [";
                        else std::cout << "Partition #" << i << "("<< elembin[i].size() <<"): [";
                        for (idx_t j = 0; j < elembin[i].size(); j++) {
                            if (j) std::cout << ",";
                            std::cout << elembin[i][j];
                        }
                        std::cout << "]" << std::endl;
                    }
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
                if (verbose) std::cout << "Sizeof(real_t) = " << sizeof(real_t) << std::endl;
                real_t *xs = new real_t[params.num_nodes];
                real_t *ys = new real_t[params.num_nodes];
                real_t *zs = NULL;
                if (params.num_dim >= 3) zs = new real_t[params.num_nodes];
                ex_get_coord(readFID, xs, ys, zs);
                
                if (verbose) {
                    std::cout << "Node Coordinates: [";
                    for (idx_t i = 0; i < params.num_nodes; i++) {
                        if (i) std::cout << ",";
                        std::cout << "(" << xs[i] << "," << ys[i] << "," << (zs ? zs[i] : 0) << ")";
                    }
                    std::cout << "]" << std::endl;
                }
                ex_put_coord(writeFID, xs, ys, zs);
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

                // Write new element blocks
                idx_t currPart = 0;
                for (idx_t i = 0; i < nparts + 1; i++) {
                    size_t num_elems_per_block = elembin[i].size();
                    idx_t num_nodes_per_elem = -1;
                    if (num_nodes_per_elem == 0 && i == nparts) continue;
                    if (verbose) std::cout << "num_elems_per_block=" << num_elems_per_block << std::endl;
                    for (size_t j = 0; j < num_elems_per_block; j++) {
                        if (verbose) std::cout << "Index:" << j << ", Size: " << elembin[i].size() << std::endl;
                        idx_t elemIdx = elembin[i][j];
                        if (verbose) std::cout << "elemIdx=" << elemIdx << std::endl;
                        num_nodes_per_elem = abs(elementsIdx[elemIdx + 1] - elementsIdx[elemIdx]);
                        if (verbose) std::cout << "num_nodes_per_elem=" << num_nodes_per_elem << std::endl;
                        break;
                    }
                    if (num_nodes_per_elem == -1 && verbose) {
                        std::cerr << "Was not able to deduce the # of nodes per elem for block #" << i << "!" << std::endl;
                        continue;
                    }

                    // Note: There could be faces and sides per entry!!! Need a more general solution!
                    ex_put_block(writeFID, EX_ELEM_BLOCK, currPart, elemtype, num_elems_per_block, num_nodes_per_elem, 0, 0, 0);

                    idx_t *connect = new idx_t[num_elems_per_block * num_nodes_per_elem];
                    idx_t idx = 0;
                    if (verbose) std::cout << "Connectivity for Block #" << currPart << ": [ ";
                    for (idx_t elemIdx : elembin[i]) {
                        if (verbose) std::cout << "[ ";
                        for (idx_t j = elementsIdx[elemIdx]; j < elementsIdx[elemIdx+1]; j++) {
                            connect[idx++] = nodesInElements[j]+1;
                            if (verbose) std::cout << nodesInElements[j]+1 << " ";
                        }
                        if (verbose) std::cout << "]";
                    }
                    if (verbose) std::cout << "]" << std::endl;
                    if (idx == 0 && verbose) {
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
                    real_t *dist_fact = new real_t[num_nodes_in_set];

                
                    assert(!ex_get_set(readFID, EX_NODE_SET, ids[i], node_list, NULL));
                    
                    if (verbose) {
                        std::cout << "Nodeset #" << ids[i] << ": [";
                        for (int j = 0; j < num_nodes_in_set; j++) {
                            if (j) std::cout << ",";
                            std::cout << node_list[j];
                        }
                        std::cout << "]" << std::endl;
                    }
                
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
                    if (verbose) std::cout << "prop_names[" << i << "] = " << prop_names[i] << std::endl;
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

    };
};