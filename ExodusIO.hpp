// TPetra data structures
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Zoltan2_Adapter.hpp>
#include <Zoltan2_XpetraCrsMatrixAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_XpetraMultiVectorAdapter.hpp>
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
#include <thread>
#include <atomic>

/*
    Developer Notes:

    - This file contains the core logic required to assemble matrices from an Exodus-II mesh file, as well as to decompose
      and visualize the decomposition. Users should invoke `open` to open the Exodus-II mesh file to read in, and should
      invoke `create` to open the Exodus-II mesh file to write out. `assemble` constructs the PDE for the steady-state heat
      equation, while `getMatrix` constructs the raw Laplacian matrix which includes all degree-of-freedom and non-degree-of-freedom
      nodes, and while is not necessarily useful on its own basis, can be rather useful when constructing a PDE based on sidesets,
      such as ones revolving around pressure applied to faces of the mesh.
    - Tpetra::Map::createOneToOne cannot be used to handle ghosted vertices when using ParMETIS as it will utilize a new
      `directoryMap_` that is distributed contiguous and uniform, and uses it's `remoteIndexList` to redistribute
      the vertices in a way that does not respect the original map. This discards the work performed by ParMETIS to
      partition the mesh, and so we instead distribute the ghost vertices in our own way.
    - For extremely large meshes, decomposing the Exodus file into multiple files (`decomp`) will likely be necessary since the
      Exodus API reads in O(N) data entirely into memory very often, where N is the number of nodes. The current code base would
      require some rather significant tweaking and rewriting as right now there is an assumption that each process is reading the
      same mesh file.
    - Sometimes, O(N) (for N nodes) of data is stored on a single node, i.e. for the node map (mapping of node indices used when presenting
      the Exodus mesh in an application like Cubit or Paraview) or the solution to the problem (Tpetra::MultiVector with only 1 column),
      but we make sure to never have the entire matrix on a single node. This is a massive difference, as larger and more complex meshes
      can have as much a maximum number of cells per row k=25 which would be O(N*k) memory. Hence while it could be better, it is not
      naive. 
    
    METIS:
    
    - METIS is a library that performs sequential partitioning of a graph or a mesh. Unlike its cousin, ParMETIS, it has some
      unique properties and additional functionality due to the entire mesh or graph being allocated on a single node. For example,
      METIS can return a partitioning scheme for the nodes as well as elements when partitioning a mesh. Otherwise, it is very much
      similar to ParMETIS. Please consort the METIS manual for more information.


    ParMETIS:
    
    - ParMETIS is a library that performs parallel partitioning of a graph or a mesh. It is a parallel version of METIS, and has some
      limitations compared to METIS, such as only providing the partitioning scheme of elements and not nodes, as well as having less
      functionality. The Distributed CSR format enforces either a relabeling of vertices or requires starting with a naive distribution,
      as processes must hold contiguous ranges of vertices, and it can be no less complex than that. Please consort the ParMETIS manual
      for more information.

    ExodusII:

    - The SEACAS Exodus II library is a C interface to the Exodus II mesh file format. It is a low-level tool for interfacing with
      Exodus-II mesh files. There are some nuances and rules that must be strictly followed, such as ensuring that the declared
      length of the scalar type or index types (4 byte or 8 byte) must match what is passed to certain functions, and is at times
      inconsistent (i.e. in the case of the indices). The librar is not type-safe, and so it is strongly recommended that one adheres
      to the Exodus-II manual and Doxygen-generated documentation that can be found online.
*/

namespace ExodusIO {

    /*
        A distributed object (i.e. all processes should call these functions), as it is stateful and holds process-specific and cached
        information. Invoke `open` to read in an Exodus-II mesh, `create` to write out an Exodus-II mesh, and `assemble` to assemble
        the Steady-State Heat Equation for the mesh; `writeSolution` to write out results of the Steady-State Heat Equation, and finally
        `decompose` to write out the solution and partitioned mesh to the output Exodus-II mesh file specified in `create`.
    */
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

            // Reads in the Exodus file specified in the 'open' function. The nodesets specified in the Exodus file is used to
            // distinguish known from unknown nodes. The known nodes, which are nodes specified in the nodesets, only contribute to the
            // overall degree of a given node, but the Laplacian will only be based on the degrees of freedom. I.E. if an unknown node n is
            // adjacent to a known node m, then m will contribute to the overall degree of n, but will not count as being adjacent to n.
            // The resulting Laplacian will be smaller than the original Laplacian (i.e. Laplacian including both known and unknown nodes),
            // which will then be partitioned via ParMETIS. The multivector X will be randomizd but will be the same size as the number of
            // degrees of freedom, and the multivector B wiil consist of the sum of known nodes that the unknown is adjacent to.
            // The CrsMatrix A has the following properties: 
            //      A[i,j] = -1 if i and j are adjacent (i and j are unknowns)
            //      A[i,i] = Degree(i) (Degree(i) = adjacent known and unknown)
            // Note: sidesets are unused; if you want to make use of sidesets, i.e. marking elements rather than nodes as unknown,
            // see 'getMatrix' function.
            bool assemble(Teuchos::RCP<Tpetra::CrsMatrix<>> *A, Teuchos::RCP<Tpetra::MultiVector<>> *X, Teuchos::RCP<Tpetra::MultiVector<>> *B, bool verbose=false) {
                typedef Tpetra::Details::DefaultTypes::global_ordinal_type global_t;
                typedef Tpetra::Details::DefaultTypes::local_ordinal_type local_t;
                
                auto comm = Tpetra::getDefaultComm();
                auto rank = Teuchos::rank(*comm);
                auto ranks = Teuchos::size(*comm);
                auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
                if (readFID == -1) return false;

                /////////////////////////////////////////////////////////////////////
                // 1. Read in Mesh from Exodus File
                /////////////////////////////////////////////////////////////////////

                // Gather all data we need to pass to ParMETIS - Each MPI Process is doing this...
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

                // WARNING: Nodemap reads everything on a single node!
                node_map = new int[params.num_nodes];
                ex_get_id_map(readFID, EX_NODE_MAP, node_map);

                if (verbose && rank == 0) {
                    std::cout << "Node Map: {" << std::endl;
                    for (int i = 0; i < params.num_nodes; i++) {
                        std::cout << "\t" << i << "->" << node_map[i] << std::endl;
                    }
                    std::cout << "}" << std::endl;
                }

                // Obtain nodeset information from Exodus file
                // Constraint: Exodus requires reading in the entire nodeset into memory
                // at once... this means that we can end up with an Out-of-Memory problem,
                // so each process only keeps track of local nodeset information. 
                auto initialMap = Teuchos::rcp(new Tpetra::Map<>(params.num_nodes, 0, comm));
                nodeSetMap.clear();
                int *ids = new int[params.num_node_sets];
                ex_get_ids(readFID, EX_NODE_SET, ids);
                
                for (idx_t i = 0; i < params.num_node_sets; i++) {
                    idx_t num_nodes_in_set = -1, num_df_in_set = -1;
                    assert(!ex_get_set_param(readFID, EX_NODE_SET, ids[i], &num_nodes_in_set, &num_df_in_set));
                                    
                    idx_t *node_list = new idx_t[num_nodes_in_set];
                    real_t *dist_fact = new real_t[num_nodes_in_set];

                
                    assert(!ex_get_set(readFID, EX_NODE_SET, ids[i], node_list, NULL));
                    
                    for (int j = 0; j < num_nodes_in_set; j++) {
                        nodeSetMap[ids[i]].insert(node_list[j]-1);
                    }
                    delete[] node_list;
                    delete[] dist_fact;
                }
                delete[] ids;

                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << " has " << nodeSetMap.size() << " node sets" << std::endl;
                            for (auto &idSet : nodeSetMap) {
                                std::cout << "Nodeset #" << idSet.first << " (" << idSet.second.size() << "): {";
                                int idx = 0;
                                for (auto id : idSet.second) {
                                    if (idx++) std::cout << ",";
                                    std::cout << node_map[id];
                                }
                                std::cout << "}" << std::endl;
                            }
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                /////////////////////////////////////////////////////////////////////
                // 2. Relabel nodes that are eliminated and create Laplacian
                /////////////////////////////////////////////////////////////////////
                
                // First eliminate local indices 
                std::map<local_t, local_t> relabelLocalIndexForward; // initialMap -> reducedMap
                std::map<local_t, local_t> relabelLocalIndexBackward; // reducedMap -> initialMap
                local_t newIdx = 0;
                for (local_t i = initialMap->getMinLocalIndex(); i < initialMap->getMaxLocalIndex(); i++) {
                    global_t gid = initialMap->getGlobalElement(i);
                    // Check if gid is in any of the nodeset maps
                    bool found = false;
                    for (auto &idSet : nodeSetMap) {
                        if (idSet.second.count(gid)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        relabelLocalIndexForward[i] = newIdx;
                        relabelLocalIndexBackward[newIdx] = i;
                        newIdx++;
                    }
                }

                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << " has " << relabelLocalIndexForward.size() << " relabelLocalIndexForward indices" << std::endl;
                            for (auto &idx : relabelLocalIndexForward) {
                                std::cout << "Local index " << node_map[initialMap->getGlobalElement(idx.first)] << " -> " << idx.second << std::endl;
                            }
                            std::cout << "Process #" << rank << " has " << relabelLocalIndexBackward.size() << " relabelLocalIndexBackward entries" << std::endl;
                            for (auto &idx : relabelLocalIndexBackward) {
                                std::cout << "Local index " << idx.first << " -> " << node_map[initialMap->getGlobalElement(idx.second)] << std::endl;
                            }
                        }
                        Teuchos::barrier(*comm);
                    }
                }
                auto reducedMap = Teuchos::rcp(new Tpetra::Map<>(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), newIdx, 0, Tpetra::getDefaultComm()));
                // Helper functions for interacting with relabeled nodes
                // initialMap.local -> reducedMap.global
                auto forwardLocalToGlobal = [&](local_t localIdx) -> global_t {
                    assert(initialMap->isNodeLocalElement(localIdx));
                    if (!relabelLocalIndexForward.count(localIdx)) {
                        return invalid;
                    }
                    auto newLocalIdx = relabelLocalIndexForward[localIdx];
                    assert(reducedMap->isNodeLocalElement(newLocalIdx));
                    return reducedMap->getGlobalElement(newLocalIdx);
                };
                // reducedMap.local -> initialMap.global
                auto backwardLocalToGlobal = [&](local_t localIdx) -> global_t {
                    assert(reducedMap->isNodeLocalElement(localIdx));
                    if (!relabelLocalIndexBackward.count(localIdx)) {
                        return invalid;
                    }
                    auto newLocalIdx = relabelLocalIndexBackward[localIdx];
                    assert(initialMap->isNodeLocalElement(newLocalIdx));
                    return initialMap->getGlobalElement(newLocalIdx);
                };
                // initialMap.global -> reducedMap.local
                auto forwardGlobalToLocal = [&](global_t globalIdx) -> local_t {
                    // assert(initialMap->isNodeGlobalElement(globalIdx));
                    auto localIdx = initialMap->getLocalElement(globalIdx);
                    if (!relabelLocalIndexForward.count(localIdx)) {
                        return invalid;
                    }
                    auto newLocalIdx = relabelLocalIndexForward[localIdx];
                    assert(reducedMap->isNodeLocalElement(newLocalIdx));
                    return newLocalIdx;
                };
                // reducedMap.global -> initialMap.local
                auto backwardGlobalToLocal = [&](global_t globalIdx) -> local_t {
                    // assert(reducedMap->isNodeGlobalElement(globalIdx));
                    auto localIdx = reducedMap->getLocalElement(globalIdx);
                    if (!relabelLocalIndexBackward.count(localIdx)) {
                        return invalid;
                    }
                    auto newLocalIdx = relabelLocalIndexBackward[localIdx];
                    assert(initialMap->isNodeLocalElement(newLocalIdx));
                    return newLocalIdx;
                };
                // initialMap.global -> reducedMap.global
                auto forwardGlobalToGlobal = [&](global_t globalIdx) -> global_t {
                    local_t localIdx = forwardGlobalToLocal(globalIdx);
                    if (localIdx == invalid) return invalid;
                    return reducedMap->getGlobalElement(localIdx);
                };
                // reducedMap.global -> initialMap.global
                auto backwardGlobalToGlobal = [&](global_t globalIdx) -> global_t {
                    local_t localIdx = backwardGlobalToLocal(globalIdx);
                    if (localIdx == invalid) return invalid;
                    return initialMap->getGlobalElement(localIdx);
                };
                auto isDOF = [&](global_t globalIdx) -> bool {
                    for (auto &idSet : nodeSetMap) {
                        if (idSet.second.count(globalIdx)) {
                            return false;
                        }
                    }
                    return true;
                };

                // We need to create the Laplacian with only adjacency of the degree-of-freedoms, but also have the
                // degree of both degree-of-freedom and non-degree-of-freedom neighbors. The current process obtains
                // the adjacency of each node that it owns by scanning through each element block and adding nodes in each
                // element to the set for each node representing adjacency. 

                ids = new int[params.num_elem_blk];
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
                std::map<idx_t, std::set<idx_t>> adjacency;
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
                    for (idx_t j = 0; j < num_elem_in_block * num_nodes_per_elem; j += num_nodes_per_elem) {
                        for (idx_t k = j; k < j + num_nodes_per_elem; k++) {
                            if (initialMap->isNodeGlobalElement(connect[k]-1)) {
                                // Check if it belongs to a nodeset (i.e. is a non-DOF node)
                                // Note, we can just check if relabelMap is -1 for the local index since
                                // we have already computed whether or not a node is a DOF or not.
                                global_t globalIdx = forwardGlobalToGlobal(connect[k]-1);
                                if (globalIdx != invalid) {
                                    for (idx_t l = j; l < j + num_nodes_per_elem; l++) {
                                        if (l == k) continue;
                                        // Note that we may insert non-DOF nodes, which is fine since we need to get the degree anyway
                                        adjacency[globalIdx].insert(connect[l]);
                                    }
                                }
                            }
                        }
                    }
                    delete[] connect;
                }

                std::map<idx_t, std::set<idx_t>> newAdjacency;
                for (auto &idSet : adjacency) {
                    for (global_t node : idSet.second) {
                        if (!isDOF(node-1)) continue;
                        newAdjacency[idSet.first].insert(node);
                    }
                }

                if (verbose) {
                    Teuchos::barrier(*comm);
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << std::endl;
                            std::cout << "Forward Mappings: {" << std::endl;
                            for (auto fwd : relabelLocalIndexForward) {
                                global_t lhs = initialMap->getGlobalElement(fwd.first);
                                global_t rhs = forwardGlobalToGlobal(initialMap->getGlobalElement(fwd.first));
                                std::cout << "\t" << lhs << "(" << node_map[lhs] << ") -> " << rhs << std::endl;
                            }
                            std::cout << "}" << std::endl;
                            std::cout << "Adjacency: {" << std::endl;
                            for (auto &idSet : newAdjacency) {
                                std::cout << "\t" << node_map[backwardGlobalToGlobal(idSet.first)] << " => [";
                                bool needComma = false;
                                for (global_t node : idSet.second) {
                                    if (needComma) std::cout << ",";
                                    std::cout << node_map[node-1];
                                    needComma = true;
                                }
                                std::cout << "]" << std::endl;
                            }
                            std::cout << "}" << std::endl;
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                // Compute global column size
                size_t maxColPerRow = 0;
                for (auto &idSet : newAdjacency) {
                    maxColPerRow = std::max(maxColPerRow, idSet.second.size() + 1);
                }
                auto laplacian = Teuchos::rcp(new Tpetra::CrsMatrix<>(reducedMap, maxColPerRow));
                

                /////////////////////////////////////////////////////////////////////
                // 3. Fill out the Laplacian from Adjacency information
                /////////////////////////////////////////////////////////////////////

                // All processes must communicate with each other to share this new mapping from initialMapping -> reducedMapping
                // Build local global-to-global indices to be shared across all processes
                // Each process has an array of indices that acts as a dense mapping from initialMap to reduceMap.
                // The `getRemoteIndexList` gives you information such as the node IDs (process ids) and local indices, which correspond
                // to the aforementioned dense mapping. By pooling together the set of neighboring indices specified in `newAdjacency`, 
                // and obtaining their global indices (obtained by indexing directly into the remote process' dense mappings), we can
                // cut down the amount of space stored on each node. We use an MPI Window allowing other processes to directly GET the
                // information without the extra tedium and difficulty involved with two-sided communication.
                MPI_Win mappingWindow;
                global_t *globalIndices;
                MPI_Win_allocate(sizeof(global_t) * initialMap->getNodeNumElements(), sizeof(global_t), MPI_INFO_NULL, MPI_COMM_WORLD, &globalIndices, &mappingWindow);
                for (local_t i = initialMap->getMinLocalIndex(); i < initialMap->getMaxLocalIndex(); i++) {
                    // Gather initialMap -> reducedMap mappings
                    globalIndices[i] = forwardLocalToGlobal(i);
                    assert(globalIndices[i] != invalid || !isDOF(initialMap->getGlobalElement(i)));
                }

                // Our sparse mappings of initialMap -> reducedMap based on newAdjacency; should include local mappings too
                std::map<global_t, global_t> sparseMapping; // global initialMap -> global reducedMap
                std::set<global_t> neighborsToQuery;
                for (auto &idSet : newAdjacency) {
                    global_t id = idSet.first;
                    sparseMapping[backwardGlobalToGlobal(id)] = id;
                    assert(sparseMapping[backwardGlobalToGlobal(id)] != invalid);
                    for (global_t node : idSet.second) {
                        if (initialMap->isNodeGlobalElement(node-1)) continue;
                        neighborsToQuery.insert(node-1);
                    }
                }

                if (reducedMap->isDistributed()) {
                    // This newAdjacency set contains global ids in the initialMap that we are adjacent to; we need to translate
                    // them into global ids in the reducedMap. We call `getRemoteIndexList` which yields the local id (index in the
                    // 1D array of globalMappings [initialMap -> reducedMap]) and the process it is allocated on.
                    Teuchos::Array<global_t> GIDList(neighborsToQuery.size());
                    {
                        int i = 0;
                        for (auto &id : neighborsToQuery) {
                            GIDList[i++] = id;
                        }
                    }
                    Teuchos::Array<int> nodeIDs(GIDList.size());
                    Teuchos::Array<local_t> LIDList(GIDList.size());
                    assert(initialMap->getRemoteIndexList(GIDList, nodeIDs, LIDList) == Tpetra::LookupStatus::AllIDsPresent);
                    
                    // Optimization: Sort based on NodeIDs so that we can, in bulk, retrieve elements by locking the MPI Window only once.
                    std::vector<std::tuple<int, global_t, local_t>> sortedTriple; // (rank, globalIdx, localIdx)
                    for (int i = 0; i < nodeIDs.size(); i++) {
                        sortedTriple.push_back(std::make_tuple(nodeIDs[i], GIDList[i], LIDList[i]));
                    }
                    std::sort(sortedTriple.begin(), sortedTriple.end(), [](const std::tuple<int, global_t, local_t> &a, const std::tuple<int, global_t, local_t> &b) {
                        return std::get<0>(a) < std::get<0>(b);
                    });

                    if (verbose) {
                        for (int i = 0; i < ranks; i++) {
                            if (i == rank) {
                                std::cout << "Process #" << rank << " has " << sortedTriple.size() << " neighbors to query" << std::endl;
                                for (auto &t : sortedTriple) {
                                    std::cout << "\t" << std::get<0>(t) << ": " << std::get<1>(t) << " -> " << std::get<2>(t) << std::endl;
                                }
                            }
                            Teuchos::barrier(*comm);
                        }
                    }

                    // Access remote global mappings with optimization that we only need to unlock once we find a new nodeID
                    // First allocate the receiving buffers
                    std::vector<global_t> perProcessMapping[ranks];
                    int perProcessMappingSize = 0;
                    bool isFirstNode = true;
                    int currNodeID = -1;
                    for (auto &t : sortedTriple) {
                        int nodeID = std::get<0>(t);
                        global_t globalIdx = std::get<1>(t);
                        local_t localIdx = std::get<2>(t);
                        if (isFirstNode) {
                            isFirstNode = false;
                            currNodeID = nodeID;
                        }
                        
                        if (currNodeID != nodeID) {
                            perProcessMapping[currNodeID].resize(perProcessMappingSize+1);
                            currNodeID = nodeID;
                            perProcessMappingSize = 0;
                        }

                        perProcessMappingSize++;
                    }
                    assert(currNodeID != -1);
                    perProcessMapping[currNodeID].resize(perProcessMappingSize+1);
                    
                    // Next, fetch the global id mappings from each processor
                    isFirstNode = true;
                    currNodeID = -1;
                    perProcessMappingSize = 0;
                    for (auto &t : sortedTriple) {
                        int nodeID = std::get<0>(t);
                        global_t globalIdx = std::get<1>(t);
                        local_t localIdx = std::get<2>(t);
                        if (isFirstNode) {
                            isFirstNode = false;
                            MPI_Win_lock(MPI_LOCK_SHARED, nodeID, 0, mappingWindow);
                            currNodeID = nodeID;
                        }

                        if (currNodeID != nodeID) {
                            MPI_Win_unlock(currNodeID, mappingWindow);
                            currNodeID = nodeID;
                            perProcessMappingSize = 0;
                            MPI_Win_lock(MPI_LOCK_SHARED, nodeID, 0, mappingWindow);
                        }

                        perProcessMapping[nodeID][perProcessMappingSize] = invalid;
                        MPI_Get(&perProcessMapping[nodeID][perProcessMappingSize++], 1, sizeof(global_t) == 8 ? MPI_LONG : MPI_INT, nodeID, localIdx, 1, sizeof(global_t) == 8 ? MPI_LONG : MPI_INT, mappingWindow);
                    }
                    MPI_Win_unlock(currNodeID, mappingWindow);

                    MPI_Win_free(&mappingWindow);
                    
                    // Now construct the actual mapping now that all asynchronous communication has ended.
                    isFirstNode = true;
                    currNodeID = -1;
                    perProcessMappingSize = 0;
                    for (auto &t : sortedTriple) {
                        int nodeID = std::get<0>(t);
                        global_t globalIdx = std::get<1>(t);
                        local_t localIdx = std::get<2>(t);
                        if (isFirstNode) {
                            isFirstNode = false;
                            currNodeID = nodeID;
                        }
                        
                        if (currNodeID != nodeID) {
                            currNodeID = nodeID;
                            perProcessMappingSize = 0;
                        }

                        sparseMapping[globalIdx] = perProcessMapping[nodeID][perProcessMappingSize++];
                    }
                }

                // Construct reverse mapping
                for (auto &idMap : sparseMapping) {
                    global_t i = idMap.first;
                    global_t j = idMap.second;
                    globalIDMap[j] = i;
                }

                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << " obtained " << sparseMapping.size() << " sparse mappings!" << std::endl;
                            for (auto mappings : sparseMapping) {
                                std::cout << "  " << node_map[mappings.first] << " -> " << mappings.second << std::endl;
                            }
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                // Add all elements to the matrix and force communication
                for (auto &idxRow : newAdjacency) {
                    idx_t id = idxRow.first;
                    auto &row = idxRow.second;
                    Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> cols(row.size() + 1);
                    Teuchos::Array<Tpetra::CrsMatrix<>::scalar_type> vals(row.size() + 1);
                    int idx = 0;
                    for (auto cell : row) {
                        global_t globalIdx = sparseMapping[cell-1];
                        // Only do so if it a DOF node (check relabel mapping)
                        cols[idx] = globalIdx;
                        vals[idx++] = -1;
                    }
                    cols.resize(idx+1);
                    vals.resize(idx+1);
                    cols[idx] = id;
                    vals[idx] = adjacency[id].size();
                    laplacian->insertGlobalValues(id, cols, vals);
                }
                laplacian->fillComplete(reducedMap, reducedMap);


                if (verbose) {
                    auto matrix = laplacian;
                    auto rank = Tpetra::getDefaultComm()->getRank();
                    auto ranks = Tpetra::getDefaultComm()->getSize();
                    auto rows = matrix->getGlobalNumRows();
                    auto map = matrix->getRowMap();

                    for (int row = 0; row <= rows; row++) {
                        if (map->isNodeGlobalElement(row)) {
                            std::cout << "Process #" << rank << ": " << node_map[backwardGlobalToGlobal(row)] << " => [";
                            Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> cols(matrix->getNumEntriesInGlobalRow(row));
                            Teuchos::Array<Tpetra::CrsMatrix<>::scalar_type> vals(matrix->getNumEntriesInGlobalRow(row));
                            size_t sz;
                            matrix->getGlobalRowCopy(row, cols, vals, sz);
                            std::vector<std::pair<Tpetra::CrsMatrix<>::global_ordinal_type, Tpetra::CrsMatrix<>::scalar_type>> entries;
                            for (size_t i = 0; i < cols.size(); i++) entries.push_back(std::make_pair(cols[i], vals[i]));
                            std::sort(entries.begin(), entries.end());
                            for (int i = 0; i < cols.size(); i++) {
                                if (i) std::cout << ",";
                                std::cout << "(" << entries[i].first << "," << entries[i].second << ")";
                            }
                            std::cout << "]" << std::endl;
                            std::flush(std::cout);
                        }
                        Teuchos::barrier(*Tpetra::getDefaultComm());
                    }
                }

                /////////////////////////////////////////////////////////////////////
                // 4. Partition the nodal matrix via Zoltan2 and friends (i.e. export)
                /////////////////////////////////////////////////////////////////////

                Teuchos::ParameterList param;
                param.set("partitioning_approach", "partition");
                param.set("algorithm", "parmetis");
                Zoltan2::XpetraCrsMatrixAdapter<Tpetra::CrsMatrix<>> adapter(laplacian);
                Zoltan2::PartitioningProblem<Zoltan2::XpetraCrsMatrixAdapter<Tpetra::CrsMatrix<>>> problem(&adapter, &param);
                try {
                    problem.solve();
                }
                catch (std::exception &e) {
                    std::cerr << "Exception returned from solve(). " << e.what() << std::endl;
                    return false;
                }
                adapter.applyPartitioningSolution(*laplacian.getConst(), *A, problem.getSolution());

                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                    laplacian->describe(*ostr, Teuchos::EVerbosityLevel::VERB_EXTREME);
                    Teuchos::barrier(*comm);
                }

                *X = Teuchos::rcp(new Tpetra::MultiVector<>((*A)->getDomainMap(),1));
                srand(time(NULL));
                (*X)->randomize();
                auto _B = Teuchos::rcp(new Tpetra::MultiVector<>(laplacian->getRangeMap(),1));

                // For the original distribution, we can scan through the original set of Adjacency to include the
                // non-DOF adjacent nodes, and then sum their values.
                for (auto &idxRow : adjacency) {
                    idx_t id = idxRow.first;
                    auto &row = idxRow.second;
                    int sum = 0;
                    for (auto cell : row) {
                        for (auto &idSet : nodeSetMap) {
                            if (idSet.second.count(cell-1)) {
                                sum += idSet.first;
                                break;
                            }
                        }
                    }
                    global_t boundaryConditionIdx = backwardGlobalToGlobal(id);
                    assert(boundaryConditionIdx != invalid);
                    auto data = _B->get1dViewNonConst();
                    data[_B->getMap()->getLocalElement(id)] = sum;
                }
                
                Zoltan2::XpetraMultiVectorAdapter<Tpetra::MultiVector<>> vectorAdapter(_B);
                vectorAdapter.applyPartitioningSolution(*_B, *B, problem.getSolution());
                
                // Since the mapping has likely changed again after partitioning via Zoltan2, we need to
                // gather the mapping of indices on process #0, because process #0 is resposible for
                // writing to the Exodus-II file.
                if (rank == 0) {
                    // Receive all users mappings
                    for (int i = 1; i < ranks; i++) {
                        std::vector<std::pair<global_t, global_t>> otherGlobalMappings;
                        int sz = 0;
                        MPI_Status status;
                        MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
                        MPI_Get_count(&status, MPI_UNSIGNED_LONG, &sz);
                        otherGlobalMappings.resize(sz);
                        MPI_Recv(&otherGlobalMappings[0], sizeof(std::pair<global_t, global_t>) * sz, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (auto &mapping : otherGlobalMappings) {
                            globalIDMap.insert(mapping);
                        }
                    }
                } else {
                    std::vector<std::pair<global_t, global_t>> ourGlobalMappings(globalIDMap.size());
                    {
                        int i = 0;
                        for (auto &idxRow : globalIDMap) {
                            ourGlobalMappings[i++] = std::make_pair(idxRow.first, idxRow.second);
                        }
                    }
                    // Send size and then mappings
                    size_t sz = ourGlobalMappings.size();
                    MPI_Send(&ourGlobalMappings[0], sizeof(std::pair<global_t, global_t>) * sz, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                }
                
                return true;
            }

            // Reads in the Exodus file specified in the 'open' function, partitions it 
            // according to ParMETIS, and then returns the Compressed Sparse Row Matrix.
            // The returned matrix is a Node x Node matrix, not one based on elements.
            // Note: The returned matrix is partitioned based on _elements_, and not based
            // on nodes. Right now it gives the Laplacian of the entire mesh, which is singular
            // and therefore not fit for a solver... but, if a particular PDE was meant to be
            // make use of the sidesets (which mark elements, not nodes, as unknown), then perhaps
            // this code could be useful.
            bool getMatrix(Teuchos::RCP<Tpetra::CrsMatrix<>> *ret, std::map<int, std::set<idx_t>> &nodeSetMap, bool verbose=false) {
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
                    if (verbose) comm->barrier();
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

                // To construct the Tpetra::Map, we must redistributing the ghosted nodes. Each process
                // only has access to its own local portion of the mesh, and so you can have nodes that are
                // contained in the elements that span multiple processes. We must therefore redistribute
                // the ghosted nodes, and to do that we must determine which nodes are to be ghosted (i.e.
                // take the intersection of nodes for each pair-wise process and then gather them).
                // There is a constraint that we cannot allow a single process to hold the entire mesh in
                // memory, but we relax this to allow for one process to hold all ghosted nodes at once, which
                // is bound to be significantly smaller in size. However, to ease this along, we construct an
                // MPI_Window to allow for one-sided fetching of each process' ghosted nodes.
                MPI_Win nodeWindow;
                MPI_Win nodeSizeWindow;
                size_t nodeSize = nodeIndices.size();
                idx_t *nodes;
                MPI_Alloc_mem(nodeSize * sizeof(idx_t), MPI_INFO_NULL, &nodes);
                for (int i = 0; i < nodeSize; i++) {
                    nodes[i] = nodeIndices[i];
                }
                MPI_Win_create(nodes, nodeSize * sizeof(idx_t), sizeof(idx_t), MPI_INFO_NULL, MPI_COMM_WORLD, &nodeWindow);
                MPI_Win_create(&nodeSize, sizeof(size_t), sizeof(size_t), MPI_INFO_NULL, MPI_COMM_WORLD, &nodeSizeWindow);

                std::set<idx_t> ghostedNodeSet;
                std::map<int, std::set<idx_t>> ghostedNodeMap; // rank -> set of ghosted nodes
                for (int i = 0; i < ranks; i++) {
                    if (rank != i) {
                        MPI_Win_lock(MPI_LOCK_SHARED, i, 0, nodeSizeWindow);
                        // Fetch the ghosted nodes from process #i
                        size_t theirNodeSize = 0;
                        MPI_Get(&theirNodeSize, 1, MPI_LONG, i, 0, 1, MPI_LONG, nodeSizeWindow);
                        MPI_Win_unlock(i, nodeSizeWindow);
                        if (verbose) std::cout << "Process #" << i << " received theirGhostedNodesSize = " << theirNodeSize << std::endl;
                        MPI_Win_lock(MPI_LOCK_SHARED, i, 0, nodeWindow);
                        std::vector<idx_t> theirNodes(theirNodeSize);
                        MPI_Get(theirNodes.data(), theirNodeSize, sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG, i, 0, theirNodeSize, sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG, nodeWindow);
                        MPI_Win_unlock(i, nodeWindow);
                        
                        std::vector<idx_t> intersection;
                        std::set_intersection(nodeIndices.begin(), nodeIndices.end(), theirNodes.begin(), theirNodes.end(), std::inserter(intersection, intersection.end()));
                        ghostedNodeSet.insert(intersection.begin(), intersection.end());
                        ghostedNodeMap[i].insert(intersection.begin(), intersection.end());
                    }
                }
                MPI_Win_free(&nodeWindow);
                MPI_Win_free(&nodeSizeWindow);

                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (rank == i) {
                            std::cout << "Process #" << rank << " has " << ghostedNodeSet.size() << " ghosted nodes" << std::endl;
                            std::cout << "Ghosted Nodes: {";
                            int idx = 0;
                            for (auto &node : ghostedNodeSet) {
                                if (idx++) std::cout << ",";
                                std::cout << node;
                            }
                            std::cout << "}" << std::endl;
                            for (int j = 0; j < ranks; j++) {
                                if (rank == j) continue;
                                std::cout << rank << "->" << j << "(" << ghostedNodeMap[j].size() << "): {";
                                int idx = 0;
                                for (auto &node : ghostedNodeMap[j]) {
                                    if (idx++) std::cout << ",";
                                    std::cout << node;
                                }
                                std::cout << "}" << std::endl;
                            }
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                // Ideal: First heuristic - Process which uses the ghosted nodes the most
                //        Second heuristic - Process with the lowest ID
                std::map<idx_t, std::vector<int>> nodeToRank; // node -> [rank]
                for (int i = 0; i < ranks; i++) {
                    if (rank != i) {
                        for (auto node : ghostedNodeMap[i]) {
                            nodeToRank[node].push_back(i);
                        }
                    }
                }

                // Compute the frequency in which each ghost node is used and send it to the processes that
                // share that particular ghosted node. We create a two vectors, one holding the ghost node
                // ids, and one holding the frequency, which are both sent. 
                std::map<idx_t, int> nodeToFreq;
                for (auto& row : adjacents) {
                    for (auto& id : row.second) {
                        nodeToFreq[id]++;
                    }
                }

                std::vector<std::pair<std::vector<idx_t>, std::vector<int>>> idFreqPair(ranks);
                for (auto &nodeRanks : nodeToRank) {
                    idx_t node = nodeRanks.first;
                    std::vector<int> &rankList = nodeRanks.second;
                    for (auto &rank : rankList) {
                        idFreqPair[rank].first.push_back(node);
                        idFreqPair[rank].second.push_back(nodeToFreq[node]);
                    }
                }

                // Now that we have computed the frequencies to be sent for each processor, we send them.
                // Phase 1: Send out all node list and frequencies (size -> nodeList -> freqList)
                std::vector<MPI_Request> requests(3 * (ranks - 1));
                for (int i = 0; i < ranks; i++) {
                    std::vector<idx_t> &nodeList = idFreqPair[i].first;
                    std::vector<int> &freqList = idFreqPair[i].second;
                    size_t nodeSize = nodeList.size();
                    size_t freqSize = freqList.size();
                    assert(nodeSize == freqSize);

                    MPI_Isend(&nodeSize, 1, MPI_LONG, i, 0, MPI_COMM_WORLD, &requests[i * 3]);
                    MPI_Isend(nodeList.data(), nodeSize, sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG, i, 0, MPI_COMM_WORLD, &requests[i * 3 + 1]);
                    MPI_Isend(freqList.data(), freqSize, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i * 3 + 2]);          
                }

                // Wait for all the messages to be sent
                assert(MPI_Waitall(3 * (ranks - 1), requests.data(), MPI_STATUSES_IGNORE) == MPI_SUCCESS);

                // Phase 2: Receive all node list and frequencies (size -> nodeList -> freqList)
                std::vector<std::pair<std::vector<idx_t>, std::vector<int>>> idFreqPairReceived(ranks);
                for (int i = 0; i < ranks; i++) {
                    if (rank == i) continue;

                    std::vector<idx_t> &nodeList = idFreqPairReceived[i].first;
                    std::vector<int> &freqList = idFreqPairReceived[i].second;
                    size_t nodeSize;
                    MPI_Recv(&nodeSize, 1, MPI_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    nodeList.resize(nodeSize);
                    MPI_Recv(nodeList.data(), nodeSize, sizeof(idx_t) == 4 ? MPI_INT : MPI_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    freqList.resize(nodeSize);
                    MPI_Recv(freqList.data(), nodeSize, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                // Now that we have the frequency information, begin pruning the indices that we do not own.
                // For any node indices, compare our frequency to that of other processes. If our frequency is
                // less than that of the other process, then we do not own that node.
                std::vector<int> nodeIndicesToRemove;
                for (auto &id : nodeIndices) {
                    int freq = nodeToFreq[id];
                    bool maxFreq = true;
                    int owningRank = rank;

                    for (int i = 0; i < ranks; i++) {
                        if (rank == i) continue;

                        // TODO: If nodeList is sorted, we can just do a binary search for O(log(n)) time complexity!
                        std::vector<idx_t> &nodeList = idFreqPairReceived[i].first;
                        std::vector<int> &freqList = idFreqPairReceived[i].second;
                        int idx = 0;
                        for (auto &node : nodeList) {
                            if (node == id) {
                                if (freq > freqList[idx]) {
                                    // Good, do nothing
                                } else if (freq < freqList[idx]) {
                                    // Note: If we ever need to figure out which rank owns the ghosted node, we can do so here...
                                    maxFreq = false;
                                } else { // freq == freqList[idx]
                                    owningRank = std::min(owningRank, std::min(rank, i));
                                }
                                break;
                            }
                            idx++;
                        }
                    }

                    if (!maxFreq || owningRank != rank) {
                        nodeIndicesToRemove.push_back(id);
                    }
                }

                // Remove indices by swapping with the end and pop_back
                for (auto id : nodeIndicesToRemove) {
                    nodeIndices.erase(std::remove(nodeIndices.begin(), nodeIndices.end(), id));
                }

                Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> indices(nodeIndices.size());
                idx = 0;
                for (auto x : nodeIndices) {
                    indices[idx++] = x;
                }
                
                // Note: createOneToOne creates a new Tpetra::Map, which would discard the original
                // partitioning, given to use by ParMETIS... can't use this...
                auto map = Teuchos::rcp(new Tpetra::Map<>(params.num_nodes, indices, 0, comm));
                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                    Teuchos::barrier(*comm);
                    map->describe(*ostr, Teuchos::EVerbosityLevel::VERB_EXTREME);
                    Teuchos::barrier(*comm);
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

                // Debugging Code: Gather all nodeIndices on a single rank and check for duplicates
                if (verbose) {
                    if (rank == 0) {
                        std::set<idx_t> debugSet(nodeIndices.begin(), nodeIndices.end());
                        for (int i = 0; i < ranks; i++) {
                            if (rank == i) continue;
                            int sz = 0;
                            MPI_Status status;
                            MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
                            MPI_Get_count(&status, MPI_INT, &sz);
                            MPI_Recv(&sz, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            std::vector<idx_t> recv(sz);
                            MPI_Recv(recv.data(), sz, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            for (auto id : recv) {
                                if (debugSet.find(id) != debugSet.end()) {
                                    std::cout << "Duplicate node index " << id << " on process " << i << std::endl;
                                } else {
                                    debugSet.insert(id);
                                }
                            }
                        }
                    } else {
                        int sz = nodeIndices.size();
                        MPI_Send(nodeIndices.data(), sz, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    }
                }

                assert(map->isOneToOne());

                /////////////////////////////////////////////////////////////////////
                // 5. Construct the nodal matrix from the constructed map.
                /////////////////////////////////////////////////////////////////////

                // Note: insertGlobalValues will add the values together, resulting in values that are not -1,
                // which would not be compliant with a Laplacian matrix. So we need to call completeFill
                // to force communication and global coherence of the matrix, and then resumeFill so that we can
                // use `replaceGlobalValues` to fix the values back to -1 (using getView).

                // Note: We have to send the adjacency information to the process we decide to give the vertex to, as each process
                // only knows of elements in its own local portion of the mesh.

                // maxColumnsPerRow + 1 due to including degree along the diagonal
                auto matrix = Teuchos::rcp(new Tpetra::CrsMatrix<>(map, maxColumnsPerRow+1));
                
                // Add all elements to the matrix and force communication
                for (auto &idxRow : adjacents) {
                    idx_t id = idxRow.first;
                    auto &row = idxRow.second;
                    Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> cols(row.size() + 1);
                    Teuchos::Array<Tpetra::CrsMatrix<>::scalar_type> vals(row.size() + 1);
                    int idx = 0;
                    for (auto cell : row) {
                        cols[idx] = cell;
                        vals[idx++] = -1;
                    }
                    cols[idx] = id;
                    vals[idx] = row.size();
                    matrix->insertGlobalValues(id, cols, vals);
                }
                matrix->fillComplete(map, map);

                matrix->resumeFill();
                for (auto row : nodeIndices) {
                    Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> cols(matrix->getNumEntriesInGlobalRow(row));
                    Teuchos::Array<Tpetra::CrsMatrix<>::scalar_type> vals(matrix->getNumEntriesInGlobalRow(row));
                    size_t sz;
                    matrix->getGlobalRowCopy(row, cols, vals, sz);
                    for (int i = 0; i < cols.size(); i++) {
                        if (vals[i] > 0) {
                            vals[i] = vals.size() - 1;
                        } else {
                            vals[i] = -1;
                        }
                    }
                    matrix->replaceGlobalValues(row, cols, vals);
                }
                matrix->fillComplete(map, map);
                
                if (verbose) {
                    auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
                    matrix->describe(*ostr, Teuchos::EVerbosityLevel::VERB_MEDIUM);
                    Teuchos::barrier(*comm);
                }
                
                *ret = matrix;

                // We must return the nodeset information for this mesh, as it is used for specifying boundary
                // conditions. We create a mapping of each id to the set of nodes in that nodeset, but only if
                // that node belongs to our local process.
                ids = new int[params.num_node_sets];
                ex_get_ids(readFID, EX_NODE_SET, ids);
                
                for (idx_t i = 0; i < params.num_node_sets; i++) {
                    idx_t num_nodes_in_set = -1, num_df_in_set = -1;
                    assert(!ex_get_set_param(readFID, EX_NODE_SET, ids[i], &num_nodes_in_set, &num_df_in_set));
                                    
                    idx_t *node_list = new idx_t[num_nodes_in_set];
                    real_t *dist_fact = new real_t[num_nodes_in_set];

                
                    assert(!ex_get_set(readFID, EX_NODE_SET, ids[i], node_list, NULL));
                    
                    for (int j = 0; j < num_nodes_in_set; j++) {
                        if (map->isNodeGlobalElement(node_list[j])) {
                            nodeSetMap[ids[i]].insert(node_list[j]);
                        }
                    }
                    delete[] node_list;
                    delete[] dist_fact;
                }
                delete[] ids;

                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << " has " << nodeSetMap.size() << " node sets" << std::endl;
                            for (auto &idSet : nodeSetMap) {
                                std::cout << "Nodeset #" << idSet.first << " (" << idSet.second.size() << "): {";
                                int idx = 0;
                                for (auto id : idSet.second) {
                                    if (idx++) std::cout << ",";
                                    std::cout << id;
                                }
                                std::cout << "}" << std::endl;
                            }
                        }
                        Teuchos::barrier(*comm);
                    }
                }


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
                    idx_t num_elem_in_block = 0;
                    idx_t num_nodes_per_elem = 0;
                    idx_t num_edges_per_elem = 0;
                    idx_t num_faces_per_elem = 0;
                    idx_t num_attr = 0;
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
                        if (verbose) std::cout << connect[j]+1;
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

                std::vector<idx_t> elembin[nparts];
                idx_t idx = 0;
                for (idx_t i=0; i<params.num_elem_blk; i++) {
                    idx_t num_elem_in_block = 0;
                    idx_t num_nodes_per_elem = 0;
                    idx_t num_edges_per_elem = 0;
                    idx_t num_faces_per_elem = 0;
                    idx_t num_attr = 0;
                    if (ex_get_block(readFID, EX_ELEM_BLOCK, ids[i], elemtype,&num_elem_in_block,&num_nodes_per_elem, &num_edges_per_elem, &num_faces_per_elem, &num_attr)) return false;
    
                    idx_t *connect = new idx_t[num_elem_in_block * num_nodes_per_elem];
                    ex_get_elem_conn(readFID, ids[i], connect);
                    idx_t part = epart[idx];

                    // For each element in the block, find the partition it belongs to and add it to that bin
                    for (idx_t j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
                        // We have reached the end of an element; assign it the index `idx` and move it into the elembin of that partition
                        if ((j+1) % num_nodes_per_elem == 0) {
                            // Element owns this element (which is the `idx`th element)
                            elembin[part].push_back(idx);
                            // Check if we are processing the last element in this block; if not,
                            // then we move on to the next partition (bounds-checking)
                            if ((idx + 1) < params.num_elem) part = epart[++idx];
                        }
                    }
                    delete[] connect;
                }

                if (verbose) {    
                    for (idx_t i = 0; i < nparts; i++) {
                        std::cout << "Partition #" << i << "("<< elembin[i].size() <<"): [";
                        for (idx_t j = 0; j < elembin[i].size(); j++) {
                            if (j) std::cout << ",";
                            std::cout << elembin[i][j];
                        }
                        std::cout << "]" << std::endl;
                    }
                }

                idx_t numparts = 0;
                for (idx_t i = 0; i < nparts; i++) {
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
                for (idx_t i = 0; i < nparts; i++) {
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

                /* read node set properties */
                if (verbose) std::cout << "Reading nodesets!" << std::endl;
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

                if (verbose) std::cout << "Written nodesets!" << std::endl;
                
                /* read and write individual side sets */
                
                if (verbose) std::cout << "Reading sidesets!" << std::endl;
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

                if (verbose) std::cout << "Written sidesets!" << std::endl;

                if (verbose) std::cout << "Reading sideset properties!" << std::endl;
                
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

                if (verbose) std::cout << "Written sideset properties!" << std::endl;
                
                /* read and write QA records */
                {
                    if (verbose) std::cout << "Reading QA records!" << std::endl;
                    idx_t num_qa_rec = -1;
                    ex_inquire(readFID, EX_INQ_QA, &num_qa_rec, &fdum, cdum);
                    char *qa_record[num_qa_rec][4];
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
                    
                    if (verbose) std::cout << "Written QA records!" << std::endl;
                }

                /* read and write information records */
                if (verbose) std::cout << "Reading information records!" << std::endl;

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

                if (verbose) std::cout << "Written information records!" << std::endl;

                // Write node maps
                if (verbose) std::cout << "Writing node maps!" << std::endl;
                int *node_map = new int[params.num_nodes];
                ex_get_node_num_map(readFID, node_map);
                ex_put_node_num_map(writeFID, node_map);

                return true;
            }

            // Handle writing node variables for a given timestep
            bool writeSolution(Teuchos::RCP<Tpetra::MultiVector<>> vec, const int timestep, bool verbose = false) {
                auto comm = Tpetra::getDefaultComm();
                int rank = comm->getRank();
                int ranks = comm->getSize();
                real_t *node_vals = nullptr;
                if (rank == 0) {

                    // Gather all non-DOF nodes and their values. These needs to be written at the current timestep,
                    // and they are assumed to be constant at each timestep.
                    node_vals = new real_t[params.num_nodes];
                    for (int i = 0; i < params.num_nodes; i++) node_vals[i] = 0.0f;
                    for (auto &idSet : nodeSetMap) {
                        int id = idSet.first;
                        auto &set = idSet.second;
                        for (auto node : set) {
                            node_vals[node] = id;
                        }
                    }
                }

                // Gather all solutions to a single node... note that there is no other way but to
                // read in _all_ solutions to a single node and write them back.
                // WARNING: Assumes 1-to-1 between indices and values!
                auto indices = vec->getMap()->getNodeElementList();
                auto values = vec->get1dView();
                assert(indices.size() == values.size());

                // Gather across all nodes
                std::vector<size_t> gatheredIndices[ranks];
                std::vector<double> gatheredValues[ranks];
                if (rank == 0) {
                    // Collect everyone elses results
                    for (int i = 0; i < ranks; i++) {
                        if (rank == i) continue;
                        size_t sz = 0;
                        MPI_Recv(&sz, 1, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (sz == 0) {
                            if (verbose) std::cout << "Process #" << i << " has no values to send." << std::endl;
                            continue;
                        }
                        gatheredIndices[i].resize(sz);
                        gatheredValues[i].resize(sz);   
                        MPI_Recv(&gatheredIndices[i][0], sz, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(&gatheredValues[i][0], sz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (verbose) std::cout << "Process #0 received " << sz << " values from Process #" << i << std::endl;
                    }
                } else {
                    // Send my results
                    size_t sz = indices.size();
                    MPI_Send(&sz, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
                    if (sz != 0) {
                        MPI_Send(&indices[0], sz, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
                        MPI_Send(&values[0], sz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                }

                // Take the gathered results and collect them into `node_vals`
                if (rank == 0) {
                    // TODO: This may clobber variable names, please make more dynamic by querying the current # of nodal variables
                    // and appending them.
                    char *varName = "Steady-State Heat Solution";
                    real_t t = 0.0;
                    if (!printedTimeZero) {
                        assert(ex_put_variable_param(writeFID, EX_NODAL, 1) == 0);
                        assert(ex_put_variable_names(writeFID, EX_NODAL, 1, &varName) == 0);
                        printedTimeZero = true;
                        assert(ex_put_time(writeFID, 1, &t) == 0);
                        assert(ex_put_nodal_var(writeFID, 1, 1, params.num_nodes, node_vals) == 0);
                    }

                    t = (real_t)timestep;                    
                    assert(ex_put_time(writeFID, timestep+1, &t) == 0);

                    for (int i = 0; i < ranks; i++) {
                        if (rank == i) {
                            for (int i = 0; i < indices.size(); i++) {
                                node_vals[globalIDMap[indices[i]]] = values[i];
                            }
                        }
                        for (size_t j = 0; j < gatheredIndices[i].size(); j++) {
                            assert(globalIDMap.count(gatheredIndices[i][j]));
                            node_vals[globalIDMap[gatheredIndices[i][j]]] = gatheredValues[i][j];
                        }
                    }
                    assert(ex_put_nodal_var(writeFID, timestep+1, 1, params.num_nodes, node_vals) == 0);

                    if (verbose) {
                        std::cout << "Wrote " << params.num_nodes << " nodal values for timestep " << timestep << "!" << std::endl;
                        std::cout << "[";
                        for (int i = 0; i < params.num_nodes; i++) {
                            if (i) std::cout << ",";
                            std::cout << node_vals[i];
                        }
                        std::cout << "]" << std::endl;
                    }
                    delete[] node_vals;
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
            bool printedTimeZero = false;

            // Cached values so we don't need to repeatedly query the Exodus file between calls
            ex_init_params params;
            std::map<int, std::set<idx_t>> nodeSetMap;
            int *node_map = nullptr;
            

            // Each process will, after calling the `assemble` method, will have a dense mapping of
            // each global index to the global index it corresponded to in the original mesh. This is
            // because we need to acquire the original mesh node ids to interact with the Exodus API, such
            // as when writing out the result. To avoid keeping all of the ids on a single node, each process
            // only stores id mappings for global indices that they own. In the event that a process requires
            // the global mappings of some other process, we can set up an MPI_Win for one-sided communication.
            std::map<Tpetra::CrsMatrix<>::global_ordinal_type, Tpetra::CrsMatrix<>::global_ordinal_type> globalIDMap;
            

            // Each process must already be able to and have computed some mapping of local indices to global indices in the other map in question.
            // This function essentially retrieves this mapping after some type of redistribution, i.e. after partitioning the matrix. For example,
            // if we have computed the mapping from one matrix to another and then partition the matrix again, the initial mappings will not left on
            // the original processes. This method will move those indices to the correct processes, defined by the partition function.
            // localIDMapper - a map of local indices in fromMap to global indices in the new target map
            // fromMap - the map from which we are moving indices that have mappings already computed for them.
            // GIDList - a list of global indices that we are moving from fromMap to the new target map
            // indexMapping - Reference to a map of global to global indices, which will be updated by this function
            void getIndexMappingAfterRedistribution(Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> GIDList, Teuchos::RCP<Tpetra::Map<>> fromMap, std::map<Tpetra::CrsMatrix<>::global_ordinal_type, Tpetra::CrsMatrix<>::global_ordinal_type> &indexMapping, std::function<Tpetra::CrsMatrix<>::global_ordinal_type(Tpetra::CrsMatrix<>::local_ordinal_type)> localIDMapper, bool verbose) {
                typedef Tpetra::Details::DefaultTypes::global_ordinal_type global_t;
                typedef Tpetra::Details::DefaultTypes::local_ordinal_type local_t;

                auto comm = Tpetra::getDefaultComm();
                int rank = comm->getRank();
                int ranks = comm->getSize();
                auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

                MPI_Win mappingWindow;
                global_t *globalIndices;
                MPI_Win_allocate(sizeof(global_t) * fromMap->getNodeNumElements(), sizeof(global_t), MPI_INFO_NULL, MPI_COMM_WORLD, &globalIndices, &mappingWindow);
                for (local_t i = fromMap->getMinLocalIndex(); i < fromMap->getMaxLocalIndex(); i++) {
                    globalIndices[i] = localIDMapper(i);
                }

                Teuchos::Array<int> nodeIDs(GIDList.size());
                Teuchos::Array<local_t> LIDList(GIDList.size());
                assert(fromMap->getRemoteIndexList(GIDList, nodeIDs, LIDList) == Tpetra::LookupStatus::AllIDsPresent);
                
                // Optimization: Sort based on NodeIDs so that we can, in bulk, retrieve elements by locking the MPI Window only once.
                std::vector<std::tuple<int, global_t, local_t>> sortedTriple; // (rank, globalIdx, localIdx)
                for (int i = 0; i < nodeIDs.size(); i++) {
                    sortedTriple.push_back(std::make_tuple(nodeIDs[i], GIDList[i], LIDList[i]));
                }
                std::sort(sortedTriple.begin(), sortedTriple.end(), [](const std::tuple<int, global_t, local_t> &a, const std::tuple<int, global_t, local_t> &b) {
                    return std::get<0>(a) < std::get<0>(b);
                });

                if (verbose) {
                    for (int i = 0; i < ranks; i++) {
                        if (i == rank) {
                            std::cout << "Process #" << rank << " has " << sortedTriple.size() << " neighbors to query" << std::endl;
                            for (auto &t : sortedTriple) {
                                std::cout << "\t" << std::get<0>(t) << ": " << std::get<1>(t) << " -> " << std::get<2>(t) << std::endl;
                            }
                        }
                        Teuchos::barrier(*comm);
                    }
                }

                // Access remote global mappings with optimization that we only need to unlock once we find a new nodeID
                // First allocate the receiving buffers
                std::vector<global_t> perProcessMapping[ranks];
                int perProcessMappingSize = 0;
                bool isFirstNode = true;
                int currNodeID = -1;
                for (auto &t : sortedTriple) {
                    int nodeID = std::get<0>(t);
                    global_t globalIdx = std::get<1>(t);
                    local_t localIdx = std::get<2>(t);
                    if (isFirstNode) {
                        isFirstNode = false;
                        currNodeID = nodeID;
                    }
                    
                    if (currNodeID != nodeID) {
                        perProcessMapping[currNodeID].resize(perProcessMappingSize+1);
                        currNodeID = nodeID;
                        perProcessMappingSize = 0;
                    }

                    perProcessMappingSize++;
                }
                assert(currNodeID != -1);
                perProcessMapping[currNodeID].resize(perProcessMappingSize+1);
                
                // Next, fetch the global id mappings from each processor
                isFirstNode = true;
                currNodeID = -1;
                perProcessMappingSize = 0;
                for (auto &t : sortedTriple) {
                    int nodeID = std::get<0>(t);
                    global_t globalIdx = std::get<1>(t);
                    local_t localIdx = std::get<2>(t);
                    if (isFirstNode) {
                        isFirstNode = false;
                        MPI_Win_lock(MPI_LOCK_SHARED, nodeID, 0, mappingWindow);
                        currNodeID = nodeID;
                    }

                    if (currNodeID != nodeID) {
                        MPI_Win_unlock(currNodeID, mappingWindow);
                        currNodeID = nodeID;
                        perProcessMappingSize = 0;
                        MPI_Win_lock(MPI_LOCK_SHARED, nodeID, 0, mappingWindow);
                    }

                    perProcessMapping[nodeID][perProcessMappingSize] = invalid;
                    MPI_Get(&perProcessMapping[nodeID][perProcessMappingSize++], 1, sizeof(global_t) == 8 ? MPI_LONG : MPI_INT, nodeID, localIdx, 1, sizeof(global_t) == 8 ? MPI_LONG : MPI_INT, mappingWindow);
                }
                MPI_Win_unlock(currNodeID, mappingWindow);

                MPI_Win_free(&mappingWindow);
                
                // Now construct the actual mapping now that all asynchronous communication has ended.
                isFirstNode = true;
                currNodeID = -1;
                perProcessMappingSize = 0;
                for (auto &t : sortedTriple) {
                    int nodeID = std::get<0>(t);
                    global_t globalIdx = std::get<1>(t);
                    local_t localIdx = std::get<2>(t);
                    if (isFirstNode) {
                        isFirstNode = false;
                        currNodeID = nodeID;
                    }
                    
                    if (currNodeID != nodeID) {
                        currNodeID = nodeID;
                        perProcessMappingSize = 0;
                    }

                    indexMapping[globalIdx] = perProcessMapping[nodeID][perProcessMappingSize++];
                }
            }
    };
};