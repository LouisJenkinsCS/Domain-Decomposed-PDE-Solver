#include "exodusII.h"
#include <string>
#include <iostream>
#include <metis.h>
#include <vector>
#include <cstdlib>
#include <cstring>

/*
    An ExodusII wrapper that provides some C++ bindings
*/

namespace ExodusIO {

    class IO {
        public:
            IO() {}
            
            // Opens the read in exodus file
            bool open(std::string fname, bool read_only = false) {
                int cpuWS = 8;
                int ioWS = 8;
                float version = 0.;

                int tmpFid = ex_open(fname.c_str(), read_only ? EX_READ : EX_WRITE, &cpuWS, &ioWS, &version);
                if (tmpFid <= 0) {
                    return false;
                }
                readFID = tmpFid;
                return true;
            }

            // Opens the writen out exodus file (modification of read in exodus file)
            bool create(std::string fname) {
                int cpuWS = 8;
                int ioWS = 8;

                int tmpFid = ex_create(fname.c_str(), EX_CLOBBER, &cpuWS, &ioWS);
                if (tmpFid <= 0) {
                    return false;
                }
                writeFID = tmpFid;
                return true;
            }

            bool decompose() {
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
                    << "\n# of Element Maps: " << params.num_elem_maps << "\n# of Face Maps" << params.num_face_maps << std::endl;

                int *ids = new int[params.num_elem_blk];
                int num_elem_in_block;
                int num_nodes_per_elem;
                int num_edges_per_elem;
                int num_faces_per_elem;
                int num_attr;
                char elemtype[MAX_STR_LENGTH+1];

                if (ex_get_ids(readFID, EX_ELEM_BLOCK, ids)) return false;
                int elementsIdx[params.num_elem + 1];
                std::vector<int> nodesInElements;
                int elemIdx = 0;
                int nodeIdx = 0;

                // using the element block parameters read the element block info
                for (int i=0; i<params.num_elem_blk; i++) {
                    if (ex_get_block(readFID, EX_ELEM_BLOCK, ids[i], elemtype,&num_elem_in_block,&num_nodes_per_elem, &num_edges_per_elem, &num_faces_per_elem, &num_attr)) return false;
                    std::cout << "Block #" << i << " has the following..."
                        << "\n\t# of Elements: " << num_elem_in_block
                        << "\n\t# of Nodes per Element: " << num_nodes_per_elem
                        << "\n\t# of Edges per Element: " << num_edges_per_elem
                        << "\n\t# of Faces per Element: " << num_faces_per_elem
                        << "\n\t# of Attributes: " << num_attr
                        << "\n\tElement Type: " << elemtype << std::endl;

                        int *connect = new int[num_elem_in_block * num_nodes_per_elem];
                        ex_get_elem_conn(readFID, ids[i], connect);
                        std::cout << "Block #" << i << ": {";
                        for (int j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
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
                for (int i = 0; i < params.num_elem + 1; i++) {
                    if (i) std::cout << ",";
                    std::cout << elementsIdx[i];
                }
                std::cout << "}" << std::endl;
                std::cout << "Nodes: {";
                for (int i = 0; i < nodesInElements.size(); i++) {
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
                idx_t nparts = 3;
                real_t *tpwgts = nullptr;
                idx_t *options = nullptr;
                idx_t objval = 0;
                idx_t *epart = new int[ne];
                idx_t *npart = new int[nn];
                for (int i = 0; i < ne; i++) epart[i] = 0;
                for (int i = 0; i < nn; i++) npart[i] = 0;
                // Note: We are assuming that there is only one Element Type in this mesh...
                if (strncmp(elemtype, "TETRA", 5)) {
                    ncommon = 3;
                } else if (strncmp(elemtype, "TRI", 3)) {
                    ncommon = 2;
                } else {
                    std::cerr << "Currently unsupported element type for mesh: " << elemtype << std::endl;
                }
                std::cout << "Calling METIS_PartMeshNodal with " << nparts << " partitions." << std::endl;
                int retval = METIS_PartMeshDual(&ne, &nn, eptr, eind, vwgt, vsize, &ncommon, &nparts, tpwgts, options, &objval, epart, npart);
                if (retval != METIS_OK) {
                    std::cout << "Error Code: " << (retval == METIS_ERROR_INPUT ? "METIS_ERROR_INPUT" : (retval == METIS_ERROR_MEMORY ? "METIS_ERROR_MEMORY" : "METIS_ERROR"));
                    return false;
                }

                std::cout << "ObjVal = " << objval << std::endl;
                std::cout << "Element Partition: {";
                for (int i = 0; i < ne; i++) {
                    if (i) std::cout << ",";
                    std::cout << epart[i];
                }
                std::cout << "}" << std::endl;
                std::cout << "Node Partition: {";
                for (int i = 0; i < nn; i++) {
                    if (i) std::cout << ",";
                    std::cout << npart[i];
                }
                std::cout << "}" << std::endl;
                
                if (npart[0] == -2) npart[0] = 0;

                std::vector<idx_t> elembin[nparts + 1];
                for (int i=0; i<params.num_elem_blk; i++) {
                    if (ex_get_block(readFID, EX_ELEM_BLOCK, ids[i], elemtype,&num_elem_in_block,&num_nodes_per_elem, &num_edges_per_elem, &num_faces_per_elem, &num_attr)) return false;
    
                    int *connect = new int[num_elem_in_block * num_nodes_per_elem];
                    ex_get_elem_conn(readFID, ids[i], connect);
                    int idx = 0;
                    idx_t part = epart[idx];
                    for (int j = 0; j < num_elem_in_block * num_nodes_per_elem; j++) {
                        if (npart[connect[j]-1] != part && npart[connect[j]-1] != -2) {
                            j += num_nodes_per_elem - (j % num_nodes_per_elem) - 1;
                            elembin[nparts].push_back(idx);
                            if (idx < num_elem_in_block) part = epart[idx];
                            idx++;
                            continue;
                        }
                        if ((j+1) % num_nodes_per_elem == 0 && idx + 1 < num_elem_in_block) {
                            elembin[part].push_back(idx);
                            part = epart[++idx];
                        }
                    }
                    delete[] connect;
                }
                    
                for (int i = 0; i < nparts + 1; i++) {
                    if (i == nparts) std::cout << "Any Partition: [";
                    else std::cout << "Partition #" << i << ": [";
                    for (int j = 0; j < elembin[i].size(); j++) {
                        if (j) std::cout << ",";
                        std::cout << elembin[i][j];
                    }
                    std::cout << "]" << std::endl;
                }

                // Write out new header
                ex_put_init(writeFID, params.title, params.num_dim, params.num_nodes, params.num_elem, nparts, params.num_node_sets, params.num_side_sets);
                
                // Writes out node coordinations
                real_t *xs = new real_t[params.num_nodes];
                real_t *ys = new real_t[params.num_nodes];
                real_t *zs = NULL;
                if (params.num_dim >= 3) zs = new real_t[params.num_nodes];
                ex_get_coord(readFID, xs, ys, zs);
                ex_put_coord(writeFID, xs, ys, zs);
                // NOTE: Bad Free when trying to delete[] the ys and zs, but not xs... Debug Later!
                // delete[] xs;
                // delete[] ys;
                // if (params.num_dim >= 3) delete[] zs;

                // Write out coordinate names
                char *coord_names[params.num_dim];
                for (int i = 0; i < params.num_dim; i++) {
                    coord_names[i] = new char[MAX_STR_LENGTH + 1];
                }
                ex_get_coord_names(readFID, coord_names);
                ex_put_coord_names(writeFID, coord_names);
                for (int i = 0; i < params.num_dim; i++) {
                    delete[] coord_names[i];
                }

                // Write element map
                int *elem_map = new int[params.num_elem];
                ex_get_map(readFID, elem_map);
                ex_put_map(writeFID, elem_map);
                delete[] elem_map;

                // Write new element blocks
                for (int i = 0; i < nparts + 1; i++) {
                    idx_t num_elems_per_block = elembin[i].size();
                    idx_t num_nodes_per_elem = -1;
                    for (idx_t elemIdx : elembin[i]) {
                        num_nodes_per_elem = elementsIdx[elemIdx + 1] - elementsIdx[elemIdx];
                        break;
                    }
                    if (num_nodes_per_elem == -1) {
                        std::cerr << "Was not able to deduce the # of nodes per elem!" << std::endl;
                        abort();
                    }
                    // Note: There could be faces and sides per entry!!! Need a more general solution!
                    ex_put_block(writeFID, EX_ELEM_BLOCK, i, elemtype, num_elems_per_block, num_nodes_per_elem, 0, 0, 0);

                    idx_t *connect = new idx_t[num_elems_per_block * num_nodes_per_elem];
                    int idx = 0;
                    for (idx_t elemIdx : elembin[i]) {
                        for (idx_t j = elementsIdx[elemIdx]; j < elementsIdx[elemIdx+1]; j++) {
                            connect[idx++] = nodesInElements[j];
                        }
                    }
                    ex_put_conn(writeFID, EX_ELEM_BLOCK, i, connect, NULL, NULL);
                    delete[] connect;
                }
                return true;
            }

            ~IO() {
                if (readFID != -1) {
                    ex_close(readFID);
                }
            }

        private:
            int readFID = -1;
            int writeFID = -1;
    };
}