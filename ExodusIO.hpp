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
                idx_t nparts = 2;
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
                    ncommon = 3;
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
                    
                for (int i = 0; i < nparts + 1; i++) {
                    if (i == nparts) std::cout << "Any Partition(" << elembin[i].size() << "): [";
                    else std::cout << "Partition #" << i << "("<< elembin[i].size() <<"): [";
                    for (int j = 0; j < elembin[i].size(); j++) {
                        if (j) std::cout << ",";
                        std::cout << elembin[i][j];
                    }
                    std::cout << "]" << std::endl;
                }

                idx_t numparts = 0;
                for (int i = 0; i < nparts + 1; i++) {
                    idx_t num_nodes_per_elem = -1;
                    for (idx_t elemIdx : elembin[i]) {
                        num_nodes_per_elem = elementsIdx[elemIdx + 1] - elementsIdx[elemIdx];
                        break;
                    }
                    if (num_nodes_per_elem > 0) {
                        numparts++;
                    }
                }

                // Write out new header
                ex_put_init(writeFID, params.title, params.num_dim, params.num_nodes, params.num_elem, numparts, params.num_node_sets, params.num_side_sets);

                // Writes out node coordinations
                real_t *xs = new real_t[params.num_nodes];
                real_t *ys = new real_t[params.num_nodes];
                real_t *zs = NULL;
                if (params.num_dim >= 3) zs = new real_t[params.num_nodes];
                ex_get_coord(readFID, xs, ys, zs);
                std::cout << "Node Coordinates: [";
                for (int i = 0; i < params.num_nodes; i++) {
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
                for (int i = 0; i < params.num_dim; i++) {
                    coord_names[i] = new char[MAX_STR_LENGTH + 1];
                }
                ex_get_coord_names(readFID, coord_names);
                ex_put_coord_names(writeFID, coord_names);
                for (int i = 0; i < params.num_dim; i++) {
                    delete[] coord_names[i];
                }

                // Write element map
                idx_t *elem_map = new idx_t[params.num_elem];
                ex_get_map(readFID, elem_map);
                ex_put_map(writeFID, elem_map);
                delete[] elem_map;

                for (int i = 0; i < nparts + 1; i++) {
                    if (i == nparts) std::cout << "Any Partition(" << elembin[i].size() << "): [";
                    else std::cout << "Partition #" << i << "("<< elembin[i].size() <<"): [";
                    for (int j = 0; j < elembin[i].size(); j++) {
                        if (j) std::cout << ",";
                        std::cout << elembin[i][j];
                    }
                    std::cout << "]" << std::endl;
                }

                // Write new element blocks
                int currPart = 0;
                for (int i = 0; i < nparts + 1; i++) {
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
                        std::cerr << "Was not able to deduce the # of nodes per elem!" << std::endl;
                        continue;
                    }

                    // Note: There could be faces and sides per entry!!! Need a more general solution!
                    ex_put_block(writeFID, EX_ELEM_BLOCK, currPart, elemtype, num_elems_per_block, num_nodes_per_elem, 0, 0, 0);

                    idx_t *connect = new idx_t[num_elems_per_block * num_nodes_per_elem];
                    int idx = 0;
                    std::cout << "Connectivity for Block #" << currPart << ": [ ";
                    for (idx_t elemIdx : elembin[i]) {
                        std::cout << "[ ";
                        for (idx_t j = elementsIdx[elemIdx]; j < elementsIdx[elemIdx+1]; j++) {
                            connect[idx++] = nodesInElements[j];
                            std::cout << nodesInElements[j] << " ";
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

                ids = new idx_t[params.num_node_sets];
                ex_get_ids(readFID, EX_NODE_SET, ids);
                
                for (int i = 0; i < params.num_node_sets; i++) {
                    idx_t num_nodes_in_set = -1, num_df_in_set = -1;
                    ex_get_set_param(readFID, EX_NODE_SET, ids[i], &num_nodes_in_set, &num_df_in_set);
                
                    ex_put_set_param(writeFID, EX_NODE_SET, ids[i], num_nodes_in_set, num_df_in_set);
                    
                    idx_t *node_list = new idx_t[num_nodes_in_set];
                    real_t *dist_fact = new real_t[num_nodes_in_set];
                
                    ex_get_set(readFID, EX_NODE_SET, ids[i], node_list, NULL);
                
                    ex_put_set(writeFID, EX_NODE_SET, ids[i], node_list, NULL);
                
                    if (num_df_in_set > 0) {
                        ex_get_set_dist_fact(readFID, EX_NODE_SET, ids[i], dist_fact);
                        ex_put_set_dist_fact(writeFID, EX_NODE_SET, ids[i], dist_fact);
                    }
                
                    delete[] node_list;
                    delete[] dist_fact;
                }
                delete[] ids;

                /* read node set properties */
                idx_t num_props;
                real_t fdum;
                char *cdum = NULL;
                ex_inquire(readFID, EX_INQ_NS_PROP, &num_props, &fdum, cdum);
                
                char *prop_names[3];
                for (int i = 0; i < num_props; i++) {
                    prop_names[i] = new char[MAX_STR_LENGTH + 1];
                }
                idx_t *prop_values = new idx_t[params.num_node_sets];
                
                ex_get_prop_names(readFID, EX_NODE_SET, prop_names);
                ex_put_prop_names(writeFID, EX_NODE_SET, num_props, prop_names);
                
                for (int i = 0; i < num_props; i++) {
                    ex_get_prop_array(readFID, EX_NODE_SET, prop_names[i], prop_values);
                    ex_put_prop_array(writeFID, EX_NODE_SET, prop_names[i], prop_values);
                }
                for (int i = 0; i < num_props; i++) {
                    delete[] prop_names[i];
                }
                delete[] prop_values;
                
                /* read and write individual side sets */
                
                ids = new idx_t[params.num_side_sets];
                
                ex_get_ids(readFID, EX_SIDE_SET, ids);
                
                for (int i = 0; i < params.num_side_sets; i++) {
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
                
                for (int i = 0; i < num_props; i++) {
                    prop_names[i] = new char[MAX_STR_LENGTH + 1];
                }
                
                ex_get_prop_names(readFID, EX_SIDE_SET, prop_names);
                
                for (int i = 0; i < num_props; i++) {
                    for (int j = 0; j < params.num_side_sets; j++) {
                        idx_t prop_value;
                        ex_get_prop(readFID, EX_SIDE_SET, ids[j], prop_names[i], &prop_value);
                    
                        if (i > 0) { /* first property is ID so it is already stored */
                            ex_put_prop(writeFID, EX_SIDE_SET, ids[j], prop_names[i], prop_value);
                        }
                    }
                }
                for (int i = 0; i < num_props; i++) {
                    delete[] prop_names[i];
                }
                delete[] ids;
                
                /* read and write QA records */
                idx_t num_qa_rec = -1;
                ex_inquire(readFID, EX_INQ_QA, &num_qa_rec, &fdum, cdum);
                char *qa_record[2][4];
                for (int i = 0; i < num_qa_rec; i++) {
                    for (int j = 0; j < 4; j++) {
                        qa_record[i][j] = new char[(MAX_STR_LENGTH + 1)];
                    }
                }
                
                ex_get_qa(readFID, qa_record);
                ex_put_qa(writeFID, num_qa_rec, qa_record);
                
                for (int i = 0; i < num_qa_rec; i++) {
                    for (int j = 0; j < 4; j++) {
                    delete[] qa_record[i][j];
                    }
                }
                /* read and write information records */
                idx_t num_info = -1;
                ex_inquire(readFID, EX_INQ_INFO, &num_info, &fdum, cdum);
                char *info[num_info];
                for (int i = 0; i < num_info; i++) {
                    info[i] = new char[MAX_LINE_LENGTH + 1];
                }
                
                ex_get_info(readFID, info);
                ex_put_info(writeFID, num_info, info);
                
                for (int i = 0; i < num_info; i++) {
                    delete[] info[i];
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