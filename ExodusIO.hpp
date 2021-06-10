#include "exodusII.h"
#include <string>
#include <iostream>
#include <metis.h>


/*
    An ExodusII wrapper that provides some C++ bindings
*/

namespace ExodusIO {

    class IO {
        public:
            IO() {}

            bool open(std::string fname, bool read_only = false) {
                int cpuWS = 8;
                int ioWS = 8;
                float version = 0.;

                int tmpFid = ex_open(fname.c_str(), read_only ? EX_READ : EX_WRITE, &cpuWS, &ioWS, &version);
                if (tmpFid <= 0) {
                    return false;
                }
                fid = tmpFid;
                return true;
            }

            bool create(std::string fname, bool clobber = false) {
                int cpuWS = 8;
                int ioWS = 8;

                int tmpFid = ex_create(fname.c_str(), clobber ? EX_CLOBBER : EX_NOCLOBBER, &cpuWS, &ioWS);
                if (tmpFid <= 0) {
                    return false;
                }
                fid = tmpFid;
                return true;
            }

            bool decompose() {
                if (fid == -1) return false;

                // Gather all data we need to pass to Metis
                ex_init_params params;
                if (ex_get_init_ext(fid,&params)) {
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

                if (ex_get_ids(fid, EX_ELEM_BLOCK, ids)) return false;

                // using the element block parameters read the element block info
                for (int i=0; i<params.num_elem_blk; i++) {
                    if (ex_get_block(fid, EX_ELEM_BLOCK, ids[i], elemtype,&num_elem_in_block,&num_nodes_per_elem, &num_edges_per_elem, &num_faces_per_elem, &num_attr)) return false;
                    std::cout << "Block #" << i << " has the following..."
                        << "\n\t# of Elements: " << num_elem_in_block
                        << "\n\t# of Nodes per Element: " << num_nodes_per_elem
                        << "\n\t# of Edges per Element: " << num_edges_per_elem
                        << "\n\t# of Faces per Element: " << num_faces_per_elem
                        << "\n\t# of Attributes: " << num_attr
                        << "\n\tElement Type: " << elemtype << std::endl;

                        int connect[num_elem_in_block * num_nodes_per_elem];
                        ex_get_elem_conn(fid, ids[i], connect);
                }
                return true;
            }

            ~IO() {
                if (fid != -1) {
                    ex_close(fid);
                }
            }

        private:
            int fid = -1;

    };
}