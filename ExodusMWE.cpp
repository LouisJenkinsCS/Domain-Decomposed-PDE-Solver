#include "exodusII.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cassert>

typedef int64_t idx_t;
typedef double real_t;

int main(void) {
    int cpuWS = 8;
    int ioWS = 8;
    float version = 0.;

    int exoid = ex_open("data/rectangle-tris.exo", EX_READ, &cpuWS, &ioWS, &version);
    if (exoid <= 0) {
        return -1;
    }

    ex_init_params params;
    if (ex_get_init_ext(exoid,&params)) {
        return -1;
    }
    std::cout << "Title: " << params.title << "\n# of Dimensions: " << params.num_dim  << "\n# of Blobs: " << params.num_blob << "\n# of Assembly: " << params.num_assembly
                    << "\n# of Nodes: " << params.num_nodes << "\n# of Elements: " << params.num_elem << "\n# of Faces: " << params.num_face
                    << "\n# of Element Blocks: " << params.num_elem_blk << "\n# of Face Blocks: " << params.num_face_blk << "\n# of Node Sets: " << params.num_node_sets 
                    << "\n# of Side Sets: " << params.num_side_sets << "\n# of Face Sets: " << params.num_face_sets << "\n# of Node Maps: " << params.num_node_maps
                    << "\n# of Element Maps: " << params.num_elem_maps << "\n# of Face Maps: " << params.num_face_maps 
                    << "\n# of Bytes in idx_t: " << sizeof(idx_t) << "\n# of Bytes in real_t: " << sizeof(real_t) << std::endl;

    idx_t *ids = new idx_t[params.num_elem_blk];
    assert(ex_get_ids(exoid, EX_ELEM_BLOCK, ids));
    for (idx_t i = 0; i < params.num_elem_blk; i++) {
        std::cout << "Native int64_t value: " << ids[i];
        std::cout << "Truncated int value: " << (int)ids[i];
    }
    return -1;
}