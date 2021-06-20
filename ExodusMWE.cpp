#include "exodusII.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

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
        << "\n# of Element Maps: " << params.num_elem_maps << "\n# of Face Maps" << params.num_face_maps << std::endl;

    // Writes out node coordinations
    float *xs = new float[params.num_nodes];
    std::memset(xs, sizeof(float), params.num_nodes);
    float *ys = new float[params.num_nodes];
    std::memset(ys, sizeof(float), params.num_nodes);
    float *zs = NULL;
    if (params.num_dim >= 3) {
        zs = new float[params.num_nodes];
        std::memset(zs, sizeof(float), params.num_nodes);
    }
    if (ex_get_coord(exoid, xs, ys, zs)) return -1;
    std::cout << "Node Coordinates: [";
    for (int i = 0; i < params.num_nodes; i++) {
        if (i) std::cout << ",";
        std::cout << "(" << xs[i] << "," << ys[i] << "," << (zs ? zs[i] : 0) << ")";
    }
    std::cout << "]" << std::endl;
    delete[] xs;
    delete[] ys;
    if (params.num_dim >= 3) delete[] zs;
    return -1;
}