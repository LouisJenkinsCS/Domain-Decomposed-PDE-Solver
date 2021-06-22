#include "ExodusIO.hpp"

int main(void) {
    ExodusIO::IO io;
    if (!io.open("data/tet-cube.exo", true)) {
        std::cerr << "Failed to open Exodus file!" << std::endl;
        return -1;
    }
    if (!io.create("output.exo")) {
        std::cerr << "Failed to create output Exodus file!" << std::endl;
        return -1;
    }
    if (!io.decompose(4)) {
        std::cerr << "Failed to decompose!" << std::endl;
        return -1;
    }
    return 0;
}