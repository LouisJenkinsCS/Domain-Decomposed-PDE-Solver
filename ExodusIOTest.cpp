#include "ExodusIO.hpp"

int main(void) {
    ExodusIO::IO io;
    io.open("data/tet-cube.exo", true);
    io.create("output.exo");
    io.decompose();
}