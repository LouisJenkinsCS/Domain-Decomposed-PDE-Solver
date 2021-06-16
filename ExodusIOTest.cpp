#include "ExodusIO.hpp"

int main(void) {
    ExodusIO::IO io;
    io.open("data/rectangle-tris.exo", true);
    io.create("output.exo");
    io.decompose();
}