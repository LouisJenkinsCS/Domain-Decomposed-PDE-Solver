#include "ExodusIO.hpp"

int main(void) {
    ExodusIO::IO io;
    io.open("2blocks.exo", true);
    io.decompose();
}