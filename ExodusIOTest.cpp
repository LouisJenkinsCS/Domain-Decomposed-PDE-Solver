#include "ExodusIO.hpp"

int main(void) {
    ExodusIO::IO io;
    io.open("data/bolted_bracket.exo", true);
    io.decompose();
}