#include "ExodusIO.hpp"

int main(int argc, char *argv[]) {
    Tpetra::ScopeGuard tscope(&argc, &argv);
    {
        ExodusIO::IO io;
        if (!io.open("data/tet-cube.exo", true)) {
            std::cerr << "Failed to open Exodus file!" << std::endl;
            return -1;
        }
        if (!io.create("output.exo")) {
            std::cerr << "Failed to create output Exodus file!" << std::endl;
            return -1;
        }
        Teuchos::RCP<Tpetra::CrsMatrix<>> ret;
        if (!io.getMatrix(&ret)) {
            std::cerr << "Failed to getMatrix!!" << std::endl;
            return -1;
        }
    }

    return 0;
}