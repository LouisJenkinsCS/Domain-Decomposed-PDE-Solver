#include "ExodusIO.hpp"
#include <Teuchos_CommandLineProcessor.hpp>

int main(int argc, char *argv[]) {
    Tpetra::ScopeGuard tscope(&argc, &argv);
    {
        Teuchos::CommandLineProcessor cmdp(false, false);
        std::string inputFile = "";
        bool verbose = false;
        cmdp.setOption("input", &inputFile, "Exodus file to decompose");
        cmdp.setOption("verbose", "no-verbose", &verbose, "Whether or not to be verbose with output (usefulf or debugging) [default=false]");
        cmdp.parse(argc, argv);
        auto comm = Tpetra::getDefaultComm();
        int rank = Teuchos::rank(*comm);
        if (inputFile.empty()) {
            if (rank == 0) std::cerr << "No input file was provided; use the '--input' parameter!" << std::endl;
            return EXIT_FAILURE;
        }
        if (Teuchos::size(*comm) == 1) {
            std::cerr << "Requires at least 2 MPI Processors for this Example!" << std::endl;
            return EXIT_FAILURE;
        }
        ExodusIO::IO io;
        if (!io.open(inputFile, true)) {
            std::cerr << "Process #" << rank << ": Failed to open input Exodus file '" << inputFile << "'" << std::endl;
            return EXIT_FAILURE;
        }

        Teuchos::RCP<Tpetra::CrsMatrix<>> A;
        Teuchos::RCP<Tpetra::MultiVector<>> X;
        Teuchos::RCP<Tpetra::MultiVector<>> B;
        std::map<int, std::set<idx_t>> nodeSetMap;
        if (!io.assemble(&A, &X, &B, verbose)) {
            std::cerr << "Process #" << rank <<  ": Failed to getMatrix!!" << std::endl;
            return EXIT_FAILURE;
        }
    }

    return 0;
}