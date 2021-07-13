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
        ExodusIO::IO io;
        if (!io.open(inputFile, true)) {
            std::cerr << "Process #" << rank << ": Failed to open input Exodus file '" << inputFile << "'" << std::endl;
            return EXIT_FAILURE;
        }

        Teuchos::RCP<Tpetra::CrsMatrix<>> ret;
        if (!io.getDual(&ret, verbose)) {
            std::cerr << "Process #" << rank <<  ": Failed to getDual!!" << std::endl;
            return EXIT_FAILURE;
        }

        auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
        ret->describe(*ostr, verbose ? Teuchos::EVerbosityLevel::VERB_EXTREME : Teuchos::EVerbosityLevel::VERB_MEDIUM);
    }

    return 0;
}