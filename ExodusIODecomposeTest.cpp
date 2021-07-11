#include "ExodusIO.hpp"
#include <Teuchos_CommandLineProcessor.hpp>


int main(int argc, char *argv[]) {
    Teuchos::CommandLineProcessor cmdp(false, false);
    std::string inputFile = "";
    std::string outputFile = "";
    size_t numPartitions = 0;
    bool verbose = false;
    cmdp.setOption("input", &inputFile, "Exodus file to decompose");
    cmdp.setOption("output", &outputFile, "Decomposed output derived from input file");
    cmdp.setOption("partitions", &numPartitions, "Number of partitions to decompsoe the original mesh over");
    cmdp.setOption("verbose", "no-verbose", &verbose, "Whether or not to be verbose with output (usefulf or debugging) [default=false]");
    cmdp.parse(argc, argv);
    if (inputFile.empty()) {
        std::cerr << "No input file was provided; use the '--input' parameter!" << std::endl;
        return EXIT_FAILURE;
    }
    if (outputFile.empty()) {
        std::cerr << "No output file was provided; use the '--output' parameter!" << std::endl;
        return EXIT_FAILURE;
    }
    if (numPartitions == 0) {
        std::cerr << "Number of partitions to decompose the mesh has not been provided; use the '--partitions' parameter!" << std::endl;
        return EXIT_FAILURE;
    }
    ExodusIO::IO io;
    if (!io.open(inputFile, true)) {
        std::cerr << "Failed to open input Exodus file '" << inputFile << "'" << std::endl;
        return EXIT_FAILURE;
    }
    if (!io.create(outputFile)) {
        std::cerr << "Failed to create output Exodus file '" << outputFile << "'" << std::endl;
        return EXIT_FAILURE;
    }
    if (!io.decompose(numPartitions, verbose)) {
        std::cerr << "Failed to decompose the input file!" << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}