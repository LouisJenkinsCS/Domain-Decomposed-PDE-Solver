#include "ExodusIO.hpp"
#include <Teuchos_CommandLineProcessor.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Ifpack2_Factory.hpp>
#include <sstream>
#include <sys/time.h>

/*
    MueLu crashes in Amesos' 'transpose' function, so we use IFPACK2 instead.

    Developer Note: MueLu and IFPACK2 both provide what are known as 'preconditioners', which
    essentially is a matrix applied to either side of the linear equation (in this case the right-hand side)
    and has the effect that it will reduces the _condition number_ of the function (matrix) it is applied to.
    The condition number is a measure of how much a change in the input results in a change in the output;
    in a well-conditioned function, a slight change in the input will result in a slight change in the output;
    in a poorly-conditioned function, a slight change in the input will result in a _substantial_ change in the output.
    Hence the need to condition the matrix, since the goal is to reach convergence (i.e. think gradient descent).

    Developer Note: An explicit `printMultiVector` and `printCrsMatrix` exists as the `describe` function only prints out
    values at rank 0, and so that the combined matrix can be seen in the proper order. MPI processes, even if they flush and
    use barriers, can result in reordering of the output that may result in non-sensical output or even improperly interleaved
    output. To remedy this, each MPI process outputs the data structures along side the timestamp that each process has output
    the given row/column. Barriers are used to ensure that the timestamps reflect the desired ordering, and gets reordered via
    `mpi_output_combiner.py`.
*/

uint64_t getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

// Prints out a CrsMatrix; each process goes row-by-row and prints out
// its row values to the `output` stream. The output should be coalesced via `mpi_output_combiner.py`
void printCrsMatrix(const Teuchos::RCP<const Tpetra::CrsMatrix<>> matrix, std::ofstream &output, bool sparse=true) {
    auto rank = Tpetra::getDefaultComm()->getRank();
    auto ranks = Tpetra::getDefaultComm()->getSize();
    auto rows = matrix->getGlobalNumRows();
    auto map = matrix->getRowMap();

    for (int row = 0; row <= rows; row++) {
        if (map->isNodeGlobalElement(row)) {
            output << row << ": [";
            Teuchos::Array<Tpetra::CrsMatrix<>::global_ordinal_type> cols(matrix->getNumEntriesInGlobalRow(row));
            Teuchos::Array<Tpetra::CrsMatrix<>::scalar_type> vals(matrix->getNumEntriesInGlobalRow(row));
            size_t sz;
            matrix->getGlobalRowCopy(row, cols, vals, sz);
            std::vector<std::pair<Tpetra::CrsMatrix<>::global_ordinal_type, Tpetra::CrsMatrix<>::scalar_type>> entries;
            for (size_t i = 0; i < cols.size(); i++) entries.push_back(std::make_pair(cols[i], vals[i]));
            std::sort(entries.begin(), entries.end());
            for (int i = 0; i < cols.size(); i++) {
                if (i) output << ",";
                output << "(" << entries[i].first << "," << entries[i].second << ")";
            }
            output << "] ~" << getTime() << "~" << std::endl;
            std::flush(std::cout);
        }
        Teuchos::barrier(*Tpetra::getDefaultComm());
    }
}

// Prints out a multi-vector with a single column; each process goes row-by-row and prints out
// its value to the `output` stream. The output should be coalesced via `mpi_output_combiner.py`
void printMultiVector(const Teuchos::RCP<const Tpetra::MultiVector<>> X, std::ofstream &output) {
    auto rank = Tpetra::getDefaultComm()->getRank();
    auto ranks = Tpetra::getDefaultComm()->getSize();
    auto N = X->getGlobalLength();
    auto map = X->getMap();
    Teuchos::barrier(*Tpetra::getDefaultComm());
    for (int col = 0; col < X->getNumVectors(); col++) {
        auto vec = X->getVector(col);
        auto copy = vec->get1dView();

        for (int row = 0; row <= vec->getGlobalLength(); row++) {
            if (vec->getMap()->isNodeGlobalElement(row)) {
                output << row << ": [" << copy[vec->getMap()->getLocalElement(row)] << "] ~" << getTime() << "~" << std::endl;
            }
            Teuchos::barrier(*Tpetra::getDefaultComm());
        }
        Teuchos::barrier(*Tpetra::getDefaultComm());
    }
}

// Solves for ax=b
void belosSolver(const Teuchos::RCP<Tpetra::CrsMatrix<>> _A, const Teuchos::RCP<Tpetra::MultiVector<> > X, const Teuchos::RCP<const Tpetra::MultiVector<> > B, size_t numIterations, double tolerance, ExodusIO::IO &io, bool verbose) {
    Teuchos::RCP<Tpetra::Operator<>> A = _A;
    Teuchos::RCP<Tpetra::Operator<>> _M;
    Teuchos::RCP<const Tpetra::Operator<>> M;
    Teuchos::ParameterList plist;
    Ifpack2::Factory ifpackFactory;
    Teuchos::RCP<Ifpack2::Preconditioner<>> prec = ifpackFactory.create("ILUT", _A.getConst());
    prec->setParameters (plist);
    prec->initialize();
    prec->compute();
    M = prec.getConst();
    
    // Set up the solver
    size_t iterations = 0;
    auto solverOptions = Teuchos::rcp(new Teuchos::ParameterList());
    solverOptions->set ("Maximum Iterations", (int) 1);
    solverOptions->set ("Convergence Tolerance", tolerance);
    
    Belos::SolverFactory<double, Tpetra::MultiVector<>, Tpetra::Operator<>> factory;
    Teuchos::RCP<Belos::SolverManager<double, Tpetra::MultiVector<>, Tpetra::Operator<>>> solver = factory.create("GMRES", solverOptions);
    
    auto problem = Teuchos::rcp(new Belos::LinearProblem<double, Tpetra::MultiVector<>, Tpetra::Operator<>>(A, X, B));
    problem->setRightPrec(M);
    problem->setProblem();
    solver->setProblem(problem);
    Belos::ReturnType result;
    // TODO: This will not work!
    for (int i = 0; i < numIterations; i++) {
        result = solver->solve();
        io.writeSolution(X, i, verbose);
        iterations++;
        if (result == Belos::Converged) {
            if (Tpetra::getDefaultComm()->getRank() == 0) {
                std::cout << "The Belos solve took " << i << " iteration(s) to reach "
                "a relative residual tolerance of " << solver->achievedTol() << "." << std::endl;
            }
            break;
        } else if (solver->achievedTol() <= tolerance) {
            if (Tpetra::getDefaultComm()->getRank() == 0) {
                std::cout << "The Belos solve took " << iterations << " iteration(s), but did not converge. Achieved tolerance = "
                        << solver->achievedTol() << "." << std::endl;
            }
            break;
        }
        solver->reset(Belos::ResetType::Problem);
        solver->setProblem(problem);
    }
    
    if (Tpetra::getDefaultComm()->getRank() == 0) {
        std::cout << "The Belos solve took " << iterations << " iteration(s), but did not converge. Achieved tolerance = "
                << solver->achievedTol() << "." << std::endl;
    }
}

int main(int argc, char *argv[]) {
    Tpetra::ScopeGuard tscope(&argc, &argv);
    {
        Teuchos::CommandLineProcessor cmdp(false, false);
        std::string inputFile = "";
        std::string outputPrefix = "mpi-proc-";
        std::string solution = "solution.exo";
        bool verbose = false;
        size_t numIterations = 300;
        size_t reportAfterIterations = 10;
        double tolerance = 1e-14;
        cmdp.setOption("input", &inputFile, "Exodus file to decompose", true);
        cmdp.setOption("verbose", "no-verbose", &verbose, "Whether or not to be verbose with output (usefulf or debugging) [default=false]");
        cmdp.setOption("iterations", &numIterations, "Maximum number of iterations that the solver will run for [default=300]");
        cmdp.setOption("reportAfterIterations", &reportAfterIterations, "Number of iterations between checks for convergence, and if --verbose, how often information gets logged [default=10]");
        cmdp.setOption("tolerance", &tolerance, "Tolerance for convergence [default=1e-14]");
        cmdp.setOption("outputPrefix", &outputPrefix, "Prefix for output files, will create prefix for output $PREFIX-$RANK.out [default=mpi-proc-]");
        cmdp.setOption("solution", &solution, "The name of the file that contains the solved steady-state heat equation [default='solution.exo']");
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

        std::stringstream outputFile;
        outputFile << outputPrefix << rank << ".out";
        std::ofstream output(outputFile.str());
        if (!output.good()) {
            std::cerr << "Process #" << rank << ": Failed to open output file '" << outputFile.str() << "'" << std::endl;
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

        if (rank == 0) std::cout << "Printing out CrsMatrix" << std::endl;
        auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
        A->describe(*ostr, verbose ? Teuchos::EVerbosityLevel::VERB_EXTREME : Teuchos::EVerbosityLevel::VERB_MEDIUM);
        output << "[Laplacian: A]" << std::endl;
        printCrsMatrix(A, output);

        // Invoke solver...
        // The linear equation being solved for is 'AX = B', where A is the laplacian matrix,
        // X is the solution (with initial randomized guess), and B is the desired values to converge to.
        if (rank == 0)  std::cout << "Printing out multivector B" << std::endl;
        output << "[RHS: B]" << std::endl;
        printMultiVector(B, output);

        srand(time(NULL));
        X->randomize();
        if (rank == 0) {
            if (!io.create(solution)) {
                std::cerr << "Process #" << rank << ": Failed to create output file '" << solution << "'" << std::endl;
            }
            io.decompose(std::max(2, comm->getSize()), verbose);
        }
        belosSolver(A, X, B, numIterations, tolerance, io, verbose);
        if (rank == 0) std::cout << "Printing out multivector X" << std::endl;
        output << "[Solution: X]" << std::endl;
        printMultiVector(X, output);
    }

    return 0;
}