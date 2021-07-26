#include "ExodusIO.hpp"
#include <Teuchos_CommandLineProcessor.hpp>

// Power method for estimating the eigenvalue of maximum magnitude of
// a matrix.  This function returns the eigenvalue estimate.
//
// We don't intend for you to write your own eigensolvers; the Anasazi
// package provides them.  You should instead see this class as a
// surrogate for a Tpetra interface to a Trilinos package.
//
// TpetraOperatorType: the type of the Tpetra::Operator specialization
//   used to represent the sparse matrix or operator A.
//
// Tpetra::Operator implements a function from one
// Tpetra::(Multi)Vector to another Tpetra::(Multi)Vector.
// Tpetra::CrsMatrix implements Tpetra::Operator; its apply() method
// computes a sparse matrix-(multi)vector multiply.  It's typical for
// numerical algorithms that use Tpetra objects to be templated on the
// type of the Tpetra::Operator specialization.  We do so here, and
// thus demonstrate how you can use the public typedefs in Tpetra
// classes to write generic code.
//
// One could use a templated function here instead of a templated
// class with a static (class) method.  I prefer the class approach
// because one can lift typedefs out of the function into the class.
// It tends to makes the function declaration easier to read.
template <class TpetraOperatorType>
class PowerMethod {
public:
    typedef typename TpetraOperatorType::scalar_type scalar_type;
    typedef typename TpetraOperatorType::local_ordinal_type local_ordinal_type;
    typedef typename TpetraOperatorType::global_ordinal_type global_ordinal_type;
    typedef typename TpetraOperatorType::node_type node_type;
    // The type of a Tpetra vector with the same template parameters as
    // those of TpetraOperatorType.
    typedef Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> vec_type;
    // The type of the norm of the above Tpetra::Vector specialization.
    typedef typename vec_type::mag_type magnitude_type;
    // Run the power method and return the eigenvalue estimate.
    //
    // Input arguments:
    //
    // A: The sparse matrix or operator, as a Tpetra::Operator.
    // niters: Maximum number of iterations of the power method.
    // tolerance: If the 2-norm of the residual A*x-lambda*x (for the
    //   current eigenvalue estimate lambda) is less than this, stop
    //   iterating.  The complicated expression for the type ensures that
    //   if the type of entries in the matrix A (scalar_type) is complex,
    //   then we'll be using a real-valued type ("magnitude") for the
    //   tolerance.  (You can't compare complex numbers using less than,
    //   so you can't test for convergence using a complex number.)
    // out: output stream to which to print the current status of the
    //   power method.
    static scalar_type
    run (const TpetraOperatorType& A, const int niters, const magnitude_type tolerance, std::ostream& out) {
        using std::endl;
        typedef Teuchos::ScalarTraits<scalar_type> STS;
        typedef Teuchos::ScalarTraits<magnitude_type> STM;
        const int myRank = A.getMap ()->getComm ()->getRank ();
        // Create three vectors for iterating the power method.  Since the
        // power method computes z = A*q, q should be in the domain of A and
        // z should be in the range.  (Obviously the power method requires
        // that the domain and the range are equal, but it's a good idea to
        // get into the habit of thinking whether a particular vector
        // "belongs" in the domain or range of the matrix.)  The residual
        // vector "resid" is of course in the range of A.
        vec_type q (A.getDomainMap ());
        vec_type z (A.getRangeMap ());
        vec_type resid (A.getRangeMap ());
        // Fill the iteration vector z with random numbers to start.
        // Don't have grand expectations about the quality of our
        // pseudorandom number generator, but it is usually good enough
        // for eigensolvers.
        z.randomize ();
        // lambda: Current approximation of the eigenvalue of maximum magnitude.
        // normz: 2-norm of the current iteration vector z.
        // residual: 2-norm of the current residual vector 'resid'.
        //
        // Teuchos::ScalarTraits defines what zero and one means for any
        // type.  Most number types T know how to turn a 0 or a 1 (int)
        // into a T.  I have encountered some number types in C++ that do
        // not.  These tend to be extended-precision types that define
        // number operators and know how to convert from a float or
        // double, but don't have conversion operators for int.  Thus,
        // using Teuchos::ScalarTraits makes this code maximally general.
        scalar_type lambda = STS::zero ();
        magnitude_type normz = STM::zero ();
        magnitude_type residual = STM::zero ();
        const scalar_type one  = STS::one ();
        const scalar_type zero = STS::zero ();
        // How often to report progress in the power method.  Reporting
        // progress requires computing a residual, which can be expensive.
        // However, if you don't compute the residual often enough, you
        // might keep iterating even after you've converged.
        const int reportFrequency = 10;
        // Do the power method, until the method has converged or the
        // maximum iteration count has been reached.
        for (int iter = 0; iter < niters; ++iter) {
            normz = z.norm2 ();       // Compute the 2-norm of z
            q.scale (one / normz, z); // q := z / normz
            A.apply (q, z);           // z := A * q
            lambda = q.dot (z);       // Approx. max eigenvalue
            // Compute and report the residual norm every reportFrequency
            // iterations, or if we've reached the maximum iteration count.
            if (iter % reportFrequency == 0 || iter + 1 == niters) {
                resid.update (one, z, -lambda, q, zero); // z := A*q - lambda*q
                residual = resid.norm2 (); // 2-norm of the residual vector
                if (myRank == 0) {
                    out << "Iteration " << iter << ":" << endl
                        << "- lambda = " << lambda << endl
                        << "- ||A*q - lambda*q||_2 = " << residual << endl;
                }
            }
            if (residual < tolerance) {
                if (myRank == 0) {
                    out << "Converged after " << iter << " iterations" << endl;
                }
                break;
            } else if (iter+1 == niters) {
                if (myRank == 0) {
                    out << "Failed to converge after " << niters
                        << " iterations" << endl;
                }
                break;
            }
        }
        return lambda;
    }
};

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

        Teuchos::RCP<Tpetra::CrsMatrix<>> ret;
        if (!io.getMatrix(&ret, verbose)) {
            std::cerr << "Process #" << rank <<  ": Failed to getMatrix!!" << std::endl;
            return EXIT_FAILURE;
        }

        auto ostr = Teuchos::VerboseObjectBase::getDefaultOStream();
        ret->describe(*ostr, verbose ? Teuchos::EVerbosityLevel::VERB_EXTREME : Teuchos::EVerbosityLevel::VERB_MEDIUM);
        // Run the power method and report the result.
        auto lambda = PowerMethod<Tpetra::CrsMatrix<>>::run(*ret, 500, 1.0e-2, std::cout);
        std::cout << "Lambda = " << lambda << std::endl;
    }

    return 0;
}