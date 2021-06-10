// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER
#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_XpetraCrsMatrixAdapter.hpp>
#include <Zoltan2_XpetraMultiVectorAdapter.hpp>
#include <iostream>

template<class CrsMatrixType>
void outputMatrix(const Teuchos::RCP<const Teuchos::Comm<int> >& comm, Teuchos::RCP<CrsMatrixType>& matrix) {
    using Teuchos::arcp;
    using Teuchos::ArrayRCP;
    using Teuchos::ArrayView;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::Time;
    using Teuchos::TimeMonitor;
    using Teuchos::tuple;
    typedef typename CrsMatrixType::scalar_type scalar_type;
    typedef typename CrsMatrixType::local_ordinal_type LO;
    typedef typename CrsMatrixType::global_ordinal_type GO;
    ArrayView<const GO> myGlobalElements = matrix->getRowMap()->getNodeElementList ();
    auto map = matrix->getRowMap();
    typedef typename ArrayView<const GO>::const_iterator iter_type;
    for (iter_type it = myGlobalElements.begin(); it != myGlobalElements.end(); ++it) {
        const LO i_local = *it;
        const GO i_global = map->getGlobalElement (i_local);
        Teuchos::ArrayView<const LO> ixs;
        Teuchos::ArrayView<const double> vals;
        matrix->getLocalRowView(i_global, ixs, vals);
        std::cout << "Rank #" << comm->getRank() << ": Index " << i_global << " has neighbors: " << ixs << std::endl;
    }
}

template<class CrsMatrixType>
Teuchos::RCP<CrsMatrixType>
redistributeMatrix (const Teuchos::RCP<const Teuchos::Comm<int> >& comm, Teuchos::RCP<const CrsMatrixType>& matrix)
{
    using Teuchos::TimeMonitor;
    using Teuchos::Time;


    typedef Zoltan2::XpetraCrsMatrixAdapter<CrsMatrixType> MatrixAdapter_t;
    Teuchos::RCP<Time> timer = TimeMonitor::getNewCounter ("Sparse matrix redistribution");
    TimeMonitor monitor (*timer);

    Teuchos::ParameterList param;
    param.set("partitioning_approach", "partition");
    param.set("algorithm", "parmetis");
    MatrixAdapter_t adapter(matrix);
    Zoltan2::PartitioningProblem<MatrixAdapter_t> problem(&adapter, &param);
    std::cout << "Solving Problem..." << std::endl;

    try {
        problem.solve();
    }
    catch (std::exception &e) {
        std::cout << "Exception returned from solve(). " << e.what() << std::endl;
        exit(-1);
    }

    std::cout << "Applying Solution to Problem..." << std::endl;
    Teuchos::RCP<CrsMatrixType> redistribMatrix;
    adapter.applyPartitioningSolution(*matrix, redistribMatrix,
                                    problem.getSolution());
    return redistribMatrix;
}
// Create and return a simple example CrsMatrix, with row distribution
// over the given Map.
//
// CrsMatrixType: The type of the Tpetra::CrsMatrix specialization to use.
template<class CrsMatrixType>
Teuchos::RCP<const CrsMatrixType>
createMatrix (const Teuchos::RCP<const Teuchos::Comm<int> >& comm, const Teuchos::RCP<const typename CrsMatrixType::map_type>& map)
{
  using Teuchos::arcp;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::Time;
  using Teuchos::TimeMonitor;
  using Teuchos::tuple;
  typedef Tpetra::global_size_t GST;
  // Fetch typedefs from the Tpetra::CrsMatrix.
  typedef typename CrsMatrixType::scalar_type scalar_type;
  typedef typename CrsMatrixType::local_ordinal_type LO;
  typedef typename CrsMatrixType::global_ordinal_type GO;
  // Create a timer for sparse matrix creation.
  RCP<Time> timer = TimeMonitor::getNewCounter ("Sparse matrix creation");
  // Time the whole scope of this routine, not counting timer lookup.
  TimeMonitor monitor (*timer);
  // Create a Tpetra::Matrix using the Map, with dynamic allocation.
  // Row is distributed by map, with 11 columns distributed over the same
  // process that owns that row. (11x11 matrix representing mesh)
  RCP<CrsMatrixType> A (new CrsMatrixType (map, 11));
  if (comm->getRank() == 0) {
    A->insertGlobalValues(1, tuple(2L,4L,5L), tuple(1.0,1.0,1.0));
    A->insertGlobalValues(2, tuple(1L,3L,5L,6L), tuple(1.0,1.0,1.0,1.0));
    A->insertGlobalValues(3, tuple(2L,6L,7L), tuple(1.0,1.0,1.0));
    A->insertGlobalValues(4, tuple(1L,5L,9L), tuple(1.0,1.0,1.0));
    A->insertGlobalValues(5, tuple(1L,2L,4L,6L,9L,10L), tuple(1.0,1.0,1.0,1.0,1.0,1.0));
    A->insertGlobalValues(6, tuple(2L,3L,5L,7L,8L,10L,11L), tuple(1.0,1.0,1.0,1.0,1.0,1.0,1.0));
    A->insertGlobalValues(7, tuple(3L,6L,8L), tuple(1.0,1.0,1.0));
    A->insertGlobalValues(8, tuple(6L,7L,11L), tuple(1.0,1.0,1.0));
    A->insertGlobalValues(9, tuple(4L,5L,10L), tuple(1.0,1.0,1.0));
    A->insertGlobalValues(10, tuple(5L,6L,9L,11L), tuple(1.0,1.0,1.0,1.0));
    A->insertGlobalValues(11, tuple(6L,8L,10L), tuple(1.0,1.0,1.0));
  } 
  // Finish up the matrix.
  A->fillComplete ();
  return A;
}

void
example (const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
         std::ostream& out)
{
  using std::endl;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::Time;
  using Teuchos::TimeMonitor;
  typedef Tpetra::global_size_t GST;
  // Set up Tpetra typedefs.
  typedef Tpetra::CrsMatrix<> crs_matrix_type;
  typedef Tpetra::Map<> map_type;
  typedef Tpetra::Map<>::global_ordinal_type global_ordinal_type;
  const int myRank = comm->getRank ();
  // The global number of rows in the matrix A to create.  We scale
  // this relative to the number of (MPI) processes, so that no matter
  // how many MPI processes you run, every process will have 10 rows.
  const GST numGlobalIndices = 21;
  const global_ordinal_type indexBase = 1;
  // Construct a Map that is global (not locally replicated), but puts
  // all the equations on MPI Proc 0.
  if (myRank == 0) {
    out << "Construct Process 0 Map" << endl;
  }
  RCP<const map_type> procZeroMap;
  {
    const size_t numLocalIndices = (myRank == 0) ? numGlobalIndices : 0;
    procZeroMap = rcp (new map_type (numGlobalIndices, numLocalIndices,
                                     indexBase, comm));
  }
  // Construct a Map that puts approximately the same number of
  // equations on each processor.
  if (myRank == 0) {
    out << "Construct global Map" << endl;
  }
  RCP<const map_type> globalMap =
    rcp (new map_type (numGlobalIndices, indexBase, comm,
                       Tpetra::GloballyDistributed));
  // Create a sparse matrix using procZeroMap.
  if (myRank == 0) {
    out << "Create sparse matrix using Process 0 Map" << endl;
  }
  RCP<const crs_matrix_type> A = createMatrix<crs_matrix_type> (comm, procZeroMap);
  if (myRank == 0) {
      out << "Redistribute sparse matrix" << endl;
  }
  outputMatrix(comm, A);
  RCP<crs_matrix_type> B = redistributeMatrix(comm, A);
  outputMatrix(comm, B);
}

int
main (int argc, char *argv[])
{
  using Teuchos::TimeMonitor;
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
  {
    auto comm = Tpetra::getDefaultComm ();

    const int myRank = comm->getRank ();
    // Make global timer for sparse matrix redistribution.
    // We will use (start and stop) this timer in example().
    example (comm, std::cout); // Run the whole example.
    // This tells the Trilinos test framework that the test passed.
    TimeMonitor::summarize (std::cout);

    if (myRank == 0) {
        std::cout << "End Result: TEST PASSED" << std::endl;
    }
  }
  return 0;
}