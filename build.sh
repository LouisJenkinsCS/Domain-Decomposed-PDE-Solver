#!/bin/bash -v

# Parameters
PARMETIS_INCLUDE=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/parmetis-4.0.3-loyrkrfdly5flgxdnywxibvqtg2kyz76/include
TRILINOS_INCLUDE=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/trilinos-13.0.1-fjd4ka7rh56hi7fnu6m2huhqhz3qzpma/include
PARMETIS_LIB=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/parmetis-4.0.3-loyrkrfdly5flgxdnywxibvqtg2kyz76/lib
TRILINOS_LIB=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/trilinos-13.0.1-fjd4ka7rh56hi7fnu6m2huhqhz3qzpma/lib/
PARMETIS_LDFLAGS="-lparmetis"
TRILINOS_LDFLAGS="-ltpetra -ltpetraext -ltpetrainout -lkokkoscore -lkokkosalgorithms -lteuchoscore -lteuchoscomm -lteuchosparameterlist -lzoltan2 -lxpetra -lxpetra-sup -lgaleri-xpetra -lgaleri-epetra -ltpetraclassiclinalg -ltpetraclassiclinalg -lzoltan -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosnumerics"
CCFLAGS="-std=c++17 -O3"

# Build Examples
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $CCFLAGS $TRILINOS_LDFLAGS Example1.cc -o exec/Example1
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $CCFLAGS $TRILINOS_LDFLAGS Example2.cc -o exec/Example2
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $CCFLAGS $TRILINOS_LDFLAGS Example3.cc -o exec/Example3
# mpic++ -I$TRILINOS_INCLUDE -I$PARMETIS_INCLUDE -L$TRILINOS_LIB -L$PARMETIS_LIB $CCFLAGS $TRILINOS_LDFLAGS $PARMETIS_LDFLAGS Example4.cc -o exec/Example4
mpic++ -I$TRILINOS_INCLUDE -I$PARMETIS_INCLUDE -L$TRILINOS_LIB -L$PARMETIS_LIB $CCFLAGS $TRILINOS_LDFLAGS $PARMETIS_LDFLAGS SimpleExampleImpl.cc -o exec/SimpleExampleImpl