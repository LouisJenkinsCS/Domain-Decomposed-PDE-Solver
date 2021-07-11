#!/usr/local/bin/bash

export COLOR_NC='\e[0m' # No Color
export COLOR_BLACK='\e[0;30m'
export COLOR_GRAY='\e[1;30m'
export COLOR_RED='\e[0;31m'
export COLOR_LIGHT_RED='\e[1;31m'
export COLOR_GREEN='\e[0;32m'
export COLOR_LIGHT_GREEN='\e[1;32m'
export COLOR_BROWN='\e[0;33m'
export COLOR_YELLOW='\e[1;33m'
export COLOR_BLUE='\e[0;34m'
export COLOR_LIGHT_BLUE='\e[1;34m'
export COLOR_PURPLE='\e[0;35m'
export COLOR_LIGHT_PURPLE='\e[1;35m'
export COLOR_CYAN='\e[0;36m'
export COLOR_LIGHT_CYAN='\e[1;36m'
export COLOR_LIGHT_GRAY='\e[0;37m'
export COLOR_WHITE='\e[1;37m'

failure () {
    printf "$COLOR_RED $1 $COLOR_NC\n"
}

success () {
    printf "$COLOR_GREEN $1 $COLOR_NC\n"
}

findPath () {
    echo "Searching for spack package '$1'"
    if [ -z `which spack` ]; then
        failure "Was unable to locate 'spack'! Please install and try again!"
        exit -1
    fi
    path=`spack find -p $1 | egrep -o "/.*$1.*" | head -1`
    if [ -z "$path" ]; then 
        failure "Was unable to locate the spack package '$1'; Try running 'spack install $1'"
        exit -1
    fi
    success "Found spack package '$1' in path '$path'"
    spack load --first $1
    declare -g `echo $1 | egrep -o "\w+" | head -1 | tr [a-z] [A-Z]`_INCLUDE="$path/include/"
    declare -g `echo $1 | egrep -o "\w+" | head -1 | tr [a-z] [A-Z]`_LIB="$path/lib/"
}

# Parameters
findPath "parmetis"
findPath "trilinos"
findPath "netcdf-c"
findPath "exodusii"
findPath "metis"
PARMETIS_LDFLAGS="-lparmetis"
TRILINOS_LDFLAGS="-ltpetra -ltpetraext -ltpetrainout -lkokkoscore -lkokkosalgorithms -lteuchoscore -lteuchoscomm -lteuchosparameterlist -lzoltan2 -lxpetra -lxpetra-sup -lgaleri-xpetra -lgaleri-epetra -ltpetraclassiclinalg -ltpetraclassiclinalg -lzoltan -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosnumerics -lmetis"
CCFLAGS="-std=c++17 -O0 -g -ggdb3 -fsanitize=address"

# PARMETIS_INCLUDE=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/parmetis-4.0.3-loyrkrfdly5flgxdnywxibvqtg2kyz76/include
# TRILINOS_INCLUDE=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/trilinos-13.0.1-fjd4ka7rh56hi7fnu6m2huhqhz3qzpma/include
# NETCDF_INCLUDE=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/netcdf-c-4.8.0-3l5g45insmzgtbxdfk4vr6ynm4vtyvpz/include
# EXODUSII_INCLUDE=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/exodusii-2021-04-05-zfmygn6ucn3me4asovbn7oa7gwkiiyee/include
# METIS_INCLUDE=/usr/local/Cellar/metis/5.1.0/include/
# PARMETIS_LIB=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/parmetis-4.0.3-loyrkrfdly5flgxdnywxibvqtg2kyz76/lib
# TRILINOS_LIB=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/trilinos-13.0.1-fjd4ka7rh56hi7fnu6m2huhqhz3qzpma/lib/
# NETCDF_LIB=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/netcdf-c-4.8.0-3l5g45insmzgtbxdfk4vr6ynm4vtyvpz/lib/
# EXODUSII_LIB=/Users/LouisJenkinsCS/spack/opt/spack/darwin-bigsur-broadwell/apple-clang-12.0.5/exodusii-2021-04-05-zfmygn6ucn3me4asovbn7oa7gwkiiyee/lib/
# METIS_LIB=/usr/local/lib/

# Build Examples
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $CCFLAGS $TRILINOS_LDFLAGS $CCFLAGS Example1.cc -o exec/Example1
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $CCFLAGS $TRILINOS_LDFLAGS $CCFLAGS Example2.cc -o exec/Example2
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $CCFLAGS $TRILINOS_LDFLAGS $CCFLAGS Example3.cc -o exec/Example3
# mpic++ -I$TRILINOS_INCLUDE -I$PARMETIS_INCLUDE -L$TRILINOS_LIB -L$PARMETIS_LIB $CCFLAGS $TRILINOS_LDFLAGS $PARMETIS_LDFLAGS Example4.cc -o exec/Example4
# mpic++ -I$TRILINOS_INCLUDE -I$PARMETIS_INCLUDE -L$TRILINOS_LIB -L$PARMETIS_LIB $CCFLAGS $TRILINOS_LDFLAGS $PARMETIS_LDFLAGS SimpleExampleImpl.cc -o exec/SimpleExampleImpl
# mpic++ -I$EXODUSII_INCLUDE -L$EXODUSII_LIB -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $TRILINOS_LDFLAGS -L$PARMETIS_LIB -L$METIS_LIB $PARMETIS_LDFLAGS -I$PARMETIS_INCLUDE $CCFLAGS -Wall -lexodus ExodusIOTest.cpp -o exec/ExodusIOTest
# install_name_tool -change libmetis.dylib /usr/local/lib/libmetis.dylib exec/ExodusIOTest
# g++-11 -Wall -lexodus -I/usr/local/Cellar/open-mpi/4.1.1_2/include/ -I$NETCDF_INCLUDE -I$EXODUSII_INCLUDE -L$EXODUSII_LIB $CCFLAGS ExodusMWE.cpp -o exec/ExodusMWE
mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $TRILINOS_LDFLAGS -L$PARMETIS_LIB -I$PARMETIS_INCLUDE $CCFLAGS -Wall -lexodus -lparmetis ExodusIODecomposeTest.cpp -o exec/ExodusIODecomposeTest
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $TRILINOS_LDFLAGS -L$PARMETIS_LIB -I$PARMETIS_INCLUDE $CCFLAGS -Wall -lexodus -lparmetis ExodusIOTest.cpp -o exec/ExodusMatrixTest
