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
TRILINOS_LDFLAGS="-ltpetra -ltpetraext -ltpetrainout -lkokkoscore -lkokkosalgorithms -lteuchoscore -lteuchoscomm -lteuchosparameterlist -lzoltan2 -lxpetra -lxpetra-sup -lgaleri-xpetra -lgaleri-epetra -ltpetraclassiclinalg -ltpetraclassiclinalg -lzoltan -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosnumerics -lmetis -lmuelu -lbelos -lbelostpetra -lmuelu-adapters -lmuelu-interface -lifpack2"
CCFLAGS="-std=c++17 -O0 -g -ggdb3 -fsanitize=address -Woverloaded-virtual"

# Compile
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $TRILINOS_LDFLAGS -L$PARMETIS_LIB -I$PARMETIS_INCLUDE $CCFLAGS -lexodus -lparmetis ExodusIODecomposeTest.cpp -o exec/ExodusIODecomposeTest
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $TRILINOS_LDFLAGS -L$PARMETIS_LIB -I$PARMETIS_INCLUDE $CCFLAGS -lexodus -lparmetis ExodusDualMatrixTest.cpp -o exec/ExodusDualMatrixTest
# mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $TRILINOS_LDFLAGS -L$PARMETIS_LIB -I$PARMETIS_INCLUDE $CCFLAGS -lexodus -lparmetis ExodusMatrixTest.cpp -o exec/ExodusMatrixTest
mpic++ -I$TRILINOS_INCLUDE -L$TRILINOS_LIB $TRILINOS_LDFLAGS -L$PARMETIS_LIB -I$PARMETIS_INCLUDE $CCFLAGS -lexodus -lparmetis BelosMueLuSolver.cpp -o exec/BelosMueLuSolver
