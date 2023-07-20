#!/usr/bin/env bash

###############################################
# test file to see if plumed is imported correctly

set -e

package_name="gromacs"
package_version="2022.5"
package_variant="plumed"
package_description="GROMACS is a package for performing molecular dynamics, i.e. simulate the Newtonian equations of motion for systems with hundreds to millions of particles. This version patched with plumed 2.7.2 (with libmatheval)."
SRC_ARCHIVE=${SRC_ARCHIVE:-ftp://ftp.gromacs.org/pub/gromacs/gromacs-$package_version.tar.gz}

cluster=${cluster:-$(/shared/ucl/apps/cluster-bin/whereami)}

# source includes/source_includes.sh
module purge
module load beta-modules
module load gcc-libs/10.2.0
module load cmake
module load compilers/gnu/10.2.0
# On Myriad, module requires UCX
if [ $cluster == "myriad" ]
then
    module load numactl/2.0.12
    module load binutils/2.36.1/gnu-10.2.0
    module load ucx/1.9.0/gnu-10.2.0
fi
module load mpi/openmpi/4.0.5/gnu-10.2.0
module load openblas/0.3.13-serial/gnu-10.2.0
module load python3/3.9-gnu-10.2.0
module load libmatheval/1.1.11
module load flex/2.5.39
# module load plumed/2.7.2/gnu-10.2.0

# importing the local version of plumed rather than the global one
export LD_LIBRARY_PATH=/home/zcbtar9/software/plumed/2.9.0/gnu-10.2.0/openblas/lib:$LD_LIBRARY_PATH
export PATH=/home/zcbtar9/software/plumed/2.9.0/gnu-10.2.0/openblas/bin:$PATH

plumed help

# Check if PLUMED_KERNEL is empty or unset, returns error if it is
: "${PLUMED_KERNEL:?"parameter null or not set, need to load a plumed module"}"
