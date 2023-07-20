module load beta-modules
module unload -f compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load cmake
module load compilers/gnu/10.2.0
module load numactl/2.0.12
module load binutils/2.36.1/gnu-10.2.0
module load ucx/1.9.0/gnu-10.2.0
module load mpi/openmpi/4.0.5/gnu-10.2.0
module load python3/3.9-gnu-10.2.0
module load libmatheval
module load flex
module load plumed-2.9.0-gnu-10.2.0

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/zcbtar9/software/plumed/2.9.0/gnu-10.2.0/openblas/lib"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/zcbtar9/software/gromacs-2022.5-plumed-2.9.0/lib64"

export PATH="$PATH:/home/zcbtar9/software/plumed/2.9.0/gnu-10.2.0/openblas/bin"

export PATH="$PATH:/home/zcbtar9/software/gromacs-2022.5-plumed-2.9.0/bin"