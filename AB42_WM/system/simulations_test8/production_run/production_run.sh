#RUNFILE FOR KATHLEEN

#!/bin/bash --l

#$ -S /bin/bash

#$ -l h_rt=1:00:0

#$ -l mem=1G

#$ -pe mpi 2

#$ -N AB42_WM-production-test8-1.1

#$ -cwd


WORKDIR=`pwd`



 
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
module load plumed/2.7.2/gnu-10.2.0
module load gromacs/2021.3/plumed/gnu-10.2.0

# export PLUMED_NUM_THREADS=$OMP_NUM_THREADS

 

cd $WORKDIR
 

#RESTART

mpirun -np 2 gmx_mpi mdrun -s production_run_input.tpr -multidir ../replica{0..1} -plumed ../../../plumed/plumed.dat -o production_run.trr -x production_run.xtc -c production_run_output.gro -g production_run.log -e production_run.edr -v -noappend -cpt 60 -cpnum -maxh 0.9 -nt 2 &> production_run1.log
