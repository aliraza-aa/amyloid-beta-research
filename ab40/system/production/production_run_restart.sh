#RUNFILE FOR KATHLEEN

#!/bin/bash --l

#$ -S /bin/bash

#$ -l h_rt=24:00:0

#$ -l mem=1G

#$ -pe mpi 400

#$ -N AB42_WM-production-run-1.15

#$ -cwd

#$ -m bes

#$ -M zcbtar9@ucl.ac.uk

WORKDIR=`pwd`

 
module unload -f compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load mpi/openmpi/4.0.5/gnu-10.2.0
module load openblas/0.3.13-serial/gnu-10.2.0
module load python3/3.9-gnu-10.2.0
module load libmatheval
module load flex
module use ~/modulefiles
module load plumed-2.9.0-gnu-10.2.0
module load gromacs-2022.5-plumed-2.9.0-sp

# export PLUMED_NUM_THREADS=$OMP_NUM_THREADS

 

cd $WORKDIR
 

#RESTART

gerun gmx_mpi mdrun -cpi state -multidir r{0..39} -plumed ../../../plumed/plumed.dat -deffnm production -v -noappend -cpt 60 -cpnum -maxh 23 &> production_run.log