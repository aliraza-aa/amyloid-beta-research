#RUNFILE FOR KATHLEEN

#!/bin/bash --l

#$ -S /bin/bash

#$ -l h_rt=12:00:0

#$ -l mem=1G

#$ -N AB42_WM-nvt-600

#$ -pe mpi 40

#$ -cwd

#$ -m bes

# send email to me
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

gerun gmx_mpi mdrun -s nvt_600_input.tpr -o nvt_600.trr -x nvt_600.xtc -c nvt_600_output.gro -g nvt_600.log -e nvt.edr -v &> nvt_600.log