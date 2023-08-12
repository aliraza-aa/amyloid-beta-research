#RUNFILE FOR KATHLEEN

#!/bin/bash --l

#$ -S /bin/bash

#$ -l h_rt=12:00:0

#$ -l mem=1G

#$ -N AB42-em-2

#$ -pe mpi 48

#$ -cwd

# send email to me
#$ -M zcbtar9@ucl.ac.uk

WORKDIR=`pwd`

 

export OMP_NUM_THREADS=1

 
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
module use ~/modulefiles
module load plumed-2.9.0-gnu-10.2.0
module load gromacs-2022.5-plumed-2.9.0

# export PLUMED_NUM_THREADS=$OMP_NUM_THREADS

 

cd $WORKDIR
 

#RESTART

gerun gmx_mpi_d mdrun -s nvt_input.tpr -multidir system/simulations/confirmation{0..47} -o nvt.trr -x nvt.xtc -c nvt_output.gro -g nvt.log -e nvt.edr -v -maxh 10 -ntomp $OMP_NUM_THREADS &> nvt_run.log

# gerun gmx_mpi_d mdrun -s em_input.tpr -o system/simulations/confirmation0/nvt.trr -x system/simulations/confirmation0/nvt.xtc -c system/simulations/confirmation0/nvt_output.gro -g system/simulations/confirmation0/nvt.log -e system/simulations/confirmation0/nvt.edr -v -maxh 10