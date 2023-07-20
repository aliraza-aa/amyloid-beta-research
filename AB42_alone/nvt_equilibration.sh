#RUNFILE FOR KATHLEEN

#!/bin/bash --l

#$ -S /bin/bash

#$ -l h_rt=24:00:0

#$ -l mem=1G

#$ -N AB42-nvt-equilibration

#$ -pe mpi 240

#$ -cwd

WORKDIR=`pwd`

 

export OMP_NUM_THREADS=1

 

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/ucbtghe/opt/plumed2-2.6.0/lib"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/ucbtghe/opt/gromacs-2019.4/lib64"

export PATH="$PATH:/home/ucbtghe/opt/plumed2-2.6.0/bin"

export PATH="$PATH:/home/ucbtghe/opt/gromacs-2019.4/bin"

# export PLUMED_NUM_THREADS=$OMP_NUM_THREADS

 

cd $WORKDIR

 

#RESTART

gerun gmx_mpi_d mdrun -s system/simulations/nvt_input -multidir system/simulations/confirmation{0..47} -v -maxh 22 -cpi state -noappend -cpt 60 -cpnum -ntomp $OMP_NUM_THREADS &> log