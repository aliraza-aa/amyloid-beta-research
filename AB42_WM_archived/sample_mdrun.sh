#RUNFILE FOR KATHLEEN

#!/bin/bash --login

#$ -S /bin/bash

#$ -l h_rt=48:00:0

#$ -l mem=1G

#$ -N pep_C26m

#$ -pe mpi 240

#$ -cwd

 

WORKDIR=`pwd`

 

export OMP_NUM_THREADS=1

 

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/ucbtghe/opt/plumed2-2.6.0/lib"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/ucbtghe/opt/gromacs-2019.4/lib64"

export PATH="$PATH:/home/ucbtghe/opt/plumed2-2.6.0/bin"

export PATH="$PATH:/home/ucbtghe/opt/gromacs-2019.4/bin"

export PLUMED_NUM_THREADS=$OMP_NUM_THREADS

 

cd $WORKDIR

 

#RESTART

mpirun -n 240 gmx_mpi mdrun -npme 1 -s topol -multidir r{0..47} -v -maxh 22 -cpi state -noappend -cpt 60 -cpnum -ntomp $OMP_NUM_THREADS &> log