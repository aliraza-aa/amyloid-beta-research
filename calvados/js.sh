#!/bin/bash --l

#$ -S /bin/bash

#$ -l h_rt=1:00:0

#$ -l mem=1G

#$ -N ab42-111-MPI

#$ -pe mpi 42

#$ -cwd

#$ -m bes

# send email to me
#$ -M zcbtar9@ucl.ac.uk

WORKDIR=`pwd`

module load python3/recommended
source ~/calvados/calv/bin/activate

cd $WORKDIR


gerun python calv_modified.py --name ab42-111-MPI --Hc6 1 --Hc13 1 --Hc14 1 &> ab42-111-MPI.log
