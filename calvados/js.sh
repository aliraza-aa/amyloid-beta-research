#!/bin/bash --l

#$ -S /bin/bash

#$ -l h_rt=1:00:0

#$ -l mem=1G

#$ -N calvados-test

#$ -pe mpi 80

#$ -cwd

#$ -m bes

# send email to me
#$ -M zcbtar9@ucl.ac.uk

WORKDIR=`pwd`

module load python3/recommended
source calv/bin/activate

cd $WORKDIR


gerun python calv.py &> calvados.log
