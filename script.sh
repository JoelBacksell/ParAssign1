#!/bin/bash -l
#SBATCH -A g2017012
#SBATCH -t 15:00
#SBATCH -N 1

mpirun -np 4 --map-by-core ./FOX 1440

