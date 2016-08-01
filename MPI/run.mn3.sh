#!/bin/bash
#BSUB -n 4
#BSUB -J spike_asyn_back
#BSUB -W 00:05
#BSUB -oo job.out
#BSUB -eo job.err
#BSUB -R "span[ptile=2]"
#BSUB -x
#BSUB -q bsc_debug
date
source environment.sh
date

export OMP_NUM_THREADS=8
mpirun ./main
date
