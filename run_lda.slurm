#!/bin/bash

#SBATCH --job-name=lda_new                                # Job name
#SBATCH --ntasks=40                                   # Number of CPUs
#SBATCH --time=172:00:00                               # Time limit hrs:min:sec
#SBATCH --output=log_files/lda_run_slurm.log          # Standard output and error log
#SBATCH --exclude=gpu01                               # Don't hog the GPU node


# memory
ulimit -s unlimited

# location
/home/mashrekur/nlp_abstracts/topic_modeling_hydrology
pwd

# load conda environment
source /home/mashrekur/.bashrc
conda activate nlp

# run code
python3 2_run_lda.py 
