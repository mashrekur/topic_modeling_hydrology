#!/bin/bash

#SBATCH --job-name=dtm                                # Job name
#SBATCH --ntasks=40                                   # Number of CPUs
#SBATCH --time=72:00:00                               # Time limit hrs:min:sec
#SBATCH --output=log_files/dtm_run_slurm.log          # Standard output and error log
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
python3 DTM.py 
