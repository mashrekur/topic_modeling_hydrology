#!/bin/bash

#SBATCH --job-name=diversity_paperwise                                # Job name
#SBATCH --ntasks=40                                   # Number of CPUs
#SBATCH --time=10000:00:00                               # Time limit hrs:min:sec
#SBATCH --output=log_files/run_diversity.log          # Standard output and error log
#SBATCH --exclude=gpu01                               # Don't hog the GPU node


# memory
ulimit -s unlimited

# location
/home/mashrekur/nlp_abstracts/topic_modeling_hydrology
pwd

# load conda environment
source /home/mashrekur/.bashrc


# run code
python3 paperwise_diversity_prototype.py 
