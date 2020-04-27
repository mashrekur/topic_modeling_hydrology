#!/bin/bash

#SBATCH --job-name=opt_top                            # Job name
#SBATCH --ntasks=40                                   # Number of CPUs
#SBATCH --time=72:00:00                               # Time limit hrs:min:sec
#SBATCH --output=log_files/optimal_topics_slurm.log   # Standard output and error log
#SBATCH --exclude=gpu01                               # Don't hog the GPU node


# memory
ulimit -s unlimited

# location
cd /home/gsnearing/topic_modeling_hydrology 
pwd

# load conda environment
source /home/gsnearing/.bashrc
conda activate nlp

# run code
python3 1_optimal_topics.py
