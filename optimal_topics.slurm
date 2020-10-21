#!/bin/bash

#SBATCH --job-name=optimal_topics              # Job name
#SBATCH --ntasks=40                    # Run on multiple CPUs
#SBATCH --time=172:00:00               # Time limit hrs:min:sec
#SBATCH --output=log_files/optimal_topics_slurm.log          # Standard output and error log
#SBATCH --exclude=gpu01               #Exclude the GPU


#memory
ulimit -s unlimited


#location
cd  /home/mashrekur/nlp_abstracts/topic_modeling_hydrology
pwd

# load conda environment
source /home/mashrekur/.bashrc
conda activate nlp

# run code
python3 1_optimal_topics.py

