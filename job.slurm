#!/bin/bash

#SBATCH --job-name=lda_run              # Job name
#SBATCH --ntasks=40                    # Run on multiple CPUs
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
#SBATCH --output=log_files/lda_run_slurm.log          # Standard output and error log
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
python3 2_run_LDA.py
