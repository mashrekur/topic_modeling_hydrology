#!/bin/bash

#SBATCH --job-name=top_jsd                                # Job name
#SBATCH --ntasks=40                                   # Number of CPUs
#SBATCH --time=10000000:00:00                               # Time limit hrs:min:sec
#SBATCH --output=../log_files/interjsd.log          # Standard output and error log
#SBATCH --exclude=gpu01                               # Don't hog the GPU node


# memory
ulimit -s unlimited

# location
/home/mashrekur/nlp_abstracts/topic_modeling_hydrology/topicwise_matrices
pwd

# load conda environment
source /home/mashrekur/.bashrc
conda activate nlp

# run code
python3 intertopic_jsd.py  
