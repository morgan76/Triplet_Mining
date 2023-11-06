#!/bin/bash
#SBATCH --job-name=test_job 
#SBATCH --gpus=1                    
#SBATCH --partition A40
#SBATCH --cpus-per-task=15          
#SBATCH --distribution=block:block  
#SBATCH --time=20:00:00            
#SBATCH --output=job_test_output%j.log


set -x

source ~/anaconda3/bin/activate

conda init bash
conda activate segmentation

cd /tsi/data_doctorants/mbuisson/Triplet_Mining

python3 trainer.py mel ../msaf_/datasets/RWC-Pop



