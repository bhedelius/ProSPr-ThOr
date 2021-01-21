#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH --gpus=1
#SBATCH -p m9g

restart_train() {
    echo "Closing, submitting new job"
    sbatch train.sh
    date
    exit 1
}

trap 'restart_train' TERM

source /fslhome/bryceeh/.bash_profile
echo "Activating conda"
conda activate thesis
echo "Starting training"
python train.py

sbatch train.sh
exit 1
