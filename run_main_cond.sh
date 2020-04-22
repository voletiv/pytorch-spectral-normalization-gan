#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu     # Yoshua pays for your job
#SBATCH --cpus-per-task=8                # Ask for _ CPUs
#SBATCH --gres=gpu:1                     # Ask for _ GPU
#SBATCH --mem=47G                        # Ask for _GB of RAM
#SBATCH --time=20:00:00                  # The job will run for _ hours
#SBATCH -o /scratch/voletivi/slurm-sngan-%j.out  # Write the log in $SCRATCH

norm='batch'
save='/home/voletivi/scratch/sngan_christiancosgrove_cifar10/exp'

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

date
nvidia-smi

module load python/3.6
source ~/myenvs/vode/bin/activate

python main_cond.py \
        --data_dir '/home/voletivi/scratch/Datasets/CIFAR10' \
        --out_dir $save \
        --norm $norm \
        --model 'resnet' \
        --loss 'hinge'

date
