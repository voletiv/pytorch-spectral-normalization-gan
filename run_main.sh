#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu     # Yoshua pays for your job
#SBATCH --cpus-per-task=10                # Ask for _ CPUs
#SBATCH --gres=gpu:1                     # Ask for _ GPU
#SBATCH --mem=47G                        # Ask for _GB of RAM
#SBATCH --time=20:00:00                  # The job will run for _ hours
#SBATCH -o /scratch/voletivi/slurm-sngan-%j.out  # Write the log in $SCRATCH

norm='batch'
save='/home/voletivi/scratch/sngan_christiancosgrove_cifar10/exp'

epochs=1000
bs=256
di=5
g=256
d=128
lr=0.0002
beta1=0.0
beta2=0.9

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

date
nvidia-smi

module load python/3.6
source ~/myenvs/vode/bin/activate

cp /scratch/voletivi/Datasets/CIFAR10 $SLURM_TMPDIR/

python main_cond.py \
        --out_dir $save \
        --data_dir $SLURM_TMPDIR/CIFAR10 \
        --n_epochs $epochs \
        --batch_size $bs \
        --disc_iters $di \
        --gen_size $g \
        --disc_size $d \
        --lr $lr \
        --beta1 $beta1 \
        --beta2 $beta2 \
        --norm $norm \
        --model 'resnet' \
        --loss 'hinge'

date
