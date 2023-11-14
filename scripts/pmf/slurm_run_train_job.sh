#!/bin/sh
#
#SBATCH --job-name="5S-10Q-pmf"
#SBATCH --account=education-3me-msc-ro
#SBATCH --partition=gpu
#SBATCH --time=11:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G

module load miniconda3/4.12.0
conda activate pmf
export shots=5
export queries=10

srun python --version
srun python main.py --output outputs/adapted_5way-${shots}shot-${queries}query_outer --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport ${shots} --nQuery ${queries} --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10