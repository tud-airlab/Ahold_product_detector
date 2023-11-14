#!/bin/sh
#
#SBATCH --job-name="pmf_test_all_jobs"
#SBATCH --account=education-3me-msc-ro
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G

module load miniconda3/4.12.0
conda activate pmf

srun python --version
export folder=5way-5shot-7-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-5shot-4-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-5shot-2-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-5shot-1-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-3shot-10-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-3shot-7-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-3shot-4-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-3shot-2-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-3shot-1-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-1shot-10-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-1shot-7-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-1shot-4-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-1shot-2-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval
export folder=5way-1shot-1-query
srun python main.py --output outputs/test_${folder} --resume outputs/${folder}/best.pth --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda:0 --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10 --eval