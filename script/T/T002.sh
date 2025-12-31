#!/bin/bash
#PBS -P rp06
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=38:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/RL-test/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/RL-test/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 src/main.py \
  --dataset cub_200_2011 --data_root ./data \
  --val_split train_split --train_val_ratio 0.1 \
  --model vit_base_patch16_224 --pretrained \
  --epochs 3 --warmup_epochs 0 --train_last_n 0 \
  --lr 1e-3 --lambda_l2 1e-2 \
  --batch_size 32 --num_workers 0 \
  --augment \
  --output_dir ./T002 --run_name T002 \
  --seed 42
  >> T002.log 2>&1