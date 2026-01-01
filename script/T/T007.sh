#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=02:45:00
#PBS -l wd
#PBS -l storage=scratch/yp87

module load cuda/12.6.2

source /scratch/yp87/sl5952/CARROT/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -u src/main.py \
  --dataset cotton80 --data_root ./data \
  --model vit_base_patch16_dinov3 --pretrained \
  --epochs 280 \
  --lr 1e-3 \
  --batch_size 1024 --num_workers 0 \
  --img_size 256 \
  --seed 42 >> T007.log 2>&1