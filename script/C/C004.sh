#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=8GB
#PBS -l walltime=28:00:00
#PBS -l wd
#PBS -l storage=scratch/yp87

module load cuda/12.6.2

source /scratch/yp87/sl5952/CARROT/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -u src/main.py \
  --dataset cub_200_2011 --data_root ./data \
  --model vit_base_patch8_224.augreg2_in21k_ft_in1k --pretrained \
  --epochs 1000 \
  --batch_size 256 --num_workers 0 \
  --img_size 224 \
  --optimizer adamw \
  --lr 0.0005 \
  --carrot --carrot_k 4 \
  --seed 42 >> C004.log 2>&1