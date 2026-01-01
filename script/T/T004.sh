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
  --dataset cub_200_2011 --data_root ./data \
  --model tiny_vit_21m_512.dist_in22k_ft_in1k --pretrained \
  --epochs 280 \
  --batch_size 16 --num_workers 0 \
  --img_size 512 \
  --seed 42 >> T004.log 2>&1