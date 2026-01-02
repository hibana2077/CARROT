#!/bin/bash
#PBS -P yp87
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=14GB
#PBS -l walltime=05:45:00
#PBS -l wd
#PBS -l storage=scratch/yp87

module load cuda/12.6.2

source /scratch/yp87/sl5952/CARROT/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -u src/main.py \
  --dataset cub_200_2011 --data_root ./data \
  --model swinv2_base_window16_256.ms_in1k --pretrained \
  --epochs 280 \
  --batch_size 16 --num_workers 0 \
  --img_size 256 \
  --seed 42 >> BS010.log 2>&1