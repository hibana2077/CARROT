#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=08:00:00
#PBS -l wd
#PBS -l storage=scratch/yp87

module load cuda/12.6.2

source /scratch/yp87/sl5952/CARROT/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -u src/main.py \
  --dataset cotton80 --data_root ./data \
  --model xcit_tiny_12_p8_384.fb_dist_in1k --pretrained \
  --epochs 1000 \
  --batch_size 32 --num_workers 0 \
  --img_size 384 \
  --use_carrot \
  --carrot_lambda 0.02 --carrot_alpha 20 --carrot_topm 40 --carrot_conf_topk 20 --carrot_warmup_epochs 10 \
  --seed 42 >> BTC006.log 2>&1