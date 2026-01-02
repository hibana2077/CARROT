#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=10GB
#PBS -l walltime=12:45:00
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
  --batch_size 256 --num_workers 0 \
  --img_size 256 \
  --use_carrot \
  --carrot_lambda 0.1 --carrot_alpha 10 --carrot_topm 20 --carrot_conf_topk 20 --carrot_warmup_epochs 10 \
  --seed 42 >> C002.log 2>&1


# lambda ∈ {0.02, 0.05, 0.1, 0.2, 0.4}