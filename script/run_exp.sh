#!/bin/bash

# Example script to run CARROT experiment

# 1. Run with default settings
echo "Running with default settings..."
python src/main.py

# 2. Run with command line arguments
echo "Running with custom arguments..."
python src/main.py --dataset cotton80 --batch_size 16 --sigma_s 0.8

# 3. Run with config file
echo "Running with config file..."
python src/main.py --config configs/default.yaml

# 4. Run with config file AND override with CLI
echo "Running with config file + CLI override..."
python src/main.py --config configs/default.yaml --diffusion_t 2.0
