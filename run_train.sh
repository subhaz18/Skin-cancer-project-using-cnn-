#!/usr/bin/env bash
source .venv/bin/activate
python train_ham10000.py --data_dir data --mode multiclass --img_size 224 --batch 32 --epochs 20
