#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 torchrun \
 --master_port=29500 \
 --nproc_per_node=8 train_deepcad.py \
 --config_path config/trainer.yaml
