#!/bin/bash
# Activate jpcgpu environment and run the training pipeline
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate jpcgpu

# python _00_aggregate_test.py --dataset CIFAR10
# For SPEECHCOMMANDS, automatic scaling handles max_t1 and max_steps
python _00_aggregate_test.py --dataset SPEECHCOMMANDS