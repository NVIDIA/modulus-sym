#!/bin/bash
# move the trained checkpoints to correct folder
mkdir 'outputs/run_mode=eval'
cp -r outputs/three_fin_flow/ 'outputs/run_mode=eval/'
cp -r outputs/three_fin_thermal/ 'outputs/run_mode=eval/'

# run the inferencing (uncomment the below line if limited GPU memory)
# export CUDA_VISIBLE_DEVICES=""
python three_fin_flow.py run_mode=eval
python three_fin_thermal.py run_mode=eval

# run the design optimization
python three_fin_design.py
