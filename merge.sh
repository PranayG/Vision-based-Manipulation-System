#!/bin/bash

# python merge.py --checkpoint_path log/checkpoint_detection.tar --debug
#python merge_grasping.py --checkpoint_path log/checkpoint_detection.tar --debug
num_iterations=2
i=0 
while [ $i -lt $num_iterations ]
do
    # i=$((i+1))
    /home/rstaion/miniconda3/envs/fr3_llm_env/bin/python home_pos.py
    # /home/rstaion/miniconda3/envs/anygrasp-env2/bin/python merge_grasping.py --checkpoint_path log/checkpoint_detection.tar --debug 
    # /home/rstaion/miniconda3/envs/anygrasp-env2/bin/python merge_grasping_singlecamera.py --checkpoint_path log/checkpoint_detection.tar --debug
    /home/rstaion/miniconda3/envs/anygrasp-env2/bin/python handcam.py --checkpoint_path log/checkpoint_detection.tar --debug --top_down_grasp
    echo "GRASP POSE ESTIMATED"
    echo "GRASPING INITIATED..."
    
    # /home/rstaion/miniconda3/envs/fr3_llm_env/bin/python move.py
    /home/rstaion/miniconda3/envs/fr3_llm_env/bin/python hand_move.py


    sleep 3
    



done
