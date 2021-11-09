#!/bin/bash


export OMP_NUM_THREADS=1
export gpu_num=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"

exp_name="(oiv6)BGNN-3-3-learnable_scaling-inst_drop0.9"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relBGNN_oiv6.yaml" \
       EXPERIMENT_NAME "$exp_name" \
        SOLVER.IMS_PER_BATCH $[3*$gpu_num] \
        TEST.IMS_PER_BATCH $[$gpu_num] \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 


