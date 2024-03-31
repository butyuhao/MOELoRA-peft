#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`
export OMP_NUM_THREADS=64

export MASTER_ADDR=localhost
export MASTER_PORT=9901
export WORLD_SIZE=8
export GPUS_PER_NODE=8

# alias cuda118='export CUDA_HOME=/mnt/petrelfs/share_data/llm_env/dep/cuda-11.8'
# alias cuda117='export CUDA_HOME=/mnt/petrelfs/share/new-cuda-11.7'

# cuda118
 
export WANDB_ENTITY=butyuhao
export WANDB_PROJECT=EduChat_DIRECT_HQ


MODEL="/cpfs01/user/chenqin.p/yi-34b/Yi-34B-ds" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.


torchrun \
    --nproc_per_node $GPUS_PER_NODE --nnodes 1 --node_rank 0 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT finetune.py \
    --model_name_or_path $MODEL \
    --bf16 True \
    --output_dir "../educhat_yi-34b-hq-3-7" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 1600 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune/ds_config_zero3-zk.json \
    --data_config "educhat-hq-2-24" \
    --dataloader_num_workers 8
