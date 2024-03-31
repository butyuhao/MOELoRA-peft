#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


export MASTER_ADDR=localhost
export MASTER_PORT=9901
export WORLD_SIZE=8
export GPUS_PER_NODE=8

# alias cuda118='export CUDA_HOME=/mnt/petrelfs/share_data/llm_env/dep/cuda-11.8'
# alias cuda117='export CUDA_HOME=/mnt/petrelfs/share/new-cuda-11.7'

# cuda118
 
export WANDB_ENTITY=butyuhao
export WANDB_PROJECT=EduChat_DIRECT_HQ


MODEL="/cpfs01/user/chenqin.p/model_cached/Qwen1.5-14B-Chat" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.


torchrun \
    --nproc_per_node $GPUS_PER_NODE --nnodes 1 --node_rank 0 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT finetune.py \
    --model_name_or_path $MODEL \
    --bf16 True \
    --output_dir "../educhat_qwen1.5_direct_hq-3-10-14B" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune/ds_config_zero3.json \
    --data_config "educhat-hq-2-24" \
    --dataloader_num_workers 8
