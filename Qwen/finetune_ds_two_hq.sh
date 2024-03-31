#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export WORLD_SIZE=4
export GPUS_PER_NODE=4

# alias cuda118='export CUDA_HOME=/mnt/petrelfs/share_data/llm_env/dep/cuda-11.8'
# alias cuda117='export CUDA_HOME=/mnt/petrelfs/share/new-cuda-11.7'

# cuda118


MODEL="../educhat_qwen_base/checkpoint-18480" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.

echo SLURM_PROCID
echo $SLURM_PROCID

torchrun \
    --nproc_per_node $GPUS_PER_NODE --nnodes 1 --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT finetune.py \
    --model_name_or_path $MODEL \
    --bf16 True \
    --output_dir "../educhat_qwen_hq" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
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
    --data_config "educhat-hq-1-2" \
    --dataloader_num_workers 16 
