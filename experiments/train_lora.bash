lora_rank=32
lora_trainable="gate_proj,k_proj,o_proj,down_proj,v_proj,q_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
model_name_or_path="/cpfs01/user/chenqin.p/model_cached/Llama-2-7b-chat-hf"   
your_data_path="data"  
your_checkpopint_path="saved/moelora/2experts_q"  
MAX_SOURCE_LENGTH=512
peft_path=""  

export WANDB_DISABLED="true"

# Training Command
deepspeed --num_gpus=4 --master_port $MASTER_PORT run_mlora.py \
    --deepspeed src/ds.config \
    --data_config bigfive_task_q\
    --do_train \
    --use_original_llama \
    --use_peft \
    --train_file $your_data_path/train.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs=5 \
    --save_strategy="epoch" \
    --logging_steps 100 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16 \
    --lora_name default \
    --expert_num 2 \
    --task_num 32 \
    --task_embedding_dim 64

# deepspeed --num_gpus=4 --master_port $MASTER_PORT run_mlora.py \
#     --deepspeed src/ds.config \
#     --data_config bigfive_task\
#     --do_train \
#     --train_file $your_data_path/train.json \
#     --cache_dir $your_data_path \
#     --prompt_column input \
#     --response_column target \
#     --overwrite_cache \
#     --model_name_or_path $model_name_or_path \
#     --output_dir $your_checkpopint_path \
#     --overwrite_output_dir \
#     --max_source_length $MAX_SOURCE_LENGTH \
#     --max_target_length 196 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --max_steps ${MAX_STEPS} \
#     --logging_steps 100 \
#     --save_steps ${SAVE_STEPS} \
#     --learning_rate $LR \
#     --lora_rank ${lora_rank} \
#     --trainable ${lora_trainable} \
#     --modules_to_save ${modules_to_save} \
#     --lora_dropout ${lora_dropout} \
#     --fp16 \
#     --lora_name moelora \
#     --expert_num 10
#     --task_num 32 \
#     --task_embedding_dim 64
