lora_rank=160
lora_trainable="gate_proj,k_proj,o_proj,down_proj,v_proj,q_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
MAX_STEPS=4000
SAVE_STEPS=500
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
model_name_or_path="/cpfs01/user/chenqin.p/model_cached/Llama-2-7b-chat-hf"   
your_data_path="data"  
your_checkpopint_path="saved/moelora"  
MAX_SOURCE_LENGTH=512


deepspeed --num_gpus=1 --master_port $MASTER_PORT run_mlora.py \
    --do_predict \
    --data_config bigfive_task_test \
    --test_file /cpfs01/user/chenqin.p/dyh/MOELoRA-peft/data/bigfive_questionnaire.xlsx \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir results/pred/moelora \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 256 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate \
    --lora_name moelora \
    --expert_num 10 \
    --task_num 32 \
    --task_embedding_dim 64 \
    --use_no_peft