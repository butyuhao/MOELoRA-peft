lora_rank=64
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

# # Training Command
# deepspeed --num_gpus=4 --master_port $MASTER_PORT run_mlora.py \
#     --deepspeed src/ds.config \
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
#     --expert_num 8

list1=(1 2 3 4 5)
list2=(325 651 977 1303 1625)

# 使用zip命令将两个列表合并，并循环遍历它们
for ((i=0; i<${#list1[@]}; i++)); do
    item1=${list1[$i]}
    item2=${list2[$i]}

    deepspeed --num_gpus=1 --master_port $MASTER_PORT run_mlora.py \
    --do_predict \
    --data_config bigfive_task_test \
    --peft_path "/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/saved/moelora/4experts_dimensional_q_4_21_balancing_CE/checkpoint-${item2}" \
    --prediction_output_path "/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/eval/llama/4expert/epoch${item1}_4_21_balancing" \
    --test_file /cpfs01/user/chenqin.p/dyh/MOELoRA-peft/data/bigfive_questionnaire.xlsx \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir results/pred/moelora \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 1024 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --lora_name moelora \
    --expert_num 4 \
    --task_num 10 \
    --task_embedding_dim 64

done


