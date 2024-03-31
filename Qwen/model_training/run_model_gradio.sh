# CUDA_VISIBLE_DEVICES=1 python ./tools/model_cli.py --model_path /data/ecnu/llama-7b-dyh-psy \
# --top_k 50 \
# --temperature 1.0 \
# --do-sample True \
# --max_new_tokens 512

# CUDA_VISIBLE_DEVICES=1 python ./tools/model_cli.py --model_path /data/ecnu/llama-7b-lzk \
# --top_k 50 \
# --temperature 1.0 \
# --do-sample True \
# --max_new_tokens 512

# CUDA_VISIBLE_DEVICES=0 python ./tools/model_cli.py --model_path /data/ecnu/EduChat/Open-Assistant/model/model_training/bellellama-7b-psy-chat/checkpoint-30 \
# --top_k 50 \
# --temperature 1.0 \
# --do-sample True \
# --max_new_tokens 512

CUDA_VISIBLE_DEVICES=0 python ./educhat_gradio.py --model_path /data/ecnu/llama_model_7b_p2 \
--top_k 50 \
--do_sample True \
--max_new_tokens 512