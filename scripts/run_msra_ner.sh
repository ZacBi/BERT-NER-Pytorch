export TASK_DATA_PATH=/home/ubuntu/workspace/github/Chatbot/data/msra_ner
export MODEL_PATH=/home/ubuntu/workspace/data/model-zoo/ernie_base_128_pytorch
export OUTPUT_DIR=/home/ubuntu/workspace/github/Chatbot/outputs
export WORKSPACE=/home/ubuntu/workspace/github/Chatbot/model

python3 ${WORKSPACE}/ner_train.py \
    --train_file ${TASK_DATA_PATH}/train.tsv \
    --predict_file ${TASK_DATA_PATH}/dev.tsv \
    --model_type bert \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}/experiments/exp4 \
    --log_dir ${OUTPUT_DIR}/experiments/exp4/runs \
    --task_name msra\
    --num_labels 7 \
    --max_seq_len 256 \
    --do_train \
    --evaluate_during_training \
    --do_lower_case \
    --per_gpu_train_batch_size 16 \
    --learning_rate 5e-5 \
    --layer_norm_eps 1e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 6 \
    --eval_steps 100 \
    --save_steps 1000 \
    --seed 1