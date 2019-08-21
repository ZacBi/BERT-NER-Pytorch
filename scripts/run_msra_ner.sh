export TASK_DATA_PATH=/
export MODEL_PATH=/
export OUTPUT_DIR=/
export WORKSPACE=/

python3 ${WORKSPACE}/ner_train.py \
    --train_file ${TASK_DATA_PATH}/train.tsv \
    --predict_file ${TASK_DATA_PATH}/test.tsv \
    --model_type bert \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}/experiments/exp3 \
    --log_dir ${WORKSPACE}/runs \
    --task_name msra\
    --num_labels 7 \
    --max_seq_len 256 \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --per_gpu_train_batch_size 16 \
    --learning_rate 5e-5 \
    --layer_norm_eps 1e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 6 \
    --eval_steps 50 \
    --save_steps 1000 \
    --seed 1