export SQUAD_DIR=../data/sed
# export SQUAD_DIR=../data/squad_v2
export MODEL_NAME=../../output_dir/checkpoint-31902
export OUTPUT_MODEL_DIR=./models
export TOK_NAME="../../tok"

# CUDA_VISIBLE_DEVICES="0,1" 
CUDA_VISIBLE_DEVICES=0
python3 ../transformers-3.3.1/examples/question-answering/run_squad.py \
  --model_type roberta \
  --model_name_or_path $MODEL_NAME \
  --config_name $MODEL_NAME \
  --tokenizer_name $TOK_NAME \
  --do_train \
  --do_eval \
  --version_2_with_negative \
  --data_dir $SQUAD_DIR \
  --train_file train.json \
  --predict_file valid.json \
  --per_gpu_train_batch_size 4\
  --learning_rate 3e-5 \
  --do_lower_case \
  --num_train_epochs 40 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir $OUTPUT_MODEL_DIR \
  --overwrite_output_dir \