#!/bin/bash

CHECKPOINT_FILE=''



export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

python /data/coding/test_metric/infer_by_chat.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /data/coding/BiRD_data/Test_GC.json \
    --top_p 0.3 \
    --max_new_tokens 80


wait
echo "Finish GC Inference."