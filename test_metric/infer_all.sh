#!/bin/bash
 
CHECKPOINT_FILE='/data/coding/code_space/output_qwen'

CUDA_VISIBLE_DEVICES=0 python /data/coding/test_metric/infer.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /data/coding/BiRD_data/Test_VG.json \
    --top_p 03 \
    --max_new_tokens 80 &

wait
echo "Finish VG Inference."

CUDA_VISIBLE_DEVICES=0 python /data/coding/test_metric/infer.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /data/coding/BiRD_data/Test_RO.json \
    --top_p 03 \
    --max_new_tokens 80 &

wait
echo "Finish RO Inference."

CUDA_VISIBLE_DEVICES=0 python /data/coding/test_metric/infer.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /data/coding/BiRD_data/Test_GC.json \
    --top_p 03 \
    --max_new_tokens 80 &

wait
echo "Finish GC Inference."

CUDA_VISIBLE_DEVICES=0 python /data/coding/test_metric/infer.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /data/coding/BiRD_data/Test_MII.json \
    --top_p 03 \
    --max_new_tokens 100 &

wait
echo "Finish MII Inference."