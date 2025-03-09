#!/bin/bash


python /data/coding/test_metric/eval_MII.py --json_name /data/coding/test_metric/code_space--output_qwen_nosample_nopad/Test_MII_pred_top0.95_inferv3.jsonl &
wait
echo "Finish MII Inference."