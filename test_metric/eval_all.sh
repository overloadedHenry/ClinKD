#!/bin/bash

JSON_FILE=''

python /data/coding/test_metric/eval_GC.py --json_name ${JSON_FILE}/Test_GC_pred_bfloat16_top0.3_inferv3.jsonl &
wait
echo "Finish GC Inference."

python /data/coding/test_metric/eval_VG.py --json_name ${JSON_FILE}/Test_VG_pred_bfloat16_top0.3_inferv3.jsonl &
wait
echo "Finish VG Inference."

python /data/coding/test_metric/eval_RO.py --json_name ${JSON_FILE}/Test_RO_pred_bfloat16_top0.3_inferv3.jsonl &
wait
echo "Finish RO Inference."

python /data/coding/test_metric/eval_MII.py --json_name ${JSON_FILE}/Test_MII_pred_bfloat16_top0.3_inferv3.jsonl &
wait
echo "Finish MII Inference."