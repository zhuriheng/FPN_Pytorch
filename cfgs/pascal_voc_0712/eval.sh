#!/usr/bin/env bash
set -eux
set -o pipefail

source activate torch0.3.1py2.7

cd ../../

exp_name="res101_fpn_baseline"
dataset="pascal_voc_0712"
net="res101"
log_time="2019-09-09-21-41"
checkepoch="11"

CUDA_VISIBLE_DEVICES=2 \
python test_net.py --exp_name ${exp_name} \
                        --dataset $dataset \
                        --net $net \
                        --log_time $log_time \
                        --checksession 1 \
                        --checkepoch $checkepoch \
                        --checkpoint 8274 \
                        --cuda \
                        --soft_nms \
                  > eval_${exp_name}_${dataset}_${net}_${log_time}_${checkepoch}_soft_nms.log 2>&1 &
#                    --load_name
#                    --soft_nms
