#!/usr/bin/env bash
set -eux
set -o pipefail

source activate torch0.3.1py2.7

cd ../../

dataset="pascal_voc"
net="res152"
log_time="2019-10-16-23-47"
checksession="2"
checkepoch="9"
checkpoint="2504"

CUDA_VISIBLE_DEVICES=2 \
python test_net.py --dataset ${dataset} \
                        --net ${net} \
                        --log_time ${log_time} \
                        --checksession ${checksession} \
                        --checkepoch $checkepoch \
                        --checkpoint ${checkpoint} \
                        --cuda \
                  > eval_${dataset}_${net}_FPN_v0.${checksession}_${checkepoch}.log 2>&1 &
#                    --load_name
#                    --soft_nms
