#!/usr/bin/env bash
set -eux
set -o pipefail

source activate torch0.3.1py2.7

cd ../../

dataset="pascal_voc"
net="res50"
log_time="2019-10-16-23-43"
checksession="1"
checkepoch="6"
checkpoint="10021"

CUDA_VISIBLE_DEVICES=0 \
python test_net.py --dataset ${dataset} \
                        --net ${net} \
                        --log_time ${log_time} \
                        --checksession ${checksession} \
                        --checkepoch $checkepoch \
                        --checkpoint ${checkpoint} \
                        --cuda \
                        --soft_nms \
                  > eval_${dataset}_${net}_FPN_v0.${checksession}_${checkepoch}_soft_nms.log 2>&1 &
#                    --load_name
#                    --soft_nms
