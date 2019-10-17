#!/usr/bin/env bash
set -eux
set -o pipefail

source activate torch0.3.1py2.7

cd ../../

# train res101
#CUDA_VISIBLE_DEVICES=0  \
#nohup python trainval_net.py \
#                        --dataset pascal_voc \
#                        --net res101 \
#                        --bs 1 \
#                        --nw 4 \
#                        --lr 1e-3 \
#                        --epochs 7 \
#                        --lr_decay_step 5 \
#                        --cuda \
#                        --s 1 \
#                        --use_tfboard True \
#             > pascal_voc_res101_fpn_v0.1.log 2>&1 &


# train res50
#CUDA_VISIBLE_DEVICES=1  \
#nohup python trainval_net.py \
#                        --dataset pascal_voc \
#                        --net res50 \
#                        --bs 1 \
#                        --nw 4 \
#                        --lr 1e-3 \
#                        --epochs 7 \
#                        --lr_decay_step 5 \
#                        --cuda \
#                        --s 1 \
#                        --use_tfboard True \
#             > pascal_voc_res50_fpn_v0.1.log 2>&1 &


# train res152
CUDA_VISIBLE_DEVICES=2  \
nohup python trainval_net.py \
                        --dataset pascal_voc \
                        --net res152 \
                        --bs 1 \
                        --nw 2 \
                        --lr 1e-3 \
                        --epochs 7 \
                        --lr_decay_step 5 \
                        --cuda \
                        --s 1 \
                        --use_tfboard True \
             > pascal_voc_res152_fpn_v0.1.log 2>&1 &