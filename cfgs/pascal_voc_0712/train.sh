#!/usr/bin/env bash
set -eux
set -o pipefail

source activate torch0.3.1py2.7

cd ../../

CUDA_VISIBLE_DEVICES=2  \
nohup python trainval_net.py \
                        --exp_name res101_fpn_baseline \
                        --dataset pascal_voc_0712 \
                        --net res101 \
                        --bs 4 \
                        --nw 5 \
                        --lr 1e-3 \
                        --epochs 12 \
                        --save_dir ckpts \
                        --cuda \
                        --use_tfboard True \
             > pascal_voc_0712_res101_fpn_baseline_v0.1.log 2>&1 &