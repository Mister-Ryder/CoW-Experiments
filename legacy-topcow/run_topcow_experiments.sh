#!/bin/bash
# run_topcow_v4.sh — 多分类版本实验
# bash run_topcow_v4.sh [GPU] [TRAIN_DIR] [TEST_DIR]

GPU=${1:-0}
TRAIN=${2:-./TopCoW_processed}
TEST=${3:-./crown_processed}
EP=200; BS=2; OUT=./runs_topcow_v4

echo "=== TopCoW TGF v4 (multiclass) | GPU=$GPU ==="

for EXP in \
    "tgf_clean       tgf        0.0" \
    "tgf_n30         tgf        0.3" \
    "tgf_n50         tgf        0.5" \
    "nogate_clean    tgf_nogate 0.0" \
    "nogate_n30      tgf_nogate 0.3" \
    "nogate_n50      tgf_nogate 0.5" \
    "unet_clean      unet       0.0"
do
    set -- $EXP; NAME=$1; MODEL=$2; NOISE=$3
    echo -e "\n>>> $NAME ($MODEL, noise=$NOISE)"
    CUDA_VISIBLE_DEVICES=$GPU python topcow_train.py \
        --exp_name $NAME --model_type $MODEL --noise_rate $NOISE \
        --train_dir $TRAIN --test_dir $TEST \
        --output_dir $OUT --epochs $EP --batch_size $BS
done

echo -e "\n=== Summary ==="
python topcow_train.py --summarize --output_dir $OUT --exp_name _