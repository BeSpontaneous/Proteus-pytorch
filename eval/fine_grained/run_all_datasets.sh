#!/bin/bash

# datasets=("aircraft" "caltech101" "cars" "cifar10" "cifar100" "dtd" "flowers" "food" "pets" "sun397" "voc2007" "cub" "gtsrb" "country211" "mnist")
datasets=("aircraft" "caltech101" "cars" "cifar10" "cifar100" "dtd" "flowers" "food" "pets" "sun397" "voc2007" "cub")

echo "" > accuracy.log

for dataset in "${datasets[@]}"
do
    echo "Running code for dataset: $dataset"

    CUDA_VISIBLE_DEVICES=0 python main_downstream_linear_dinov2.py -b 32 \
        --model 'vit_large' \
        --pretrained path_pretrained_model \
        --dataset "$dataset" --max-iter 500 >> accuracy.log;

    if [ $? -eq 0 ]; then
        echo "Run for dataset $dataset finished successfully."
    else
        echo "Run for dataset $dataset failed."
    fi
done
