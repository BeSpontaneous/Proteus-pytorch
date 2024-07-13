


python -m torch.distributed.launch --nproc_per_node=4 --use_env dinov2/eval/linear_proteus.py \
    --config-file dinov2/configs/eval/vitl14_pretrain.yaml \
    --pretrained-weights path_pretrained_model \
    --train-dataset ImageNet:split=TRAIN:root=imagenet_path:extra=imagenet_path \
    --val-dataset ImageNet:split=VAL:root=imagenet_path:extra=imagenet_path \
    --batch-size 128 --epoch-length 2502 \
    --epochs 10 \
    --output-dir log/linear_probing;