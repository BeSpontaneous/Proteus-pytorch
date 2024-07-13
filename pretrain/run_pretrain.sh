

#### access DINOv2

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 128 --warmup-epochs 5 --epochs 300 \
    --data-set IMNET --data-path imagenet_path \
    --teacher-model vit_large --target_model vit_base --model models_proteus_dinov2 \
    --patch_size 14 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 1.0 --lambda_patch 1.0 \
    --output_dir log/DINOv2_training;



#### access SynCLR

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 128 --warmup-epochs 5 --epochs 300 \
    --data-set IMNET --data-path imagenet_path \
    --teacher-model vit_large --target_model vit_base --model models_proteus_synclr \
    --teacher-path pretrained_synclr_path \
    --patch_size 14 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 1.0 --lambda_patch 1.0 \
    --output_dir log/SynCLR_training;



#### access CLIP

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 128 --warmup-epochs 5 --epochs 300 \
    --data-set IMNET --data-path imagenet_path \
    --teacher-model vit_large --target_model vit_base --model models_proteus_clip \
    --teacher-path pretrained_clip_path \
    --patch_size 14 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 0.0 --lambda_patch 0.0 \
    --output_dir log/CLIP_training;