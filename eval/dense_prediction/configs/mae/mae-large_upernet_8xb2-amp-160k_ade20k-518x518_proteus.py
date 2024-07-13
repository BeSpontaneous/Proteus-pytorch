_base_ = [
    '../_base_/models/upernet_mae.py', '../_base_/datasets/ade20k_518.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (518, 518)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/dinov2_vitl14_ibot_mse_aligned_old_aug_ema_wo_mixup.pth',
    backbone=dict(
        type='MAE',
        img_size=(518, 518),
        patch_size=14,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23]),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=150, channels=1024),
    auxiliary_head=dict(in_channels=1024, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=(518, 518), stride=(345, 345)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.65),
    constructor='LayerDecayOptimizerConstructor')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
