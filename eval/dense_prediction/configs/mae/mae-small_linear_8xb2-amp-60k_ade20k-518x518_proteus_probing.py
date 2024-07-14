_base_ = [
    '../_base_/models/linear_mae_probing.py', '../_base_/datasets/ade20k_518.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_60k.py'
]
crop_size = (518, 518)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/dinov2_vits14_proteus.pth',
    backbone=dict(
        type='MAE_fix',
        img_size=(518, 518),
        patch_size=14,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],
        frozen_stages=12),
    decode_head=dict(
        type='FCNHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        channels=1536,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=(518, 518), stride=(345, 345)))


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=8e-4, betas=(0.9, 0.999), weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=60000,
        by_epoch=False,
    )
]


# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

find_unused_parameters = True