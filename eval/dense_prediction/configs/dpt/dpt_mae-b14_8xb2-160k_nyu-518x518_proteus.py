_base_ = [
    '../_base_/models/dpt_mae_depth.py', '../_base_/datasets/nyu_518x518.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_25k.py'
]
crop_size = (518, 518)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/dinov2_vitb14_proteus.pth',
    backbone=dict(
        type='MAE',
        img_size=(518, 518),
        patch_size=14,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=0.1,
        output_cls_token=True),
    decode_head=dict(
        type='DPTHead_depth',
        in_channels=(768, 768, 768, 768),
        channels=256,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        num_classes=1,
        readout_type='project',
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
    ),
    test_cfg=dict(mode='slide_flip', crop_size=(518, 518), stride=(129, 129)))
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone

default_hooks = dict(
    checkpoint=dict(save_best='rmse', rule='less', max_keep_ckpts=1))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=25000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4, num_workers=2)
val_dataloader = dict(batch_size=4, num_workers=2)
test_dataloader = val_dataloader
