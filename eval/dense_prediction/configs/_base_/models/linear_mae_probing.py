norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
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
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
