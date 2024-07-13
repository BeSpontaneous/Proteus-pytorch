norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0)
model = dict(
    type='DepthEstimator',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/vit-b16_p16_224-80ecf9dd.pth', # noqa
    backbone=dict(
        type='MAE',
        img_size=(518, 518),
        patch_size=14,
        in_channels=3,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
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
        in_channels=(384, 384, 384, 384),
        channels=256,
        embed_dims=384,
        post_process_channels=[48, 96, 192, 384],
        num_classes=1,
        readout_type='project',
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
    ),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable