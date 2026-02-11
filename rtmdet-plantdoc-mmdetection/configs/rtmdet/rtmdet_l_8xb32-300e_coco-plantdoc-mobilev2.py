_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py', './rtmdet_tta.py'
]
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[32, 96, 320],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=29,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'CocoDataset'
data_root = '/home/featurize/data/PlantDoc/'
backend_args = None
metainfo = {
    'classes': ('Bell_pepper leaf spot', 'Potato leaf early blight', 'Strawberry leaf', 'grape leaf', 'grape leaf black rot', 'Tomato leaf', 'Bell_pepper leaf', 'Potato leaf', 'Peach leaf', 'Corn leaf blight', 'Apple Scab Leaf', 'Cherry leaf', 'Tomato leaf bacterial spot', 'Tomato leaf yellow virus', 'Corn Gray leaf spot', 'Apple rust leaf', 'Raspberry leaf', 'Blueberry leaf', 'Squash Powdery mildew leaf', 'Tomato mold leaf', 'Tomato Early blight leaf', 'Tomato leaf late blight', 'Tomato Septoria leaf spot', 'Tomato leaf mosaic virus', 'Potato leaf late blight', 'Apple leaf', 'Corn rust leaf', 'Soyabean leaf', 'Tomato two spotted spider mites leaf' ),
    'palette': [
        (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
        (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
        (153, 69, 1), (120, 166, 157), (0, 182, 199),
        (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
        (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
        (183, 130, 88), (155,10,187), (10,145,190), (120,120,10),
        (50,200,200), (200,50,200), (200,200,50), (0,100,200),
        (100,0,200), (200,100,0)
    ]
}

train_dataloader = dict(
    # batch_size=8,
    # num_workers=2,
    batch_size=16,
    num_workers=2,
    batch_sampler=None,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VOC2007/ImageSets/voc07_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    # batch_size=5, num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VOC2007/ImageSets/voc07_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'VOC2007/ImageSets/voc07_test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    proposal_nums=(100, 1, 10))
test_evaluator = val_evaluator

# train_dataloader = dict(
#     batch_size=32,
#     num_workers=10,
#     batch_sampler=None,
#     pin_memory=True,
#     dataset=dict(pipeline=train_pipeline))
# val_dataloader = dict(
#     batch_size=5, num_workers=10, dataset=dict(pipeline=test_pipeline))
# test_dataloader = val_dataloader

max_epochs = 300
stage2_num_epochs = 20
base_lr = 0.004
interval = 10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

# val_evaluator = dict(proposal_nums=(100, 1, 10))
# test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        # max_keep_ckpts=3,  # only keep latest 3 checkpoints
        save_best='auto'
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]
