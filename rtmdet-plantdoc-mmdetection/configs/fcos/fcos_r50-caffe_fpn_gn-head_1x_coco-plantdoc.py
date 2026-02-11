_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=29,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/featurize/data/PlantDoc/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

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
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
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
    classwise=False,
    format_only=False,
    backend_args=backend_args)
# iou_thrs=0.5
test_evaluator = val_evaluator

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'))

load_from = None
resume = False

