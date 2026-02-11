_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='CenterNet',
    # use caffe img_norm
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
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
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        # There is a chance to get 40.3 after switching init_cfg,
        # otherwise it is about 39.9~40.1
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=29,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        hm_min_radius=4,
        hm_min_overlap=0.8,
        more_pos_thresh=0.2,
        more_pos_topk=9,
        soft_weight_on_reg=False,
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/featurize/data/PlantDoc/'

backend_args = None

# single-scale training is about 39.3
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
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

# training schedule for 4x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00025,
        by_epoch=False,
        begin=0,
        end=4000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    # Experiments show that there is no need to turn on clip_grad.
    paramwise_cfg=dict(norm_decay_mult=0.))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'))
