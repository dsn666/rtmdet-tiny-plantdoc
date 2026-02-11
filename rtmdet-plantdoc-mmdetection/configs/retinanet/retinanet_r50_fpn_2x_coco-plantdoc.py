_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=80,
    )
)

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
train_cfg = dict(max_epochs=48)

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
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
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'))

load_from = None
resume = False
