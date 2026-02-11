# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/featurize/data/PlantDoc/'

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

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
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

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
