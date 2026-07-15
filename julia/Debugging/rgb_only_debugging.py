model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=2048,
        num_classes=300, #300 for wlasl300
        dropout=0.5),
    test_cfg = dict(average_clips='prob'))

dataset_type = 'PoseDataset'
data_root = '../WLASL300/WLASL_300_compressed'
#for the RGB stream, the same ann file as for the Pose stream can be used
ann_file = 'julia/WLASL300/pyskl_mediapipe_annos_2d_denormalized_NOSE_FACE_HANDS_compressed.pkl'
label_mappings='julia/WLASL300/label_mapping.txt'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='DecordInit', label_mapping_file='julia/WLASL300/label_mapping.txt'),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False), #first resizing bigger than second
    #dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    #dict(type='Resize', scale=(448, 448), keep_ratio=False),
    #dict(type='Flip', flip_ratio=0.5), #use flipping 
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='DecordInit', label_mapping_file='julia/WLASL300/label_mapping.txt'),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(448, 448), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=10),
    dict(type='DecordInit', label_mapping_file='julia/WLASL300/label_mapping.txt'),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(448, 448), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4, #batch size, decrease videos per gpu to prevent cuda out of memory, Attention: adapt lr accordingly
    workers_per_gpu=2, #decrease worker
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, split='train', ann_file=ann_file, data_prefix=data_root, pipeline=train_pipeline)),
    val=dict(type=dataset_type, split='val', ann_file=ann_file, data_prefix=data_root, pipeline=val_pipeline),
    test=dict(type=dataset_type, split='test', ann_file=ann_file, data_prefix=data_root, pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)  # adapted lr
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 240
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5), save_best='auto')
#early stopping
early_stopping = dict(
    monitor='loss',
    phase='val',
    patience=15,
    min_delta=0.00001,
    max_epochs=240,
    mode='min')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/julia/RGBPose_Conv3d/rgb_only_hands_only_lr_0_01' #TODO


