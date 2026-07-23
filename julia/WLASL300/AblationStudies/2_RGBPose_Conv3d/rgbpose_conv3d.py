N_HAND_LANDMARKS = 21 #x2 for both hands

backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=6,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1)),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        in_channels= 2 * N_HAND_LANDMARKS, 
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1)))
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=300, 
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1., 1.])
test_cfg = dict(average_clips='prob')
model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    test_cfg=test_cfg)

dataset_type = 'PoseDataset'
data_root = '../WLASL300/WLASL_300_compressed'
ann_file = 'julia/WLASL300/pyskl_mediapipe_annos_2d_denormalized_NOFACE_NOBODY.pkl' #TODO 
label_mappings='julia/WLASL300/label_mapping.txt'

left_kp = list(range(0, 21))
right_kp = list(range(21, 42)) 

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=48), num_clips=1),
    dict(type='DecordInit', label_mapping_file=label_mappings),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False), 
    dict(type='RandomResizedCrop', area_range=(0.8, 1.0)),
    dict(type='Resize', scale=(448, 448), keep_ratio=False), 
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25), #scale by 0.25 as RGB channel is 4x bigger than Pose channel
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=48), num_clips=1),
    dict(type='DecordInit', label_mapping_file=label_mappings),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(448, 448), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=48), num_clips=10),
    dict(type='DecordInit', label_mapping_file=label_mappings),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(448, 448), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]

data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, split='train', data_prefix=data_root, pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', data_prefix=data_root, pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', data_prefix=data_root, pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 120
#early stopping
early_stopping = dict(
    monitor='loss',
    phase='val',
    patience=10,
    min_delta=0.00001,
    max_epochs=240,
    mode='min')
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5), key_indicator='RGBPose_1:1_top1_acc')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/julia/RGBPose_Conv3d/rgbpose_conv3d_3' #TODO
load_from = 'work_dirs/julia/RGBPose_Conv3d/rgbpose_conv3d_3/rgbpose_conv3d_init_3.pth' #TODO
