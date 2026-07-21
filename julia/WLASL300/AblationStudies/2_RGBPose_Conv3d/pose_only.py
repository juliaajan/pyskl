N_HAND_LANDMARKS = 21 #x2 for both hands

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels= 2 * N_HAND_LANDMARKS, #number of keypoints
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1), #no spatial downsampling in stem layer
        pool1_stride=(1, 1), #no temporal downsampling in stem layer
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2), #stride 2 to reduce spatial dimensions in each resnet block
        temporal_strides=(1, 1, 1)), #temporal stride 1, framerate stays the same in each resenet block
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=300, #300 for wlasl300
        dropout=0.5),
    test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset'
ann_file = 'julia/WLASL300/pyskl_mediapipe_annos_2d_denormalized_NO_KPS_FROM_BODYMODEL.pkl' #TODO 

#left and right hand keypoints each with 21 kps starting at index 4, left hand is extracted first
left_kp = list(range(0, 21))
right_kp = list(range(21, 42)) 

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomResizedCrop', area_range=(0.8, 1.0)),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True,  left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
         dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)), #train split
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline), #val split
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline)) #test split
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0003) 
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
work_dir = './work_dirs/julia/RGBPose_Conv3d/pose_only_hands_only_lr_0_01_fr48_uncompressedAnnos_Resize08_otherConfig' #TODO

