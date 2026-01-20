
N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21 #x2 for both hands

model = dict(
    type='Recognizer3D', #
    backbone=dict(
        type='ResNet3dSlowOnly', #
        in_channels=N_FACE_LANDMARKS + N_BODY_LANDMARKS + 2 * N_HAND_LANDMARKS, #number of keypoints, 543
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=300, #300 for wlasl300
        dropout=0.5),
    test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset' #
ann_file = 'julia/WLASL300/pyskl_mediapipe_annos.pkl' #TODO
#if flipping is used, these need to be adapted based on mediapipe
#left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
#right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='KeypointTo2D'), #remove 3d-coordinate of keypoints
    dict(type='DeNormalizeKeypoints'),  #denormalize mediapipe keypoints from [0, 1] to original image shape
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    #dict(type='Resize', scale=(-1, 64)),
    #dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    #dict(type='Resize', scale=(56, 56), keep_ratio=False),
    #remove flipping
    #dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False), #with_limb kontrolliert ob heatmap nur aus punkten (keypoints) oder auch verbindungen zwischen den kps generiert
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='KeypointTo2D'), #remove 3d-coordinate of keypoints
    dict(type='DeNormalizeKeypoints'),  #denormalize mediapipe keypoints from [0, 1] to original image shape
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    #dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='KeypointTo2D'), #remove 3d-coordinate of keypoints
    dict(type='DeNormalizeKeypoints'),  #denormalize mediapipe keypoints from [0, 1] to original image shape
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    #dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=False), #add: double=True, left_kp=left_kp, right_kp=right_kp to double the test data by flipping it horizontally
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8, #decrease videos per gpu to prevent cuda out of memory
    workers_per_gpu=2, #decrease worker
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)), #train split
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline), #val split
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline)) #test split
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 24 #epochs
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/julia/mediapipe_wlasl300' #TODO
