


#change: FACE_LANDMARK_INDICES, in_channels, flip keypoints, linked annotation file, work_dir name, in_channels,  flip keypoints

#order of extracted keypoints: nose (1), face, left hand (21), right hand (21)
FACE_LANDMARK_INDICES = [46, 52, 53, 65, 295, 283, 282, 276, 7, 159, 155, 145, 382, 386, 249, 374, 324, 13, 78, 14] #TODO
N_NOSE_LANDMARKS = 1
N_HAND_LANDMARKS = 21 #x2 for both hands

model = dict(
    type='Recognizer3D', 
    backbone=dict(
        type='ResNet3dSlowOnly', 
        in_channels= N_NOSE_LANDMARKS + len(FACE_LANDMARK_INDICES) + 2 * N_HAND_LANDMARKS, #number of keypoints
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1), #no downsampling in first 
        pool1_stride=(1, 1), #no downsampling
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2), #downsampling by factor 2 in all 3 stages of ResNet3dSlowOnly
        temporal_strides=(1, 1, 2)), #downsampling by factor 2 only in the last of 3 stages off ResNet3dSlowOnly
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=300, #300 for wlasl300
        dropout=0.5),
    test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset' #
ann_file = 'julia/WLASL300/pyskl_mediapipe_annos_2d_denormalized_NOSE_FACE_HANDS.pkl' #TODO 

#Define kepoints for "Flip" Module
#point 0 is the nose which is centered, and therefore neither left nor right keypoint
#keypoints 1-20 are face keypoints
#keypoints 21-41 are left hand keypoints, keypoints 42-62 are right hand keypoints
#[46, 52, 53, 65, 295, 283, 282, 276, 7, 159, 155, 145, 382, 386, 249, 374, 324, 13, 78, 14] 
#right face: keypoints 46, 52, 53, 65    , 7, 159, 155, 145,      324
#left face: 295, 283, 282, 276     , 382, 386, 249, 374,       78
#kps 13 and 14 (index 18 and 20) (and 0 for the nose) are centered
left_face = [1, 2, 3, 4, 9, 10, 11, 12, 17]
right_face = [5, 6, 7, 8, 13, 14, 15, 16, 19]
#left and right hand keypoints each with 21 kps starting at index 21, left hand is extracted first
left_hand = list(range(21, 42))
right_hand = list(range(42, 63)) 
left_kp = left_face + left_hand
right_kp = right_face + right_hand

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48), #divide video in n segments and choose one random frame from each segment
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    #resize the heatmap (not the original image) by subject-centered cropping
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)), #?
    dict(type='Resize', scale=(128, 128), keep_ratio=False), #warum zwei mal?
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False), #with_limb kontrolliert ob heatmap nur aus punkten (keypoints) oder auch verbindungen zwischen den kps generiert, joints (no limbs) ist für SLR besser, s. heatmap visualization
    dict(type='FormatShape', input_format='NCTHW_Heatmap'), #?
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]), #?
    dict(type='ToTensor', keys=['imgs', 'label']) #?
    #PackActionInputs ?
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(128, 128), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10), #for faster inference, multi-clip testing can be disabled here by setting num_clips=1
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(128, 128), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'), #?
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]), #?
    dict(type='ToTensor', keys=['imgs']) #?
]
data = dict(
    videos_per_gpu=4, #batch size, decrease videos per gpu to prevent cuda out of memory, Attention: adapt lr accordingly
    workers_per_gpu=2, #decrease worker
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)), #train split
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline), #val split
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline)) #test split
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0003) #adapt lr linear to batch size
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 240 #epochs, attention: will be overwritten if early_stopping is used and max_epochs is set lower than total_epochs
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
log_level = 'INFO'
work_dir = './work_dirs/julia/AblationStudies/hands_nose_face_flip_128px_lr_0_01_correctedFlip' #TODO
load_from = 'work_dirs/julia/AblationStudies/hands_nose_face_flip_128px_lr_0_001/epoch_90.pth'

