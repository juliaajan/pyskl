
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21 #x2 for both hands

#TODO: auf slowfast ändenr und gucken was anpassen muss
#TODO: muss ich depth setzen? Vielen nutzen depth=50, pretrained=None
model = dict(
    type='Recognizer3D', #
    backbone=dict(
        type='ResNet3dSlowOnly', 
        in_channels= N_BODY_LANDMARKS + 2 * N_HAND_LANDMARKS, #number of keypoints
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
ann_file = 'julia/WLASL300/pyskl_mediapipe_annos_2d_denormalized_NOFACE.pkl' #TODO 

#Define kepoints for "Flip" Module
#point 0 is the nose which is centered, and therefore neither left nor right keypoint
left_body = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
right_body = [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
#left and right hand keypoints each with 21 kps starting at index 33, left hand is extracted first
left_hand = list(range(33, 54))
right_hand = list(range(54, 75)) 
left_kp = left_body + left_hand
right_kp = right_body + right_hand

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
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0003) #adapt lr linear to batch size
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
    patience=3,
    min_delta=0.01,
    max_epochs=240,
    mode='min')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/julia/mediapipe_wlasl300_noface_240epochs_flip_earlyStopping_loss_Resize128_correctedValLoss_lr_0_001' #TODO
