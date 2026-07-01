import os
import argparse
import copy as cp

import cv2
import moviepy.editor as mpy
import numpy as np
from mmcv import load
from tqdm import tqdm

skeleton_map = dict(
    mediapipe_hand=[
        # Hand connections (21 keypoints per hand) inside the hand keypoints
        (0, 1, 't'), (1, 2, 'f'), (2, 3, 'f'), (3, 4, 'f'),      # Thumb
        (0, 5, 't'), (5, 6, 'lu'), (6, 7, 'lu'), (7, 8, 'lu'),   # Index
        (0, 9, 't'), (9, 10, 'ru'), (10, 11, 'ru'), (11, 12, 'ru'),  # Middle
        (0, 13, 't'), (13, 14, 'ld'), (14, 15, 'ld'), (15, 16, 'ld'),  # Ring
        (0, 17, 't'), (17, 18, 'rd'), (18, 19, 'rd'), (19, 20, 'rd')   # Pinky
    ],
    #face connections inside the FACE keypoints
    #order: [46, 52, 53, 65, 295, 283, 282, 276, 7, 159, 155, 145, 382, 386, 249, 374, 324, 13, 78, 14]
    mediapipe_mouth=[
        (0, 1, 'lf'), (1, 2, 'rf'), (2, 3,'rf'), (3, 0,'lf'), #left and right mouth corners
    ]
)

def load_frames(vid):
    vid = cv2.VideoCapture(vid)
    images = []
    success, image = vid.read()
    while success:
        images.append(np.ascontiguousarray(image[..., ::-1]))
        success, image = vid.read()
    return images


# Visualize 2D Skeletons with the original RGB Video
def display_keypoints_on_video(anno_pickle_path, path_video, output_path):
    data = load(anno_pickle_path)
    annotations = data['annotations']

    #extract the video_id from the given video path
    #and try to find it in the annotations file 
    video_id = os.path.splitext(os.path.basename(path_video))[0]
    anno = None
    for a in annotations:
        if a['frame_dir'] == video_id:
            anno = a
            break
    if anno is None:
        raise ValueError(f"Video {video_id} not found in annotation pickle file.")

    video = Vis2DPoseMediaPipe(
    anno, 
    video=path_video,
    show_mouth=True,    
    show_hands=True,     
    thre=0.0,             # Confidence threshold
    fps=25
)
    
    output_file = os.path.join(output_path, f'skeleton_vis_mouth_hands_{video_id}.mp4')
    video.write_videofile(output_file)
    print(f"Saved visualized video to {output_file}")



def Vis2DPoseMediaPipe(item, thre=0.0, out_shape=(1080, 1920), fps=25, video=None, show_mouth=True, show_hands=True):
    tx = cp.deepcopy(item)
    item = tx

    if isinstance(item, str):
        item = load(item)

    kp = item['keypoint'] #shape: (M, T, V, C),
    print("Keypoint shape:", kp.shape)

    num_keypoints = kp.shape[-2]
    #check if face is included
    
    if 'keypoint_score' in item:
        kpscore = item['keypoint_score']
        print("Keypoint score shape:", kpscore.shape)
        #concatenate the keypoint scores to kp
        kp = np.concatenate([kp, kpscore[..., None]], -1) #shape now: (M, T, V, C+1)

    total_frames = None
    assert len(kp.shape) == 4, f"Expected 4 dimensions for each keypoint (M, T, V, C+1), got {len(kp.shape)}"
    total_frames = item.get('total_frames', kp.shape[1])


    #load video frames
    if video is None:
        frames = [np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255 
                  for i in range(total_frames)]
    else:
        frames = load_frames(video)
        original_shape = frames[0].shape[:2]
        if out_shape is None:
            out_shape = original_shape
        #keypoints need to be scaled to the desired output scale, if it differs from input scale
        if out_shape != original_shape:
            #out_shape has format (height, width), but keypoints have format (x, y) -> width*x, height*y
            scale_height = out_shape[0] / original_shape[0] #scale of output height to input height
            scale_width = out_shape[1] / original_shape[1] #scale of output width to input width
            kp = kp.copy()
            kp[..., 0] *= scale_width  # multiply x-cord of kp with scale_width
            kp[..., 1] *= scale_height  # multiply y-cord of kp with scale_height
        frames = [cv2.resize(f, (out_shape[1], out_shape[0])) for f in frames]

    assert kp.shape[-1] == 3, f"Expected 2 coords and one score for each keypoint, got {kp.shape[-1]}"

    #prepare keypoints per frame
    kps = [kp[:, i] for i in range(total_frames)]

    mouth_edges = skeleton_map['mediapipe_mouth']
    hand_edges = skeleton_map['mediapipe_hand']
    
    color_map = {
        #BGR format! Not RGB
        'ru': ((0, 0x96, 0xc7), (0x3, 0x4, 0x5e)),
        'rd': ((0xca, 0xf0, 0xf8), (0x48, 0xca, 0xe4)),
        'lu': ((0x9d, 0x2, 0x8), (0x3, 0x7, 0x1e)),
        'ld': ((0xff, 0xba, 0x8), (0xe8, 0x5d, 0x4)),
        't': ((0xee, 0x8b, 0x98), (0xd9, 0x4, 0x29)),
        'f': ((0x8d, 0x99, 0xae), (0x2b, 0x2d, 0x42)),
        'rf': ((0xff, 0xff, 0), (0xff, 0xff, 0)), #right face yellow 
        'lf': ((0xff, 0, 0xff), (0xff, 0, 0xff)), #left face puprle

    }

    for i in tqdm(range(total_frames)):
        kp_frame = kps[i]        
        for m in range(kp_frame.shape[0]):  # For each person
            ske = kp_frame[m]
            assert len(ske) == 46, f"Expected 46 keypoints (mouth + 2 hands), got {len(ske)}"

            # Draw mouth keypoints
            if show_mouth:
                mouth_kps = ske[0:4] 
                print([(i, mouth_kps[i][0], mouth_kps[i][1]) for i in range(4)])
                
                for st, ed, co in mouth_edges:
                    j1, j2 = mouth_kps[st], mouth_kps[ed]
                    j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                    if kp.shape[-1] == 3:
                        conf = min(j1[2], j2[2])
                    else:
                        conf = 0.0
                    
                    if conf > thre:
                        co_tup = color_map[co]
                        color = [x + (y - x) * (conf - thre) / 0.8 for x, y in zip(co_tup[0], co_tup[1])]
                        color = tuple([int(x) for x in color])
                        frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=1)

            # Draw hands
            if show_hands:
                left_hand = ske[4:25]
                right_hand = ske[25:46]

                # Draw left hand
                for st, ed, co in hand_edges:
                    j1, j2 = left_hand[st], left_hand[ed]
                    j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                    #set the confidence
                    if kp.shape[-1] == 3:
                        conf = min(j1[2], j2[2]) #index 2 = confidence score
                    else:
                        conf = 0.0  #fallback if no confidence provided - assume valid keypoint
                    if conf > thre:
                        color = (255, 0, 0)  # Red for left hand
                        frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)
                
                # Draw right hand
                for st, ed, co in hand_edges:
                    j1, j2 = right_hand[st], right_hand[ed]
                    j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                    #set the confidence
                    if kp.shape[-1] == 3:
                        conf = min(j1[2], j2[2]) #index 2 = confidence score
                    else:
                        conf = 0.0  #fallback if no confidence provided - assume valid keypoint
                    if conf > thre:
                        color = (0, 0, 255)  # Blue for right hand
                        frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)

    return mpy.ImageSequenceClip(frames, fps=fps)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualise the keypoints on the original video')
    #path to the video lsit created by wlasl_prepare_mediapipe_extraction.py
    parser.add_argument('anno_pickle_file', type=str, help='the path to the annotation pickle file')
    parser.add_argument('video_file', type=str, help='the path to the input video file')
    parser.add_argument('output_path', type=str, help='the path to folder where the output video should be saved')
    args = parser.parse_args()

    #the pickle file must have a .pkl suffix
    assert args.anno_pickle_file.endswith('.pkl')
    display_keypoints_on_video(args.anno_pickle_file, args.video_file, args.output_path)


    #example: 32167


