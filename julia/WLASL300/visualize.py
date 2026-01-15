import os
import argparse
import copy as cp

import cv2
import moviepy.editor as mpy
import numpy as np
from mmcv import load
from tqdm import tqdm



#adapted from pyskl/demo/vis_skeleton.ipynb
skeleton_map = dict(
    mediapipe_body=[
        # Body connections (33 keypoints, indices 0-32)
        (0, 1, 'f'), (1, 2, 'f'), (2, 3, 'f'), (3, 7, 'f'),  # Face
        (0, 4, 'f'), (4, 5, 'f'), (5, 6, 'f'), (6, 8, 'f'),  # Face
        (9, 10, 'f'),  # Mouth
        (11, 12, 't'), (11, 13, 'lu'), (13, 15, 'lu'),  # Left arm
        (12, 14, 'ru'), (14, 16, 'ru'),  # Right arm
        (11, 23, 'ld'), (23, 25, 'ld'), (25, 27, 'ld'),  # Left leg
        (12, 24, 'rd'), (24, 26, 'rd'), (26, 28, 'rd'),  # Right leg
        (23, 24, 't'),  # Hip
        (27, 29, 'ld'), (27, 31, 'ld'),  # Left foot
        (28, 30, 'rd'), (28, 32, 'rd')   # Right foot
    ],
    mediapipe_hand=[
        # Hand connections (21 keypoints per hand)
        (0, 1, 't'), (1, 2, 'f'), (2, 3, 'f'), (3, 4, 'f'),      # Thumb
        (0, 5, 't'), (5, 6, 'lu'), (6, 7, 'lu'), (7, 8, 'lu'),   # Index
        (0, 9, 't'), (9, 10, 'ru'), (10, 11, 'ru'), (11, 12, 'ru'),  # Middle
        (0, 13, 't'), (13, 14, 'ld'), (14, 15, 'ld'), (15, 16, 'ld'),  # Ring
        (0, 17, 't'), (17, 18, 'rd'), (18, 19, 'rd'), (19, 20, 'rd')   # Pinky
    ],
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
    annotations = load(anno_pickle_path)

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
    show_face=True,      # Face als Punkte (optional)
    show_hands=True,      # Hände anzeigen
    show_body=True,       # Körper anzeigen
    thre=0.3,             # Confidence threshold
    fps=24
)
    
    output_file = os.path.join(output_path, f'skeleton_vis_{video_id}.mp4')
    video.write_videofile(output_file)
    print(f"Saved visualized video to {output_file}")



def Vis2DPoseMediaPipe(item, thre=0.2, out_shape=(540, 960), fps=24, video=None, 
                       show_face=True, show_hands=True, show_body=True):
    tx = cp.deepcopy(item)
    item = tx

    if isinstance(item, str):
        item = load(item)

    kp = item['keypoint'] #shape: (1, 95, 543, 3)
    print("Keypoint shape:", kp.shape)

    num_keypoints = kp.shape[-2]
    #check if face is included
    has_face = (num_keypoints == 543)
    
    if 'keypoint_score' in item:
        kpscore = item['keypoint_score']
        print("Keypoint score shape:", kpscore.shape)
        #concatenate the keypoint scores to kp
        kp = np.concatenate([kp, kpscore[..., None]], -1)

    total_frames = None
    assert len(kp.shape) == 4, f"Expected 4 dimensions for each keypoint, got {len(kp.shape)}"
    total_frames = item.get('total_frames', kp.shape[1])


    #load video frames
    if video is None:
        frames = [np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255 
                  for i in range(total_frames)]
    else:
        frames = load_frames(video)
        if out_shape is None:
            out_shape = frames[0].shape[:2]
        frames = [cv2.resize(x, (out_shape[1], out_shape[0])) for x in frames]

    #scale keypoints to output shape
    assert kp.shape[-1] in [3, 4], f"Expected 3 or 4 coords, got {kp.shape[-1]}"
    img_shape = item.get('img_shape', out_shape)
    img_height, img_width = img_shape[0], img_shape[1]
    out_height, out_width = out_shape[0], out_shape[1]
    kp[..., 0] *= out_width / img_width
    kp[..., 1] *= out_height / img_height

    #prepare keypoints per frame
    kps = [kp[:, i] for i in range(total_frames)]


    body_edges = skeleton_map['mediapipe_body']
    hand_edges = skeleton_map['mediapipe_hand']
    
    color_map = {
        'ru': ((0, 0x96, 0xc7), (0x3, 0x4, 0x5e)),
        'rd': ((0xca, 0xf0, 0xf8), (0x48, 0xca, 0xe4)),
        'lu': ((0x9d, 0x2, 0x8), (0x3, 0x7, 0x1e)),
        'ld': ((0xff, 0xba, 0x8), (0xe8, 0x5d, 0x4)),
        't': ((0xee, 0x8b, 0x98), (0xd9, 0x4, 0x29)),
        'f': ((0x8d, 0x99, 0xae), (0x2b, 0x2d, 0x42))
    }


    #temporal debug code 
    for i in range(min(5, total_frames)):  # Erste 5 Frames
        kp_frame = kps[i]
        ske = kp_frame[0]
        print(f"\nFrame {i}:")
        print(f"  Nose (kp 0): x={ske[0][0]:.1f}, y={ske[0][1]:.1f}, conf={ske[0][3]:.3f}")
        print(f"  Left wrist (kp 15): x={ske[15][0]:.1f}, y={ske[15][1]:.1f}, conf={ske[15][3]:.3f}")
        print(f"  Right wrist (kp 16): x={ske[16][0]:.1f}, y={ske[16][1]:.1f}, conf={ske[16][3]:.3f}")

    for i in tqdm(range(total_frames)):
        kp_frame = kps[i]
        for m in range(kp_frame.shape[0]):  # For each person
            ske = kp_frame[m]
            
            # Draw body (keypoints 0-32)
            if show_body:
                for st, ed, co in body_edges:
                    j1, j2 = ske[st], ske[ed]
                    j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                    #set the confidence (was appended from keyypoint_score to kp)
                    if kp.shape[-1] == 4:
                        conf = min(j1[3], j2[3])  #index 3 = confidence score
                    else:
                        conf = 1.0  #fallback if no confidence provided
                    
                    if conf > thre:
                        co_tup = color_map[co]
                        color = [x + (y - x) * (conf - thre) / 0.8 for x, y in zip(co_tup[0], co_tup[1])]
                        color = tuple([int(x) for x in color])
                        frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)
            
            # Draw hands
            if show_hands:
                if has_face:
                    # Left hand: keypoints 501-521
                    left_hand = ske[501:522]
                    # Right hand: keypoints 522-542
                    right_hand = ske[522:543]
                else:
                    # Without face: keypoints 33-53 (left), 54-74 (right)
                    left_hand = ske[33:54]
                    right_hand = ske[54:75]
                
                # Draw left hand
                for st, ed, co in hand_edges:
                    j1, j2 = left_hand[st], left_hand[ed]
                    j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                    #set the confidence
                    if kp.shape[-1] == 4:
                        conf = min(j1[3], j2[3]) #index 3 = confidence score
                    else:
                        conf = 1.0  #fallback if no confidence provided
                    if conf > thre:
                        color = (255, 0, 0)  # Blue for left hand
                        frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)
                
                # Draw right hand
                for st, ed, co in hand_edges:
                    j1, j2 = right_hand[st], right_hand[ed]
                    j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                    #set the confidence
                    if kp.shape[-1] == 4:
                        conf = min(j1[3], j2[3]) #index 3 = confidence score
                    else:
                        conf = 1.0  #fallback if no confidence provided
                    if conf > thre:
                        color = (0, 0, 255)  # Red for right hand
                        frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)
            
            # Optionally draw face keypoints as points (too many for skeleton)
            if show_face and has_face:
                face_kps = ske[33:501]  # 468 face keypoints
                for fkp in face_kps:
                    x, y = int(fkp[0]), int(fkp[1])
                     #set the confidence
                    if kp.shape[-1] == 4:
                        conf = fkp[3]  
                    else:
                        conf = 1.0
                    if conf > thre:
                        frames[i] = cv2.circle(frames[i], (x, y), 1, (0, 255, 0), -1)

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


