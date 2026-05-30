import argparse
import os

from mmcv import load
from julia.WLASL300.mediapipe.hands_upperbody.visualize import Vis2DPoseMediaPipe

def load_annotations(anno_pickle_path, path_video):
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
    
    return anno, video_id


def display_upperbody_hands(anno_pickle_path, path_video):
    anno, video_id = load_annotations(anno_pickle_path, path_video)

    video = Vis2DPoseMediaPipe(
    anno, 
    video=path_video,
    show_face=False,      # Face als Punkte (optional)
    show_hands=True,      # Hände anzeigen
    show_body=False,       # Körper anzeigen
    show_upperbody=True,   # Oberkörper anzeigen
    show_shoulders_arms=False,  # Schultern und Arme anzeigen
    thre=0.0,             # Confidence threshold
    fps=25
)
    return video, video_id

def display_shoulders_arms_hands(anno_pickle_path, path_video):
    anno, video_id = load_annotations(anno_pickle_path, path_video)

    video = Vis2DPoseMediaPipe(
    anno, 
    video=path_video,
    show_face=False,      # Face als Punkte (optional)
    show_hands=True,      # Hände anzeigen
    show_body=False,       # Körper anzeigen
    show_upperbody=False,   # Oberkörper anzeigen
    show_shoulders_arms=True,  # Schultern und Arme anzeigen
    thre=0.0,             # Confidence threshold
    fps=25
)
    return video, video_id



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualise upperbody and shoulder_hands keypoints on the original video')
    #path to the video list created by wlasl_prepare_mediapipe_extraction.py
    parser.add_argument('anno_pickle_file_upperbody', type=str, help='the path to the annotation pickle file containing upperbody and hand kps')
    parser.add_argument('anno_pickle_file_shoulders_arms', type=str, help='the path to the annotation pickle file containing shoulder and arm kps')
    parser.add_argument('video_file', type=str, help='the path to the input video file')
    parser.add_argument('output_path', type=str, help='the path to folder where the output videos should be saved')
    args = parser.parse_args()

    #the pickle files must have a .pkl suffix
    assert args.anno_pickle_file_upperbody.endswith('.pkl')
    assert args.anno_pickle_file_shoulders_arms.endswith('.pkl')

    #visualize upperbody keypoints
    video_upperbody_hands, video_id = display_upperbody_hands(args.anno_pickle_file_upperbody, args.video_file)
    output_file_upperbody = os.path.join(args.output_path, f'skeleton_vis_noface_upperbody{video_id}.mp4')
    video_upperbody_hands.write_videofile(output_file_upperbody)
    print(f"Saved visualized video with upperbody keypoints to {output_file_upperbody}")


    #visualize shoulder and hand keypoints
    video_shoulders_arms_hands, video_id = display_shoulders_arms_hands(args.anno_pickle_file_shoulders_arms, args.video_file)
    output_file_shoulders_arms = os.path.join(args.output_path, f'skeleton_vis_noface_shoulders_arms{video_id}.mp4')
    video_shoulders_arms_hands.write_videofile(output_file_shoulders_arms)
    print(f"Saved visualized video with upperbody keypoints to {output_file_shoulders_arms}")


    #example: 32167


