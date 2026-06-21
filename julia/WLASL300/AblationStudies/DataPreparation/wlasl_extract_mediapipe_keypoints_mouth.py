#Starting from the annotation file including keypoints for the nose, face and hands,
#this file removes the hip nose, eyebrows and eye keypoints to enable training with mouth_hands.py

import argparse
import os
import numpy as np
from mmcv import load, dump


def remove_nose_eyebrows_eyes_keypoints(input_path):
    """
    Remove nose (index 0) and the first 16 face keypoints (indices 1-16),
    keep the following 4 keypoints for the mouth and the rest of the hand keypoints


    Input format: (persons, frames, keypoints, 2) with 67 keypoints
    Output format: (persons, frames, keypoints, 2) with 46 keypoints
    """

    print(f"Loading annotations from: {input_path}")
    anno_dict = load(input_path)
    
    #extract annotations and split dict, split stays the same
    annotations = anno_dict['annotations']
    split_dict = anno_dict['split']
    
    #upperbody keypoints are indices 0-24, followed by 21 for left hand and 21 for right hand
    #indices 23 and 24 (hips) should be removed
    indices_to_keep = [i for i in range(17, 68)]
    print(f"Indices to keep: {indices_to_keep}")

    assert (len(indices_to_keep) == 46), f"Expected to keep 46 keypoints, but got {len(indices_to_keep)}"
    

    for i, anno in enumerate(annotations):
        #print progress
        if (i+1) % 100 == 0:
            print(f"Processing video {i+1}/{len(annotations)}")

        #remove hip keypoints, keep the rest
        #keypoint shape: (num_persons, num_frames, num_keypoints, 2 coordinates)
        anno['keypoint'] = anno['keypoint'][:, :, indices_to_keep, :]
        
        #remove hip keypoint scores, keep the rest
        #keypoint_score shape: (num_persons, num_frames, num_keypoints)
        anno['keypoint_score'] = anno['keypoint_score'][:, :, indices_to_keep]
    
    #create output dictionary
    output_dict = {
        'split': split_dict,
        'annotations': annotations
    }
    
    #save results
    output_path = os.path.split(input_path)[0] #get path to input file without filename
    output_filename = "pyskl_mediapipe_annos_2d_denormalized_MOUTH_HANDS.pkl"
    output_file= os.path.join(output_path, output_filename) 

    dump(output_dict, output_file)

    print(f"Successfully saved annotations with removed nose, eyebrow and eye keypoints to: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Remove nose, eyebrow and eye from annotation-pickle')
    #fileName: pyskl_mediapipe_annos_2d_denormalized_NOSE_FACE_HANDS.pkl

    parser.add_argument('input_path', type=str, help='input path to the annotation pickle file containing upperbody and face keypoints')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    assert "pyskl_mediapipe_annos_2d_denormalized_NOSE_FACE_HANDS.pkl" in args.input_path

    
    remove_nose_eyebrows_eyes_keypoints(args.input_path)