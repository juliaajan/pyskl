
import argparse
from mmcv import load, dump
from pyskl.smp import *
import os
#oriented on pyskl/examples/extract_diving48_skeleton/diving48_example.ipynb step1+2

#create a lookup table that maps video_ids to their corresponding glosses
def create_label_lookup(path_wlasl_json):
    video_id_to_gloss_lookup = {}
    
    #open the WLASL json file where video_ids and their corresponding glosses (words) are stored
    with open(path_wlasl_json) as wlasl_file:
        content = json.load(wlasl_file)
    
    #go through all gloss entries
    for entry in content:
        gloss = entry['gloss']
        #go through alll videos for that gloss
        for video in entry['instances']:
            video_id =video.get("video_id")
            video_id_to_gloss_lookup[video_id] = gloss

    return video_id_to_gloss_lookup
     

#create a label file that contains the full path to each video followed by its corresponding gloss (label)
#(see examples/extract_diving48_skeleton/diving48_example.ipynb step 2)
#TODO: prüfen wo das label file erstellt wird
def create_label_file(path_wlasl300, video_id_to_gloss_lookup):
    lines = []
    name_label_file = "wlasl300.list"

    #generate video list for 2d skeleton extraction with mediapipe (as in diving48_example.ipynb)
    for root, dirs, files in os.walk(path_wlasl300):
        for file in files:
            #for wlasl300, all videos are in mp4 format. If this is not the case, this line needs to be adapted or removed
            if file.lower().endswith(".mp4"):
                video_id = file.split('.mp4')[0]  #split right before the .mp4 ending and get the video id (=file_id)
                full_path = os.path.join(root, file)
                label =  video_id_to_gloss_lookup[video_id] #get thee corresponding label(gloss) for this video from the lookup table

                line = f"{full_path} {label}"
                lines.append(line)
                print("Added line:", line)

    mwlines(lines, name_label_file)
    print(f"Created {name_label_file} with {len(lines)} entries")
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Once downloaded the WLASL300 from kaggle, this script does the preprocessing to extract mediapipe keypoints')
    parser.add_argument('path_wlasl300', type=str, help='path to folder containing the train, test, val folders in the wlasl300 dataset, e.g. WLASL/WLASL_300')
    #parser.add_argument('output_path_to_annotation_file', type=str, help='path where the annotation file will be created')
    parser.add_argument('path_wlasl_json', type=str, help='path to the WLASL json file containing video_ids and their corresponding glosses')
    args = parser.parse_args()

    video_id_to_gloss_lookup = create_label_lookup(args.path_wlasl_json)
    create_label_file(args.path_wlasl300, video_id_to_gloss_lookup)

