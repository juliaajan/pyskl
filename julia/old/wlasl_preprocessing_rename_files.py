#script to go over the video files created by  downloading and preprocessing the WLASL videos
# this step is a preparation for the ntu_pose_extraction.py file
#to create an annotation file and rename all video files according to the annotation file
#so that ntu_pose_extraction.py can get the gloss directly from the filename and save the gloss (label) in the pickle file
import argparse
import json
import os

#save all mappings from video_id to their corresponding gloss for fast lookup of the glosses
#when renaming the files according to their gloss
video_id_to_gloss_lookup = {}


#create a new file (annotation_file) that contains all glosses (words) in the WLASL dataset, one per line
#additionally, create a lookup table that maps video_ids to their corresponding glosses
def create_annotation_file(path_to_annotation_file, path_wlasl_json):
    print("Creating annotation file...")
   

    #open the WLASL json file where video_ids and their corresponding glosses (words) are stored
    with open(path_wlasl_json) as wlasl_file:
        content = json.load(wlasl_file)

    #gloss_index = 0 #index = rownr in the annotation file 

    #use text mode 'w' to make sure to overwrite a possibly already existing file
    with open(path_to_annotation_file, 'w') as anno_file:
        for entry in content:
            gloss = entry['gloss']
            #write a line to the annotation file for each gloss entry and add a linebreak
            anno_file.write(f"{gloss}\n")
            print(f"Wrote gloss '{gloss}' to annotation file.")

            #for all entries in this gloss (=all videos belongign to this word), save their video ids in a lookup table
            for video in entry['instances']:
                video_id =video.get("video_id")
                video_id_to_gloss_lookup[video_id] = gloss
            #HIER EFRTIG
    print("Done creating annotation file")


def rename_files(path_to_annotation_file, path_video_files):
    print("Start renaming video files according to their glosses...")
    #ATTENTION: the gloss_identifier has to be identical to the one in ntu_pose-extraction.py
    #also, zfill(5) has to match 
    gloss_identifier = "_GL" #string after which the gloss number will be appended to the filename, eg 12345.mp4, belongs to gloss 10 -> 12345_GLOSS00010.mp4


    #iterate over all video files, check in which line of the annotation file the corresponding gloss entry is, and rename the file accordingly
    for file_name in os.listdir(path_video_files):
        #TODO: in if-condition prüfen ob video file ist, aber gibt mehrere endungen (mp4, avi, mov etc) if file_name.endswith('....') and
        if gloss_identifier not in file_name:  #make sure to not rename already renamed files again
            video_id = file_name.split('.')[0]  #split right before the . ending and get the video id (=file_id) #TODO: statt vorm . splitten vor der Video-Endung splitten (mp4, avi, mov etc)
            video_format = file_name.split('.')[1]  #get the video format/ending (e.g. mp4)

            try:
                #get the gloss of this video from the lookup table
                gloss = video_id_to_gloss_lookup[video_id]
                print(f"Found video {video_id}, belongs to gloss {gloss}")

                #get the line in which this gloss appears in the annotation file
                with open(path_to_annotation_file, 'r') as anno_file:
                    for line_number, line in enumerate(anno_file, start=1):
                        if line.strip() == gloss:
                            gloss_line_number = line_number
                            #rename the video file by adding the line in which the gloss appears 
                            new_file_name = video_id + gloss_identifier + str(gloss_line_number).zfill(5) + "." + video_format
                            os.rename(os.path.join(path_video_files, file_name), os.path.join(path_video_files, new_file_name))
                            print("Renamed file", file_name, "to", new_file_name)
                            break
            except KeyError:
                print(f"\033[31mNo gloss found for video id {video_id}\033[0m")
                continue

    print("Done renaming video files.")
            
            #TODO: check to which gloss this video_id belongs (either int he annotation file or in the hash table)
            #TODO: get the number of the gloss (=row in the annotation file)
            #TODO: append the gloss number with a leading "G" (for Gloss) to the filename
            #HIER WEITER MACHEN
            #TODO: da ich das file umbenenne, kann es sein dass in endlos schleife läuft weil immer "neue" (umbenannte) files gefunden werden

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Once downloaded all WLASL videos, this script created an annotation file containing all glosses (words), and renames the downloaded videos by adding their gloss to the filename.')
    parser.add_argument('path_video_files', type=str, help='path to folder containing all downloaded and preprocessed WLASL video files, will likely end with WLASL/start_kit/videos')
    parser.add_argument('path_wlasl_json', type=str, help='path where the WLASL json file containing file details, video ids, glosses, ... is located, e.g. WLASL/WLASL_v0.3.json')
    parser.add_argument('path_to_annotation_file', type=str, help='path where the annotation file with gloss-labels will be created')
    args = parser.parse_args()

    path_to_annotation_file = args.path_to_annotation_file
    #check if the given path is a directory or a filename, if dir, append 'annotation.txt' to it
    if os.path.isdir(path_to_annotation_file):
        path_to_annotation_file = os.path.join(path_to_annotation_file, 'annotation.txt')


    create_annotation_file(path_to_annotation_file, args.path_wlasl_json)
    rename_files(path_to_annotation_file, args.path_video_files)
