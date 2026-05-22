#script to split the WLASL dataset into train, val, test sets

#pickle files with skeleton data are downloaded and extracted from opehands (MultiSign ISLR)
#https://openhands.ai4bharat.org/en/latest/instructions/datasets.html

#WLASL_v0.3.json file from the WLASL repository specifies for each video if it belongs to train, val or test set
#https://github.com/dxli94/WLASL/blob/master/start_kit/WLASL_v0.3.json


#TODO: muss noch anpassen nachdem ich die pickle files nichtmehr von openHands fertig runterlade sondern selbst mit ntuu_extrace_pose escript erstelle
import json
import pickle
from pathlib import Path
import os

#specify path to pickle file and WLASL json file
path_pickle_files = "../data/WLASL/WLASL/wlasl_poses_pickle"
path_wlasl_json = "../WLASL/start_kit/WLASL_v0.3.json"

#count number of pickle files
nr_files = sum(1 for x in Path(path_pickle_files).glob('*') if x.is_file()) 


#open the WLASL json file that specifies train, val, test splits
def get_train_test_val_ids(path_wlasl_json):
    with open(path_wlasl_json) as file:
        content = json.load(file)

    train_ids = []
    test_ids = []
    val_ids = []

    for gloss_entry in content:
        for inst in gloss_entry["instances"]:
            if inst.get("split") == "train":
                train_ids.append(inst.get("video_id"))
            elif inst.get("split") == "test":
                test_ids.append(inst.get("video_id"))
            elif inst.get("split") == "val":
                val_ids.append(inst.get("video_id"))
            else:
                print(f"\033[31mNo train/test/val indicator found for video with id {inst.get("video_id")}\033[0m")
    return train_ids, test_ids, val_ids

def move_to_folders(train_ids, test_ids, val_ids, path_pickle_files):
    #create folders for train, test and val sets
    trainpath = os.path.join(path_pickle_files, 'train')
    testpath = os.path.join(path_pickle_files, 'test')
    valpath = os.path.join(path_pickle_files, 'val')
    print(trainpath)

    if not os.path.exists(trainpath):
        os.makedirs(trainpath)
    if not os.path.exists(testpath):
        os.makedirs(testpath)
    if not os.path.exists(valpath):
        os.makedirs(valpath)
    
    #iterate over all pickle files and move them to corresponding folder
    for file_name in os.listdir(path_pickle_files):
        if file_name.endswith('.pkl'):
            file_id = file_name.split('.')[0]  #split right before the .pkl ending and get the video id
            source = os.path.join(path_pickle_files, file_name)
            if file_id in train_ids:
                destination = os.path.join(trainpath, file_name)
                print(f"Moving file {file_name} to {trainpath}")
                os.rename(source, destination)
            elif file_id in test_ids:
                destination = os.path.join(testpath, file_name)
                print(f"Moving file {file_name} to {testpath}")
                os.rename(source, destination)
            elif file_id in val_ids:
                destination = os.path.join(valpath, file_name)
                print(f"Moving file {file_name} to {valpath}")
                os.rename(source, destination)
            else:
                print(f"\033[31mCould not move because no train/test/val indicator found for file {file_name}\033[0m")
    
        

#get the ids that belong to train, test and val sets from the WLASL json file
train_ids, test_ids, val_ids = get_train_test_val_ids(path_wlasl_json)

#should be: train: 14289, test: 2878, val: 3916
print("Number of train samples:", len(train_ids))
print("Number of test samples:", len(test_ids))
print("Number of val samples:", len(val_ids))

print("Total  number of pickle files: ", nr_files)
print("Total number of samples2: ", len(train_ids) + len(test_ids) + len(val_ids))

if(nr_files != len(train_ids) + len(test_ids) + len(val_ids)):
    print("\033[31mError: Number of files does not match number of samples in WLASL json file!\033[0m")

#here, the number of pickle files and the number of train/test/val samples is not the same, 
#probably because some videos are not available anymore on youtube and have been removed from the WLASL json but not from the pickle files
#12 files won't be able to be moved because of missing train/test/val annotation
move_to_folders(train_ids, test_ids, val_ids, path_pickle_files)


