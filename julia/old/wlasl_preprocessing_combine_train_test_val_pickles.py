#script to split the WLASL dataset into train, val, test sets

#pickle files created with ntu_pose_extraction.py are all in one folder

#WLASL_v0.3.json file from the WLASL repository specifies for each video if it belongs to train, val or test set
#https://github.com/dxli94/WLASL/blob/master/start_kit/WLASL_v0.3.json


import argparse
import json
import pickle
from pathlib import Path
import os


#open the WLASL json file that specifies train, val, test splits
def get_train_test_val_ids(path_wlasl_json):
    with open(path_wlasl_json) as file:
        content = json.load(file)

    train_ids = []
    test_ids = []
    val_ids = []

    for entry in content:
       for inst in entry['instances']:
            if inst.get("split") == "train":
                train_ids.append(inst.get("video_id"))
            elif inst.get("split") == "test":
                test_ids.append(inst.get("video_id"))
            elif inst.get("split") == "val":
                val_ids.append(inst.get("video_id"))
            else:
                video_id = inst.get("video_id")
                print(f"\033[31mNo train/test/val indicator found for video with id {video_id}\033[0m")
    return train_ids, test_ids, val_ids


def combine_pickle_files(train_ids, test_ids, val_ids, path_pickle_files):

    path_train_pickle = os.path.join(path_pickle_files, 'wlasl_train_combined.pkl')
    path_test_pickle = os.path.join(path_pickle_files, 'wlasl_test_combined.pkl')
    path_val_pickle = os.path.join(path_pickle_files, 'wlasl_val_combined.pkl')

    train_data = []
    test_data = []
    val_data = []

    
    #iterate over all pickle files and move them to corresponding folder
    for file_name in os.listdir(path_pickle_files):
        #skip files that are not pickle files
        if not file_name.endswith('.pkl'):
            continue
        # skip already combined files
        if file_name.endswith("_combined.pkl"):
            continue

        file_id = file_name.split('_GL')[0]  #split right before the gloss identifier to get the video id (=file name)
        file_path = os.path.join(path_pickle_files, file_name)
        # load pickle file
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if file_id in train_ids:
            train_data.append(data)
            print(f"Added data of {file_name} to train_data")
        elif file_id in test_ids:
            test_data.append(data)
            print(f"Added data of {file_name} to test_data")
        elif file_id in val_ids:
            val_data.append(data)
            print(f"Added data of {file_name} to val_data")
        else:
            print(f"\033[31mNo train/test/val indicator found for file {file_name}\033[0m")

    # save combined pickles
    with open(path_train_pickle, "wb") as f:
        pickle.dump(train_data, f)

    with open(path_test_pickle, "wb") as f:
        pickle.dump(test_data, f)

    with open(path_val_pickle, "wb") as f:
        pickle.dump(val_data, f)  
    print("All pickle files successfully combined into train, test and val files.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine pickle files for train, test and val sets')
    parser.add_argument('path_pickle', type=str, help='path to the folder containing all unsorted pickle files')
    parser.add_argument('path_WLASL_json', type=str, help='path to the WLASL json file specifying train/test/val splits, e.g. WLASL/WLASL_v0.3.json')

    args = parser.parse_args()

    path_pickle_files = args.path_pickle
    path_wlasl_json = args.path_WLASL_json
    
    #get the ids that belong to train, test and val sets from the WLASL json file
    train_ids, test_ids, val_ids = get_train_test_val_ids(path_wlasl_json)
    #should be: train: 14289, test: 2878, val: 3916 (old numbers)
    print("Number of train samples:", len(train_ids))
    print("Number of test samples:", len(test_ids))
    print("Number of val samples:", len(val_ids))

    #count number of pickle files
    nr_files = sum(1 for x in Path(path_pickle_files).glob('*') if x.is_file()) 
    print("Total  number of pickle files: ", nr_files)
    print("Total number of samples2: ", len(train_ids) + len(test_ids) + len(val_ids))

    if(nr_files != len(train_ids) + len(test_ids) + len(val_ids)):
        print("\033[31mError: Attention: Number of files does not match number of samples in WLASL json file!\033[0m")

    #here, the number of pickle files and the number of train/test/val samples is not the same, 
    #probably because some videos are not available anymore on youtube and have been removed from the WLASL json but not from the pickle files
    #12 (old number ) files won't be able to be moved because of missing train/test/val annotation
    combine_pickle_files(train_ids, test_ids, val_ids, path_pickle_files)


