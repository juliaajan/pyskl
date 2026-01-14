#in pickle file gucken und schauen ob die auch train/test/val drins tehen haben
#(dann hätte preprocessing step sparen können)


#TODO: alle pickle files in ein file (pro train/test/val) kombinieren
import argparse
import pickle
import json
import numpy as np


def _convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj



#shape=(261, 75))} #09500
#shape=(21, 75))} #/val/00295s
def unpickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    data_conv = _convert(data)
    with open(path.replace('.pkl', '.json'), "w") as jf:
        json.dump(data_conv, jf, ensure_ascii=False, indent=2)
    print("Converted pickle file to json and saved as:", path.replace('.pkl', '.json'))

    #print("file:", path)

    #print(type(data))
    #print(data) 

#path_to_file = 'WLASL_julia/12320_GL00003.pkl'
#path_to_file = '../data/WLASL/WLASL/wlasl_poses_pickle/val/00295.pkl'
#unpickle(path_to_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unpickle a pickle file and save it as a json')
    parser.add_argument('pickle_file', type=str, help='path and name of pickle file')
    args = parser.parse_args()

    unpickle(args.pickle_file)



