#in pickle file gucken und schauen ob die auch train/test/val drins tehen haben
#(dann hätte preprocessing step sparen können)


#TODO: alle pickle files in ein file (pro train/test/val) kombinieren
import argparse
import pickle
import json
import numpy as np
from mmcv import load


def unpickle(path):
    try:
        data = load(path)
        print(f"Loaded with mmcv.load")
    except Exception as e1:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded with pickle.load")
        except Exception as e2:
            print(f"Error with mmcv: {e1}")
            print(f"Error with pickle: {e2}")
            raise
    
    #dont let numpy shorten the output with "..." but print the whole array
    np.set_printoptions(threshold=np.inf, linewidth=200)
    
    output_path = path.replace('.pkl', '.txt')
    with open(output_path, "w") as f:
        f.write(str(data))
    print("Converted pickle file to txt and saved as:", output_path)

#path_to_file = 'WLASL_julia/12320_GL00003.pkl'
#path_to_file = '../data/WLASL/WLASL/wlasl_poses_pickle/val/00295.pkl'


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Unpickle a pickle file and save it as a json')
  parser.add_argument('pickle_file', type=str, help='path and name of pickle file')
  args = parser.parse_args()

  unpickle(args.pickle_file)






#TODO: kann ggf auch als text datei statt als json speichern
######für pickle files runtergeladen von OpenHands docu für WLASL
#path_to_file = '../data/WLASL/WLASL/wlasl_poses_pickle/val/00295.pkl'

#Aufbau:
#{'keypoints': array1, 'confidences': array2},
#keypoints array: (num_frames, 543, 3) = (num_frames, num_keypoints, 3)
    #for each frame, each keypoint has x,y and z koordinates
    #x and y are the (normalized) coordinates of the keypoint within the image frame (x, y direction)
    #z is the depth coordinate (distance from camera) that is estimated by mediapipe
#confidence shape: (num_frames, 543), (num_frames, num_keypoints)
#mediapipe hat 33 + 468 + 21 + 21 = 543 keypoints
#wir brauchen nur die für die hand (21 für linke und 21 für rechte hand)