import mmcv
try:
    from mmcv import load #for remote labpc
except (ImportError, ModuleNotFoundError):
    try:
        from mmengine.fileio import load #for windows
    except (ImportError, ModuleNotFoundError) as e:
        raise ModuleNotFoundError(f"Weder mmcv.load noch mmengine.fileio.load konnten importiert werden. Fehlermeldung: {e}" )

import argparse
import pickle
import numpy as np


def unpickle_specific_id(path, video_id):
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

    if not isinstance(data, dict):
        raise ValueError("Expected pickle content to be a dict with keys 'split' and 'annotations'.")

    #get the annotations list from the dict
    annotations = data.get("annotations")
    if annotations is None:
        raise KeyError("Key 'annotations' not found in pickle file.")
    if not isinstance(annotations, list):
        raise ValueError("Expected 'annotations' to be a list.")

    #find the nanotation with the frame_dir that matches the provided video_id
    matching = [anno for anno in annotations if str(anno.get("frame_dir")) == str(video_id)]

    if len(matching) == 0:
        raise ValueError(f"No annotation found for video_id='{video_id}'.")
    if len(matching) > 1:
        raise ValueError(
            f"Found {len(matching)} annotations for video_id='{video_id}'. Expected exactly one."
        )

    selected_annotation = matching[0]

    #dont let numpy shorten the output with "..." but print the whole array
    #only use this line for short files, when using annotations for a single video!
    np.set_printoptions(threshold=np.inf, linewidth=200)

    output_path = path.replace('.pkl', '') + "_" + video_id + ".txt"
    with open(output_path, "w") as f:
        f.write(str(selected_annotation))
    print("Saved annotations of video {video_id} to:", output_path)



#path_to_file = 'WLASL_julia/12320_GL00003.pkl'
#path_to_file = '../data/WLASL/WLASL/wlasl_poses_pickle/val/00295.pkl'


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Unpickle a pickle file and save the annotations of a specific video as a txt')
  parser.add_argument('pickle_file', type=str, help='path and name of pickle file')
  parser.add_argument('video_id', type=str, help='ID of the video to extract annotations for')

  args = parser.parse_args()

  unpickle_specific_id(args.pickle_file, args.video_id)