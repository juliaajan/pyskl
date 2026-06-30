
import argparse
import os
import io
from mmcv import load
from mmcv.fileio import FileClient

file_client = None

def get_videoreader(filename):
    global file_client
    filename_compressed = filename.replace("WLASL_300", "WLASL_300_compressed")

    try:
        import decord
    except ImportError:
        raise ImportError(
            'Please run "pip install decord" to install Decord first.')

    if file_client is None:
        file_client = FileClient('disk')
    file_obj = io.BytesIO(file_client.get(filename_compressed))
    container = decord.VideoReader(file_obj, num_threads=1)
    try:
        return container
    except Exception as e:
        print(f"Error occurred while loading video: {filename_compressed}: {e}")
        raise ValueError(f"Error occurred while loading video: {filename_compressed}: {e}")
        
def collect_videos(root):
    uncompressed_path = os.path.join(root, "WLASL_300")
    videos = []
    for dirpath, _, filenames in os.walk(uncompressed_path):
        for filename in filenames:
            if filename.endswith('.mp4'):
                videos.append(os.path.join(dirpath, filename))
    return videos

def compare_frame_lengths(files, anno_path):
    anno_file = load(anno_path)
    annotations = anno_file['annotations']
    count_mismatch = 0
    count_severe_mismatch = 0

    for file in files:
        #extract video_id from the file path
        video_id = os.path.splitext(os.path.basename(file))[0]
    
        #get the correct frame annotation
        for a in annotations:
            if a['frame_dir'] == video_id:
                num_frames_anno = a['total_frames']
                break
        
        #get video reader as used in loading.py
        result_videoreader = get_videoreader(file)
        num_frames_compressed = len(result_videoreader)

        if num_frames_anno != num_frames_compressed:
            print(f"##Video id {video_id} --- Frames from anno: {num_frames_anno}, Frames from (compressed) video reader: {num_frames_compressed}")
            count += 1
            if abs(num_frames_anno - num_frames_compressed) > 1:
                count_severe_mismatch += 1

    print(f"Checked all videos, {count_mismatch} videos have frame length mismatches.")
    print(f"WARNING: {count_severe_mismatch} videos have severe frame length mismatches of > 1 frame.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Check length of comrpessed video files')
    parser.add_argument('video_folder_path', type=str, help='Folder to WLASL_300 and WLASL_300_compressed')
    parser.add_argument('anno_path', type=str, help='Path to annotation file')

    args = parser.parse_args()


    #collect all mp4 videos from train, test and val folders
    videos = collect_videos(args.video_folder_path) 
    compare_frame_lengths(videos, args.anno_path)



