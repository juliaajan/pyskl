
import multiprocessing as mp
import os
import subprocess
from pathlib import Path

import cv2

#from pyskl.smp import *

input_folder = '../WLASL300/WLASL_300'
output_folder = '../WLASL300/WLASL_300_compressed'

def get_shape(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def collect_videos(root):
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.mp4'):
                videos.append(os.path.join(dirpath, filename))
    return videos

#adopted from configs/rgbpose_con3d/compress:nturgbd.py
def compress(src, dest, shape, target_size=540, fps=-1):
    w, h = shape
    #scale video to height=targetsize, width= 2*targetsize for videos in landscape mode (Querformat), the other way around for portrait format (Hochformat)
    scale_str = f'-vf scale=-2:{target_size}' if w >= h else f'-vf scale={target_size}:-2'
    fps_str = f'-r {fps}' if fps > 0 else '' #keep frame rate from original video, if present(-1) 
    quality_str = '-q:v 1' #highest quality
    vcodec_str = '-c:v mpeg4' # '-c:v libx264' change encoder to mpeg4 as libx264 is not available
    cmd = f'ffmpeg -y -loglevel error -i {src} -threads 1 {quality_str} {scale_str} {fps_str} {vcodec_str} {dest}'
    os.system(cmd)

    print("Compressed video saved to: ", dest)


def compress_wlasl300(file):
    src = file
    shape = get_shape(src) #tuple

    rel_path = os.path.relpath(src, input_folder) #eg train/12345.mp4
    #add train, test, val folders in WLALS_300_compressed
    dest = os.path.join(output_folder, rel_path) 
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    compress(src, dest, shape)

    print("Source: ", src)
    print("Destination: ", dest)
    print("Shape: ", shape)



if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    #collect all mp4 videos from train, test and val folders
    files = collect_videos(input_folder) #TODO: Path?

    #processes each video individually
    #pool = mp.Pool(1)
    #pool.map(compress_wlasl300, files)
    for file in files:
        compress_wlasl300(file)

    #python "julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/compress_wlasl300.py"


