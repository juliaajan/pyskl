
import multiprocessing as mp
import os
import cv2
import subprocess
from pathlib import Path
import argparse
import pickle
import numpy as np


TARGET_SIZE = 540

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
def compress(src, dest, shape, fps=-1):
    w, h = shape
    #scale video to height=targetsize, width= 2*targetsize for videos in landscape mode (Querformat), the other way around for portrait format (Hochformat)
    scale_str = f'-vf scale=-2:{TARGET_SIZE}' if w >= h else f'-vf scale={TARGET_SIZE}:-2'
    fps_str = f'-fps_mode passthrough' if fps > 0 else '' #keep frame rate from original video, if present(-1) 
    quality_str = '-q:v 1' #highest quality
    vcodec_str = '-c:v mpeg4' # '-c:v libx264' change encoder to mpeg4 as libx264 is not available
    cmd = f'ffmpeg -y -loglevel error -i {src} -threads 1 {quality_str} {scale_str} {fps_str} {vcodec_str} {dest}'
    os.system(cmd)

    print("Compressed video saved to: ", dest)


def compress_wlasl300(file, input_path, output_path):
    src = file
    shape = get_shape(src) #tuple

    rel_path = os.path.relpath(src, input_path) #eg train/12345.mp4
    #add train, test, val folders in WLALS_300_compressed
    dest = os.path.join(output_path, rel_path) 
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    compress(src, dest, shape)

    print("Source: ", src)
    print("Destination: ", dest)
    print("Shape: ", shape)


def compress_failed_file_windows(video_id, input_path, output_path):
    video_file = None
    for dirpath, _, filenames in os.walk(input_path):
        for filename in filenames:
            if video_id in filename and filename.endswith('.mp4'):
                video_file = os.path.join(dirpath, filename)
                break

    shape = get_shape(video_file) #tuple

    rel_path = os.path.relpath(video_file, input_path) #eg train/12345.mp4
    #add train, test, val folders in WLALS_300_compressed
    dest = os.path.join(output_path, rel_path) 
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if video_id in dest:
        import imageio_ffmpeg
        print("Compressing video: ", video_id)
        print("Source: ", video_file)
        print("Destination: ", dest)
        print("Shape: ", shape)

        w, h = shape
        scale_str = f'scale=-2:{TARGET_SIZE}' if w >= h else f'scale={TARGET_SIZE}:-2'

        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel", "error",
            "-i", video_file,
            "-threads", "1",
            "-q:v", "1",
            "-vf", scale_str,
            "-c:v", "mpeg4",
            dest
        ]

        subprocess.run(cmd)
        print("Compressed video saved to: ", dest)


def compress_single_annotatio(anno, input_folder, output_folder):
    video_id = anno['frame_dir']
    input_video_path = None
    output_video_path = None

    for dirpath, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith('.mp4') and Path(filename).stem == video_id:
                input_video_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(input_video_path, input_folder)
                output_video_path = os.path.join(output_folder, rel_path)
                break
        if input_video_path is not None:
            break

    if input_video_path is None or output_video_path is None:
        raise ValueError(f'Could not resolve input/output video path for video id {video_id}')

    original_shape_wh = get_shape(input_video_path)
    compressed_shape_wh = get_shape(output_video_path)

    # get_shape returns (width, height), annotation stores (height, width)
    original_shape_hw = (original_shape_wh[1], original_shape_wh[0])
    compressed_shape_hw = (compressed_shape_wh[1], compressed_shape_wh[0])

    #assert that the actual, original size matches shape indicated in the annotation file
    assert tuple(anno['img_shape']) == original_shape_hw, (
        f"img_shape mismatch for {video_id}: annotation={anno['img_shape']} vs video={original_shape_hw}")

    #check that the compressed shape is correctly scaled according to the target size
    orig_w, orig_h = original_shape_wh
    comp_w, comp_h = compressed_shape_wh
    if orig_w >= orig_h:
        assert comp_h == TARGET_SIZE, (
            f"Compressed height mismatch for {video_id}: expected {TARGET_SIZE}, got {comp_h}")
    else:
        assert comp_w == TARGET_SIZE, (
            f"Compressed width mismatch for {video_id}: expected {TARGET_SIZE}, got {comp_w}")


    #finally, scale keypoints and update annotation
    scale_x = comp_w / orig_w
    scale_y = comp_h / orig_h

    if 'keypoint' in anno and anno['keypoint'] is not None:
        keypoint = np.asarray(anno['keypoint'])
        keypoint_dtype = keypoint.dtype
        keypoint = keypoint.astype(np.float32, copy=True)
        keypoint[..., 0] *= scale_x
        keypoint[..., 1] *= scale_y
        anno['keypoint'] = keypoint.astype(keypoint_dtype, copy=False)

    #update the image shape of the annotation
    anno['img_shape'] = compressed_shape_hw
    anno['original_shape'] = compressed_shape_hw

    print("Successfully compressed annotation: ", video_id)

    return anno



def parse_args():
    parser = argparse.ArgumentParser(
        description='Compress WLASL300 videos and rescale annotations to the new video size.')
    parser.add_argument('--input-video-path', default=None, help='Folder that holds the original WLASL300 videos, e.g. ../WLASL300/WLASL_300')
    parser.add_argument('--output-video-path', default=None, help='Folder to save the compressed videos')
    parser.add_argument('--ann-file', default=None, help='Annotation file holding keypoints that should be resized to new compressed video dimensions')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_video_path, exist_ok=True)
    #compress_failed_file_windows("35364", args.input_video_path, args.output_video_path) #uncomment to compress a specific video


    #collect all mp4 videos from train, test and val folders
    files = collect_videos(args.input_video_path)

    #processes each video individually
    for file in files:
        try:
            compress_wlasl300(file, args.input_video_path, args.output_video_path)
        except Exception as e:
           print(f"Error occurred while processing video: {file}: {e}")

    #python "julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/compress_wlasl300.py"
    #input_folder = '../WLASL300/WLASL_300'
    #output_folder = '../WLASL300/WLASL_300_compressed'

