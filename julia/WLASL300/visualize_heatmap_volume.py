import os
import cv2
import os.path as osp
import decord
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import urllib
import moviepy.editor as mpy
import random as rd
from pyskl.smp import *
from mmpose.apis import vis_pose_result
from mmpose.models import TopDown
from mmcv import load, dump
import matplotlib.cm as cm
from pyskl.datasets.pipelines import Compose
#original source: pyskl/demo/visualize_heatmap_volume.ipynb



#set fonts 
FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.6
FONTCOLOR = (255, 255, 255)
BGBLUE = (0, 119, 182)
THICKNESS = 1
LINETYPE = 1


def add_label(frame, label, BGCOLOR=BGBLUE):
    threshold = 30
    def split_label(label):
        label = label.split()
        lines, cline = [], ''
        for word in label:
            if len(cline) + len(word) < threshold:
                cline = cline + ' ' + word
            else:
                lines.append(cline)
                cline = word
        if cline != '':
            lines += [cline]
        return lines
    
    if len(label) > 30:
        label = split_label(label)
    else:
        label = [label]
    label = ['Action: '] + label
    
    sizes = []
    for line in label:
        sizes.append(cv2.getTextSize(line, FONTFACE, FONTSCALE, THICKNESS)[0])
    box_width = max([x[0] for x in sizes]) + 10
    text_height = sizes[0][1]
    box_height = len(sizes) * (text_height + 6)
    
    cv2.rectangle(frame, (0, 0), (box_width, box_height), BGCOLOR, -1)
    for i, line in enumerate(label):
        location = (5, (text_height + 6) * i + text_height + 3)
        cv2.putText(frame, line, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    return frame
    

def vis_skeleton(vid_path, anno, category_name=None, ratio=0.5):
    vid = decord.VideoReader(vid_path)
    frames = [x.asnumpy() for x in vid]
    
    h, w, _ = frames[0].shape
    new_shape = (int(w * ratio), int(h * ratio))
    frames = [cv2.resize(f, new_shape) for f in frames]
    
    assert len(frames) == anno['total_frames']
    #TODO: stimmt die shape hier?
    # The shape is N x T x K x 3
    kps = np.concatenate([anno['keypoint'], anno['keypoint_score'][..., None]], axis=-1)
    kps[..., :2] *= ratio
    # Convert to T x N x K x 3
    kps = kps.transpose([1, 0, 2, 3])
    vis_frames = []

    # we need an instance of TopDown model, so build a minimal one
    model = TopDown(backbone=dict(type='ShuffleNetV1'))

    for f, kp in zip(frames, kps):
        bbox = np.zeros([0, 4], dtype=np.float32)
        result = [dict(keypoints=k) for k in kp]
        
        vis_frame = vis_pose_result(model, f, result)
        
        if category_name is not None:
            vis_frame = add_label(vis_frame, category_name)
        
        vis_frames.append(vis_frame)
    return vis_frames




if __name__ == '__main__':
    # We assume the annotation is already prepared
    #ann_file = '../data/gym/gym_hrnet.pkl' #TODO
    #output_dir = ''#TODO
    #category_mappings = #TODO
    #video_file = 'C:/path/to/your/wlasl/videos/1234.mp4'  # TODO: ANPASSEN!


    #get parser arguments
    parser = argparse.ArgumentParser(
        description='Visualise heatmaps')
    parser.add_argument('ann_file', type=str, help='path to the pickle file, eg wlasl300/annos.pkl')
    parser.add_argument('category_mappings', type=str, help='path to the mappings of gloss indices to gloss names, eg wlasl300/label_mappings.txt')
    parser.add_argument('video_file', type=str, help='path to video that should be used for visualization, eg WLASL/start_kit/videos/1234.mp4')
    parser.add_argument('output_dir', type=str, help='path to where the vsualization videos should be saved')
    args = parser.parse_args()

    ann_file = args.ann_file
    output_dir = args.output_dir
    category_mappings = args.category_mappings
    video_file = args.video_file




    #TODO prüfen
    keypoint_pipeline = [
        dict(type='PoseDecode'),
        dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
        dict(type='Resize', scale=(-1, 64)),
        dict(type='CenterCrop', crop_size=64),
        dict(type='GeneratePoseTarget', with_kp=True, with_limb=False)
    ]

    #TODO prüfen
    limb_pipeline = [
        dict(type='PoseDecode'),
        dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
        dict(type='Resize', scale=(-1, 64)),
        dict(type='CenterCrop', crop_size=64),
        dict(type='GeneratePoseTarget', with_kp=False, with_limb=True)
    ]


    from pyskl.datasets.pipelines import Compose
    def get_pseudo_heatmap(anno, flag='keypoint'):
        assert flag in ['keypoint', 'limb']
        pipeline = Compose(keypoint_pipeline if flag == 'keypoint' else limb_pipeline)
        return pipeline(anno)['imgs']

    def vis_heatmaps(heatmaps, channel=-1, ratio=8):
        # if channel is -1, draw all keypoints / limbs on the same map
        import matplotlib.cm as cm
        heatmaps = [x.transpose(1, 2, 0) for x in heatmaps]
        h, w, _ = heatmaps[0].shape
        newh, neww = int(h * ratio), int(w * ratio)
        
        if channel == -1:
            heatmaps = [np.max(x, axis=-1) for x in heatmaps]
        cmap = cm.viridis
        heatmaps = [(cmap(x)[..., :3] * 255).astype(np.uint8) for x in heatmaps]
        heatmaps = [cv2.resize(x, (neww, newh)) for x in heatmaps]
        return heatmaps



    # Load mapping
    lines = mrlines(category_mappings)
    label_data = [x.strip().split('\t') for x in lines]
    wlasl_categories = [label_name for idx, label_name in sorted(label_data, key=lambda x: int(x[0]))]
    wlasl_annos = load(ann_file)['annotations']


    #get the video
    video_file_basename= osp.basename(video_file) #only filename and ending, e.g. 1234.mp4
    frame_dir = osp.splitext(video_file_basename)[0] #get filename without .mp4 ending
    #vid_path = osp.join(video_file, vid) use video_file directly
    anno_matches = [x for x in wlasl_annos if x['frame_dir'] == frame_dir]
    if len(anno_matches) == 0:
        print("No annotation found for video:", video_file, " with frame_dir:", frame_dir)
        exit(1)

    anno = anno_matches[0]
    print("Found annotation for video:", video_file)

    # Visualize Skeleton
    print(f"Creating skeleton visualization for {anno['frame_dir']}...")
    vis_frames = vis_skeleton(video_file, cp.deepcopy(anno), wlasl_categories[anno['label']])
    vid = mpy.ImageSequenceClip(vis_frames, fps=24)
    #save as video
    skeleton_output = osp.join(output_dir, f'{anno["frame_dir"]}_skeleton.mp4')
    vid.write_videofile(skeleton_output, codec='libx264', audio=False, logger=None)
    print(f"Saved skeleton video: {skeleton_output}")


    #create heatmap
    keypoint_heatmap = get_pseudo_heatmap(cp.deepcopy(anno))
    keypoint_mapvis = vis_heatmaps(keypoint_heatmap)
    keypoint_mapvis = [add_label(f, wlasl_categories[anno['label']]) for f in keypoint_mapvis]
    vid = mpy.ImageSequenceClip(keypoint_mapvis, fps=24)
    #save as video
    heatmap_output = osp.join(output_dir, f'{anno["frame_dir"]}_keypoint_heatmap.mp4')
    vid.write_videofile(heatmap_output, codec='libx264', audio=False, logger=None)
    print(f"Saved heatmap video: {heatmap_output}")

    #limbs
    limb_heatmap = get_pseudo_heatmap(cp.deepcopy(anno), 'limb')
    limb_mapvis = vis_heatmaps(limb_heatmap)
    limb_mapvis = [add_label(f, wlasl_categories[anno['label']]) for f in limb_mapvis]
    vid = mpy.ImageSequenceClip(limb_mapvis, fps=24)
    #save as video
    limb_output = osp.join(output_dir, f'{anno["frame_dir"]}_limb_heatmap.mp4')
    vid.write_videofile(limb_output, codec='libx264', audio=False, logger=None)
    print(f"Saved limb heatmap video: {limb_output}")