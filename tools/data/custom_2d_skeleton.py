# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import decord
import mmcv
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm

import pyskl  # noqa: F401
from pyskl.smp import mrlines

try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose  # noqa: F401
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

pyskl_root = osp.dirname(pyskl.__path__[0])
default_det_config = f'{pyskl_root}/demo/faster_rcnn_r50_fpn_1x_coco-person.py'
default_det_ckpt = (
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
default_pose_config = f'{pyskl_root}/demo/hrnet_w32_coco_256x192.py'
default_pose_ckpt = (
    'https://download.openmmlab.com/mmpose/top_down/hrnet/'
    'hrnet_w32_coco_256x192-c78dce93_20200708.pth')


#öffnet die Videodatei , iteriert über alle frames im video, konvertiert jedes frame von decord-format in numpy-array
#shape pro frame: (h, w, 3) (RGB Bild)
def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


#TODO: kann ich einstellen dass ich nur eine person pro frame erkennen will?
#TODO: was wenn hier keine person erkannt wird?
#TODO: kann ich coco-person benutzen hierfür? (Glaube ja, weil er das ja nur nutzt um personen zu erkennen, noch nicht um deren pose keypoints zu extrahieren)
##mmdetection: detectiert Objekte (hier: person) im frame
#nutzt faster r-cnn (trainiert auf coco-person klasse)
#gibt bounding boxes zurück [x1, y1, x2, y2, confidence_score]
#results.append(result): Sammelt alle Detections pro Frame
# Beispiel für Frame 0:
# det_results[0] = array([[x1, y1, x2, y2, 0.95],  # Person 1, 95% Konfidenz
#                         [x1, y1, x2, y2, 0.82]]) # Person 2, 82% Konfidenz
def detection_inference(model, frames):
    results = []
    for frame in frames:
        result = inference_detector(model, frame)
        results.append(result)
    return results #diese results gehen dann in pose_inference rein


#extrahiert 
def pose_inference(anno_in, model, frames, det_results, compress=False):
    #Vorbereitung:
    anno = cp.deepcopy(anno_in)
    assert len(frames) == len(det_results) 
    num_person = max([len(x) for x in det_results]) # # Max. Personen in irgendeinem Frame
    total_frames = len(frames)
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = num_person

    #brauche nicht, compress ist standardmäßig auf false, ist dazu da nur die tatsächlich erkannten keypoints zu speichern (und nicht für die nicht-erkannten Nullen zu speichern)
    if compress:
        kp, frame_inds = [], []
        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
            for j, item in enumerate(pose):
                kp.append(item['keypoints']) # Nur tatsächliche Detections speichern
                frame_inds.append(i) # Merke Frame-Index
        anno['keypoint'] = np.stack(kp).astype(np.float16) # (N, 17, 3), N = Summe aller Detections
        anno['frame_inds'] = np.array(frame_inds, dtype=np.int16)  # (N,)
    
    #standardfall: compress = False
    else:
        # Erstelle festes Array für alle Personen × Frames × 17 Keypoints × 3 (x, y, confidence_score)
        #für jede person (von num_person) in jedem frame (von total_frames) speichere deren 17 keypoints jewils mit x,y,confidence_score
        #TODO: ich kann hier ggf die Person-Dimension weglassen (weil nur eine Person pro Video)
        #TODO: ich habe hier nicht 17 keypoints x 3 (x,y,confidence) sondern 75 keypoints x 3 (oder x4 mit confidence?) (x,y,z)
        kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)
        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Konvertiere Bounding Boxes zu MMPose-Format
            # Align input format
            d = [dict(bbox=x) for x in list(d)] ## [x1,y1,x2,y2,score] → {'bbox': [x1,y1,x2,y2,score]}

            # Pose-Estimation für alle Personen in diesem Frame
            # Input:  frame = numpy array (H,W,3)
            # bboxes = [{'bbox': [x1,y1,x2,y2,score]}, ...]
            # Output: [{'keypoints': array(17,3)}, ...]  # 17 COCO-Keypoints mit (x,y,conf)
            #TODO: hier stattdessen mediapipe detection machen
            pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
            # # pose = [{'keypoints': array(17,3)}, {'keypoints': array(17,3)}, ...]
            for j, item in enumerate(pose):
                kp[j, i] = item['keypoints']
        
        ## Trenne x,y und confidence
        anno['keypoint'] = kp[..., :2].astype(np.float16)
        #TODO: laut pyskl/tools/data/README.md sind keypoint_scores nur für 2D skeletons required, aber ich habe doch 3D skeletons oder nicht? Oder habe ich nur 3D keypoints?
        anno['keypoint_score'] = kp[..., 2].astype(np.float16)

        #Resultat:
        #anno = {
            #'keypoint': np.array (num_person, total_frames, 17, 2),  # x,y
            #'keypoint_score': np.array (num_person, total_frames, 17)  # confidence
        #}
    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    # parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    # parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local_rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    parser.add_argument('--compress', action='store_true', help='whether to do K400-style compression')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def main():
    #holt sich die argumente aus terminal
    args = parse_args()
    assert args.out.endswith('.pkl')

    lines = mrlines(args.video_list) #liest die videoliste ein in der "videopath label" steht
    lines = [x.split() for x in lines] # Teilt jede Zeile in [pfad, label]

    # * We set 'frame_dir' as the base name (w/o. suffix) of each video
    assert len(lines[0]) in [1, 2]
    if len(lines[0]) == 1:  # Nur Pfad -> brauch ich nciht, meine video_list hat Pfad + Label
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines]
    else:  # Pfad + Label
        #stellt sicher dass label ein int ist
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]

    #prüfen ob ein GPU oder mehrere GPUs, ggf videos gleichmäßig auf GPUs verteilen
    #brauche nicht unbedingt, weil ich nur eine GPU habe
    if args.non_dist:
        my_part = annos # Alle Videos auf 1 GPU
        os.makedirs(args.tmpdir, exist_ok=True)
    else:
        init_dist('pytorch', backend='nccl')  # Multi-GPU initialisieren
        rank, world_size = get_dist_info()
        if rank == 0:
            os.makedirs(args.tmpdir, exist_ok=True)
        dist.barrier()
        my_part = annos[rank::world_size] # Jede GPU bekommt Teil der Videos

    #Modelle laden für Personenerkennung und keypoint-pose ectraction
    #TODO: gucken ob es mediapipe hier schon gibt, sonst muss hinzufügen
    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda') # Faster R-CNN (Person Detection)
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    #TODO: hier stattdessen mediapipe detection machen
    #checkpoint file ist für pre-trained weights, wenn keins angebe mache ich kein pretraining
    #TODO: prüfen mit und ohne pretraining, weil das pose ist kann wahrsch. nicht coco pretraining nutzen
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda') # HRNet (Pose Estimation)

    #main part: pose extraction und keypoint extraction durchführen
    results = []
    #für jedes video:
    for anno in tqdm(my_part):
        #frames extrahieren
        frames = extract_frame(anno['filename'])
        #person detection durchführen
        det_results = detection_inference(det_model, frames)
        # * Get detection results for human (# Nur COCO-Klasse 0 (Person))
        det_results = [x[0] for x in det_results]
        #Bounding boxes filtern (zu kleine boxes oder zu kleine scores rausfiltern)
        #TODO: muss ggf noch werte anpassen
        for i, res in enumerate(det_results):
            # * filter boxes with small scores
            res = res[res[:, 4] >= args.det_score_thr]
            # * filter boxes with small areas
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            det_results[i] = res

        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        #pose-keypoints extracton
        anno = pose_inference(anno, pose_model, frames, det_results, compress=args.compress)
        #entferne Pfad (weil nicht mehr nötig)
        anno.pop('filename')
        results.append(anno)

    #speichern (je nachdmem ob single oder multi-GPU)
    if args.non_dist:
        mmcv.dump(results, args.out) # Speichert direkt wlasl300_annos.pkl
    else:
        # Multi-GPU: Jede GPU speichert Teil
        mmcv.dump(results, osp.join(args.tmpdir, f'part_{rank}.pkl'))
        dist.barrier()

        if rank == 0:
            parts = [mmcv.load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
            rem = len(annos) % world_size
            if rem:
                for i in range(rem, world_size):
                    parts[i].append(None)

            ordered_results = []
            for res in zip(*parts):
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:len(annos)]
            mmcv.dump(ordered_results, args.out)


# wlasl300_annos.pkl enthält Liste von:
""" [
  {
    'frame_dir': '12320',
    'label': 'book',
    'img_shape': (1080, 1920),
    'total_frames': 120,
    'num_person_raw': 1,
    'keypoint': array (1, 120, 17, 2),
    'keypoint_score': array (1, 120, 17)
  },
  {...}, ...
] """

if __name__ == '__main__':
    main()
