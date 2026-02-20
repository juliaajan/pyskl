
import argparse
from mmcv import load, dump
import cv2
import os, gc
import numpy as np
import mediapipe as mp


mp_holistic = mp.solutions.holistic
N_HAND_LANDMARKS = 21

#lädt für ein video alle frames
def load_frames_from_video(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        success, img = vidcap.read()
        if not success:
            break
        #OpenCV opens in BGR, but mediapipe expects RGB, so convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    vidcap.release()
    # cv2.destroyAllWindows()
    return np.asarray(frames)


def get_holistic_keypoints(frames):
    """
    For videos, it's optimal to create with `static_image_mode=False` for each video.
    https://google.github.io/mediapipe."""
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)
    
    keypoints = []
    confs = []
    img_shape = frames[0].shape[:2]

    #für jeden frame
    for frame in frames:
        results = holistic.process(frame)
        
        lh_data, lh_conf = process_other_landmarks(results.left_hand_landmarks, N_HAND_LANDMARKS, img_shape)
        rh_data, rh_conf = process_other_landmarks(results.right_hand_landmarks, N_HAND_LANDMARKS, img_shape)



        #führe alle generierten keypoints und conf scores zusammen
        data = np.concatenate([lh_data, rh_data])
        # data.shape = (42, 2) = 21+21 with (x, y) each
        conf = np.concatenate([lh_conf, rh_conf])
        # conf.shape = (42,)

        keypoints.append(data)
        confs.append(conf)

    # TODO: Reuse the same object when this issue is fixed: https://github.com/google/mediapipe/issues/2152
    holistic.close()  # Schließe MediaPipe-Modell
    del holistic # Lösche Objekt
    gc.collect()  # Force Garbage Collection

    #konvertiert von Liste zu np Array (hier sind kp und confs für ALLE frames)
    keypoints = np.stack(keypoints) # (T, 42, 2) — T Frames, 42 Keypoints, 2 Koordinaten (x,y)
    confs = np.stack(confs) # (T, 42) — Confidence pro Keypoint
    return keypoints, confs


def process_other_landmarks(component, n_points, img_shape):
    kps = np.zeros((n_points, 2))
    conf = np.zeros(n_points)
    
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y] for p in landmarks])
        #Denormalisiere x und y anhand ihrer image shape, sodass sie nicht mehr im Bereich (0,1) sonder in Pixelwerten im Bereich (W, H) vorliegen
        h, w = img_shape
        kps[:, 0] *= w  #x
        kps[:, 1] *= h  #y
        #setze confidence für alle Punkte azf 1
        #TODO: stimmt es, dass mediapipe keine confidence scores für face/hand landmarks liefert?
        conf = np.ones(n_points)
    return kps, conf


def mediapipe_inference(video_path, label=0):
    """Extrahiert MediaPipe-Keypoints für ein einzelnes Video."""
    print(f"Start processing video: {video_path}")
    
    # Frames laden
    frames = load_frames_from_video(video_path)
    shape = frames[0].shape[:2]
    
    # Keypoints extrahieren
    pose_kps, pose_confs = get_holistic_keypoints(frames)  # (T, 42, 2), (T, 42)
    
    # Füge Person-Dimension hinzu: (1, T, 42, 2)
    keypoints = pose_kps[np.newaxis, ...]
    confidences = pose_confs[np.newaxis, ...]
    
    # Annotation erstellen
    anno = {
        'frame_dir': os.path.basename(video_path).split('.')[0],
        'label': label,
        'img_shape': shape,
        'original_shape': shape,
        'total_frames': len(frames),
        'num_person_raw': 1,
        'keypoint': keypoints.astype(np.float16),
        'keypoint_score': confidences.astype(np.float16)
    }
    
    print(f"Successfully extracted keypoints - Mean confidence: {np.mean(pose_confs):.3f}")
    
    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract MediaPipe keypoints from a single video')
    parser.add_argument('video', type=str, help='Path to the video file')
    parser.add_argument('output', type=str, help='Path to save the output pickle file')
    parser.add_argument('--label', type=int, default=0, help='Label for the video (default: 0)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Prüfe ob Video existiert
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    
    # Extrahiere Keypoints
    anno = mediapipe_inference(args.video, label=args.label)
    
    # Speichere Ergebnis
    dump(anno, args.output)
    print(f"Saved annotation to: {args.output}")
