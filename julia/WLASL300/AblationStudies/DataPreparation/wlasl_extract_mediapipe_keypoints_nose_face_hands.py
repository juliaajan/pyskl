
import argparse
from mmcv import load, dump
from pyskl.smp import mrlines, mwlines
import cv2
import os, gc
import numpy as np
import mediapipe as mp
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
from mmcv import load, dump
from tqdm import tqdm



#oriented on pyskl/tools/data/custom_2d_skeleton.py
#and Ai4Bharat/OpenHands/scripts/mediapipe_extract.py



# ============= OpenHands-style Keypoint Extraction ============= #source: Ai4Bharat/OpenHands/scripts/mediapipe_extract.py

mp_holistic = mp.solutions.holistic

FACE_LANDMARK_INDICES = [46, 52, 53, 65, 295, 283, 282, 276, 7, 159, 155, 145, 382, 386, 249, 374, 324, 13, 78, 14] #TODO
N_NOSE_LANDMARKS = 1 # only keep the nose (index 0)
N_UPPER_BODY_LANDMARKS = 25
N_HAND_LANDMARKS = 21 #this number must be doubles for both hands

#Counter: Zählt verarbeitete Videos über mehrere parallele Prozesse hinweg.
#RawValue: Shared Memory (alle Prozesse sehen denselben Wert).
#Lock: Verhindert, dass zwei Prozesse gleichzeitig increment() aufrufen (Race Condition).
class Counter(object):
    # https://stackoverflow.com/a/47562583/
    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue("i", initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        return self.val.value


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
        #optional: resize frame
        # img = cv2.resize(img, (640, 480))
        frames.append(img)

    vidcap.release()
    # cv2.destroyAllWindows()
    return np.asarray(frames)


#holt ALLE keypoints für ein video (body, face, hands) indem Methoden von oben dafür aufruft
def get_holistic_keypoints(frames):
    """
    For videos, it's optimal to create with `static_image_mode=False` for each video.
    https://google.github.io/mediapipe/solutions/holistic.html#static_image_mode
    """

    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)

    keypoints = []
    confs = []

    img_shape = frames[0].shape[:2]

    #für jeden frame
    for frame in frames:
        results = holistic.process(frame)

        nose_data, nose_conf = process_body_landmarks(results.pose_landmarks, N_NOSE_LANDMARKS, img_shape)
        face_data, face_conf = process_face_landmarks(results.face_landmarks, FACE_LANDMARK_INDICES, img_shape)
        lh_data, lh_conf = process_hand_landmarks(results.left_hand_landmarks, N_HAND_LANDMARKS, img_shape)
        rh_data, rh_conf = process_hand_landmarks(results.right_hand_landmarks, N_HAND_LANDMARKS, img_shape)

        #concatenate all generated keypoints and their scores
        data = np.concatenate([nose_data, face_data, lh_data, rh_data])
        # data.shape = (K, 2)
        conf = np.concatenate([nose_conf, face_conf, lh_conf, rh_conf])
        # conf.shape = (K,)

        keypoints.append(data)
        confs.append(conf)

    # TODO: Reuse the same object when this issue is fixed: https://github.com/google/mediapipe/issues/2152
    holistic.close()  #close MediaPipe model
    del holistic #delete object
    gc.collect()  #force garbage colllector

    #konvertiert von Liste zu np Array (hier sind kp und confs für ALLE frames)
    keypoints = np.stack(keypoints) # (T, K, 2) — T Frames, K Keypoints, 2 Koordinaten (x,y)
    confs = np.stack(confs) # (T, K) — Confidence pro Keypoint
    return keypoints, confs


#extrahiere BODY keypoints (x,y) und confidence score aus mediapipe body/pose landmarks
def process_body_landmarks(component, n_points, img_shape):
    #initialisiere leere arrays (falls keine landmarks erkannt werden)
    kps = np.zeros((n_points, 2))  # (K, 2) für Pose
    conf = np.zeros(n_points) # (K,) für Confidence

    #wenn landmark vorhanden
    if component is not None:
        landmarks = component.landmark  #MediaPipe LandmarkList
        #Extrahiere x und y für jeden Punkt, lasse z aus - nur die ersten n_points (z.B. 25)
        kps = np.array([[p.x, p.y] for p in landmarks])[:n_points]
        #Denormalisiere x und y anhand ihrer image shape, sodass sie nicht mehr im Bereich (0,1) sonder in Pixelwerten im Bereich (W, H) vorliegen
        h, w = img_shape
        kps[:, 0] *= w  #x
        kps[:, 1] *= h  #y
        #Extrahiere Visibility (Konfidenz) für Body-Keypoints - nur die ersten n_points
        conf = np.array([p.visibility for p in landmarks])[:n_points]

    return kps, conf

#extrahiere LEFT HAND, RIGHT HAND keypoints (x,y) und confidence score
def process_hand_landmarks(component, n_points, img_shape):
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

#extrahiere FACE keypoints (x,y) und confidence score
def process_face_landmarks(component, landmark_indices, img_shape):
    n_points = len(landmark_indices)
    kps = np.zeros((n_points, 2))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        # Extrahiere nur die spezifizierten Keypoints anhand der Indices
        kps = np.array([[landmarks[i].x, landmarks[i].y] for i in landmark_indices])
        
        #Denormalisiere x und y anhand ihrer image shape, sodass sie nicht mehr im Bereich (0,1) sonder in Pixelwerten im Bereich (W, H) vorliegen
        h, w = img_shape
        kps[:, 0] *= w  #x
        kps[:, 1] *= h  #y

        #setze confidence für alle Punkte azf 1
        conf = np.ones(n_points)
    return kps, conf




def process_single_video(anno):
    print("Start processing video: ", anno['filename'])
    try:
        frames = load_frames_from_video(anno['filename'])
        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        anno['original_shape'] = shape
        
        anno = mediapipe_inference(anno, frames)
        return anno
    except Exception as e:
        print(f"ERROR processing {anno['filename']}: {e}")
        return None


def mediapipe_inference(anno_in, frames):
    """Extract MediaPipe keypoints and format as custom_2d_skeleton annotation."""
    import copy as cp
    anno = cp.deepcopy(anno_in)
    
    total_frames = len(frames)
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = 1 #assume one person per video in SLR
    
    # Extract keypoints
    pose_kps, pose_confs = get_holistic_keypoints(frames)  # (T, 67, 2), (T, 67)
    
    keypoints = pose_kps
    confidences = pose_confs
    
    #add person dimension
    keypoints = keypoints[np.newaxis, ...]  # (1, T, num_keypoints, 2)
    
    # Store in annotation
    anno['keypoint'] = keypoints.astype(np.float16)
    anno['keypoint_score'] = confidences[np.newaxis, ...].astype(np.float16)  # (1, T, num_keypoints)

    print("Successfully extracted MediaPipe keypoints for video:", anno_in['filename'], "mean confidence for this video: ", np.mean(confidences))
    
    return anno



def create_label_mapping(lines):
    #get all unique labels from the video list (lines contains path + label)
    unique_labels = sorted(set(x[1] for x in lines))
    
    #map each lapelname to an int
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    #save as txt file
    label_mapping_file = os.path.join(args.output_path, "label_mapping.txt")
    with open(label_mapping_file, 'w') as f:
        for label, idx in label_to_int.items():
            f.write(f"{idx}\t{label}\n")
    
    print(f"Found {len(unique_labels)} unique labels")
    print(f"Saved label mapping to: {label_mapping_file}")
    
    return label_to_int



##################main#############
def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract keypoints from videos using MediaPipe and create annotation-pickle files in format suitable for pyskl training')
    #path to the video lsit created by wlasl_prepare_mediapipe_extraction.py
    parser.add_argument('videolist', type=str, help='the path to the list of source videos that contains lines with the video paths and their corresponding labels, e.g. my-folder/wlasl300.list')
    parser.add_argument('output_path', type=str, help='the path where the output pickle file will be saved')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    #TODO: Achtung: das läuft auf CPU, ggf wie in custom_2d_skeleton.py auf GPU umstellen
    #anzahl cpu-cores ermitteln
    #Alternativ: als argument übergeben
    n_cores = multiprocessing.cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing")

    #load video list
    assert args.videolist.endswith('.list')
    lines = mrlines(args.videolist) #liest die videoliste ein in der "videopath label" steht
    lines = [x.split() for x in lines] # Teilt jede Zeile in [pfad, label]

    #make sure that the video list is not empty
    if len(lines) == 0:
        raise ValueError(f"Video list is empty: {args.videolist}")

    #make sure that each line has 2 entries: path and label
    for i, line in enumerate(lines):
        if len(line) != 2:
            raise ValueError(f"Line {i} must have 2 elements (path + label), got {len(line)}: {line}")

    #convert the label from string to int
    label_to_int = create_label_mapping(lines)
    annos = [dict(frame_dir=os.path.basename(x[0]).split('.')[0], filename=x[0], label=label_to_int[x[1]]) for x in lines]

    print(f"Start processing {len(annos)} videos")

    # Parallel processing with joblib
    results = Parallel(n_jobs=n_cores, backend="loky")(delayed(process_single_video)(anno)
        for anno in tqdm(annos, desc="Extracting keypoints"))

    #filter out failed videos
    #TODO: prüfen ob so richtig
    results = [r for r in results if r is not None]
    print(f"Successfully processed {len(results)}/{len(annos)} videos")

    #create the split dict
    split_dict = {}
    for anno in results:
        #extract split from filename path
        filename = anno['filename']
        path_parts = filename.replace('\\', '/').split('/')
        
        #find train/test/path in folder structure (only works because WLASL300 splits videos in train/test/val folders)
        split_name = None
        for part in path_parts:
            if part in ['train', 'test', 'val']:
                split_name = part
                break
        if split_name is None:
            raise ValueError(f"Could not determine split for {filename}, no train/test/val folder found in path")
        #append the video_id (frame_dir) to the corresponding split list
        if split_name not in split_dict:
            split_dict[split_name] = []
        split_dict[split_name].append(anno['frame_dir'])

        #remove the filename field from the anno file
        anno.pop('filename')

        #final annotation file with split and annotations
        output_dict = {
            'split': split_dict,
            'annotations': results
        }


    #save results
    output_file= os.path.join(args.output_path, "pyskl_mediapipe_annos_2d_denormalized_NOSE_FACE_HANDS.pkl")
    dump(output_dict, output_file)
    print(f"Saved annotations to: {output_file}")
    print(f"Split distribution: {[(k, len(v)) for k, v in split_dict.items()]}")



#############################------
    #TODO: ab hier alles weiter, parallele verarbeitung, speichern etc
    #Pfade definieren
    #DIR = "AUTSL/train/" #input pfad
    #SAVE_DIR = "AUTSL/holistic_poses/" #output: keypoints

    #os.makedirs(SAVE_DIR, exist_ok=True)

    #alle video pfade sammeln
    #file_paths = [] #TODO: kann auch über anno file machen
    #save_paths = []
   # for file in os.listdir(DIR):
   #     if "color" in file: # Nur Videos mit "color" im Namen TODO: ändern
   #         file_paths.append(os.path.join(DIR, file)) ## z.B. "AUTSL/train/video_001_color.mp4"
   #         save_paths.append(os.path.join(SAVE_DIR, file.replace(".mp4", ""))) # z.B. "AUTSL/holistic_poses/video_001_color"

    #Parallele Verarbeitung mit joblib, führe für jedes video aus
   # Parallel(n_jobs=n_cores, backend="loky")(
    #    delayed(gen_keypoints_for_video)(path, save_path)
    #    for path, save_path in tqdm(zip(file_paths, save_paths))
   # )
