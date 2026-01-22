
import argparse
from mmcv import load, dump
import pyskl
from pyskl.smp import mrlines, mwlines
import cv2
import os, sys, gc
import time
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

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
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




#erstellt keypoints für frames aus video
#stattdessen: process_single_video in main nutzen
#def gen_keypoints_for_video(video_path, save_path):
  #  if not os.path.isfile(video_path):
  #      print("SKIPPING MISSING FILE:", video_path)
  #      return
  #  frames = load_frames_from_video(video_path)
  #  gen_keypoints_for_frames(frames, save_path)


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



#stattdessen: mediapipe_inference
#def gen_keypoints_for_frames(frames, save_path):
#    #alle keypoints holen (body und rest)
#    pose_kps, pose_confs = get_holistic_keypoints(frames) # pose_kps.shape = (T, 543, 3) — alle Keypoints (Body + Face + Hands)
#    
#    #TODO: das hier wegnehmen, will face behalten
#    #gesichts keypoits und confidences entfernen, nur Hände und Körper behalten
#    body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1) # pose_confs.shape = (T, 543) — Confidences
#    confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)
#
#    #TODO: ändern wie der hier speichert
#    d = {"keypoints": body_kps, "confidences": confs}
#    # d = {
#    #     'keypoints': array (T, 75, 3),
#    #     'confidences': array (T, 75)
#    # }
#
#    #speichert als pickle datei
#    #TODO: ändern
#    with open(save_path + ".pkl", "wb") as f:
#        pickle.dump(d, f, protocol=4)


#holt ALLE keypoints für ein video (body, face, hands) indem Methoden von oben dafür aufruft
def get_holistic_keypoints(frames):
    """
    For videos, it's optimal to create with `static_image_mode=False` for each video.
    https://google.github.io/mediapipe/solutions/holistic.html#static_image_mode
    """

    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)

    keypoints = []
    confs = []

    #für jeden frame
    for frame in frames:
        results = holistic.process(frame)

        body_data, body_conf = process_body_landmarks(results.pose_landmarks, N_BODY_LANDMARKS)
        face_data, face_conf = process_other_landmarks(results.face_landmarks, N_FACE_LANDMARKS)
        lh_data, lh_conf = process_other_landmarks(results.left_hand_landmarks, N_HAND_LANDMARKS)
        rh_data, rh_conf = process_other_landmarks(results.right_hand_landmarks, N_HAND_LANDMARKS)

        #führe alle generierten keypoints und conf scores zusammen
        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        # data.shape = (543, 3) = 33+468+21+21
        conf = np.concatenate([body_conf, face_conf, lh_conf, rh_conf])
        # conf.shape = (543,)

        keypoints.append(data)
        confs.append(conf)

    # TODO: Reuse the same object when this issue is fixed: https://github.com/google/mediapipe/issues/2152
    holistic.close()  # Schließe MediaPipe-Modell
    del holistic # Lösche Objekt
    gc.collect()  # Force Garbage Collection

    #konvertiert von Liste zu np Array (hier sind kp und confs für ALLE frames)
    keypoints = np.stack(keypoints) # (T, 543, 3) — T Frames, 543 Keypoints, 3 Koordinaten (x,y,z)
    confs = np.stack(confs) # (T, 543) — Confidence pro Keypoint
    return keypoints, confs


#extrahiere BODY keypoints (x,y,z) und confidence score aus mediapipe body/pose landmarks
def process_body_landmarks(component, n_points):
    #initialisiere leere arrays (falls keine landmarks erkannt werden)
    kps = np.zeros((n_points, 3))  # (33, 3) für Pose
    conf = np.zeros(n_points) # (33,) für Confidence

    #wenn landmark vorhanden
    if component is not None:
        landmarks = component.landmark  #MediaPipe LandmarkList
        #Extrahiere x,y,z für jeden Punkt
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        #Extrahiere Visibility (Konfidenz) für Body-Keypoints
        conf = np.array([p.visibility for p in landmarks])

        #kps   # shape (33, 3), dtype float, Werte 0-1 (normalisiert)
            # [[x1, y1, z1], [x2, y2, z2], ...]
        #conf  # shape (33,), dtype float, Werte 0-1 (Visibility-Score)
    #returne BODY keypoints und confidence scores
    return kps, conf

#extrahiere FACE, LEFT HAND, RIGHT HAND keypoints (x,y,z) und confidence score
def process_other_landmarks(component, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        #setze confidence für alle Punkte azf 1
        #TODO: stimmt es, dass mediapipe keine confidence scores für face/hand landmarks liefert?
        conf = np.ones(n_points)
    return kps, conf



################## custom made  #####################################

#das ist der Teil der in custom_2d_skeleton in main unter for anno in tqdm(my_parts) steht
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


#das ist der Teil der wie pose_inference in custom_2d_skeleton ist und dort für keypoint extraction zuständig ist
def mediapipe_inference(anno_in, frames):
    """Extract MediaPipe keypoints and format as custom_2d_skeleton annotation."""
    import copy as cp
    anno = cp.deepcopy(anno_in)
    
    total_frames = len(frames)
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = 1 #assume one person per video in SLR
    
    # Extract keypoints
    pose_kps, pose_confs = get_holistic_keypoints(frames)  # (T, 543, 3), (T, 543)
    
    #if keep_face:
    # Keep all 543 keypoints (body + face + hands)
    keypoints = pose_kps
    confidences = pose_confs
    #num_keypoints = 543
    #else:
        # Only body + hands (75 keypoints)
       # body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1)
        #confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)
        #keypoints = body_kps
        #confidences = confs
       # num_keypoints = 75
    
    # Format: (1, T, num_keypoints, 3) — add person dimension like NTU
    keypoints = keypoints[np.newaxis, ...]  # (1, T, num_keypoints, 3)
    
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
    output_file= os.path.join(args.output_path, "pyskl_mediapipe_annos.pkl")
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
