import cv2
import os, sys, gc
import time
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm
import multiprocessing
from joblib import Parallel, delayed
from natsort import natsorted
from glob import glob
import math
import pickle

#Quelle: https://github.com/AI4Bharat/OpenHands/blob/main/scripts/mediapipe_extract.py

#impoertiere mediapipe holistic module
mp_holistic = mp.solutions.holistic

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21

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


#holt ALLE keypoints für ein video (body, face, hands) indem Methoden von oben dafür aufruft
def get_holistic_keypoints(
    frames, holistic=mp_holistic.Holistic(static_image_mode=False, model_complexity=2)
):
    """
    For videos, it's optimal to create with `static_image_mode=False` for each video.
    https://google.github.io/mediapipe/solutions/holistic.html#static_image_mode
    """

    keypoints = []
    confs = []

    #für jeden frame
    for frame in frames:
        results = holistic.process(frame)

        body_data, body_conf = process_body_landmarks(
            results.pose_landmarks, N_BODY_LANDMARKS
        )
        face_data, face_conf = process_other_landmarks(
            results.face_landmarks, N_FACE_LANDMARKS
        )
        lh_data, lh_conf = process_other_landmarks(
            results.left_hand_landmarks, N_HAND_LANDMARKS
        )
        rh_data, rh_conf = process_other_landmarks(
            results.right_hand_landmarks, N_HAND_LANDMARKS
        )

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




def gen_keypoints_for_frames(frames, save_path):
    #alle keypoints holen (body und rest)
    pose_kps, pose_confs = get_holistic_keypoints(frames) # pose_kps.shape = (T, 543, 3) — alle Keypoints (Body + Face + Hands)
    
    #TODO: das hier wegnehmen, will face behalten
    #gesichts keypoits und confidences entfernen, nur Hände und Körper behalten
    body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1) # pose_confs.shape = (T, 543) — Confidences
    confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)

    d = {"keypoints": body_kps, "confidences": confs}
    # d = {
    #     'keypoints': array (T, 75, 3),
    #     'confidences': array (T, 75)
    # }

    #speichert als pickle datei
    with open(save_path + ".pkl", "wb") as f:
        pickle.dump(d, f, protocol=4)


#lädt für ein video alle frames
def load_frames_from_video(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        success, img = vidcap.read()
        if not success:
            break
        #konvertiert von BGR zu RGB
        #TODO: prüfen ob das nötig ist, ob Bilder vorher wirklich BGR sind
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #optional: resize frame
        # img = cv2.resize(img, (640, 480))
        frames.append(img)

    vidcap.release()
    # cv2.destroyAllWindows()
    return np.asarray(frames)


#lädt alle frames aus einem ordner (statt aus Video)
#(wird nicht benutzt)
def load_frames_from_folder(frames_folder, patterns=["*.jpg"]):
    images = []
    for pattern in patterns:
        images.extend(glob(f"{frames_folder}/{pattern}"))
    images = natsorted(list(set(images)))  # remove dupes
    if not images:
        exit(f"ERROR: No frames in folder: {frames_folder}")

    frames = []
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return np.asarray(frames)

#erstellt keypoints für frames aus video
def gen_keypoints_for_video(video_path, save_path):
    if not os.path.isfile(video_path):
        print("SKIPPING MISSING FILE:", video_path)
        return
    frames = load_frames_from_video(video_path)
    gen_keypoints_for_frames(frames, save_path)

#wird nicht benutzt:
#erstellt keypoints für frames aus folder (statt aus video)
def gen_keypoints_for_folder(folder, save_path, file_patterns):
    frames = load_frames_from_folder(folder, file_patterns)
    gen_keypoints_for_frames(frames, save_path)


def generate_pose(dataset, save_folder, worker_index, num_workers, counter):
    ## 1. Berechne welche Videos dieser Worker verarbeiten soll
    num_splits = math.ceil(len(dataset) / num_workers)
    end_index = min((worker_index + 1) * num_splits, len(dataset))
    #jeder worker verarbeitet seine videos und holt aus den keypoints
    for index in range(worker_index * num_splits, end_index):
        imgs, label, video_id = dataset.read_data(index)
        save_path = os.path.join(save_folder, video_id)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #in jedem worker-prozess
        gen_keypoints_for_frames(imgs, save_path)
        counter.increment()


#(wird nicht benutzt)
def dump_pose_for_dataset(
    dataset, save_folder, num_workers=multiprocessing.cpu_count()
):
    #erstellt output ordner
    os.makedirs(save_folder, exist_ok=True)
    #initialisiere worker lsite und shared counter
    processes = []
    counter = Counter()
    #startet worker process
    for i in tqdm(range(num_workers), desc="Creating sub-processes..."):
        p = multiprocessing.Process(
            target=generate_pose, args=(dataset, save_folder, i, num_workers, counter)
        )
        p.start()
        processes.append(p)

    #progress bar für alle worker
    total_samples = len(dataset)
    with tqdm(total=total_samples) as pbar:
        while counter.value < total_samples:
            pbar.update(counter.value - pbar.n)
            time.sleep(2)

    #warte auf alle worker
    for i in range(num_workers):
        processes[i].join() #blockiere bis prozess i fertig ist
    print(f"Pose data successfully saved to: {save_folder}")


if __name__ == "__main__":
    # gen_keypoints_for_video("/home/gokulnc/data-disk/datasets/Chinese/CSL/word/color/000/P01_01_00_0._color.mp4", "sample.pkl")
    #anzahl cpu-cores ermitteln
    n_cores = multiprocessing.cpu_count()

    #Pfade definieren
    DIR = "AUTSL/train/" #input pfad
    SAVE_DIR = "AUTSL/holistic_poses/" #output: keypoints

    os.makedirs(SAVE_DIR, exist_ok=True)

    #alle video pfade sammeln
    file_paths = []
    save_paths = []
    for file in os.listdir(DIR):
        if "color" in file: # Nur Videos mit "color" im Namen TODO: ändern
            file_paths.append(os.path.join(DIR, file)) ## z.B. "AUTSL/train/video_001_color.mp4"
            save_paths.append(os.path.join(SAVE_DIR, file.replace(".mp4", ""))) # z.B. "AUTSL/holistic_poses/video_001_color"

    #Parallele Verarbeitung mit joblib
    Parallel(n_jobs=n_cores, backend="loky")(
        delayed(gen_keypoints_for_video)(path, save_path)
        for path, save_path in tqdm(zip(file_paths, save_paths))
    )
