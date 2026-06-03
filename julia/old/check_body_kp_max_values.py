import argparse
import pickle
from datetime import datetime
import os

#To check the max x and y and confidence values of the keypoints in an annotation file
#currently for anno file with keypoints for body and hands only (no face)

#N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21
num_keypoints =  N_BODY_LANDMARKS +  N_HAND_LANDMARKS*2 #number of used keypoints, e.g. 
output_handle = None

def write_output(message):
    if output_handle:
        #print to file (if output_path was given)
        print(message, file=output_handle)

    #print to console
    print(message)
       


def unpickle(path, num_videos=5000, output_dir=None):
    global output_handle
    
    #create logging file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"keypoints_max_values_{timestamp}.txt")
        output_handle = open(output_file, 'w')
        print(f"Output wird gespeichert in: {output_file}\n")


    #open annotation file
    with open(path, "rb") as f:
        data = pickle.load(f)

    annotations = data['annotations']

    total_videos = len(annotations)
    write_output(f"Gesamtanzahl Videos: {total_videos}")
    write_output(f"Analysiere die ersten {min(num_videos, total_videos)} Videos\n") 

    #create a list to store the max values of each keypoint
    first_video = annotations[0]
    first_kps = first_video['keypoint'] # Shape: [M x T x V x C] = [num persons x num frames x num keypoints x num coordinates]
    M, T, V, C = first_kps.shape
    #create a matrix that stores x, y, conf score and video_id for each max keypoint, initialized with -1
    keypoint_max_values = [[-1, -1, -1, "video_id"] for v in range(V)] 
    #attention max values of x, y and the score do NOT necessarily belong to the same video and frame, we store their max values independently

    #iterate through all videos and check max value for x and y coordinates and their corresponding confidence score for each keypoint
    for video_dir in range(min(num_videos, total_videos)):
        video = annotations[video_dir] #iterate through the videos
        kps = video['keypoint']  # Shape: [M x T x V x C]
        kp_scores = video['keypoint_score']
        M, T, V, C = kps.shape  # M=Personen, T=Frames, V=Keypoints, C=Koordinaten

        for person_idx in range(M): #iterate through the persons in the video (should be only one for SLR)
            for frame in range(T): #iterature through each video frame
                frame_kps = kps[person_idx, frame, :, :]  # Shape: [V x C]
                frame_scores = kp_scores[person_idx, frame, :]  # Shape: [V]

                for kp in range(V): #iterate through each keypoint
                    x, y = frame_kps[kp, :2]  #x and y coordinates
                    score = frame_scores[kp]  #confidence score
                    video_id = video['frame_dir'] 
                    #check if x, y and conf score values are greater than the current max values for this keypoint, if yes, replace the max values with the new ones
                    if x > keypoint_max_values[kp][0]: 
                        keypoint_max_values[kp][0] = x
                    if y > keypoint_max_values[kp][1]:
                        keypoint_max_values[kp][1] = y
                    if score > keypoint_max_values[kp][2]:
                        keypoint_max_values[kp][2] = score
                        keypoint_max_values[kp][3] = video_id #save at which video_id the max confidence was found



    write_output("Max values for each keypoint (x, y, confidence score):")
    write_output("="*50)
    write_output("max values for BODY keypoints:")
    for i in range(N_BODY_LANDMARKS):
        max_x, max_y, max_conf, video_id = keypoint_max_values[i]
        write_output(f"Keypoint {i}: max x: {max_x}, max y: {max_y}, max confidence score: {max_conf} found in video: {video_id}")

    write_output("="*50)
    write_output("max values for LEFT HAND keypoints:")
    for i in range(N_BODY_LANDMARKS, N_BODY_LANDMARKS + N_HAND_LANDMARKS):
        max_x, max_y, max_conf, video_id = keypoint_max_values[i]
        write_output(f"Keypoint {i}: max x: {max_x}, max y: {max_y}, max confidence score: {max_conf} found in video: {video_id}")

    write_output("="*50)
    write_output("max values for RIGHT HAND keypoints:")  
    for i in range(N_BODY_LANDMARKS + N_HAND_LANDMARKS, num_keypoints):
        max_x, max_y, max_conf, video_id = keypoint_max_values[i]
        write_output(f"Keypoint {i}: max x: {max_x}, max y: {max_y}, max confidence score: {max_conf} found in video: {video_id}")



    if output_handle:
        output_handle.close()
        print(f"Saved max keypoint values to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Check values of keypoints in annotation file')
    parser.add_argument('anno', type=str, help='path and name of anno file that contains the extracted keypoints')
    parser.add_argument('--output-dir', type=str, default=None, help='optional output directory to save results to a txt file')
    args = parser.parse_args()

    unpickle(args.anno, output_dir=args.output_dir)

    #python julia/old/check_body_kp_max_values.py "C:\Users\juja\Desktop\Julia\Studium\Uni\Master\Masterarbeit\Code\pyskl_mediapipe_annos_2d_denormalized_NOFACE.pkl" --output-dir "C:\Users\juja\Desktop\Julia\Studium\Uni\Master\Masterarbeit\Results\AblationStudies\1_KeypointSelection"