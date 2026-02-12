import argparse
import pickle

#To check the max x and y and confidence values of the keypoints in an annotation file
#currently for anno file with keypoints for body and hands only (no face)

#N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21
num_keypoints =  N_BODY_LANDMARKS +  N_HAND_LANDMARKS*2 #number of used keypoints, e.g. 
       
def unpickle(path, num_videos=100):
    with open(path, "rb") as f:
        data = pickle.load(f)

    annotations = data['annotations']

    total_videos = len(annotations)
    print(f"Gesamtanzahl Videos: {total_videos}")
    print(f"Analysiere die ersten {min(num_videos, total_videos)} Videos\n") 

    #create a list to store the max values of each keypoint
    first_video = annotations[0]
    first_kps = first_video['keypoint'] # Shape: [M x T x V x C] = [num persons x num frames x num keypoints x num coordinates]
    M, T, V, C = first_kps.shape
    keypoint_max_values = [[-10, -10, -10] for _ in range(V)] # List of each keypoints, inner list for max x/y and confidence score values of each keypoint
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
                    x, y = frame_kps[kp, :2]  # x and y coordinates
                    score = frame_scores[kp]  # confidence score
                    #check if x and y values are greater than the current max values for this keypoint, if yes, replace the max values with the new ones
                    if x > keypoint_max_values[kp][0]: 
                        keypoint_max_values[kp][0] = x
                    if y > keypoint_max_values[kp][1]:
                        keypoint_max_values[kp][1] = y
                    if score > keypoint_max_values[kp][2]:
                        keypoint_max_values[kp][2] = score



    print("Max values for each keypoint (x, y, confidence score):")
    print("="*50)
    print("max values for BODY keypoints:")
    for i in range(N_BODY_LANDMARKS):
        max_x, max_y, max_conf = keypoint_max_values[i]
        print(f"Keypoint {i}: max x: {max_x}, max y: {max_y}, max confidence score: {max_conf}")
       # print(f"max x {i}: {keypoint_max_values[i][0]}")
       # print(f"max y {i}: {keypoint_max_values[i][1]}")
       # print(f"max conf {i}: {keypoint_max_values[i][2]}")

    print("="*50)
    print("max values for LEFT HAND keypoints:")
    for i in range(N_BODY_LANDMARKS, N_BODY_LANDMARKS + N_HAND_LANDMARKS):
        max_x, max_y, max_conf = keypoint_max_values[i]
        print(f"Keypoint {i}: max x: {max_x}, max y: {max_y}, max confidence score: {max_conf}")
       # print(f"max x {i}: {keypoint_max_values[i][0]}")
       # print(f"max x {i}: {keypoint_max_values[i][0]}")
       # print(f"max y {i}: {keypoint_max_values[i][1]}")
       # print(f"max conf {i}: {keypoint_max_values[i][2]}")

    print("="*50)
    print("max values for RIGHT HAND keypoints:")  
    for i in range(N_BODY_LANDMARKS + N_HAND_LANDMARKS, num_keypoints):
        max_x, max_y, max_conf = keypoint_max_values[i]
        print(f"Keypoint {i}: max x: {max_x}, max y: {max_y}, max confidence score: {max_conf}")
       # print(f"max x {i}: {keypoint_max_values[i][0]}")
       # print(f"max x {i}: {keypoint_max_values[i][0]}")
       # print(f"max y {i}: {keypoint_max_values[i][1]}")
       # print(f"max conf {i}: {keypoint_max_values[i][2]}")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Check values of keypoints in annotation file')
    parser.add_argument('anno', type=str, help='path and name of anno file that contains the extracted keypoints')
    args = parser.parse_args()

    unpickle(args.anno)