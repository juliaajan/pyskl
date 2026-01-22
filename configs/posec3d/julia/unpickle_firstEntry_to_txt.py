import pickle
#um pickle files in txt umzuwandeln und zu testen


path = '../pyskl_mediapipe_annos.pkl'
#path = 'C:\\Users\\juja\\Downloads\\diving48_hrnet.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
    #C:/Users/juja/Desktop\Julia\Studium\Uni\Master\Masterarbeit\Code

    anno = data['annotations'][0]
    if 'keypoint' in anno:
        kp = anno['keypoint']
        if kp.shape[-1] == 3:  #if keypoint has 3 coordinates (x,y,z), remove z
            kp_only_2coords = kp[..., :2]
            print("##Keypoint shape is 3, now reduced to 2")
            print("2D keypoint shape:", kp_only_2coords.shape)
    else: 
        print("No keypoint found in first annotation")





# Prüfe ersten Eintrag
anno = data['annotations'][0]
print(f"Keypoint shape: {anno['keypoint'].shape}")
print("first frame")
print(anno['keypoint'][0])  # Keypoints des ersten Frames anzeigen

#save as txt
with open(path.replace('.pkl', '.txt'), 'w') as f:
    f.write(str(data['annotations'][0]))
print(f"Wrote first annotation to {path.replace('.pkl', '.txt')}")


#labels = [anno['label'] for anno in data['annotations']]
#print(f"Min label: {min(labels)}")  # Sollte 0 sein
#print(f"Max label: {max(labels)}")  # Sollte 299 sein
#print(f"Unique labels: {len(set(labels))}")  # Sollte 300 sein