import copy as cp
import argparse
import os
import pickle
import numpy as np


def denormalize(anno_in):
        anno = cp.deepcopy(anno_in)

        h, w = anno['img_shape']
        keypoint = anno['keypoint']
        
        #ATTENTION: some keypoints have values greater than 1.0 because they are estimated to be out of the image
        #max_val = np.max(keypoint[..., :2])
        #if max_val > 1.0:
        #    print("Warning: Keypoint values greater than 1.0 detected before denormalization.", max_val)


        #multiply normalized keypoints with image width and heigth
        keypoint = keypoint.copy()
        keypoint[..., 0] *= w  #x
        keypoint[..., 1] *= h  #y

        print("Denormalized keypoints for one annotation")

        #max_val = np.max(keypoint[..., :2])
        #print("Max keypoint value AFTER denormalization:", max_val)
        
        
        anno['keypoint'] = keypoint
        return anno



def keypointTo2D(anno_in):
    anno = cp.deepcopy(anno_in)
    if 'keypoint' in anno:
        kp = anno['keypoint']
        if kp.shape[-1] == 3:  #if keypoint has 3 coordinates (x,y,z), remove z
            anno['keypoint'] = kp[..., :2]
    print("Converted keypoints to 2D")
    return anno

def save_anno(output_dict, anno_path):
    output_file= os.path.join(os.path.dirname(anno_path), "pyskl_mediapipe_annos_2d_denormalized.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f"Saved new annotation file with normalized 2d keypoints to {output_file}")


def save_first_anno_as_txt(output_dict, anno_path):
    output_file= os.path.join(os.path.dirname(anno_path), "pyskl_mediapipe_annos_2d_denormalized_firstAnno.txt")
    #save the first annotation and all split values
    first_output = {
        'split': output_dict['split'],
        'annotations': [output_dict['annotations'][0]]
    }

    with open(output_file, 'w') as f:
         f.write(str(first_output))
    print(f"Wrote first annotation and all split values to {output_file}")
     
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test adding the denormalize and 2d to 3d pipeline steps to an annotation file')
    parser.add_argument('anno_path', type=str, help='path to the annotation file, e.g. ../pyskl_mediapipe_annos.pkl')
    args = parser.parse_args()

    anno_path = args.anno_path

    with open(anno_path, 'rb') as f:
        data = pickle.load(f) #data contains split and annotations

    processed_annotations = [] #SICHER DASS DAS ALS LISTE GEHT?
    for anno in data['annotations']:
        anno_2d = keypointTo2D(anno)
        anno_denormalized = denormalize(anno_2d)
        processed_annotations.append(anno_denormalized) 
    
    # Erstelle neues output_dict mit den verarbeiteten Annotations
    output_dict = {
        'split': data['split'],  # Behalte die original split info
        'annotations': processed_annotations
    }

    #NEW
    save_anno(output_dict, anno_path)
    save_first_anno_as_txt(output_dict, anno_path)


    #anno_2d = keypointTo2D(anno)
    #anno_denormalized = denormalize(anno_2d)
    #save_anno(anno_denormalized, anno_path)
    #save_anno_as_txt(anno_denormalized, anno_path)
    #TODO:     ergebnis als txt speichern


