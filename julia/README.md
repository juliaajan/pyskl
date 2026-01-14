# Explanation of added files

## wlasl_prepare_mediapipe_extraction.py
As the name suggests, this file helps to prepare the the wlasl300 dataset before the keypoint extraction with mediapipe can start.
After downloading the Wlasl300 with the following commands:


- `mkdir WLASL300`
- `cd WLASL300`
- `#!/bin/bash curl -L -o path/wlasl300.zip\ https://www.kaggle.com/api/v1/datasets/download/thtrnphc/wlasl300`
- `unzip wlasl300.zip`
- `cd WLASL_300`
- `ls` → should show three folders: train, test and val

Use this file to create a label file for the downloaded videos, which will later be needed to use `wlasl_extract_mediapipe_keypoints.py`.

It will create a new file called `wlasl300.list` containing all video paths and their corresponding labels in the following format:

```bash
WLASL300/train/12345.mp4 book
WLASL300/val/23467.mp4 since
WLASL300/train/98765.mp4 read
WLASL300/train/00123.mp4 slow
WLASL300/test/00678.mp4 africa
...
```



## wlasl_extract_mediapipe_keypoints

This file creates the annotation file in the format needed by pyskl, which contains information about the frame, the detected keypoints and its corresponding confidence score.

```bash 
[
        {
        'frame_dir': '13245',
        'label': 'drink',
        'img_shape': (720, 1280),
        'total_frames': 95,
        'num_person_raw': 1,
        'keypoint': array(1, 95, 543, 3), #543 keypoints, 3 coordinates (x,y,z) each
        'keypoint_score': array(1, 95, 543) 
    },
    {...}, 
    ...
] 
```

## unpickle.py
This file helps to unpickle and print the content of a single pickle file in order to understand its structure.



