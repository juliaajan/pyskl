# RGBPose Conv3d
RGBPose Conv3d is a model that incorporates not only the skeletal heatmaps, but also an RGB stream with the original sign videos.

# Training

## Step 1. Data Preprosessing
First, download and unzip the RGB videos from WLASL300 in a folder called "WLASL300", and compress the raw video with compress_wlasl300.py. This adds a new folder called WLASL_300_compressed and places the videos rescaled to `960x540` inside. Additionally, it takes the annotation file with the skeletal keypoints and compresses the keypoints to the according size. It is important that the compressed skeletal file is used for the RGB stream, as videos are cropped in the train pipeline to a bounding box around the extracted keypoints. If the uncompressed annotation file is used, the relevant keypoints might be cropped out of the rescaled videos.

`python julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/compress_wlasl300.py  --input-video-path ../WLASL300/WLASL_300 --output-video-path ../WLASL300/WLASL_300_compressed --ann-file  julia/WLASL300/pyskl_mediapipe_annos_2d_denormalized_NOFACE_NOBODY.pkl `

You can use `check_compressed_files.py` to make sure that all compressed videos are processed correctly and still have the same number of frames as before the compression. A frame mismatch of one between the original video and the compressed video is tolerated, but larger mismatches will result in an termination of the training process, as this means that a video was broken during the compression and might not show the compelte sign execution any more. In case you have frame mismatches, please try to compress the video again and maybe try a different envoder in compress_wlasl300.py. 
    `python julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/check_compressed_files. py ../WLASL300 julia/WLASL300/pyskl_mediapipe_annos_2d_denormalized_NOSE_FACE_HANDS.pkl`                                                                                                                                                             

## Step 2. Pretraining
First, we train with the RGB and skeletal heatmaps separately, to generate weights that will be used to initialise the final RGBPose_Conv3d model.
Make sure to use the compressed annotation file created with compress_wlasl300.py!

```bash
# Train the RGB-only model (1 GPU)
bash tools/dist_train.sh julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/rgb_only.py 1 --validate --test-last --test-best
# Train the Pose-only model (1 GPU)
bash tools/dist_train.sh julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/pose_only.py 1 --validate --test-last --test-best
```