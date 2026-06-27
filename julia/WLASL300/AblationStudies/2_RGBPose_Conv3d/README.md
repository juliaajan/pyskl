# RGBPose Conv3d
RGBPose Conv3d is a model that incorporates not only the skeletal heatmaps, but also an RGB stream with the original sign videos.

# Training

## Step 1. Data Preprosessing
First, download and unzip the RGB videos from WLASL300 in a folder called "WLASL300", and compress the raw video with compress_wlasl300.py. This adds a new folder called WLASL_300_compressed and places the videos rescaled to `960x540` inside.

## Step 2. Pretraining
First, we train with the RGB and skeletal heatmaps separately, to generate weights that will be used to initialise the final RGBPose_Conv3d model.

```bash
# Train the RGB-only model (1 GPU)
bash tools/dist_train.sh julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/rgb_only.py 1 --validate --test-last --test-best
# Train the Pose-only model (1 GPU)
bash tools/dist_train.sh julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/pose_only.py 1 --validate --test-last --test-best
```