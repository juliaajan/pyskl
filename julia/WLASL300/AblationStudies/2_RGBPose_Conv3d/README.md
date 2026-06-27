# RGBPose Conv3d
RGBPose Conv3d is a model that incorporates not only the skeletal heatmaps, but also an RGB stream with the original sign videos.

# Training

## Data Preprosessing
First, download the RGB videos from WLASL300 and compress the raw video with compress_wlasl300.py. This adds a new folder called WLASL_300_compressed and places the videos rescaled to `960x540` inside.