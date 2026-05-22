# Explanation of Evaluation files

## create_confusion_matrix.py
This file creates confusion matrices for a given pkl file. Confusion matrices can be produced using no normalization, normalization of the true labels, the predicted labels, or normalizing both the true and the predicted labels. Normally, normalization of the true labels is used. To change this, specify the desired normalization by changing the parameters of np.save, np.savetxtx and visualize_confusion_matrix at the end of the file to cm_raw, cm_true, cm_pred or cm_all.

This file also saves the values as npy and csv, and can check if any of the ground truth labels are missing in the rpediction, meaning that a ground truth label was not once predicted. This can be achieved by uncommenting the line calling analyze_missing_classes.

Args: The desired pkl file 
Use with `python create_confusion_matrix.py best_pred.pkl`


## plot_accuracy__and_learningrate.py
This file plots train loss and accuracy as well as validation loss an accuracy and saves the plots at the given path. Additionally, all validation losses and their corresponding epochs are saved in a dedicated txt file.

Note: The file uses the .log.json file that is created during training to extract training loss and accuracy and validation accuracy, and the .log file to extract the validation losses. It is assumed that both files have the same filename.

Note: In case the training process was interrupted, a new log file is created, starting where the old training ended. In this case, both the .log.json and the .log file need to be merged manually into one file to make sure that values from all epochs can be extracted.

Args: the .log.json file that was created during training
Use with `python plot_accuracy_and_learningrate.py 20260520_110512.log.json`


## plot_accuracy__and_learningrate2.py
Acts the same way as plot_accuracy__and_learningrate.py, with the only difference being that two separate files are used to extract training and validation losses/accuracies. First path specifies the log file for the training values, second one for the validation values

Args: the .log.json file that was created during training, one for trainign and one for validation accuracies/losses
Use with `python plot_accuracy_and_learningrate.py 20260520_110512_train.log.json 20260520_110512_val.log.json`


## unpickle_pth.py
Convert a pth file to txt to easily view its content.

Args: the pth file
Use with `python unpickle_pth.py epoch_10.pth`

## visualize_heatmap_volumes.py
Creates and saves joint and limb heatmaps for the given video and its annotation file.

Args:
-annotation file containing the extracted keypoints for all videos
-path to a file containing mappings of gloss indices to gloss names, here: wlasl300/label_mappings.txt
-video_id (not the path!) of the video that should be used, eg "07383"
-output path where the vsualization videos should be saved

Use with: `python visualize_heatmap_volume.py pyskl_mediapipe_annos.pkl julia/WLASL300/label_mapping.txt WLASL_300/test/language/32167.mp4 julia/WLASL300/vis_heatmaps_01_20`



## visualize_skeleton.py (former visualize.py)
Visualizes the with MediaPipe extracted skeletal keypoints over the original video.

Args:
-the annotation file that was created with wlasl_prepare_mediapipe_extraction.py and that contains the with MediaPipe extracted keypoints (of all videos in the WLASL dataset)
-the path to the input video file, that is used as a background video on which the skeleton will be plotted over. Note: The correct annotation for the specified video is loaded from the annotationfile using the video_id extracted from the filename. Therefore, the video file must be named after 'video_id.mp4', e.g. '32167.mp4'
-the output folder where to save the visualization

Use with  `python visualize.py julia/WLASL300/pyskl_mediapipe_annos.pkl /data/0janssen/WLASL300/WLASL_300/test/language/32167.mp4 julia/WLASL300 `

Note: The parameters of Vis2DPoseMediaPipe can be adapted to specify which keypoints should be displayed onto the video, e.g. face, hands, upper_body, full body
