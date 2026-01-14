# Explanation of added files
## wlasl_preprocessing_rename_files.py
> For preparing the data of the WLASL dataset 

After downloading and preprocessing the video files for the WLASL (https://github.com/dxli94/WLASL/tree/master), you need to create an annotationfile containing all glosses and rename the files by appending their corresponding glosses. These are needed to be able to use ntu_preprocessing.py

In the wlasl_preprocessing_rename_files.py script, set the path to the folder with the downloaded and preprocessed videos (e.g. WLASL/start_kit/videos), the path to the json file (e.g. WLASL/WLASL_v0.3.json when you have cloned the whole WLASL repo, else download the `WLASL_v0.3.json` file manually from https://github.com/dxli94/WLASL/tree/master/start_kit and set the path accordingly), and the path to the folder where your annotation file should be created.
 
### example:
when the WLASL json contains the following glosses: book, drink, computer, before, ... <br >
The annotation file will look like this:
 ```
 book
 drink
 computer
 before
 ...
```
And a video called "1234.mp4" belonging to the gloss "computer" will be renamed to "1234_GL00003.mp4" (because its gloss is in the third line of the annotation file)

> The pickle file that will be generated with ntu_pose_extraction will use label: 2 for this video (since it starts couting at index 0)


## unpickle.py
After downloading and sorting the pickle file, this file helps to unpickle and print the content of a single pickle file in order to understand its structure.
