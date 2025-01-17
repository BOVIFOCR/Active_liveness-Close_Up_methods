## Description
This repository holds the implementation of the active liveness methods [Camera Close-up](https://www.sciencedirect.com/science/article/abs/pii/S0167404822000281), [Face Close-up](https://ink.library.smu.edu.sg/sis_research/4513/) and [Hybrid Close-up]().


## Requirements
- [Python 3.8+](https://www.python.org/downloads/)
- [pytorch 2.2.2](https://pytorch.org/)

## Dataset organization
The dataset must be organized in three directories: live, spoof and annotation.

Each sample is a set of image frames that must be in its own subdirectory inside the live or spoof directory.

The frame files of every sample must be named such that when alphabetically ordered, the file order is the same as they appear on the source video.

The overall organization is represented below:

```
<path_to_dataset>/
    ├── live/
    │   ├── sample_0/
    │   │   ├── frame_0.png
    │   │   ├── frame_1.png
    │   │   ├── ...
    │   │   └── frame_a.png
    │   ├── sample_1/
    │   │   ├── frame_0.png
    │   │   ├── frame_1.png
    │   │   ├── ...
    │   │   └── frame_b.png
    │   ├── ...
    │   └── sample_n/
    │       ├── frame_0.png
    │       ├── frame_1.png
    │       ├── ...
    │       └── frame_c.png
    ├── spoof/
    │   ├── sample_n+1/
    │   │   ├── frame_0.png
    │   │   ├── frame_1.png
    │   │   ├── ...
    │   │   └── frame_x.png
    │   ├── sample_n+2/
    │   │   ├── frame_0.png
    │   │   ├── frame_1.png
    │   │   ├── ...
    │   │   └── frame_y.png
    │   ├── ...
    │   └── sample_n+m/
    │       ├── frame_0.png
    │       ├── frame_1.png
    │       ├── ...
    │       └── frame_z.png
    └── annotations/
        └── annotations.csv
```

## Execution

The implemented methods have five different functionalities: pre-process, select frames, extract features, train and test model.

#### Pre-process

Although it is not mandatory, it is highly recomended to pre-process the dataset to reduce execution time of multiple executions of Camera Close-up, Face Close-up and Hybrid Close-up methods.

Pre-processing the dataset will generate three csv files:live_pre_processed.csv, spoof_pre_processed.csv and pre_processed.csv. The first and second files contain a pre-processed distortion feature vector of each frame of each live and spoof samples. The latter is a marged file of the previous two generated files. The following command can be executed to pre-process the used dataset:
```
PATH_TO_METHOD=<path_to_this_directory>
METHOD=<cameraCloseUp.py or faceCloseUp.py or hybridCloseUp.py>
PATH_TO_PRE_PROCESSED=<path_to_pre_processed_directory>
PATH_TO_DATASET=<path_to_dataset>
DLIB_DAT=<path to shape_predictor_68_face_landmarks.dat and mmod_human_face_detector.dat directory>
python3 $PATH_TO_METHOD/$METHOD --pre_processed=$PATH_TO_PRE_PROCESSED --dataset=$PATH_TO_DATASET --dlib_dat_files=$DLIB_DAT --process_dataset
```

For Hybrid Close-up method, it is possible to set the ```--encoder``` argument to determine the used encoder.

#### Frame selection

This funcitonality will execute the frame selector module of the desired method. The csv files containing the samples of train, validation and test partitions must have the columns:

- **label**: containing either live or spoof
- **sampleID**: containing the directory name of the sample

Example:

```
label, sampleID
live, sample_0
live, sample_1
...
live, sample_f
spoof, sample_0
spoof, sample_1
...
spoof, sample_g
```

The following command can be executed to select frames:

```
PATH_TO_METHOD=<path_to_this_directory>
METHOD=<cameraCloseUp.py or faceCloseUp.py or hybridCloseUp.py>
PATH_TO_PRE_PROCESSED=<path_to_pre_processed_directory/pre_processed.csv>
PATH_TO_DATASET=<path_to_dataset>
DLIB_DAT=<path to shape_predictor_68_face_landmarks.dat and mmod_human_face_detector.dat directory>
TRAIN_CSV=<path_to_train_list.csv>
VAL_CSV=<path_to_val_list.csv>
TEST_CSV=<path_to_test_list.csv>
N=<number of frames to select from each sample>
python3 $PATH_TO_METHOD/$METHOD --pre_processed=$PATH_TO_PRE_PROCESSED --dataset=$PATH_TO_DATASET --dlib_dat_files=$DLIB_DAT --N=$N --train=$TRAIN_CSV --val=$VAL_CSV --test=$TEST_CSV --select_frames
```
This command produces three files: frames_train.csv, frames_val.csv and frames_test.csv ;Each file contains a row with its selected frames for each sample of its respective dataset partition.

For Camera Close-up and Hybrid Close-up methods, it is possible to set the ```--b``` argument to determine the number of bins used to divide the input video.


#### Feature extraction

This funcitonality will execute the feature extraction module of the desired method. The csv files containing the selected frames of each sample can be generated by the frame selection functionality. The following command can be executed to extract features:

```
PATH_TO_METHOD=<path_to_this_directory>
METHOD=<cameraCloseUp.py or faceCloseUp.py or hybridCloseUp.py>
PATH_TO_PRE_PROCESSED=<path_to_pre_processed_directory/pre_processed.csv>
PATH_TO_DATASET=<path_to_dataset>
DLIB_DAT=<path to shape_predictor_68_face_landmarks.dat and mmod_human_face_detector.dat directory>
TRAIN_CSV=<path_to_frame_selection/frames_train.csv>
VAL_CSV=<path_to_frame_selection/frames_val.csv>
TEST_CSV<path_to_frame_selection/frames_test.csv>
N=<number of selected frames>
python3 $PATH_TO_METHOD/$METHOD --pre_processed=$PATH_TO_PRE_PROCESSED --dataset=$PATH_TO_DATASET --dlib_dat_files=$DLIB_DAT --N=$N --train=$TRAIN_CSV --val=$VAL_CSV --test=$TEST_CSV --extract_features
```
This command produces three files: features_train.csv, features_val.csv and features_test.csv ;Each file contains a row with the normalized distortion feature vector of each sample of its respective dataset partition.

If the ```--pre_processed``` argument is not declared, the method will not benefit from any pre-processed data and will compute from scratch all desired feature vectors.

For Hybrid Close-up method, it is possible to set the ```--encoder``` argument to determine the used encoder.

#### Train

This funcitonality will execute the training routine of the desired method. The csv files containing the input features of each sample can be generated by the extract features functionality. The following command can be executed to train:

```
PATH_TO_METHOD=<path_to_this_directory>
METHOD=<cameraCloseUp.py or faceCloseUp.py or hybridCloseUp.py>
TRAIN_CSV=<path_to_extracted_features/features_train.csv>
VAL_CSV=<path_to_extracted_features/features_train.csv>
N=<number of selected frames>
python3 $PATH_TO_METHOD/$METHOD --N=$N --train=$TRAIN_CSV --val=$VAL_CSV --learn
```
This command produces one file: <method>_weights.pth containig the weights of the model with the best HTER at the validation partition.

For Hybrid Close-up method, it is possible to set the ```--encoder``` argument to determine the used encoder.

#### Test

This funcitonality will execute the test routine of the desired method. The csv file containing the input features of each sample can be generated by the extract features functionality. The following command can be executed to test:

```
PATH_TO_METHOD=<path_to_this_directory>
METHOD=<cameraCloseUp.py or faceCloseUp.py or hybridCloseUp.py>
TEST_CSV=<path_to_extracted_features/features_test.csv>
WEIGHTS=<path_to_model_weights/<method>_weights.pth>
N=<number of selected frames>
python3 $PATH_TO_METHOD/$METHOD --N=$N --test=$TEST_CSV --weights=$WEIGHTS --eval --v
```

This command will print the Accuracy, HTER and F1-score of the model with weights described by the ```$WEIGHTS``` file on the test partition.

For Hybrid Close-up method, it is possible to set the ```--encoder``` argument to determine the used encoder.

#### Pipelined execution

It is also possible to execute the frame selection, feature extraction and training functionalities at once with the command:

```
PATH_TO_METHOD=<path_to_this_directory>
METHOD=<cameraCloseUp.py or faceCloseUp.py or hybridCloseUp.py>
PATH_TO_PRE_PROCESSED=<path_to_pre_processed_directory/pre_processed.csv>
PATH_TO_DATASET=<path_to_dataset>
DLIB_DAT=<path to shape_predictor_68_face_landmarks.dat and mmod_human_face_detector.dat directory>
TRAIN_CSV=<path_to_train_list.csv>
VAL_CSV=<path_to_val_list.csv>
TEST_CSV=<path_to_test_list.csv>
N=<number of frames to select from each sample>
python3 $PATH_TO_METHOD/$METHOD --pre_processed=$PATH_TO_PRE_PROCESSED --dataset=$PATH_TO_DATASET --dlib_dat_files=$DLIB_DAT --N=$N --train=$TRAIN_CSV --val=$VAL_CSV --test=$TEST_CSV --select_frames --extract_features --learn
```

For Camera Close-up and Hybrid Close-up methods, it is possible to set the ```--b``` argument to determine the number of bins used to divide the input video. Moreover, for Hybrid Close-up method, it is possible to set the ```--encoder``` argument to determine the used encoder.
