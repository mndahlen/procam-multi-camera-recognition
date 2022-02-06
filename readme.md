# Background
In surveillance settings, there could be cameras positioned at different locations with different angles of view. I will attempt to build a system that detects people in different video streams from different views and identifies them as the same person independent of camera position and angle. Several suitable datasets are publically available from the TEV apartment at the Bruno Kessler Institute. Existing object detection models such as YOLO can be tried and used for the detection and identification of persons across multiple cameras.

You can find more information about the Hallway dataset here: https://tev.fbk.eu/resources/hallwaylab

# Results
![Alt Text](docs/presentation/demo_videos/gif_final_model_instance_4.gif)


# Usage
Run main.py in src/

# Description of files and directories
## data/
contains all datasets and scripts for generating data.
- **hallway_639/**: Dataset with 639 generated images from instance 0 of hallway.
- **hallway_1192/**: Dataset with 1192 generated images from instance 0 of hallway.
- **hallway_1192_augmented/**: Hallway_1192 augmented with horizontal flips and noise.
- **hallway_1192_augmented_zero_padded/**: Hallway_1192 with augmentations AND zero padding for rescale to resnet 224x224. 
- **hallway_1192_cropped/**: Hallway_1192 augmented as before and zero padded, also with crop augmentation (removes upper/lower/left/right half of image).
- **augment_data.py**: Main file for augmentating dataset once it has been generated using generate_dataset.py.
- **create_csv.py**: When a dataset has been created, use this to create csv with labels for all images.
- **generate_dataset.py**: Generates a dataset by taking images from hallway (specifically instance 0 of hallway), detects persons with yolo and cuts them out with boundingbox. The pictures will be manually labeled to the correct person. Which is which person is seen by examining the datasets.

## models/
- **resnet18_hallway_639_1_5**: Resnet 18 trained on hallway_639, 1 epoch, batch size 5.
- **resnet18_hallway_639_1_10**: Resnet 18 trained on hallway_639, 1 epoch, batch size 10.
- **resnet18_hallway_639_1_20**: Resnet 18 trained on hallway_639, 1 epoch, batch size 20.
- **resnet18_hallway_639_2_20**: Resnet 18 trained on hallway_639, 2 epochs, batch size 20.
- **resnet18_hallway_639_3_20**: Resnet 18 trained on hallway_639, 3 epochs, batch size 20.
- **resnet18_hallway_1192_2_20**: Resnet 18 trained on hallway_1192, 2 epochs, batch size 20.
- **resnet18_hallway_1192_3_20**: Resnet 18 trained on hallway_1192, 3 epochs, batch size 20.
- **resnet18_hallway_1192_augmented_3_20**: Resnet 18 trained on hallway_1192_augmented, 3 epochs, batch size 20.
- **resnet18_hallway_1192_augmented_zero_padded_3_20**: Resnet 18 trained on hallway_1192_augmented_zero_padded, 3 epochs, batch size 20.
- **resnet18_hallway_1192_cropped_3_20**: Resnet 18 trained on hallway_1192_cropped, 3 epochs, batch size 20.
## src/
- **FeatureExtractor.py**: Wrapper of Resnet(Or any model for extracting features) for use in main script.
- **helpers.py**: Various helper functions for different modules in this repository.

- **main.py**: Main script for demonstrating project.
- **PersonDataset.py**: Dataloader for dataset.
- **test_feature_similarity.py**: Script for testing use of resnet for extracting visual person similarities. Also plots the cosine distance results.
- **train_resnet.py**: Trains resnet for use as a feature extractor.
## video/
Just material for presentation of the project. Enjoy!
