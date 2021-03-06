import torch
import cv2
import os
from helpers import *
from FeatureExtractor import FeatureExtractor

"""
Main script for demonstrating project.
"""

# Pytorch models
device = torch.device("cpu")
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
feature_extractor = FeatureExtractor("models/resnet18_hallway_1192_cropped_3_20.tar",zero_pad=True)

# Parameters
IMGDIR = "data/hallway"
cam_instance = 4
num_frames = 60
main_selected_person_idx = 0 # Test with different values and see what happens! Warning, for some cases the idx can be too high and the program will crash.
T = 0.9
FEATURE_START_BUFFER = -1


# Run script
c1_persons = None
c2_persons = None
c3_persons = None
c4_persons = None
for frame in range(num_frames):    
    frame_idx = idx_to_string(frame)
    c1 = cv2.imread(os.path.join(IMGDIR,"c1/00000{}/00{}{}.jpg".format(cam_instance,cam_instance,frame_idx)))
    c2 = cv2.imread(os.path.join(IMGDIR,"c2/00000{}/00{}{}.jpg".format(cam_instance,cam_instance,frame_idx)))
    c3 = cv2.imread(os.path.join(IMGDIR,"c3/00000{}/00{}{}.jpg".format(cam_instance,cam_instance,frame_idx)))
    c4 = cv2.imread(os.path.join(IMGDIR,"c4/00000{}/00{}{}.jpg".format(cam_instance,cam_instance,frame_idx)))

    # Execute yolo on camera images
    c1_yolo_results = yolo(c1)
    c2_yolo_results = yolo(c2)
    c3_yolo_results = yolo(c3)
    c4_yolo_results = yolo(c4)

    # Extract bbox from yolo result
    c1_bbox = get_bbox(c1_yolo_results)
    c2_bbox = get_bbox(c2_yolo_results)
    c3_bbox = get_bbox(c3_yolo_results)
    c4_bbox = get_bbox(c4_yolo_results)

    # Update bbox and features for all persons, create new or match on feature if needed.
    c1_persons = feature_extractor.get_cam_persons(c1_bbox,c1,c1_persons,T_bbox=100,T_sim = 0.8)
    c2_persons = feature_extractor.get_cam_persons(c2_bbox,c2,c2_persons,T_bbox=100,T_sim = 0.8)
    c3_persons = feature_extractor.get_cam_persons(c3_bbox,c3,c3_persons,T_bbox=100,T_sim = 0.8)
    c4_persons = feature_extractor.get_cam_persons(c4_bbox,c4,c4_persons,T_bbox=100,T_sim = 0.8)

    # Get most similar persons to selected person from camera 1
    if frame > FEATURE_START_BUFFER:
        person_camera_idxs = feature_extractor.get_closest_persons(main_selected_person_idx, c1_persons, c2_persons, c3_persons, c4_persons)
    else:
        person_camera_idxs = [0,0,0,0]
    
    # Define colors
    blue =   (255,0,0)
    red = (0,0,255)
    green =  (0,255,0)

    # Draw bounding boxes
    c1_w_bbox = draw_bbox_on_im(c1_bbox, c1, red, selected=True, selected_bbox=c1_persons[person_camera_idxs[0]]["bbox"])
    c2_w_bbox = draw_bbox_on_im(c2_bbox, c2, red, selected=True, selected_bbox=c2_persons[person_camera_idxs[1]]["bbox"])
    c3_w_bbox = draw_bbox_on_im(c3_bbox, c3, red, selected=True, selected_bbox=c3_persons[person_camera_idxs[2]]["bbox"])
    c4_w_bbox = draw_bbox_on_im(c4_bbox, c4, red, selected=True, selected_bbox=c4_persons[person_camera_idxs[3]]["bbox"])
    # Add num persons counter
    c1_w_bbox = draw_text_on_im(c1_w_bbox, str(len(c1_persons)),(750,950))
    c2_w_bbox = draw_text_on_im(c2_w_bbox, str(len(c2_persons)),(750,950))
    c3_w_bbox = draw_text_on_im(c3_w_bbox, str(len(c3_persons)),(750,950))
    c4_w_bbox = draw_text_on_im(c4_w_bbox, str(len(c4_persons)),(750,950))
    # Draw frame num
    c1_w_bbox = draw_text_on_im(c1_w_bbox, str(frame),(250,950), color =(0,255,0))
    # Merge four views to one display
    merged_im = merge_4_im(c1_w_bbox,c2_w_bbox,c3_w_bbox,c4_w_bbox)

    cv2.imshow("test",merged_im)
    cv2.waitKey(1) 

# If still open, close all opened windows
cv2.destroyAllWindows() 
