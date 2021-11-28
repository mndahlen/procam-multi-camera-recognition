import torch
import cv2
import os
import time
from helpers import *
from FeatureExtractor import FeatureExtractor
device = torch.device("cpu")

IMGDIR = "../data/hallway"

def update_cam_persons(bbox,im,persons,T):
    if not persons:
        persons = {}
    
    num_detected_persons = bbox.shape[0]
   
    for i in range(0,num_detected_persons):
        b = bbox[i]
        x1 = int(b[0])
        y1 = int(b[1])
        x2 = int(b[2])
        y2 = int(b[3])
        person_im = im[y1:y2,x1:x2] 
    pass



# models
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
feature_extractor = FeatureExtractor()

cam_instance = 0
num_frames = 1000
T = 0.9

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

    c1_yolo_results = yolo(c1)
    c2_yolo_results = yolo(c2)
    c3_yolo_results = yolo(c3)
    c4_yolo_results = yolo(c4)

    c1_bbox = get_bbox(c1_yolo_results)
    c2_bbox = get_bbox(c2_yolo_results)
    c3_bbox = get_bbox(c3_yolo_results)
    c4_bbox = get_bbox(c4_yolo_results)

    c1_persons = feature_extractor.get_cam_persons_3(c1_bbox,c1,c1_persons,T_bbox=100,T_sim = 0.8)
    #c2_persons = feature_extractor.get_cam_persons(c2_bbox,c2,c2_persons,T)
    #c3_persons = feature_extractor.get_cam_persons(c3_bbox,c3,c3_persons,T)
    #c4_persons = feature_extractor.get_cam_persons(c4_bbox,c4,c4_persons,T)
    print(len(c1_persons))
    blue =   (255,0,0)
    red = (0,0,255)
    green =  (0,255,0)

    c1_w_bbox = draw_bbox_on_im(c1_bbox, c1, red, selected=True, selected_bbox=c1_persons[1]["bbox"])
    c2_w_bbox = draw_bbox_on_im(c2_bbox, c2, red)
    c3_w_bbox = draw_bbox_on_im(c3_bbox, c3, red)
    c4_w_bbox = draw_bbox_on_im(c4_bbox, c4, red)
    merged_im = merge_4_im(c1_w_bbox,c2_w_bbox,c3_w_bbox,c4_w_bbox)

    cv2.imshow("test",merged_im)
    cv2.waitKey(1) 
    #closing all open windows 
cv2.destroyAllWindows() 
