import cv2
import numpy as np
from PIL import Image, ImageOps

def idx_to_string(idx):
    if idx < 10:
        idx_str = "00" + str(idx)
    elif idx < 100:
        idx_str = "0" + str(idx)
    else:
        idx_str = str(idx)
    return idx_str

def get_bbox(yolo_result):
    person_detections = yolo_result.pandas().xyxy[0]
    person_detections = person_detections.loc[person_detections['name'] == 'person']
    bbox = person_detections[['xmin','ymin','xmax','ymax']].to_numpy()
    return bbox

def draw_bbox_on_im(bbox, im, color, selected=False, selected_bbox=None): 
    num_detected_persons = bbox.shape[0]
   
    for i in range(0,num_detected_persons):
        b = bbox[i]
        x1 = int(b[0])
        y1 = int(b[1])
        x2 = int(b[2])
        y2 = int(b[3])
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
    if selected:
        print("SELECTED")
        x1 = int(selected_bbox[0])
        y1 = int(selected_bbox[1])
        x2 = int(selected_bbox[2])
        y2 = int(selected_bbox[3])
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return im

def draw_text_on_im(img, text, coord=(10,500), color =(255,255,255)):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 10
    fontColor              = color
    thickness              = 10
    lineType               = 20

    img = cv2.putText(img,text, 
        coord, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    return img

def merge_4_im(im1,im2,im3,im4):
    height = int(im1.shape[0]/4)
    width = int(im1.shape[1]/4)
    resized_1 = cv2.resize(im1, (width,height), interpolation = cv2.INTER_AREA)
    resized_2 = cv2.resize(im2, (width,height), interpolation = cv2.INTER_AREA)
    resized_3 = cv2.resize(im3, (width,height), interpolation = cv2.INTER_AREA)
    resized_4 = cv2.resize(im4, (width,height), interpolation = cv2.INTER_AREA)
    im_1_2 = np.concatenate((resized_1, resized_2), axis=1)
    im_3_4 = np.concatenate((resized_3, resized_4), axis=1)
    im_1_2_3_4 = np.concatenate((im_1_2, im_3_4), axis=0)
    
    return im_1_2_3_4

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)