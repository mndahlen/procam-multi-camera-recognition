import torch
import cv2
import os
IMGDIR = "data/hallway"
DATADIR = "data/hallway_persons_0"
PRESENTATIONDIR = "data/presentation"
# Model
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

max_im_index = 40
cameras = ['c1','c2','c3','c4']
for camera in cameras:
        img = cv2.imread(os.path.join(IMGDIR,"{}/000000/000000.jpg".format(camera)))

        results = yolo(img)

        person_detections = results.pandas().xyxy[0]
        person_detections = person_detections.loc[person_detections['name'] == 'person']
        bbox = person_detections[['xmin','ymin','xmax','ymax']].to_numpy()
        num_persons = bbox.shape[0]
        
        img_draw_bbox = img.copy()
        for i in range(0,num_persons):
            b = bbox[i]
            x1 = int(b[0])
            y1 = int(b[1])
            x2 = int(b[2])
            y2 = int(b[3])
            cv2.imwrite(os.path.join(PRESENTATIONDIR,"{}_000_{}.png".format(camera,i)),img[y1:y2,x1:x2])
            cv2.rectangle(img_draw_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(PRESENTATIONDIR,"{}_bbox.png".format(camera)),img_draw_bbox)