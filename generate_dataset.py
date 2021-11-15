import torch
import cv2
import os
IMGDIR = "data/hallway"
DATADIR = "data/hallway_persons_0"
# Model
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

max_im_index = 40
cameras = ['c1','c2','c3','c4']
for camera in cameras:
    for idx in range(0,max_im_index + 1):
        if idx < 10:
            idx_str = "00" + str(idx)
        elif idx < 100:
            idx_str = "0" + str(idx)
        else:
            idx_str = str(idx)

        print("Camera: {}, idx: {}\n".format(camera, idx_str))

        img = cv2.imread(os.path.join(IMGDIR,"{}/000000/000{}.jpg".format(camera,idx_str)))

        results = yolo(img)

        person_detections = results.pandas().xyxy[0]
        person_detections = person_detections.loc[person_detections['name'] == 'person']
        bbox = person_detections[['xmin','ymin','xmax','ymax']].to_numpy()
        num_persons = bbox.shape[0]

        for i in range(0,num_persons):
            b = bbox[i]
            x1 = int(b[0])
            y1 = int(b[1])
            x2 = int(b[2])
            y2 = int(b[3])
            cv2.imwrite(os.path.join(DATADIR,"{}_{}_{}.png".format(camera,idx_str,i)),img[y1:y2,x1:x2])
