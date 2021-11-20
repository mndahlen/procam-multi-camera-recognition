import torch
import cv2
import os
IMGDIR = "data/hallway"

# Model
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

if 1:
    # Images
    img = cv2.imread(os.path.join(IMGDIR,"c1/000002/002087.jpg"))
    results = yolo(img)

    person_detections = results.pandas().xyxy[0]
    person_detections = person_detections.loc[person_detections['name'] == 'person']
    print("test")
    bbox = person_detections[['xmin','ymin','xmax','ymax']].to_numpy()
    num_persons = bbox.shape[0]

    img_draw_bbox = img.copy()
    for i in range(0,num_persons):
        b = bbox[i]
        x1 = int(b[0])
        y1 = int(b[1])
        x2 = int(b[2])
        y2 = int(b[3])
        cv2.imwrite("person_{}_2087.png".format(i),img[y1:y2,x1:x2])
        cv2.rectangle(img_draw_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("test",img_draw_bbox)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 
