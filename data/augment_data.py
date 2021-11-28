import numpy as np
import pandas as pd
import cv2

labels = pd.read_csv("hallway_1192/data_labels.csv")
for index, row in labels.iterrows():
    print(row['imname'], row['label'])

def flip_horizontal(image):
    return np.flip(image, axis=1)