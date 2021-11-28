import numpy as np
import pandas as pd
import cv2
import os

def flip_horizontal(image):
    return np.flip(image, axis=1)

def add_noise(image, mean, sigma):
    gauss = np.abs(np.random.normal(mean,sigma,img.shape))
    noisy = np.uint8(image + gauss)

    return noisy



IMGDIR = "hallway_1192_augmented/persons"

labels = pd.read_csv("hallway_1192_augmented/data_labels.csv")
for index, row in labels.iterrows():
    imname = row['imname'][:-4]
    label = row['label']
    img = cv2.imread(os.path.join(IMGDIR,imname + ".png"))
    print(imname)

    imname_flipped = imname + "_horizontal_flip.png"
    imname_noise = imname + "_gaussian_0_30.png"

    img_flipped = flip_horizontal(img)
    img_noise = add_noise(img, 0, 30)
    labels = labels.append({'imname': imname_flipped, 'label': label}, ignore_index=True)
    labels = labels.append({'imname': imname_noise, 'label': label}, ignore_index=True)
    cv2.imwrite(os.path.join(IMGDIR,imname_flipped),img_flipped)
    cv2.imwrite(os.path.join(IMGDIR,imname_noise),img_noise)

labels.to_csv("hallway_1192_augmented/data_labels_augmented.csv", index=False)
