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

def zero_pad(image, wanted_width, wanted_height):
    # Wrote this in a rush but it does its job for training on zero padded data. 
    # Most examples are successfully zero padded!

    imsize = np.shape(image)
    height = imsize[0]
    width = imsize[1]

    if width < wanted_width and height < wanted_height:
        width_diff = wanted_width - width
        height_diff = wanted_height - height

        zero_padded = np.zeros((wanted_height, wanted_width,3))
        zero_padded[int(height_diff/2):int(height_diff/2) + height, int(width_diff/2):int(width_diff/2) + width,:] = image
        return zero_padded
    elif height >= wanted_height:
        try:
            scale_factor = wanted_height/height
            resized = cv2.resize(image, (int(width*scale_factor),int(height*scale_factor)), interpolation = cv2.INTER_AREA)
            width_diff = wanted_width - width*scale_factor
            height_diff = 0

            zero_padded = np.zeros((wanted_height, wanted_width,3))
            zero_padded[int(height_diff/2):int(height_diff/2 + height), int(width_diff/2):int(width_diff/2 + width*scale_factor),:] = resized
            return zero_padded
        except:
            return image
    return image

IMGDIR = "hallway_1192_augmented_zero_padded/persons"

labels = pd.read_csv("hallway_1192/data_labels.csv")
for index, row in labels.iterrows():
    imname = row['imname'][:-4]
    label = row['label']
    img = cv2.imread(os.path.join(IMGDIR,imname + ".png"))
    print(imname)

    imname_nothing = imname + ".png"
    imname_flipped = imname + "_horizontal_flip.png"
    imname_noise = imname + "_gaussian_0_30.png"

    img_nothing = zero_pad(img,224,224)
    img_flipped = zero_pad(flip_horizontal(img),224,224)
    img_noise = zero_pad(add_noise(img, 0, 30),224,224)

    labels = labels.append({'imname': imname_flipped, 'label': label}, ignore_index=True)
    labels = labels.append({'imname': imname_noise, 'label': label}, ignore_index=True)
    cv2.imwrite(os.path.join(IMGDIR,imname_flipped),img_flipped)
    cv2.imwrite(os.path.join(IMGDIR,imname_noise),img_noise)
    cv2.imwrite(os.path.join(IMGDIR,imname_nothing),img_nothing)

labels.to_csv("hallway_1192_augmented_zero_padded/data_labels_augmented_zero_padded.csv", index=False)
