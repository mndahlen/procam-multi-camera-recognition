import numpy as np
import pandas as pd
import cv2
import os

"""
Augment data using some standard augmentations.
"""

def flip_horizontal(image):
    return np.flip(image, axis=1)

def add_noise(image, mean, sigma):
    gauss = np.abs(np.random.normal(mean,sigma,img.shape))
    noisy = np.uint8(image + gauss)

    return noisy

def resize_with_padding(img, expected_size):
    im_height = img.shape[0]
    im_width = img.shape[1]

    if im_height > expected_size[0]:
        scale_factor = expected_size[0]/im_height
        im_height = int(im_height*scale_factor)
        im_width = int(im_width*scale_factor)
        img = cv2.resize(img, (im_width,im_height), interpolation = cv2.INTER_AREA)
    
    delta_width = expected_size[1] - im_width
    delta_height = expected_size[0] - im_height
    pad_width = delta_width // 2
    pad_height = delta_height // 2

    side1 = np.zeros((im_height, pad_width,3))
    side2 = np.zeros((im_height, delta_width - pad_width,3)) 
    top1 = np.zeros((pad_height, expected_size[0],3))
    top2 = np.zeros((delta_height - pad_height, expected_size[0],3))

    padded = np.concatenate((side1,img,side2), axis = 1)
    padded = np.concatenate((top1,padded,top2), axis = 0)

    return padded

def block_image(img, type_):
    if type_ == "top_half":
        img[0:img.shape[0]//2, :, :] = 0
    elif type_ == "bottom_half":
        img[img.shape[0]//2:img.shape[0], :, :] = 0
    elif type_ == "left_half":
        img[:, 0:img.shape[1]//2, :] = 0
    elif type_ == "right_half":
        img[:, img.shape[1]//2:img.shape[1], :] = 0
    return img

if 0:
    IMGDIR_LOAD = "hallway_1192/persons"
    IMGDIR_SAVE = "hallway_1192_cropped/persons"

    labels = pd.read_csv("hallway_1192/data_labels.csv")
    for index, row in labels.iterrows():
        imname = row['imname'][:-4]
        label = row['label']
        img = cv2.imread(os.path.join(IMGDIR_LOAD,imname + ".png"))
        print(imname)

        imname_nothing = imname + ".png"
        imname_flipped = imname + "_horizontal_flip.png"
        imname_noise = imname + "_gaussian_0_30.png"
        imname_top_half = imname + "_top_half.png"
        imname_bottom_half = imname + "_bottom_half.png"
        imname_left_half = imname + "_left_half.png"
        imname_right_half = imname + "_right_half.png"

        img_nothing = resize_with_padding(img,(224,224))
        img_flipped = resize_with_padding(flip_horizontal(img),(224,224))
        img_noise = resize_with_padding(add_noise(img, 0, 30),(224,224))
        img_top_half = resize_with_padding(block_image(img.copy(), "top_half"),(224,224))
        img_bottom_half = resize_with_padding(block_image(img.copy(), "bottom_half"),(224,224))
        img_left_half = resize_with_padding(block_image(img.copy(), "left_half"),(224,224))
        img_right_half =  resize_with_padding(block_image(img.copy(), "right_half"),(224,224))

        labels = labels.append({'imname': imname_flipped, 'label': label}, ignore_index=True)
        labels = labels.append({'imname': imname_noise, 'label': label}, ignore_index=True)
        labels = labels.append({'imname': imname_top_half, 'label': label}, ignore_index=True)
        labels = labels.append({'imname': imname_bottom_half, 'label': label}, ignore_index=True)
        labels = labels.append({'imname': imname_left_half, 'label': label}, ignore_index=True)
        labels = labels.append({'imname': imname_right_half, 'label': label}, ignore_index=True)

        cv2.imwrite(os.path.join(IMGDIR_SAVE,imname_nothing),img_nothing)
        cv2.imwrite(os.path.join(IMGDIR_SAVE,imname_flipped),img_flipped)
        cv2.imwrite(os.path.join(IMGDIR_SAVE,imname_noise),img_noise)
        cv2.imwrite(os.path.join(IMGDIR_SAVE,imname_top_half),img_top_half)
        cv2.imwrite(os.path.join(IMGDIR_SAVE,imname_bottom_half),img_bottom_half)
        cv2.imwrite(os.path.join(IMGDIR_SAVE,imname_left_half),img_left_half)
        cv2.imwrite(os.path.join(IMGDIR_SAVE,imname_right_half),img_right_half)

    labels.to_csv("hallway_1192_cropped/data_labels_cropped.csv", index=False)
else:
    print("Disabled!")