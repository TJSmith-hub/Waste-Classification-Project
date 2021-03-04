import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

cardboard_ims = load_images_from_folder("new_waste_images/cardboard")
glass_ims = load_images_from_folder("new_waste_images/cardboard")
metal_ims = load_images_from_folder("new_waste_images/cardboard")
paper_ims = load_images_from_folder("new_waste_images/cardboard")
plastic_ims = load_images_from_folder("new_waste_images/cardboard")
trash_ims = load_images_from_folder("new_waste_images/cardboard")

