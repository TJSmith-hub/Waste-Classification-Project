#import required libraries
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

#dataset class
class dataset:
    classes = ["cardboard","glass","metal","paper","plastic","trash"]
    original_images = []
    original_labels = []
    new_images = []
    new_labels = []

    #contructor
    def __init__(self, mode):
        #load original images at full resolution and label
        if mode == "original":
            print("Loading images...")
            i = 0
            folder = "o_waste_images"
            subfolders = os.listdir(folder)
            for subf in subfolders:
                for filename in os.listdir(os.path.join(folder,subf)):
                    img = Image.open(os.path.join(folder,subf,filename))
                    self.original_images.append(np.asarray(img))
                    self.original_labels.append(i)
                i += 1
                print(subf)
        #load saved modified image dataset
        elif mode == "new":
            print("Loading dataset...")
            folder = "waste_dataset/images"
            #load images
            for filename in os.listdir(folder):
                img = Image.open(os.path.join(folder,filename))
                self.new_images.append(np.asarray(img))
            #load labels
            with open("waste_dataset/waste_labels.txt", 'r') as myfile:
                reader = csv.reader(myfile)
                self.new_labels = list(map(int, next(reader)))

    #get dataset images and labels
    def get_dataset(self):
        return self.new_images, self.new_labels

    #save modified images and labels
    def save_dataset(self):
        print("Saving", len(self.new_images), "images")
        #save labels as csv file
        with open("waste_dataset/waste_labels.txt", 'w', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow(self.new_labels)
        i = 0
        #save images to one folder
        for image in self.new_images:
            temp = Image.fromarray(image)
            temp.save(os.path.join("waste_dataset/waste_dataset","image" + str(i) + ".jpg"))
            i += 1
    
    #resize all images to spesified resolution
    def resize_images(self, res):
        for image in self.original_images:
            temp = Image.fromarray(image)
            temp = temp.resize((res[0],res[1]))
            self.new_images.append(np.asarray(temp))
        self.new_labels = self.original_labels.copy()
        print(np.shape(self.new_images))

    #perform a left to right flip of all images and append to dataset
    def add_flipped_images(self):
        n = len(self.new_images)
        print("Flipping",n,"images")
        for i in range(0,n):
            temp = Image.fromarray(self.new_images[i])
            temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
            self.new_images.append(np.asarray(temp))
            self.new_labels.append(self.new_labels[i])

    #permorm rotation of all images 3 times and append to dataset
    def add_rotated_images(self):
        n = len(self.new_images)
        print("Rotating",n,"images")
        for i in range(0,n*3):
            temp = Image.fromarray(self.new_images[i])
            temp = temp.transpose(Image.ROTATE_90)
            self.new_images.append(np.asarray(temp))
            self.new_labels.append(self.new_labels[i])

    #perform a 3 flips of all images and append to dataset
    def add_flipped_images2(self):
        n = len(self.new_images)
        print("Flipping",n,"images")
        for i in range(0,n):
            temp = Image.fromarray(self.new_images[i])
            flip1 = temp.transpose(Image.FLIP_LEFT_RIGHT)
            self.new_labels.append(self.new_labels[i])
            self.new_images.append(np.asarray(flip1))
            flip2 = temp.transpose(Image.FLIP_TOP_BOTTOM)
            self.new_images.append(np.asarray(flip2))
            self.new_labels.append(self.new_labels[i])
            flip3 = flip2.transpose(Image.FLIP_LEFT_RIGHT)
            self.new_images.append(np.asarray(flip3))
            self.new_labels.append(self.new_labels[i])

    #plot example 5x5 grid of images
    def plot(self):
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.new_images[i*100])
            plt.xlabel(self.classes[self.new_labels[i*100]])
        plt.show()