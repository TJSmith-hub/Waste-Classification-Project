import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

class dataset:
    classes = ["cardboard","glass","metal","paper","plastic","trash"]
    original_images = []
    original_labels = []
    new_images = []
    new_labels = []
    def __init__(self, mode):
        if mode == "original":
            i = 0
            folder = "o_waste_images"
            subfolders = os.listdir(folder)
            for subf in subfolders:
                for filename in os.listdir(os.path.join(folder,subf)):
                    img = Image.open(os.path.join(folder,subf,filename))
                    self.original_images.append(np.asarray(img))
                    self.original_labels.append(i)
                i += 1
                print("Loaded images: ", subf)
        elif mode == "new":
            print("Loading dataset...")
            folder = "waste_dataset"
            for filename in os.listdir(folder):
                img = Image.open(os.path.join(folder,filename))
                self.new_images.append(np.asarray(img))
            with open("waste_labels", 'r') as myfile:
                reader = csv.reader(myfile)
                self.new_labels = list(map(int, next(reader)))

    def load_images(self):
        return self.new_images, self.new_labels

    def save_dataset(self):
        #save labels as csv file
        with open("waste_labels", 'w', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow(self.new_labels)
        i = 0
        for image in self.new_images:
            temp = Image.fromarray(image)
            temp.save(os.path.join("waste_dataset","image" + str(i) + ".jpg"))
            i += 1
    
    def resize_images(self, res):
        for image in self.original_images:
            temp = Image.fromarray(image)
            temp = temp.resize((res[0],res[1]))
            self.new_images.append(np.asarray(temp))
        self.new_labels = self.original_labels.copy()
        print(np.shape(self.new_images))

    def add_flipped_images(self):
        n = len(self.new_images)
        print("Flipping",n,"images")
        for i in range(0,n):
            temp = Image.fromarray(self.new_images[i])
            temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
            self.new_images.append(np.asarray(temp))
        print("generating labels")
        for i in range(0,n):
            self.new_labels.append(self.new_labels[i])

    def add_rotated_images(self):
        n = len(self.new_images)
        print("Rotating",n,"images")
        for i in range(0,n):
            temp = Image.fromarray(self.new_images[i])
            temp = temp.transpose(Image.ROTATE_90)
            self.new_images.append(np.asarray(temp))
            temp = temp.transpose(Image.ROTATE_90)
            self.new_images.append(np.asarray(temp))
            temp = temp.transpose(Image.ROTATE_90)
            self.new_images.append(np.asarray(temp))
        print("generating labels")
        for i in range(0,n*3):
            self.new_labels.append(self.new_labels[i])

    def add_flipped_images2(self):
        n = len(self.new_images)
        print("Rotating",n,"images")
        for i in range(0,n*3):
            temp = Image.fromarray(self.new_images[i])
            temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
            self.new_images.append(np.asarray(temp))
        print("generating labels")
        for i in range(0,n*3):
            self.new_labels.append(self.new_labels[i])


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