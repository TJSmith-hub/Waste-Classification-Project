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
    def __init__(self):
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

    def load_images(self):
        return self.new_images, self.new_labels

    def save_dataset(self):
        #save labels as csv file
        with open("waste_labels", 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(self.new_labels)
        i = 0
        for image in self.new_images:
            temp = Image.fromarray(image)
            temp.save(os.path.join("waste_dataset","image" + str(i) + ".jpg"))
            i += 1
    
    def resize_images(self, x, y):
        for image in self.original_images:
            temp = Image.fromarray(image)
            temp = temp.resize((x,y))
            self.new_images.append(np.asarray(temp))
        self.new_labels = self.original_labels.copy()
        print(np.shape(self.new_images))
        
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