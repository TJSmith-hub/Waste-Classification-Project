import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

classes = ["cardboard","glass","metal","paper","plastic","trash"]

def load_images(folder):
    images = []
    labels = []
    i = 0
    subfolders = os.listdir(folder)
    for subf in subfolders:
        for filename in os.listdir(os.path.join(folder,subf)):
            img = Image.open(os.path.join(folder,subf,filename))
            images.append(np.asfarray(img).astype('uint8'))
            labels.append(i)
        i += 1
        print("Loaded images: ", subf)
    return images, labels

waste_images, waste_labels = load_images('new_waste_images')

print(np.shape(waste_images))

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(waste_images[i])
    plt.xlabel(classes[waste_labels[i]])
#plt.show()

waste_images = np.divide(waste_images,255.0)

train_images,test_images,train_labels,test_labels = train_test_split(waste_images, waste_labels,test_size=0.3)

train_images = tf.convert_to_tensor(train_images)
train_labels = tf.convert_to_tensor(train_labels)
test_images = tf.convert_to_tensor(test_images)
test_labels = tf.convert_to_tensor(test_labels)


print(train_images)

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)