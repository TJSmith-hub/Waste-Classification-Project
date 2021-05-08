import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from PIL import Image
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import process_data as pd

dataset = pd.dataset("original")
dataset.resize_images([64,64])
dataset.add_flipped_images2()
#dataset.save_dataset()
#dataset.plot()

waste_images, waste_labels = dataset.load_images()

waste_images = np.divide(waste_images,255.0)
train_images,test_images,train_labels,test_labels = train_test_split(waste_images, waste_labels,test_size=0.3, random_state=0)

train_images = tf.convert_to_tensor(train_images)
train_labels = tf.convert_to_tensor(train_labels)
test_images = tf.convert_to_tensor(test_images)
test_labels = tf.convert_to_tensor(test_labels)

print(len(waste_images), "total images")

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=train_images[0].shape))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.9))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(6, activation='softmax'))
model.summary()

#set up step decay function to decay the learning rate
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.9
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	print(lrate)
	return lrate
lrate = callbacks.LearningRateScheduler(step_decay)
#set up early stoping
es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
#add callbacks
callbacks.CallbackList = []

model.compile(optimizer=optimizers.SGD( learning_rate=0.001, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

start = time.time()
history = model.fit(train_images, train_labels, epochs=800, validation_data=(test_images, test_labels), callbacks=callbacks.CallbackList, verbose=1)
print("Training time : ", time.time() - start)

if (os.path.exists('models/WasteNet.h5')):
	os.remove('models/WasteNet.h5')
model.save('models/WasteNet.h5')

test_loss, test_acc = model.evaluate(test_images,  test_labels)
print(test_acc)

predictions = model.predict(test_images)

y_pred = np.argmax(predictions,axis=1)

print(classification_report(test_labels, y_pred, target_names=dataset.classes))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


c_matrix = confusion_matrix(test_labels, y_pred, normalize='true')
plt.matshow(c_matrix)
for (i, j), z in np.ndenumerate(c_matrix):
    plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', c='white')
plt.xticks(range(6),dataset.classes)
plt.yticks(range(6),dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#------------------------------Prediction Visualisation from---------------------------------
#https://www.tensorflow.org/tutorials/keras/classification
#modified to work with this particular code

def plot_image(i, predictions_array, true_label, img):
	true_label, img = true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(dataset.classes[predicted_label],100*np.max(predictions_array),dataset.classes[true_label]),color=color)

def plot_value_array(i, predictions_array, true_label):
	true_label = true_label[i]
	plt.grid(False)
	plt.xticks(range(6))
	plt.yticks([])
	thisplot = plt.bar(range(6), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, 2*i+1)
	plot_image(i, predictions[i], test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2*i+2)
	plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
#--------------------------------------------------------------------------------------