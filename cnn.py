#################################################
#				CODE INSPIRED BY				#
#https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
#TODO: MAKE THE SAVING SYSTEM WORK BETTER!
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os
import joblib
import numpy as np


labels = ['drawings', 'engraving']#,'iconography', 'painting', 'sculpture']
img_size = 224

#Function getting the images from different folders
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data,dtype=object)

if __name__ == '__main__':
	train=get_data('train')
	val=get_data('val')
	filename='image_model.h5'
	nb_it=20
	#loading the model
	try:
		model = keras.models.load_model(filename)
	except:
		print("File does not exist")
	l=[]
	for i in train:
		if(i[1]==0):
			l.append("drawings")
		else:
			l.append("engraving")
	sns.set_style('darkgrid')
	sns.countplot(x=l)

	#Show a drawings image
	plt.figure(figsize = (5,5))
	plt.imshow(train[1][0])
	plt.title(labels[train[0][1]])

	#Show an engraving one
	plt.figure(figsize = (5,5))
	plt.imshow(train[-1][0])
	plt.title(labels[train[-1][1]])

	#plt.show()

	#DATA PREPROCESSING

	x_train = []
	y_train = []
	x_val = []
	y_val = []

	for feature, label in train:
	  x_train.append(feature)
	  y_train.append(label)

	for feature, label in val:
	  x_val.append(feature)
	  y_val.append(label)

	# Normalize the data
	x_train = np.array(x_train) / 255
	x_val = np.array(x_val) / 255

	x_train.reshape(-1, img_size, img_size, 1)
	y_train = np.array(y_train)

	x_val.reshape(-1, img_size, img_size, 1)
	y_val = np.array(y_val)

	datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


	datagen.fit(x_train)

	model = Sequential()
	model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
	model.add(MaxPool2D())

	model.add(Conv2D(32, 3, padding="same", activation="relu"))
	model.add(MaxPool2D())

	model.add(Conv2D(64, 3, padding="same", activation="relu"))
	model.add(MaxPool2D())
	model.add(Dropout(0.4))

	model.add(Flatten())
	model.add(Dense(128,activation="relu"))
	model.add(Dense(2, activation="softmax"))

	model.summary()

	opt = Adam(lr=0.000001) #optimizer
	model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
	history = model.fit(x_train,y_train,epochs = nb_it , validation_data = (x_val, y_val))
	#epochs : 500 default
	#plot
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(nb_it)
	#Save the model to the disk
	model.save(filename)


	plt.figure(figsize=(15, 15))
	plt.subplot(2, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(2, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()