from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import json
import keras
import shap
import tensorflow as tf
import numpy as np
import skimage.io 
import skimage.segmentation

import cv2
import os
img_size=164
#Function getting the images from different folders
def get_images(data_dir):
	data = []
	path = data_dir 
	for img in os.listdir(data_dir):
		try:
			img_arr = skimage.io.imread(os.path.join(path, img))
			resized_arr=skimage.transform.resize(img_arr, (164,164))
			data.append([resized_arr])
		except Exception as e:
			print(e)
	return data
	#return np.array(data)

labels = ['drawings', 'engraving','iconography', 'painting']
#pictures=get_data('train')
# load pre-trained model and choose two images to explain
#model = ResNet50(weights='imagenet')
#OUR MODEL:
model = keras.models.load_model("image_model.h5")

X = skimage.io.imread("dessin2.jpg")
X=skimage.transform.resize(X, (164,164))

Z = skimage.io.imread("saint-michel.jpg")
Z=skimage.transform.resize(Z, (164,164))

x_test=np.array([X,Z])
# select a set of background examples to take an expectation over
background = get_images('expl')#[np.random.choice(x_train.shape[0], 100, replace=False)]
for img in background:
	print(img.shape, "pouet pouet")
# explain predictions of the model on four images
e = shap.DeepExplainer(model, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(x_test)

# plot the feature attributions
shap.image_plot(shap_values, -x_test)