import keras
import shap
import tensorflow as tf
import numpy as np
import cv2
import skimage.io 
import skimage.segmentation

#Based on the Github documentation https://github.com/slundberg/shap
#To use it, you need a few jpg images (We used, as commended by the documentation, 100 images) in the folder PATH
#PATH OF THE IMAGES USED BY DEEP EXPLAINER
PATH="expl/*.jpg"

#TWO TEST IMAGES
img1="dessin2.jpg"
img2="saint-michel.jpg"


#default size used by our model
img_size=164

#The labels in the order they appear in the plot
labels = ['drawings', 'engraving','iconography', 'painting']


#Loading our model:
model = keras.models.load_model("image_model.h5")

#test pictures
X = skimage.io.imread(img1)
Z = skimage.io.imread(img2)

#For the model
X=skimage.transform.resize(X, (img_size,img_size))
Z=skimage.transform.resize(Z, (img_size,img_size))

x_test=np.array([X,Z])
# select a set of background examples to take an expectation over

import glob
background = []
for img in glob.glob(PATH):
    n= cv2.imread(img)
    n = cv2.resize(n, (img_size, img_size))
    background.append(n)
background=np.array(background)

# explain predictions of the model on our images
e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(x_test,check_additivity=False)


# plot the feature attributions
shap.image_plot(shap_values, x_test)