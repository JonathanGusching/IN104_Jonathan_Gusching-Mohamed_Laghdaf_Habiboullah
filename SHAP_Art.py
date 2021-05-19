import keras
import shap
import tensorflow as tf
import numpy as np
import skimage.io 
import skimage.segmentation

#First version of SHAP, THE RESULTS AREN'T EXTREMELY SATISFYING.
#FOR A BETTER VERSION SEE SHAP_Art_Deep.py


#Loading our model
model = keras.models.load_model("image_model.h5")

X = skimage.io.imread("dessin2.jpg")
X=skimage.transform.resize(X, (164,164))

Z = skimage.io.imread("saint-michel.jpg")
Z=skimage.transform.resize(Z, (164,164))


class_names=np.array(["drawings","engraving","iconography","painting"])
# define a masker that is used to mask out partitions of the input image, this one uses a blurred background
masker = shap.maskers.Image("inpaint_telea", X.shape)

# By default the Partition explainer is used for all  partition explainer
explainer = shap.Explainer(model, masker)#, output_names=class_names)
Y=np.array([X,Z])
# here we use 500 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer(Y, max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])
#print(class_names.shape)
shap.image_plot(shap_values)#,labels=class_names)
