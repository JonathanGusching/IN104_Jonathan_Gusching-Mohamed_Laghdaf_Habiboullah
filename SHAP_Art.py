from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import json
import keras
import shap
import tensorflow as tf
import numpy as np
import skimage.io 
import skimage.segmentation

# load pre-trained model and choose two images to explain
#model = ResNet50(weights='imagenet')
#OUR MODEL:
model = keras.models.load_model("image_model.h5")

X = skimage.io.imread("mona_lisa.png")
X=skimage.transform.resize(X, (164,164))
print(X.shape) 
#def f(X):
#    tmp = X.copy()
#    preprocess_input(tmp)
#    return model(tmp)
#X, y = shap.datasets.imagenet50()

#load the ImageNet class names as a vectorized mapping function from ids to names
#url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
#with open(shap.datasets.cache(url)) as file:
#    class_names = [v[1] for v in json.load(file).values()]

#OUR CLASSES:
class_names={0:'drawings',1:'engraving',2:'iconography',3:'paintings'}   

# define a masker that is used to mask out partitions of the input image, this one uses a blurred background
masker = shap.maskers.Image("inpaint_telea", X.shape)
print(X.shape)
# By default the Partition explainer is used for all  partition explainer
explainer = shap.Explainer(model, masker, output_names=class_names)
print(X.shape)
print(explainer)
Y=np.array([X])
print(Y.shape)
# here we use 500 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer([X], max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])
shap.image_plot(shap_values)
