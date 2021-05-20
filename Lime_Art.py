#Imports
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings


#Importing our model
warnings.filterwarnings('ignore') 
model = keras.models.load_model('image_model.h5') #Loading our trained model
#just in case it doesn't work, source : https://www.tensorflow.org/guide/keras/save_and_serialize


#Read  and pre-processe the image

Xi = skimage.io.imread("mona_lisa.png")
Xi = skimage.transform.resize(Xi, (164,164)) 
Xi = (Xi - 0.5)*2 #Inception pre-processing


#Predict class of input image
labels={0:'drawings',1:'engraving',2:'iconography',3:'painting'}
np.random.seed()
preds = model.predict(Xi[np.newaxis,:,:,:])
print("Probabilities: (drawing,engraving, iconography, painting)",preds)
#decode_predictions(preds)[0]
top_pred_classes = preds[0].argsort()[-5:][::-1]
#Step1 Create perturbations

superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
num_superpixels = np.unique(superpixels).shape[0]

num_perturb = 200

perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

def perturb_image(img,perturbation,segments):
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1 
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image




#Step 2: Use ML classifier to predict classes of new generated images


predictions = []
for pert in perturbations:
  perturbed_img = perturb_image(Xi,pert,superpixels)
  pred = model.predict(perturbed_img[np.newaxis,:,:,:])
  predictions.append(pred)

predictions = np.array(predictions)


#Step 3: Compute distances between the original image and each of the perturbed images and compute weights (importance) of each perturbed image

original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
kernel_width = 0.25
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function


#Step 4: Use perturbations, predictions and weights to fit an explainable (linear) model

class_to_explain = top_pred_classes[0]
simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
coeff = simpler_model.coef_[0]



num_top_features = 10
top_features = np.argsort(coeff)[-num_top_features:] 



mask = np.zeros(num_superpixels) 
mask[top_features]= True #Activate top superpixels
skimage.io.imshow(perturb_image(Xi/2+0.5,mask,superpixels) )
plt.title(labels[np.argmax(preds)])
plt.show()
