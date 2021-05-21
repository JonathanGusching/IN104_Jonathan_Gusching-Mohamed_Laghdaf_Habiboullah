# IN104 Project : XAI
## Foreword :

Many libraries have been used, maybe you will need to install some of them. We have noticed that on Ubuntu, there are displaying issues when executed directly from the terminal (the ones that need HTML).

## Introduction :

In this Github repository, you will find :
- The codes used for XAI
- The codes used for training (cnn.py). You might either use it to train the CNN used in the XAI codes use the .h5 file directly (image_model.h5)
- A few images used for testing
- Requirements.txt which include which packages to import for running the code.

## Methods :

The methods used for XAI are:
1. Images :
	- LIME (LIME_Art.py)
	- SHAP (SHAP_Art.py)
	- Grad-CAM (CAM_Art.py)
2. Tabular Dataset :
	- XGB
	- SHAP
	- ALE (all of them are in the file Explainable_Tabular.py)
	- LIME (LIME_Tabul.py)
3. Text Dataset :
	- Sequence Classification Explainer (provided by the transformers library; the file is SEQ_CLASS_Text.py)
	- SHAP (SHAP_Text.py)


For the image explanability, we used a dataset of pieces of art : drawings, engravings, iconography, and paintings. 
However, the available pre-trained models weren't offering a ready-to-use DL algorithm for that. So we had to implement our own convolutional neural network thanks to tutorials (cf cnn.py).

## HOW TO DO:
Normally, the only thing you need to download is the data sets. You can then theoretically execute the code using the usual Python way.
If you want to execute cnn.py, you need the dataset in the same folder as cnn.py
