
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lime import lime_text
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#uses PyTorch

device="cpu"
#torch.device(device)

#We can't use our initial dataset (1GB), too heavy, so instead... :
file_name="clothes.csv"


ds = pd.read_csv(file_name)
example=ds.iloc[15,4] #Getting the right column, i.e. the one with the review

#Importing the model
model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#model=model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name) #Note to myself: Tokenization= getting the words in the text, individually, and ignoring white spaces

# With both the model and tokenizer initialized we are now able to get explanations on an example text.


#Explaining using Sequence Classification Explainer from the transformers explainer
from transformers_interpret import SequenceClassificationExplainer
cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)
word_attributions = cls_explainer(example)
cls_explainer.visualize("visualize.html")

#FIRST ATTEMPT: Explaining using LIME:
from lime.lime_text import LimeTextExplainer
#def predictor(texts):
#	outputs = model(**tokenizer(texts, return_tensors="pt", padding=True).to(device))
#	probas = F.softmax(outputs.logits).detach().numpy() #memory problem here
#	return probas

#class_names=['positive','negative','neutral']
#explainer = LimeTextExplainer(class_names=class_names)

#exp=explainer.explain_instance(example,predictor)#,num_features=20, num_samples=2000)
## LIME END

#https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Positive%20vs.%20Negative%20Sentiment%20Classification.html

#SECOND ATTEMPT
#import eli5
#from eli5.lime import TextExplainer

#te = TextExplainer(model,random_state=42)
#te.fit(example, model.eval())
#te.show_prediction(target=example)#target_names=twenty_train.target_names)

#THIRD ATTEMPT :
#model = ClassificationModel('roberta', 'PATH_TO_FINE-TUNED_MODEL', use_cuda=True, args=args)

# define softmax function
#def softmax(x):
#    e_x = np.exp(x - np.max(x))
#    return e_x / e_x.sum(axis=0)
# define prediction function
#def predict_probs(texts):
    #predictions = model.predict(texts)
#	outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
#	x = np.array(list(outputs)[1])
#	return np.apply_along_axis(softmax, 1, x)
#class_names=['positive','negative','neutral']
#explainer = LimeTextExplainer(class_names=class_names)
# explain instance with LIME
#exp = explainer.explain_instance(example, predict_probs, num_features=6)
import transformers
import shap

# load a transformers pipeline model
model = transformers.pipeline('sentiment-analysis', return_all_scores=True)

# explain the model on two sample inputs
explainer = shap.Explainer(model) 
shap_values = explainer(["What a great movie! ...if you have no taste."])

# visualize the first prediction's explanation for the POSITIVE output class
shap.plots.text(shap_values[0, :, "POSITIVE"])
explainer.visualize("visualize.html")