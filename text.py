import pandas as pd
import numpy as np
from lime import lime_text
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#uses PyTorch

device="cuda"
torch.device(device)

#We can't use our initial dataset (1GB), too heavy, so instead... :
file_name="Reviews.csv"


ds = pd.read_csv(file_name)
ds=ds[1:100]
example=ds.iloc[56,9] #Getting the right column, i.e. the one with the review

#Importing the model
model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model=model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# With both the model and tokenizer initialized we are now able to get explanations on an example text.


#Explaining using Sequence Classification Explainer from the transformers explainer
from transformers_interpret import SequenceClassificationExplainer
cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)
word_attributions = cls_explainer(example)
cls_explainer.visualize("visualize.html")

#Explaining using LIME:
from lime.lime_text import LimeTextExplainer
def predictor(texts):
	outputs = model(**tokenizer(texts, return_tensors="pt", padding=True).to(device))
	probas = F.softmax(outputs.logits).detach().numpy()
	return probas

class_names=['positive','negative','neutral']
explainer = LimeTextExplainer(class_names=class_names)

exp=explainer.explain_instance(example,predictor,num_features=20, num_samples=2000)
