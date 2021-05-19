import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lime import lime_text
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
#uses PyTorch


#We can't use our initial dataset (3GB), too heavy, so instead... :
file_name="clothes.csv"


ds = pd.read_csv(file_name)
example=ds.iloc[15,4] #Getting the right column (4), i.e. the one with the review


classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True)
tokenizer=classifier.tokenizer
model=classifier.model

#Explaining using Sequence Classification Explainer from the transformers explainer
from transformers_interpret import SequenceClassificationExplainer
cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)
word_attributions = cls_explainer(example)
cls_explainer.visualize("visualize.html")