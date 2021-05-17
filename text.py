import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#We can't use our initial dataset (1GB), too heavy, so instead... :
file_name="Reviews.csv"


ds = pd.read_csv(file_name)
ds=ds.iloc[56,9] #Getting the right column, i.e. the one with the review

#Importing the model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# With both the model and tokenizer initialized we are now able to get explanations on an example text.


#Explaining using Sequence Classification Explainer from the transformers explainer
from transformers_interpret import SequenceClassificationExplainer
cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)
word_attributions = cls_explainer(ds)
cls_explainer.visualize("visualize.html")

#Other explainability methods