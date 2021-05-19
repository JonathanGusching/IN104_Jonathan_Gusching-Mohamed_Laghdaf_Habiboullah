import pandas as pd
import transformers
import shap


file_name="clothes.csv"


ds = pd.read_csv(file_name)
example=ds.iloc[15,4] #Getting the right column (4), i.e. the one with the review

# load a transformers pipeline model
classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True)

# explain the model on two sample inputs
explainer = shap.Explainer(classifier) 
shap_values = explainer(example)

# visualize the first prediction's explanation for the POSITIVE output class
shap.plots.text(shap_values[0, :, "POSITIVE"])
