import pandas as pd
import transformers
import shap

#Opening the document
file_name="clothes.csv"
ds = pd.read_csv(file_name)
example=ds.iloc[15,4] #Getting the right column (4), i.e. the one with the review

n=1 #default's prediction

# load a transformers pipeline model
model = transformers.pipeline('sentiment-analysis', return_all_scores=True)

# explain the model on two sample inputs
explainer = shap.Explainer(model) 
shap_values = explainer([example])
# visualize the n-th prediction's explanation for the POSITIVE output class

shap.plots.text(shap_values[n-1, :, "POSITIVE"])