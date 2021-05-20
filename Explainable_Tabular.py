import pandas as pd
import xgboost
from xgboost import XGBClassifier
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

#Loading the data

train_df=pd.read_csv('US_Accidents_Dec20_Updated.csv')
train_df=train_df.iloc[:1000,] # For the ALE method because it took a lot of time for executing
#Choosing features and encoding them

train_df.rename(columns={ 'Temperature(F)': 'Temperature', 'Wind_Chill(F)':'Wind_Chill', 'Distance(mi)':'Distance','Humidity(%)':'Humidity','Pressure(in)':'Pressure','Wind_Speed(mph)':'Wind_Speed', 'Visibility(mi)':'Visibility'},inplace=True)
train_df = train_df.replace(to_replace='None', value=np.nan).dropna()
train_df['target']=np.where(train_df['Severity']==4, 0, 1)
train_df["Bump"] = train_df["Bump"].astype(int)
features= ['Temperature','Wind_Chill','Humidity','Pressure','Wind_Speed','Visibility','Bump']
target = ['target']
X_train, X_test, y_train, y_test = train_test_split(
    train_df[features], train_df[target], test_size = 0.2, random_state=42)

#Training the model with XGBClassifier
classifier = XGBClassifier(
    max_depth=12,
    subsample=0.33,
    objective='binary:logistic',
    n_estimators=100,
    learning_rate = 0.01)
eval_set = [(X_train,y_train), (X_test,y_test)]

classifier.fit(
    X_train, y_train.values.ravel(), 
    early_stopping_rounds=12, 
    eval_metric=["error", "logloss"],
    eval_set=eval_set, 
    verbose=True
)
classifier.score(X_test,y_test)




#Applying SHAP for explainability
import shap

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_train)

shap_values_df = pd.DataFrame(shap_values)

shap.summary_plot(shap_values, X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar")




#Applying LIME for explainability

import lime
explainer = lime.lime_tabular.LimeTabularExplainer(X_train[features].astype(int).values,  
mode='classification',training_labels=y_train['target'],feature_names=features)

def prob(data):
    return np.array(list(zip(1-classifier.predict(data),classifier.predict(data))))

i = 1 #for the row i
exp = explainer.explain_instance(X_test.loc[i,features], prob, num_features=5)

exp.show_in_notebook(show_all=False)




#Applying ALE for explainability
from alepython import ale_plot

matplotlib.rc("figure", figsize=(9, 6))
ale_plot(
    classifier,
    X_test,
    X_test.columns[:1],
    bins=20,
    monte_carlo=True,
    monte_carlo_rep=100,
    monte_carlo_ratio=0.6,
)

matplotlib.rc("figure", figsize=(9, 6))
ale_plot(classifier, X_test, X_test.columns[1:3], bins=10)




#Applying PDP for explainability for 1 feature

#Binary feature='Bump' 
## 1.1 target distribution through feature 'Bump'
fig, axes, summary_df = info_plots.target_plot(
    df=train_df, feature='Bump', feature_name='Bump', target='target'
)
_ = axes['bar_ax'].set_xticklabels([0, 1])

#summary_df

## 1.2 check prediction distribution through feature 'Bump'
fig, axes, summary_df = info_plots.actual_plot(
    model=classifier, X=train_df[features], feature='Bump', feature_name='Bump'
)

#summary_df

## 1.3 pdp for feature 'Bump'

pdp_Bump = pdp.pdp_isolate(
    model=classifier, dataset=train_df, model_features=features, feature='Bump'
)

# default
fig, axes = pdp.pdp_plot(pdp_Bump, 'Bump')
_ = axes['pdp_ax'].set_xticklabels([1, 0])#here the plot may be empty so there is no relation with target !?


# 3. numeric feature: Temperature
## 3.1 target distribution through feature 'Temperature'

fig, axes, summary_df = info_plots.target_plot(
    df=train_df, feature='Temperature', feature_name='T', target='target', show_percentile=True
)
#summary_df

## 3.2 check prediction distribution through feature 'Temperature'
fig, axes, summary_df = info_plots.actual_plot(
    model=classifier, X=train_df[features], feature='Temperature', feature_name='T', 
    show_percentile=True
)

#summary_df


## 3.3 pdp for feature 'Temperature'
pdp_Temp = pdp.pdp_isolate(
    model=classifier, dataset=train_df, model_features=features, feature='Temperature'
)

fig, axes = pdp.pdp_plot(pdp_Temp, 'Temperature')

fig, axes = pdp.pdp_plot(pdp_Temp, 'Temperature', plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
        pdp_Temp, 'Temperature', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)
