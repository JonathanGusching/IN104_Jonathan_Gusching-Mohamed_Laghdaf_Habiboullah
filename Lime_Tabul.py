import pandas as pd
import numpy as np
import lightgbm as lgb
import lime
import lime.lime_tabular
# For converting textual categories to integer labels 
from sklearn.preprocessing import LabelEncoder

# for creating train test split
from sklearn.model_selection import train_test_split

train_df=pd.read_csv('US_Accidents_Dec20_Updated.csv')

# data preparation

train_df.rename(columns={ 'Temperature(F)': 'Temperature', 'Wind_Chill(F)':'Wind_Chill', 'Distance(mi)':'Distance','Humidity(%)':'Humidity','Pressure(in)':'Pressure','Wind_Speed(mph)':'Wind_Speed', 'Visibility(mi)':'Visibility'},inplace=True)
train_df = train_df.replace(to_replace='None', value=np.nan).dropna()
train_df['target']=np.where(train_df['Severity']==4, 0, 1)
train_df["Bump"] = train_df["Bump"].astype(int)
features= ['Temperature','Wind_Chill','Humidity','Pressure','Wind_Speed','Visibility','Bump','City_le']
target = ['target']
# label encoding textual data

le = LabelEncoder()
train_df['City_le']=le.fit_transform(train_df['City'])

# using train test split to create validation set

X_train, X_test, y_train, y_test = train_test_split(
    train_df[features], train_df[target], test_size = 0.3)

# def lgb_model(X_train,y_train,X_test,y_test,lgb_params):
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)


# specify configurations as a dict

lgb_params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric':'binary_logloss',
    'metric': {'l2', 'auc'},
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': None,
    'num_iteration':100,
    'num_threads':7,
    'max_depth':12,
    'min_data_in_leaf':100,
    'alpha':0.5}

# training the lightgbm model

model = lgb.train(lgb_params,lgb_train,num_boost_round=100,valid_sets=lgb_eval,early_stopping_rounds=20)


# this is required as LIME requires class probabilities in case of classification example
# LightGBM directly returns probability for class 1 by default 
def prob(data):
    return np.array(list(zip(1-model.predict(data),model.predict(data))))
    
# asking for explanation for LIME model

explainer = lime.lime_tabular.LimeTabularExplainer(train_df[model.feature_name()].astype(int).values,  
mode='classification',training_labels=train_df[target],feature_names=model.feature_name())

i = 1
exp = explainer.explain_instance(train_df.loc[i,features].astype(int).values, prob, num_features=4)

exp.show_in_notebook()#We used here jupyter notebook to show it 
