#STEP1: Process the data and split into a training and test set
import pandas as pd
import numpy as np
# import pickle
import joblib
# Importing the dataset
df = pd.read_csv('US_Accidents_Dec20_Updated.csv')
df=df.iloc[:100000]


df.rename(columns={ 'Temperature(F)': 'Temperature', 'Wind_Chill(F)':'Wind_Chill', 'Distance(mi)':'Distance','Humidity(%)':'Humidity','Pressure(in)':'Pressure','Wind_Speed(mph)':'Wind_Speed', 'Visibility(mi)':'Visibility'},inplace=True)
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
df = df.replace(to_replace='None', value=np.nan).dropna()
df['target']=np.where(df['Sunrise_Sunset']=='Day', 0, 1)
dataframe=df[['Temperature','Wind_Chill','Humidity','Pressure','Severity','target','Wind_Speed','Visibility']]

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


feature_columns = []

# numeric cols
for header in ['Humidity','Temperature', 'Wind_Chill','Pressure','Wind_Speed','Severity','Visibility']:
  feature_columns.append(feature_column.numeric_column(header))


# indicator_column_names = ['Weather_Condition']
# for col_name in indicator_column_names:
#   categorical_column = feature_column.categorical_column_with_vocabulary_list(
#       col_name, dataframe[col_name].unique())
#   indicator_column = feature_column.indicator_column(categorical_column)
#   feature_columns.append(indicator_column)


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.1),
  layers.Dense(1)
])

model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(train_ds,validation_data=val_ds,epochs=100)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model,filename)
