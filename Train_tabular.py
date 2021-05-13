#STEP1: Process the data and split into a training and test set
import pandas as pd

# Importing the dataset
df = pd.read_csv('US_Accidents_Dec20_Updated.csv')
del df['Astronomical_Twilight;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;']
X = df.iloc[0:10000,2:].values
y = df.iloc[0:10000, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
length=X.shape[1]
for i in range(length):
    if type(X[:,i][0])==str or type(X[:,i][0])==bool:
        labelencoder_X_ = LabelEncoder()
        X[:, i] = labelencoder_X_.fit_transform(X[:, i])
#ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
#X = ct.fit_transform(X)
#removing the dummy variable


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




#Step 2: Build the Artificial Neural Network Structure

# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential #used to initialize the NN
from keras.layers import Dense  #used to build the hidden Layers
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu', input_dim =X_train.shape[1] ))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
#classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Step 3: Train the Artificial Neural Network

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 64, epochs = 100)

#Step 4: Use the model to predict!

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
