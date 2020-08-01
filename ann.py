import pandas as pd
import numpy as np
import tensorflow as tf

#   Part1 - Pre-processing.---------------------------------------------------------------------------------------------

'''Importing the dataset'''
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#   Encoding categorical data

#   LabelEncoding the Gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#   OneHotEncoding the 'Geograhy' column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#   Splitting dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#   Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#   Part2 - Building the Artificial Neural Network.---------------------------------------------------------------------

#   Initializing the ANN
ann = tf.keras.models.Sequential()

#   Adding the input and first hidden layer
ann. add(tf.keras.layers.Dense(units=6, activation='relu'))

#   Adding the second hidden layer
ann. add(tf.keras.layers.Dense(units=6, activation='relu'))

#   Adding the output layer
ann. add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#   Part3 - Training the Artificial Neural Network.---------------------------------------------------------------------

#   Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#   Training the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#   Part4 - Making Prediction and Evaluating the model.-----------------------------------------------------------------

#   Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1))

#   Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)