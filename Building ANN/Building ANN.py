# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:11:53 2020

@author: Sai Kumar
"""

#Building ANN

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#dataset.columns
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X1 = LabelEncoder()#For Country names(Not Ordinal data)
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])

labelencoder_X2 = LabelEncoder()#For Country names
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
#Creating Dummy Variables for the country col
#onehotencoder = OneHotEncoder(categories = [1])
#X = onehotencoder.fit_transform(X).toarray()
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

#Taking from the 2nd col
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 101)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Making an ANN
#Import required packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #Applied to neurons so that some of them are randomly disabled

#When we have overfitting, apply dropout to all the layers


#Initialise the ANN
classifier = Sequential()

#Adding the input layer and the 1st hidden layer
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))#Tip: # Hidden layers = Avg. of the #i/p layers & o/p layers(11+1/2)
classifier.add(Dropout(p = 0.1)) #0.1=10%---dropping 1 neuron

#Adding second hidden layer
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling an ANN(Applying Stochastic Gradient Descent)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Convert the probabilities we got in y_pred to True/False or 1/0(predicted Results)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Predicting a new observation
#Geography: 'France', Credit score:600, Gender:Male, Age:40, Tenure:3, Balance:60000
#No. of Products:2 , Has Credit Card: Yes, Is Active Member: Yes, Estimated Salary:50000
new_pred = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred>0.5)


#Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#KerasClassifier basically requires a function to be its argument
def building_classifier():
    #Initialise the ANN
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))#Tip: # Hidden layers = Avg. of the #i/p layers & o/p layers(11+1/2)
    classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#Create a global classifier as we built in the fn
classifier = KerasClassifier(build_fn = building_classifier, batch_size = 10, epochs = 100)
#Accuracies in a list(10 fold)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#n_jpbs-> #CPUs to use (-1 is to use all CPUs)
mean = accuracies.mean()
variance = accuracies.std()

#Tuning ANN
#Drop Out Regularization used to overcome the overfitting in DL
#Implementing Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

#KerasClassifier basically requires a function to be its argument
def building_classifier(optimizer):
    #Initialise the ANN
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))#Tip: # Hidden layers = Avg. of the #i/p layers & o/p layers(11+1/2)
    classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#Create a global classifier as we built in the fn
classifier = KerasClassifier(build_fn = building_classifier)

#GridSearch
hyperparams = {'batch_size' : [25,32],
               'epochs' : [100, 250],
               'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator= classifier,
                           param_grid = hyperparams,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_


#{'batch_size': 25, 'epochs': 250, 'optimizer': 'rmsprop'}
#Best_accuracy : 0.845















