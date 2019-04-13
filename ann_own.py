# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:08:49 2019

@author: Arnav Kumar Nath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values   #All index from 3 to 12            #Row no and customer id and surname has no impact so not included in independent variable.
y = dataset.iloc[:, 13].values                 #If the customer has exited is the dependent variable we need to find from the x values
                                            #ANN will find the independent variables which will have more impact on the 'exit of the customer'

#Encode the categorical values like spain, germany into numerical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#For geography :
labelencoder_X_1 = LabelEncoder()
X[:, 1 ]         = labelencoder_X_1.fit_transform(X[:, 1])    #Geography and Gender are the categorical values
                 
 #here 1 and 2 are the index of the columns
#Categories are not ordinal i.e Spain not greater than Germany so we do one hot encoder
#For Gender : No need to create dummy variable as it will remove one column 
labelencoder_X_2 = LabelEncoder()
X[:, 2 ]         = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder  =  OneHotEncoder(categorical_features = [1])    #1 for geography 
X              = onehotencoder.fit_transform(X).toarray()
X= X[: , 1:]            # To remove the dummy variable



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling is compulsory in ANN as we dont want a independent variable dominating 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Part 2 - Let us make our ANN 

# Importing Keras Library and packages
import keras 

#To inialise our network and create the layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initialising the ANN by creating a classfier as it classifies
classifier = Sequential()


#Adding Layers :
                  #output_dim = no of nodes in our hidden layers(11+1/2 as  output produce 2 binary value)   #input_dim = no of independent variables

#Create the first hidden layer
classifier.add = (Dense (output_dim = 6, 
                         init       = 'uniform', 
                         activation = 'relu', 
                         input_dim  = 11))
                        #" uniform init " means "uniform" weight distribution
                        #rectifier function for hidden layers as it rectifies
                        
classifier.add(Dropout(p = 0.1 ))     #INPUT_DIM is compulsory for first hidden layer
#p = fraction of neurons we want to drop or remove



#Creating the second hidden layer
classifier.add = (Dense (output_dim = 6, 
                         init       = 'uniform', 
                         activation = 'relu'))
classifier.add(Dropout(p = 0.1 ))



#Creating the output layer
#Sigmoid function is best for output layer as it gives probability of whether to leave or stay
#Output_dim = 1 as output will be whether yes or no (0 for stays and 1 for exits)
classifier.add = (Dense (output_dim = 1, 
                         init       = 'uniform', 
                         activation = 'sigmoid'))




#Compiling the ANN
classifier.compile(optimizer = 'adam',                      #adam optimiser is a algorithm for stochastic gradient descent
                   loss      = 'binary_crossentropy',
                   metrics   = ['accuracy'])      #Binary for only two outcomes(1 or 0) and we get a logarithmic loss
                                                #catgorical cross entropy for more than 2 values
                                                #Accuracy metrics to increase accuracy every time we run
                                                
#Fitting the ANN to the training set
                                                
#epoch means number of training it receives 
#batch_size means number of observations after which  we want to update                                           
classifier.fit(X_train,
               y_train, 
               batch_size = 10,
               nb_epoch   = 100)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Setting the threshold range so that it returns True if y_pred > 0.5
y_pred = (y_pred > 0.5)     
# We get the result in the form of True and False
----------------------------------------------------
#Predicting a single new observation
"""geography = France
Credit = 600
Gender = Male
Age= 40
Tenure= 3
balance = $60000
Products = 2
Credit Card = Yes
Active = Yes
Estimated Salary = $50000"""
#To add in horizontal we add double bracket
#first two are dummy variables
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


 



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#If y>0.5 then the customer is leaving



#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequencial
from keras.layers import Dense
#It requires a function that will build the architecture of the ANN
def build_classifier():
    classifier = Sequential()
    classifier.add = (Dense (output_dim = 6, 
                             init       = 'uniform', 
                             activation = 'relu', 
                             input_dim  = 11))
    
    classifier.add = (Dense (output_dim = 6, 
                             init       = 'uniform', 
                             activation = 'relu'))
    
    classifier.add = (Dense (output_dim = 1, 
                             init       = 'uniform', 
                             activation = 'sigmoid'))
    
    classifier.compile(optimizer = 'adam',                      
                       loss      = 'binary_crossentropy',
                       metrics   = ['accuracy'])
    return classifier
    
#Global Clasifier which will be built on k fold cross validation

classifier = KerasClassifier(build_fn = build_classifier, 
                             batch_size = 10,
                             nb_epoch = 100)

#k=10 fold returns relevant measurement of accuracies i.e 10 accuracies for 10 batches
accuracies = cross_val_score(estimator = classifier,
                             X      = X_train,
                             y      = y_train,
                             cv     = 10, 
                             n_jobs = -1)   
    #cv means value of k
    #n_jobs means no of CPUs used to compute (-1 means all) and run parallel computation
    #low bias and lowe variance means high accuracies and low difference
    #K fold gives relevant results and take the means 
    
mean     = accuracies.mean()
variance = accuracies.std()

#Dropout regularisation to reduce overfitting which reduces performances and we have high accuracies in train set than test set
#Added berfore
#Tuning the ANN
#grid search returns the best paramater from k fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn_model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add = (Dense (output_dim = 6, 
                             init       = 'uniform', 
                             activation = 'relu', 
                             input_dim  = 11))
    
    classifier.add = (Dense (output_dim = 6, 
                             init       = 'uniform', 
                             activation = 'relu'))
    
    classifier.add = (Dense (output_dim = 1, 
                             init       = 'uniform', 
                             activation = 'sigmoid'))
    
    classifier.compile(optimizer = 'adam',                      
                       loss      = 'binary_crossentropy',
                       metrics   = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)


#Create a dictionary for the hyper paramters we want to optimise
parameters = {'batch_size' : [25, 32],
              'nb_epoch'   : [100, 500],
              'optimizer'  : ['adam', 'rmsprop']}

#Creating the grid search object
grid_search = GridSearchCV(estimator  = classifier,
                           param_grid = parameters,
                           scoring    = 'accuracy', 
                           CV         = 10)

#Fitting GRID Serach in our model
grid_search = grid_search.fit(X_train, y_train)


#Finding the best
best_parameters = grid_search.best_params_
best_accuracy   = grid_search.best_score_

#Execute the first part and tuning part 