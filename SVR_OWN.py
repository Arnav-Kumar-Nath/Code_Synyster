#Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset=pd.read_csv('Position_Salaries.csv')

#Import datas
#Only Levels Column is taken as Independent as it is same as Position
#To specifiy it as matrix not as vector we made 1:2 instead of just 1
#Always make X a matrix to avoid Error
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


#Splitting the dataset
"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)"""

#Feature Scaling
#Linear Regression has inbuilt Feature Scaling Library
#Three Double Quotes are used to save as comments
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')  #To know what we are doing use kernel
regressor.fit(X,y)
 


#Prediciting a new result
y_pred=regressor.predict(6.5)




#Visualizing the SVR Results
plt.scatter(X,y,color='red')
#Plot the y values as predicted values of X_grid 
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position Level' )
plt.ylabel('Salary')
plt.show()

#We get a blue straight horizontal line .WHY?
