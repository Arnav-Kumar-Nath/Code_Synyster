#Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset=pd.read_csv('Salary_Data.csv')

#Import datas
#Salary is dependent on Years of Experience so  Salary=y
X=dataset.iloc[:,:-1].values   #All columns except last
y=dataset.iloc[:,1].values

#----------------------------------------------------------------------------------
#Splitting the data set into train and test set
#(30 observations: let 20 to train and 10 to test )
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
#---------------------------------------------------------------------------------------


#Feature Scaling
#Three Double Quotes are used to save as comments
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#------------------------------------------------------------------------------------------
#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#Which dataset we want to fit the regressor 
regressor.fit(X_train,y_train)

#The regressor learned the correaltion of the dependent and independent values

#Predicting the Test Set Results
#Take the regressor that we trained
#y_pred will have the predicted values of the salaries of the employees
y_pred=regressor.predict(X_test)

#----------------------------------------------------------------------------------------------
#Visualizing the training Set Results of 20 observations:

#Making a ScatterPlot of PYPLOT (X_train as X coordinate
plt.scatter(X_train,y_train,color='red')

#Making a plot between X train and Y prediction of Train Set (not Test Set)
plt.plot(X_train,regressor.predict(X_train),color='blue')


plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Predicted values is blue line and line is quite linear and hence there is linear dependencies
#Real values are red dots and hence some predictions are quite accurate

#-------------------------------------------------------------------------------------------
#Visualizing the TESTSET RESULTS of 10 observations:

#Making a ScatterPlot of PYPLOT (X_test as X coordinate)
plt.scatter(X_test,y_test,color='red')

#Making a plot between X train and Y prediction of Train Set (not Test Set) again 
#It should not change as our regressor has already trained on Training Set
#and we obtained our unique model and hence we get the same regression line
plt.plot(X_train,regressor.predict(X_train),color='blue')


plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
