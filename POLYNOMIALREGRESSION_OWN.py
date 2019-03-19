#Polynomial Regression for One Independent and One Dependent Value
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

#---------------------------------------------------------------------------------------------
# No need of Splitting the data set into train and test set
#To get very accurate prediction ,no need to split into sets
#We need most information as possible as observations are less
"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)"""

#Feature Scaling
#Linear Regression has inbuilt Feature Scaling Library
#Three Double Quotes are used to save as comments
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#------------------------------------------------------------------------------------

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)


#FItting Polynomial Rgression to the dataset
from sklearn.preprocessing  import PolynomialFeatures
#It will transform the original matrix to a matrix of X ^powers
poly_reg=PolynomialFeatures(degree=4)    #Upto power 4 and adding three extra term
X_poly=poly_reg.fit_transform(X)  
#It creates x0=1,1.....;x1=1,2,3,.........; (x2=1,4,9,16,.......; x3=1,8,27,......;x4=1,16,81,...these are the extra terms)

#Fitting the transformed polynomial model to our linear regression model
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


#Visualising the linear Regression Model
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level' )
plt.ylabel('Salary')
plt.show()

#We dont get much great prediction


#Visuailizing the Polynomial Model

#Making a array of values between 1-10 changing at 0.1 steps For a better curve
X_grid=np.arange(min(X),max(X),0.1)
#To reshape numpy array into matrix of 90 lines and 1 column
X_grid=X_grid.reshape((len(X_grid),1))   
plt.scatter(X,y,color='red')
#Plot the y values as predicted values of X_grid 
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level' )
plt.ylabel('Salary')
plt.show()


#Predicting a new result with Linear Regression 
#Prediction of salary of level 6.5
lin_reg.predict(6.5)        
 #Result:$330378 


#Prediciting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
#Result:$158862