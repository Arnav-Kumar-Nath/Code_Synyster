#Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset=pd.read_csv('50_Startups.csv')

#Import data values
#Excluding last column
X=dataset.iloc[:,:-1].values

#Column[4]:Profit is taken as dependent data 
y=dataset.iloc[:,4].values

#Encoding the state categorical data only as it
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#Change the independent data only as other columns are not categories
labelencoder_X=LabelEncoder()

#Fit the LabelEncoder into State column=[3] and Transform it
X[:,3]=labelencoder_X.fit_transform(X[:,3])        
#States are changed into 0,1,2 

#--------------------------------------------------------------------------------------------

#Create dummy variables so that machine doesnt think states are given priority 
#and give everyone same priority
#Applying OneHotEncoder to column =[3]
onehotencoder=OneHotEncoder(categorical_features=[3])   
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
#Removed the first column of X 
X=X[:,1:]


#Splitting the data set into train and test set
from sklearn.cross_validation import train_test_split

#50 observations( let 10 obs to test set and 40 to train set)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling(Library would do it manually)
#Three Double Quotes are used to save as comments
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#Fitting MLR to the training Set

from sklearn.linear_model import LinearRegression

#Created an object 'regressor' of the class 'LinearRegression'
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test Set Results
y_pred=regressor.predict(X_test)

#---------------------CHECKING IMPACTFUL INDEPENDENT VARIABLE USING BACKWARD ELIMINATION-----------------------------------------------------------------------

#Building the optimal model using Backward Elimination
#To check which independent value has greater impact
#What happens if we remove the less impact variables
import statsmodels.formula.api as sm

#In the formula ,b0 is constant and x0 is considered equal to 1
#y=b0+b1*x1+b2*x2+........
#In the regression b0 is not considered 
#So we have to consider b0 somehow in the model
#So we have to add a column of 1s before
#Array name=X, Np created a matrix of 1s of 50 rows and 1 column
#astype = int specifies it integer value because of datatype error
#To add it in the beginning, let us append X and reverse the process
#X=np.append(arr=X,values=np.ones((50,1)).astype(int),axis=1)
X= np.append(arr=np.ones((50,1)).astype(int),values= X,axis=1)


#Optimal X which dominates or have more impact
"""Select all independent variable and remove them one by one which are not impactful"""
#Step 1: Select a significance level to stay in the model (SL=0.5)
X_opt=X[:,[0,1,2,3,4,5]]

#Step2 : Fit the full model with all possible predictors:using Ordinary Least Square Class 
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()

#Step3 : Consider the predictor with highest P value
#(if P>SL=0.5, continue or else model is ready)
#Less P value , more significant or impactful by using a function "Summary()"
regressor_OLS.summary()
#constant has index0, x1 has column[1]
#-------------------------------------------------------------------------------
#x2 has highest p value, so it will be remove in  X_opt
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#x1 =column[1]is the first dummy variable and has highest p variable

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#x2=column[4] from remaining has highest p value which shows [4]
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#X2=column[5] from remaining has highest p value

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Both have 0 p value 