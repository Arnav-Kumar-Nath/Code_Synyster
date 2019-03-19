

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


#Fitting the regressor
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


y_pred=regressor.predict(6.5)






#Visualizing the Decision Tree Results
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
#Plot the y values as predicted values of X_grid 
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff(Decision Tree Regression)')
plt.xlabel('Position Level' )
plt.ylabel('Salary')
plt.show()