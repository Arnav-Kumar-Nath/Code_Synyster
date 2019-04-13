#author: Arnav Kumar Nath


#CLASSIFICATION 

#K Nearest Neighbors

#ONLY FITTING PART IS DIFFERENT , REST IS SAME

#Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
#Taking Age and Salary as X
X=dataset.iloc[:,[2,3]].values
#Taking Purchased as y
y=dataset.iloc[:,4].values

#Splitting the sets as observations are more
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
#y doesnt need scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)



#Fitting KNN Class to our Training Set


#ONLY CHANGE THIS PART , REST SAME  
#n_neighbors=no of neighbors to use
#p=power parameter for the Minkowski metric(default=2)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)  
classifier.fit(X_train,y_train)       #LEarning the correlations so as to predict the results  

#Predicting the Test Set Results
y_pred=classifier.predict(X_test) 


#Making the Confusion Matrix
#To see if it has learnt the predictions and how many it made correct predictions
from sklearn.metrics import confusion_matrix       #Functions has _ separated so it is not a class
cm=confusion_matrix(y_test,y_pred) 

        #Ground value of y_true means y_test(real set)
        #We got only 7 wrong predictions

#Visualising the Training Set Results
#Red region who dont purchased 
#Green region who bought 
#Prediction Boundary separates the 0 and 1
from matplotlib.colors import ListedColormap
#CREATED A NEW SET TO AVOID CONFUSION AND REPLACE EASILY EVERYWHERE
X_set,y_set=X_train, y_train

#ARRAY X1,X2 ---AGE =[0] and SALARY=[1] 
#Taking min-1 and max+1 so that the points arent squuezed
#step=resolution
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop= X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1, stop= X_set[:,1].max()+1,step=0.01))


#CONTOUR BODY
#predict function is used 
plt.contour(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

#SCATTER DOTS BODY 
#With the loop, we plotted all the data points 
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i), label=j)
    
   
plt.title('K Nearest Neighbor (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



#Visualising the Test Set Results"""
#Red dots who dont purchased 
#Green dots who bought 
#Prediction Boundary separates the 0 and 1  (Left side didnt bought)
from matplotlib.colors import ListedColormap
X_set,y_set=X_test, y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop= X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1, stop= X_set[:,1].max()+1,step=0.01))

#Contour BODY
plt.contour(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())


#SCATTER DOTS BODY

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i), label=j)
    
   
plt.title('K Nearest Neighbor (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()