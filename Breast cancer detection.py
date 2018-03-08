import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
#importing the data
data = pd.read_csv('data.csv')
data1 = data.copy(deep = True)
#preprocessing the data and getting to know it better
#print(data1.head(10))
#print(data1.columns)
data1['diagnosis'] = data1['diagnosis'].apply(lambda x : 1 if x =='M' else 0 )

toDrop_cols = ['id', 'diagnosis' , 'Unnamed: 32']
X = data1.drop(toDrop_cols, axis=1)
y = data1['diagnosis']
#now we need to split the data into three parts train,test and validation 
X_train,X_test,y_train,y_test= train_test_split(X,y , test_size=.20,random_state=0)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=.20,random_state=0)

my_pipeline = make_pipeline(XGBClassifier(n_estimators = 1000,learning_rate = 0.05))
my_pipeline.fit(X_train,y_train)
prediction = my_pipeline.predict(X_test)
print("validation  Accuracy..: ",accuracy_score(y_test ,prediction ))
print("test acc :" ,accuracy_score(y_val, my_pipeline.predict(X_val)))

#yields a val acc of 98% and a test of 96%

#training the model using a different Algorithm 
"""
test_accuracy = np.empty(25)
train_accuracy = np.empty(25)
val_accuracy = np.empty(25)
for n in range(1,25):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    val_accuracy[n] = knn.score(X_val, y_val) 
    test_accuracy[n] = knn.score(X_test,y_test)
    train_accuracy[n] = knn.score(X_train, y_train)
n1=np.argmax(test_accuracy)
n2=np.argmax(val_accuracy) 
print(test_accuracy[n1])
print(val_accuracy[n2])
"""  
#yields a test acc of 96 % and a val of 97%
