import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv")
x=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed','employ', 'retire', 'gender']].values
y=df['custcat'].values
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

k=4
neigh=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
yhat=neigh.predict(x_test)
print('Train set accuracy: ',metrics.accuracy_score(y_train,neigh.predict(x_train)))
print('Test set accuracy: ',metrics.accuracy_score(y_test,yhat))

      