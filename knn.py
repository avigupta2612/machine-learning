import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv")
x=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed','employ', 'retire', 'gender']].values
y=df['custcat'].values
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

k=10
mean_acc=np.zeros((k-1))
std_acc=np.zeros((k-1))
ConfusionMx=[];
for n in range(1,k):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
print(mean_acc)
print(std_acc)
plt.plot(range(1,k),mean_acc,'g')
plt.fill_between(range(1,k),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc,alpha=0.10)
plt.legend(('Accuracy','+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('No of neighbors')
plt.show()
print('best accuracy was',mean_acc.max(),'with k=',mean_acc.argmax()+1)