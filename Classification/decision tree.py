import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

df=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv')
print(df.head())
x=df[['Age','Sex','BP','Cholesterol','Na_to_K']].values
y=df['Drug']
a1=preprocessing.LabelEncoder()
a2=preprocessing.LabelEncoder()
a3=preprocessing.LabelEncoder()
a1.fit(['F','M'])
a2.fit(['LOW','NORMAL','HIGH'])
a3.fit(['NORMAL','HIGH'])
x[:,1]=a1.transform(x[:,1])
x[:,2]=a2.transform(x[:,2])
x[:,3]=a3.transform(x[:,3])
print(x[0:5])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=4)
dtree=DecisionTreeClassifier()
dtree.fit(xtrain,ytrain)
yhat=dtree.predict(xtest)
print('accuracy : ',metrics.accuracy_score(ytest,yhat))
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(dtree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(ytrain), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')