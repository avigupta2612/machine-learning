import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df=pd.read_csv('heart.csv')
print(df.columns)
df=df.astype('object')
print(df.dtypes)

x=df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']].values
print(x[0:5])

y=df['target']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=4)
dtree=DecisionTreeClassifier(criterion='entropy',max_depth=14)
dtree.fit(xtrain,ytrain)
yhat=dtree.predict(xtest)

print('Accuracy: ',metrics.accuracy_score(ytest,yhat))
from sklearn.tree import export_graphviz
# Export as dot file
featureNames = df.columns[0:13]
targetNames = df["target"].unique().tolist()
export_graphviz(dtree, out_file='tree.dot', 
                feature_names = featureNames,
                class_names = targetNames,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();
