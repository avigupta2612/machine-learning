import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv('energydata_complete.csv')
df1=df.head(2000)
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = df1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
print(df1.columns)
y=np.asarray(df1['V'])
df1=df1.drop(['Press_mm_hg'],axis=1)
xtrain,xtest,ytrain,ytest= train_test_split(df1,y,test_size=0.2,random_state=3)
print(xtrain.shape)
lm=LinearRegression()
lm.fit(xtrain[['RH_1']],ytrain)
yhat=lm.predict(xtest[['RH_1']])
from sklearn.metrics import r2_score
print(r2_score(yhat,ytest))
print(yhat[0:10])
print(ytest[0:10])