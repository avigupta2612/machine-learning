import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
df=pd.read_csv('AirQualityUCI.csv',sep=';')
df=df.drop(['Date','Time', 'Unnamed: 15', 'Unnamed: 16'],axis=1)

print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)

'''def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 10
    height = 6
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()'''
'''def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-100, 600])
    plt.ylabel('Price')
    plt.legend()'''
'''for i in ['CO(GT)','C6H6(GT)','T','RH','AH']:
    df[i]=df[i].astype('object')
    df[i]=df[i].map(lambda x: x.split(','))
    df[i]=df[i].map(lambda x: '.'.join(x))'''
df['T']=df['T'].astype('object')
df['T']=df['T'].map(lambda x: x.split(','))
df['T']=df['T'].map(lambda x: '.'.join(x))
df=df.dropna()
print(df.head())
print(df.shape)
print(df.dtypes)
'''corr = df.corr()

y=df['T']
df=df.drop(['T'],axis=1)
lm=LinearRegression()
xtrain,xtest,ytrain,ytest=train_test_split(df,y,test_size=0.2,random_state=3)
lm.fit(xtrain[['C6H6(GT)','PT08.S5(O3)']],ytrain)
yhat=lm.predict(xtest[['C6H6(GT)','PT08.S5(O3)']])
yhat1=lm.predict(xtrain[['C6H6(GT)','PT08.S5(O3)']])
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(xtrain[['C6H6(GT)','CO(GT)']])
x_test_pr = pr.fit_transform(xtest[['C6H6(GT)','CO(GT)']])
lm1=LinearRegression()
lm1.fit(x_train_pr,ytrain)
yhat_pr=lm1.predict(x_test_pr)
print(pr)

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
#DistributionPlot(ytest, yhat, "Actual Values (Train)", "Predicted Values (Train)", Title)
print('R2-score linear:',r2_score(yhat,ytest))
print('R2-score non linear:',r2_score(yhat_pr,ytest))
#PollyPlot(xtrain[['C6H6(GT)']], xtest[['C6H6(GT)']], ytrain, ytest,lm1,pr)
print(lm.intercept_)
print(lm.coef_)'''