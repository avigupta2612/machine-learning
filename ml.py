import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Datafiniti_Womens_Shoes.csv')
print(df.head())
print(df.columns)
df1=df[['id','brand','prices.amountMax','prices.amountMin']]
df1['price']=(df['prices.amountMax']+df['prices.amountMin'])/2
print(df1.dtypes)
print(df['brand'].value_counts())
df1=df1.drop(['prices.amountMax','prices.amountMin'],axis=1)
dfg=df1.groupby(['brand'], as_index=False).max()
dfg=dfg.sort_values(by=['price'],ascending=False)
print(dfg)