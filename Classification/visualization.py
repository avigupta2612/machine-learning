import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
df=pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',sheet_name='Canada by Citizenship',skiprows=range(20))
print(df.head())
print(df.shape)
df.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)
df.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
df.set_index('Country',inplace=True)
df.columns = list(map(str, df.columns))
df['Total']=df.sum(axis=1)
years=list(map(str,range(1980,2014)))
print(df.columns)
df1=df.loc[['India','China','Japan'],years]
df2=pd.DataFrame(df[years].sum(axis=0))
df2=df2.reset_index()
df2.columns=['year','total']
df2['year']=df2['year'].astype(int)
df2.plot(kind='scatter',figsize=(10,8),x='year',y='total',color='blue')
plt.xlabel('Years')
plt.ylabel('Total')
plt.show()
print(df2.head())