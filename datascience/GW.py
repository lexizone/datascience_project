
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import metrics

data = pd.read_csv('Environment_Temperature_change_E_All_Data_NOFLAG.csv', encoding='latin-1')
data2 = pd.read_csv('FAOSTAT_data_11-24-2020.csv')

#print("Data1:\n",data.info())
#print("Data2:\n",data2.info())

# Due to world politics being in contant flux many countries have come into existence or have gone the way of the dodo. 
# This means our dataset is incomplete as shown by the above plot where the bright lines show null/nan values in our dataset. 
# A simple thing we do to remedy this is to just remove the rows which contain nan values.
plt.figure(figsize=(20,8))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#plt.show()

# After getting rid of the extra columns that are not needed the dataset now looks like the table below.
data=data.dropna()

data = data.rename(columns={'Area':'Country'})
#data=data[data['Element']=='Temperature change']
data = data.drop(columns=['Area Code','Months Code','Element Code','Unit'])
TempC = data.loc[data.Months.isin(['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December'])]

#print(TempC.head())


# look at how many countries we have in the data set.
#print(TempC.Country.unique())

# The table below show the data present for just Thailand.
TH = TempC.loc[TempC.Country=='Thailand']
#print(TH)

# make a simple plot to see how the temperature varies over the year.
plt.figure(figsize=(15,10))

sns.lineplot(x=TH.Months.loc[TH.Element=='Temperature change'],y=TH.Y1971.loc[TH.Element=='Temperature change'], label='Y1975')
sns.lineplot(x=TH.Months.loc[TH.Element=='Temperature change'],y=TH.Y1981.loc[TH.Element=='Temperature change'], label='Y1985')
sns.lineplot(x=TH.Months.loc[TH.Element=='Temperature change'],y=TH.Y1991.loc[TH.Element=='Temperature change'], label='Y1995')
sns.lineplot(x=TH.Months.loc[TH.Element=='Temperature change'],y=TH.Y2001.loc[TH.Element=='Temperature change'], label='Y2005')
sns.lineplot(x=TH.Months.loc[TH.Element=='Temperature change'],y=TH.Y1961.loc[TH.Element=='Temperature change'], label='Y2015')

plt.xlabel('Months')
plt.ylabel('Temperature change (C)')
plt.title('Temperature Change in Thailand')
#plt.show()

TH = TH.melt(id_vars=['Country','Months','Element'],var_name='Year', value_name='TempC')
TH['Year'] = TH['Year'].str[1:].astype('str')
#print(TH.info())

# replot the temperature change over the year and the standard deviation provided for each month.
plt.figure(figsize=(15,15))
plt.subplot(211)
for i in TH.Year.unique():
    plt.plot(TH.Months.loc[TH.Year==str(i)].loc[TH.Element=='Temperature change'],TH.TempC.loc[TH.Year==str(i)].loc[TH.Element=='Temperature change'],linewidth=0.5)
plt.plot(TH.Months.unique(),TH.loc[TH.Element=='Temperature change'].groupby(['Months']).mean(),'r',linewidth=2.0,label='Average')
plt.xlabel('Months',)
plt.xticks(rotation=90)
plt.ylabel('Temperature change')
plt.title('Temperature Change in Thailand')
plt.legend()

plt.subplot(212)
plt.plot(TH.Months.loc[TH.Year=='1995'].loc[TH.Element=='Standard Deviation'],TH.TempC.loc[TH.Year=='1995'].loc[TH.Element=='Standard Deviation']) 
plt.xlabel('Year')
plt.xticks(rotation=90)
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of Temperature Change in Thailand')

plt.subplots_adjust(hspace=0.65,bottom=0.16,top=0.945)
#plt.show()

# look at the how the data is spread over the different years and how the mean temperature changes.
plt.figure(figsize=(15,10))
plt.scatter(TH['Year'].loc[TH.Element=='Temperature change'],TH['TempC'].loc[TH.Element=='Temperature change'])
plt.plot(TH.loc[TH.Element=='Temperature change'].groupby(['Year']).mean(),'r',label='Average')
plt.axhline(y=0.0, color='k', linestyle='-')
plt.xlabel('Year')
plt.xticks(np.linspace(0,58,20),rotation=90)
plt.ylabel('Temperature change')
plt.legend()
plt.title('Temperature Change in Thailand')
#plt.show()

# look at the histogram of the temperature changes
plt.figure(figsize=(15,10))
sns.histplot(TH.TempC.loc[TH.Element=='Temperature change'],kde=True,stat='density')
plt.axvline(x=0.0, color='b', linestyle='-')
plt.xlabel('Temperature change')
plt.title('Temperature Change in Thailand')
#plt.show()

# World Temperature

TempC=TempC.melt(id_vars=['Country','Months','Element'],var_name='Year', value_name='TempC')
TempC['Year'] = TempC['Year'].str[1:].astype('str')
#print(TempC)


# To make sure country groupings such as EU or Africa don't skew our calculations we remove them for the world data list. 
# So we are just left with individual countires. 
# We can keep the regions data in the different dataset in case we want to use it later.
regions=TempC[TempC.Country.isin(['World', 'Africa',
       'Eastern Africa', 'Middle Africa', 'Northern Africa',
       'Southern Africa', 'Western Africa', 'Americas',
       'Northern America', 'Central America', 'Caribbean',
       'South America', 'Asia', 'Central Asia', 'Eastern Asia',
       'Southern Asia', 'South-Eastern Asia', 'Western Asia', 'Europe',
       'Eastern Europe', 'Northern Europe', 'Southern Europe',
       'Western Europe', 'Oceania', 'Australia and New Zealand',
       'Melanesia', 'Micronesia', 'Polynesia', 'European Union',
       'Least Developed Countries', 'Land Locked Developing Countries',
       'Small Island Developing States',
       'Low Income Food Deficit Countries',
       'Net Food Importing Developing Countries', 'Annex I countries',
       'Non-Annex I countries', 'OECD'])]

TempC=TempC[~TempC.Country.isin(['World', 'Africa',
       'Eastern Africa', 'Middle Africa', 'Northern Africa',
       'Southern Africa', 'Western Africa', 'Americas',
       'Northern America', 'Central America', 'Caribbean',
       'South America', 'Asia', 'Central Asia', 'Eastern Asia',
       'Southern Asia', 'South-Eastern Asia', 'Western Asia', 'Europe',
       'Eastern Europe', 'Northern Europe', 'Southern Europe',
       'Western Europe', 'Oceania', 'Australia and New Zealand',
       'Melanesia', 'Micronesia', 'Polynesia', 'European Union',
       'Least Developed Countries', 'Land Locked Developing Countries',
       'Small Island Developing States',
       'Low Income Food Deficit Countries',
       'Net Food Importing Developing Countries', 'Annex I countries',
       'Non-Annex I countries', 'OECD'])]

#print(TempC)


# Now we can look at the distribution data. Lets first look at a histogram for temperature change.
plt.figure(figsize=(15,10))
sns.histplot(TempC.TempC.loc[TempC.Element=='Temperature change'],kde=True,stat='density')
plt.axvline(x=0.0, color='b', linestyle='-')
plt.xlabel('Temperature change')
plt.title('Temperature Change distribution of the World')
plt.xlim(-5,5)
#plt.show()


# Let us calculate some averages that we can easily use for our plots.
# Average for the whole world
AvgT=TempC.loc[TempC.Element=='Temperature change'].groupby(['Year'],as_index=False).mean()
# Average for every country
AvgTC=TempC.loc[TempC.Element=='Temperature change'].groupby(['Country','Year'],as_index=False).mean()

# We can also do a scatter plot, like before, for different years for all the countries and plot the world average.
plt.figure(figsize=(15,10))
plt.scatter(TempC['Year'].loc[TempC.Element=='Temperature change'],TempC['TempC'].loc[TempC.Element=='Temperature change'])
plt.plot(AvgT.Year,AvgT.TempC,'r',label='Average')
plt.axhline(y=0.0, color='k', linestyle='-')
plt.xlabel('Year')
plt.xticks(np.linspace(0,58,20),rotation=90)
plt.ylabel('Temperature change')
plt.legend()
plt.title('Temperature Change of the World')
#plt.show()

# Finally we can plot the temperatures for each country and plot the world average on top.
plt.figure(figsize=(15,10))
for i in AvgTC.Country.unique():
    plt.plot(AvgTC.Year.loc[AvgTC.Country==str(i)],AvgTC.TempC.loc[AvgTC.Country==str(i)],linewidth=0.5)

plt.plot(AvgT.Year,AvgT.TempC,'r' ,label='Average' ,linewidth=2.2)
plt.axhline(y=0.0, color='k', linestyle='-')
plt.xlabel('Year')
plt.xticks(np.linspace(0,58,20),rotation=90)
plt.ylabel('Average Temperature change')
plt.legend()
plt.title('Average Temperature Change of the World')
#plt.show()


# Test-Train Split
MonthV={'January':'1', 'February':'2', 'March':'3', 'April':'4', 'May':'5', 'June':'6', 'July':'7','August':'8', 'September':'9', 'October':'10', 'November':'11', 'December':'12'}
TempC=TempC.replace(MonthV)
#print(TempC.head())

y=TempC['TempC'].loc[TempC.Element=='Temperature change']
X=TempC.drop(columns=['TempC','Country','Months','Element']).loc[TempC.Element=='Temperature change']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8,test_size=0.2,random_state=64)



#Regression model : Simple Linear Regression
LR = LinearRegression()
LR.fit(X_train,y_train)
LRpreds = LR.predict(X_valid)

#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, LRpreds)))

# to check the Actual value and Predicited value
plt.figure(figsize=(15,8))
plt.plot(y_valid-LRpreds,'o')
plt.axhline(y=0.0, color='k', linestyle='-')
plt.ylabel('Actual value - Predicited value')
#plt.show()

# Fit the model to the training data
LR.fit(X, y)
#print(LR.fit(X, y))


# create artifical data that we can use to test what the model predicts for the future.
# Creating prediction data
LR_test=pd.DataFrame({'Year':np.random.randint(1961,2060, size=1000)})
LR_test=LR_test.sort_values(by=['Year']).reset_index(drop=True).astype(str)
#T_test=pd.DataFrame(np.arange(2020, 2046),columns=['Year']).astype(str)

# Generate test predictions
preds_test = LR.predict(LR_test)
LR_test['TempC']=pd.Series(preds_test, index=LR_test.index)


#Regression model : Polynomial regression
PR2_mod = Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', LinearRegression(fit_intercept=False))])

#PR3_mod = Pipeline([('poly', PolynomialFeatures(degree=5)),
#                  ('linear', LinearRegression(fit_intercept=False))])
# Fit the model to the training data
PR2_mod.fit(X, y)
#PR3_mod.fit(X, y)

# Creating prediction data
PR2_test=pd.DataFrame({'Year':np.random.randint(1961,2060, size=1000)})
PR2_test=PR2_test.sort_values(by=['Year']).reset_index(drop=True).astype(str)

#PR3_test=pd.DataFrame({'Year':np.random.randint(1995,2060, size=1000)})
#PR3_test=PR3_test.sort_values(by=['Year']).reset_index(drop=True).astype(str)

# Generate test predictions
pred2_test = PR2_mod.predict(PR2_test)
#pred3_test = PR3_mod.predict(PR3_test)

PR2_test['TempC']=pd.Series(pred2_test, index=PR2_test.index)
#PR3_test['TempC']=pd.Series(pred3_test, index=PR3_test.index)


# Plotting Results

plt.figure(figsize=(15,10))
for i in AvgTC.Country.unique():
    plt.plot(AvgTC.Year.loc[AvgTC.Country==str(i)],AvgTC.TempC.loc[AvgTC.Country==str(i)],linewidth=0.5)

plt.plot(AvgT.Year,AvgT.TempC,'r',linewidth=2.0)
plt.plot(LR_test.Year.unique(),LR_test.groupby('Year').mean(),'b',linewidth=2.0,label='Linear Model')
plt.plot(PR2_test.Year.unique(),PR2_test.groupby('Year').mean(),'g',linewidth=2.0,label='Poly-2 Model')
#plt.plot(PR3_test.Year.unique(),PR3_test.groupby('Year').mean(),'c',linewidth=2.0,label='Poly-5 Model')
plt.axhline(y=0.0, color='k', linestyle='-')
plt.xticks(np.linspace(0,100,40),rotation=90)
plt.xlabel('Year')
plt.ylabel('Average Temperature change')
plt.title('Average Temperature Change of the World')
plt.legend()
plt.show()