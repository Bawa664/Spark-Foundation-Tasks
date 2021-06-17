#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# setting the style of matplotlib
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings("ignore")

#importing dataset
supermarket=pd.read_csv("C:\python csv dataset\SampleSuperstore.csv")
print(supermarket.head())
print(supermarket.info())
print(supermarket.describe())
print(supermarket.dtypes)

#checking for missing vaiue
print(supermarket.isnull().sum())

#checking for duplicate values
print(supermarket.duplicated().sum())

#Dropping the duplicate values
supermarket.drop_duplicates(inplace=True)
print(supermarket.shape)

#Displaying the unique data
print(supermarket.nunique())

#Dropping of irrelevant columns like we have postal code in the dataset
drop=supermarket.drop(columns='Postal Code',axis=1,inplace=True)
print(supermarket)

sales_supermarket=supermarket.groupby('Category',as_index=False)['Sales'].sum()
subcat_supermarket=supermarket.groupby(['Category','Sub-Category'])['Sales'].sum()
subcat_supermarket['Sales']=map(int,subcat_supermarket)
print(sales_supermarket)

#Exploratory Data Analysis
supermarket.plot(x='Quantity',y='Sales',style='.')
plt.title('Quantity vs Sales')
plt.xlabel('Quantity')
plt.ylabel('Sales')
plt.grid()
plt.show()

supermarket.plot(x='Discount',y='Profit',style='.')
plt.title('Discount vs profit')
plt.xlabel('Discount')
plt.ylabel('Profit')
plt.grid()
plt.show()

sns.pairplot(supermarket)
plt.show()

sns.pairplot(supermarket,hue='Region')
plt.show()

supermarket['Category'].value_counts()
sns.countplot(x=supermarket['Category'])
plt.show()

print(supermarket.corr())

sns.heatmap(supermarket.corr(),annot=True)
plt.show()

fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(10,7));

sns.countplot(supermarket['Category'],ax=axs[0][0])
sns.countplot(supermarket['Segment'],ax=axs[0][1])
sns.countplot(supermarket['Ship Mode'],ax=axs[1][0])
sns.countplot(supermarket['Region'],ax=axs[1][1])
axs[0][0].set_title('Category',fontsize=20)
axs[0][1].set_title('Segment',fontsize=20)
axs[1][0].set_title('Ship Mode',fontsize=20)
axs[1][1].set_title('Region',fontsize=20)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(ncols=2, nrows = 2, figsize = (10,10))
sns.distplot(supermarket['Sales'], color = 'red',  ax = axs[0][0])
sns.distplot(supermarket['Profit'], color = 'green',  ax = axs[0][1])
sns.distplot(supermarket['Quantity'], color = 'orange',  ax = axs[1][0])
sns.distplot(supermarket['Discount'], color = 'blue',  ax = axs[1][1])
axs[0][0].set_title('Sales Distribution', fontsize = 20)
axs[0][1].set_title('Profit Distribution', fontsize = 20)
axs[1][0].set_title('Quantity distribution', fontsize = 20)
axs[1][1].set_title('Discount Distribution', fontsize = 20)
plt.show()

plt.title('Region')
plt.pie(supermarket['Region'].value_counts(),labels=supermarket['Region'].value_counts().index,autopct='%1.1f%%')
plt.show()

plt.title('Ship Mode')
plt.pie(supermarket['Ship Mode'].value_counts(),labels=supermarket['Ship Mode'].value_counts().index,autopct='%1.1f%%')
plt.show()

supermarket.groupby('Segment')['Profit'].sum().sort_values().plot.bar()
plt.title("Profits on various Segments")
plt.show()

supermarket.groupby('Region')['Profit'].sum().sort_values().plot.bar()
plt.title("Profits on various Regions")
plt.show()


plt.figure(figsize=(14,6))
supermarket.groupby('State')['Profit'].sum().sort_values().plot.bar()
plt.title("Profits on various Regions")
plt.show()



