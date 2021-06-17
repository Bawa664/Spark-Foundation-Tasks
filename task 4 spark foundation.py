import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# setting the style of matplotlib
plt.style.use("fivethirtyeight")

tm = pd.read_csv("C:\python csv dataset\globalterrorismdb_0718dist.csv")
print(tm.head())

print(tm.dtypes)

print(tm.count())

print(tm.shape)

#checking  if the values is null or not
tm.isnull().sum()

print(tm.head())

#EDA
#Number of terrorist attacks in each and every year
EDA=tm['iyear'].value_counts(dropna=False).sort_index()
print(EDA)

print('Country with Highest Terrorist Attacks:',tm['country_txt'].value_counts().index[0])
print('Regions with Highest Terrorist Attacks:',tm['region_txt'].value_counts().index[0])

#Number of Terrorist attacks each year
year = tm['iyear'].unique()
years_count = tm['iyear'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = year,
           y = years_count,
           palette = "tab10")
plt.xticks(rotation = 50)
plt.xlabel('Attacking Year',fontsize=20)
plt.ylabel('Number of Attacks Each Year',fontsize=20)
plt.title('Attacks In Years',fontsize=30)
plt.show()

#Terrorist activities by region in Each Year
pd.crosstab(tm.iyear, tm.region).plot(kind='area',stacked=False,figsize=(20,10))
plt.title('Terrorist Activities By Region In Each Year',fontsize=25)
plt.ylabel('Number of Attacks',fontsize=20)
plt.xlabel("Year",fontsize=20)
plt.show()


#top 10 most affected cities
tm['city'].value_counts().to_frame().sort_values('city',axis=0,ascending=False).head(10).plot(kind='bar',figsize=(20,10),color='blue')
plt.xticks(rotation = 50)
plt.xlabel("city",fontsize=15)
plt.ylabel("Number of attack",fontsize=15)
plt.title("Top 10 most effected city",fontsize=20)
plt.show()

#Terrorist attack locations in each year
terror_region=pd.crosstab(tm.iyear,tm.region)
terror_region.plot(figsize=(15,7))
plt.show()

