#importing Necessary Libraries
import pydataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydataset

# setting the style of matplotlib
plt.style.use("fivethirtyeight")

#reading Data
ds=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
ds.head()
print(ds.describe())

#years 
x=ds.iloc[:,0]
y=ds.iloc[:,1]
print(x)
print(y)
plt.scatter(x,y)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title("Student_scores%")
plt.show()

#collecting x and y
x=ds['Hours'].values
y=ds['Scores'].values
print(x,y)


#Mean X and Y
mean_x=np.mean(x)
mean_y=np.mean(y)

#total number of values
n=len(x)

#y=m*x+c
#c=y-m+x

#Using the formula to calculate m and c
numer = 0
denom = 0
for i in range(n):
    numer+=(x[i]-mean_x)*(y[i]-mean_y)
    denom+=(x[i]-mean_x)**2
m = numer/denom
c = mean_y-(m*mean_x)

#print coefficiencies
plt.scatter(x,y)
print('Slope and Intercept : ',m,c)

print('scores = ',m,"hours+",c)

# Plotting Values and Regression Line
max_x = np.max(x)
min_x = np.min(x)
# Calculating line values x and y
x = np.linspace(min_x,max_x,1000)
y = c+m*x


#Ploting Line
plt.plot(x,y, color='red',label='Regression Line')
#Ploting Scatter Points
plt.scatter(x,y,c='blue',label='Scatter plot')


def myfunc(x):
    return m * x + c
mymodel=list(map(myfunc,x))

plt.scatter(x,y)
plt.plot(x,mymodel)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title("Student_scores%")
plt.show()


# Data preprocessing
X = ds.iloc[:, :-1].values  
y = ds.iloc[:, 1].values
print(x)


# Splitting the data into test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Training the model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
print("---------Model Trained---------")


#Making Predictions
print(x_test)
y_pred=reg.predict(x_test)


#Accuracy of the model
reg.score(x_test,y_test)

#plotting the regression Lines for Train & Test

plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title("Hours vs scores(TRAIN)")
plt.scatter(x_train,y_train)
plt.plot(x_train,reg.predict(x_train),color='red')
plt.show()


plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title("Hours vs scores(TEST)")
plt.scatter(x_test,y_test)
plt.plot(x_test,reg.predict(x_test),color='red')
plt.show()

#predicting the score
x=9.25
m=9.7758033907
c=2.4836734053
y=m*x+c
print(y)
