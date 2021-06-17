#Importing the Libraries
#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style for the matplotlib
plt.style.use('seaborn')

#To ignore Warnings
import warnings
warnings.filterwarnings('ignore')


#Importing the data
df = pd.read_csv('C:\python csv dataset\Iris.csv')
print(df.shape)
print(df.describe())

#Checking if the values is null or not
print(df.isnull().sum())

#Checking the data-types of features
print(df.dtypes)

#Spliting the data for the model building
# Now lets start making the model
# First let split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop(['Id','Species'], axis=1), df['Species'], test_size=0.2, random_state=0)

#Cross validation of the data
# Cross_validation of data and importing the Decision tree classifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
dtc = DecisionTreeClassifier()
#dtc.fit(x_train, y_train)
acc = cross_val_score(dtc, x_train, y_train, scoring = "accuracy", cv = 10)
print(acc)

#Training and predicting the dataset
# Training and predicting
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

#Checking the accuracy through confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt="0.2f",cmap="coolwarm")
print("accuracy is ", accuracy_score(y_test,y_pred))
plt.show()

#Visualising the Decision tree classifier
# Visualising the decision tree graph
plt.figure(figsize=(25,20))
plot_tree(dtc,feature_names=df.columns,class_names=df['Species'].value_counts().index,filled=True)
plt.show()
