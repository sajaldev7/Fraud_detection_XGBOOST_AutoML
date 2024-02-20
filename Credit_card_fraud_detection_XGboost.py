import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from tpot import TPOTClassifier
#from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score
import pandas_profiling
import ydata_profiling
#%%
#loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('C:/Users/SDev/creditcard.csv')
#%%
#print 1st 5 rows of the dataset
credit_card_data.head()
#%%
#print last 5 rows of the dataset
credit_card_data.tail()
#%%
#ydata_profiling.ProfileReport(credit_card_data)
#%%
#prof =pandas_profiling.ProfileReport(credit_card_data, explorative=True, dark_mode= True)
#prof.to_file('output.html')
#%%
credit_card_data.info()
#%%
#check is there any missing values or not
credit_card_data.isnull().sum()
#%%
credit_card_data.duplicated().sum()
#%%
credit_card_data.drop_duplicates(inplace = True)
#%%
#distribution of legit and fraud transaction
credit_card_data['Class'].value_counts()
#%%
sns.countplot(x='Class', data=credit_card_data)
#%%
#0--> for Legit Transaction & 1 for Fraudulent Transaction
#store legit data into legit variable
legit=credit_card_data[credit_card_data.Class ==0]
#store fraud data into fraud variable
fraud=credit_card_data[credit_card_data.Class ==1]
#%%
print(legit.shape)
print(fraud.shape)
#%%
#statistical measures of the data
legit.Amount.describe()
#%%
fraud.Amount.describe()
#%%
#compare the values for both transcaton
credit_card_data.groupby('Class').mean()
#%%
#taking random 492 values from legit data
legit_sample = legit.sample(n=473)
#%%
#joint/ concatenating two dataFrame
new_dataset= pd.concat([legit_sample, fraud], axis=0)
#%%
new_dataset.head()
#%%
new_dataset.tail()
#%%
#after =pandas_profiling.ProfileReport(new_dataset, explorative=True, dark_mode= True)
#after.to_file('after.html')
#%%
#width 30 and height 30
plt.figure(figsize=(30,30))
#graphical representation of data where the individual values contained in a matrix as colors
#basically shows relation between one variable to another variable
sns.heatmap(new_dataset.corr(),cmap='summer', annot=True, square=True,  )
plt.show()
#%%
new_dataset['Class'].value_counts()
#%%
sns.countplot(x='Class', data=new_dataset)
#%%
new_dataset.groupby('Class').mean()
#%%
#Spliting the data into Features and Targets
X= new_dataset.drop(columns='Class', axis=1)
Y= new_dataset['Class']
#%%
print(X)
print(Y)
#%%
#Split the data into Traning data and Testing data
#the class distribution in the target variable 'Y' is preserved in both the training and testing sets
#every time you run this code with random_state=2, the data will be split in the same way, ensuring consistent results
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
#%%
print(X.shape, X_train.shape, X_test.shape)
#%%
#Model Training
model = RobustScaler()
#%%
model.fit(X_train, Y_train)
#%%
#Accuracy Score
xgb= XGBClassifier(n_estimators=10,max_depth=12,learning_rate=.1)
#%%
xgb.fit(X_train , Y_train)
#%%
print (xgb.score(X_train , Y_train))
print (xgb.score(X_test , Y_test))
#%%
con = confusion_matrix(Y_test,xgb.predict(X_test))
con
#%%
from mlxtend.plotting import plot_confusion_matrix
plot_confusion_matrix(con)
#%%
Y_pred = xgb.predict(X_test)
Y_pred
#%%
print(classification_report(Y_test,Y_pred))
#%%
#using AutoML
tpot = TPOTClassifier(
    generations=5,  # Number of iterations
    population_size=20,  # Number of pipelines in each generation
    verbosity=2,  # Show progress
    random_state=42,
    config_dict='TPOT sparse',  # Use a configuration for sparse data
    n_jobs=-1  # Use all available CPU cores
)
#%%
tpot.fit(X_train, Y_train)












