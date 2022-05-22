import numpy as np
import pandas as pd
import sklearn
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from src import config

df=pd.read_csv('https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv')
df=df.drop(['PassengerId','Name', 'Ticket','Cabin', 'Embarked'],axis=1)
df['Sex']=df['Sex'].map(lambda x: 1 if x=='male' else 0)
df['Age']=df['Age'].fillna(df['Age'].median())
df.to_csv(config.DATA,index=False)

#You use only Pclass, Sex, Age, SibSp (Siblings aboard), Parch (Parents/children aboard), and Fare to predict whether a passenger survived.

print(df.columns)


