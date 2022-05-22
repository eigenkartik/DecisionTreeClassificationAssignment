import pandas
import pandas as pd
import numpy as np
from src import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import accuracy_score, f1_score

def train(data):
    df=pd.read_csv(config.DATA)
    x=df.drop(['Survived'], axis=1)
    print(x.columns)
    y=df['Survived']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    #scaler=StandardScaler() #isn't required for Decision tree
    model=DecisionTreeClassifier(max_depth=3)
    model.fit(x_train,y_train)
    print(model.score(x_train,y_train))
    print(model.score(x_test,y_test))
    joblib.dump(model,config.MODEL)

#
# df=pd.read_csv(config.DATA)
# df.info()

train(config.DATA)

def predict(data):
    x_predict=pd.DataFrame(data, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    model=joblib.load(config.MODEL)
    prediction=model.predict(x_predict)
    print ('prediction is',prediction)

predict([[1,0,38.0,1,0,71.2833]])