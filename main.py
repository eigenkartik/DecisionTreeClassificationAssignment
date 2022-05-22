import numpy as np
from flask import Flask,render_template,request
import joblib
import pandas as pd

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        pclass=request.form['pclass']
        Sex=request.form['Sex']
        Age=float(request.form['Age'])
        SibSp=request.form['Sibsp']
        Parch=request.form['Parch']
        Fare=float(request.form['Fare'])
        x_predict=np.array([[pclass,Sex,Age,SibSp,Parch,Fare]])
        model=joblib.load('clfmodel.pkl')
        prediction=model.predict(x_predict)
        if prediction==1:
            return render_template('result.html', prediction='Person has survived')
        else:
            return render_template('result.html', prediction='Person has not survived')
    except Exception as e:
        return e



if __name__=='__main__':
    app.run(debug=True)

