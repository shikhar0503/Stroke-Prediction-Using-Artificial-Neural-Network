from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pickle

MODEL_PATH ='Strokes_Predictor.h5'
model = load_model(MODEL_PATH)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        age = float(request.form['age'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        heart_disease = float(request.form['heart_disease'])
        hypertension = float(request.form['hypertension'])
        Married = float(request.form['Married'])
        formerly_smoked = float(request.form['formerly_smoked'])
        self_employed = float(request.form['self_employed'])
        bmi = float(request.form['bmi'])

        data = np.array([[age,avg_glucose_level,heart_disease,hypertension,Married,formerly_smoked,self_employed,bmi]])
        my_prediction = model.predict(data)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)