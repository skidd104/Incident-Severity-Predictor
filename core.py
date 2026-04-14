from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('models/car_accident_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('accident_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Region': int(request.form['region']),
            'Area Type': int(request.form['area']),
            'Road Conditions': int(request.form['road']),
            'Time of Day': int(request.form['time']),
            'Driver Age': int(request.form['age']),
            'Sex of Driver': int(request.form['sex']),
            'Vehicle Type': int(request.form['v_type']),
            'Number of Vehicles': int(request.form['n_vehicles'])
        }

        test_df = pd.DataFrame([data])
        
        pred_num = model.predict(test_df)[0]
        prob = model.predict_proba(test_df)[0]

        severity_labels = {0: 'Fatal', 1: 'Serious', 2: 'Slight'}
        result = severity_labels.get(pred_num, "Unknown")
        
        confidence = f"{max(prob)*100:.2f}%"
        status_class = result.lower() 

        return render_template('accident_index.html', 
                               prediction=result, 
                               confidence=confidence,
                               status=status_class)
    except Exception as e:
        return render_template('accident_index.html', prediction="Error", confidence=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5001)