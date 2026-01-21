from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and preprocessors
MODEL_PATH = 'model/titanic_survival_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
LE_SEX_PATH = 'model/le_sex.pkl'
LE_EMBARKED_PATH = 'model/le_embarked.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le_sex = joblib.load(LE_SEX_PATH)
le_embarked = joblib.load(LE_EMBARKED_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Validate inputs
        if not (1 <= pclass <= 3):
            return jsonify({'error': 'Pclass must be between 1 and 3'})
        
        if age < 0 or age > 120:
            return jsonify({'error': 'Age must be between 0 and 120'})
        
        if fare < 0:
            return jsonify({'error': 'Fare must be non-negative'})
        
        # Encode categorical variables
        sex_encoded = le_sex.transform([sex])[0]
        embarked_encoded = le_embarked.transform([embarked])[0]
        
        # Prepare features for prediction
        features = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': 'Survived' if prediction == 1 else 'Did Not Survive',
            'survival_probability': f"{probability[1]:.2%}",
            'death_probability': f"{probability[0]:.2%}",
            'passenger_details': {
                'Passenger Class': pclass,
                'Sex': sex.capitalize(),
                'Age': age,
                'Fare': f"${fare:.2f}",
                'Embarked': embarked
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)