print("✅ Flask script loaded.")

# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and normalizer
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/normalizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        age = int(request.form['age'])
        gender = request.form['gender']  # Male/Female
        total_bilirubin = float(request.form['total_bilirubin'])
        direct_bilirubin = float(request.form['direct_bilirubin'])
        alk_phos = float(request.form['alk_phos'])
        alt = float(request.form['alt'])
        ast = float(request.form['ast'])
        total_protein = float(request.form['total_protein'])
        albumin = float(request.form['albumin'])
        agr = float(request.form['agr'])

        # Convert gender to number
        gender_encoded = 1 if gender.lower() == "male" else 0

        # Prepare input for prediction
        input_data = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin, alk_phos,
                                alt, ast, total_protein, albumin, agr]])
        
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        result = "⚠️ High risk of Liver Cirrhosis" if prediction == 1 else "✅ Likely Healthy"
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True)
