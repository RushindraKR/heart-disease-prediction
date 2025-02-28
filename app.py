from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Collect all 13 features
        features = np.array([[  
            float(data["age"]),
            float(data["sex"]),
            float(data["cp"]),
            float(data["trestbps"]),
            float(data["cholesterol"]),
            float(data["fbs"]),
            float(data["restecg"]),
            float(data["thalach"]),
            float(data["exang"]),
            float(data["oldpeak"]),
            float(data["slope"]),
            float(data["ca"]),
            float(data["thal"])
        ]])

        # Predict
        prediction = model.predict(features)[0]

        return jsonify({"prediction": "Positive" if prediction == 1 else "Negative"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
