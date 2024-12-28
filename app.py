from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.joblib')

@app.route('/')
def index():
    return "<center><h1>Flask Machine Learning API - House Price Prediction</h1></center>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract the features from the incoming JSON
    features = np.array(data['features']).reshape(1, -1)
    
    # Get the prediction from the model
    prediction = model.predict(features)
    
    # Return the prediction as a JSON response
    return jsonify({'predicted_house_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

