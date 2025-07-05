from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load the model (update path to your downloaded model)
model = joblib.load('iris_RandomForest.pkl')  # or iris_LogisticRegression.pkl

# Get feature names for reference
iris = load_iris()
feature_names = iris.feature_names

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "iris_classifier"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Handle different input formats
        if 'instances' in data:
            # MLflow format
            features = np.array(data['instances'])
        elif 'data' in data:
            # Custom format
            features = np.array(data['data'])
        else:
            # Direct array format
            features = np.array(data)
        
        # Make prediction
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Get class names
        class_names = iris.target_names
        
        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'classes': [class_names[pred] for pred in predictions],
            'feature_names': feature_names
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': str(type(model).__name__),
        'feature_names': feature_names,
        'target_names': iris.target_names.tolist(),
        'n_features': len(feature_names)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
