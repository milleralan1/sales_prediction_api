from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)

model = None
MODEL_PATH = 'sales_model.joblib'

# Allowed values based on training data 
VALID_VALUES = {
    'Store_Type': ['S1', 'S2', 'S3', 'S4'],
    'Location_Type': ['L1', 'L2', 'L3', 'L4', 'L5'],
    'Region_Code': ['R1', 'R2', 'R3', 'R4']
}

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return True
        return False
    except Exception:
        return False

def validate_input_data(data):
    required_fields = ['Store_id', 'Store_Type', 'Location_Type', 
                      'Region_Code', 'Date', 'Holiday', 'Discount']
    
    # Check for missing fields [cite: 1]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    
    # Strict Value Validation 
    for field, allowed in VALID_VALUES.items():
        if data[field] not in allowed:
            return False, f"Invalid {field}. Must be one of: {', '.join(allowed)}"
    
    return True, "Valid"

def prepare_prediction_data(data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
    df['Store_id'] = df['Store_id'].astype(str)
    
    if 'Discount' in df.columns and df['Discount'].dtype == 'object':
        df['Discount'] = df['Discount'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    numeric_cols = ['Holiday', 'Discount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

@app.route('/')
def home(): return render_template('index.html')

@app.route('/predict-ui')
def predict_ui(): return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        if model is None: return jsonify({'error': 'Model not loaded'}), 503
        data = request.get_json()
        
        is_valid, message = validate_input_data(data)
        if not is_valid: return jsonify({'error': 'Validation Error', 'message': message}), 400
        
        input_df = prepare_prediction_data(data)
        prediction = model.predict(input_df)
        
        return jsonify({
            'success': True,
            'prediction': {'sales': round(float(prediction[0]), 2), 'store_id': data['Store_id']}
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if load_model():
        app.run(host='0.0.0.0', port=7008, debug=True)