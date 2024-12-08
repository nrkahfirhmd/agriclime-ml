from flask import Flask, request, jsonify
import pickle 
import pandas as pd 
import numpy as np 
import requests
from dotenv import load_dotenv
import os
from datetime import datetime as date
from datetime import timedelta
import tensorflow as tf

app = Flask(__name__)

load_dotenv()
key = os.getenv("API_KEY")

SCALER_PATH = "scaler.pkl"
min_max_scaler = None

try:
    with open(SCALER_PATH, 'rb') as f:
        min_max_scaler = pickle.load(f)    
except Exception as e:
    print(f"Error loading model: {e}")

MODEL_PATH = "forecasting.h5"  
forecast = None

try:    
    forecast = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

def fetch_data(location="Bandung"):
    try:
        now = date.now()
        
        if now.hour <= 6:
            target_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            target_date = now.strftime("%Y-%m-%d")
            
        response = requests.get('http://api.weatherapi.com/v1/history.json?key={}&q={}&dt={}'.format(key, location, target_date))
        
        response.raise_for_status()
        data = response.json()
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None    
    
def preprocess_data(data):
    now = date.now()    
    
    df = []
    for hour in range(now.hour + 1):
        field = {
            'time': data['forecast']['forecastday'][0]['hour'][hour]['time'],
            'temp': data['forecast']['forecastday'][0]['hour'][hour]['temp_c'],
            'wind_speed': data['forecast']['forecastday'][0]['hour'][hour]['wind_kph'],
            'wind_degree': data['forecast']['forecastday'][0]['hour'][hour]['wind_degree'],            
            'pressure': data['forecast']['forecastday'][0]['hour'][hour]['pressure_in'],
            'precip': data['forecast']['forecastday'][0]['hour'][hour]['precip_in'],
            'humidity': data['forecast']['forecastday'][0]['hour'][hour]['humidity'],
            'cloud': data['forecast']['forecastday'][0]['hour'][hour]['cloud'],
            'uv': data['forecast']['forecastday'][0]['hour'][hour]['uv'],
            'weather': data['forecast']['forecastday'][0]['hour'][hour]['condition']['code'],
        }   
        
        df.append(field)            

    return pd.DataFrame(df)

def min_max_scale(data):    
    return min_max_scaler.transform(data)

def create_sequences(data, sequence_length=6):
    X = []    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])

    return X

def predict_forecast(data):    
    data_array = np.array(data)
    data_reshaped = data_array.reshape(1, 6, 6) 
    
    prediction = forecast.predict(data_reshaped)
    
    reshaped_prediction = prediction.reshape(-1, prediction.shape[-1])
    
    return min_max_scaler.inverse_transform(reshaped_prediction)

# PREDICT 1 HOUR
@app.route("/predict", methods=["GET"])
def run():    
    # try:
    data = fetch_data()
    df = preprocess_data(data)
    df['time'] = pd.to_datetime(df['time'])
    
    to_predict = df.drop(['time', 'weather', 'precip', 'uv'], axis=1)
    
    data_scaled = min_max_scale(pd.DataFrame(to_predict))
            
    sequences = create_sequences(data_scaled)
    
    sequences_list = np.array(sequences).tolist()
    
    prediction = predict_forecast(sequences_list[-1])
    
    predicted_data = []
    predicted_data.append(str(df['time'].iloc[-1] + timedelta(hours=1)))
    predicted_data.append(prediction.tolist())
    predicted_data.append(df['precip'].iloc[-1])
    predicted_data.append(df['uv'].iloc[-1])
    
    flattened_data = np.concatenate([[predicted_data[0]], np.array(predicted_data[1]).flatten(), [predicted_data[2], predicted_data[3]]])

    flattened_data[7], flattened_data[5] = flattened_data[5], flattened_data[7]
    
    classified = predict_classify([flattened_data[1:]])    
    
    return jsonify({"data": flattened_data.tolist(), "weather": classified.tolist()}), 200
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    
MODEL_PATH = "classification.h5"  
classify = None

try:    
    classify = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    
SCALER_PATH = "standard_scaler.pkl"
standard_scaler = None

try:
    with open(SCALER_PATH, 'rb') as f:
        standard_scaler = pickle.load(f)    
except Exception as e:
    print(f"Error loading model: {e}")

def standard_scale(data):
    return standard_scaler.transform(data)

ENCODER_PATH = "label_encoder.pkl"
label_encoder = None

try:
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)    
except Exception as e:
    print(f"Error loading model: {e}")

def weather_label(prediction):
    return label_encoder.inverse_transform([prediction])[0]

def predict_classify(data):
    scaled_data = standard_scale(data)
    
    result = classify.predict(scaled_data)
    
    result_label = weather_label(np.argmax(result[0]))
    
    return result[0]

# CHECK API HEALTH
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"}), 200

if __name__ == "__main__":
    app.run(debug=True)