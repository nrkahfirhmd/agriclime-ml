from flask import Flask, request, jsonify
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

MODEL_PATH = "forecasting.keras"  
forecast = None

try:    
    forecast = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

def fetch_data(location):
    try:
        now = date.now()
        
        if now.hour <= 6:
            next_date = now.strftime("%Y-%m-%d")
            target_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            response = requests.get('http://api.weatherapi.com/v1/history.json?key={}&q={}&dt={}'.format(key, location, target_date))
            target_data = response.json()        
            response = requests.get('http://api.weatherapi.com/v1/history.json?key={}&q={}&dt={}'.format(key, location, next_date))
            next_data = response.json()       
            data = fetch_before_six(target_data, next_data)        
        else:
            target_date = now.strftime("%Y-%m-%d")
            response = requests.get('http://api.weatherapi.com/v1/history.json?key={}&q={}&dt={}'.format(key, location, target_date))
            data = response.json()                        
        
        response.raise_for_status()           
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None          

def fetch_before_six(data_prev, data_next):
    now = date.now() 
    data = []
    
    for hour in range(24):
        data.append(data_prev['forecast']['forecastday'][0]['hour'][hour])
    
    for hour in range(now.hour + 1):
        data.append(data_next['forecast']['forecastday'][0]['hour'][hour])  
    
    return data
    
def preprocess_data(data):
    now = date.now()    
    
    df = []
    if now.hour <= 6:
        for hour in range(len(data)):
            field = {
                'time': data[hour]['time'],
                'temp': data[hour]['temp_c'],
                'wind_speed': data[hour]['wind_kph'],
                'wind_degree': data[hour]['wind_degree'],            
                'pressure': data[hour]['pressure_in'],
                'precip': data[hour]['precip_in'],
                'humidity': data[hour]['humidity'],
                'cloud': data[hour]['cloud'],
                'uv': data[hour]['uv'],
                'weather': data[hour]['condition']['code'],
            }   
            
            df.append(field)    
    else:
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
    
    return reshaped_prediction

# PREDICT 1 HOUR
@app.route("/predict/<location>", methods=["GET"])
def run(location):    
    try:
        data = fetch_data(location)
        df = preprocess_data(data)
        df['time'] = pd.to_datetime(df['time'])
        
        to_predict = df.drop(['time', 'weather', 'precip', 'uv'], axis=1)    
                
        sequences = create_sequences(np.array(to_predict))
        
        sequences_list = np.array(sequences).tolist()
        
        prediction = predict_forecast(sequences_list[-1])
        
        predicted_data = []
        predicted_data.append(str(df['time'].iloc[-1] + timedelta(hours=1)))
        prediction = prediction[0]
        prediction[0] = round(prediction[0], 1)
        prediction[1] = round(prediction[1], 1)
        prediction[2] = round(prediction[2])
        prediction[3] = round(prediction[3], 2)
        prediction[4] = round(prediction[4])
        prediction[5] = round(prediction[5])
        predicted_data.append(prediction)
        predicted_data.append(df['precip'].iloc[-1])
        predicted_data.append(df['uv'].iloc[-1])
        
        flattened_data = np.concatenate([[predicted_data[0]], np.array(predicted_data[1]).flatten(), [predicted_data[2], predicted_data[3]]])

        flattened_data[7], flattened_data[5] = flattened_data[5], flattened_data[7]
        
        classified = predict_classify([np.array(flattened_data[1:], dtype=float).reshape(-1, 8)])   
        
        return jsonify({"data": flattened_data.tolist(), "weather": classified}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
MODEL_PATH = "classification.keras"  
classify = None

@app.route("/forecast/<location>/<hours>", methods=["GET"])
def run_multiple(location, hours):
    # try:
    hours = int(hours)
    data = fetch_data(location)
    df = preprocess_data(data)
    
    df['time'] = pd.to_datetime(df['time'])
    
    predictions, classify = predict_multiple_hours(df, hours)
    
    for i in range(len(predictions)):
        predictions[i] = predictions[i].tolist()
    
    return jsonify({"data": predictions, "weather": classify}), 200
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

def predict_multiple_hours(data, hours=12):
    predictions = []
    classify = []
    
    to_predict = data.drop(['time', 'weather', 'precip', 'uv'], axis=1)
    
    time = data['time'].iloc[-1]
    
    for _ in range(hours):
        sequences = create_sequences(np.array(to_predict))        
        
        prediction = predict_forecast(sequences[-1])
        
        to_predict = np.vstack([to_predict[1:], prediction[0]])
        
        predicted_data = []
        time = time + timedelta(hours=1)
        predicted_data.append(str(time))
        prediction = prediction[0]
        prediction[0] = round(prediction[0], 1)
        prediction[1] = round(prediction[1], 1)
        prediction[2] = round(prediction[2])
        prediction[3] = round(prediction[3], 2)
        prediction[4] = round(prediction[4])
        prediction[5] = round(prediction[5])
        predicted_data.append(prediction)
        predicted_data.append(data['precip'].iloc[-1])
        predicted_data.append(data['uv'].iloc[-1])
        
        flattened_data = np.concatenate([[predicted_data[0]], np.array(predicted_data[1]).flatten(), [predicted_data[2], predicted_data[3]]])

        flattened_data[7], flattened_data[5] = flattened_data[5], flattened_data[7]
        
        classified = predict_classify([np.array(flattened_data[1:], dtype=float).reshape(-1, 8)])   
        
        classify.append(classified)
        
        predictions.append(flattened_data)
    
    return predictions, classify

try:    
    classify = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

def weather_label(prediction):    
    label = ['Clear', 'Cloudy', 'Drizzle', 'Rain','Storm/Thunder', 'Snow', 'Sleet', 'Fog', 'Extreme Weather']
    
    return label[prediction]

def predict_classify(data):
    result = classify.predict(data)
    
    result_label = weather_label(np.argmax(result[0]))
    
    return result_label

# CHECK API HEALTH
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"}), 200

if __name__ == "__main__":
    app.run(debug=True)