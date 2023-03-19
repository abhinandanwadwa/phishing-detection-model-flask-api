from flask import Flask, request, jsonify
import pickle
# import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

# Define the route for prediction
@app.route("/predict", methods=["GET"])
def get_prediction():
    prediction = model.predict([[77.0, 23.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.220779221, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 4.0, 2.0, 19.0, 2.0, 32.0, 19.0, 32.0, 15.75, 19.0, 14.66666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.966666667, 0.033333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 80.0, 20.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 77.0, 5767.0, 0.0, 0.0, 1.0, 2.0]])[0]

    response = {"prediction": prediction}
    return jsonify(response)