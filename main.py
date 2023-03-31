from flask import Flask, request, jsonify
import pickle
# import pandas as pd

app = Flask(__name__)

model = pickle.load(open('random_forest_classifier.pkl', 'rb'))

# Define the route for prediction
@app.route("/predict", methods=["POST"])
def get_prediction():
    data = request.json
    data = data['input']

    prediction = model.predict([data])[0]

    response = {"prediction": prediction}
    return jsonify(response)