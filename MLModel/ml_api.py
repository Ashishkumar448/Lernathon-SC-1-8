
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To allow JS frontend access

model = joblib.load('rf_pipe.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return jsonify({'severity': str(pred)})

if __name__ == "_main_":
    app.run(port=5000)