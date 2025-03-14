import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model and vectorizer (pickled in Phase 1)
model = pickle.load(open("best_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    # Transform and predict
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    # Return raw prediction (label) as JSON
    return jsonify({ "prediction": int(pred) })

if __name__ == '__main__':
    app.run(debug=True)
