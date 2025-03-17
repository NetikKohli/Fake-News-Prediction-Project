import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("best_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label_str = "Real" if pred == 0 else "Fake"
    # Return both numeric label and human-readable string
    return jsonify({
        "label": int(pred),
        "prediction": label_str
    })

if __name__ == '__main__':
    app.run(debug=True)
