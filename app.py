import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model and vectorizer
try:
    model = pickle.load(open("best_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not available"}), 500

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Invalid input, 'text' field is required"}), 400

    text = data.get('text', '')
    try:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        label_str = "Real" if pred == 0 else "Fake"
        return jsonify({
            "label": int(pred),
            "prediction": label_str
        })
    except Exception as ex:
        return jsonify({"error": f"Prediction failed: {ex}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
