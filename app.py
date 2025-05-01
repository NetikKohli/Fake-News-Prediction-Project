import pickle
import pandas as pd
import re
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained model and vectorizer (Phase 1)
try:
    model = pickle.load(open("best_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    model = None
    vectorizer = None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text.lower())
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return re.sub(r'\\s+', ' ', text).strip()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    title = author = content = ""
    if request.method == "POST":
        title = request.form.get("title", "")
        author = request.form.get("author", "")
        content = request.form.get("content", "")
        combined_text = f"{title} {author} {content}"
        cleaned = clean_text(combined_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        result = "Real News" if prediction == 0 else "Fake News"

    # Load live predictions (if available)
    live_results = []
    try:
        df_live = pd.read_csv("live_predictions.csv")
        # Take the latest 10 for display
        df_show = df_live.sort_values(by="text").tail(10)
        for _, row in df_show.iterrows():
            live_results.append({
                "text": row["text"],
                "prediction": row["prediction_str"]
            })
    except Exception:
        live_results = []

    return render_template("my.html",
                           result=result,
                           title=title,
                           author=author,
                           content=content,
                           live_results=live_results)

if __name__ == "__main__":
    app.run(debug=True)
