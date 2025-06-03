from flask import Flask, render_template, request
import pickle
import re
import os

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("best_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Text cleaning function
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text).lower())
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    title = author = content = ""
    if request.method == "POST":
        title = request.form.get("title", "")
        author = request.form.get("author", "")  # Can be empty
        content = request.form.get("content", "")

        # Combine all inputs
        combined_text = f"{title} {author} {content}"
        cleaned = clean_text(combined_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        result = "Real News" if prediction == 0 else "Fake News"

    return render_template("index.html", result=result, title=title, author=author, content=content)


if __name__ == "__main__":
    app.run(debug=True)
