from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load local model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="./roberta-base-sentiment140"
)

@app.route('/')
def home():
    return "Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = sentiment_pipeline(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
