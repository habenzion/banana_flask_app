from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import base64

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("banana_model.keras")

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")

    processed = preprocess(image)
    prediction = model.predict(processed)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Convert image to base64 (for displaying back in UI)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        "prediction": class_names[class_index],
        "confidence": round(confidence * 100, 2),
        "image": img_str
    })

if __name__ == "__main__":
    app.run(debug=True)
