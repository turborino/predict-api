# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model("model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream).resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr)
    decoded = decode_predictions(preds, top=5)[0]
    return jsonify({'predictions': [(label, float(prob * 100)) for (_, label, prob) in decoded]})
