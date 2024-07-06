from flask import Flask, request, jsonify
from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Parameters for loading data and images
detection_model_path = 'faceDetectModel.xml'
emotion_model_path = 'emoDetectModel.hdf5'

# Loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def detect_emotions(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        return {
            "emotion_probabilities": {emotion: float(prob) for emotion, prob in zip(EMOTIONS, preds)},
            "detected_emotion": label,
            "probability": float(emotion_probability)
        }
    else:
        return {"error": "No face detected"}

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
   
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image provided"}), 400

    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    result = detect_emotions(image_path)
    os.remove(image_path)

    return jsonify(result)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)