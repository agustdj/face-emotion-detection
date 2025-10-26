import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from mtcnn import MTCNN

# --- Page config ---
st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊")

# --- Load Mini-Xception model ---
@st.cache_resource
def load_emotion_model():
    model = load_model("models/fer2013_mini_XCEPTION.119-0.65.hdf5", compile=False)
    return model

model = load_emotion_model()

# --- Only 3 classes: Angry, Happy, Neutral ---
index_map = [0, 3, 6]  
labels_3 = ['Angry', 'Happy', 'Neutral']

# --- Initialize MTCNN face detector ---
detector = MTCNN()

# --- Predict emotion function ---
def predict_emotion(image):
    # MTCNN expects RGB
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_img)

    if len(results) == 0:
        return "No Face", 0

    # Use first detected face
    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
    face_img = rgb_img[y:y+h, x:x+w]

    # Convert to grayscale for Mini-Xception
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=(0, -1))

    # Predict emotion
    pred = model.predict(face_resized, verbose=0)
    pred_subset = pred[0, index_map]
    emotion = labels_3[np.argmax(pred_subset)]
    confidence = np.max(pred_subset) * 100
    return emotion, confidence

# --- Streamlit UI ---
st.title("😊 Facial Emotion Recognition")
st.write("Predict emotions: Angry 😡 | Happy 😄 | Neutral 😐")
option = st.radio("Select image source:", ("📷 Webcam", "🖼️ Upload Image"))

# --- Webcam ---
if option == "📷 Webcam":
    picture = st.camera_input("Capture face image")

    if picture:
        img = Image.open(picture)
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        emotion, confidence = predict_emotion(img_bgr)

        if emotion == "No Face":
            st.warning("No face detected!")
            st.image(img_np, use_container_width=True)
        else:
            st.success(f"🧠 Predicted Emotion: **{emotion}** ({confidence:.2f}%)")
            st.image(img_np, caption=f"Emotion: {emotion}", use_container_width=True)

# --- Upload image ---
elif option == "🖼️ Upload Image":
    uploaded = st.file_uploader("Upload a face image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        emotion, confidence = predict_emotion(img_bgr)

        if emotion == "No Face":
            st.warning("No face detected!")
            st.image(img_np, use_container_width=True)
        else:
            st.success(f"🧠 Predicted Emotion: **{emotion}** ({confidence:.2f}%)")
            st.image(img_np, caption=f"Emotion: {emotion}", use_container_width=True)
