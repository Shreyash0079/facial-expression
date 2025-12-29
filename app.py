import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("model.h5")

# Emotion Labels
emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Streamlit UI
st.set_page_config(page_title="Emotion Detector", page_icon="😺", layout="centered")

st.title("🎭 FER-2013 Emotion Recognition")
st.write("Upload a face image and let AI guess the emotion.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def predict_emotion(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    prediction = model.predict(img)
    idx = np.argmax(prediction)
    return emotion_labels[idx], prediction[0]

if uploaded_file:
    # show uploaded image
    image_np = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded Image", width=250)

    # Save temp image for prediction
    with open("temp.jpg", "wb") as f:
        f.write(image_np)

    emotion, probs = predict_emotion("temp.jpg")
    st.markdown(f"### 🧠 Predicted Emotion: **{emotion}**")

    # Show probability bar chart
    st.bar_chart(probs)
