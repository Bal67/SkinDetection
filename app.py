import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import git

# Clone the GitHub repository
repo_url = 'https://github.com/Bal67/SkinDetection'
repo_dir = '/tmp/SkinDetection'

if not os.path.exists(repo_dir):
    git.Repo.clone_from(repo_url, repo_dir)

# Load the fine-tuned model
model_path = os.path.join(repo_dir, 'models', 'finetuned_mobilenetv2.h5')

model = None

# Check if the model file exists and load the model
if os.path.exists(model_path):
    try:
        # If you have custom objects, define them here
        custom_objects = {}  # Replace with actual custom objects if any
        model = load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file not found at {model_path}")

# Define the list of skin conditions
conditions = [
    'allergic contact dermatitis',
    'basal cell carcinoma',
    'dariers disease',
    'ehlers danlos syndrome',
    'erythema multiforme',
    'folliculitis',
    'granuloma pyogenic',
    'granuloma annulare',
    'hailey hailey disease',
    'kaposi sarcoma',
    'keloid',
    'lichen planus',
    'lupus erythematosus',
    'melanoma',
    'mycosis fungoides',
    'myiasis',
    'nematode infection',
    'neutrophilic dermatoses',
    'photodermatoses',
    'pityriasis rosea',
    'psoriasis',
    'scabies',
    'scleroderma',
    'squamous cell carcinoma',
    'tungiasis',
    'vitiligo'
]

# Preprocess the image
def preprocess_image(image):
    img = ImageOps.fit(image, (128, 128), Image.LANCZOS)
    img_array = np.asarray(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict the skin condition
def predict_condition(image):
    preprocessed_image = preprocess_image(image)
    if model:
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        return conditions[predicted_class], confidence
    else:
        return None, None

# Streamlit app
st.title("Skin Condition Predictor")
st.write("Upload an image of a skin condition and the app will predict the possible condition from the list below:")
st.write(", ".join(conditions))

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Predict the condition
    condition, confidence = predict_condition(image)
    if condition and confidence:
        st.write(f"Prediction: {condition} with confidence {confidence:.2f}")
    else:
        st.write("Model not loaded properly. Unable to classify the image.")

st.write("**Disclaimer:** This application can only guess the condition from the list provided and should not be used as a medical diagnosis.")


