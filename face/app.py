# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Real-Time Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background-color: #1f77b4;
        text-align: center;
        color: white;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

class EmotionDetector:
    def __init__(self, model_path):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.colors = {
            'Angry': '#FF0000',
            'Disgust': '#008000', 
            'Fear': '#800080',
            'Happy': '#FFD700',
            'Sad': '#0000FF',
            'Surprise': '#00FF00',
            'Neutral': '#808080'
        }
        
        # Load model with caching
        @st.cache_resource
        def load_emotion_model(model_path):
            return load_model(model_path)
        
        self.model = load_emotion_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def preprocess_face(self, face_roi):
        """Preprocess face for emotion prediction"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype('float32') / 255.0
        reshaped = normalized.reshape(1, 48, 48, 1)
        return reshaped
    
    def detect_emotions_image(self, image):
        """Detect emotions in an image"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        results = []
        processed_img = img_array.copy()
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = img_array[y:y+h, x:x+w]
            
            # Preprocess and predict
            processed_face = self.preprocess_face(face_roi)
            predictions = self.model.predict(processed_face, verbose=0)[0]
            
            # Get top emotion
            emotion_idx = np.argmax(predictions)
            emotion = self.emotions[emotion_idx]
            confidence = predictions[emotion_idx]
            
            # Store results
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'all_predictions': predictions
            })
            
            # Draw on image
            color = self.hex_to_bgr(self.colors[emotion])
            cv2.rectangle(processed_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(processed_img, f"{emotion} ({confidence:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return processed_img, results
    
    def hex_to_bgr(self, hex_color):
        """Convert hex color to BGR for OpenCV"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Real-Time Emotion Recognition</h1>', unsafe_allow_html=True)
    
    # Initialize detector
    detector = EmotionDetector('emotion_model.h5')
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Image Upload", "Webcam", "About"]
    )
    
    if app_mode == "Image Upload":
        image_upload_mode(detector)
    elif app_mode == "Webcam":
        webcam_mode(detector)
    elif app_mode == "About":
        about_mode()

def image_upload_mode(detector):
    st.header("📷 Image Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing faces"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process image
        with st.spinner("Detecting emotions..."):
            processed_img, results = detector.detect_emotions_image(image)
            
            with col2:
                st.subheader("Processed Image")
                # Convert BGR to RGB for display
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                st.image(processed_img_rgb, use_column_width=True)
        
        # Display results
        if results:
            st.subheader("📊 Emotion Analysis")
            
            for i, result in enumerate(results):
                with st.expander(f"Face {i+1} - {result['emotion']} ({result['confidence']:.2%})", expanded=True):
                    # Confidence bars for all emotions
                    st.write("**Emotion Distribution:**")
                    
                    for idx, emotion in enumerate(detector.emotions):
                        conf = result['all_predictions'][idx]
                        color = detector.colors[emotion]
                        
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            st.write(f"{emotion}:")
                        with col_b:
                            bar_html = f"""
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {conf*100}%; background-color: {color};">
                                    {conf:.1%}
                                </div>
                            </div>
                            """
                            st.markdown(bar_html, unsafe_allow_html=True)
        else:
            st.warning("No faces detected in the image.")

def webcam_mode(detector):
    st.header("📹 Webcam Live Detection")
    
    st.warning("""
    **Note:** Webcam feature works when running Streamlit locally. 
    For deployed apps, consider using image upload instead.
    """)
    
    # Webcam input using Streamlit's camera input
    img_file_buffer = st.camera_input("Take a picture for emotion detection")
    
    if img_file_buffer is not None:
        # Read image
        bytes_data = img_file_buffer.getvalue()
        image = Image.open(img_file_buffer)
        
        # Process image
        with st.spinner("Analyzing emotions..."):
            processed_img, results = detector.detect_emotions_image(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Your Photo")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Emotion Analysis")
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                st.image(processed_img_rgb, use_column_width=True)
            
            # Display emotion results
            if results:
                st.success(f"Detected {len(results)} face(s)")
                
                for i, result in enumerate(results):
                    emotion = result['emotion']
                    confidence = result['confidence']
                    color = detector.colors[emotion]
                    
                    st.markdown(f"""
                    <div class="emotion-card" style="border-left: 5px solid {color};">
                        <h3>Face {i+1}: {emotion} ({confidence:.1%})</h3>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("No faces detected. Try another photo!")

def about_mode():
    st.header("ℹ️ About This App")
    
    st.markdown("""
    ## Real-Time Emotion Recognition
    
    This application uses deep learning to detect emotions from facial expressions in real-time.
    
    ### Features:
    - 📷 **Image Upload**: Upload images and detect emotions
    - 📹 **Webcam Support**: Real-time emotion detection using your webcam
    - 🎯 **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    - 📊 **Confidence Scores**: See prediction confidence for each emotion
    
    ### Technology Stack:
    - **Backend**: TensorFlow/Keras with CNN
    - **Frontend**: Streamlit
    - **Face Detection**: OpenCV Haar Cascades
    - **Model**: Custom-trained CNN on FER2013 dataset
    
    ### How to Use:
    1. Select your preferred mode (Image Upload or Webcam)
    2. Upload an image or take a photo
    3. View the emotion analysis results
    4. See confidence scores for all emotion classes
    
    ### Emotion Classes:
    """)
    
    emotions_info = {
        'Angry': '😠 Facial expressions showing anger or frustration',
        'Disgust': '🤢 Expressions of disgust or revulsion', 
        'Fear': '😨 Faces showing fear or surprise',
        'Happy': '😊 Expressions of happiness and joy',
        'Sad': '😢 Faces showing sadness or disappointment',
        'Surprise': '😲 Expressions of surprise or shock',
        'Neutral': '😐 Neutral facial expressions'
    }
    
    for emotion, description in emotions_info.items():
        st.write(f"- **{emotion}**: {description}")

if __name__ == "__main__":
    main()