# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import io
import os

# Page configuration
st.set_page_config(
    page_title="Real-Time Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 5px 0;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 12px;
        line-height: 20px;
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
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return None
            return load_model(model_path)
        
        self.model = load_emotion_model(model_path)
    
    def preprocess_image(self, image):
        """Preprocess image for emotion prediction"""
        # Convert to grayscale and resize
        if image.mode != 'L':
            image = image.convert('L')
        image = image.resize((48, 48))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype('float32') / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)
        
        return img_array
    
    def detect_emotions_simple(self, image):
        """Simple emotion detection without face detection"""
        try:
            # Preprocess the entire image
            processed_img = self.preprocess_image(image)
            
            # Predict emotions
            predictions = self.model.predict(processed_img, verbose=0)[0]
            
            # Get results
            emotion_idx = np.argmax(predictions)
            emotion = self.emotions[emotion_idx]
            confidence = predictions[emotion_idx]
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': predictions
            }
            
        except Exception as e:
            st.error(f"Error in emotion detection: {e}")
            return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Emotion Recognition System</h1>', unsafe_allow_html=True)
    
    # Check if model exists
    model_path = 'emotion_model.h5'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please make sure it's in the same directory.")
        st.info("""
        **To fix this:**
        1. Upload your trained `emotion_model.h5` file
        2. Make sure it's in the same directory as app.py
        3. Redeploy the app
        """)
        return
    
    # Initialize detector
    try:
        detector = EmotionDetector(model_path)
        if detector.model is None:
            return
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Image Analysis", "About"]
    )
    
    if app_mode == "Image Analysis":
        image_analysis_mode(detector)
    elif app_mode == "About":
        about_mode()

def image_analysis_mode(detector):
    st.header("📷 Image Emotion Analysis")
    
    st.info("""
    **How to use:**
    1. Upload a clear facial image
    2. The system will analyze the dominant emotion
    3. View confidence scores for all emotions
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a facial image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a face for emotion analysis"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_column_width=True)
        
        # Process image
        with st.spinner("🔍 Analyzing emotions..."):
            result = detector.detect_emotions_simple(image)
            
            with col2:
                st.subheader("📊 Emotion Analysis")
                
                if result:
                    emotion = result['emotion']
                    confidence = result['confidence']
                    color = detector.colors[emotion]
                    
                    # Display main result
                    st.markdown(f"""
                    <div class="emotion-card" style="border-left-color: {color};">
                        <h3>🎯 Dominant Emotion: {emotion}</h3>
                        <h4>Confidence: {confidence:.1%}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display all emotion confidences
                    st.subheader("📈 Emotion Distribution")
                    
                    for idx, emotion_name in enumerate(detector.emotions):
                        conf = result['all_predictions'][idx]
                        color = detector.colors[emotion_name]
                        
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            st.write(f"{emotion_name}:")
                        with col_b:
                            bar_html = f"""
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {conf*100}%; background-color: {color};">
                                    {conf:.1%}
                                </div>
                            </div>
                            """
                            st.markdown(bar_html, unsafe_allow_html=True)
                    
                    # Emotion insights
                    st.subheader("💡 Emotion Insights")
                    if emotion == "Happy":
                        st.success("😊 This appears to be a happy expression!")
                    elif emotion == "Sad":
                        st.warning("😢 This appears to be a sad expression.")
                    elif emotion == "Angry":
                        st.error("😠 This appears to be an angry expression.")
                    elif emotion == "Surprise":
                        st.info("😲 This appears to be a surprised expression.")
                    else:
                        st.info(f"🎭 Detected as {emotion} expression.")
                        
                else:
                    st.error("❌ Could not analyze the image. Please try another image.")

def about_mode():
    st.header("ℹ️ About This App")
    
    st.markdown("""
    ## Emotion Recognition System
    
    This application uses deep learning to detect emotions from facial expressions.
    
    ### 🎯 Features:
    - **Image Analysis**: Upload images for emotion detection
    - **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    - **Confidence Scores**: See prediction confidence for each emotion
    - **Clean Interface**: Easy-to-use web interface
    
    ### 🔧 Technology:
    - **Backend**: TensorFlow/Keras with CNN
    - **Frontend**: Streamlit
    - **Model**: Custom-trained CNN on FER2013 dataset
    
    ### 📊 Dataset:
    - **FER2013 Dataset**: 35,887 facial images
    - **7 Emotion Categories**
    - **48×48 pixel grayscale images**
    
    ### 🎭 Emotion Classes:
    - **😠 Angry**: Expressions of anger or frustration
    - **🤢 Disgust**: Expressions of disgust or revulsion
    - **😨 Fear**: Expressions of fear or anxiety
    - **😊 Happy**: Expressions of happiness and joy
    - **😢 Sad**: Expressions of sadness or disappointment
    - **😲 Surprise**: Expressions of surprise or shock
    - **😐 Neutral**: Neutral facial expressions
    
    ### 🚀 How to Use:
    1. Go to **Image Analysis** tab
    2. Upload a clear facial image
    3. View the emotion analysis results
    4. See confidence scores for all emotions
    """)

if __name__ == "__main__":
    main()
