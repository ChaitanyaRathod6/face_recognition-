Real-Time Emotion Recognition System
A deep learning-based web application that detects human emotions from facial expressions in real-time using Convolutional Neural Networks (CNN).

https://img.shields.io/badge/Emotion-Recognition-blue https://img.shields.io/badge/Python-3.8%252B-green https://img.shields.io/badge/TensorFlow-2.13-orange https://img.shields.io/badge/Streamlit-1.28-red

🎭 Features
Real-time Emotion Detection: Detect 7 different emotions from facial expressions

Multiple Input Methods: Upload images or use webcam for live detection

Confidence Scoring: View prediction confidence for each emotion

Beautiful UI: Interactive Streamlit web interface

Model Insights: Visualize emotion distribution with confidence bars

🎯 Detected Emotions
The system can detect the following 7 emotions:

😠 Angry - Expressions of anger or frustration

🤢 Disgust - Expressions of disgust or revulsion

😨 Fear - Expressions of fear or anxiety

😊 Happy - Expressions of happiness and joy

😢 Sad - Expressions of sadness or disappointment

😲 Surprise - Expressions of surprise or shock

😐 Neutral - Neutral facial expressions

📊 Dataset
This project uses the FER2013 (Facial Expression Recognition 2013) dataset:

Dataset Information
Name: FER2013

Source: Kaggle - FER2013 Dataset

Size: 35,887 grayscale images

Image Dimensions: 48×48 pixels

Classes: 7 emotion categories

Split: Training (28,709), PublicTest (3,589), PrivateTest (3,589)

Dataset Structure
text
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
🛠️ Technology Stack
Backend
Python 3.8+

TensorFlow/Keras - Deep learning framework

OpenCV - Computer vision and image processing

NumPy - Numerical computations

Frontend
Streamlit - Web application framework

PIL/Pillow - Image processing

Matplotlib - Data visualization (for training)

Model Architecture
Convolutional Neural Network (CNN)

Input: 48×48 grayscale images

Output: 7 emotion classes with softmax activation

Layers: Multiple Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense layers

📁 Project Structure
text
emotion-recognition/
├── app.py                 # Streamlit web application
├── train.py              # Model training script
├── model.py              # CNN model architecture
├── dataset_loader.py     # Data loading and preprocessing
├── requirements.txt      # Python dependencies
├── emotion_model.h5     # Trained model weights
└── README.md            # Project documentation
🚀 Quick Start
Prerequisites
Python 3.8 or higher

pip (Python package manager)

Installation
Clone the repository

bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
Install dependencies

bash
pip install -r requirements.txt
Download the FER2013 dataset

Option 1: Download from Kaggle

Option 2: Use the provided download script

python
python download_dataset.py
Training the Model (Optional)
If you want to train the model yourself:

bash
python train.py
This will:

Load and preprocess the FER2013 dataset

Train the CNN model for 30 epochs

Save the trained model as emotion_model.h5

Generate training history plots

Running the Application
Start the Streamlit app

bash
streamlit run app.py
Open your browser and go to http://localhost:8501

Choose your input method:

Image Upload: Upload facial images

Webcam: Use your camera for real-time detection

🌐 Deployment
Deploy on Streamlit Cloud (Free)
Push your code to GitHub

Go to Streamlit Cloud

Connect your GitHub repository

Deploy with one click

Deploy on Hugging Face Spaces
Create a new Space on Hugging Face

Select Streamlit as SDK

Upload all project files

Your app will be automatically deployed

📈 Model Performance
The trained model achieves:

Training Accuracy: ~60-65%

Validation Accuracy: ~65-70%

Test Accuracy: ~65-70%

Note: Performance may vary based on training parameters and dataset quality.

🎮 Usage Examples
Image Upload
Select "Image Upload" mode

Upload a facial image (JPG, JPEG, or PNG)

View detected emotions with confidence scores

See emotion distribution across all classes

Webcam Detection
Select "Webcam" mode

Allow camera access

View real-time emotion detection

See live confidence scores

🔧 Customization
Adding New Emotions
Update the emotion dictionary in dataset_loader.py

Retrain the model with new labeled data

Modify the frontend in app.py

Model Improvements
Experiment with different CNN architectures

Try transfer learning with pre-trained models

Implement data augmentation techniques

Fine-tune hyperparameters

🤝 Contributing
We welcome contributions! Please feel free to:

Fork the repository

Create feature branches

Submit pull requests

Report issues and suggestions


🙏 Acknowledgments
FER2013 Dataset: Provided by Pierre-Luc Carrier and Aaron Courville

TensorFlow/Keras: Deep learning framework

Streamlit: For the amazing web app framework

OpenCV: Computer vision library



📚 References
Goodfellow, I., et al. "Challenges in representation learning: A report on the black box." (2013)

TensorFlow Documentation

Streamlit Documentation

OpenCV Documentation

Made with ❤️ using Python, TensorFlow, and Streamlit

If you find this project helpful, please give it a ⭐ on GitHub!

