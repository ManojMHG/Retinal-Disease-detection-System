# Retinal-Disease-detection-System
A deep learning–based retinal disease detection system using CNN models and a Flask web application to classify retinal images and provide disease insights.

Retinal Disease Detection System 👁️🧠
📌 Overview

This project is a Retinal Disease Detection System that uses Deep Learning (CNN) to analyze retinal images and detect eye-related diseases.
The system is implemented as a Flask-based web application, allowing users to upload retinal images and receive predictions along with treatment information.

This project is useful for medical image analysis, academic research, and healthcare-related AI applications.

🚀 Features

Retinal image classification using trained deep learning models

Web-based interface for image upload and prediction

Pre-trained CNN model (.h5) for accurate detection

Disease-specific treatment information

Scalable and modular project structure

🛠️ Technologies Used
🔹 Programming & Frameworks

Python

Flask

TensorFlow / Keras

OpenCV

NumPy

🔹 Deep Learning

Convolutional Neural Networks (CNN)

Pre-trained model (best_model_overall.h5)

🔹 Frontend

HTML

CSS

Bootstrap (via templates & static folders)

⚙️ System Workflow

User uploads a retinal image through the web interface

Image is preprocessed using OpenCV

CNN model predicts the retinal disease

Prediction result is displayed on the web page

Treatment information is shown based on disease type

📂 Project Structure
Retinal-Disease-Detection/
│── app.py                    # Main Flask application
│── model.py                  # Model loading and prediction logic
│── models.py                 # CNN architecture
│── database.py               # Database handling
│── treatment_info.py         # Disease treatment details
│── best_model_overall.h5     # Trained deep learning model
│── requirements.txt          # Required Python packages
│── templates/                # HTML files
│── static/                   # CSS, JS, images
│── Dataset/                  # Training dataset
│── Images/                   # Sample images
│── run.bat                   # Windows run script
│── run.txt                   # Execution notes
