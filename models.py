import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

def get_grad_cam(model, img_array, layer_name=None):
    # If no layer name provided, try to find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
    
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    # Then compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    
    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    # Build a ponderated map of filters according to gradients importance
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    
    for index, w in enumerate(weights):
        cam += w * output[:, :, index]
    
    # Apply ReLU and resize
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))
    cam = cam / np.max(cam)
    
    return cam

def predict_image(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    
    # Define labels
    labels = ['ARMD','BRVO','CME','CWS','HTN']
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None, None
    
    original_img = img.copy()
    img = cv2.resize(img, (256, 256))
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_label = labels[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    # Generate Grad-CAM
    cam = get_grad_cam(model, img_array)
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Superimpose heatmap on original image
    original_img_resized = cv2.resize(original_img, (256, 256))
    original_img_resized = np.float32(original_img_resized) / 255
    superimposed_img = heatmap * 0.4 + original_img_resized
    superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
    
    # Save the Grad-CAM visualization
    grad_cam_filename = f"grad_cam_{os.path.basename(image_path)}"
    grad_cam_path = os.path.join(os.path.dirname(image_path), grad_cam_filename)
    cv2.imwrite(grad_cam_path, superimposed_img)
    
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted class: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print("All class probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"  {labels[i]}: {prob:.4f}")
    
    return predicted_label, confidence, grad_cam_filename, predicted_class_idx
# model_path="best_model_overall.h5"
# fpath="./Images/HTN/578.jpg"
# print(predict_image(model_path,fpath))