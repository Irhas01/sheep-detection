import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Download model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    response = requests.get(image_path)
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224))  # Resize to the model's expected input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

# Function to make predictions
def predict(image_path):
    image = load_and_preprocess_image(image_path)
    image = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]  # Add batch dimension
    predictions = model(image)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Example usage
image_url = "https://upload.wikimedia.org/wikipedia/commons/f/ff/Domestic_goat_kid_in_capeweed.jpg"
predicted_class = predict(image_url)

# Load and display the image with the predicted label
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted class index: {predicted_class}")
plt.show()
