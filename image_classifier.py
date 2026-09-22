import tensorflow as tf
import numpy as np
from PIL import Image

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)

# Ask user for image
image_path = input("Enter the image path: ")

# Open image
image = Image.open(image_path).convert("RGB")

# Resize image
image = image.resize((224, 224))

# Convert image to NumPy array
image_array = np.array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Preprocess image
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)

# Make prediction
predictions = model.predict(image_array)

# Decode prediction
results = tf.keras.applications.mobilenet_v2.decode_predictions(
    predictions,
    top=3
)[0]

print("\n==============================")
print("      IMAGE PREDICTION")
print("==============================")

for _, label, probability in results:
    print(f"{label}: {probability * 100:.2f}%")