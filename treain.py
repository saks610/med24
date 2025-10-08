from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('plant_disease_classifier.h5')

# Load an image you want to predict
img_path = 'path_to_image.jpg'  # Replace with the image path
img = image.load_img(img_path, target_size=(128, 128))

# Convert the image to a numpy array and normalize it
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)

# Output the result
if prediction[0] > 0.5:
    print("The plant is diseased.")
else:
    print("The plant is healthy.")
