import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
.\AImodule.py

# -----------------------------------------
# Train the Model
# -----------------------------------------

# Set up data generators for training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the images
    rotation_range=30,  # Random rotations
    width_shift_range=0.2,  # Random shifts
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill empty spaces after transformations
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

# Load training and validation data
train_dir = 'dataset/train'  # Directory with training images
val_dir = 'dataset/test'     # Directory with validation images

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode='binary'  # Binary classification: healthy or diseased
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Define a CNN model
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Save the model for later use
model.save('plant_disease_classifier.h5')


# -----------------------------------------
# Predict Using the Trained Model
# -----------------------------------------

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
