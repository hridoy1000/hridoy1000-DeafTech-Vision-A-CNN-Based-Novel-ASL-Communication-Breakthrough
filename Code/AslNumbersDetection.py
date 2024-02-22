import imghdr
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Function to load and preprocess the images from the directory
def load_images(directory, image_size):
    images = []
    labels = []

    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        # Check if the entry is a directory
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                filepath = os.path.join(label_dir, filename)

                # Check if the file is a valid image
                if imghdr.what(filepath) is not None:
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (image_size, image_size))
                    images.append(image)
                    labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    return images, labels


# Set the image size for resizing (you can adjust this as needed)
image_size = 64
batch_size = 64

# Directory containing the dataset
dataset_directory = '/Users/hridoy/PycharmProjects/aslresearch/Datasets/asl_digits_dataset'

# Load images and labels
images, labels = load_images(dataset_directory, image_size)

# Normalize the pixel values to [0, 1]
images /= 255.0

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Normalize the pixel values to [0, 1]
train_images /= 255.0
test_images /= 255.0

# Reshape the images for CNN input
train_images = train_images.reshape(-1, image_size, image_size, 1)
test_images = test_images.reshape(-1, image_size, image_size, 1)

# Data augmentation generator for training data
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

# Apply data augmentation to the training data
train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)

# Build a more complex CNN model with regularization and dropout
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])


# Define a learning rate schedule
def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch >= 20:
        return initial_lr * 0.1
    return initial_lr


# Compile the model with learning rate scheduler
optimizer = optimizers.Adam(learning_rate=lr_schedule(0))
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks for learning rate scheduling and early stopping
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Train the model with augmented data and use validation data for evaluation
epochs = 32
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=(test_images, test_labels),
    callbacks=[lr_scheduler, early_stopping]
)

# Plot the error and accuracy graphs for the new model
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the new model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'New Test Accuracy: {test_accuracy}')
print(f'New Test Loss: {test_loss}')