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


def load_train_images(directory, image_size):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                filepath = os.path.join(label_dir, filename)
                if imghdr.what(filepath) is not None:
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (image_size, image_size))
                    images.append(image)
                    labels.append(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    return images, labels


def load_test_images(directory, image_size):
    images = []
    labels = []

    # Go through each image file in the directory
    for filename in os.listdir(directory):
        if not os.path.isdir(filename):  # Ensure we're processing a file and not a directory
            filepath = os.path.join(directory, filename)

            if imghdr.what(filepath) is not None:
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (image_size, image_size))
                images.append(image)
                # Extract label name from the filename (remove '_test.jpg')
                label = filename.split('_')[0]
                labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    return images, labels


image_size = 64
batch_size = 64

train_images, train_labels = load_train_images(
    '/Users/hridoy/PycharmProjects/aslresearch/Datasets/asl_alphabet_test_dataset', image_size)
test_images, test_labels = load_test_images(
    '/Users/hridoy/PycharmProjects/aslresearch/Datasets/asl_alphabet_train_dataset', image_size)

label_encoder = LabelEncoder()
label_encoder.fit(train_labels)
train_labels = label_encoder.transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
                                                                      stratify=train_labels, random_state=42)

train_images /= 255.0
test_images /= 255.0
val_images /= 255.0

train_images = train_images.reshape(-1, image_size, image_size, 1)
test_images = test_images.reshape(-1, image_size, image_size, 1)
val_images = val_images.reshape(-1, image_size, image_size, 1)

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)

# model

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
    layers.Dense(len(np.unique(train_labels)), activation='softmax')
])


# learning rate
def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch >= 20:
        return initial_lr * 0.1
    return initial_lr


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule(0))
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

epochs = 50
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=(val_images, val_labels),
    callbacks=[lr_scheduler, early_stopping]
)

# Plotting
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

# plt.tight_layout()
plt.show()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
