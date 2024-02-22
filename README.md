**DeafTech-Vision-A-CNN-Based-Novel-ASL-Communication-Breakthrough**

This repository contains code for training and evaluating convolutional neural networks (CNNs) to recognize American Sign Language (ASL) gestures. The research focuses on recognizing both ASL digits and alphabets.

### Datasets: Provided in the Dataset file 
1. **asl_digits_dataset**: This dataset includes images depicting ASL digits (0-9).
2. **asl_alphabet_train_dataset**: This dataset comprises images representing various ASL alphabet signs, intended for model training.
3. **asl_alphabet_test_dataset**: This dataset contains additional images of ASL alphabet signs, suitable for model testing.

### Usage:
1. **Loading and Preprocessing**: Images are loaded and preprocessed using OpenCV and NumPy. The `load_train_images` and `load_test_images` functions are utilized for loading images and labels from the respective directories.

2. **Data Augmentation**: Data augmentation techniques such as rotation, zoom, and horizontal flip are applied using TensorFlow's `ImageDataGenerator`.

3. **Model Architecture**: The CNN model architecture includes multiple convolutional and max-pooling layers, followed by fully connected layers with dropout regularization.

4. **Training and Evaluation**: The model is trained using the training data and evaluated on the validation and test datasets. Learning rate scheduling and early stopping callbacks are employed during training to improve performance.

### Files:
- `Code`: Containing the all codes
-   `AslAlphabetsDetection.py`: Containing the code for loading, preprocessing, training, and evaluating the ASL Alphabets model.
-   `AslNumbersDetection.py`: Containing the code for loading, preprocessing, training, and evaluating the ASL Digits model.
- `Datasets`: Containing Public dataset description and link. 
- `README.md`: This readme file provides an overview of the project and usage instructions.

### Dependencies:
- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- TensorFlow

### Instructions:
1. Clone the repository to your local machine.
2. Ensure all dependencies are installed.
3. Run the `AslAlphabetsDetection.py.py` `AslNumbersDetection.py` and  file to train and evaluate the ASL model.


### Restriction: Unauthorized use of this codebase without permission is strictly prohibited.



