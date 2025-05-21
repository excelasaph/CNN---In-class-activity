# CNN - Image Classification Project

This project demonstrates a basic image classification task using a Convolutional Neural Network (CNN) built with TensorFlow/Keras in Google Colab.

## Project Description

The goal of this project is to train a neural network to classify images into different categories. In this specific implementation, the model is trained on a small dataset of images to perform a binary classification.

The project involves:

- Loading and preprocessing image data.
- Building a simple CNN model.
- Training the model on the prepared data.
- Evaluating the model's performance.
- Making predictions on new images.

## Setup and Usage

This project is designed to be run in Google Colab.

1.  **Open in Colab:**
    *   Upload the notebook file (`.ipynb`) to your Google Drive.
    *   Open the notebook in Google Colab.

2.  **Upload Images:**
    *   You will need to upload the image files specified in the code to your Colab environment's `/content/` directory or adjust the file paths in the code to where you store them. The current code expects the images:
        *   `/content/gary-sandoz-0Q5d7Qe2Wko-unsplash.jpg`
        *   `/content/manja-vitolic-gKXKBY-C-Dk-unsplash.jpg`
        *   `/content/vander-films-aPvB8KMIh5w-unsplash.jpg`
    *   The images can be found in the `/Image` directory
    *   Make sure these files exist in the specified location before running the code.

3.  **Run the Notebook:**
    *   Go through each code cell and run them sequentially.
    *   The notebook will load the data, build and train the model, and provide an evaluation of its performance.

## Code Overview

The core components of the code are:

-   **Loading and Preprocessing:** Reads image files, resizes them to a fixed dimension (28x28 pixels), converts them to NumPy arrays, and assigns numerical labels.
-   **Data Splitting:** Divides the dataset into training and testing sets.
-   **Model Definition:** Creates a `Sequential` Keras model with:
    *   Convolutional layers (`Conv2D`) for feature extraction.
    *   Max pooling layers (`MaxPooling2D`) for downsampling.
    *   A `Flatten` layer to convert the 2D features into a 1D vector.
    *   Dense (fully connected) layers (`Dense`) for classification.
-   **Model Compilation:** Configures the model with an optimizer (`adam`), a loss function (`sparse_categorical_crossentropy`), and metrics (`accuracy`).
-   **Model Training:** Trains the model on the training data for a specified number of epochs.

## Dependencies

The project relies on the following Python libraries:

-   `tensorflow`
-   `numpy`
-   `sklearn` (specifically `sklearn.model_selection`)
-   `PIL` (Pillow)

These libraries are typically pre-installed in Google Colab.

## Suggested Next Steps

-   **Evaluate the model on the test set** to understand its performance on unseen data.
-   **Visualize predictions** to see examples of correct and incorrect classifications.
-   **Add more data** to improve the model's generalization capabilities.
-   **Implement data augmentation** to artificially increase the training data size and diversity.
-   **Experiment with different model architectures** or hyperparameters.
-   **Save the trained model** for future use.
