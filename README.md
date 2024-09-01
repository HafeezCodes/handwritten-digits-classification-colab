# Handwritten Digits Classification

This project implements two deep learning models for handwritten digits classification using TensorFlow and Keras, developed in a Google Colab environment. The models are trained on the MNIST dataset, which consists of grayscale images of handwritten digits from 0 to 9.

## Project Overview

The objective of this project is to build and compare two neural network models for digit classification:

1. **Model without Hidden Layers:** A simple neural network with only an input layer and an output layer.
2. **Model with Hidden Layers:** A neural network that includes one hidden layer to learn more complex features.

The code is designed to run in a Google Colab notebook. If you want to run it locally, ensure you have the required packages installed.

## Installation

To set up your environment locally, you need to install the required Python libraries. 

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/HafeezCodes/handwritten-digits-classification-colab.git
   cd handwritten-digits-classification-colab
   ```

2. **Open the Colab Notebook**

   Open the `handwritten_digits_classification.ipynb` file in Google Colab by uploading it or linking your GitHub repository directly in Colab.

3. **Run the Notebook**

   Follow the instructions in the notebook to execute the code cells. The notebook includes steps for loading the dataset, defining and training the model, and evaluating its performance.

## Model Architectures

### Model without Hidden Layers
- Input Layer: Accepts images of shape (28, 28)
- Flatten Layer: Flattens the 2D images into 1D vectors
- Output Layer: Fully connected layer with 10 neurons and sigmoid activation for multi-class classification

### Model with Hidden Layer
- Input Layer: Accepts images of shape (28, 28)
- Flatten Layer: Flattens the 2D images into 1D vectors
- Hidden Layer: Fully connected layer with 100 neurons and ReLU activation
- Output Layer: Fully connected layer with 10 neurons and sigmoid activation for multi-class classification

## Training

The model is trained using the MNIST dataset with the following configuration:

- **Optimizer:** Adam
- **Loss Function:** Sparse categorical crossentropy
- **Metrics:** Accuracy
- **Epochs:** 5


