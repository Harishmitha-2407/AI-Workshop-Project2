 <div align="center">

# 🐱🐶 Dogs vs Cats Image Classification Using CNN

### Deep Learning | Computer Vision | TensorFlow | Keras

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Evaluation-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

**An end-to-end Convolutional Neural Network (CNN) project that learns visual features from images and classifies them into two categories: Cats and Dogs.**

</div>

---

## 📌 Table of Contents

- [✨ Project Overview](#-project-overview)
- [🎯 Objectives](#-objectives)
- [🧠 Introduction to CNN](#-introduction-to-cnn)
- [📊 Dataset Description](#-dataset-description)
- [🏗️ Model Architecture](#️-model-architecture)
- [⚙️ Technologies Used](#️-technologies-used)
- [🚀 Installation and Setup](#-installation-and-setup)
- [🔬 Project Workflow](#-project-workflow)
- [📈 Model Training](#-model-training)
- [📊 Model Evaluation](#-model-evaluation)
- [📉 Visualizations](#-visualizations)
- [💾 Saving the Model](#-saving-the-model)
- [📁 Project Structure](#-project-structure)
- [🚀 Future Improvements](#-future-improvements)
- [⚠️ Limitations](#️-limitations)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Project Overview

Image classification is an important application of computer vision that enables computers to recognize and categorize objects in images.

This project implements a **Convolutional Neural Network (CNN)** to perform binary image classification by distinguishing between images of cats and dogs.

The model is developed using TensorFlow and Keras. It processes color images, extracts meaningful visual features through convolutional layers, and predicts the probability that an image belongs to one of the two classes.

The notebook includes dataset discovery, image loading, preprocessing, data augmentation, model construction, training, validation, performance evaluation, visualization, and model saving.

### 🎯 Project Goal

To build and evaluate a CNN that learns image features automatically and classifies input images into the categories **Cat** or **Dog**.

### 🌟 Key Highlights

- Image classification using deep learning.
- Three convolutional layers for hierarchical feature extraction.
- Image resizing to a consistent resolution of 128 × 128 pixels.
- Pixel normalization to the range 0–1.
- Data augmentation using horizontal flipping and random rotation.
- Dropout regularization to help reduce overfitting.
- Early stopping based on validation loss.
- Classification report and confusion matrix.
- Training and validation accuracy and loss graphs.
- Saving the trained model in Keras format.

## 🎯 Objectives

The main objectives of this project are:

1. Understand the fundamentals of image classification.
2. Load and organize image datasets using TensorFlow/Keras.
3. Prepare images for deep-learning model training.
4. Apply image normalization and data augmentation.
5. Build a CNN with multiple convolutional and pooling layers.
6. Learn visual features such as edges, textures, shapes, and patterns.
7. Train the network using the Adam optimizer.
8. Monitor training and validation performance.
9. Evaluate classification results using accuracy, precision, recall, and F1-score.
10. Analyze incorrect predictions using a confusion matrix.
11. Visualize learning curves to identify possible overfitting.
12. Save the trained model for future predictions.

## 🧠 Introduction to CNN

A **Convolutional Neural Network (CNN)** is a deep-learning architecture commonly used for image processing and computer vision.

Unlike traditional machine-learning approaches that may require manually designed image features, a CNN can learn useful visual representations directly from image pixels.

### How a CNN Works

**1. Convolution Layer**

Applies learnable filters to image data to detect features such as edges, textures, and shapes.

**2. Activation Function**

The ReLU activation function introduces non-linearity, allowing the network to learn more complex patterns.

**3. Max Pooling Layer**

Reduces the spatial dimensions of feature maps while retaining prominent information.

**4. Flatten Layer**

Converts the extracted feature maps into a one-dimensional vector.

**5. Fully Connected Layer**

Combines the extracted features to learn patterns useful for classification.

**6. Output Layer**

Uses a sigmoid activation function to produce a probability between 0 and 1 for binary classification.

The final probability is converted into a predicted class using a threshold of 0.5.

## 📊 Dataset Description

The project expects an image dataset containing separate training and validation directories for cats and dogs.

The notebook searches common locations in Google Colab and the Keras datasets directory to locate the expected folders.

### Expected Dataset Structure

```text
cats_and_dogs_filtered/
│
├── train/
│   ├── cats/
│   │   ├── cat_image_1.jpg
│   │   ├── cat_image_2.jpg
│   │   └── ...
│   │
│   └── dogs/
│       ├── dog_image_1.jpg
│       ├── dog_image_2.jpg
│       └── ...
│
└── validation/
    ├── cats/
    │   ├── cat_image_1.jpg
    │   └── ...
    │
    └── dogs/
        ├── dog_image_1.jpg
        └── ...
```

The example filenames above are illustrative. The actual number and names of image files depend on the dataset provided.

### Dataset Configuration

| Property | Configuration |
|---|---|
| Task | Binary image classification |
| Classes | Cats and Dogs |
| Image type | RGB color images |
| Input image size | 128 × 128 pixels |
| Input channels | 3 |
| Batch size | 32 |
| Training directory | `train/` |
| Validation directory | `validation/` |
| Label mode | Binary |
| Random seed | 42 for training dataset loading |

The notebook uses `image_dataset_from_directory()` to load images and infer their labels from the subdirectory names.

**Important:** The notebook does not download or extract the dataset itself. The expected training and validation folders must already be available in a searched directory. The exact dataset size is not specified in the notebook.

## 🏗️ Model Architecture

The CNN is implemented using TensorFlow/Keras and the Sequential API.

### Network Structure

```text
Input Image: 128 × 128 × 3
           |
           ▼
     Rescaling Layer
       Pixel / 255
           |
           ▼
   Random Horizontal Flip
           |
           ▼
     Random Rotation
           |
           ▼
   Convolution Layer 1
      32 Filters, 3×3
        ReLU Activation
           |
           ▼
      Max Pooling 2×2
           |
           ▼
   Convolution Layer 2
      64 Filters, 3×3
        ReLU Activation
           |
           ▼
      Max Pooling 2×2
           |
           ▼
   Convolution Layer 3
      128 Filters, 3×3
        ReLU Activation
           |
           ▼
      Max Pooling 2×2
           |
           ▼
        Flatten
           |
           ▼
    Dense Layer: 128
        ReLU Activation
           |
           ▼
       Dropout: 0.5
           |
           ▼
     Dense Layer: 1
     Sigmoid Activation
           |
           ▼
    Cat / Dog Prediction
```

### Layer Configuration

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 128 × 128 × 3 | Accepts RGB images |
| Rescaling | Factor 1/255 | Normalizes pixel values |
| RandomFlip | Horizontal | Augments training images |
| RandomRotation | Factor 0.05 | Introduces small rotations |
| Conv2D 1 | 32 filters, 3 × 3 | Learns low-level visual features |
| MaxPooling2D 1 | 2 × 2 | Reduces feature-map dimensions |
| Conv2D 2 | 64 filters, 3 × 3 | Learns intermediate features |
| MaxPooling2D 2 | 2 × 2 | Reduces spatial dimensions |
| Conv2D 3 | 128 filters, 3 × 3 | Learns higher-level features |
| MaxPooling2D 3 | 2 × 2 | Further reduces feature-map dimensions |
| Flatten | One-dimensional output | Prepares features for dense layers |
| Dense | 128 neurons, ReLU | Learns classification patterns |
| Dropout | 0.5 | Regularizes the dense representation |
| Output | 1 neuron, sigmoid | Predicts the binary class probability |

### Why Use Three Convolutional Layers?

The network uses progressively larger numbers of filters: 32, 64, and 128.

This design allows successive layers to learn increasingly complex visual representations. Early layers may detect edges and simple textures, while deeper layers can combine those patterns into more complex features.

### Why Use Max Pooling?

Max pooling reduces the dimensions of the feature maps, lowering computational requirements and helping the model focus on prominent activations.

### Why Use Dropout?

The dropout layer uses a rate of 0.5 during training. It randomly deactivates a portion of the dense-layer activations, helping reduce reliance on particular neurons and potentially limiting overfitting.

### Why Use Sigmoid?

The output layer contains one neuron with sigmoid activation. Its output represents the estimated probability for the positive class, with the class interpretation determined by the dataset's class ordering.

The notebook uses a threshold of 0.5 to convert probabilities into predicted labels.

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep-learning framework |
| Keras | CNN construction and training |
| NumPy | Numerical operations and prediction processing |
| Matplotlib | Accuracy and loss visualization |
| Scikit-learn | Classification report and confusion matrix |
| Google Colab | Cloud-based notebook execution |
| Jupyter Notebook | Interactive development environment |

The notebook installs `tensorflow-datasets`, but its image-loading pipeline uses Keras's `image_dataset_from_directory()` rather than the TensorFlow Datasets API.

## 🚀 Installation and Setup

### Option 1: Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `CNN_Dogs_vs_Cats(1).ipynb`.
3. Ensure that the dataset has the expected directory structure.
4. Place the dataset in a directory searched by the notebook.
5. Run the notebook cells in order.
6. Review the model summary, training output, evaluation metrics, and graphs.
7. Locate the saved model at `/content/dog_cat_cnn.keras`.

The notebook's initial installation cell runs:

```python
!pip install -q tensorflow-datasets
```

TensorFlow, NumPy, Matplotlib, and Scikit-learn must also be available in the execution environment.

### Option 2: Run Locally

Install Python and Jupyter Notebook, then install the required packages:

```bash
python -m pip install tensorflow tensorflow-datasets numpy matplotlib scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and execute its cells in sequence.

**Dataset requirement:** The notebook searches `/content` and the user's Keras datasets directory. If you run it locally or store the dataset elsewhere, update the search roots or provide the correct training and validation paths.

## 🔬 Project Workflow

### Step 1: Install Dependencies

Install the required Python libraries for deep learning, image processing, visualization, and evaluation.

### Step 2: Locate the Dataset

Search the available directories for the expected training and validation folders, each containing `cats` and `dogs` subdirectories.

### Step 3: Load the Images

Use `tf.keras.utils.image_dataset_from_directory()` to load images into batches.

```python
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
```

The training dataset is loaded with binary labels and a fixed random seed. The validation dataset is loaded without shuffling.

### Step 4: Normalize Image Pixels

The model applies:

```python
Rescaling(1.0 / 255)
```

This scales pixel values from the usual 0–255 range to approximately 0–1.

Normalization helps provide a consistent numerical input range for training.

### Step 5: Apply Data Augmentation

The model includes:

```python
RandomFlip("horizontal")
RandomRotation(0.05)
```

These transformations introduce small variations in training images and can help the model generalize to different orientations.

### Step 6: Build the CNN

Create three convolutional blocks, each followed by a max-pooling layer. The extracted features are passed through a flatten layer, a dense layer, dropout, and a sigmoid output.

### Step 7: Compile the Model

The notebook uses the following configuration:

```python
cnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

- **Adam:** Updates model weights during training.
- **Binary cross-entropy:** Measures prediction error for binary classification.
- **Accuracy:** Measures the proportion of correctly classified samples.

### Step 8: Train the Model

Train the network using the training dataset and evaluate validation performance after each epoch.

The notebook allows up to 20 epochs and uses early stopping.

### Step 9: Evaluate the Model

Evaluate the network on the validation dataset and calculate classification metrics.

### Step 10: Visualize Results

Generate training and validation accuracy and loss curves. Review the classification report and confusion matrix to understand model performance.

### Step 11: Save the Model

Save the trained CNN in Keras format for future reuse.

## 📈 Model Training

The notebook trains the network with the following configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Metric | Accuracy |
| Maximum epochs | 20 |
| Batch size | 32 |
| Early stopping monitor | `val_loss` |
| Early stopping patience | 5 epochs |
| Restore best weights | Enabled |
| Prediction threshold | 0.5 |

### Early Stopping

The project uses:

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Training stops when validation loss fails to improve for five consecutive epochs.

Restoring the best weights allows the model to return to the weights associated with the lowest observed validation loss.

### Dataset Prefetching

The notebook also uses:

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
```

Prefetching helps prepare batches while the model processes previous batches, potentially improving input-pipeline efficiency.

## 📊 Model Evaluation

The model is evaluated using the validation dataset.

### 1. Validation Loss

Validation loss measures the binary cross-entropy between the model's predicted probabilities and the true labels.

Lower validation loss generally indicates better probabilistic predictions on the validation data.

### 2. Validation Accuracy

Validation accuracy measures the fraction of validation images correctly classified by the model.

The notebook prints validation loss and validation accuracy after training.

### 3. Classification Report

The project uses Scikit-learn's `classification_report()` to calculate:

- **Precision:** How many predicted examples of a class are correct.
- **Recall:** How many actual examples of a class are identified correctly.
- **F1-score:** The harmonic mean of precision and recall.
- **Support:** The number of actual examples belonging to each class.

These metrics provide a more detailed understanding of classification performance than accuracy alone.

### 4. Confusion Matrix

The confusion matrix compares actual classes against predicted classes.

For binary classification, the four outcomes are:

- **True Negative (TN):** Negative class correctly predicted.
- **False Positive (FP):** Negative class incorrectly predicted as positive.
- **False Negative (FN):** Positive class incorrectly predicted as negative.
- **True Positive (TP):** Positive class correctly predicted.

The matrix helps identify which types of classification errors occur most frequently.

### 5. Training and Validation Curves

The notebook plots accuracy and loss across training epochs.

These curves help analyze:

- Whether the model learns useful patterns.
- Whether validation performance improves alongside training performance.
- Whether the model may be overfitting.
- Whether training stops before the maximum number of epochs.

**Performance note:** The notebook source defines the evaluation process but does not contain recorded output metrics. Actual validation accuracy, loss, precision, recall, and F1-score should be added after running the notebook successfully.

## 📉 Visualizations

The project generates two main learning-curve plots.

### CNN Accuracy

The accuracy graph compares training accuracy with validation accuracy across epochs.

It helps show how classification performance changes during training and whether the model generalizes to the validation data.

### CNN Loss

The loss graph compares training loss with validation loss across epochs.

It helps identify possible overfitting when training loss continues to decrease while validation loss begins to increase.

### Why Are These Visualizations Important?

Visualizations make it easier to understand model behavior, detect possible issues, and decide whether further experimentation is required.

The notebook also prints the confusion matrix and classification report for numerical evaluation.

## 💾 Saving the Model

After training and evaluation, the notebook saves the model using:

```python
cnn_model.save("/content/dog_cat_cnn.keras")
```

The saved file is named:

`dog_cat_cnn.keras`

It contains the trained Keras model and its learned weights, allowing the model to be loaded again without repeating training.

### Loading the Saved Model

```python
import tensorflow as tf

cnn_model = tf.keras.models.load_model(
    "/content/dog_cat_cnn.keras"
)
```

When using the saved model for new images, ensure that the images are resized to 128 × 128 pixels and provided in RGB format.

The saved model already includes its rescaling and augmentation layers. During inference, Keras automatically disables training-only behavior such as random augmentation when the model is called in inference mode.

In Google Colab, download the saved file if you want to retain it after the session ends.

## 📁 Project Structure

A recommended repository structure is:

```text
CNN-Dogs-vs-Cats/
│
├── CNN_Dogs_vs_Cats(1).ipynb
│
├── README.md
│
├── dog_cat_cnn.keras
│
└── dataset/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    │
    └── validation/
        ├── cats/
        └── dogs/
```

### File Descriptions

- `CNN_Dogs_vs_Cats(1).ipynb`: Contains the complete CNN implementation, dataset discovery, image loading, model training, evaluation, visualization, and saving steps.
- `README.md`: Provides project documentation, setup instructions, model details, and evaluation explanations.
- `dog_cat_cnn.keras`: Generated model file saved after the notebook runs successfully.
- `dataset/`: Represents the recommended location for the training and validation images.

The dataset and model file are not automatically included in the repository simply because they appear in this example structure.

## 🚀 Future Improvements

The project can be improved and extended in several ways.

### 1. Improve Classification Performance

Experiment with learning rates, batch sizes, network depth, filter counts, and training settings.

### 2. Increase Dataset Diversity

Use a larger and more varied dataset to help the model learn from differences in lighting, backgrounds, poses, and image quality.

### 3. Use Transfer Learning

Experiment with pretrained architectures such as MobileNetV2, EfficientNet, or ResNet to compare their performance against the custom CNN.

### 4. Add a Single-Image Prediction Interface

Build a simple interface that allows users to upload an image and receive a predicted class and probability.

### 5. Improve Model Evaluation

Evaluate the model on a separate test set that is not used for training or model selection. Compare multiple runs and report their results consistently.

### 6. Add Prediction Visualizations

Display uploaded images alongside their predicted class and confidence score.

### 7. Deploy the Model

Integrate the saved CNN into a small web application using Flask, FastAPI, or Streamlit.

## ⚠️ Limitations

- The model is a binary classifier trained for cats and dogs.
- Its performance depends on the quality and diversity of the available dataset.
- The notebook does not specify the total number of images.
- The notebook does not establish a verified benchmark accuracy.
- Validation performance alone does not establish generalization to all real-world images.
- Data augmentation does not guarantee improved accuracy.
- A separate test set is recommended for an unbiased final evaluation.
- The model expects images with the input dimensions and channel format used during training.

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes.
4. Test the notebook with a valid dataset.
5. Update the documentation if needed.
6. Submit a pull request describing your improvements.

## 📄 License

No license is specified in the supplied notebook. If you publish this project publicly, add a license file that defines how others may use, modify, and distribute your code.

Also check the license and usage conditions of the dataset you choose to include or reference.

---

<div align="center">

### 🐾 Turning Images into Intelligent Predictions

**Built with Python, TensorFlow, Keras, and Convolutional Neural Networks.**

⭐ If you find this project useful, consider giving the repository a star on GitHub.

</div>
