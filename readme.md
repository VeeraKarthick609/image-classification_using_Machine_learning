# Celebrity Image Classifier

This project aims to classify celebrity images using a machine learning model. The code includes data cleaning, image cropping, feature engineering, and model training using Support Vector Machines (SVM) and Logistic Regression. The final model is saved for future use.

## Table of Contents

- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Saved Model](#saved-model)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset consists of images of various celebrities. The images are preprocessed to extract relevant features for training the machine learning model.

## Data Cleaning

The data cleaning process involves cropping images to focus on faces with at least two eyes. This step is essential for creating a consistent and relevant dataset for model training.

## Feature Engineering

Feature engineering includes extracting features from the images, combining raw and wavelet-transformed images, and preparing the data for training the model.

## Model Training

The model is trained using three different algorithms: Support Vector Machines (SVM), Random Forest, and Logistic Regression. Grid search is employed to find the best hyperparameters for each algorithm.

## Evaluation

The models are evaluated using a confusion matrix to assess their performance on a test dataset. The results are visualized for better interpretation.

## Saved Model

The trained model with the best performance is saved for future use. The model is stored in a pickle file, and the class dictionary used for encoding is saved as a JSON file.

## Usage

To use the saved model for prediction, load it using the joblib library. Example code is provided in the notebook.

```python
import joblib

# Load the model
model = joblib.load('saved_model.pkl')

# Make predictions
predictions = model.predict(new_data)
