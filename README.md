# Rainfall-Predictor
This repository contains a machine learning project aimed at predicting daily rainfall using a 10-year meteorological dataset with 14,5460 samples and 22 features. It includes data preprocessing, feature selection, and the implementation of Decision Trees, KNN, and SVM models to enhance prediction accuracy.
# Weather Prediction Model - 10-Year Meteorological Dataset 🌦️

This project aims to predict rainfall using a 10-year historical weather dataset. By leveraging various preprocessing and machine learning techniques, the model can effectively identify rainfall patterns and make predictions based on meteorological data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Objective](#objective)
- [Challenges](#challenges)
- [Solution Approach](#solution-approach)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview

This project explores the relationship between various meteorological features and rainfall occurrence. Using the dataset, we preprocess and build machine learning models to predict if it will rain on a given day.

## Dataset

The dataset contains 145,460 samples over 22 features, collected across various weather stations. Key features include:
- **Date**: Observation date
- **Station Code**: Unique ID for each weather station
- **Temperature**: Daily minimum and maximum temperatures (°C)
- **Rainfall**: Daily rainfall amount (mm)
- **Evaporation**: Evaporation level (mm)
- **Sunshine**: Hours of sunshine
- **Wind Data**: Wind direction, gust trajectory, and speed
- **Humidity**: Relative humidity at different times of day
- **Cloudiness**: Cloud cover percentage
- **Air Pressure**: Atmospheric pressure (hPa)

## Objective

The goal is to predict rainfall (binary classification) based on weather data, focusing on:
1. **Data preprocessing** to handle missing values and feature engineering
2. **Feature transformation** and encoding of non-numeric features
3. **Model training** using Decision Trees, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)
4. **Evaluation** using confusion matrices and metrics like accuracy, precision, recall, and F1-score

## Challenges

- **Missing Data**: Handling missing values with imputation strategies based on feature correlation or deletion.
- **Feature Encoding**: Using Label Encoding for ordinal features and One-Hot Encoding for nominal features.
- **Outliers**: Identifying and removing outliers using z-score calculations.
- **Imbalanced Classes**: Addressing the 3:1 class imbalance in target labels, exploring balancing techniques to improve accuracy and reduce bias.
- **Model Selection and Hyperparameter Tuning**: Using cross-validation and hyperparameter tuning to select optimal model settings.

## Solution Approach

1. **Data Preprocessing**:
   - Dropped duplicate samples and removed samples with null target values.
   - Applied transformations for date-related features and encoded categorical data using One-Hot or Label Encoding as appropriate.
   - Imputed missing values for correlated features, using mean or model-based imputation.
   - Removed outliers based on z-score thresholding.

2. **Feature Selection**:
   - Calculated feature correlations and selected those with an absolute correlation greater than 0.3 with the target.
   - Reduced the dataset to key features for effective training.

3. **Modeling**:
   - Trained three models: Decision Tree, K-Nearest Neighbors, and SVM.
   - For Decision Tree, optimized the depth through cross-validation.
   - For KNN, tested various `k` values and found `k=3` to be optimal.
   - For SVM, utilized Grid Search for hyperparameter tuning, although default parameters were used due to computational limits.

4. **Model Evaluation**:
   - Divided data into train, validation, and test sets.
   - Used Accuracy, Precision, Recall, and F1 Score for model performance.
   - Analyzed confusion matrices to evaluate true positives, false positives, false negatives, and true negatives.

## Model Evaluation

The project uses various performance metrics to evaluate the models:
- **Accuracy** = Correct Predictions / Total Predictions
- **Precision** = True Positives / (True Positives + False Positives)
- **Recall** = True Positives / (True Positives + False Negatives)
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)

## Technologies Used

- **Python**: Core language
- **Pandas & Numpy**: Data manipulation and preprocessing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-Learn**: Machine learning models and evaluation metrics


## Usage

1. Run data preprocessing:
    ```python
    python data_preprocessing.py
    ```
2. Train models:
    ```python
    python model_training.py
    ```
3. Evaluate results:
    ```python
    python model_evaluation.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

