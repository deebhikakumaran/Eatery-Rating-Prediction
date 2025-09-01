# Eatery Rating Prediction

Predict the rating given to an eatery by a customer based on their review and location. This project builds a machine learning model to learn from training data and submit predictions for test data.

## Overview

This project aims to predict customer ratings for eateries using both textual reviews and numerical location data (latitude, longitude). The workflow includes data cleaning, feature engineering, model training, and deployment via a Streamlit app.

## Workflow Summary

### 1. Data Preparation

- **Loading Data:** Imported training and test datasets.
- **Cleaning:** Removed duplicates, handled missing values, and dropped irrelevant columns (i.e - store name, category, rating count, category).
- **Outlier Detection:** Identified and removed outliers in latitude and longitude using the IQR method.

### 2. Feature Engineering

- **Text Preprocessing:** Cleaned review text, removed stopwords, and generated word clouds for visualization.
- **Vectorization:** Transformed review text into numerical features using TF-IDF vectorization.
- **Scaling:** Scaled latitude and longitude using `StandardScaler`.

### 3. Model Building

- **ColumnTransformer:** Combined scaled numerical features and vectorized text features.
- **Model Selection:** Trained several models (Random Forest, AdaBoost, Gradient Boosting, Decision Tree, KNN, Logistic Regression, Dummy Classifier).
- **Hyperparameter Tuning:** Used `RandomizedSearchCV` and `BayesSearchCV` for optimization.
- **Best Model:** Selected the best-performing model based on accuracy.

### 4. Prediction & Submission

- **Prediction:** Used the trained model to predict ratings for the test set.
- **Submission:** Saved predictions in the required format for competition submission.

## Streamlit App: Eatery Rating Predictor

An interactive web app built with Streamlit to predict the rating a customer might give to an eatery based on their review and location.

### Features

- **Inputs:**  
  - Latitude  
  - Longitude  
  - Customer Review (text)

- **Prediction:**  
  - Combines location and review text features  
  - Uses a trained model and vectorizer (`rating_model.pkl`, `vectorizer.pkl`)  
  - Displays the predicted rating instantly

### Run Locally

1. Make sure `rating_model.pkl` and `vectorizer.pkl` are in the project directory.
2. Install Streamlit if not already installed:

   ```
   pip install streamlit
   ```
3. Start the app:

   ```
   streamlit run app.py
   ```

### Example Usage

- Enter latitude and longitude of the eatery.
- Paste or type the customer’s review.
- Click **Predict Rating** to see the predicted rating.

---

For details on data processing and model training, see `code.ipynb`.  
For live predictions, use `app.py`.

---
