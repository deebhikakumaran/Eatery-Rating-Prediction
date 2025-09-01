# McDonald's Rating Prediction

This project is a machine learning solution for predicting the star rating of a McDonald's review. It is built to handle complex data types—such as natural language text and time-based expressions and is deployed with a web application for user interaction.

Check Live Demo: https://mcdonalds-rating-predictor.streamlit.app/

## Approach

The solution is divided into two main components: model training and a web application for inference.

### 1\. Data Preprocessing

The model uses a `ColumnTransformer` to apply different preprocessing steps to specific columns:

  * **Review Text (`review`):** A custom `BertVectorizer` class is used to convert the review text into dense numerical vectors. This approach leverages a pre-trained BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`) to capture the semantic meaning and sentiment of the review, which is more effective than traditional methods like TF-IDF for this task.

  * **Review Time (`review_time`):** A custom `TextReviewTimeTransformer` converts human-readable time strings (e.g., "5 years ago," "a day ago") into a numerical feature representing the number of days since the review was written. This feature is then scaled using `StandardScaler`.

  * **Branch Location (`store_id`):** A unique integer ID is assigned to each branch based on its latitude and longitude. Missing location data is treated as a separate category, receiving a `store_id` of `-1`. This ensures the model can account for all data, even incomplete records.

  * **Other Numerical Features (`rating_count`):** This feature is scaled using `StandardScaler` to normalize its range.

### 2\. Model Pipeline

All preprocessing steps are chained together in a `ColumnTransformer`, which then feeds the processed data into a `StackingClassifier` for the final prediction. This entire workflow is encapsulated in a scikit-learn `Pipeline`, ensuring that the same transformations are applied consistently to new data during prediction.

The trained pipeline and a mapping of branch names to their `store_id` are saved using the `pickle` library.

### 3\. Streamlit Application

A simple web application is built using the Streamlit library to demonstrate the model's functionality. The app loads the pickled model and the branch-to-ID mapping. Users can select a branch name from a dropdown menu, input a review, and provide a rating count. The app uses the `store_map` to convert the selected branch name into its corresponding `store_id` and feeds the full input data into the saved pipeline to get a real-time rating prediction.

## File Descriptions

  * `code.py`: The script for training the model and saving the pipeline and `store_id` map.
  * `app.py`: The Streamlit web application that loads the trained model and performs predictions.
  * `train.csv`: The dataset used for training the model.
  * `test.csv`: The dataset used for evaluating the model.
  * `final_pipeline.pkl`: The pickled scikit-learn pipeline object after training.
  * `store_map.pkl`: The pickled dictionary mapping branch names to their unique `store_id`.

## Run Locally

1.  **Clone or download the project files.**

    ```bash
    git clone https://github.com/deebhikakumaran/Eatery-Rating-Prediction.git
    ```
2.  **Install the required libraries:**

    ```bash
    pip install pandas numpy scikit-learn streamlit transformers torch
    ```
3.  **Train the model:** Run the `code.py` script. This will generate the necessary `.pkl` files.

    ```bash
    python code.py
    ```
4.  **Launch the web application:** Once the model files are generated, run the `app.py` script to start the Streamlit app.

    ```bash
    streamlit run app.py
    ```
5.  **Access the app:** A local URL will be provided in your terminal. Open this URL in your web browser to interact with the prediction interface.

---

This project was developed as part of the Machine Learning Practice (MLP) course in the BS Data Science and Programming program.

---