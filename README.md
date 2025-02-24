IMDB Sentiment Analysis with TensorFlow

Overview

This project trains a deep learning model using TensorFlow to classify movie reviews as positive or negative based on the IMDB dataset. The model utilizes TextVectorization, Convolutional Neural Networks (CNNs), and word embeddings to understand and predict sentiment.

Features

Downloads and preprocesses the IMDB movie reviews dataset.

Converts text reviews into numerical sequences.

Implements a CNN-based sentiment analysis model.

Trains and evaluates the model on labeled IMDB reviews.

Saves the trained model for future sentiment prediction.

Provides a function to predict sentiment on new reviews.

Requirements

Make sure you have the following dependencies installed:

pip install tensorflow numpy pandas

Dataset

The IMDB dataset consists of 50,000 movie reviews labeled as:

Positive (1)

Negative (0)

We use the tf.keras.datasets.imdb module to load and process the data.

Model Architecture

The sentiment analysis model consists of:

Embedding Layer - Converts words into dense vectors.

Conv1D Layer - Extracts meaningful features from text.

GlobalMaxPooling1D - Reduces dimensionality while keeping important features.

Dense Layer - Learns deeper patterns in the data.

Dropout Layer - Prevents overfitting.

Output Layer (Sigmoid Activation) - Produces a probability score (positive/negative).

How to Run the Project

Clone this repository:

git clone https://github.com/your-repo/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

Run the Python script:

python main.py

The script will:

Download and preprocess the IMDB dataset.

Train the model.

Evaluate accuracy.

Save the trained model as imdb_sentiment_model.keras.

Predict sentiment for a sample review.

Predicting Sentiment

After training, you can use the predict_sentiment function to classify new movie reviews:

sample_text = "This movie was really great! I enjoyed every moment of it."
sentiment, confidence = predict_sentiment(sample_text)
print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")

Output Example:

Validation accuracy: 0.8765

Sample text: This movie was really great! I enjoyed every moment of it.
Predicted sentiment: Positive (confidence: 0.9234)

Model Saving & Loading:

The trained model is saved as imdb_sentiment_model.keras. You can load it later using:

from tensorflow import keras
model = keras.models.load_model('imdb_sentiment_model.keras')


