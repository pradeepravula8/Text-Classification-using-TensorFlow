import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys


if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
tf.keras.utils.disable_interactive_logging()  


TextVectorization = tf.keras.layers.TextVectorization


tf.random.set_seed(42)


MAX_WORDS = 10000  
MAX_LEN = 200     
BATCH_SIZE = 32
EPOCHS = 5

def load_and_preprocess_data():
    try:
       
        print("Downloading IMDB dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=MAX_WORDS,
            path="imdb.npz"  
        )
        
        print("Loading word index...")
        
        word_index = tf.keras.datasets.imdb.get_word_index(
            path="imdb_word_index.json"
        )
        
       
        word_index = {k:(v+3) for k,v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2
        word_index["<UNUSED>"] = 3
        
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        
        def decode_review(text):
            try:
                return ' '.join([reverse_word_index.get(i, '?') for i in text])
            except Exception as e:
                print(f"Error decoding review: {e}")
                return ""
        
        print("Converting reviews to text...")
        texts_train = [decode_review(x) for x in x_train]
        texts_test = [decode_review(x) for x in x_test]
        
        
        all_texts = texts_train + texts_test
        all_labels = np.concatenate([y_train, y_test])
        
        return all_texts, all_labels
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def create_model(vocab_size, embedding_dim=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=MAX_LEN),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def main():
    try:
        
        print("Loading and preprocessing data...")
        texts, labels = load_and_preprocess_data()
        
        
        print("Vectorizing text...")
        vectorizer = TextVectorization(
            max_tokens=MAX_WORDS,
            output_sequence_length=MAX_LEN,
            standardize='lower_and_strip_punctuation'
        )
        
        
        print("Adapting vectorizer...")
        vectorizer.adapt(texts)
        
        print("Converting texts to sequences...")
       
        padded_sequences = vectorizer(texts).numpy()
        
        # Split the data
        indices = np.arange(len(texts))
        np.random.shuffle(indices)
        padded_sequences = padded_sequences[indices]
        labels = labels[indices]
        
        
        num_validation_samples = int(0.2 * len(texts))
        x_train = padded_sequences[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = padded_sequences[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]
        
        
        print("Creating and compiling model...")
        model = create_model(MAX_WORDS)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        
        print("Training model...")
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val)
        )
        
       loss, accuracy = model.evaluate(x_val, y_val)
        print(f"\nValidation accuracy: {accuracy:.4f}")
        
        
        model.save('imdb_sentiment_model.keras')
        
        
        def predict_sentiment(text):
            sequences = vectorizer([text])
            prediction = model.predict(sequences)[0][0]
            return "Positive" if prediction > 0.5 else "Negative", prediction

       
        sample_text = "This movie was really great! I enjoyed every moment of it."
        sentiment, confidence = predict_sentiment(sample_text)
        print(f"\nSample text: {sample_text}")
        print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
