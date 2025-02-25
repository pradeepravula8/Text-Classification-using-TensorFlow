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
