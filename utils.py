# utils.py
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dense, Dropout

def load_data(file_path_or_buffer, text_column, sentiment_column, sentiment_map, sample_size=None):
    if isinstance(file_path_or_buffer, str):
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_csv(file_path_or_buffer)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    df[text_column] = df[text_column].astype(str).str.lower()
    df[text_column] = df[text_column].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
    df[text_column] = df[text_column].str.replace(r'@\w+', '', regex=True)
    df[text_column] = df[text_column].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    
    df['sentiment'] = df[sentiment_column].str.lower().map(sentiment_map)
    df = df.dropna(subset=['sentiment'])
    
    if len(df) < 100:
        raise ValueError("Dataset is too small after preprocessing. Please use a larger dataset.")
    
    invalid_labels = df['sentiment'].isna() | ~df['sentiment'].isin([0, 1, 2])
    if invalid_labels.any():
        raise ValueError(f"Invalid sentiment labels found: {df[invalid_labels][sentiment_column].unique()}")
    
    return df

def preprocess_text(texts, max_words, max_len, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

def preprocess_single_text(text, tokenizer, max_len):
    text = str(text).lower()
    text = text.replace(r'http\S+|www\S+|https\S+', '')
    text = text.replace(r'@\w+', '')
    text = text.replace(r'[^a-zA-Z\s]', '')
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded_sequence

def build_rnn_model(vocab_size, embedding_dim=100, max_len=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        SimpleRNN(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(vocab_size, embedding_dim=100, max_len=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(vocab_size, embedding_dim=100, max_len=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model