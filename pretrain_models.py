# pretrain_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import os
from utils import load_data, preprocess_text, build_rnn_model, build_lstm_model, build_bilstm_model

# Create directory for pre-trained models
if not os.path.exists('pretrained_models'):
    os.makedirs('pretrained_models')

# Load and preprocess the dataset
dataset_path = 'Tweets.csv'  # Update this path to your dataset location
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df = load_data(dataset_path, text_column='text', sentiment_column='airline_sentiment', sentiment_map=sentiment_map, sample_size=14640)
texts = df['text'].values
labels = df['sentiment'].values

max_words = 5000
max_len = 100
X, tokenizer = preprocess_text(texts, max_words, max_len)
vocab_size = min(len(tokenizer.word_index) + 1, max_words)

# Save the tokenizer
import pickle
with open('pretrained_models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train and save each model
models = {
    'RNN': build_rnn_model(vocab_size, max_len=max_len),
    'LSTM': build_lstm_model(vocab_size, max_len=max_len),
    'Bi-LSTM': build_bilstm_model(vocab_size, max_len=max_len)
}

for model_name, model in models.items():
    print(f"Training {model_name} model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    model_path = os.path.join('pretrained_models', f'{model_name.lower().replace("-", "_")}_model.keras')
    model.save(model_path)
    print(f"{model_name} model saved to {model_path}")