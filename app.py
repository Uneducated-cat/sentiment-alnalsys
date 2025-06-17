import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import os
import pickle
from sklearn.model_selection import train_test_split
from utils import load_data, preprocess_text, preprocess_single_text, build_rnn_model, build_lstm_model, build_bilstm_model

# Base directory for absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Create directory for saving fine-tuned models
if not os.path.exists(os.path.join(BASE_DIR, 'saved_models')):
    os.makedirs(os.path.join(BASE_DIR, 'saved_models'))

# Modern dark theme CSS
CSS = """
<style>
:root {
    --primary: #1a1a1a;
    --secondary: #2d2d2d;
    --accent: #4e8cff;
    --text: #f0f0f0;
    --border: #444;
    --success: #28a745;
    --warning: #ffc107;
    --danger: #dc3545;
    --negative: #ff4d4d;
    --neutral: #ffcc00;
    --positive: #00cc66;
}

* {
    font-family: 'Segoe UI', Roboto, sans-serif;
}

body {
    background-color: var(--primary);
    color: var(--text);
}

.stApp {
    background-color: var(--primary);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
}

.st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
    border-color: var(--border) !important;
}

.card {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    background: linear-gradient(145deg, #2d2d2d, #252525);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}

.card-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
}

.card-content {
    font-size: 15px;
    color: var(--text);
    margin-top: 5px;
}

.confidence-bar {
    height: 6px;
    background: #333;
    border-radius: 3px;
    margin-top: 8px;
    overflow: hidden;
}

.confidence-level {
    height: 100%;
    border-radius: 3px;
}

.stButton > button {
    background: linear-gradient(to right, #4e8cff, #6a5af9);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 4px 12px rgba(78, 140, 255, 0.3);
}

.stTextInput > div > div > input, 
.stTextArea > textarea {
    background-color: var(--secondary) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stSelectbox > div > div > div {
    background-color: var(--secondary) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stFileUploader > section > div {
    background-color: var(--secondary) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

.stProgress > div > div > div > div {
    background-color: var(--accent) !important;
}

.stAlert {
    border-radius: 8px !important;
}

.model-icon {
    margin-right: 8px;
    font-size: 20px;
}

.debug-panel {
    background-color: #2a2a2a;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
    border-left: 4px solid var(--accent);
}

.prob-bar {
    height: 20px;
    margin: 5px 0;
    border-radius: 4px;
    display: flex;
    overflow: hidden;
}

.prob-label {
    width: 80px;
    padding: 0 10px;
    display: flex;
    align-items: center;
    font-weight: bold;
    background: rgba(0,0,0,0.2);
}

.prob-value {
    flex-grow: 1;
    display: flex;
    align-items: center;
    padding: 0 10px;
}
</style>
"""

# Load tokenizer
@st.cache_resource
def load_tokenizer(use_pretrained=True):
    folder = 'pretrained_models' if use_pretrained else 'saved_models'
    path = os.path.join(BASE_DIR, folder, 'tokenizer.pkl')
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# Load models
@st.cache_resource
def load_models(use_pretrained=True):
    models = {}
    folder = 'pretrained_models' if use_pretrained else 'saved_models'
    model_paths = {
        'RNN': os.path.join(BASE_DIR, folder, 'rnn_model.keras'),
        'LSTM': os.path.join(BASE_DIR, folder, 'lstm_model.keras'),
        'Bi-LSTM': os.path.join(BASE_DIR, folder, 'bi_lstm_model.keras')
    }
    for name, path in model_paths.items():
        if not os.path.exists(path):
            st.warning(f"{name} model not found at {path}. Please {'train' if use_pretrained else 'fine-tune'} the model first.")
        else:
            try:
                models[name] = load_model(path)
            except Exception as e:
                st.error(f"Error loading {name} model: {e}")
    return models

# Sentiment color mapping
def sentiment_color(sentiment):
    return {
        'Negative': 'var(--negative)',
        'Neutral': 'var(--neutral)',
        'Positive': 'var(--positive)'
    }[sentiment]

# Streamlit app
def main():
    # Set page config FIRST - this must be the first Streamlit command
    st.set_page_config(
        page_title="Sentiment Analysis Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply CSS after page config
    st.markdown(CSS, unsafe_allow_html=True)
    
    # Header with logo
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998664.png", width=60)
    with col2:
        st.title("Sentiment Analysis Dashboard")
    st.caption("Analyze text sentiment with deep learning models | Fine-tune models with your dataset")

    # Create tabs for Prediction and Fine-Tuning
    tab1, tab2 = st.tabs(["🔮 Predict Sentiment", "⚙️ Fine-Tune Models"])

    # Prediction Tab
    with tab1:
        st.subheader("Model Selection")
        col1, col2 = st.columns(2)
        with col1:
            use_pretrained = st.toggle("Use Pre-Trained Models", value=True, key="use_pretrained_predict")
        with col2:
            debug_mode = st.toggle("Enable Debug Mode", value=False, key="debug_mode")
        
        tokenizer = load_tokenizer(use_pretrained)
        models = load_models(use_pretrained)
        
        available_models = list(models.keys())
        selected_models = st.multiselect(
            "Select Models to Use",
            ['RNN', 'LSTM', 'Bi-LSTM'],
            default=available_models if available_models else [],
            key="predict_models"
        )
        
        st.divider()
        st.subheader("Input Text")
        input_option = st.radio("Choose input method:", 
                               ["Enter Text", "Upload File"], 
                               horizontal=True)
        
        headlines = []
        if input_option == "Enter Text":
            headlines_input = st.text_area("Enter headlines (one per line)", height=150,
                                          placeholder="Enter your text here...\nExample:\nThe company reported record profits this quarter\nNew product launch delayed indefinitely")
            if headlines_input:
                headlines = [line.strip() for line in headlines_input.split('\n') if line.strip()]
        else:
            uploaded_file = st.file_uploader("Upload CSV/Excel (Max 200MB)", 
                                            type=['csv', 'xlsx'], 
                                            accept_multiple_files=False, 
                                            key="predict_uploader")
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    with st.expander("File Preview (First 5 rows)"):
                        st.dataframe(df.head())
                    
                    columns = df.columns.tolist()
                    text_column = st.selectbox("Select Text Column", columns, key="predict_text_column")
                    headlines = df[text_column].astype(str).tolist()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        if st.button("Analyze Sentiment", type="primary", use_container_width=True):
            if not headlines:
                st.warning("Please enter headlines or upload a file")
            elif not selected_models:
                st.warning("Please select at least one model")
            elif not models:
                st.error("No models loaded. Please ensure models exist or fine-tune models")
            else:
                # Progress bar
                progress_bar = st.progress(0)
                total_headlines = len(headlines)
                results = []
                
                for idx, headline in enumerate(headlines):
                    X_input = preprocess_single_text(headline, tokenizer, max_len=100)
                    headline_results = []
                    
                    for model_name in selected_models:
                        if model_name in models:
                            try:
                                model = models[model_name]
                                pred_proba = model.predict(X_input, verbose=0)
                                pred_label = np.argmax(pred_proba, axis=1)[0]
                                pred_class = ['Negative', 'Neutral', 'Positive'][pred_label]
                                pred_confidence = float(pred_proba[0][pred_label])
                                
                                headline_results.append({
                                    'model': model_name,
                                    'sentiment': pred_class,
                                    'confidence': pred_confidence,
                                    'probs': pred_proba[0].tolist()  # Store all probabilities
                                })
                            except Exception as e:
                                st.error(f"Error predicting with {model_name}: {e}")
                    
                    results.append({
                        'headline': headline,
                        'predictions': headline_results,
                        'X_input': X_input  # Store input for debug
                    })
                    
                    # Update progress
                    progress = (idx + 1) / total_headlines
                    progress_bar.progress(progress)
                
                # Display results
                st.divider()
                st.subheader("Analysis Results")
                
                # Debug info - show tokenization if enabled
                if debug_mode and tokenizer:
                    with st.expander("🔍 Tokenizer Information", expanded=True):
                        st.write(f"Tokenizer word index size: {len(tokenizer.word_index)}")
                        st.write(f"Tokenizer configuration: {tokenizer.get_config()}")
                
                for result in results:
                    with st.expander(f"**{result['headline']}**", expanded=False):
                        cols = st.columns(len(result['predictions']))
                        
                        for idx, pred in enumerate(result['predictions']):
                            with cols[idx]:
                                color = sentiment_color(pred['sentiment'])
                                st.markdown(f"""
                                <div class="card">
                                    <div class="card-title">
                                        <span class="model-icon">🧠</span> {pred['model']}
                                    </div>
                                    <div class="card-content">
                                        <strong style="color: {color};">{pred['sentiment']}</strong><br>
                                        Confidence: {pred['confidence']:.2%}
                                    </div>
                                    <div class="confidence-bar">
                                        <div class="confidence-level" style="width: {pred['confidence']*100:.0f}%; background: {color};"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Debug panel for detailed probabilities
                        if debug_mode:
                            st.markdown("---")
                            st.subheader("Debug Information")
                            
                            # Show tokenization details
                            st.write("**Tokenized Input:**")
                            non_zero_tokens = [str(tok) for tok in result['X_input'][0] if tok != 0]
                            st.code(f"Token IDs: {', '.join(non_zero_tokens)}")
                            
                            if tokenizer:
                                try:
                                    decoded_text = tokenizer.sequences_to_texts([result['X_input'][0]])[0]
                                    st.code(f"Decoded Text: {decoded_text}")
                                except:
                                    st.warning("Could not decode tokens to text")
                            
                            # Show detailed probabilities for each model
                            for pred in result['predictions']:
                                st.write(f"**{pred['model']} Probabilities:**")
                                
                                # Create probability bars
                                classes = ['Negative', 'Neutral', 'Positive']
                                colors = ['var(--negative)', 'var(--neutral)', 'var(--positive)']
                                
                                for i, cls in enumerate(classes):
                                    prob = pred['probs'][i]
                                    st.markdown(f"""
                                    <div class="prob-bar">
                                        <div class="prob-label" style="color: {colors[i]}">{cls}</div>
                                        <div class="prob-value" style="background: linear-gradient(90deg, {colors[i]} {prob*100:.1f}%, #333 {prob*100:.1f}%);">
                                            {prob:.4f}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

    # Fine-Tuning Tab
    with tab2:
        st.subheader("Dataset Configuration")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv", key="train_uploader")
        
        if uploaded_file is not None:
            try:
                df_preview = pd.read_csv(uploaded_file)
                with st.expander("Dataset Preview", expanded=True):
                    st.dataframe(df_preview.head())
                uploaded_file.seek(0)

                col1, col2 = st.columns(2)
                with col1:
                    text_column = st.text_input("Text Column Name", value="text")
                with col2:
                    sentiment_column = st.text_input("Sentiment Column Name", value="sentiment")

                st.subheader("Label Mapping")
                st.info("Map your sentiment labels to numerical values")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    neg_label = st.text_input("Negative Label", value="negative").lower()
                with col2:
                    neu_label = st.text_input("Neutral Label", value="neutral").lower()
                with col3:
                    pos_label = st.text_input("Positive Label", value="positive").lower()
                    
                sentiment_map = {
                    neg_label: 0,
                    neu_label: 1,
                    pos_label: 2
                }

                if st.button("Process Dataset", type="primary", key="load_data"):
                    df = load_data(uploaded_file, text_column, sentiment_column, sentiment_map)
                    st.session_state['df'] = df
                    st.success("Data processed successfully!")
                    st.write(f"Class distribution:")
                    st.bar_chart(df['sentiment'].value_counts())

                if 'df' in st.session_state:
                    df = st.session_state['df']
                    texts = df['text'].values
                    labels = df['sentiment'].values
                    
                    st.divider()
                    st.subheader("Training Configuration")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        max_words = st.slider("Vocabulary Size", 1000, 10000, 5000, 1000)
                        max_len = st.slider("Sequence Length", 50, 200, 100, 10)
                    with col2:
                        model_choice = st.selectbox("Model to Fine-Tune", ["RNN", "LSTM", "Bi-LSTM"])
                        epochs = st.slider("Training Epochs", 1, 20, 5, 1)
                    
                    if st.button("Start Fine-Tuning", type="primary", key="train_button"):
                        with st.spinner("Preprocessing data..."):
                            X, tokenizer = preprocess_text(texts, max_words, max_len)
                            vocab_size = min(len(tokenizer.word_index) + 1, max_words)
                            
                            # Save tokenizer
                            tokenizer_path = os.path.join(BASE_DIR, 'saved_models', 'tokenizer.pkl')
                            with open(tokenizer_path, 'wb') as f:
                                pickle.dump(tokenizer, f)
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
                        
                        # Load pre-trained model
                        pretrained_path = os.path.join(BASE_DIR, 'pretrained_models', f'{model_choice.lower().replace("-", "_")}_model.keras')
                        if not os.path.exists(pretrained_path):
                            st.error(f"Pre-trained {model_choice} model not found!")
                        else:
                            model = load_model(pretrained_path)
                            
                            st.info(f"Fine-tuning {model_choice} model with {len(X_train)} samples")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
                            
                            for epoch in range(epochs):
                                status_text.text(f"Epoch {epoch+1}/{epochs} in progress...")
                                history = model.fit(
                                    X_train, 
                                    y_train, 
                                    epochs=1,
                                    batch_size=32, 
                                    validation_split=0.2,
                                    callbacks=[early_stopping],
                                    verbose=0
                                )
                                # Update progress
                                progress = (epoch + 1) / epochs
                                progress_bar.progress(progress)
                            
                            # Save model
                            model_path = os.path.join(BASE_DIR, 'saved_models', f'{model_choice.lower().replace("-", "_")}_model.keras')
                            model.save(model_path)
                            
                            st.success(f"Fine-tuning complete! Model saved to {model_path}")
                            st.balloons()
            except Exception as e:
                st.error(f"Error processing dataset: {str(e)}")

if __name__ == "__main__":
    main()