# Sentiment Analysis Dashboard

![Dashboard Screenshot](https://via.placeholder.com/800x400.png?text=Sentiment+Analysis+Dashboard)

The Sentiment Analysis Dashboard is a powerful web application that leverages deep learning models to analyze text sentiment. Built with Streamlit and TensorFlow, this tool allows users to either predict sentiment using pre-trained models or fine-tune models with custom datasets for enhanced accuracy.

## Key Features

- 🧠 **Three Deep Learning Models**: RNN, LSTM, and Bi-LSTM architectures
- 🔮 **Real-time Sentiment Prediction**: Analyze text or files instantly
- ⚙️ **Model Fine-Tuning**: Customize models with your own datasets
- 🔍 **Debug Mode**: Detailed insights into model predictions
- 🎨 **Modern Dark UI**: Sleek, professional interface with visualizations
- 📊 **Interactive Visualizations**: Confidence bars and probability distributions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

### Predict Sentiment
1. Select between pre-trained models or your fine-tuned models
2. Choose models to use (RNN, LSTM, Bi-LSTM)
3. Enter text directly or upload a CSV/Excel file
4. Click "Analyze Sentiment" to see results
5. Enable Debug Mode for detailed prediction insights

### Fine-Tune Models
1. Upload your CSV dataset
2. Configure text and sentiment columns
3. Map sentiment labels to numerical values
4. Set vocabulary size and sequence length
5. Select a model to fine-tune
6. Start the fine-tuning process
7. Use your custom models for prediction

## Project Structure

```
sentiment-analysis-dashboard/
├── app.py                  # Main application file
├── utils.py                # Helper functions
├── requirements.txt        # Dependencies
├── saved_models/           # Custom models storage
├── pretrained_models/      # Pre-trained models
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## Dependencies

- Python 3.8+
- Streamlit
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- NLTK

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues, please [open an issue](https://github.com/yourusername/sentiment-analysis-dashboard/issues) on GitHub.

---

**Created with ❤️ by Your Name**  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat&logo=github)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile)
