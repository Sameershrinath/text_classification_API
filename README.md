# 📧 Spam Email/SMS Classifier

A machine learning-powered spam detection application built with Streamlit and scikit-learn using Multinomial Naive Bayes algorithm.

## 🚀 Features

- **Single Message Classification**: Classify individual emails or SMS messages
- **Batch Processing**: Upload CSV files or process multiple messages at once
- **Real-time Predictions**: Instant spam detection with confidence scores
- **Interactive Examples**: Pre-loaded examples of spam and legitimate messages
- **Text Preprocessing**: Advanced NLP preprocessing pipeline
- **Model Performance**: High accuracy spam detection with optimized precision
- **User-friendly Interface**: Clean, intuitive Streamlit web interface

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Algorithm**: Multinomial Naive Bayes
- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Text Processing**: NLTK (Natural Language Toolkit)
- **Data Handling**: Pandas, NumPy
- **Model Persistence**: Pickle

## 📊 Model Details

### Preprocessing Pipeline
1. **Lowercase conversion**: Normalize text case
2. **Tokenization**: Split text into individual words
3. **Alphanumeric filtering**: Remove special characters
4. **Stopword removal**: Remove common English words
5. **Stemming**: Reduce words to their root form using Porter Stemmer
6. **TF-IDF Vectorization**: Convert text to numerical features (max 3000 features)

### Algorithm
- **Multinomial Naive Bayes**: Optimized for text classification
- **Training**: Trained on SMS Spam Collection dataset
- **Performance**: High precision to minimize false positives

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Install Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### 2. Model Setup
First, make sure you have the trained model files:
- `model.pkl` - Trained Multinomial Naive Bayes model
- `tfidf.pkl` - Fitted TF-IDF vectorizer

**Option A: If you have the model files**
- Place `model.pkl` and `tfidf.pkl` in the project directory

**Option B: If you need to train the model**
1. Run your `Model_training.ipynb` notebook
2. The notebook will generate `model.pkl` and `tfidf.pkl` files
3. Or run: `python setup_model.py` to check model files

### 3. Run the Application
```bash
# Method 1: Using batch file (Windows)
run_app.bat

# Method 2: Manual command
streamlit run Main.py
```

### 4. Access the App
Open your browser and go to: `http://localhost:8501`

## 📁 Project Structure

```
Spam_classifier/
├── Main.py                    # Main Streamlit application
├── Model_training.ipynb       # Model training notebook
├── setup_model.py            # Model setup and verification script
├── requirements.txt          # Python dependencies
├── run_app.bat              # Windows batch file to run the app
├── README.md                # This file
├── model.pkl                # Trained model (generated)
├── tfidf.pkl                # TF-IDF vectorizer (generated)
└── Data/                    # Dataset folder
    └── spam.csv             # Training dataset
```

## 🎯 How to Use

### Single Message Classification
1. Go to the "🔍 Single Message" tab
2. Enter your email or SMS message in the text area
3. Click "🔍 Classify Message"
4. View the prediction result with confidence score

### Batch Processing
1. Go to the "📄 Batch Processing" tab
2. **File Upload**: Upload a CSV file with messages
3. **Manual Input**: Enter multiple messages (one per line)
4. Process all messages and download results

### Examples
1. Go to the "📈 Examples" tab
2. Explore pre-loaded spam and legitimate message examples
3. See how the model classifies different types of messages

## 📊 Sample Predictions

### Spam Examples
- "FREE! Win a £1000 cash prize! Text WIN to 85233 now!" → **SPAM** (95% confidence)
- "URGENT! Your account will be closed. Click here to verify immediately." → **SPAM** (92% confidence)

### Ham (Legitimate) Examples
- "Hi, are we still meeting for lunch tomorrow at 12pm?" → **HAM** (98% confidence)
- "Your appointment is confirmed for Monday at 3pm." → **HAM** (97% confidence)

## 🔧 Technical Details

### Model Training Process
1. **Data Loading**: SMS Spam Collection dataset
2. **Data Cleaning**: Remove unnecessary columns, handle missing values
3. **Label Encoding**: Convert spam/ham to 1/0
4. **Text Preprocessing**: Apply full preprocessing pipeline
5. **Feature Extraction**: TF-IDF vectorization with 3000 features
6. **Model Training**: Multinomial Naive Bayes
7. **Evaluation**: Accuracy, precision, recall metrics
8. **Model Saving**: Pickle serialization

### Key Functions
- `transform_text()`: Applies the complete preprocessing pipeline
- `predict_spam()`: Makes predictions using the trained model
- `load_model()`: Loads saved model and vectorizer (cached)

## 📋 Dependencies

```
streamlit==1.28.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.25.2
nltk==3.8.1
pickle-mixin==1.0.2
```

## 🎨 Features Overview

### 🔍 Single Message Tab
- Text area for message input
- Real-time classification
- Confidence scores for both spam and ham
- Processed text visualization
- Step-by-step preprocessing explanation

### 📄 Batch Processing Tab
- CSV file upload with column selection
- Manual multi-message input
- Batch prediction results
- Summary statistics
- Downloadable results

### 📈 Examples Tab
- Pre-loaded spam examples
- Pre-loaded legitimate message examples
- Real-time prediction on examples
- Processed text visualization

## ⚠️ Important Notes

- **Educational Purpose**: This application is for educational and demonstration purposes
- **Model Limitations**: Trained on SMS/email data, may not work perfectly on all message types
- **Manual Verification**: Always verify important messages manually
- **Context Matters**: Consider the sender and context when evaluating messages
- **Performance**: Model accuracy depends on training data quality and diversity

## 🛡️ Security Considerations

- The model processes text locally (no external API calls)
- No user data is stored or transmitted
- All processing happens in your local environment
- Model files are loaded once and cached for performance

## 🔄 Model Retraining

To retrain the model with new data:
1. Update the dataset in `Data/spam.csv`
2. Run the `Model_training.ipynb` notebook
3. New `model.pkl` and `tfidf.pkl` files will be generated
4. Restart the Streamlit app

## 📞 Troubleshooting

### Common Issues

1. **"Model files not found"**
   - Ensure `model.pkl` and `tfidf.pkl` are in the same directory
   - Run `python setup_model.py` to check

2. **"NLTK data not found"**
   - The app will automatically download required NLTK data
   - Or manually run: `nltk.download('punkt')` and `nltk.download('stopwords')`

3. **"Import errors"**
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ recommended)

4. **"Streamlit not found"**
   - Install Streamlit: `pip install streamlit`
   - Verify installation: `streamlit --version`

## 🎓 Learning Resources

- [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [NLTK Documentation](https://www.nltk.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📈 Performance Metrics

The model achieves:
- **High Precision**: Minimizes false positives (legitimate messages marked as spam)
- **Good Recall**: Identifies most spam messages
- **Balanced Accuracy**: Performs well on both spam and legitimate messages

## 🤝 Contributing

To improve the model:
1. Add more diverse training data
2. Experiment with different algorithms
3. Enhance preprocessing techniques
4. Improve the user interface
5. Add more features (email headers, metadata analysis)

---

**Built with ❤️ for spam detection and email security education**
