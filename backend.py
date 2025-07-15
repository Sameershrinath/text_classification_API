from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Global variables to store model and vectorizer
model = None
tfidf = None

def load_model():
    """Load the trained model and vectorizer"""
    global model, tfidf
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open('tfidf.pkl', 'rb') as tfidf_file:
            tfidf = pickle.load(tfidf_file)
        
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        return False


# Initialize FastAPI app
app = FastAPI(
    title="Spam Classifier API",
    description="API for spam message classification using Machine Learning",
    version="1.0.0"
)

# Load the model when the application starts
if not load_model():
    raise RuntimeError("Failed to load model files. Please ensure 'model.pkl' and 'tfidf.pkl' are in the same directory.")

if model is None or tfidf is None:
    raise HTTPException(status_code=500, detail="Model not loaded properly")

# Pydantic models for request/response
class MessageRequest(BaseModel):
    text: str

class BatchMessageRequest(BaseModel):
    messages: List[str]

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    spam_probability: float
    not_spam_probability: float
    processed_text: str

class BatchPredictionResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, int]

def transform_text(text: str) -> str:
    """
    Transform text using the same preprocessing steps as in model training
    """
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y.copy()
    y.clear()
    
    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y.copy()
    y.clear()
    
    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

def predict_spam(text: str) -> tuple:
    """Predict if the text is spam or not spam"""
    global model, tfidf
    
    if model is None or tfidf is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Transform the text
    transformed_text = transform_text(text)
    
    # Vectorize the text
    vectorized_text = tfidf.transform([transformed_text])
    
    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0]
    
    return prediction, probability, transformed_text

@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "Spam Classifier API is running!"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None and tfidf is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_single_message(request: MessageRequest):
    """Predict if a single message is spam or not"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty message provided")
        
        prediction, probability, processed_text = predict_spam(request.text)
        
        return PredictionResponse(
            prediction="Spam" if prediction == 1 else "Not_Spam",
            probability=probability[1] if prediction == 1 else probability[0],
            spam_probability=probability[1],
            not_spam_probability=probability[0],
            processed_text=processed_text
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_messages(request: BatchMessageRequest):
    """Predict spam for multiple messages"""
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        results = []
        spam_count = 0
        not_spam_count = 0
        error_count = 0
        
        for i, message in enumerate(request.messages, 1):
            try:
                if not message.strip():
                    results.append({
                        "message_number": i,
                        "message": message,
                        "prediction": "Error",
                        "spam_probability": 0.0,
                        "not_spam_probability": 0.0,
                        "processed_text": "",
                        "error": "Empty message"
                    })
                    error_count += 1
                    continue
                
                prediction, probability, processed_text = predict_spam(message)
                
                pred_label = "Spam" if prediction == 1 else "Not_Spam"
                
                results.append({
                    "message_number": i,
                    "message": message,
                    "prediction": pred_label,
                    "spam_probability": probability[1],
                    "not_spam_probability": probability[0],
                    "processed_text": processed_text,
                    "error": None
                })
                
                if prediction == 1:
                    spam_count += 1
                else:
                    not_spam_count += 1
                    
            except Exception as e:
                results.append({
                    "message_number": i,
                    "message": message,
                    "prediction": "Error",
                    "spam_probability": 0.0,
                    "not_spam_probability": 0.0,
                    "processed_text": "",
                    "error": str(e)
                })
                error_count += 1
        
        return BatchPredictionResponse(
            results=results,
            summary={
                "total_messages": len(request.messages),
                "spam_count": spam_count,
                "not_spam_count": not_spam_count,
                "error_count": error_count
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch prediction: {str(e)}")

@app.get("/model/info")
def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "Multinomial Naive Bayes",
        "vectorizer": "TF-IDF (max 3000 features)",
        "preprocessing": "Tokenization, Stemming, Stop word removal",
        "training_data": "SMS Spam Collection Dataset",
        "model_loaded": model is not None and tfidf is not None
    }

@app.get("/examples")
def get_examples():
    """Get example messages for testing"""
    return {
        "spam_examples": [
            "FREE! Win a £1000 cash prize! Text WIN to 85233 now!",
            "URGENT! Your account will be closed. Click here to verify immediately.",
            "Congratulations! You've won a free iPhone! Claim now at freephone.com",
            "Hot singles in your area! Meet them tonight! Register free!",
            "WINNER! You've been selected for a £500 shopping voucher. Reply STOP to opt out."
        ],
        "not_spam_examples": [
            "Hi, are we still meeting for lunch tomorrow at 12pm?",
            "Your appointment is confirmed for Monday at 3pm. Please arrive 10 minutes early.",
            "Thanks for your help today. The presentation went really well!",
            "Can you pick up some milk on your way home? Thanks!",
            "Meeting has been rescheduled to 2pm in conference room B."
        ]
    }

