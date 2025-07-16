import streamlit as st
import requests
import pandas as pd
import json

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"
# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #4ECDC4;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    .spam-result {
        background-color: #FFE5E5;
        color: #D32F2F;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin-top: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .Not_Spam-result {
        background-color: #E8F5E8;
        color: #2E7D32;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-top: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin-bottom: 2rem;
        color:#000000;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin-bottom: 1rem;
        color:#000000;
    }
    .example-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #9E9E9E;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin-bottom: 1rem;
        color:#000000;
    }
</style>
""", unsafe_allow_html=True)

def check_api_connection():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_single_message(text):
    """Make a prediction for a single message"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection Error: {str(e)}"

def predict_batch_messages(messages):
    """Make predictions for multiple messages"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"messages": messages},
            timeout=30
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection Error: {str(e)}"

def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def get_examples():
    """Get example messages from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/examples", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Main header
st.markdown('<h1 class="main-header">📧 Spam Email Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect spam emails using Machine Learning</p>', unsafe_allow_html=True)

# Check API connection
if not check_api_connection():
    st.markdown("""
    <div class="error-box">
        <h3>⚠️ API Connection Error</h3>
        <p>Unable to connect to the backend API. Please make sure:</p>
        <ul>
            <li>The backend server is running on <code>http://localhost:8000</code></li>
            <li>Run the backend with: <code>python backend.py</code> or <code>uvicorn backend:app --reload</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Get model information
model_info = get_model_info()

# Sidebar information
st.sidebar.title("📊 Model Information")
if model_info:
    st.sidebar.markdown(f"""
    **Model Details:**
    - **Algorithm**: {model_info.get('model_type', 'Unknown')}
    - **Vectorizer**: {model_info.get('vectorizer', 'Unknown')}
    - **Preprocessing**: {model_info.get('preprocessing', 'Unknown')}
    - **Training Data**: {model_info.get('training_data', 'Unknown')}

    **Status:**
    - Model Loaded: {"✅" if model_info.get('model_loaded') else "❌"}
    
    **Performance:**
    - High accuracy in spam detection
    - Optimized for precision to minimize false positives
    """)
else:
    st.sidebar.error("Unable to load model information")

# Info box
st.markdown("""
<div class="info-box">
    <h3>🔍 How it works:</h3>
    <p>This classifier uses Natural Language Processing and Machine Learning to analyze text messages and emails. 
    It processes the text through several steps including tokenization, stemming, and feature extraction using TF-IDF, 
    then uses a Multinomial Naive Bayes algorithm to classify the message as spam or legitimate (Not_Spam).</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["🔍 Single Message", "📄 Batch Processing", "📈 Examples"])

with tab1:
    st.header("Enter a Message to Classify")
    
    # Text input area
    input_text = st.text_area(
        "Enter your message here:",
        height=150,
        placeholder="Type your email or SMS message here...",
        help="Enter the complete message you want to classify"
    )
    
    # Prediction button
    if st.button("🔍 Classify Message", type="primary", use_container_width=True):
        if input_text.strip():
            with st.spinner("Analyzing message..."):
                result, error = predict_single_message(input_text)
                
                if error:
                    st.error(f"Error: {error}")
                elif result:
                    # Display results
                    if result['prediction'] == 'Spam':
                        st.markdown(f"""
                        <div class="spam-result">
                            <h3>🚨 SPAM DETECTED!</h3>
                            <p>This message is classified as <strong>SPAM</strong></p>
                            <p>Confidence: {result['spam_probability']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="Not_Spam-result">
                            <h3>✅ LEGITIMATE MESSAGE</h3>
                            <p>This message is classified as <strong>Not_Spam</strong></p>
                            <p>Confidence: {result['not_spam_probability']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show prediction probabilities
                    st.subheader("📊 Prediction Probabilities")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Not_Spam (Not Spam)", f"{result['not_spam_probability']:.2%}")
                    
                    with col2:
                        st.metric("Spam", f"{result['spam_probability']:.2%}")
                    
                    # Show processed text
                    with st.expander("🔧 Processed Text", expanded=False):
                        st.text("Processed text:")
                        st.code(result['processed_text'], language="text")
                        
                        st.markdown("**Processing Steps:**")
                        st.markdown("""
                        1. Convert to lowercase
                        2. Tokenize words
                        3. Remove non-alphanumeric characters
                        4. Remove stopwords and punctuation
                        5. Apply stemming
                        6. Vectorize using TF-IDF
                        """)
        else:
            st.warning("Please enter a message to classify.")

with tab2:
    st.header("Batch Processing")
    st.markdown("Upload a CSV file or enter multiple messages for batch classification.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Select column for classification
            text_column = st.selectbox("Select the column containing text messages:", df.columns)
            
            if st.button("🔍 Classify All Messages"):
                with st.spinner("Processing all messages..."):
                    messages = [str(text) for text in df[text_column].tolist()]
                    result, error = predict_batch_messages(messages)
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif result:
                        # Process results
                        predictions = []
                        probabilities = []
                        
                        for res in result['results']:
                            predictions.append(res['prediction'])
                            probabilities.append(res['spam_probability'])
                        
                        # Add results to dataframe
                        df['Prediction'] = predictions
                        df['Spam_Probability'] = probabilities
                        
                        # Display results
                        st.subheader("📊 Results")
                        st.dataframe(df)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Spam Messages", result['summary']['spam_count'])
                        
                        with col2:
                            st.metric("Not_Spam Messages", result['summary']['not_spam_count'])
                        
                        with col3:
                            st.metric("Total Messages", result['summary']['total_messages'])
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="spam_classification_results.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Manual batch input
    st.subheader("Manual Batch Input")
    batch_text = st.text_area(
        "Enter multiple messages (one per line):",
        height=200,
        placeholder="Message 1\nMessage 2\nMessage 3\n..."
    )
    
    if st.button("🔍 Classify Batch Messages"):
        if batch_text.strip():
            messages = [msg.strip() for msg in batch_text.split('\n') if msg.strip()]
            
            with st.spinner("Processing batch messages..."):
                result, error = predict_batch_messages(messages)
                
                if error:
                    st.error(f"Error: {error}")
                elif result:
                    # Display results
                    results_data = []
                    for res in result['results']:
                        message = res['message']
                        results_data.append({
                            "Message #": res['message_number'],
                            "Message": message[:50] + "..." if len(message) > 50 else message,
                            "Prediction": res['prediction'],
                            "Spam Probability": f"{res['spam_probability']:.2%}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df)
                    
                    # Summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Spam Messages", result['summary']['spam_count'])
                    with col2:
                        st.metric("Not_Spam Messages", result['summary']['not_spam_count'])

with tab3:
    st.header("📈 Example Messages")
    
    # Get examples from API
    examples = get_examples()
    
    if examples:
        # Display examples with predictions
        spam_examples = examples.get('spam_examples', [])
        not_spam_examples = examples.get('not_spam_examples', [])
        
        # Spam examples
        if spam_examples:
            st.subheader("🔍 Spam Examples")
            
            for i, example in enumerate(spam_examples, 1):
                with st.expander(f"Example {i}: {example[:50]}..."):
                    st.write(f"**Full Message:** {example}")
                    
                    # Get prediction for example
                    result, error = predict_single_message(example)
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif result:
                        if result['prediction'] == 'Spam':
                            st.markdown(f"""
                            <div class="spam-result">
                                <strong>Prediction: SPAM</strong><br>
                                Confidence: {result['spam_probability']:.2%}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="Not_Spam-result">
                                <strong>Prediction: Not_Spam</strong><br>
                                Confidence: {result['not_spam_probability']:.2%}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show processed text
                        st.text("Processed text:")
                        st.code(result['processed_text'], language="text")
        
        # Not spam examples
        if not_spam_examples:
            st.subheader("🔍 Not_Spam (Legitimate) Examples")
            
            for i, example in enumerate(not_spam_examples, 1):
                with st.expander(f"Example {i}: {example[:50]}..."):
                    st.write(f"**Full Message:** {example}")
                    
                    # Get prediction for example
                    result, error = predict_single_message(example)
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif result:
                        if result['prediction'] == 'Spam':
                            st.markdown(f"""
                            <div class="spam-result">
                                <strong>Prediction: SPAM</strong><br>
                                Confidence: {result['spam_probability']:.2%}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="Not_Spam-result">
                                <strong>Prediction: Not_Spam</strong><br>
                                Confidence: {result['not_spam_probability']:.2%}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show processed text
                        st.text("Processed text:")
                        st.code(result['processed_text'], language="text")
    else:
        st.error("Unable to load examples from API")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🛡️ This spam classifier is for educational and demonstration purposes.</p>
    <p>Built with Streamlit and FastAPI | Powered by Multinomial Naive Bayes</p>
</div>
""", unsafe_allow_html=True)

# Warning box
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Notes:</h4>
    <ul>
        <li>This model is trained on SMS/email data and may not work perfectly on all types of messages</li>
        <li>Always verify important messages manually, especially if they seem suspicious</li>
        <li>The model's performance depends on the quality and diversity of training data</li>
        <li>Consider the context and sender when evaluating messages</li>
        <li>Make sure the backend API is running before using the application</li>
    </ul>
</div>
""", unsafe_allow_html=True)
