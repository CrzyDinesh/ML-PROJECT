import pandas as pd
import re
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==================== SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

st.set_page_config(page_title="📩 SMS Spam Classifier", layout="wide")
st.title("📩 SMS Spam Classifier")
st.markdown("*Classify SMS messages as spam or legitimate using machine learning*")

# ==================== CONSTANTS ====================
MODEL_PATH = "spam_model.pkl"
VECT_PATH = "vectorizer.pkl"
# Updated default path to your file location
DEFAULT_FILE_PATH = r"C:\2 year\binary game\spamming\sms-spam-classifier-main\SMSSpamCollection"

# ==================== LOAD DATA ====================
@st.cache_data(show_spinner=False)
def load_data(file_path):
    """Load and validate the SMS dataset"""
    try:
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        
        df = pd.read_csv(file_path, sep='\t', names=['labels', 'messages'], encoding='latin-1')
        
        # Validate data
        if df.empty:
            return None, "Dataset is empty"
        if 'labels' not in df.columns or 'messages' not in df.columns:
            return None, "Invalid dataset format"
        
        return df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

# File upload option
st.sidebar.header("📁 Data Source")
upload_option = st.sidebar.radio("Choose data source:", ["Use Local Path", "Upload File"])

df = None
error = None

if upload_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload SMSSpamCollection file", type=['txt', 'tsv', 'csv'])
    if uploaded_file:
        df, error = load_data(uploaded_file)
    else:
        st.info("👆 Please upload your dataset file in the sidebar")
        st.stop()
else:
    file_path = st.sidebar.text_input("Enter file path:", value=DEFAULT_FILE_PATH)
    if file_path:
        df, error = load_data(file_path)
    else:
        st.warning("⚠️ Please enter a file path")
        st.stop()

if error:
    st.error(f"❌ {error}")
    st.info("💡 **Tip**: Upload the file using the sidebar or check your file path")
    st.stop()

st.success(f"✅ Dataset loaded successfully!")
st.info(f"📊 Total messages: {len(df)}")

# ==================== OVERVIEW ====================
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
spam_count = (df['labels'] == 'spam').sum()
ham_count = (df['labels'] == 'ham').sum()
spam_pct = (spam_count / len(df)) * 100

col1.metric("Total Messages", len(df))
col2.metric("Spam", spam_count)
col3.metric("Ham", ham_count)
col4.metric("Spam %", f"{spam_pct:.1f}%")

# Show sample messages
if st.checkbox("Show sample messages"):
    st.write("**Sample Spam Messages:**")
    st.dataframe(df[df['labels'] == 'spam']['messages'].head(3), use_container_width=True)
    st.write("**Sample Ham Messages:**")
    st.dataframe(df[df['labels'] == 'ham']['messages'].head(3), use_container_width=True)

# ==================== PREPROCESSING ====================
def clean_message(msg):
    """Clean and preprocess text message"""
    msg = re.sub('[^a-zA-Z]', ' ', str(msg))
    msg = msg.lower().split()
    msg = [stemmer.stem(w) for w in msg if w not in stop_words and len(w) > 2]
    return ' '.join(msg)

@st.cache_data(show_spinner=False)
def preprocess(df):
    """Apply preprocessing to all messages"""
    df = df.copy()
    df['cleaned'] = df['messages'].apply(clean_message)
    return df

with st.spinner("🔄 Preprocessing messages..."):
    df = preprocess(df)
    st.write("✅ Preprocessing complete")

# ==================== MODEL TRAINING ====================
@st.cache_resource(show_spinner=False)
def train_and_evaluate(_df):
    """Train model and return all necessary components"""
    # Prepare data
    y = (_df['labels'] == 'spam').astype(int).values
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        _df['cleaned'], y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)
    
    # Balance training data
    rus = RandomUnderSampler(random_state=42)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
    
    # Train model
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_bal, y_train_bal)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    return model, vectorizer, X_test, y_test, y_pred

# Train or load model
if st.sidebar.checkbox("Retrain Model", value=False):
    with st.spinner("🔄 Training model..."):
        model, vectorizer, X_test, y_test, y_pred = train_and_evaluate(df)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECT_PATH)
        st.sidebar.success("✅ Model trained and saved!")
elif os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    # Still need to prepare test data for evaluation
    _, _, X_test, y_test, y_pred = train_and_evaluate(df)
    st.sidebar.info("📁 Loaded saved model")
else:
    with st.spinner("🔄 Training model (first time)..."):
        model, vectorizer, X_test, y_test, y_pred = train_and_evaluate(df)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECT_PATH)
        st.sidebar.success("✅ Model trained!")

# ==================== EVALUATION ====================
st.subheader("📈 Model Performance")

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Display metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{acc*100:.2f}%")
col2.metric("Precision", f"{precision*100:.2f}%")
col3.metric("Recall", f"{recall*100:.2f}%")
col4.metric("F1-Score", f"{f1*100:.2f}%")
col5.metric("Specificity", f"{specificity*100:.2f}%")

# Detailed metrics
with st.expander("📋 Detailed Classification Report"):
    st.text(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix
col1, col2 = st.columns(2)

with col1:
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    st.pyplot(fig)
    plt.close()

with col2:
    st.write("**Metrics Breakdown**")
    metrics_data = {
        'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
        'Count': [tp, tn, fp, fn],
        'Description': [
            'Correctly identified spam',
            'Correctly identified ham',
            'Ham classified as spam',
            'Spam classified as ham'
        ]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

# ==================== USER INPUT ====================
st.subheader("💬 Test Your Own Message")
st.write("Enter an SMS message below to classify it as spam or legitimate.")

user_msg = st.text_area("Message:", height=100, placeholder="e.g., Congratulations! You've won a free iPhone. Click here to claim...")

col1, col2 = st.columns([1, 5])
with col1:
    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.rerun()

if classify_btn:
    if user_msg.strip():
        try:
            # Preprocess and predict
            msg_clean = clean_message(user_msg)
            msg_vec = vectorizer.transform([msg_clean])
            pred = model.predict(msg_vec)[0]
            conf = model.predict_proba(msg_vec)[0]
            
            # Display result
            st.markdown("---")
            if pred == 1:
                st.error(f"🚨 **SPAM DETECTED**")
                confidence = conf[1] * 100
                st.metric("Confidence", f"{confidence:.1f}%")
                if confidence > 90:
                    st.warning("⚠️ High confidence spam - likely malicious")
            else:
                st.success(f"✅ **LEGITIMATE MESSAGE**")
                confidence = conf[0] * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show processed text
            with st.expander("🔍 View Processed Text"):
                st.code(msg_clean, language=None)
                st.caption(f"Original length: {len(user_msg)} chars | Processed length: {len(msg_clean)} chars")
                
        except Exception as e:
            st.error(f"❌ Error during classification: {str(e)}")
    else:
        st.warning("⚠️ Please enter a message to classify.")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Built with Streamlit | Model: Multinomial Naive Bayes | Features: TF-IDF")
