import streamlit as st
import requests
import torch
import joblib
import re
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ddgs import DDGS
from firecrawl import Firecrawl

# ------------------ FIRECRAWL CONFIG ------------------
firecrawl = Firecrawl(api_key="fc-1f9c34727ff04d6883b79a7b51f494f8")

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(
    page_title="Political Bias Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: #e0e7ff;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .result-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .result-content {
        font-size: 1.1rem;
        color: #475569;
    }
    .info-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border: 1px solid #fde68a;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1fae5;
        border: 1px solid #a7f3d0;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    div[data-testid="stRadio"] > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
    }
    .stTextInput>label, .stTextArea>label {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
    }
    hr {
        border: none;
        border-top: 2px solid #e2e8f0;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>Political Bias Analyzer</h1>
        <p>Analyze political bias of news articles using BERT, SVM, and Random Forest models</p>
    </div>
""", unsafe_allow_html=True)

# ------------------ PREPROCESSING ------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

    # Basic cleanup
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"\s+", " ", text)  # normalize spaces

    # Keep sentence structure but remove non-essential characters
    text = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\s-]", "", text)

    # Lowercase but preserve meaning
    text = text.lower()

    # Lemmatize while keeping important short words
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words or len(w) > 2]
    return " ".join(words)

# ------------------ MODEL LOAD ------------------
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)
    bert_model.load_state_dict(torch.load("bert_weights/bert_type_classifier.pt", map_location=torch.device("cpu")), strict=False)
    bert_model.eval()

    svm_bias_vectorizer = joblib.load("models/svm_bias_vectorizer.joblib")
    svm_model_bias = joblib.load("models/svm_model_bias.pkl")
    svm_vectorizer_subtype = joblib.load("models/svm_vectorizer_subtype.pkl")
    svm_model_subtype = joblib.load("models/svm_model_subtype.pkl")

    rf_vectorizer_bias = joblib.load("models/seperate_vectorizer1.joblib")
    rf_model_bias = joblib.load("models/seperate_bias_model.pkl")
    rf_vectorizer_subtype = joblib.load("models/seperate_vectorizer2.joblib")
    rf_model_subtype = joblib.load("models/seperate_subtype_model.pkl")

    return (
        tokenizer,
        bert_model,
        svm_bias_vectorizer,
        svm_model_bias,
        svm_vectorizer_subtype,
        svm_model_subtype,
        rf_vectorizer_bias,
        rf_model_bias,
        rf_vectorizer_subtype,
        rf_model_subtype,
    )

(
    tokenizer,
    bert_model,
    svm_bias_vectorizer,
    svm_model_bias,
    svm_vectorizer_subtype,
    svm_model_subtype,
    rf_vectorizer_bias,
    rf_model_bias,
    rf_vectorizer_subtype,
    rf_model_subtype,
) = load_models()

# ------------------ LABEL MAPPINGS ------------------
bert_id2label = {0: "Left", 1: "Right", 2: "Center"}
svm_bias_id2label = {0: "Left", 1: "Center", 2: "Right"}
svm_bias_label2id = {"Left": 0, "Center": 1, "Right": 2}
svm_subtype_id2label = {
    0: "Conservative",
    1: "Capitalist",
    2: "Nationalist",
    3: "Center",
    4: "Liberal",
    5: "Secular",
    6: "Socialist",
}
rf_bias_id2label = svm_bias_id2label
rf_subtype_id2label = svm_subtype_id2label

# ------------------ DUCKDUCKGO SEARCH ------------------
def ddg_news_search(query: str, top_k: int = 5):
    """Fetch top news results from DuckDuckGo instead of Brave."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=top_k, region="us-en"):
                title = r.get("title") or "Untitled"
                url = r.get("href") or r.get("url") or ""
                if title and url:
                    results.append({"title": title, "url": url})
        return results
    except Exception:
        return []

# ------------------ FIRECRAWL SCRAPER ------------------
def fetch_article_text(url):
    try:
        with st.spinner("Fetching article content..."):
            doc = firecrawl.scrape(url, formats=["markdown", "html", "summary"])

            # Join markdown into one string
            article_text = " ".join(doc.markdown) if isinstance(doc.markdown, list) else str(doc.markdown)
            summary = doc.summary if hasattr(doc, "summary") else ""
            
            return article_text, summary
    except Exception as e:
        st.markdown(f'<div class="warning-box"><strong>Error:</strong> Failed to fetch article - {str(e)}</div>', unsafe_allow_html=True)
        return "", ""


# ------------------ MODEL PREDICTIONS ------------------
def predict_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    return bert_id2label[pred], conf

def predict_with_svm(clean_input, pred):
    if not clean_input:
        st.markdown('<div class="warning-box">Input text is empty for SVM prediction.</div>', unsafe_allow_html=True)
        return "Unknown", "Unknown"
    
    svm_bias_pred = svm_bias_label2id.get(pred, 0)
    svm_subtype_features = svm_vectorizer_subtype.transform([clean_input]).toarray()
    svm_combined = np.hstack((svm_subtype_features, np.array([[svm_bias_pred]])))
    svm_subtype_pred = svm_model_subtype.predict(svm_combined)[0]

    return (
        svm_bias_id2label.get(svm_bias_pred, "Unknown"),
        svm_subtype_id2label.get(svm_subtype_pred, "Unknown"),
    )

def predict_with_rf(clean_input):
    if not clean_input:
        st.markdown('<div class="warning-box">Input text is empty for RF prediction.</div>', unsafe_allow_html=True)
        return "Unknown", "Unknown"
    
    rf_bias_features = rf_vectorizer_bias.transform([clean_input])
    rf_bias_pred = rf_model_bias.predict(rf_bias_features)[0]

    rf_subtype_features = rf_vectorizer_subtype.transform([clean_input])
    rf_combined = hstack((rf_subtype_features, np.array([[rf_bias_pred]])))
    rf_subtype_pred = rf_model_subtype.predict(rf_combined)[0]

    return (
        rf_bias_id2label.get(rf_bias_pred, "Unknown"),
        rf_subtype_id2label.get(rf_subtype_pred, "Unknown"),
    )

# ------------------ MAIN UI ------------------
st.markdown("## Select Input Method")
st.markdown("---")

option = st.radio(
    "Choose how you want to provide the article:",
    ("Paste Article Link", "Manual Text Entry", "Search News Topic"),
    help="Select your preferred method to analyze an article"
)

if option == "Search News Topic":
    st.markdown("### Search for News Articles")
    topic = st.text_input(
        "Enter a political topic:",
        placeholder="e.g., 'gun control', 'immigration', 'climate change'",
        help="Search for recent news articles on a specific topic"
    )

    if st.button("Search News", key="search_btn"):
        if topic.strip():
            with st.spinner("Searching for articles..."):
                results = ddg_news_search(topic)
                
            if results:
                st.markdown('<div class="success-box"><strong>Search Results:</strong> Found ' + str(len(results)) + ' articles</div>', unsafe_allow_html=True)
                st.markdown("### Top Articles")
                
                for idx, r in enumerate(results, 1):
                    st.markdown(f"{idx}. [{r['title']}]({r['url']})", unsafe_allow_html=True)
                
                st.markdown('<div class="info-box">Click on an article link above, then copy the URL and use the "Paste Article Link" option to analyze it.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">No articles found. Try a different search term.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">Please enter a search topic.</div>', unsafe_allow_html=True)


elif option == "Paste Article Link":
    st.markdown("### Analyze Article from URL")
    
    input_url = st.text_input(
        "Paste the article URL:",
        placeholder="https://example.com/news-article",
        help="Enter the full URL of the news article you want to analyze"
    )

    # Initialize session variables once
    if "article_text" not in st.session_state:
        st.session_state.article_text = ""
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "url_fetched" not in st.session_state:
        st.session_state.url_fetched = ""

    fetch_clicked = st.button("Fetch Article", key="fetch_btn")

    # Fetch text when button clicked
    if fetch_clicked and input_url.strip():
        text, summary = fetch_article_text(input_url)

        if text:
            st.session_state.article_text = clean_text(text)
            st.session_state.summary = summary
            st.session_state.url_fetched = input_url
            st.markdown('<div class="success-box"><strong>Success:</strong> Article fetched and processed successfully.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box"><strong>Warning:</strong> No text could be extracted from the URL.</div>', unsafe_allow_html=True)
    
    # Display summary if it exists in session state
    if st.session_state.get("summary", "").strip():
        st.markdown('<div class="info-box"><strong>Article Summary:</strong><br>' + st.session_state.summary + '</div>', unsafe_allow_html=True)



elif option == "Manual Text Entry":
    st.markdown("### Paste Article Text Directly")
    article_text = st.text_area(
        "Enter or paste the article text:",
        height=200,
        placeholder="Paste the full article text here...",
        help="Paste the complete article text for analysis"
    )
    
    if article_text.strip():
        st.session_state.article_text = clean_text(article_text)
        st.markdown(f'<div class="info-box"><strong>Text Length:</strong> {len(article_text)} characters</div>', unsafe_allow_html=True)

# ------------------ RUN ANALYSIS ------------------
st.markdown("---")
st.markdown("## Analysis")

if st.button("Analyze Article", key="analyze_btn", type="primary"):
    article_text = st.session_state.get("article_text", "").strip()
    
    if not article_text:
        st.markdown('<div class="warning-box"><strong>Warning:</strong> Please fetch or enter an article first.</div>', unsafe_allow_html=True)
    else:
        # Display summary before analysis if it exists
        if st.session_state.get("summary", "").strip():
            st.markdown('<div class="info-box"><strong>Article Summary:</strong><br>' + st.session_state.summary + '</div>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing article with multiple models..."):
            # Predictions
            bert_bias, bert_conf = predict_with_bert(article_text)
            svm_bias, svm_subtype = predict_with_svm(article_text, bert_bias)
            # rf_bias, rf_subtype = predict_with_rf(article_text)

        # Display results with professional styling
        st.markdown("### Analysis Results")
        st.markdown("---")
        
        # BERT Results
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
                <div class="result-card" style="border-left-color: #3b82f6;">
                    <div class="result-title">BERT Classification</div>
                    <div class="result-content">
                        <strong>Bias:</strong> {bert_bias}<br>
                        <strong>Confidence:</strong> {bert_conf:.2%}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence_color = "#10b981" if bert_conf > 0.7 else "#f59e0b" if bert_conf > 0.5 else "#ef4444"
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: {confidence_color};">{bert_conf:.0%}</div>
                    <div style="color: #64748b; font-size: 0.9rem;">Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        
        # SVM Results
        st.markdown(f"""
            <div class="result-card" style="border-left-color: #8b5cf6;">
                <div class="result-title">SVM Classification</div>
                <div class="result-content">
                    <strong>Bias:</strong> {svm_bias}<br>
                    <strong>Subtype:</strong> {svm_subtype}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Random Forest Results (commented out)
        # st.markdown(f"""
        #     <div class="result-card" style="border-left-color: #ec4899;">
        #         <div class="result-title">Random Forest Classification</div>
        #         <div class="result-content">
        #             <strong>Bias:</strong> {rf_bias}<br>
        #             <strong>Subtype:</strong> {rf_subtype}
        #         </div>
        #     </div>
        # """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Analysis completed using fine-tuned machine learning classifiers")