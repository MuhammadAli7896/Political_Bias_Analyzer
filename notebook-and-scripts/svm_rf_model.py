import joblib
import numpy as np
from scipy.sparse import hstack
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


svm_bias_vectorizer = joblib.load("svm_bias_vectorizer.joblib")
svm_model_bias = joblib.load("svm_model_bias.pkl")
svm_vectorizer_subtype = joblib.load("svm_vectorizer_subtype.pkl")
svm_model_subtype = joblib.load("svm_model_subtype.pkl")

rf_vectorizer_bias = joblib.load("seperate_vectorizer1.joblib")
rf_model_bias = joblib.load("seperate_bias_model.pkl")
rf_vectorizer_subtype = joblib.load("seperate_vectorizer2.joblib")
rf_model_subtype = joblib.load("seperate_subtype_model.pkl")


def clean_text(text):
    lemma = WordNetLemmatizer()
    swords = stopwords.words("english")
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^a-zA-Z0-9 ]", " ", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemma.lemmatize(word) for word in tokens if word not in swords]
    return " ".join(tokens)

def predict_text(text):
    clean_input = (text)

    svm_bias_features = svm_bias_vectorizer.transform([clean_input]).toarray()
    svm_bias_pred = svm_model_bias.predict(svm_bias_features)[0]

    svm_subtype_features = svm_vectorizer_subtype.transform([clean_input]).toarray()
    svm_combined = np.hstack((svm_subtype_features, np.array([[svm_bias_pred]])))  
    svm_subtype_pred = svm_model_subtype.predict(svm_combined)[0]

    rf_bias_features = rf_vectorizer_bias.transform([clean_input])
    rf_bias_pred = rf_model_bias.predict(rf_bias_features)[0]

    rf_subtype_features = rf_vectorizer_subtype.transform([clean_input])
    rf_combined = hstack((rf_subtype_features, np.array([[rf_bias_pred]])))  
    rf_subtype_pred = rf_model_subtype.predict(rf_combined)[0]


    print("\n--- Prediction Results ---")
    print("SVM:")
    print(f"  → Bias Type: {svm_bias_pred}")
    print(f"  → Bias Subtype: {svm_subtype_pred}")
    print("\nRandom Forest:")
    print(f"  → Bias Type: {rf_bias_pred}")
    print(f"  → Bias Subtype: {rf_subtype_pred}")

    return {
        "svm_bias": svm_bias_pred,
        "svm_subtype": svm_subtype_pred,
        "rf_bias": rf_bias_pred,
        "rf_subtype": rf_subtype_pred
    }


if __name__ == "__main__":
    user_text = input("Enter a news article text:\n")
    results = predict_text(user_text)
