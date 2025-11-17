# Political Bias Analyzer

An advanced Natural Language Processing application that detects and classifies political bias in news articles using state-of-the-art machine learning models including BERT, SVM, and Random Forest classifiers.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [API Keys Setup](#api-keys-setup)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## Overview

The Political Bias Analyzer helps users understand the political orientation of news articles by automatically classifying them into:

- **Main Bias Categories**: Left, Center, Right
- **Political Subtypes**: Socialist, Liberal, Secular, Center, Capitalist, Conservative, Nationalist

This tool addresses the growing concern of media bias and helps users make informed decisions by providing objective, data-driven insights into news content.

## Features

### Multi-Model Analysis
- **BERT Classification**: Deep semantic understanding with confidence scores
- **SVM Classification**: Fast traditional ML predictions with bias and subtype detection
- **Random Forest**: Ensemble-based robust classification

### Flexible Input Methods
- **URL Analysis**: Paste any news article URL for automatic scraping and analysis
- **Text Paste**: Directly paste article content for instant analysis
- **News Search**: Search for articles by topic using integrated search APIs

### Professional User Interface
- Clean, modern design with blue gradient theme
- Responsive layout for all devices
- Real-time feedback and loading indicators
- Color-coded confidence levels
- Persistent article summaries

### Advanced Features
- Automated web scraping with Firecrawl API
- Smart content cleaning (removes ads, navigation, boilerplate)
- Comprehensive error handling with retry logic
- Environment variable-based configuration
- Auto-reload functionality for development

## Demo

### Interface Preview
The application features three main sections:

1. **Input Selection**: Choose between URL paste, manual text entry, or news search
2. **Article Processing**: Automatic fetching and cleaning of content
3. **Results Display**: Professional cards showing bias classifications with confidence scores

### Example Analysis

```
Article: "Government proposes new healthcare reform bill"

BERT Classification:
├─ Bias: Left
└─ Confidence: 85%

SVM Classification:
├─ Bias: Left
└─ Subtype: Liberal
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/MuhammadAli7896/NLP-Project.git
cd NLP-Project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Step 5: Setup Environment Variables

Create a `.env` file in the project root (see [API Keys Setup](#api-keys-setup)):

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Step 6: Verify Model Files

Ensure the following model files are present:
- `bert_weights/bert_type_classifier.pt`
- `models/svm_bias_vectorizer.joblib`
- `models/svm_model_bias.pkl`
- `models/svm_vectorizer_subtype.pkl`
- `models/svm_model_subtype.pkl`
- `models/seperate_vectorizer1.joblib`
- `models/seperate_bias_model.pkl`
- `models/seperate_vectorizer2.joblib`
- `models/seperate_subtype_model.pkl`

## Usage

### Running the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

#### Method 1: Analyze by URL
1. Select "Paste Article Link"
2. Paste the news article URL
3. Click "Fetch Article"
4. Review the article summary
5. Click "Analyze Article"
6. View the bias analysis results

#### Method 2: Manual Text Entry
1. Select "Manual Text Entry"
2. Paste the article text directly
3. Click "Analyze Article"
4. View the bias analysis results

#### Method 3: Search News
1. Select "Search News Topic"
2. Enter a political topic (e.g., "healthcare", "immigration")
3. Click "Search News"
4. Browse the search results
5. Click on an article link
6. Copy the URL and use Method 1 to analyze

## Models

### BERT Classifier

- **Architecture**: bert-base-uncased with custom classification layer
- **Parameters**: 110M parameters
- **Output**: 3 classes (Left, Center, Right) with confidence scores
- **Features**: Bidirectional context understanding, attention mechanisms

### Support Vector Machine (SVM)

- **Type**: Dual-stage pipeline
- **Stage 1**: Bias classification (3 classes)
- **Stage 2**: Subtype classification (7 classes)
- **Features**: Fast inference, hierarchical feature engineering

### Random Forest

- **Type**: Ensemble classifier
- **Features**: Resistant to overfitting, feature importance analysis
- **Output**: Bias and subtype predictions

## Dataset

- **Total Records**: 26,810 labeled news articles
- **Source**: `dataset_making/final_validated.csv`
- **Features**: Text content, bias rating, political subtype
- **Preprocessing**: Text cleaning, lemmatization, stopword removal

### Label Distribution

**Bias Categories:**
- Left: Articles with left-leaning perspective
- Center: Politically neutral articles
- Right: Articles with right-leaning perspective

**Political Subtypes:**
- Socialist, Liberal, Secular (Left-leaning)
- Center (Neutral)
- Capitalist, Conservative, Nationalist (Right-leaning)

## Project Structure

```
NLP-Project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git exclusion rules
├── README.md                       # This file
│
├── bert_weights/
│   └── bert_type_classifier.pt     # Fine-tuned BERT model
│
├── models/
│   ├── svm_bias_vectorizer.joblib
│   ├── svm_model_bias.pkl
│   ├── svm_vectorizer_subtype.pkl
│   ├── svm_model_subtype.pkl
│   ├── seperate_vectorizer1.joblib
│   ├── seperate_bias_model.pkl
│   ├── seperate_vectorizer2.joblib
│   └── seperate_subtype_model.pkl
│
├── dataset_making/
│   ├── final_validated.csv         # Training dataset
│   └── [preprocessing scripts]
│
├── notebook-and-scripts/
│   ├── bert_model.py               # BERT training script
│   ├── svm_rf_model.py             # SVM/RF training
│   └── [Jupyter notebooks]
│
└── models_reports/
    └── [Model performance reports]
```

## API Keys Setup

The application requires API keys for full functionality. Create a `.env` file in the project root:

```env
# Required for article scraping
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Optional for Brave Search (DuckDuckGo is fallback)
BRAVE_API_KEY=your_brave_api_key_here
```

### Getting API Keys

1. **Firecrawl API** (Required):
   - Visit [firecrawl.dev](https://firecrawl.dev)
   - Sign up for an account
   - Get your API key from the dashboard

2. **Brave Search API** (Optional):
   - Visit [brave.com/search/api](https://brave.com/search/api)
   - Sign up for API access
   - Get your API key

**Note**: The application uses DuckDuckGo search as a fallback if Brave API is not configured.

## Technologies Used

### Core Technologies
- **Python 3.11**: Primary programming language
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework for BERT
- **Transformers (Hugging Face)**: Pre-trained models and tokenizers
- **scikit-learn**: Traditional ML models (SVM, Random Forest)

### NLP Libraries
- **NLTK**: Text preprocessing and tokenization
- **NumPy**: Numerical computations
- **SciPy**: Sparse matrix operations

### Web Scraping & APIs
- **Firecrawl API**: Article content extraction
- **DuckDuckGo Search**: News article discovery
- **Requests**: HTTP client for API calls

### Development Tools
- **python-dotenv**: Environment variable management
- **joblib**: Model serialization

## Contact

**Muhammad Ali**

- GitHub: [@MuhammadAli7896](https://github.com/MuhammadAli7896)
- Email: muhammadali30804@gmail.com
- Project Link: [https://github.com/MuhammadAli7896/NLP-Project](https://github.com/MuhammadAli7896/NLP-Project)

**Azlan (Team Member)**

- GitHub: [@Azlan36](https://github.com/Azlan36)
- Email: asim4500189@cloud.neduet.edu.pk

## Acknowledgments

- BERT model from [Hugging Face Transformers](https://huggingface.co/transformers/)
- Dataset sources and contributors
- Open-source community for various libraries and tools

## Future Enhancements

- [ ] Batch processing for multiple articles
- [ ] Export functionality (PDF, CSV reports)
- [ ] Visualization dashboard with charts
- [ ] Browser extension for on-the-fly analysis
- [ ] API endpoint for third-party integration

---

<div align="center">

**Made for promoting media literacy and informed news consumption**

Star this repository if you find it helpful!

</div>
