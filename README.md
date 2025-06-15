# 🎬 YouTube Comment Sentiment Analyzer

A complete pipeline for analyzing the **sentiment of YouTube video comments** using a **Multinomial Naive Bayes** classifier trained on custom-labeled data. The system fetches up to 500 top-level English comments, classifies them as **positive** or **negative**, and highlights both the sentiment ratio and the most extreme comments.  

---

## 🔍 Features

- ✅ Fetches comments from any public YouTube video (up to 500)
- 🧹 Cleans and filters comments (language, length, symbols/emojis)
- 🧠 Predicts comment sentiment using a trained **Naive Bayes** model
- 📈 Shows sentiment breakdown, positive ratio, and confidence scores
- 💬 Highlights the most positive and most negative comments
- 🌐 Frontend for user interaction

---


---
### 🔍 Model Details

- **Training Data:** Trained on DAIR-AI Text Emotion Classification Dataset containing ~42k textual comments.
- **Preprocessing:**
  - Lowercasing, punctuation & emoji removal
  - Tokenization, stop word removal, lemmatization
- **Algorithm:** Multinomial Naive Bayes (MNB)
- **Vectorizer:** `TfidfVectorizer` with up to **4-grams**

---
## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/aliraza-zaidi/yt-comments-sentiment-analysis.git
cd yt-comments-sentiment-analysis

```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Set up YouTube Data API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project and enable the **YouTube Data API v3**.
3. Generate an **API key**.
4. In your project/res folder, open `fetch_comments.py` and replace the placeholder with your API key:

```python
API_KEY = "YOUR_API_KEY"
```
---

### 4. Run the App

To launch the web-based frontend for analyzing YouTube video sentiment:

```bash
python app.py
```
---

