# 🧠 ReviewPredict — Flipkart Sentiment Analyzer

> An AI-powered web application that classifies Flipkart product reviews as **Positive**, **Neutral**, or **Negative** using an advanced NLP pipeline with negation handling, emoji recognition, lemmatization, and sentence-level voting.

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-black?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

</div>

---

## 👨‍💻 Developer

| Field | Details |
|---|---|
| **Name** |     Shubham singh |
| **Branch** | B.Tech — Computer Science Engineering |
| **Year** | 2nd Year |
| **Course** | Data Science |
| **Email** | narukasingh79@gmail.com |
| **Phone** | +91 9784965562 |

---

## 📌 Project Overview

ReviewPredict is a full-stack Data Science course project that goes beyond basic sentiment classification. It implements a **multi-step NLP pipeline** trained on **300,000+ real Flipkart reviews** and deploys the model as a complete Flask web application.

The core prediction logic uses **sentence-level majority voting** — each sentence of a review is analyzed independently, and the final sentiment is determined by the majority using Python's `Counter`. This makes it especially accurate for mixed reviews like *"Delivery was fast but the product quality is terrible."*

---

## ✨ Key Features

- 🔄 **Negation Handling** — `"not good"` becomes `"not_good"` so the model correctly understands reversed sentiment
- 😍 **Emoji Recognition** — Emojis like 😍→love, 😡→angry are mapped to words before cleaning
- 📝 **Sentence-Level Voting** — Each sentence is predicted separately; final result = `Counter().most_common(1)`
- 🌱 **WordNet Lemmatization** — Words reduced to base forms (running→run, better→good)
- 📊 **Confidence Scoring** — Every prediction shows a % confidence from `predict_proba()`
- 🌐 **Full Web App** — Login, Dashboard, My Reviews (with filters), About page
- 📱 **Fully Responsive** — Works on mobile, tablet, and desktop
- ⚡ **Loading Animations** — Page loaders, scroll reveals, smooth transitions throughout

---

## 🗂️ Project Structure

```
ReviewPredict/
│
├── app.py                    ← Flask backend (all routes + NLP pipeline)
├── tfidf.pkl                 ← Saved TF-IDF vectorizer (from notebook)
├── model.pkl                 ← Saved trained model (from notebook)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
└── templates/
    ├── login.html            ← Secure login page
    ├── dashboard.html        ← Main analyzer + chart + recent reviews
    ├── myreviews.html        ← All saved reviews with search & filter
    └── about.html            ← Project info & developer profile
```

---

## ⚙️ NLP Pipeline

Every review passes through this exact 6-step pipeline (same as the training notebook):

```
Raw Review Text
      │
      ▼
1. replace_emojis()     → 😍 → "love", 😡 → "angry"
      │
      ▼
2. clean_text()         → remove non-alpha, lowercase, remove stopwords
      │
      ▼
3. handle_negation()    → "not good" → "not_good"
      │
      ▼
4. lemmatize_text()     → "running" → "run", "better" → "good"
      │
      ▼
5. TF-IDF Transform     → vectorize with bigrams (max_features=5000)
      │
      ▼
6. Sentence Vote        → Counter(predictions).most_common(1)[0][0]
      │
      ▼
Final: Positive / Neutral / Negative  +  Confidence %
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Flipkart Product Reviews |
| **File** | `Dataset.csv` (encoding: `latin1`) |
| **Total Rows** | ~3,09,296 reviews |
| **Columns Used** | `Rate`, `Review` |
| **Labelling** | Rate ≥ 4 → Positive, Rate = 3 → Neutral, Rate ≤ 2 → Negative |

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| `MultinomialNB(alpha=0.5)` | Naive Bayes — fast baseline |
| `LogisticRegression(max_iter=1000)` | Better accuracy on complex patterns |

The best performing model and its TF-IDF vectorizer are saved as `.pkl` files using `pickle`.

---

## 🚀 How to Run

### Step 1 — Clone / Download the project

Place all files in one folder following the structure shown above.

### Step 2 — Install dependencies

```bash
pip install flask nltk scikit-learn pandas numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 3 — Generate model files (run your Jupyter Notebook)

Open and run your training notebook. Make sure the last cells save:

```python
import pickle

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

Place `tfidf.pkl` and `model.pkl` in the **same folder as `app.py`**.

### Step 4 — Run the Flask app

```bash
python app.py
```

### Step 5 — Open in browser

```
http://localhost:5000
```

---

## 🔗 Application Pages

| Route | Page | Description |
|---|---|---|
| `/login` | Login | Secure login with remember-me, loading spinner, field validation |
| `/dashboard` | Dashboard | Analyze reviews, sentence breakdown, doughnut chart, recent history |
| `/myreviews` | My Reviews | All saved reviews grouped by product, with search & sentiment filter |
| `/about` | About | Project info, NLP pipeline diagram, tech stack, developer profile |
| `/logout` | — | Clears session and redirects to login |

---

## 🐛 Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: flask` | Flask not installed | `pip install flask` |
| `ModuleNotFoundError: nltk` | NLTK not installed | `pip install nltk` |
| `FileNotFoundError: tfidf.pkl` | Model not generated yet | Run your Jupyter notebook first |
| `FileNotFoundError: model.pkl` | Model not generated yet | Run your Jupyter notebook first |
| `TemplateNotFound: dashboard.html` | Wrong folder structure | All `.html` files must be inside a `templates/` folder |
| `Address already in use` | Port 5000 is busy | Change to `app.run(port=5001)` in `app.py` |
| `LookupError: punkt` | NLTK data missing | `app.py` auto-downloads it; run once with internet |

---

## 🛠️ Tech Stack

**Backend**
- Python 3.8+
- Flask 2.3+
- scikit-learn (TF-IDF, Naive Bayes, Logistic Regression)
- NLTK (stopwords, punkt tokenizer, WordNet lemmatizer)
- Pickle (model serialization)

**Frontend**
- HTML5 + CSS3 (no framework)
- Vanilla JavaScript
- Chart.js (doughnut chart)
- Google Fonts (Playfair Display + Nunito)

**Data**
- Pandas, NumPy
- Matplotlib (training visualizations)

---

## 📝 License

This project was built as a **Data Science course assignment** by Anmol Panchal.  
Feel free to use it as a reference for your own NLP projects.

---


