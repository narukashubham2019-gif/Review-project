"""
ReviewPredict - Flipkart Sentiment Analyzer
Author : Shubham Singh | B.Tech CSE 2nd Year | Data Science
Email  : narukasingh79@gmail.com
Phone  : +91 9784965562

HOW TO RUN:
  1. pip install flask nltk scikit-learn
  2. Keep tfidf.pkl and model.pkl in same folder as app.py
  3. python app.py
  4. Open http://localhost:5000
"""

from flask import Flask, render_template, request, redirect, url_for, session
import pickle, re, os
from datetime import date
from collections import Counter

# ── NLTK setup ─────────────────────────────────────────────────────────────
import nltk

def download_nltk():
    packages = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# ── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'reviewpredict_secret_key_2024'

# ── Load Saved Model & TF-IDF ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, 'tfidf.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    MODEL_LOADED = True
except Exception as e:
    print(f"[WARNING] Could not load model: {e}")
    print("[WARNING] Running in DEMO mode. Train your model first.")
    tfidf = None
    model = None
    MODEL_LOADED = False

# ── NLP Functions (same as your notebook) ──────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
NEGATIONS   = ["not", "no", "never", "n't"]
EMOJI_MAP   = {
    "😍": "love",   "😊": "happy",        "😡": "angry",
    "😢": "sad",    "👌": "great",         "👍": "good",
    "👎": "bad",    "🤩": "amazing",       "😤": "frustrated",
    "💔": "disappointed", "🥰": "love",   "😠": "angry",
}

def replace_emojis(text):
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)
    return text

def clean_text(text):
    text = replace_emojis(str(text))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

def handle_negation(text):
    words = text.split()
    result = []
    for i in range(len(words) - 1):
        if words[i] in NEGATIONS:
            result.append(words[i] + "_" + words[i + 1])
        else:
            result.append(words[i])
    if words:
        result.append(words[-1])
    return " ".join(result)

def lemmatize_text(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

def preprocess(text):
    """Full pipeline: emoji → clean → negation → lemmatize"""
    text = clean_text(text)
    text = handle_negation(text)
    text = lemmatize_text(text)
    return text

def predict_review(review_text):
    """
    predict_review() - same logic as your notebook.
    Sentence-level voting using Counter.
    """
    if not MODEL_LOADED:
        # Demo mode: return mock result
        return {
            'sentiment':      'Positive',
            'confidence':     91.5,
            'sentences':      [review_text],
            'sentence_preds': ['Positive'],
            'demo':           True,
        }

    sentences   = sent_tokenize(review_text)
    clean_sents = [preprocess(s) for s in sentences]
    vectors     = tfidf.transform(clean_sents)
    preds       = model.predict(vectors)

    # Majority vote (your notebook logic)
    final = Counter(preds).most_common(1)[0][0]

    # Normalize labels to Title case
    label_norm = lambda s: {
        'positive': 'Positive',
        'neutral':  'Neutral',
        'negative': 'Negative',
    }.get(str(s).lower(), str(s).capitalize())

    final_label   = label_norm(final)
    sent_labels   = [label_norm(p) for p in preds]

    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(vectors)
        classes    = [str(c).lower() for c in model.classes_]
        tgt        = str(final).lower()
        idx        = classes.index(tgt) if tgt in classes else 0
        confidence = round(float(proba[:, idx].mean()) * 100, 1)
    else:
        confidence = 87.0

    return {
        'sentiment':      final_label,
        'confidence':     confidence,
        'sentences':      sentences,
        'sentence_preds': sent_labels,
        'demo':           False,
    }

# ── In-memory review store ──────────────────────────────────────────────────
# { username: [ {id, product, category, text, sentiment, confidence, date, ...} ] }
reviews_store = {}

def get_user_reviews():
    return reviews_store.get(session.get('username', ''), [])

def get_stats():
    rv = get_user_reviews()
    return {
        'total':    len(rv),
        'positive': sum(1 for r in rv if r['sentiment'] == 'Positive'),
        'neutral':  sum(1 for r in rv if r['sentiment'] == 'Neutral'),
        'negative': sum(1 for r in rv if r['sentiment'] == 'Negative'),
    }

def is_logged_in():
    return 'username' in session

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    if is_logged_in():
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            error = 'Please enter both username and password.'
        else:
            session['username'] = username
            session.permanent = ('remember' in request.form)
            return redirect(url_for('dashboard'))

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if not is_logged_in():
        return redirect(url_for('login'))

    result = None
    error  = None

    if request.method == 'POST':
        review_text  = request.form.get('review', '').strip()
        product_name = (request.form.get('product_name', '') or 'Unknown Product').strip()
        category     = (request.form.get('category', '') or '').strip()

        if not review_text:
            error = 'Please enter a review text before clicking Predict.'
        else:
            result = predict_review(review_text)

            u = session['username']
            if u not in reviews_store:
                reviews_store[u] = []

            reviews_store[u].append({
                'id':          len(reviews_store[u]),
                'product':     product_name,
                'category':    category,
                'text':        review_text,
                'sentiment':   result['sentiment'],
                'confidence':  result['confidence'],
                'sentences':   result['sentences'],
                'sent_preds':  result['sentence_preds'],
                'date':        date.today().strftime('%d %b %Y'),
            })

    recent = list(reversed(get_user_reviews()[-5:]))

    return render_template('dashboard.html',
        result   = result,
        error    = error,
        stats    = get_stats(),
        recent   = recent,
        username = session['username'],
        model_loaded = MODEL_LOADED,
    )


@app.route('/myreviews')
def myreviews():
    if not is_logged_in():
        return redirect(url_for('login'))

    grouped = {}
    for idx, r in enumerate(get_user_reviews()):
        entry = dict(r)
        entry['_index'] = idx
        grouped.setdefault(entry['product'], []).append(entry)

    return render_template('myreviews.html',
        grouped  = grouped,
        stats    = get_stats(),
        username = session['username'],
    )


@app.route('/delete_review/<int:index>', methods=['POST'])
def delete_review(index):
    if not is_logged_in():
        return redirect(url_for('login'))
    u  = session['username']
    rv = reviews_store.get(u, [])
    if 0 <= index < len(rv):
        rv.pop(index)
    return redirect(url_for('myreviews'))


@app.route('/clear_reviews', methods=['POST'])
def clear_reviews():
    if not is_logged_in():
        return redirect(url_for('login'))
    reviews_store[session['username']] = []
    return redirect(url_for('myreviews'))


@app.route('/about')
def about():
    return render_template('about.html',
        username     = session.get('username'),
        model_loaded = MODEL_LOADED,
    )


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)