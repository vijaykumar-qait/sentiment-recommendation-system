# Sentiment-Based Product Recommendation System

### IIITB AI-ML Capstone Project

A hybrid recommendation system that doesn't just suggest popular items; it also suggests items that have *good reviews*. What good is it to suggest something that sells well but everyone hates?

**Live Demo:** https://sentiment-recommendation-system.onrender.com/

> **Note:** This app runs on Render's free tier. If it hasn't been used recently, it may take 30-60 seconds to wake up. Please be patient.

---

## What This Project Does

Traditional recommendation systems look at what similar users bought. The problem? A product might be trending but have terrible reviews. This system fixes that:

1.  **Generate Candidates:** We use **Item-Based Collaborative Filtering** to find 20 products similar to your interests.
2.  **Filter by Sentiment:** We analyze ALL reviews for those 20 products using a **Random Forest** classifier and filter out the ones with negative sentiment.
3.  **Final Recommendation:** The system returns the top 5 products that are both relevant to you and highly rated by others.

Result: Products that are both relevant to you AND have genuinely good reviews

---

## The Tech Stack

* **Python 3.11.9** - Selected for Render compatibility.
* **Random Forest** - For sentiment classification (F1-score: 0.9558).
* **TF-IDF** - For text vectorization (using unigrams + bigrams).
* **Item-Based CF** - Chosen for better coverage on sparse data.
* **Flask** - A lightweight web framework for the API.
* **Render** - Cloud deployment platform.

---

## Challenges I Ran Into

### 1. Class Imbalance (88.8% Positive vs 11.2% Negative)
The dataset was heavily skewed towards positive reviews. My first model just predicted "Positive" for everything and achieved high accuracy, but it was useless for catching bad products.

* **The Fix:** I used `class_weight='balanced'` in the sklearn models. This forced the model to penalize errors on the minority class more heavily.
* **Result:** The F1-score jumped from ~0.70 to 0.9558.

### 2. Negation Words Matter
Initially, I removed all stopwords. This turned phrases like "not good" into "good," which completely flipped the sentiment.

* **The Fix:** I customized the stopword removal process to specifically keep negation words (like 'not', 'no', 'never', "don't").

### 3. User-Based CF Failed
I started with User-Based Collaborative Filtering. It failed because the coverage was only 12.75%—users simply didn't have enough overlapping ratings.

* **The Fix:** Switched to Item-Based CF. Since products have more overlapping ratings than users do, coverage improved to 91.18%.

---

## Model Performance

### Sentiment Models Compared

| Model | F1-Score | Accuracy | ROC-AUC | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9558** | **91.85%** | **91.39%** | **Selected Model** |
| XGBoost | 0.9362 | 90.63% | 90.28% | Good, but more complex |
| Naive Bayes | 0.9397 | 90.71% | 89.62% | Fast but lower accuracy |
| Logistic Regression | 0.8889 | 88.89% | 89.81% | Baseline |

### Recommendation Systems Compared

| Method | RMSE | MAE | Coverage | Notes                         |
| :--- | :--- | :--- | :--- |:------------------------------|
| User-Based CF | 1.46 | 0.85 | 12.75% | Poor coverage due to sparsity |
| **Item-Based CF** | **1.11** | **0.71** | **91.18%** | **Used in deployment**        |

---

## Project Structure

```text
├── SentimentBasedProductRecommendationSystem.ipynb  # Full ML pipeline
├── model.py                                         # Prediction logic
├── app.py                                           # Flask web app
├── templates/                                       # Single-page UI (all states
│   ├── index.html                                   # Main UI
├── models/                                          # Trained models
│   ├── best_sentiment_model.pkl                     # Random Forest
│   ├── tfidf_vectorizer.pkl                         # Text vectorizer
│   ├── product_similarity.pkl                       # CF Matrix
│   └── train_matrix.csv                             # User ratings
├── data/
│   └── processed_text_data.csv                      # Cleaned reviews
├── requirements.txt
├── runtime.txt                                      # Python version for Render
└── Procfile                                         # Gunicorn config

```
---
## How to Run Locally
Follow these steps to get the project running on your machine:
```bash
# 1. Clone the repository
git clone [https://github.com/vijaykumar-qait/sentiment-recommendation-system](https://github.com/vijaykumar-qait/sentiment-recommendation-system)
cd "Capstone -SentimentBasedProductRecommendationSystem"

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Generate models (if .pkl files are missing)
# Run the Jupyter Notebook: SentimentBasedProductRecommendationSystem.ipynb

# 5. Run the application
python app.py


### Open your browser and navigate to http://localhost:5000.
```

---
## Key Learnings
- Accuracy is not what it seems: A model can be 90% accurate on data that isn't balanced and still not be useful. Always check the F1-Score or AUC.
- Context is very important: Bigrams, which are pairs of two words, are very important for getting phrases like "not happy" in TF-IDF.
- Simplicity wins: Deep Learning is interesting, but a well-tuned Random Forest trained faster, was easier to set up, and worked great with this dataset.

##  Future Improvements
- For a better understanding of sentiment, add deep learning models like LSTM or BERT.
- For new users who don't have any history, set up a "cold start" solution.
- Add a loop for real-time feedback so that the model can be retrained when new reviews come in.