"""
model.py - ML models for product recommendation system

Contains:
- Random Forest sentiment model
- Item-Based Collaborative Filtering
- Prediction functions
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import json

# Global variables for models (loaded once)
sentiment_model = None
tfidf_vectorizer = None
train_matrix = None
product_similarity_df = None
df_full = None


def load_models():
    """
     Load all models and data at startup.
     Takes 5-10 seconds but only runs once.
     Returns True if successful.
    """
    global sentiment_model, tfidf_vectorizer, train_matrix, product_similarity_df, df_full

    try:
        print("Loading models and data...")

        # Load sentiment model (Random Forest - Best from Task 4)
        sentiment_model = joblib.load('models/best_sentiment_model.pkl')
        print("Sentiment model loaded (Random Forest, F1=0.9558)")

        # Load TF-IDF vectorizer
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        print("TF-IDF vectorizer loaded (5000 features)")

        # Load recommendation system data
        train_matrix = pd.read_csv('models/train_matrix.csv', index_col=0)
        print(f"Training matrix loaded ({train_matrix.shape[0]} users × {train_matrix.shape[1]} products)")

        # Load product similarity matrix (Item-Based CF - Best from Task 5)
        product_similarity_df = joblib.load('models/product_similarity.pkl')
        print("Product similarity matrix loaded (Item-Based CF)")

        # Load full dataset for sentiment analysis
        df_full = pd.read_csv('data/processed_text_data.csv')
        print(f"Review data loaded ({len(df_full)} reviews)")

        # Load and display system info
        with open('models/recommendation_system_info.json', 'r') as f:
            rec_info = json.load(f)
        print(f"✓ Using: {rec_info['selected_system']}")

        print("\n" + "="*80)
        print("All models loaded successfully!")
        print("="*80)
        return True

    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def predict_item_based(user, product, k=10):
    """
    Predict rating using Item-Based Collaborative Filtering

    Algorithm:
    1. Get products the user has rated
    2. Calculate similarity between target product and rated products
    3. Take top K most similar products
    4. Return weighted average of user's ratings for similar products

    """
    # Check if user exists
    if user not in train_matrix.index:
        return 0

    # Get products this user has rated
    user_ratings = train_matrix.loc[user]
    rated_products = user_ratings[user_ratings > 0].index

    if len(rated_products) == 0:
        return 0

    # Get similarities and ratings for rated products
    similarities = []
    ratings = []

    for rated_product in rated_products:
        if rated_product in product_similarity_df.index and product in product_similarity_df.columns:
            sim = product_similarity_df.loc[rated_product, product]
            if sim > 0:
                similarities.append(sim)
                ratings.append(train_matrix.loc[user, rated_product])

    if not ratings:
        return 0

    # Take top K most similar products
    if len(ratings) > k:
        top_k_idx = np.argsort(similarities)[-k:]
        similarities = [similarities[i] for i in top_k_idx]
        ratings = [ratings[i] for i in top_k_idx]

    # Weighted average: higher weight to more similar products
    predicted_rating = np.average(ratings, weights=similarities)

    return predicted_rating


def get_top_20_recommendations(username):
    """
    Generate top 20 product recommendations for a user

    Process:
    1. Find products user hasn't rated yet
    2. Predict ratings for all unrated products using Item-Based CF
    3. Sort by predicted rating
    4. Return top 20

    """
    # Validate user exists
    if username not in train_matrix.index:
        return None, f"User '{username}' not found in database"

    # Get products user hasn't rated yet
    user_rated_products = train_matrix.loc[username][train_matrix.loc[username] > 0].index
    all_products = train_matrix.columns
    unrated_products = [p for p in all_products if p not in user_rated_products]

    # Generate predictions for all unrated products
    predictions = []
    for product in unrated_products:
        pred_rating = predict_item_based(username, product)
        if pred_rating > 0:  # Only include if we can predict
            predictions.append({
                'product': product,
                'predicted_rating': pred_rating
            })

    if len(predictions) == 0:
        return None, "Could not generate recommendations for this user"

    # Create DataFrame and sort by predicted rating
    predictions_df = pd.DataFrame(predictions)
    top_20 = predictions_df.sort_values('predicted_rating', ascending=False).head(20)

    return top_20, None


def apply_sentiment_filtering(top_20):
    """
    Apply sentiment analysis to filter top 20 to top 5 products

    Process:
    1. For each product in top 20:
       - Get all reviews for that product
       - Vectorize reviews using TF-IDF
       - Predict sentiment using Random Forest model
       - Calculate positive sentiment percentage
    2. Sort by positive sentiment percentage
    3. Return top 5

    This integrates both systems:
    - Recommendation System: Generates candidates based on ratings
    - Sentiment Analysis: Validates quality based on review sentiment
    """
    product_sentiments = []

    # Analyze sentiment for each product
    for idx, row in top_20.iterrows():
        product_name = row['product']

        # Get all reviews for this product
        product_reviews = df_full[df_full['name'] == product_name]

        if len(product_reviews) == 0:
            continue

        # Get processed review text
        review_texts = product_reviews['lemmatized_text'].values

        # Vectorize using same TF-IDF vectorizer as training
        review_vectors = tfidf_vectorizer.transform(review_texts)

        # Predict sentiment using Random Forest model
        sentiment_predictions = sentiment_model.predict(review_vectors)

        # Calculate positive sentiment percentage
        positive_count = (sentiment_predictions == 'Positive').sum()
        total_count = len(sentiment_predictions)
        positive_percentage = (positive_count / total_count) * 100

        product_sentiments.append({
            'product': product_name,
            'predicted_rating': row['predicted_rating'],
            'total_reviews': total_count,
            'positive_reviews': positive_count,
            'negative_reviews': total_count - positive_count,
            'positive_percentage': positive_percentage
        })

    # Create DataFrame
    sentiment_df = pd.DataFrame(product_sentiments)

    # Sort by positive sentiment percentage and get top 5
    top_5 = sentiment_df.sort_values('positive_percentage', ascending=False).head(5)

    return top_5


def get_model_info():
    """
    Get information about loaded models
    """
    info = {
        'sentiment_model': {
            'name': 'Random Forest',
            'test_f1': 0.9558,
            'test_accuracy': 0.9185,
            'test_roc_auc': 0.9139
        },
        'recommendation_system': {
            'name': 'Item-Based Collaborative Filtering',
            'rmse': 1.1134,
            'mae': 0.7070,
            'coverage': 91.18
        },
        'data': {
            'total_reviews': len(df_full) if df_full is not None else 0,
            'users': train_matrix.shape[0] if train_matrix is not None else 0,
            'products': train_matrix.shape[1] if train_matrix is not None else 0
        },
        'models_loaded': all([
            sentiment_model is not None,
            tfidf_vectorizer is not None,
            train_matrix is not None,
            product_similarity_df is not None,
            df_full is not None
        ])
    }

    return info


# For testing this module independently
if __name__ == '__main__':
    print("="*80)
    print("Testing model.py")
    print("="*80)

    # Load models
    success = load_models()

    if success:
        print("\n" + "="*80)
        print("Model Information:")
        print("="*80)
        info = get_model_info()
        print(f"Sentiment Model: {info['sentiment_model']['name']}")
        print(f"  - F1-Score: {info['sentiment_model']['test_f1']:.4f}")
        print(f"Recommendation System: {info['recommendation_system']['name']}")
        print(f"  - RMSE: {info['recommendation_system']['rmse']:.4f}")
        print(f"  - Coverage: {info['recommendation_system']['coverage']:.2f}%")
        print(f"\nData Statistics:")
        print(f"  - Users: {info['data']['users']:,}")
        print(f"  - Products: {info['data']['products']:,}")
        print(f"  - Reviews: {info['data']['total_reviews']:,}")

        # Test prediction
        print("\n" + "="*80)
        print("Testing prediction for user 'mike':")
        print("="*80)
        top_20, error = get_top_20_recommendations('mike')
        if not error:
            print(f"Generated {len(top_20)} recommendations")
            top_5 = apply_sentiment_filtering(top_20)
            print(f"Filtered to {len(top_5)} products with sentiment analysis")
            print("\nTop 5 Products:")
            for idx, row in top_5.iterrows():
                print(f"  {row['product'][:50]}...")
                print(f"    Rating: {row['predicted_rating']:.2f}, Positive: {row['positive_percentage']:.1f}%")
        else:
            print(f"Error: {error}")
    else:
        print("Failed to load models")
