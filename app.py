"""
Flask Web Application for Sentiment-Based Product Recommendation System


Routes:
    - / : home page with username input
    - /recommend : generate and show recommendations
    - /api/recommend/<username> : API endpoint (JSON)
    - /health : health check
"""

from flask import Flask, render_template, request, jsonify
import os
import model  # Import module to access train_matrix after loading

# Import ML functions from model.py
from model import (
    load_models,
    get_top_20_recommendations,
    apply_sentiment_filtering,
    get_model_info
)

app = Flask(__name__)

# Load models when app starts
print("="*80)
print("Starting Flask application...")
print("="*80)
load_models()

@app.route('/')
def index():
    """Home page with user input"""
    # Get list of users for dropdown (sample of top 100 active users)
    user_ratings_count = model.train_matrix.sum(axis=1).sort_values(ascending=False).head(100)
    users = user_ratings_count.index.tolist()

    return render_template('index.html', users=users)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate recommendations for selected user"""
    # Check manual input first, then dropdown
    username = request.form.get('username_input')
    if not username:
        username = request.form.get('username_select')

    if not username:
        return render_template('error.html', error="Please either select a user from dropdown or type username")

    # Generate top 20 recommendations
    top_20, error = get_top_20_recommendations(username)

    if error:
        return render_template('error.html', error=error)

    # Apply sentiment filtering to get top 5
    top_5 = apply_sentiment_filtering(top_20)

    # Convert to list of dicts for template
    top_5_list = top_5.to_dict('records')

    return render_template('results.html',
                         username=username,
                         recommendations=top_5_list,
                         total_generated=len(top_20))

@app.route('/api/recommend/<username>')
def api_recommend(username):
    """API endpoint for recommendations (JSON response)"""
    # Generate top 20 recommendations
    top_20, error = get_top_20_recommendations(username)

    if error:
        return jsonify({'error': error}), 404

    # Apply sentiment filtering
    top_5 = apply_sentiment_filtering(top_20)

    # Convert to JSON
    result = {
        'username': username,
        'recommendations': top_5.to_dict('records'),
        'total_generated': len(top_20)
    }

    return jsonify(result)

@app.route('/health')
def health():
    """Health check endpoint"""
    model_info = get_model_info()
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_info['models_loaded']
    })

if __name__ == '__main__':

    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
