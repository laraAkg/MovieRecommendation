from flask import Flask, render_template, request, jsonify
import pickle
import logging
import os
from recommendation import recommend_movies
from explanation import explain_recommendation

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
try:
    with open('light_model.pkl', 'rb') as f:
        df, tfidf, indices = pickle.load(f)
    logger.info("✅ Model successfully loaded.")
except Exception as e:
    logger.error(f"❌ Failed to load the model: {e}")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main page and handle movie recommendations."""
    recommendations = []
    error_message = None
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        if not movie_title:
            error_message = "Please enter a movie title."
        else:
            recommendations = recommend_movies(movie_title, df, tfidf, indices)
            if 'error' in recommendations[0]:
                error_message = recommendations[0]['error']
                recommendations = []
    return render_template("index.html", recommendations=recommendations, error_message=error_message)

@app.route('/titles')
def titles():
    """Return a list of all movie titles."""
    title_list = df['title'].dropna().tolist()
    return jsonify(title_list)

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode)