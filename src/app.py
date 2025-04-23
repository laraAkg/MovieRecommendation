from flask import Flask, render_template, request, jsonify
import pickle
from functools import lru_cache

from recommendation import recommend_movies

app = Flask(__name__)

# --- Modell laden ---
with open('created_model/light_model.pkl', 'rb') as f:
    df, tfidf_vectorizer, indices = pickle.load(f)

# TF-IDF-Matrix einmalig berechnen
tfidf_matrix = tfidf_vectorizer.transform(df['combined_features'])

# --- Caching der Empfehlungen ---
CACHE_SIZE = 128

@lru_cache(maxsize=CACHE_SIZE)
def cached_recommend(title: str):
    """
    Wrapper um recommend_movies, cached Ergebnisse für bis zu CACHE_SIZE verschiedene Titel.
    Raises ValueError, wenn Titel nicht gefunden wird.
    """
    return recommend_movies(
        title=title,
        df=df,
        tfidf_matrix=tfidf_matrix,
        indices=indices
    )

# --- Routen ---
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error_msg = None

    if request.method == 'POST':
        movie_title = request.form.get('movie_title', '').strip()
        try:
            recommendations = cached_recommend(movie_title)
        except ValueError as e:
            error_msg = str(e)

    return render_template(
        "index.html",
        recommendations=recommendations,
        error_msg=error_msg
    )

@app.route('/titles')
def titles():
    return jsonify(df['title'].dropna().tolist())

if __name__ == '__main__':
    app.run(debug=True)
