from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

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

def explain_recommendation(base_movie, recommended_movie):
    explanation = {}

    # Compare genres
    base_genres = set(str(base_movie.get('genres', '')).split())
    recommended_genres = set(str(recommended_movie.get('genres', '')).split())
    genre_overlap = base_genres.intersection(recommended_genres)
    if genre_overlap:
        explanation['genres'] = {
            'shared': list(genre_overlap),
            'count': len(genre_overlap),
            'details': f"Gemeinsame Genres: {', '.join(genre_overlap)}"
        }

    # Compare cast
    base_cast = set(str(base_movie.get('cast', '')).split(', '))
    recommended_cast = set(str(recommended_movie.get('cast', '')).split(', '))
    cast_overlap = base_cast.intersection(recommended_cast)
    if cast_overlap:
        explanation['cast'] = {
            'shared': list(cast_overlap),
            'count': len(cast_overlap),
            'details': f"Gemeinsame Schauspieler: {', '.join(cast_overlap)}"
        }

    # Compare director
    if base_movie.get('director') == recommended_movie.get('director'):
        explanation['director'] = {
            'shared': base_movie.get('director'),
            'details': f"Gleicher Regisseur: {base_movie.get('director')}"
        }

    # Compare keywords
    base_keywords = set(str(base_movie.get('keywords', '')).split())
    recommended_keywords = set(str(recommended_movie.get('keywords', '')).split())
    keyword_overlap = base_keywords.intersection(recommended_keywords)
    if keyword_overlap:
        explanation['keywords'] = {
            'shared': list(keyword_overlap),
            'count': len(keyword_overlap),
            'details': f"Gemeinsame Stichworte: {', '.join(keyword_overlap)}"
        }

    # Compare tagline
    base_tagline = str(base_movie.get('tagline', '')).strip().lower()
    recommended_tagline = str(recommended_movie.get('tagline', '')).strip().lower()
    if base_tagline and recommended_tagline and base_tagline == recommended_tagline:
        explanation['tagline'] = {
            'shared': base_tagline,
            'details': f"Gleicher Slogan: {base_tagline}"
        }

    # Add similarity score
    explanation['similarity_score'] = {
        'score': recommended_movie.get('sim_score', 0),
        'details': f"Ähnlichkeitsscore: {recommended_movie.get('sim_score', 0):.2f}"
    }

    return explanation

def recommend_movies(title, top_n=9):
    title = title.lower().strip()
    if title not in indices:
        logger.warning(f"Title '{title}' not found in the dataset.")
        return [{'error': f"Der Titel '{title}' wurde nicht gefunden. Bitte versuche es mit einem anderen."}]

    # Get the index of the input title
    idx = indices[title]

    # Compute similarity scores
    input_vector = tfidf.transform([df.loc[idx, 'combined_features']])
    sim_scores = cosine_similarity(input_vector, tfidf.transform(df['combined_features']))[0]
    df['sim_score'] = sim_scores

    # Sort by similarity
    filtered = df.sort_values(by='sim_score', ascending=False)
    similar_movies = filtered[filtered.index != idx].head(top_n)

    # Prepare results
    base_movie = df.loc[idx]
    results = []
    for _, row in similar_movies.iterrows():
        explanation = explain_recommendation(base_movie, row)
        results.append({
            'title': row.get('title', 'Unbekannter Titel'),
            'overview': row.get('overview', 'Keine Beschreibung verfügbar'),
            'genres': row.get('genres', 'Keine Genres verfügbar'),
            'cast': ', '.join(str(row.get('cast', 'Keine Cast-Information')).split(', ')[:5]),
            'director': row.get('director', 'Kein Regisseur verfügbar'),
            'keywords': row.get('keywords', 'Keine Stichworte verfügbar'),
            'tagline': row.get('tagline', 'Kein Slogan verfügbar'),
            'production_countries': row.get('production_countries', 'Keine Länderinfo'),
            'production_companies': row.get('production_companies', 'Keine Produktionsfirmen'),
            'release_date': row.get('release_date', 'Unbekanntes Veröffentlichungsdatum'),
            'vote_average': row.get('vote_average', 0),
            'explanation': explanation
        })
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        recommendations = recommend_movies(movie_title)
    return render_template("index.html", recommendations=recommendations)

@app.route('/titles')
def titles():
    title_list = df['title'].dropna().tolist()
    return jsonify(title_list)

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode)
