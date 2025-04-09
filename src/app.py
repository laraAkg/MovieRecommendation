from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Modell laden
with open('light_model.pkl', 'rb') as f:
    df, tfidf, indices = pickle.load(f)
print(df[['combined_features']].head())

def explain_recommendation(base_movie, recommended_movie):
    explanation = {}

    # Vergleiche Genres
    base_genres = set(base_movie['genres'].split())
    recommended_genres = set(recommended_movie['genres'].split())
    genre_overlap = base_genres.intersection(recommended_genres)
    if genre_overlap:
        explanation['genres'] = {
            'shared': list(genre_overlap),
            'count': len(genre_overlap),
            'details': f"Teilt die Genres: {', '.join(genre_overlap)}"
        }

    # Vergleiche Cast
    base_cast = set(base_movie['cast'].split(', '))  # Cast korrekt als Liste verarbeiten
    recommended_cast = set(recommended_movie['cast'].split(', '))  # Cast korrekt als Liste verarbeiten
    cast_overlap = base_cast.intersection(recommended_cast)
    if cast_overlap:
        explanation['cast'] = {
            'shared': list(cast_overlap),
            'count': len(cast_overlap),
            'details': f"Gemeinsame Schauspieler: {', '.join(cast_overlap)}"
        }

    # Vergleiche Director
    if base_movie['director'] == recommended_movie['director']:
        explanation['director'] = {
            'shared': base_movie['director'],
            'details': f"Gleicher Regisseur: {base_movie['director']}"
        }

    # Vergleiche Keywords
    base_keywords = set(base_movie['keywords'].split())
    recommended_keywords = set(recommended_movie['keywords'].split())
    keyword_overlap = base_keywords.intersection(recommended_keywords)
    if keyword_overlap:
        explanation['keywords'] = {
            'shared': list(keyword_overlap),
            'count': len(keyword_overlap),
            'details': f"Teilt die Keywords: {', '.join(keyword_overlap)}"
        }

    # Vergleiche Tagline
    base_tagline = base_movie.get('tagline', '').strip().lower()
    recommended_tagline = recommended_movie.get('tagline', '').strip().lower()
    if base_tagline and recommended_tagline and base_tagline == recommended_tagline:
        explanation['tagline'] = {
            'shared': base_tagline,
            'details': f"Teilt die gleiche Tagline: {base_tagline}"
        }

    # Füge die Ähnlichkeit hinzu
    explanation['similarity_score'] = {
        'score': recommended_movie['sim_score'],
        'details': f"Ähnlichkeitsscore: {recommended_movie['sim_score']:.2f}"
    }

    return explanation

def get_director(crew_list):
    if isinstance(crew_list, list):
        for m in crew_list:
            if isinstance(m, dict) and m.get('job') == 'Director':
                return m.get('name', '')
    return ''

def recommend_movies(title, top_n=9):
    title = title.lower().strip()
    if title not in indices:
        return []

    # Index des eingegebenen Titels abrufen
    idx = indices[title]

    # Ähnlichkeitsberechnung
    input_vector = tfidf.transform([df.loc[idx, 'combined_features']])
    sim_scores = cosine_similarity(input_vector, tfidf.transform(df['combined_features']))[0]
    df['sim_score'] = sim_scores

    # Sortiere nach Ähnlichkeit
    filtered = df.sort_values(by='sim_score', ascending=False)
    similar_movies = filtered[filtered.index != idx].head(top_n)

    # Ergebnisse aufbereiten
    base_movie = df.loc[idx]
    results = []
    for _, row in similar_movies.iterrows():
        explanation = explain_recommendation(base_movie, row)
        results.append({
            'title': row['title'],
            'overview': row.get('overview', 'Keine Beschreibung vorhanden'),
            'genres': row.get('genres', 'Keine Genres vorhanden'),
            'cast': ', '.join(row.get('cast', 'Keine Cast-Informationen vorhanden').split(', ')[:5]),
            'director': row.get('director', 'Kein Regisseur vorhanden'),
            'keywords': row.get('keywords', 'Keine Keywords vorhanden'),
            'tagline': row.get('tagline', 'Keine Tagline vorhanden'),
            'production_countries': row.get('production_countries', 'Keine Produktionsländer vorhanden'),
            'production_companies': row.get('production_companies', 'Keine Produktionsfirmen vorhanden'),
            'release_date': row.get('release_date', 'Unbekanntes Veröffentlichungsdatum'),
            'vote_average': row.get('vote_average', 0),
            'explanation': explanation  # Strukturierte Erklärung
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
    app.run(debug=True)
