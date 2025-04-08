from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Modell laden
with open('light_model.pkl', 'rb') as f:
    df, tfidf, indices = pickle.load(f)
print(df[['combined_features']].head())

def get_director(crew_list):
    if isinstance(crew_list, list):
        for m in crew_list:
            if isinstance(m, dict) and m.get('job') == 'Director':
                return m.get('name', '')
    return ''

def recommend_movies(title, top_n=10):
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
    results = []
    for _, row in similar_movies.iterrows():
        results.append({
            'title': row['title'],
            'overview': row.get('overview', 'Keine Beschreibung vorhanden')[:300] + "...",
            'genres': row.get('genres', 'Keine Genres vorhanden'),
            'cast': row.get('cast', 'Keine Cast-Informationen vorhanden'),
            'director': row.get('director', 'Kein Regisseur vorhanden')
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
