from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Modell laden
with open('light_model.pkl', 'rb') as f:
    df, tfidf, indices = pickle.load(f)

def recommend_movies(title, language=None, min_year=None, min_rating=None, top_n=10):
    title = title.lower().strip()
    if title not in indices:
        return []

    idx = indices[title]
    input_vector = tfidf.transform([df.loc[idx, 'combined_features']])
    sim_scores = cosine_similarity(input_vector, tfidf.transform(df['combined_features']))[0]
    df['sim_score'] = sim_scores

    filtered = df.copy()
    if language:
        filtered = filtered[filtered.get('original_language', '').str.lower() == language.lower()]
    if min_year:
        filtered = filtered[filtered.get('release_date', '').str[:4].astype(str) >= str(min_year)]
    if min_rating:
        filtered = filtered[filtered.get('vote_average', 0) >= float(min_rating)]

    filtered = filtered.sort_values(by='sim_score', ascending=False)
    return filtered[filtered.index != idx]['title'].head(top_n).tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        language = request.form.get('language')
        min_year = request.form.get('min_year')
        min_rating = request.form.get('min_rating')
        recommendations = recommend_movies(movie_title, language, min_year, min_rating)
    return render_template('index.html', recommendations=recommendations)

@app.route('/titles')
def titles():
    title_list = df['title'].dropna().tolist()
    return jsonify(title_list)

if __name__ == '__main__':
    app.run(debug=True)
