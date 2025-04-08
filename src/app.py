from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Flask initialisieren
app = Flask(__name__)

# 🔹 Modell laden
with open('light_model.pkl', 'rb') as f:
    df, tfidf, indices = pickle.load(f)

# 🔹 Empfehlungslogik
def recommend_movies(title, top_n=10):
    title = title.lower().strip()
    if title not in indices:
        return []
    
    idx = indices[title]
    input_vector = tfidf.transform([df.loc[idx, 'combined_features']])
    sim_scores = cosine_similarity(input_vector, tfidf.transform(df['combined_features']))[0]
    similar_indices = sim_scores.argsort()[::-1][1:top_n + 1]
    return df.iloc[similar_indices]['title'].tolist()

# 🔹 Routen
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        recommendations = recommend_movies(movie_title)

    return render_template('index.html', recommendations=recommendations)

# 🔹 App starten
if __name__ == '__main__':
    app.run(debug=True)
