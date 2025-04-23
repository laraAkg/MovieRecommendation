from flask import Flask, render_template, request, jsonify
import pickle
import logging
from recommendation import recommend_movies  # aus recommendation.py

app = Flask(__name__)

# Modell laden
with open('created_model/light_model.pkl', 'rb') as f:
    df, tfidf, indices = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        recommendations = recommend_movies(movie_title, df, tfidf, indices)
    return render_template("index.html", recommendations=recommendations)

@app.route('/titles')
def titles():
    title_list = df['title'].dropna().tolist()
    return jsonify(title_list)

if __name__ == '__main__':
    app.run(debug=True)
