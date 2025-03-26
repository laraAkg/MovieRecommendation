from flask import Flask, render_template, request, jsonify
import os
import joblib
from model import HybridRecommender
import pandas as pd

app = Flask(__name__)

# Pfad zur gespeicherten Modell-Datei
MODEL_PATH = "/Users/lara/Documents/MovieRecommendation/models/hybrid_recommender_model.pkl"

# Lade das gespeicherte Modell
if os.path.exists(MODEL_PATH):
    recommender = joblib.load(MODEL_PATH)
    if recommender is None:
        raise RuntimeError(f"Das Modell konnte nicht von {MODEL_PATH} geladen werden.")
else:
    raise RuntimeError(f"Das Modell {MODEL_PATH} wurde nicht gefunden. Bitte trainiere und speichere das Modell zuerst.")

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    query = ''
    all_titles = recommender.df['title'].dropna().unique().tolist()

    if request.method == 'POST':
        query = request.form['title']
        recs = recommender.get_recommendations(query)

        # Überprüfe, ob `recs` ein DataFrame ist
        if isinstance(recs, pd.DataFrame):
            recs_list = []
            for rec in recs.to_dict(orient='records'):
                explanation = recommender.explain_recommendation(query, rec['title'])
                recs_list.append({
                    "title": rec['title'],
                    "data": rec,
                    "explanation": explanation
                })
            recommendations = recs_list
        else:
            recommendations = recs

    return render_template(
        'index.html',
        recommendations=recommendations,
        query=query,
        all_titles=all_titles
    )

if __name__ == '__main__':
    app.run(debug=True)