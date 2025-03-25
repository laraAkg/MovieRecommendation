from flask import Flask, render_template, request, jsonify
from model import HybridRecommender

app = Flask(__name__)

# Modell initialisieren
recommender = HybridRecommender('data/netflix_titles_clean.csv')
recommender.preprocess()
recommender.vectorize_and_combine()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    query = ''
    all_titles = recommender.df['title'].dropna().unique().tolist()

    if request.method == 'POST':
        query = request.form['title']
        recs = recommender.get_recommendations(query)
        if isinstance(recs, str):
            recommendations = recs
        else:
            recs_list = []
            for rec in recs.to_dict(orient='records'):
                explanation = recommender.explain_recommendation(query, rec['title'])
                recs_list.append({
                    "title": rec['title'],
                    "data": rec,
                    "explanation": explanation
                })
            recommendations = recs_list

    return render_template(
        'index.html',
        recommendations=recommendations,
        query=query,
        all_titles=all_titles
    )

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()
    title = data.get('title', None)
    if not title:
        return jsonify({"error": "Kein Titel angegeben"}), 400

    recs = recommender.get_recommendations(title)
    if isinstance(recs, str):
        return jsonify({"error": recs}), 404

    enriched_recs = []
    for rec in recs.to_dict(orient='records'):
        explanation = recommender.explain_recommendation(title, rec['title'])
        enriched_recs.append({
            "title": rec['title'],
            "data": rec,
            "explanation": explanation
        })

    return jsonify({
        "input_title": title,
        "recommendations": enriched_recs
    })

# Optional: API zum Anpassen der Gewichte (Bonus)
@app.route('/api/set_weights', methods=['POST'])
def set_weights():
    data = request.get_json()
    new_weights = data.get('weights', None)
    if not new_weights:
        return jsonify({"error": "Keine Gewichte übergeben"}), 400
    try:
        for key in recommender.weights.keys():
            if key in new_weights:
                recommender.weights[key] = float(new_weights[key])
        return jsonify({"message": "Gewichte erfolgreich angepasst", "new_weights": recommender.weights})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
