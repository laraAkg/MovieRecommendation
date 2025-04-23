import logging
from flask import Flask, render_template, request, jsonify, flash
import pickle
from functools import lru_cache

from recommendation import recommend_movies

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "dein-geheimes-key"  # nötig für flash()

# --- Modell laden + Cache wie gehabt ---
with open('created_model/light_model.pkl', 'rb') as f:
    df, tfidf_vectorizer, indices = pickle.load(f)
tfidf_matrix = tfidf_vectorizer.transform(df['combined_features'])

CACHE_SIZE = 128
@lru_cache(maxsize=CACHE_SIZE)
def cached_recommend(title: str):
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
    if request.method == 'POST':
        movie_title = request.form.get('movie_title', '').strip()
        logger.info(f"User hat angefragt: {movie_title!r}")
        try:
            recommendations = cached_recommend(movie_title)
        except ValueError as e:
            # Bekannte Fehler (Titel nicht gefunden)
            logger.warning(f"Fehler bei Eingabetitel: {e}")
            flash(str(e), category="warning")
        except Exception as e:
            # Unvorhergesehener Fehler
            logger.exception("Unerwarteter Fehler in index()")
            flash("Ein interner Fehler ist aufgetreten. Bitte versuche es später erneut.", category="danger")

    return render_template(
        "index.html",
        recommendations=recommendations
    )

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 aufgerufen: {request.path}")
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    # Logger.exception druckt Stack-Trace
    logger.exception("500 Internal Server Error")
    return render_template("500.html"), 500

@app.route('/titles')
def titles():
    return jsonify(df['title'].dropna().tolist())

if __name__ == '__main__':
    # Setze Flask-eigenes Logging etwas leiser
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    app.run(debug=True)
