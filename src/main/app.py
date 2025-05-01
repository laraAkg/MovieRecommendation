import logging
from flask import Flask, render_template, request, jsonify, flash
import pickle
from functools import lru_cache
from recommendation import recommend_movies

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "PyTH0N_Fl4sk_SeCrEt_K3y"

# --- Load model + cache setup ---
with open("created_model/light_model.pkl", "rb") as f:
    df, tfidf_vectorizer, indices = pickle.load(f)
tfidf_matrix = tfidf_vectorizer.transform(df["combined_features"])

CACHE_SIZE = 128


@lru_cache(maxsize=CACHE_SIZE)
def cached_recommend(title: str):
    """
    Get cached movie recommendations for a given title.
    Args:
        title (str): The title of the movie for which recommendations are needed.
    Returns:
        list: A list of recommended movie titles.
    """

    return recommend_movies(
        title=title, df=df, tfidf_matrix=tfidf_matrix, indices=indices
    )


# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle the main page for movie recommendations.
    Processes POST requests to fetch recommendations based on user input.
    Logs the requested title, handles errors, and renders the index page.
    """

    recommendations = []
    if request.method == "POST":
        movie_title = request.form.get("movie_title", "").strip()
        logger.info(f"User requested: {movie_title!r}")
        try:
            recommendations = cached_recommend(movie_title)
        except ValueError as e:
            # Known errors (title not found)
            logger.warning(f"Error with input title: {e}")
            flash(str(e), category="warning")
        except Exception as e:
            # Unexpected error
            logger.exception("Unexpected error in index()")
            flash(
                "An internal error occurred. Please try again later.", category="danger"
            )

    return render_template("index.html", recommendations=recommendations)


@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 triggered: {request.path}")
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(e):
    # Logger.exception prints the stack trace
    logger.exception("500 Internal Server Error")
    return render_template("500.html"), 500


@app.route("/titles", methods=["GET"])
def titles():
    return jsonify(df["title"].tolist())


if __name__ == "__main__":
    # Reduce verbosity of Flask's built-in logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(debug=True)
