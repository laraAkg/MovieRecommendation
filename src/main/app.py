import logging
from flask import Flask, render_template, request, jsonify, flash
import pickle
from recommendation import recommend_movies
import os
from recommendation import get_recommendations_knn
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

with open("created_model/light_model.pkl", "rb") as f:
    df, tfidf_vectorizer, indices, tfidf_matrix = pickle.load(f)
    
with open("created_model/knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle the main page for movie recommendations.
    Processes POST requests to fetch recommendations based on user input.
    """
    recommendations = []
    model_type = "tfidf"  # Default to TF-IDF
    if request.method == "POST":
        movie_title = request.form.get("movie_title", "").strip()
        model_type = request.form.get("model_type", "tfidf").strip() 
        logger.info(f"User requested: {movie_title!r} using model: {model_type!r}")
        try:
            if model_type == "knn":
                recommendations = get_recommendations_knn(
                    title=movie_title,
                    knn_model=knn_model,
                    tfidf_matrix=tfidf_matrix,
                    df=df,
                    indices=indices,
                )
            else:
                recommendations = recommend_movies(
                    title=movie_title,
                    df=df,
                    tfidf_matrix=tfidf_matrix,
                    indices=indices,
                )
        except ValueError as e:
            logger.warning(f"Error with input title: {e}")
            flash(str(e), category="warning")
        except Exception as e:
            logger.exception("Unexpected error in index()")
            flash(
                "An internal error occurred. Please try again later.", category="danger"
            )
    return render_template("index.html", recommendations=recommendations, model_type=model_type)

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 triggered: {request.path}")
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(e):
    logger.exception("500 Internal Server Error")
    return render_template("500.html"), 500


@app.route("/titles", methods=["GET"])
def titles():
    return jsonify(df["title"].tolist())


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(debug=True)
