import pandas as pd
import pickle
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from model_evaluator import evaluate_tfidf, train_and_evaluate_knn, plot_all_metrics
from mongodb_handler import MongoDBHandler

MODEL_DIR = "created_model"
TFIDF_PARAMS = {"stop_words": "english", "max_features": 5000, "ngram_range": (1, 2)}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"✅ Objekt erfolgreich gespeichert unter {filename}")


def build_combined_features(df):
    return (
        df["title"].fillna("") + " " +
        df["production_companies"].fillna("").astype(str) + " " +
        df["production_countries"].fillna("").astype(str) + " " +
        df["overview"].fillna("") + " " +
        df["genres"].fillna("").astype(str) + " " +
        df["keywords"].fillna("").astype(str)
    )

def build_combined_features_v2(df):
    return (
        df["title"].fillna("") + " " +
        df["overview"].fillna("") + " " +
        df["genres"].fillna("").astype(str) + " " +
        df["cast"].fillna("").astype(str) + " " +
        df["crew"].fillna("").astype(str) + " " +
        df["tagline"].fillna("") + " " +
        df["keywords"].fillna("").astype(str)
    )


def main():

    mongo_handler = MongoDBHandler(
        os.getenv("MONGO_URI"), os.getenv("DB_NAME"), os.getenv("COLLECTION_NAME")
    )
    df = mongo_handler.load_data()

    df["combined_features"] = build_combined_features(df)
    df["combined_features_V2"] = build_combined_features_v2(df)


    tfidf_model = TfidfVectorizer(**TFIDF_PARAMS)
    tfidf_matrix = tfidf_model.fit_transform(df["combined_features"])

    test_titles = ["High School Musical", "High School Musical 2", "High School Musical 3: Senior Year", "Inception", "The Dark Knight", "The Dark Knight Rises", "The Matrix", "The Matrix Reloaded", "The Matrix Revolutions"] 
    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

    results, models = train_and_evaluate_knn(tfidf_matrix, df, indices, test_titles)
    best_metric = max(results, key=lambda metric: results[metric]["accuracy"])
    logger.info(f"Beste Distanzmetrik: {best_metric}")
    plot_all_metrics(results, output_file="all_metrics_performance.png")
    best_model = models[best_metric]
    save_pickle("created_model/knn_model.pkl", best_model)
    metrics_tfidf = evaluate_tfidf(tfidf_matrix, df, indices, test_titles)
    logger.info(metrics_tfidf)

    tfidf_matrix_v2 = tfidf_model.fit_transform(df["combined_features_V2"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_pickle("created_model/light_model.pkl", (df, tfidf_model, indices, tfidf_matrix_v2))

if __name__ == "__main__":
    main()