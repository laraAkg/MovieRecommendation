import pandas as pd
import pickle
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from model_evaluator import evaluate_knn, evaluate_tfidf
from helper.mongodb_handler import MongoDBHandler

MODEL_DIR = "created_model"
TFIDF_PARAMS = {"stop_words": "english", "max_features": 5000, "ngram_range": (1, 2)}
KNN_PARAMS = {"n_neighbors": 10, "metric": "euclidean", "algorithm": "brute"}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()


def train_knn_model(tfidf_matrix, n_neighbors=10):
    knn = NearestNeighbors(**KNN_PARAMS)
    knn.fit(tfidf_matrix)
    logger.info("✅ k-NN model successfully trained.")
    return knn

def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"✅ Objekt erfolgreich gespeichert unter {filename}")


def build_combined_features(df):
    return (
        df["title"].fillna("") + " " +
        df["tagline"].fillna("") + " " +
        df["production_companies"].fillna("").astype(str) + " " +
        df["production_countries"].fillna("").astype(str) + " " +
        df["release_date"].fillna("") + " " +
        df["vote_average"].fillna("").astype(str) + " " +
        df["vote_count"].fillna("").astype(str) + " " +
        df["cast"].fillna("").astype(str) + " " +
        df["crew"].fillna("").astype(str) + " " +
        df["overview"].fillna("") + " " +
        df["genres"].fillna("").astype(str) + " " +
        df["keywords"].fillna("").astype(str)
    )


# Main workflow
def main():

    mongo_handler = MongoDBHandler(
        os.getenv("MONGO_URI"), os.getenv("DB_NAME"), os.getenv("COLLECTION_NAME")
    )
    df = mongo_handler.load_data()

    df["combined_features"] = build_combined_features(df)


    tfidf_model = TfidfVectorizer(**TFIDF_PARAMS)

    tfidf_matrix = tfidf_model.fit_transform(df["combined_features"])

    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_pickle("created_model/light_model.pkl", (df, tfidf_model, indices, tfidf_matrix))

    knn_model = train_knn_model(tfidf_matrix)

    save_pickle("created_model/knn_model.pkl", knn_model)

    test_titles = ["Inception", "The Matrix", "Titanic"]  # Beispiel-Testtitel
    metrics = evaluate_knn(knn_model, tfidf_matrix, df, indices, test_titles)
    logger.info(metrics)

    metrics_tfidf = evaluate_tfidf(tfidf_matrix, df, indices, test_titles)
    logger.info(metrics_tfidf)

if __name__ == "__main__":
    main()