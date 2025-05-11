"""
This script builds and evaluates a movie recommendation model using TF-IDF and K-Nearest Neighbors (KNN).

Modules:
- pandas: For data manipulation.
- logging: For logging messages.
- os: For environment variable and file handling.
- sklearn.feature_extraction.text.TfidfVectorizer: For TF-IDF vectorization.
- dotenv: For loading environment variables.
- model_evaluator: Custom module for model evaluation and utility functions.
- mongodb_handler: Custom module for MongoDB data handling.

Workflow:
1. Load movie data from MongoDB.
2. Generate combined feature representations for movies.
3. Train and evaluate KNN models using TF-IDF features.
4. Identify the best distance metric and save the best model.
5. Save a lightweight version of the model for future use.

Constants:
- MODEL_DIR: Directory to save the models.
- TFIDF_PARAMS: Parameters for TF-IDF vectorization.

Outputs:
- Performance metrics for different distance metrics.
- Saved models in the specified directory.
"""
import pandas as pd
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from model_evaluator import evaluate_tfidf, train_and_evaluate_knn, plot_all_metrics, save_pickle, build_combined_features, build_combined_features_v2
from mongodb_handler import MongoDBHandler

MODEL_DIR = "created_model"
TFIDF_PARAMS = {"stop_words": "english", "max_features": 5000, "ngram_range": (1, 2)}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()


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
logger.info(f"Best Distance metric: {best_metric}")
plot_all_metrics(results, output_file="all_metrics_performance.png")
best_model = models[best_metric]
save_pickle("created_model/knn_model.pkl", best_model)
metrics_tfidf = evaluate_tfidf(tfidf_matrix, df, indices, test_titles)
logger.info(metrics_tfidf)

tfidf_matrix_v2 = tfidf_model.fit_transform(df["combined_features_V2"])
os.makedirs(MODEL_DIR, exist_ok=True)
save_pickle("created_model/light_model.pkl", (df, tfidf_model, indices, tfidf_matrix_v2))
