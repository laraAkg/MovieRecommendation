import logging
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def evaluate_knn(model, tfidf_matrix, df, indices, test_titles, top_n=10):
    all_true, all_pred = [], []
    for title in test_titles:
        idx = indices.get(title.lower())
        if idx is None:
            logger.warning(f"'{title}' not found. Skipping.")
            continue
        _, neighbors = model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
        rec_titles = df.iloc[neighbors.flatten()[1:]]["title"].tolist()
        y_true, y_pred = _evaluate(rec_titles, idx, df, indices)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    return _compute_metrics(all_true, all_pred)

def evaluate_tfidf(tfidf_matrix, df, indices, test_titles, top_n=10):
    all_true, all_pred = [], []
    for title in test_titles:
        idx = indices.get(title.lower())
        if idx is None:
            logger.warning(f"'{title}' not found. Skipping.")
            continue
        sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        rec_indices = sims.argsort()[-top_n-1:-1][::-1]
        rec_titles = df.iloc[rec_indices]["title"].tolist()
        y_true, y_pred = _evaluate(rec_titles, idx, df, indices)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    return _compute_metrics(all_true, all_pred)

def _evaluate(recommended_titles, target_idx, df, indices):
    true_genres = set(df.loc[target_idx, "genres"].split(", "))
    true_keywords = set(df.loc[target_idx, "keywords"].split(", "))
    y_true = []
    y_pred = []

    for rec_title in recommended_titles:
        rec_idx = indices.get(rec_title.lower())
        if rec_idx is None:
            continue
        rec_genres = set(df.loc[rec_idx, "genres"].split(", "))
        rec_keywords = set(df.loc[rec_idx, "keywords"].split(", "))
        is_relevant = bool(true_genres & rec_genres and true_keywords & rec_keywords)
        y_true.append(1 if is_relevant else 0)
        y_pred.append(1)
    return y_true, y_pred

def _compute_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }