import logging
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_knn(model, tfidf_matrix, df, indices, test_titles, top_n=10):
    """
    Evaluates the performance of a k-Nearest Neighbors (kNN) model for movie recommendations.

    Args:
        model: The trained kNN model.
        tfidf_matrix: Sparse matrix of TF-IDF features.
        df: DataFrame containing movie data.
        indices: Dictionary mapping movie titles to their indices.
        test_titles: List of movie titles to evaluate.
        top_n: Number of top recommendations to consider (default is 10).

    Returns:
        dict: Evaluation metrics computed from true and predicted labels.
    """
    all_true, all_pred = [], []
    for title in test_titles:
        idx = indices.get(title.lower())
        if idx is None:
            logger.warning(f"'{title}' not found. Skipping.")
            continue
        _, neighbors = model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n + 1)
        rec_titles = df.iloc[neighbors.flatten()[1:]]["title"].tolist()
        y_true, y_pred = _evaluate(rec_titles, idx, df, indices)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    return _compute_metrics(all_true, all_pred)


def evaluate_tfidf(tfidf_matrix, df, indices, test_titles, top_n=10):
    """
    Evaluates the performance of a TF-IDF-based recommendation system.

    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix representing item features.
        df (pandas.DataFrame): DataFrame containing item metadata, including titles.
        indices (dict): Mapping of item titles (lowercased) to their indices in the TF-IDF matrix.
        test_titles (list of str): List of titles to evaluate recommendations for.
        top_n (int, optional): Number of top recommendations to consider. Defaults to 10.

    Returns:
        dict: A dictionary containing evaluation metrics (e.g., precision, recall, F1-score).
    """
    all_true, all_pred = [], []
    for title in test_titles:
        idx = indices.get(title.lower())
        if idx is None:
            logger.warning(f"'{title}' not found. Skipping.")
            continue
        sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        rec_indices = sims.argsort()[-top_n - 1 : -1][::-1]
        rec_titles = df.iloc[rec_indices]["title"].tolist()
        y_true, y_pred = _evaluate(rec_titles, idx, df, indices)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    return _compute_metrics(all_true, all_pred)


def _evaluate(recommended_titles, target_idx, df, indices):
    """
    Evaluate the relevance of recommended titles based on genres and keywords.

    Args:
        recommended_titles (list): List of recommended movie titles.
        target_idx (int): Index of the target movie in the DataFrame.
        df (pd.DataFrame): DataFrame containing movie information with 'genres' and 'keywords' columns.
        indices (dict): Mapping of movie titles (lowercase) to their indices in the DataFrame.

    Returns:
        tuple: Two lists - y_true (ground truth relevance) and y_pred (predicted relevance).
    """
    true_genres = set(str(df.loc[target_idx, "genres"]).split(", "))
    true_keywords = set(str(df.loc[target_idx, "keywords"]).split(", "))
    y_true = []
    y_pred = []

    for rec_title in recommended_titles:
        rec_idx = indices.get(rec_title.lower())
        if rec_idx is None:
            continue
        rec_genres = set(str(df.loc[rec_idx, "genres"]).split(", "))
        rec_keywords = set(str(df.loc[rec_idx, "keywords"]).split(", "))
        is_relevant = bool(true_genres & rec_genres or true_keywords & rec_keywords)
        y_true.append(1 if is_relevant else 0)
        y_pred.append(1)
    return y_true, y_pred


def _compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for classification.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Dictionary containing precision, recall, and accuracy scores.
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def train_and_evaluate_knn(
    tfidf_matrix, df, indices, test_titles, metrics=["euclidean", "manhattan"]
):
    """
    Train and evaluate k-NN models with different distance metrics.
    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix representing item features.
        df (pandas.DataFrame): DataFrame containing item metadata, including titles.
        indices (dict): Mapping of item titles (lowercased) to their indices in the TF-IDF matrix.
        test_titles (list of str): List of titles to evaluate recommendations for.
        metrics (list of str): List of distance metrics to evaluate.
    Returns:
        dict: A dictionary containing evaluation metrics for each distance metric.
    """
    results = {}
    models = {}
    for metric in metrics:
        logger.info(f"Training k-NN model with metric: {metric}")
        knn_model = NearestNeighbors(n_neighbors=9, metric=metric, algorithm="brute")
        knn_model.fit(tfidf_matrix)

        metrics_knn = evaluate_knn(knn_model, tfidf_matrix, df, indices, test_titles)
        results[metric] = metrics_knn
        models[metric]= knn_model
    logger.info("Training and evaluation completed.")
    logger.info("Results:")
    for metric, metric_results in results.items():
        logger.info(f"Metric: {metric}, Results: {metric_results}")
    return results, models

def plot_all_metrics(results, output_file="all_metrics_performance.png"):
    """
    Plots bar charts for multiple metrics and saves the plot as an image.

    Args:
        results (dict): A dictionary where keys are metric names and values are 
                        dictionaries mapping labels to their corresponding values.
        output_file (str): The file path to save the generated plot. Defaults to 
                           "all_metrics_performance.png".

    Returns:
        None
    """
    labels = list(next(iter(results.values())).keys())
    metrics = list(results.keys())
    data = {metric: list(results[metric].values()) for metric in metrics}

    x = range(len(labels))
    width = 0.2
    plt.figure(figsize=(12, 8))

    for i, (metric, values) in enumerate(data.items()):
        plt.bar([pos + i * width for pos in x], values, width=width, label=metric)

    plt.xlabel("Metriken")
    plt.ylabel("Werte")
    plt.title("Performance der k-NN-Modelle für verschiedene Distanzmetriken")
    plt.xticks([pos + width for pos in x], labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"✅ Grafik erfolgreich gespeichert unter {output_file}")

def save_pickle(filename, obj):
    """
    Save an object to a file using pickle.

    Args:
        filename (str): The path to the file where the object will be saved.
        obj (any): The object to be serialized and saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"Model saved under {filename}")


def build_combined_features(df):
    """
    Combine multiple text-based columns from a DataFrame into a single string for each row.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the columns 'title', 
                           'production_companies', 'production_countries', 
                           'overview', 'genres', and 'keywords'.

    Returns:
        pd.Series: A Series where each row is a concatenated string of the specified columns.
    """
    return (
        df["title"].fillna("") + " " +
        df["production_companies"].fillna("").astype(str) + " " +
        df["production_countries"].fillna("").astype(str) + " " +
        df["overview"].fillna("") + " " +
        df["genres"].fillna("").astype(str) + " " +
        df["keywords"].fillna("").astype(str)
    )

def build_combined_features_v2(df):
    """
    Combines multiple text-based columns from a DataFrame into a single string for each row.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing columns like 'title', 'overview', 
                           'genres', 'cast', 'crew', 'tagline', and 'keywords'.

    Returns:
        pd.Series: A Series where each row is a concatenated string of the specified columns.
    """
    return (
        df["title"].fillna("") + " " +
        df["overview"].fillna("") + " " +
        df["genres"].fillna("").astype(str) + " " +
        df["cast"].fillna("").astype(str) + " " +
        df["crew"].fillna("").astype(str) + " " +
        df["tagline"].fillna("") + " " +
        df["keywords"].fillna("").astype(str)
    )
