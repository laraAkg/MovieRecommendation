from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity
from explanation import explain_recommendation


def recommend_movies(
    title: str,
    df: pd.DataFrame,
    tfidf_matrix: spmatrix,
    indices: Dict[str, int],
    top_n: int = 9,
) -> List[Dict[str, Any]]:
    """
    Recommends movies based on a given movie title using cosine similarity.
    Args:
        title (str): The title of the movie for which recommendations are to be generated.
        df (pd.DataFrame): A DataFrame containing movie metadata.
        tfidf_matrix (spmatrix): A precomputed TF-IDF matrix representing movie features.
        indices (Dict[str, int]): A dictionary mapping movie titles (lowercased and stripped) to their indices in the DataFrame.
        top_n (int, optional): The number of top recommendations to return. Defaults to 9.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing details of a recommended movie, including:
            - title (str): The title of the recommended movie.
            - overview (str): A brief description of the movie.
            - genres (str): The genres associated with the movie.
            - cast (str): A comma-separated string of up to 5 cast members.
            - director (str): The director of the movie.
            - keywords (str): Keywords associated with the movie.
            - tagline (str): The tagline of the movie.
            - production_countries (str): The countries where the movie was produced.
            - production_companies (str): The companies involved in the production of the movie.
            - release_date (str): The release date of the movie.
            - vote_average (float): The average user rating of the movie.
            - explanation (str): An explanation of why the movie was recommended.
    Raises:
        ValueError: If the given movie title is not found in the indices dictionary.
    """
    key = title.lower().strip()
    if key not in indices:
        raise ValueError(f"Title '{title}' was not found.")

    idx = indices[key]
    input_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    sorted_idx = np.argsort(sim_scores)[::-1]
    rec_indices = [i for i in sorted_idx if i != idx][:top_n]
    base_movie = df.iloc[idx].to_dict()
    recommendations: List[Dict[str, Any]] = []
    for i in rec_indices:
        row = df.iloc[i].to_dict()
        row["sim_score"] = float(sim_scores[i])
        explanation = explain_recommendation(base_movie, row)
        recommendations.append(
            {
                "title": row.get("title", ""),
                "overview": row.get("overview", ""),
                "genres": row.get("genres", ""),
                "cast": ", ".join(str(row.get("cast", "")).split(", ")[:5]),
                "director": row.get("director", ""),
                "keywords": row.get("keywords", ""),
                "tagline": row.get("tagline", ""),
                "production_countries": row.get("production_countries", ""),
                "production_companies": row.get("production_companies", ""),
                "release_date": row.get("release_date", ""),
                "vote_average": row.get("vote_average", 0.0),
                "explanation": explanation,
            }
        )

    return recommendations

def get_recommendations_knn(title, knn_model, tfidf_matrix, df, indices, top_n=10):
    """
    Generate movie recommendations using the k-NN model.
    Args:
        title (str): The title of the movie to base recommendations on.
        knn_model (NearestNeighbors): The trained k-NN model.
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix of combined features.
        df (pandas.DataFrame): The DataFrame containing movie data.
        indices (pandas.Series): A mapping of movie titles to their indices.
        top_n (int): The number of recommendations to return.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing details of recommended movies.
    """
    title = title.lower()
    if title not in indices:
        raise ValueError(f"Movie '{title}' not found in the dataset.")

    idx = indices[title]
    distances, neighbors = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    similar_indices = neighbors.flatten()[1:]  # Exclude the input movie itself

    recommendations = []
    for i in similar_indices:
        row = df.iloc[i].to_dict()
        recommendations.append(
            {
                "title": row.get("title", ""),
                "overview": row.get("overview", ""),
                "genres": row.get("genres", ""),
                "cast": row.get("cast", ""),
                "director": row.get("director", ""),
                "keywords": row.get("keywords", ""),
                "tagline": row.get("tagline", ""),
                "production_countries": row.get("production_countries", ""),
                "production_companies": row.get("production_companies", ""),
                "release_date": row.get("release_date", ""),
                "vote_average": row.get("vote_average", 0.0),
            }
        )

    return recommendations