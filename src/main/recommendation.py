from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
import ast
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def format_movie_info(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        production_companies = ast.literal_eval(row.get("production_companies", ""))
        if isinstance(production_companies, list):
            production_companies = ", ".join([company["name"] for company in production_companies if "name" in company])
        else:
            production_companies = ""
    except (ValueError, SyntaxError):
        production_companies = ""

    try:
        production_countries = ast.literal_eval(row.get("production_countries", ""))
        if isinstance(production_countries, list):
            production_countries = ", ".join([country["name"] for country in production_countries if "name" in country])
        else:
            production_countries = ""
    except (ValueError, SyntaxError):
        production_countries = ""

    try:
        genres = ast.literal_eval(row.get("genres", ""))
        if isinstance(genres, list):
            genres = ", ".join([genre["name"] for genre in genres if "name" in genre])
        else:
            genres = ""
    except (ValueError, SyntaxError):
        genres = ""

    try:
        cast = ast.literal_eval(row.get("cast", ""))
        if isinstance(cast, list):
            cast = ", ".join([actor["name"] for actor in cast[:5] if "name" in actor])  # Maximal 5 Schauspieler
        else:
            cast = ""
    except (ValueError, SyntaxError):
        cast = ""

    return {
        "title": row.get("title", ""),
        "overview": row.get("overview", ""),
        "genres": genres,
        "cast": cast,
        "director": row.get("director", ""),
        "keywords": row.get("keywords", ""),
        "tagline": row.get("tagline", ""),
        "production_countries": production_countries,
        "production_companies": production_companies,
        "release_date": row.get("release_date", ""),
        "vote_average": row.get("vote_average", 0.0),
    }

def recommend_movies(
    title: str,
    df: pd.DataFrame,
    tfidf_matrix: spmatrix,
    indices: Dict[str, int],
    top_n: int = 9,
) -> List[Dict[str, Any]]:
    """
    Generate movie recommendations using cosine similarity on TF-IDF vectors.
    """
    key = title.lower().strip()
    if key not in indices:
        raise ValueError(f"Movie '{title}' not found in dataset.")

    idx = indices[key]
    input_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    sorted_idx = np.argsort(sim_scores)[::-1]
    rec_indices = [i for i in sorted_idx if i != idx][:top_n]

    return [format_movie_info(df.iloc[i].to_dict()) for i in rec_indices]

def get_recommendations_knn(
    title: str,
    knn_model: NearestNeighbors,
    tfidf_matrix: spmatrix,
    df: pd.DataFrame,
    indices: Dict[str, int],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate movie recommendations using the k-NN model.
    """
    key = title.lower().strip()
    if key not in indices:
        raise ValueError(f"Movie '{title}' not found in dataset.")

    idx = indices[key]
    distances, neighbors = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    similar_indices = neighbors.flatten()[1:]  # Exclude the input movie itself

    return [format_movie_info(df.iloc[i].to_dict()) for i in similar_indices]
