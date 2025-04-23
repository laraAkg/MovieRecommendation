"""
recommendation.py

Content-Based Movie Recommendation mit Erklärungen (XAI).
"""

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
    top_n: int = 9
) -> List[Dict[str, Any]]:
    """
    Gibt die Top-N Film-Empfehlungen für 'title' zurück.

    Args:
        title:          Originalfilmtitel (beliebige Groß-/Kleinschreibung).
        df:             DataFrame mit allen Filmen und einer Spalte 'combined_features'.
        tfidf_matrix:   Vorab berechnete TF-IDF-Matrix (Form: [n_movies, n_features]).
        indices:        Mapping von bereinigtem Titel (lower().strip()) → Zeilenindex in df.
        top_n:          Anzahl der gewünschten Empfehlungen (default: 9).

    Returns:
        Liste von Dicts, je eines pro empfohlenem Film, mit Schlüsseln:
        'title', 'overview', 'genres', 'cast', 'director',
        'keywords', 'tagline', 'production_countries',
        'production_companies', 'release_date', 'vote_average', 'explanation'.

    Raises:
        ValueError: Wenn der Titel nicht in `indices` gefunden wird.
    """
    key = title.lower().strip()
    if key not in indices:
        raise ValueError(f"Titel '{title}' wurde nicht gefunden.")

    idx = indices[key]
    # 1) Input-Vektor aus dem pre-computed tfidf_matrix ziehen
    input_vec = tfidf_matrix[idx]

    # 2) Kosinus-Ähnlichkeit zu allen Filmen
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()

    # 3) Indices der Top-N (außer dem Originalfilm)
    sorted_idx = np.argsort(sim_scores)[::-1]
    rec_indices = [i for i in sorted_idx if i != idx][:top_n]

    # 4) Basisfilm als dict
    base_movie = df.iloc[idx].to_dict()

    # 5) Empfehlungen aufbereiten
    recommendations: List[Dict[str, Any]] = []
    for i in rec_indices:
        # Film-Daten als Dict
        row = df.iloc[i].to_dict()
        # echten Similarity-Score mitgeben
        row['sim_score'] = float(sim_scores[i])

        # Erklärung erzeugen (XAI)
        explanation = explain_recommendation(base_movie, row)

        recommendations.append({
            'title':                row.get('title', ''),
            'overview':             row.get('overview', ''),
            'genres':               row.get('genres', ''),
            'cast':                 ', '.join(str(row.get('cast', '')).split(', ')[:5]),
            'director':             row.get('director', ''),
            'keywords':             row.get('keywords', ''),
            'tagline':              row.get('tagline', ''),
            'production_countries': row.get('production_countries', ''),
            'production_companies': row.get('production_companies', ''),
            'release_date':         row.get('release_date', ''),
            'vote_average':         row.get('vote_average', 0.0),
            'explanation':          explanation
        })

    return recommendations
