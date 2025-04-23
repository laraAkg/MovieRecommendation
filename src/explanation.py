import logging

logger = logging.getLogger(__name__)

def explain_recommendation(base_movie, recommended_movie):
    """Generate an explanation for why a movie was recommended."""
    explanation = {}

    # Compare genres
    base_genres = set(str(base_movie.get('genres', '')).split())
    recommended_genres = set(str(recommended_movie.get('genres', '')).split())
    genre_overlap = base_genres.intersection(recommended_genres)
    if genre_overlap:
        explanation['genres'] = {
            'shared': list(genre_overlap),
            'count': len(genre_overlap),
            'details': f"{', '.join(genre_overlap)}"
        }

    # Compare cast
    base_cast = set(str(base_movie.get('cast', '')).split(', '))
    recommended_cast = set(str(recommended_movie.get('cast', '')).split(', '))
    cast_overlap = base_cast.intersection(recommended_cast)
    if cast_overlap:
        explanation['cast'] = {
            'shared': list(cast_overlap),
            'count': len(cast_overlap),
            'details': f"{', '.join(cast_overlap)}"
        }

    # Compare director
    if base_movie.get('director') == recommended_movie.get('director'):
        explanation['director'] = {
            'shared': base_movie.get('director'),
            'details': f"{base_movie.get('director')}"
        }

    # Compare keywords
    base_keywords = set(str(base_movie.get('keywords', '')).split())
    recommended_keywords = set(str(recommended_movie.get('keywords', '')).split())
    keyword_overlap = base_keywords.intersection(recommended_keywords)
    if keyword_overlap:
        explanation['keywords'] = {
            'shared': list(keyword_overlap),
            'count': len(keyword_overlap),
            'details': f"{', '.join(keyword_overlap)}"
        }

    # Compare tagline
    base_tagline = str(base_movie.get('tagline', '')).strip().lower()
    recommended_tagline = str(recommended_movie.get('tagline', '')).strip().lower()
    if base_tagline and recommended_tagline and base_tagline == recommended_tagline:
        explanation['tagline'] = {
            'shared': base_tagline,
            'details': f"{base_tagline}"
        }

    # Add similarity score
    explanation['similarity_score'] = {
        'score': recommended_movie.get('sim_score', 0),
        'details': f"{recommended_movie.get('sim_score', 0):.2f}"
    }

    logger.info(f"Explanation for {recommended_movie.get('title', '')}: {explanation}")
    return explanation