from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(title, df, tfidf, indices, top_n=9):
    """Recommend movies based on a given title."""
    title = title.lower().strip()
    if title not in indices:
        return [{'error': f"The title '{title}' was not found. Please try another title."}]

    # Get the index of the input title
    idx = indices[title]

    # Compute similarity scores
    input_vector = tfidf.transform([df.loc[idx, 'combined_features']])
    sim_scores = cosine_similarity(input_vector, tfidf.transform(df['combined_features']))[0]
    df['sim_score'] = sim_scores

    # Sort by similarity
    filtered = df.sort_values(by='sim_score', ascending=False)
    similar_movies = filtered[filtered.index != idx].head(top_n)

    # Prepare results
    base_movie = df.loc[idx]
    results = []
    for _, row in similar_movies.iterrows():
        results.append({
            'title': row.get('title', 'Unknown title'),
            'overview': row.get('overview', 'No description available'),
            'genres': row.get('genres', 'No genres available'),
            'cast': ', '.join(str(row.get('cast', 'No cast information available')).split(', ')[:5]),
            'director': row.get('director', 'No director available'),
            'keywords': row.get('keywords', 'No keywords available'),
            'tagline': row.get('tagline', 'No tagline available'),
            'production_countries': row.get('production_countries', 'No production countries available'),
            'production_companies': row.get('production_companies', 'No production companies available'),
            'release_date': row.get('release_date', 'Unknown release date'),
            'vote_average': row.get('vote_average', 0),
            'sim_score': row['sim_score']
        })
    return results