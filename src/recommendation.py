import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from explanation import explain_recommendation  # Wichtig!

def recommend_movies(title, df, tfidf, indices, top_n=9):
    title = title.lower().strip()
    if title not in indices:
        return [{'error': f"Der Titel '{title}' wurde nicht gefunden."}]
    
    idx = indices[title]
    input_vector = tfidf.transform([df.loc[idx, 'combined_features']])
    sim_scores = cosine_similarity(input_vector, tfidf.transform(df['combined_features']))[0]
    df['sim_score'] = sim_scores

    filtered = df.sort_values(by='sim_score', ascending=False)
    similar_movies = filtered[filtered.index != idx].head(top_n)

    base_movie = df.loc[idx]
    results = []
    for _, row in similar_movies.iterrows():
        explanation = explain_recommendation(base_movie, row)  # 👈 Hier passiert XAI-Magic
        results.append({
            'title': row.get('title', ''),
            'overview': row.get('overview', ''),
            'genres': row.get('genres', ''),
            'cast': ', '.join(str(row.get('cast', '')).split(', ')[:5]),
            'director': row.get('director', ''),
            'keywords': row.get('keywords', ''),
            'tagline': row.get('tagline', ''),
            'production_countries': row.get('production_countries', ''),
            'production_companies': row.get('production_companies', ''),
            'release_date': row.get('release_date', ''),
            'vote_average': row.get('vote_average', 0),
            'explanation': explanation  # 👈 wird an Template übergeben
        })
    return results
