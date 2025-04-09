import pandas as pd
import pickle
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 1. MongoDB-Verbindung
client = MongoClient('mongodb+srv://user:user@movierecommendation.2y0pw.mongodb.net/?retryWrites=true&w=majority&appName=MovieRecommendation')
db = client['netflix_db']
collection = db['recommendation_data']

# 🔹 2. Daten laden
data = list(collection.find({}, {'_id': 0}))
df = pd.DataFrame(data)
print(f"✅ {len(df)} bereinigte Filme aus MongoDB geladen.")

# 🔹 3. Einzelne Features extrahieren
def extract_features(row):
    # Extrahiere relevante Attribute
    overview = str(row.get('overview', '')) if pd.notnull(row.get('overview')) else ''
    genres = ' '.join([g['name'] for g in row.get('genres', []) if isinstance(g, dict)])
    cast = ', '.join([c['name'] for c in row.get('cast', []) if isinstance(c, dict)])
    director = ', '.join([c['name'] for c in row.get('director', []) if isinstance(c, dict)])
    keywords = ' '.join([k['name'] for k in row.get('keywords', []) if isinstance(k, dict)])
    tagline = str(row.get('tagline', '')) if pd.notnull(row.get('tagline')) else ''
    production_countries = row.get('production_countries', 'Keine Produktionsländer vorhanden')
    production_companies = row.get('production_companies', 'Keine Produktionsfirmen vorhanden')
    release_date = row.get('release_date', 'Unbekanntes Veröffentlichungsdatum')
    vote_average = row.get('vote_average', 0)
    return overview, genres, cast, director, keywords, tagline, production_countries, production_companies, release_date, vote_average

# Wende die Funktion auf den DataFrame an
df[['overview', 'genres', 'cast', 'director', 'keywords', 'tagline', 'production_countries', 'production_companies', 'release_date', 'vote_average']] = df.apply(
    lambda row: pd.Series(extract_features(row)), axis=1
)

# 🔹 4. 'combined_features' erstellen
df['combined_features'] = df.apply(
    lambda row: ' '.join([
        row['overview'], row['genres'], row['cast'], row['director'], row['keywords'], row['tagline'],
        row['production_countries'], row['production_companies']
    ]).strip(),
    axis=1
)

# 🔹 5. TF-IDF-Vektorisierung
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# 🔹 6. Titel → Index Mapping (lowercase für spätere Suche)
df['title_lower'] = df['title'].str.lower()
indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()

# 🔹 7. Nur relevante Spalten behalten (Speicher sparen)
df_reduced = df[['title', 'title_lower', 'overview', 'genres', 'cast', 'director', 'keywords', 'tagline',
                 'production_countries', 'production_companies', 'release_date', 'vote_average', 'combined_features']]

# 🔹 8. Modell speichern
with open('light_model.pkl', 'wb') as f:
    pickle.dump((df_reduced, tfidf, indices), f)

print("✅ light_model.pkl erfolgreich gespeichert.")