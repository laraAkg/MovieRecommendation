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

# 🔹 3. Sicherstellen, dass 'combined_features' vorhanden ist
if 'combined_features' not in df.columns:
    raise ValueError("❌ 'combined_features' fehlt im DataFrame!")

# 🔹 4. TF-IDF-Vektorisierung
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# 🔹 5. Titel → Index Mapping (lowercase für spätere Suche)
df['title_lower'] = df['title'].str.lower()
indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()

# 🔹 6. Nur relevante Spalten behalten (Speicher sparen)
df_reduced = df[['title', 'title_lower', 'combined_features']]

# 🔹 7. Modell speichern
with open('light_model.pkl', 'wb') as f:
    pickle.dump((df_reduced, tfidf, indices), f)

print("✅ light_model.pkl erfolgreich gespeichert.")
