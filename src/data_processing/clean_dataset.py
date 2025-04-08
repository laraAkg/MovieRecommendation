import pandas as pd
import ast
from pymongo import MongoClient

# 🔹 1. CSV-Dateien laden
movies_df = pd.read_csv('data/movies_metadata.csv', low_memory=False)
credits_df = pd.read_csv('data/credits.csv')
keywords_df = pd.read_csv('data/keywords.csv')

# 🔹 2. IDs bereinigen & casten
movies_df = movies_df[movies_df['id'].apply(lambda x: str(x).isdigit())]
movies_df['id'] = movies_df['id'].astype(int)
credits_df['id'] = credits_df['id'].astype(int)
keywords_df['id'] = keywords_df['id'].astype(int)

# 🔹 3. Mergen der drei DataFrames
df = movies_df.merge(credits_df, on='id').merge(keywords_df, on='id')

# 🔹 4. Nur relevante Spalten
df = df[['id', 'title', 'overview', 'tagline', 'genres', 'keywords', 'cast', 'crew']]

# 🔹 5. Parser für List-of-Dict-Spalten
def parse_column(val):
    try:
        return ast.literal_eval(val) if pd.notnull(val) else []
    except:
        return []

for col in ['genres', 'keywords', 'cast', 'crew']:
    df[col] = df[col].apply(parse_column)

# 🔹 6. Helferfunktionen
def get_names(obj_list, key='name', max_items=5):
    return ' '.join([obj[key].replace(' ', '') for obj in obj_list[:max_items] if key in obj])

def get_director(crew_list):
    for member in crew_list:
        if member.get('job') == 'Director':
            return member.get('name', '').replace(' ', '')
    return ''

# 🔹 7. Feature-Text generieren (kombiniert & gewichtet)
def combine_features(row):
    overview = str(row['overview']) if pd.notnull(row['overview']) else ''
    genres = get_names(row['genres']) * 2
    keywords = get_names(row['keywords']) * 2
    cast = get_names(row['cast'])
    director = get_director(row['crew']) * 2
    tagline = str(row['tagline']) if pd.notnull(row['tagline']) else ''
    return ' '.join([overview, genres, keywords, cast, director, tagline]).strip()

df['combined_features'] = df.apply(combine_features, axis=1)

# 🔹 8. Bereinigung: fehlende Titel oder Inhalte raus
initial_count = len(df)

# Entferne Einträge ohne Titel oder kombinierte Features
df.dropna(subset=['title', 'combined_features'], inplace=True)
missing_values_removed = initial_count - len(df)

# Entferne Einträge mit leeren kombinierten Features
initial_count = len(df)
df = df[df['combined_features'].str.strip() != '']
empty_features_removed = initial_count - len(df)

# Entferne Duplikate basierend auf dem Titel
initial_count = len(df)
df.drop_duplicates(subset='title', inplace=True)
duplicates_removed = initial_count - len(df)

# 🔹 9. Titel vereinheitlichen (lowercase für späteres Mapping)
df['title'] = df['title'].str.strip()

# Vorschau
print(df[['title', 'combined_features']].head())

# Logging der Bereinigungsschritte
print(f"🔍 Anzahl der entfernten Einträge:")
print(f"- Fehlende Werte (Titel oder Features): {missing_values_removed}")
print(f"- Leere kombinierte Features: {empty_features_removed}")
print(f"- Duplikate: {duplicates_removed}")

# 🔹 10. Verbindung zu MongoDB Atlas
client = MongoClient('mongodb+srv://user:user@movierecommendation.2y0pw.mongodb.net/?retryWrites=true&w=majority&appName=MovieRecommendation')
db = client['netflix_db']
collection = db['recommendation_data']

# 🔹 11. Collection leeren und Daten hochladen
collection.delete_many({})
collection.insert_many(df.to_dict(orient='records'))

print(f"✅ {len(df)} bereinigte Filme erfolgreich in MongoDB gespeichert!")