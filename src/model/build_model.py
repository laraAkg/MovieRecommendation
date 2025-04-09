import pandas as pd
import pickle
import re
import logging
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URI = 'mongodb+srv://user:user@movierecommendation.2y0pw.mongodb.net/?retryWrites=true&w=majority&appName=MovieRecommendation'
DB_NAME = 'netflix_db'
COLLECTION_NAME = 'recommendation_data'

def connect_to_mongodb(uri, db_name, collection_name):
    """Connect to MongoDB and return the collection."""
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]

def load_data_from_mongodb(collection):
    """Load data from MongoDB and return it as a DataFrame."""
    try:
        data = list(collection.find({}, {'_id': 0}))
        if not data:
            raise ValueError("No data found in the MongoDB collection.")
        df = pd.DataFrame(data)
        logger.info(f"✅ {len(df)} cleaned movies loaded from MongoDB.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from MongoDB: {e}")
        exit(1)

def extract_features(row):
    """Extract relevant features from a row."""
    def safe_get(key, default=''):
        return str(row.get(key, default)) if pd.notnull(row.get(key)) else default

    overview = safe_get('overview')
    genres = ' '.join([g['name'] for g in row.get('genres', []) if isinstance(g, dict)])
    cast = ', '.join([c['name'] for c in row.get('cast', []) if isinstance(c, dict)])
    director = ', '.join([d['name'] for d in row.get('crew', []) if isinstance(d, dict) and d.get('job') == 'Director'])
    keywords = ' '.join([k['name'] for k in row.get('keywords', []) if isinstance(k, dict)])
    tagline = safe_get('tagline')
    production_countries = safe_get('production_countries', 'No production countries available')
    production_companies = safe_get('production_companies', 'No production companies available')
    release_date = safe_get('release_date', 'Unknown release date')
    vote_average = row.get('vote_average', 0)
    vote_count = row.get('vote_count', 0)
    return overview, genres, cast, director, keywords, tagline, production_countries, production_companies, release_date, vote_average, vote_count

def clean_text(text):
    """Clean text by removing special characters and extra spaces."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_combined_features(df):
    """Create the combined features column."""
    df['combined_features'] = df.apply(
        lambda row: clean_text(' '.join([
            row['overview'], row['genres'], row['cast'], row['director'], row['keywords'], row['tagline'],
            row['production_countries'], row['production_companies'], str(row['vote_count_normalized'])
        ])),
        axis=1
    )
    return df

def normalize_column(df, column_name):
    """Normalize a numerical column using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[f'{column_name}_normalized'] = scaler.fit_transform(df[[column_name]])
    return df

def save_model(filename, df_reduced, tfidf, indices):
    """Save the model components to a file."""
    with open(filename, 'wb') as f:
        pickle.dump((df_reduced, tfidf, indices), f)
    logger.info(f"✅ Model successfully saved to {filename}.")

# Main workflow
if __name__ == "__main__":
    # Connect to MongoDB
    collection = connect_to_mongodb(MONGO_URI, DB_NAME, COLLECTION_NAME)

    # Load data
    df = load_data_from_mongodb(collection)

    # Extract features
    df[['overview', 'genres', 'cast', 'director', 'keywords', 'tagline', 'production_countries', 
        'production_companies', 'release_date', 'vote_average', 'vote_count']] = df.apply(
        lambda row: pd.Series(extract_features(row)), axis=1
    )

    # Normalize vote_count
    df = normalize_column(df, 'vote_count')

    # Create combined features
    df = create_combined_features(df)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Title → Index Mapping
    df['title_lower'] = df['title'].str.lower()
    indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()

    # Keep only relevant columns
    df_reduced = df[['title', 'title_lower', 'overview', 'genres', 'cast', 'director', 'keywords', 'tagline',
                     'production_countries', 'production_companies', 'release_date', 'vote_average', 'vote_count',
                     'vote_count_normalized', 'combined_features']]

    # Save the model
    save_model('light_model.pkl', df_reduced, tfidf, indices)