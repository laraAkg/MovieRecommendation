import pandas as pd
import pickle
import re
import logging
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import os
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()


def connect_to_mongodb(uri, db_name, collection_name):
    """
    Connects to a MongoDB database and retrieves a specific collection.
    Args:
        uri (str): The connection URI for the MongoDB instance.
        db_name (str): The name of the database to connect to.
        collection_name (str): The name of the collection to retrieve.
    Returns:
        pymongo.collection.Collection: The specified MongoDB collection.
    """
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]

def load_data_from_mongodb(collection):
    """
    Load data from a MongoDB collection and return it as a pandas DataFrame.
    This function retrieves all documents from the specified MongoDB collection,
    excluding the '_id' field, and converts the data into a pandas DataFrame.
    If the collection is empty, a ValueError is raised.
    Args:
        collection (pymongo.collection.Collection): The MongoDB collection object 
            from which data will be retrieved.
    Returns:
        pandas.DataFrame: A DataFrame containing the data from the MongoDB collection.
    Raises:
        ValueError: If no data is found in the MongoDB collection.
        Exception: If there is an error during the data retrieval process.
    """
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
        
def safe_get(row, key, default=''):
    """
    Safely retrieves a value from a dictionary-like object, ensuring that the value is not null.
    Args:
        row (dict): The dictionary-like object to retrieve the value from.
        key (str): The key to look up in the dictionary.
        default (str, optional): The default value to return if the key is not found or the value is null. Defaults to an empty string.
    Returns:
        str: The value associated with the key, converted to a string if not null, or the default value otherwise.
    """
    return str(row.get(key, default)) if pd.notnull(row.get(key)) else default

def extract_features(row):
    """
    Extracts various features from a movie data row.
    Args:
        row (dict): A dictionary containing movie data
    Returns:
        tuple: A tuple containing extracted features
    """
    overview = safe_get(row, 'overview')
    genres = ' '.join([g['name'] for g in row.get('genres', []) if isinstance(g, dict)])
    cast = ', '.join([c['name'] for c in row.get('cast', []) if isinstance(c, dict)])
    director = ', '.join([d['name'] for d in row.get('crew', []) if isinstance(d, dict) and d.get('job') == 'Director'])
    keywords = ' '.join([k['name'] for k in row.get('keywords', []) if isinstance(k, dict)])
    tagline = safe_get(row, 'tagline')
    production_countries = safe_get(row, 'production_countries', 'No production countries available')
    production_companies = safe_get(row, 'production_companies', 'No production companies available')
    release_date = safe_get(row, 'release_date', 'Unknown release date')
    vote_average = row.get('vote_average', 0)
    vote_count = row.get('vote_count', 0)
    return overview, genres, cast, director, keywords, tagline, production_countries, production_companies, release_date, vote_average, vote_count

def clean_text(text):
    """
    Cleans the input text by performing the following operations:
    1. Removes all characters that are not alphanumeric or whitespace.
    2. Replaces multiple consecutive whitespace characters with a single space.
    3. Strips leading and trailing whitespace from the text.
    Args:
        text (str): The input string to be cleaned.
    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_combined_features(df):
    """
    Creates a new column 'combined_features' in the given DataFrame by concatenating and cleaning 
    text from multiple columns. The combined text includes information such as the movie's overview, 
    genres, cast, director, keywords, tagline, production countries, production companies, and 
    normalized vote count.
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data. It must include the 
        following columns: 'overview', 'genres', 'cast', 'director', 'keywords', 'tagline', 
        'production_countries', 'production_companies', and 'vote_count_normalized'.
    Returns:
        pandas.DataFrame: The input DataFrame with an additional column 'combined_features', 
        which contains the cleaned and concatenated text data for each row.
    """
    df['combined_features'] = df.apply(
        lambda row: clean_text(' '.join([
            row['overview'], row['genres'], row['cast'], row['director'], row['keywords'], row['tagline'],
            row['production_countries'], row['production_companies'], str(row['vote_count_normalized'])
        ])),
        axis=1
    )
    return df

def normalize_column(df, column_name):
    """
    Normalize a specified column in a DataFrame using Min-Max scaling.
    This function adds a new column to the DataFrame with normalized values
    of the specified column. The new column is named as '<column_name>_normalized'.
    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        column_name (str): The name of the column to be normalized.
    Returns:
        pandas.DataFrame: The DataFrame with an additional column containing
        the normalized values of the specified column.
    """
    scaler = MinMaxScaler()
    df[f'{column_name}_normalized'] = scaler.fit_transform(df[[column_name]])
    return df

def save_model(filename, df_reduced, tfidf, indices):
    """
    Saves the provided model components to a file using pickle.
    Args:
        filename (str): The path to the file where the model components will be saved.
        df_reduced (pandas.DataFrame): The reduced DataFrame containing processed data.
        tfidf (sklearn.feature_extraction.text.TfidfVectorizer): The TF-IDF vectorizer used for text processing.
        indices (dict or list): The indices or mapping used for recommendations or lookups.
    Returns:
        None
    """
    with open(filename, 'wb') as f:
        pickle.dump((df_reduced, tfidf, indices), f)
    logger.info(f"✅ Model successfully saved to {filename}.")

def train_knn_model(tfidf_matrix, n_neighbors=10):
    """
    Trains a k-Nearest Neighbors (k-NN) model based on the TF-IDF vectors.
    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix of combined features.
        n_neighbors (int): The number of neighbors to use for recommendations.
    Returns:
        NearestNeighbors: The trained k-NN model.
    """
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    knn.fit(tfidf_matrix)
    logger.info("✅ k-NN model successfully trained.")
    return knn

def save_knn_model(filename, knn_model):
    """
    Saves the k-NN model to a file.
    Args:
        filename (str): The path to the file where the model will be saved.
        knn_model (NearestNeighbors): The k-NN model to save.
    Returns:
        None
    """
    with open(filename, 'wb') as f:
        pickle.dump(knn_model, f)
    logger.info(f"✅ k-NN model successfully saved to {filename}.")


def get_recommendations_tfidf(title, tfidf_matrix, df, indices, top_n=10):
    """
    Generate movie recommendations using the TF-IDF model.
    Args:
        title (str): The title of the movie to base recommendations on.
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix of combined features.
        df (pandas.DataFrame): The DataFrame containing movie data.
        indices (pandas.Series): A mapping of movie titles to their indices.
        top_n (int): The number of recommendations to return.
    Returns:
        list: A list of recommended movie titles.
    """
    title = title.lower()
    if title not in indices:
        logger.error(f"Movie '{title}' not found in the dataset.")
        return []

    idx = indices[title]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    return df.iloc[similar_indices]['title'].tolist()

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
        list: A list of recommended movie titles.
    """
    title = title.lower()
    if title not in indices:
        logger.error(f"Movie '{title}' not found in the dataset.")
        return []

    idx = indices[title]
    distances, neighbors = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    similar_indices = neighbors.flatten()[1:]  # Exclude the input movie itself
    return df.iloc[similar_indices]['title'].tolist()

# Main workflow
if __name__ == "__main__":
    # Connect to MongoDB
    collection = connect_to_mongodb(os.getenv("MONGO_URI"), os.getenv("DB_NAME"), os.getenv("COLLECTION_NAME"))

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
    model_dir = 'created_model'
    os.makedirs(model_dir, exist_ok=True)
    save_model('created_model/light_model.pkl', df_reduced, tfidf, indices)

    # Train k-NN-Model
    knn_model = train_knn_model(tfidf_matrix)

    # Save k-NN-Model
    save_knn_model('created_model/knn_model.pkl', knn_model)

    # Example movie title
    movie_title = "The Dark Knight"

    # Generate recommendations using TF-IDF
    tfidf_recommendations = get_recommendations_tfidf(movie_title, tfidf_matrix, df, indices)
    logger.info(f"TF-IDF Recommendations for '{movie_title}': {tfidf_recommendations}")

    # Generate recommendations using k-NN
    knn_recommendations = get_recommendations_knn(movie_title, knn_model, tfidf_matrix, df, indices)
    logger.info(f"k-NN Recommendations for '{movie_title}': {knn_recommendations}")