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
        

def save_model(filename, df_reduced, tfidf, indices, tfidf_matrix):
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
        pickle.dump((df_reduced, tfidf, indices, tfidf_matrix), f)
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

    # Bereinige Genres
    df['genres'] = df['genres'].apply(
        lambda x: ', '.join([genre['name'] for genre in x]) if isinstance(x, list) else ''
    )

    # Bereinige Cast
    df['cast'] = df['cast'].apply(
        lambda x: ', '.join([member['name'] for member in x[:5]]) if isinstance(x, list) else ''
    )

    # Bereinige Keywords
    df['keywords'] = df['keywords'].apply(
        lambda x: ', '.join([keyword['name'] for keyword in x]) if isinstance(x, list) else ''
    )

    # Bereinige Crew (z. B. nur Regisseure)
    df['crew'] = df['crew'].apply(
        lambda x: ', '.join([member['name'] for member in x if member['job'] == 'Director']) if isinstance(x, list) else ''
    )

    # Reduziere den DataFrame auf relevante Spalten
    df_reduced = df[['title', 'overview', 'genres', 'cast', 'crew', 'keywords', 'tagline',
                     'production_countries', 'production_companies', 'release_date', 'vote_average', 'vote_count']]


    df_reduced = df[['title', 'overview', 'genres', 'cast', 'crew', 'keywords', 'tagline',
                     'production_countries', 'production_companies', 'release_date', 'vote_average', 'vote_count'
                    ]]


        # Combine text columns into a single column
    df['combined_features'] = (
        df['title'].fillna('') + ' ' +
        df['tagline'].fillna('') + ' ' +
        df['production_companies'].fillna('').astype(str) + ' ' +
        df['production_countries'].fillna('').astype(str) + ' ' +
        df['release_date'].fillna('') + ' ' +
        df['vote_average'].fillna('').astype(str) + ' ' +
        df['vote_count'].fillna('').astype(str) + ' ' +
        df['cast'].fillna('').astype(str) + ' ' +
        df['crew'].fillna('').astype(str) + ' ' +
        df['overview'].fillna('') + ' ' +
        df['genres'].fillna('').astype(str) + ' ' +
        df['keywords'].fillna('').astype(str) + ' ' 
    )
    

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

    # Save the model
    model_dir = 'created_model'
    os.makedirs(model_dir, exist_ok=True)
    save_model('created_model/light_model.pkl', df_reduced, tfidf, indices, tfidf_matrix)

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