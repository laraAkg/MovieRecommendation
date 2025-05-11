import pandas as pd
import numpy as np

def is_missing(value):
    """
    Check if a value is considered missing.
    Args:
        value: The value to check.

    Returns:
        bool: True if the value is missing, False otherwise.
    """
    return (
        pd.isnull(value) or
        (isinstance(value, str) and value.strip() == '') or
        (isinstance(value, (list, np.ndarray)) and len(value) == 0)
    )

def load_data(max_missing=2):
    """
    Loads and merges movie-related datasets, filters rows with excessive missing values, 
    and returns the cleaned DataFrame.
    Args:
        max_missing (int): Maximum number of missing values allowed per row. Defaults to 2.
    Returns:
        pandas.DataFrame: A DataFrame containing merged and filtered movie data.
    """
    movies_df = pd.read_csv('../../data/movies_metadata.csv', low_memory=False)
    credits_df = pd.read_csv('../../data/credits.csv')
    keywords_df = pd.read_csv('../../data/keywords.csv')
    movies_df = movies_df[movies_df['id'].apply(lambda x: str(x).isdigit())]
    movies_df['id'] = movies_df['id'].astype(int)
    credits_df['id'] = credits_df['id'].astype(int)
    keywords_df['id'] = keywords_df['id'].astype(int)
    df = movies_df.merge(credits_df, on='id').merge(keywords_df, on='id')
    missing_counts = df.apply(lambda row: sum(is_missing(value) for value in row), axis=1)
    df = df[missing_counts <= max_missing]

    return df

def filter_important_columns(df):
    """
    Filters the input DataFrame to retain only the important columns if they exist.
    Args:
        df (pd.DataFrame): The input DataFrame containing movie data.
    Returns:
        pd.DataFrame: A DataFrame containing only the important columns.
    """
    important_columns = [
        'title', 'genres', 'cast', 'crew', 'keywords', 'overview', 'tagline', 'vote_count',
        'production_countries', 'production_companies', 'release_date', 'vote_average'
    ]
    
    filtered_df = df[[col for col in important_columns if col in df.columns]]
    
    return filtered_df

