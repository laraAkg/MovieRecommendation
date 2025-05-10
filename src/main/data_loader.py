import pandas as pd
import numpy as np

def is_missing(value):
    return (
        pd.isnull(value) or
        (isinstance(value, str) and value.strip() == '') or
        (isinstance(value, (list, np.ndarray)) and len(value) == 0)
    )
def load_data(max_missing=2):
    # Load CSV files
    movies_df = pd.read_csv('../../data/movies_metadata.csv', low_memory=False)
    credits_df = pd.read_csv('../../data/credits.csv')
    keywords_df = pd.read_csv('../../data/keywords.csv')

    # Ensure 'id' columns are integers
    movies_df = movies_df[movies_df['id'].apply(lambda x: str(x).isdigit())]
    movies_df['id'] = movies_df['id'].astype(int)
    credits_df['id'] = credits_df['id'].astype(int)
    keywords_df['id'] = keywords_df['id'].astype(int)

    # Merge DataFrames on 'id'
    df = movies_df.merge(credits_df, on='id').merge(keywords_df, on='id')

    # Count missing values in each row
    missing_counts = df.apply(lambda row: sum(is_missing(value) for value in row), axis=1)

    # Keep rows with missing values less than or equal to the threshold
    df = df[missing_counts <= max_missing]

    return df

def filter_important_columns(df):
    """
    Filters the DataFrame to keep only the important columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing movie data.
    
    Returns:
        pd.DataFrame: A DataFrame with only the important columns.
    """
    important_columns = [
        'title', 'genres', 'cast', 'crew', 'keywords', 'overview', 'tagline', 'vote_count',
        'production_countries', 'production_companies', 'release_date', 'vote_average'
    ]
    
    # Keep only the important columns that exist in the DataFrame
    filtered_df = df[[col for col in important_columns if col in df.columns]]
    
    return filtered_df

