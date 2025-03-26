import pandas as pd
import logging
from data_processing.data_cleaning import clean_data, validate_data
from data_processing.database_utils import get_mongo_collection, save_clean_data_to_mongo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_CSV(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded records: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = load_data_from_CSV('./data/netflix_titles.csv')
    required_columns = ['title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'description', 'listed_in', 'type']
    if not validate_data(df, required_columns):
        logging.error("Data validation failed. Program will terminate.")
    else:
        df_clean = clean_data(df)
        collection = get_mongo_collection('netflix_db', 'netflix_content')
        save_clean_data_to_mongo(df_clean, collection)