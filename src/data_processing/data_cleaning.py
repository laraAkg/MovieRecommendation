import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    logging.info("Duplicates removed.")
    missing_values = df.isnull().sum()
    logging.info(f"Missing values per column before cleaning:\n{missing_values}")

    default_values = {
        'rating': 'Unknown rating',
        'director': 'Unknown director',
        'cast': 'Unknown cast',
        'country': 'Unknown country',
        'date_added': 'Unknown date added',
        'release_year': 'Unknown release year',
        'duration': 'Unknown duration',
        'description': 'Unknown description',
        'listed_in': 'Unknown listed in',
        'type': 'Unknown type',
        'title': 'Unknown title'
    }

    for column, default_value in default_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(default_value)

    if 'date_added' in df.columns:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['date_added'] = df['date_added'].fillna(pd.Timestamp.now())
        df['date_added'] = df['date_added'].dt.strftime('%Y-%m-%d')

    logging.info("Data cleaning completed.")
    return df

def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    return True