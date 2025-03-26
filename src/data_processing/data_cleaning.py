import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling duplicates, missing values, and formatting specific columns.
    """
    # Entferne Duplikate
    df = df.drop_duplicates()
    logging.info("Duplicates removed.")
    
    # Fehlende Werte vor der Bereinigung anzeigen
    missing_values = df.isnull().sum()
    logging.info(f"Missing values per column before cleaning:\n{missing_values}")

    # Standardwerte für fehlende Daten
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

    # Fehlende Werte mit Standardwerten auffüllen
    for column, default_value in default_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(default_value)

    # Formatierung der Spalte 'date_added'
    if 'date_added' in df.columns:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['date_added'] = df['date_added'].fillna(pd.Timestamp.now())
        df['date_added'] = df['date_added'].dt.strftime('%Y-%m-%d')

    # Bereinigung der 'duration'-Spalte
    if 'duration' in df.columns:
        # Extrahiere Minuten für Filme
        df['duration_mins'] = df['duration'].str.extract(r'(\d+)\s*[mM]in').astype(float)
        # Extrahiere Anzahl der Staffeln für Serien
        df['duration_seasons'] = df['duration'].str.extract(r'(\d+)\s*[sS]eason').astype(float)
        # Fülle fehlende Werte mit 0
        df['duration_mins'] = df['duration_mins'].fillna(0)
        df['duration_seasons'] = df['duration_seasons'].fillna(0)
        # Entferne die ursprüngliche 'duration'-Spalte
        df = df.drop(columns=['duration'])

    logging.info("Data cleaning completed.")
    return df
    # Standardwerte für fehlende Daten
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

    # Fehlende Werte mit Standardwerten auffüllen
    for column, default_value in default_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(default_value)

    # Formatierung der Spalte 'date_added'
    if 'date_added' in df.columns:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['date_added'] = df['date_added'].fillna(pd.Timestamp.now())
        df['date_added'] = df['date_added'].dt.strftime('%Y-%m-%d')

    # Bereinigung der 'duration'-Spalte
    if 'duration' in df.columns:
        # Extrahiere Minuten für Filme
        df['duration_mins'] = df['duration'].str.extract(r'(\d+)\s*[mM]in').astype(float)
        # Extrahiere Anzahl der Staffeln für Serien
        df['duration_seasons'] = df['duration'].str.extract(r'(\d+)\s*[sS]eason').astype(float)
        # Fülle fehlende Werte mit 0
        df['duration_mins'] = df['duration_mins'].fillna(0)
        df['duration_seasons'] = df['duration_seasons'].fillna(0)

    logging.info("Data cleaning completed.")
    return df

def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validates the input DataFrame by checking for the presence of required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    return True