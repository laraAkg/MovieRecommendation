import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt das DataFrame, indem Duplikate entfernt, fehlende Werte (als None) beibehalten 
    und bestimmte Spalten formatiert werden.
    """
    # Entferne Duplikate
    df = df.drop_duplicates()
    logging.info("Duplicates removed.")
    
    # Fehlende Werte vor der Bereinigung anzeigen
    missing_values = df.isnull().sum()
    logging.info(f"Missing values per column before cleaning:\n{missing_values}")

    # Sicherstellen, dass alle erforderlichen Spalten existieren.
    required_columns = [
        'rating', 'director', 'cast', 'country', 'date_added',
        'release_year', 'duration', 'description', 'listed_in',
        'type', 'title'
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
        else:
            # Alle NaN-Werte werden explizit zu None konvertiert.
            df[col] = df[col].where(df[col].notnull(), None)
    
    # Formatierung der Spalte 'date_added':
    if 'date_added' in df.columns:
        # Konvertiere in datetime; fehlerhafte Werte werden zu NaT
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        # Formatieren als 'YYYY-MM-DD' oder None, wenn der Wert fehlt.
        df['date_added'] = df['date_added'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)
    
    # Bereinigung der 'duration'-Spalte:
    if 'duration' in df.columns:
        # Extrahiere Minuten für Filme
        df['duration_mins'] = df['duration'].str.extract(r'(\d+)\s*[mM]in')
        # Extrahiere Anzahl der Staffeln für Serien
        df['duration_seasons'] = df['duration'].str.extract(r'(\d+)\s*[sS]eason')
        # Umwandlung in numerische Werte; falls nicht möglich, NaN, die wir dann zu None konvertieren
        df['duration_mins'] = pd.to_numeric(df['duration_mins'], errors='coerce')
        df['duration_seasons'] = pd.to_numeric(df['duration_seasons'], errors='coerce')
        df['duration_mins'] = df['duration_mins'].where(df['duration_mins'].notnull(), None)
        df['duration_seasons'] = df['duration_seasons'].where(df['duration_seasons'].notnull(), None)
        # Entferne die ursprüngliche 'duration'-Spalte
        df = df.drop(columns=['duration'])
    
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