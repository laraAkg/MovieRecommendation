# src/eda.py

import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Lädt die Netflix CSV-Daten."""
    df = pd.read_csv(filepath)
    print(f"Datensätze geladen: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt den Datensatz (fehlende Werte, Duplikate etc.)."""
    # Duplikate entfernen
    df.drop_duplicates(inplace=True)
    
    # Fehlende Werte anzeigen
    print("Fehlende Werte je Spalte:")
    print(df.isnull().sum())
    
    # Einfache Imputation: Missing 'rating' und 'director' durch 'Unknown'
    df['rating'].fillna('Unknown', inplace=True)
    df['director'].fillna('Unknown', inplace=True)
    
    # 'date_added' in datetime konvertieren
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    
    # Fehlende 'country' mit 'Unknown' füllen
    df['country'].fillna('Unknown', inplace=True)
    
    return df

def save_clean_data(df: pd.DataFrame, filepath: str):
    """Bereinigte Daten lokal abspeichern."""
    df.to_csv(filepath, index=False)
    print(f"Bereinigte Daten gespeichert unter: {filepath}")

if __name__ == "__main__":
    # Beispielausführung
    df = load_data('data/netflix_titles.csv')
    df_clean = clean_data(df)
    save_clean_data(df_clean, 'data/netflix_titles_clean.csv')
