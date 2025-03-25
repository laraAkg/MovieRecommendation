# src/database.py

import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# .env Variablen laden (für die sichere Speicherung der Connection-URI)
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")  # deine MongoDB URI aus der .env-Datei
DB_NAME = "netflix_db"
COLLECTION_NAME = "netflix_shows"

def connect_mongo():
    """Verbindung zu MongoDB Atlas aufbauen."""
    client = MongoClient(MONGO_URI, server_api={"version": "1"})
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("Verbindung zu MongoDB erfolgreich!")
    return collection

def insert_data(df: pd.DataFrame):
    """Daten aus DataFrame in MongoDB einfügen."""
    collection = connect_mongo()
    records = df.to_dict(orient="records")
    collection.delete_many({})  # Collection vor dem Laden leeren (optional)
    collection.insert_many(records)
    print(f"{len(records)} Datensätze erfolgreich in MongoDB eingefügt.")

def load_data_from_mongo() -> pd.DataFrame:
    """Daten aus MongoDB abrufen und als DataFrame zurückgeben."""
    collection = connect_mongo()
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Beispielausführung
    df = pd.read_csv('data/netflix_titles_clean.csv')
    insert_data(df)
    df_mongo = load_data_from_mongo()
    print(df_mongo.head())
