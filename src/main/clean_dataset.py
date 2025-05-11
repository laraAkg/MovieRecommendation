"""
This script performs the following tasks:
1. Loads movie data with a specified maximum number of missing values.
2. Filters the dataset to retain only important columns.
3. Cleans the dataset by removing rows with missing titles and duplicate titles.
4. Saves the cleaned dataset to a MongoDB collection.

Modules:
- logging: For logging script progress and events.
- data_loader: Contains functions to load and filter the dataset.
- mongodb_handler: Handles MongoDB operations.
- os, dotenv: For environment variable management.

Environment Variables:
- MONGO_URI: MongoDB connection URI.
- DB_NAME: Name of the MongoDB database.
- COLLECTION_NAME: Name of the MongoDB collection.
"""
import logging
from data_loader import load_data, filter_important_columns
from mongodb_handler import MongoDBHandler
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Loading data...")
df = load_data(max_missing=2)

logger.info("Filtering important columns...")
df = filter_important_columns(df)

logger.info("Cleaning data...")

df.dropna(subset=['title'], inplace=True)
df = df.drop_duplicates(subset='title', inplace=False)


logger.info("Saving data to MongoDB...")
mongo_handler = MongoDBHandler(
    os.getenv("MONGO_URI"), os.getenv("DB_NAME"), os.getenv("COLLECTION_NAME")
)

mongo_handler.save_data(df.to_dict(orient="records"))

logger.info(f"{len(df)} cleaned movies successfully saved to MongoDB!")