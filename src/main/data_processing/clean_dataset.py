from helper.mongodb_handler import MongoDBHandler
import logging
import sys
from data_loader import load_data, filter_important_columns
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Loading data...")
df = load_data(max_missing=3)

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

logger.info(f"✅ {len(df)} cleaned movies successfully saved to MongoDB!")