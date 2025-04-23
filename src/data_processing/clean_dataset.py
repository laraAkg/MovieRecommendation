from data_loader import DataLoader
from data_cleaner import DataCleaner
from mongodb_handler import MongoDBHandler
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading data...")
df = DataLoader.load_data()

# Clean data
logger.info("Cleaning data...")
df = DataCleaner.clean_data(df)

# Save data to MongoDB
logger.info("Saving data to MongoDB...")
mongo_handler = MongoDBHandler(
    os.getenv("MONGO_URI"), os.getenv("DB_NAME"), os.getenv("COLLECTION_NAME")
)
mongo_handler.save_data(df.to_dict(orient="records"))

logger.info(f"✅ {len(df)} cleaned movies successfully saved to MongoDB!")