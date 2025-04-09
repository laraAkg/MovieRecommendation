from data_loader import DataLoader
from data_cleaner import DataCleaner
from mongodb_handler import MongoDBHandler
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MONGO_URI = 'mongodb+srv://user:user@movierecommendation.2y0pw.mongodb.net/?retryWrites=true&w=majority&appName=MovieRecommendation'
DB_NAME = 'netflix_db'
COLLECTION_NAME = 'recommendation_data'

# Load data
logger.info("Loading data...")
df = DataLoader.load_data()

# Clean data
logger.info("Cleaning data...")
df = DataCleaner.clean_data(df)

# Save data to MongoDB
logger.info("Saving data to MongoDB...")
mongo_handler = MongoDBHandler(MONGO_URI, DB_NAME, COLLECTION_NAME)
mongo_handler.save_data(df.to_dict(orient='records'))

logger.info(f"✅ {len(df)} cleaned movies successfully saved to MongoDB!")