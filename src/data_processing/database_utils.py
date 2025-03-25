from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_mongo_collection(db_name: str, collection_name: str):
    client = MongoClient('mongodb+srv://user:user@movierecommendation.2y0pw.mongodb.net/?retryWrites=true&w=majority&appName=MovieRecommendation')
    db = client[db_name]
    collection = db[collection_name]
    return collection

def save_clean_data_to_mongo(df, collection):
    collection.delete_many({})
    logging.info(f"All existing documents in the collection '{collection.name}' have been deleted.")
    data_dict = df.to_dict("records")
    collection.insert_many(data_dict)
    logging.info(f"Cleaned data saved to MongoDB: {collection.name}")