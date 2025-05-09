from pymongo import MongoClient
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MongoDBHandler:
    def __init__(self, uri, db_name, collection_name):
        """
        Initializes a connection to a MongoDB collection.
        Args:
            uri (str): The MongoDB connection URI.
            db_name (str): The name of the database to connect to.
            collection_name (str): The name of the collection within the database.
        Attributes:
            client (MongoClient): The MongoDB client instance.
            db (Database): The database instance.
            collection (Collection): The collection instance within the database.
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_data(self, data, batch_size=1000):
        """
        Saves the provided data into the MongoDB collection in batches.
        This method first clears the existing data in the collection by deleting all documents.
        Then, it inserts the new data in batches of the specified size.
        Args:
            data (list[dict]): A list of dictionaries representing the data to be saved.
            batch_size (int, optional): The number of documents to insert per batch. Defaults to 1000.
        Returns:
            None
        """ 
        self.collection.delete_many({})
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            self.collection.insert_many(batch)
            print(f"Inserted batch {i // batch_size + 1} of {len(data) // batch_size + 1}")

    def load_data(self):
        """
        Load data from the MongoDB collection and return it as a pandas DataFrame.
        This function retrieves all documents from the specified MongoDB collection,
        excluding the '_id' field, and converts the data into a pandas DataFrame.
        If the collection is empty, a ValueError is raised.
        Returns:
            pandas.DataFrame: A DataFrame containing the data from the MongoDB collection.
        Raises:
            ValueError: If no data is found in the MongoDB collection.
            Exception: If there is an error during the data retrieval process.
        """
        try:
            data = list(self.collection.find({}, {'_id': 0}))
            if not data:
                raise ValueError("No data found in the MongoDB collection.")
            df = pd.DataFrame(data)
            logger.info(f"✅ {len(df)} cleaned movies loaded from MongoDB.")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from MongoDB: {e}")
            exit(1)