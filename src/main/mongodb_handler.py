from pymongo import MongoClient
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MongoDBHandler:
    def __init__(self, uri, db_name, collection_name):
        """
        Initializes the MongoDB handler with the specified URI, database name, and collection name.

        Args:
            uri (str): The connection URI for the MongoDB instance.
            db_name (str): The name of the database to connect to.
            collection_name (str): The name of the collection to interact with.
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_data(self, data, batch_size=1000):
        """
        Saves the provided data into the MongoDB collection in batches.
        Args:
            data (list): The data to be saved, represented as a list of documents.
            batch_size (int, optional): The number of documents to insert per batch. Defaults to 1000.
        Side Effects:
            Clears the existing data in the collection before inserting the new data.
            Prints the progress of batch insertion.
        """
        self.collection.delete_many({})
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            self.collection.insert_many(batch)
            print(f"Inserted batch {i // batch_size + 1} of {len(data) // batch_size + 1}")

    def load_data(self):
        """
        Loads data from the MongoDB collection into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the cleaned movies data.

        Raises:
            ValueError: If no data is found in the MongoDB collection.
            SystemExit: If an error occurs during data loading.
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