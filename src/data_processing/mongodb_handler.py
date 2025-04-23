from pymongo import MongoClient

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