from pymongo import MongoClient

class MongoDBHandler:
    def __init__(self, uri, db_name, collection_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_data(self, data, batch_size=1000):
        """Save data to MongoDB in smaller batches."""
        self.collection.delete_many({})
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            self.collection.insert_many(batch)
            print(f"Inserted batch {i // batch_size + 1} of {len(data) // batch_size + 1}")