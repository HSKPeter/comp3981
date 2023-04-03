import pymongo
from log_util import logger
from uuid import uuid4
from slack_alert import AlertSender


class MongoStorageClient:
    _default_collection_name = uuid4().hex
    _instance = None
    _alert_sender = AlertSender()
    _db_name = "sudoku"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    def __init__(self):
        self.collection_name = self._default_collection_name
        self.connection_string = "localhost:27017"
        self.client = pymongo.MongoClient()
        self.create_collection()
        self._instance = self

    def check_collection_exists(self):
        """
        Check if a mongo collection exists
        """
        return self.client.get_database(self._db_name).get_collection(self.collection_name).exists()

    def create_collection(self):
        """
        Create a collection in mongo db
        """
        collection = self.client.get_database(self._db_name).create_collection(self.collection_name)
        collection.create_index("id")

        logger.info(f"Created Mongo Collection {self.collection_name}")
        self._alert_sender.send(f"Created Mongo Collection {self.collection_name}")

    def save_data(self, data):
        """
        Update data in Mongo DB
        """
        # Update data with specific ID in Mongo DB
        query_filter = {"$id": data["id"]}

        self.client.get_database(self._db_name).get_collection(self.collection_name).update_one(query_filter, {"$set": data}, upsert=True)

        # self.client.get_database(self._db_name).get_collection(self.collection_name).insert_one(data)

    def find_node(self, node_id):
        """
        Read node data from Mongo DB
        """
        return self.client.get_database(self._db_name).get_collection(self.collection_name).find_one({"id": node_id})



if __name__ == "__main__":
    # container_name = uuid4().hex
    # print(container_name)
    c = MongoStorageClient()
    c.save_data({"node_id": "test123", "X": 123})

    print(type(c.find_node("test123")))
