from azure.storage.blob import BlobServiceClient
from log_util import logger
import os
import json
from uuid import uuid4
from threading import Lock
from slack_alert import AlertSender


class AzureStorageClient:
    _default_container_name = uuid4().hex
    _instance = None
    _alert_sender = AlertSender()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        # print(f"get instance {cls._default_container_name}")
        return cls._instance

    def __init__(self):
        if self._instance is not None:
            raise Exception("This class is a singleton!")

        self.container_name = self._default_container_name
        self.lock = Lock()
        # AzureStorageClient._default_container_name = self.container_name

        self.connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        if self.connection_string is None:
            raise Exception("No connection string provided")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = None

        if self.check_container_exists():
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
        else:
            self.create_container()
            self.container_client = self.blob_service_client.get_container_client(self.container_name)

        self._instance = self

    def check_container_exists(self):
        """
        Check if a container exists in Azure Blob Storage
        """
        return self.blob_service_client.get_container_client(self.container_name).exists()

    def create_container(self):
        """
        Create a container in Azure Blob Storage
        """
        self.blob_service_client.create_container(self.container_name)
        logger.info(f"Created Azure Storage container {self.container_name}")
        self._alert_sender.send_alert(f"Created Azure Storage container {self.container_name}")

    def upload_file(self, file_name, json_string):
        """
        Upload a json string to Azure Blob Storage as json file
        """
        with self.lock:
            blob_client = self.container_client.get_blob_client(file_name)
            blob_client.upload_blob(json_string, overwrite=True)
            return True
        # with open(f"nodes/{file_name}", "w") as f:
        #     f.write(json_string)

    def download_data(self, file_name):
        """
        Download a json file from Azure Blob Storage
        """
        while self.lock:
            blob_client = self.container_client.get_blob_client(file_name)
            data = blob_client.download_blob().readall()
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        # with open(f"nodes/{file_name}", "r") as f:
        #     return json.load(f)


if __name__ == "__main__":
    # container_name = uuid4().hex
    # print(container_name)
    azure_storage_client = AzureStorageClient()
    azure_storage_client.upload_file(json_string=json.dumps({"test": "test123"}), file_name="test.json")

    print(azure_storage_client.download_data("test.json"))
