from azure.storage.blob import BlobServiceClient
from log_util import logger
import json
from alert_util import AlertSender
import os


class AzureStorageClient:
    _alert_sender = AlertSender()

    def __init__(self, container_name):
        """
        Initialize an instance of AzureStorageClient
        """
        self.container_name = container_name

        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.connection_string = connection_string if connection_string is not None else input("Enter Azure Storage Connection String: ")

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
        self._alert_sender.send(f"Created Azure Storage container {self.container_name}")

    def upload_file(self, file_name, data):
        """
        Upload a json string to Azure Blob Storage as json file
        """
        blob_client = self.container_client.get_blob_client(file_name)
        blob_client.upload_blob(json.dumps(data), overwrite=True)

    def download_data(self, file_name):
        """
        Download a json file from Azure Blob Storage
        """
        blob_client = self.container_client.get_blob_client(file_name)
        data = blob_client.download_blob().readall()
        json_str = data.decode('utf-8')
        return json.loads(json_str)
