import requests
import json


class AlertSender:
    def __init__(self):
        self.webhook_url = ""

    def send(self, message):
        slack_payload = {"text": message}
        requests.post(self.webhook_url, data=json.dumps(slack_payload), headers={'Content-Type': 'application/json'})
