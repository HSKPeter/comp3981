import requests
import json
import os


class AlertSender:
    def __init__(self):
        self.webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    def send(self, message):
        if self.webhook_url is not None:
            slack_payload = {"text": message}
            requests.post(self.webhook_url, data=json.dumps(slack_payload), headers={'Content-Type': 'application/json'})


if __name__ == "__main__":
    alert_sender = AlertSender()
    alert_sender.send("Hello, World!")