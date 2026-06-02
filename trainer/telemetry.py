import json
import logging
import socket
import time

import requests

from trainer.constants import VECTOR_URL


HOSTNAME = socket.gethostname()

class VectorHandler(logging.Handler):
    def __init__(self, url):
        super().__init__()
        self.url = url

    def emit(self, record):
        try:
            log_entry = {
                "message": record.getMessage(),
                "level": record.levelname,
                "logger": record.name,
                "timestamp": int(time.time() * 1000),
                "server": HOSTNAME,
            }

            for key, value in record.__dict__.items():
                if key not in log_entry and not key.startswith("_"):
                    try:
                        json.dumps(value) 
                        log_entry[key] = value
                    except Exception:
                        log_entry[key] = str(value)
            requests.post(self.url, json=log_entry, timeout=0.5)
        except Exception:
            self.handleError(record)

def setup_logger():
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.addHandler(VectorHandler(VECTOR_URL))
    return logger

logger = setup_logger()
