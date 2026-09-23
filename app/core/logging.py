import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        value = {"time": datetime.now(timezone.utc).isoformat(), "level": record.levelname,
                 "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            value["exception"] = self.formatException(record.exc_info)
        return json.dumps(value)


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
