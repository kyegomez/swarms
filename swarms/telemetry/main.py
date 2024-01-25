import logging
import pymongo
import platform
import datetime


class Telemetry:
    def __init__(self, db_url, db_name):
        self.logger = self.setup_logging()
        self.db = self.setup_db(db_url, db_name)

    def setup_logging(self):
        logger = logging.getLogger("telemetry")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        return logger

    def setup_db(self, db_url, db_name):
        client = pymongo.MongoClient(db_url)
        return client[db_name]

    def capture_device_data(self):
        data = {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "time": datetime.datetime.now(),
        }
        return data

    def send_to_db(self, collection_name, data):
        collection = self.db[collection_name]
        collection.insert_one(data)

    def log_and_capture(self, message, level, collection_name):
        if level == "info":
            self.logger.info(message)
        elif level == "error":
            self.logger.error(message)
        data = self.capture_device_data()
        data["log"] = message
        self.send_to_db(collection_name, data)

    def log_import(self, module_name):
        self.logger.info(f"Importing module {module_name}")
        module = __import__(module_name, fromlist=["*"])
        for k in dir(module):
            if not k.startswith("__"):
                self.logger.info(f"Imported {k} from {module_name}")
