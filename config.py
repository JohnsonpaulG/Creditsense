import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "creditsense-dev-secret-key")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "instance", "creditsense.db"))
    JSON_SORT_KEYS = False
