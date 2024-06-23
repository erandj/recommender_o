import logging
from pymongo import MongoClient
from ..parameters import mongoDB_url, mongoDB_db_name, mongoDB_collection_name, UserModelTemplate

client = MongoClient(mongoDB_url)
db = client[mongoDB_db_name]  # Database name
collection = db[mongoDB_collection_name]  # Collection name


def save_filepath_to_model(user_id: str, file_path: str):
    """
    Saves filepath to the user's model.
    """
    model_doc = UserModelTemplate(user_id, file_path)

    result = collection.update_one({"user_id": user_id}, {"$set": model_doc._asdict()}, upsert=True)


def retrieve_model(user_id: str):
    """
    Finds filepath to the user's model.

    Return
    -------------------------
    The document with filepath as UserModelTemplate class.
    """
    document = collection.find_one({"user_id": user_id})

    user = UserModelTemplate(document["user_id"], document["file_path"])

    if not user:
        logging.warning("The user's model does not exists.")
        return

    return user