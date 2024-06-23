import logging
from pymongo import MongoClient
from ..parameters import mongoDB_url, mongoDB_db_name, mongoDB_collection_name, UserModelTemplate

client = MongoClient(mongoDB_url)
db = client[mongoDB_db_name]  # Database name
collection = db[mongoDB_collection_name]  # Collection name


def save_model(title: str, description: str, user_id: int, file_size: str, file_path: str, created_At: str):
    """
    Сохраняет модель пользователя в MongoDB.
    """
    model_doc = {
        "title": title,
        "description": description,
        "user_id": user_id,
        "file_size": file_size,
        "file_path": file_path,
        "created_At": created_At
    }

    result = collection.update_one({"user_id": user_id}, {"$set": model_doc}, upsert=True)
    logging.info(f"Model for user_id {user_id} saved successfully.")
    return result

def retrieve_model(user_id: str):
    """
    Находит модель пользователя по `user_id`.

    Return
    -------------------------
    Возвращает документ с моделью в качестве класса UserModelTemplate.
    """
    document = collection.find_one({"user_id": user_id})

    if not document:
        return None

    # Проверяем наличие всех необходимых полей
    title = document.get("title", "Default Title")
    description = document.get("description", "Default Description")
    user_id = document.get("user_id", "")
    file_size = document.get("file_size", "")
    file_path = document.get("file_path", "")
    created_At = document.get("created_At", "")

    return UserModelTemplate(title, description, user_id, file_size, file_path, created_At)

def get_all_models(user_id: str):
    """
    Получает список всех моделей из MongoDB.
    """
    models = []
    if user_id:
        cursor = collection.find({"user_id": user_id})
    else:
        cursor = collection.find({})

    for document in cursor:
        model = {
            "title": document.get("title", "Default Title"),
            "description": document.get("description", "Default Description"),
            "user_id": document.get("user_id", ""),
            "file_size": document.get("file_size", ""),
            "file_path": document.get("file_path", ""),
            "created_At": document.get("created_At", "")
        }
        models.append(model)

    return models

def delete_model(user_id: int):
    """
    Удаляет модель пользователя по user_id.
    """
    result = collection.delete_one({"user_id": user_id})
    if result.deleted_count > 0:
        logging.info(f"Model for user_id {user_id} deleted successfully.")
        return True
    else:
        logging.warning(f"Model for user_id {user_id} not found.")
        return False