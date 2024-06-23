import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import FileResponse

from src.ai.ai_model import RecommendationEngine
from src.db import mongo_connector
from .json_parser import parse_from_model
from .model import TrainDataModel
from ..parameters import root_filepath
from ..parameters import root_filepath
from fastapi.responses import StreamingResponse
from io import BytesIO
from zipfile import ZipFile
import uuid

router = APIRouter()

@router.get("/get_models_list")
async def get_models_list(user_id: str | None = None):
    """
    Получает список всех моделей в базе данных.
    """
    models_list = mongo_connector.get_all_models(user_id)

    if not models_list:
        raise HTTPException(status_code=404, detail="No models found")

    for model in models_list:
        file_name = model["file_path"].split("/")[-1]  # Получаем имя файла из пути
        model["file_path"] = f"/files/models/{model['user_id']}/{file_name}"  # Заменяем путь на URL

    return models_list

@router.post("/create_model", status_code=201)
async def create_model(model: TrainDataModel, title: str, description: str, created_At: str | None = None, user_id: str | None = None):
    """
    Создает модель пользователя.
    """
    if model is None:
        raise HTTPException(400, detail="Request body is null.")

    df, target_df = parse_from_model(model)
    if df is None or target_df is None:
        raise HTTPException(400, detail="Can't parse the data.")

    if user_id is None:
        user_id = str(uuid.uuid4())

    file_path = f"{root_filepath}/user_models/{user_id}/"

    # Создание модели и сохранение на диск
    engine = RecommendationEngine()
    engine.create(df, target_df)
    model_file_path = engine.save_model(file_path)

    # Получение размера файла
    file_size = os.path.getsize(file_path)

    # Сохранение модели в базе данных MongoDB
    result = mongo_connector.save_model(title, description, user_id, str(file_size), model_file_path, created_At)

    return {"id": user_id}



@router.get("/get_model/{user_id}")
async def get_model(user_id: str):
    """
    Получает модель пользователя по user_id.
    """
    user = mongo_connector.retrieve_model(user_id)

    if not user:
        raise HTTPException(404, "Model not found")

    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "w") as zip_file:
        for filename in os.listdir(user.file_path):
            file_path = os.path.join(user.file_path, filename)
            zip_file.write(file_path, arcname=filename)

    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=files.zip"})

@router.put("/update_model/{user_id}", status_code=200)
async def update_model(user_id: str, model: TrainDataModel, title: str, description: str, created_At: str):
    """
    Обновляет модель пользователя.
    """
    existing_model = mongo_connector.retrieve_model(user_id)

    if not existing_model:
        raise HTTPException(404, "Model not found")

    df, target_df = parse_from_model(model)
    if df is None or target_df is None:
        raise HTTPException(400, detail="Can't parse the data.")

    file_path = f"{root_filepath}/user_models/{user_id}/"

    result = mongo_connector.update_model(user_id, title, description, model.file_size, file_path, created_At)

    return {"message": "Model updated successfully"}

@router.delete("/delete_model/{user_id}", status_code=204)
async def delete_model(user_id: str):
    """
    Удаляет модель пользователя.
    """
    existing_model = mongo_connector.retrieve_model(user_id)

    if not existing_model:
        raise HTTPException(404, "Model not found")

    result = mongo_connector.delete_model(user_id)

    return {"message": "Model deleted successfully"}