from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import FileResponse

from src.ai.ai_model import RecommendationEngine
from src.db import mongo_connector
from .json_parser import parse_from_model
from .model import TrainDataModel
from ..parameters import root_filepath
from ..parameters import root_filepath

import uuid

router = APIRouter()


@router.post("/create_engine", status_code=201)
async def create_engine(model: TrainDataModel, user_id: str | None = None):
    """
    Creates the recommender model. 
    """
    if model is None:
        raise HTTPException(400, detail="Request body is null.")

    df, target_df = parse_from_model(model)
    if df is None or target_df is None:
        raise HTTPException(400, detail="Can't parse the data.")

    engine = RecommendationEngine()
    engine.create(df, target_df)

    if user_id is None:
        user_id = str(uuid.uuid4())

    file_path = f"{root_filepath}\\user_engines\\{user_id}\\"
    file_path = f"{root_filepath}\\user_engines\\{user_id}\\"
    engine.save_model(file_path)
    user = mongo_connector.save_filepath_to_model(user_id, file_path)

    return {"id": user_id}


@router.post("/get_model")
async def get_model(user_id: str):
    user = mongo_connector.retrieve_model(user_id)
    
    return FileResponse(user.file_path+'model.pkl', 
                        media_type="application/octet-stream", 
                        filename="model.pkl")
