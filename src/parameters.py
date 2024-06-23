# RandomForest
import os
from typing import List, NamedTuple

test_random_state = 10
params_random_tree = {
    'bootstrap': True,
    'max_features': 'sqrt',
    'min_samples_split': 4,
    'n_estimators': 20
}

# MongoDB
mongoDB_url = "mongodb://model_mongodb:27017"
mongoDB_db_name = "recommenderDB"
mongoDB_collection_name = "user_models"

class UserModelTemplate(NamedTuple):
    title: str
    description: str
    user_id: int
    file_size: str
    file_path: str
    created_At: str

# Other
root_filepath = os.path.dirname(os.path.abspath(__file__))
