# RandomForest
import os
from typing import NamedTuple


test_random_state = 10
params_random_tree = {
    'bootstrap': True,
    'max_features': 'sqrt',
    'min_samples_split': 4,
    'n_estimators': 20
}

# MongoDB
mongoDB_url = "mongodb://localhost:27017"
mongoDB_db_name = "recommenderDB"
mongoDB_collection_name = "user_models"

class UserModelTemplate(NamedTuple):
    user_id: int
    file_path: str

# Other
root_filepath = os.path.dirname(os.path.abspath(__file__))
