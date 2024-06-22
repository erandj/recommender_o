from pathlib import Path
import json
from pydantic import BaseModel, Field
from typing import Any, Dict, List
from ..data_types import str_categ


class _Feature(BaseModel):
    type: str
    name: str
    data: List[Any]

class TableModel(BaseModel):
    columns: List[_Feature]

class DataTableModel(TableModel):
    pass

class TargetDataTableModel(TableModel):
    pass


class TrainDataModel(BaseModel):
    data: DataTableModel
    target_data: TargetDataTableModel 
    # idk how to get it from json file. So... Yeah.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": {
                        "columns": [
                            {
                                "type": "str",
                                "name": "string",
                                "data": [
                                    "string"
                                ]
                            },
                            {
                                "type": "str_categ",
                                "name": "category string",
                                "data": [
                                    "string"
                                ]
                            },
                            {
                                "type": "int",
                                "name": "integer",
                                "data": [
                                    1
                                ]
                            },
                            {
                                "type": "float",
                                "name": "target_float",
                                "data": [
                                    0.1
                                ]
                            },
                            {
                                "type": "bool",
                                "name": "boolean",
                                "data": [
                                    True
                                ]
                            },
                            {
                                "type": "list",
                                "name": "list",
                                "data": [
                                    ["string"]
                                ]
                            },
                            {
                            "type": "date",
                            "name": "date",
                            "data": [
                                "2008-09-15"
                            ]
                        }
                        ]
                    },
                    "target_data": {
                        "columns": [
                            {
                                "type": "int",
                                "name": "target_integer",
                                "data": [
                                    1
                                ]
                            },
                            {
                                "type": "float",
                                "name": "target_float",
                                "data": [
                                    0.1
                                ]
                            },
                            {
                                "type": "bool",
                                "name": "target_boolean",
                                "data": [
                                    True
                                ]
                            }
                        ]
                    }
                }
            ]
        }
    }

    