import pandas as pd
from pandas import DataFrame, Series
from src.api.model import TableModel, TrainDataModel
from src.data_types import StrCategArray, StrCategDtype, str_categ


def model_to_df(table: TableModel) -> DataFrame:
    df: DataFrame = DataFrame([])

    columns = table.columns
    for i in columns:
        dtype = None
        data = i.data
        if i.type == "str_categ":
            dtype = StrCategDtype
            data = [None if str_data is None or str_data == '' else str_categ(str_data) for str_data in data]
        elif i.type == "list":
            new_data = []
            for list_data in data:
                new_list = []
                for str_data in list_data:
                    if str_data is None or str_data == '':
                        new_list.append(None)
                    else:
                        new_list.append(str_categ(str_data))
                new_data.append(new_list)
            data = new_data
        elif i.type != "date":
            dtype: str = i.type
            
        series = Series(data, name=i.name, dtype=dtype)

        if i.type == "date":
            series = pd.to_datetime(series) 
        df = pd.merge(df, series, how='outer', left_index=True, right_index=True)
    
    return df

def parse_from_model(model: TrainDataModel) -> tuple[DataFrame, DataFrame]:
    df: DataFrame = model_to_df(model.data)
    df_target: DataFrame = model_to_df(model.target_data)

    return df, df_target
    
