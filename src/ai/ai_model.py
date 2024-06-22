import logging
import os
import re
import numpy as np
import pandas as pd
import pickle

from pandas import DataFrame, Series
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

from typing import List, Optional, Union

from sklearn.model_selection import train_test_split
from ..data_types import str_categ
from ..parameters import params_random_tree, test_random_state


class DataProcessor:
    """
    The class to perform data processing.
    """

    @staticmethod
    def _remove_emoji(string: str) -> str:
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)

        return emoji_pattern.sub(r'', string)

    @staticmethod
    def _remove_urls(text: str) -> str:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')

        return url_pattern.sub(r'', text)

    @staticmethod
    def _perform_tfidf(column: Series) -> DataFrame:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(column)
        column_names = [i + "_tfidf" for i in vectorizer.get_feature_names_out()]

        return_df = DataFrame(
            matrix.toarray(),
            columns=column_names,
            index=column.index)
        return return_df

    @staticmethod
    def _perform_ohe(column: Series) -> DataFrame:
        """
        Performs one-hot encoding to pandas Series. 

        Parameters
        ----------
        column : Series
            The pandas Series of strings.
        type : StringTypeEnum
            The Enum of string types for plain text and categorical text.

        Returns
        -------
        Series
            The pandas DataFrame of binary values.
        """
        column_unique: List[object] = column.explode().unique().tolist()
        column_names: List[str] = list(map(lambda x: column.name + "_" + str(x) + "_ohe", column_unique))

        category_df = DataFrame(
            np.zeros(shape=(column.shape[0], len(column_names))),
            columns=column_names,
            index=column.index,
            dtype=int)

        for row_i in range(len(column)):
            category_list: List[str] = column.loc[row_i]

            for category_i in range(len(category_list)):
                category: str = category_list[category_i]
                category = column.name + "_" + str(category) + "_ohe"
                category_df.loc[row_i, category] = 1

        return category_df

    @staticmethod
    def _preprocess_text(column: Series) -> Series:
        """
        Performs preprocessing to given string Series. Lowers string and removes punctuation, emoji and urls.

        Parameters
        ----------
        column : Series
            The pandas Series of strings.

        Returns
        -------
        Series
            Preprocessed pandas Series of string.
        """
        if not pd.api.types.is_string_dtype(column.dtype):
            logging.error("Unexpected dtype of Series", exc_info=True)
            raise TypeError('Unexpected dtype of Series')

        if isinstance(column.loc[0], list):
            column = column.map(lambda x: [i.lower() for i in x])
            column = column.map(lambda x: [re.sub(r'[^\w\s]', '', i) for i in x])  # punctuation removal
            column = column.map(lambda x: [DataProcessor._remove_emoji(i) for i in x])
            column = column.map(lambda x: [DataProcessor._remove_urls(i) for i in x])
        else:
            column = column.map(lambda x: x.lower())
            column = column.map(lambda x: re.sub(r'[^\w\s]', '', x))  # punctuation removal
            column = column.map(lambda x: DataProcessor._remove_emoji(x))
            column = column.map(lambda x: DataProcessor._remove_urls(x))

        return column

    @staticmethod
    def _handle_na(column: Series, interpolate=False, drop=True) -> Series:
        """
        Dealing with missing values. 

        Parameters
        ----------
        column : Series
            The pandas Series of strings.
        interpolate : bool
            If true performs interpolation.
        drop : bool
            If true drops rows with missing values.

        Returns
        -------
        Series
            Preprocessed pandas Series.
        """
        if interpolate:
            column = column.interpolate(method='linear')
        if drop:
            column = column.dropna()

        return column

    @staticmethod
    def normalize(arr, t_min=0, t_max=1):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    @staticmethod
    def text_processing(column: Series) -> DataFrame:
        """
        Performs text vectorization for plain text and categorical text.

        Parameters
        ----------
        column : Series
            The pandas Series of strings.
        type : StringTypeEnum
            The Enum of string types for plain text and categorical text.

        Returns
        -------
        DataFrame
            The pandas DataFrame with binary or float value.
        """
        if not pd.api.types.is_string_dtype(column.dtype):
            logging.error("Unexpected dtype of Series", exc_info=True)
            raise TypeError('Unexpected dtype of Series')

        column = DataProcessor._handle_na(column)
        column = DataProcessor._preprocess_text(column)

        return_df: DataFrame
        if column.apply(type).eq(str_categ).all() or column.apply(type).eq(list).all():
            return_df = DataProcessor._perform_ohe(column)
        elif column.apply(type).eq(str).all():
            return_df = DataProcessor._perform_tfidf(column)
        else:
            logging.error("Unexpected type of string", exc_info=True)
            raise TypeError('Unexpected type of string')

        return return_df

    @staticmethod
    def bool_processing(column: Series) -> Series:
        """
        Converts boolean to binary values.

        Parameters
        ----------
        column : Series
            The pandas Series of booleans.

        Returns
        -------
        Series
            The pandas Series with binary values.
        """
        try:
            column = DataProcessor._handle_na(column)

            if pd.api.types.is_integer_dtype(column.dtype):
                if not all(x in [0, 1] for x in column):
                    raise TypeError
            elif pd.api.types.is_bool_dtype(column.dtype):
                column = column.apply(lambda x: 1 if x else 0)
            else:
                raise TypeError

            return column
        except TypeError:
            logging.error("Unexpected dtype of Series", exc_info=True)
            raise TypeError('Unexpected dtype of Series')

    @staticmethod
    def datetime_processing(column: Series) -> DataFrame:
        """
        Converts datetime to year, month and day.

        Parameters
        ----------
        column : Series
            The pandas Series of datetime.

        Returns
        -------
        DataFrame
            The pandas DataFrame with year, month and day columns.
        """
        if not pd.api.types.is_datetime64_any_dtype(column.dtype):
            logging.error("Unexpected dtype of Series", exc_info=True)
            raise TypeError('Unexpected dtype of Series')

        column = DataProcessor._handle_na(column, interpolate=True)

        return_df = DataFrame()
        return_df[column.name + '_year'] = column.dt.year
        return_df[column.name + '_month'] = column.dt.month
        return_df[column.name + '_day'] = column.dt.day

        return return_df

    @staticmethod
    def other_processing(column: Series) -> Series:
        """
        Deals with other type Series.

        Parameters
        ----------
        column : Series
            The pandas Series.

        Returns
        -------
        Series
            The processed pandas Series.
        """
        column = DataProcessor._handle_na(column, interpolate=True)

        return column


class RecommendationEngine:
    """
    The class that handles all functions of recommendation engine.
    """

    def __init__(self) -> None:
        self.reg_random_forest_model = RandomForestRegressor(**params_random_tree)

    def create(self, df: DataFrame, target_df: DataFrame) -> None:
        """
        Performs model creation. Preprocesses given data, then trains engine and saves it as file.

        Parameters
        ----------
        df : DataFrame
            The pandas DataFrame.
        target_df : DataFrame
            The pandas DataFrame.
        """
        for dtype in target_df.dtypes:
            if not pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_float_dtype(
                    dtype) and not pd.api.types.is_bool_dtype(dtype):
                logging.warn("The program can't handle non-numeric data type on target feature. Try again.")
                return

        # if new_df.shape[0] < 20 or new_target_df.shape[0] < 20:
        #     logging.warn("The data is too small. Please, use 'update' to provide new data to engine.")
        #     return

        df, target_df = self.preprocess_data(df, target_df)

        self.train(df, target_df)

    @staticmethod
    def _preprocess_data(df: DataFrame):
        for col_name in df:
            col = df[col_name]
            col_type = col.dtype

            preprocessed_data: Union[DataFrame, Series]
            if pd.api.types.is_string_dtype(col_type):
                preprocessed_data = DataProcessor.text_processing(col)
            elif pd.api.types.is_bool_dtype(col_type):
                preprocessed_data = DataProcessor.bool_processing(col)
            elif pd.api.types.is_datetime64_any_dtype(col_type):
                preprocessed_data = DataProcessor.datetime_processing(col)
            else:
                preprocessed_data = DataProcessor.other_processing(col)

            df = df.drop(columns=[col_name])
            df = df.merge(preprocessed_data, how="outer", left_index=True, right_index=True,
                          suffixes=('', '_DROP')).filter(regex="^(?!.*DROP)")

        return df

    def preprocess_data(self, df: DataFrame, target_df: DataFrame):
        """
        Performs data preprocessing.

        Parameters
        ----------
        df : DataFrame
            The pandas DataFrame.
        target_df : DataFrame
            The pandas DataFrame.

        Returns
        -------
        df : DataFrame
            The preprocessed pandas DataFrame.
        target_df : DataFrame
            The preprocessed pandas DataFrame.
        """
        try:
            if df is None or target_df is None or \
                    df.empty or target_df.empty:
                raise ValueError
        except (NameError, ValueError):
            logging.error(
                "It seems like you didn't provide any data. Please, use 'update' to provide new data to engine.")
            return

        try:
            df = self._preprocess_data(df)
            target_df = self._preprocess_data(target_df)
        except TypeError:
            logging.error(f"The preprocess has stopped, no data was saved. A data type is incorrect.")
            return

        return df, target_df

    def train(self, df: DataFrame, target_df: DataFrame):
        """
        Performs model training.

        Parameters
        ----------
        df : DataFrame
            The pandas DataFrame.
        target_df : DataFrame
            The pandas DataFrame.

        Returns
        -------
        MAE : Float | ndarray
            Mean absolute error
        MSE : Float | ndarray
            Mean squared error
        RMSE : Float | ndarray
            Root of mean squared error
        """
        X_train, X_test, y_train, y_test = train_test_split(df, target_df, test_size=0.15,
                                                            random_state=test_random_state)

        self.reg_random_forest_model.fit(X_train, y_train)

        y_pred = self.reg_random_forest_model.predict(X_test)

        # if this 15% of test data turns out to be necessary uncomment it.
        # self.reg_random_forest_model.fit(df, target_df)

        mea = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        print('MAE:', mea)
        print('MSE:', mse)
        print('RMSE:', rmse)

        return mea, mse, rmse

    def save_model(self, file_path: str):
        print(file_path)
        os.makedirs(file_path, exist_ok=True)
        with open(file_path + 'model.pkl', 'wb') as f:
            pickle.dump(self.reg_random_forest_model, f)

    def load_model(self, file_path: str):
        with open(file_path + 'model.pkl', 'rb') as f:
            self.reg_random_forest_model = pickle.load(f)

