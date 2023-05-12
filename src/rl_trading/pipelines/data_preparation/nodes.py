"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.18.8
"""
from kedro.pipeline import Pipeline, node, pipeline
from kedro.io import *
from kedro.runner import *

import pandas as pd
import pprint

from .nodes import * # your node functions

def set_dataframe_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
	return df.set_index(index_name)

def train_test_split(df: pd.DataFrame, input_params: dict) -> pd.DataFrame:
    df = df.dropna()
    print("\n========= Data Information ==========")
    print(df.info())
    print("\n")
    columns = input_params["col"]
    train_ratio = input_params["train_ratio"]
    data = df[columns]
    train_num = int(train_ratio*len(df))
    train_data = data.iloc[:train_num]
    test_data = data.iloc[train_num:]
    return train_data, test_data

