"""
This is a boilerplate pipeline 'deep_q'
generated using Kedro 0.18.8
"""
from kedro.pipeline import Pipeline, node, pipeline
from kedro.io import *
from kedro.runner import *

import os
import pandas as pd
import pickle

from .nodes import * # your node functions

def set_dataframe_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
	return df.set_index(index_name)
