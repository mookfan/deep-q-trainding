import numpy as np
import pandas as pd


class Environment(object):
    def __init__(self, 
                data: pd.DataFrame, 
                min_info: dict,
                max_info: dict,
                ):
        '''
            "min, max" are using for Normalization.
        '''
        self.data = data
        self.min_info = min_info
        self.max_info = max_info
    
    def minmax_normalize(self, row):
        norm_data = []
        for key, value in zip(self.min_info, self.max_info):
            min_v = self.min_info[key]
            max_v = self.max_info[key]
            row[key] = (row[key] - min_v) / (max_v - min_v)
        return row

    def get_state(self, t: int, n: int):
        end = t + 1
        start = t - n + 1
        block = self.data.iloc[start: end].copy()
        # apply the custom function to each row of the DataFrame
        result = block.apply(self.minmax_normalize, axis=1)
        return result
        
