import pandas as pd


class Environment(object):
    def __init__(self, 
                data: pd.DataFrame, 
                close_min: float,
                close_max: float,
                vol_min: float, 
                vol_max: float
                ):
        '''
            "close_min, close_max , vol_min, vol_max" are using for Normalization.
        '''
        self.data = data
        self.close_min = close_min
        self.close_max = close_max
        self.vol_min = vol_min
        self.vol_max = vol_max
    
    def minmax_normalize(self, row):
        price = row['close']
        vol = row['Volume']

        close_width = (self.close_max - self.close_min)
        vol_width = (self.vol_max - self.vol_min)
        
        price_norm = (price - self.close_min) / close_width
        vol_norm = (vol - self.vol_min) / vol_width
        return price_norm, vol_norm

    def get_state(self, t: int, n: int):
        end = t + 1
        start = t - n + 1
        block = self.data.iloc[start: end].copy()
        # apply the custom function to each row of the DataFrame
        result = block.apply(self.minmax_normalize, axis=1)
        # extract the individual values from the resulting DataFrame
        block.loc[:, 'close'] = result.apply(lambda x: x[0])
        block.loc[:, 'Volume'] = result.apply(lambda x: x[1])
        return block
        
