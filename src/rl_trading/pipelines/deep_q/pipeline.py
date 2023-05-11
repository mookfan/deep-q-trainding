"""
This is a boilerplate pipeline 'deep_q'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import set_dataframe_index

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=set_dataframe_index,
                inputs=['RAW_ICHI', 'params:index_name'],
                outputs="ICHI",
                name="DEEP_Q_LEARNING",
            ),
        
    ])
