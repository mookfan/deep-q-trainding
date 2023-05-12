"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import set_dataframe_index, train_test_split

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=set_dataframe_index,
                inputs=['RAW_ICHI', 'params:index_name'],
                outputs="ichi",
                name="DEEP_Q_LEARNING",
            ),
        node(
                func=train_test_split,
                inputs=['ichi', 'params:input_params'],
                outputs=["train_df", "test_df"],
                name="SPILT_DATA",
            ),
    ])

