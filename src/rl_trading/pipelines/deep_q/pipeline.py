"""
This is a boilerplate pipeline 'deep_q'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_information, print_data, train, plot_train_result

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=get_information,
                inputs=['train_df'],
                outputs="train_info",
                name="GET_TRAIN_DATA_INFO",
            ),
        node(
                func=get_information,
                inputs=['test_df'],
                outputs="test_info",
                name="GET_TEST_DATA_INFO",
            ),  
        node(
                func=print_data,
                inputs=['train_df', 'test_df'],
                outputs=None,
                name="PRINT_DATA",
            ),
        node(
                func=train,
                inputs=['train_info', 'params:model_params'],
                outputs="result_data",
                name="TRAIN_RESULT",
            ),
        node(
                func=plot_train_result,
                inputs=['result_data'],
                outputs="result_plot",
                name="PLOT_RESULT",
            ),
    ])
