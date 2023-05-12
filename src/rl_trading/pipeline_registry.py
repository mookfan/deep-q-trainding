"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from rl_trading.pipelines import data_preparation as dp 
from rl_trading.pipelines import deep_q as dq

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_preparation_pipeline = dp.create_pipeline()
    deep_q_pipeline = data_preparation_pipeline + dq.create_pipeline()

    return {
        "data preparation": data_preparation_pipeline,
        "deep q-learning": deep_q_pipeline,
        "__default__": data_preparation_pipeline + deep_q_pipeline,
    }

    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    # return pipelines
