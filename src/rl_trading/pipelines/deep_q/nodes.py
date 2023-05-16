"""
This is a boilerplate pipeline 'deep_q'
generated using Kedro 0.18.8
"""
from kedro.pipeline import Pipeline, node, pipeline
from kedro.io import *
from kedro.runner import *

from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from .nodes import * # your node functions
from .environment import Environment
from .agent import Agent

def get_information(data: pd.DataFrame) -> dict:
    data_info = {
                "max": data.max(),
                "min": data.min(),
                "df": data
    }
    return data_info

def print_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n========= Train Data Summary ==========")
    # pprint.pprint(train)
    print(train_df.describe())
    print("\n========= Test Data Summary ==========")
    # pprint.pprint(test)
    print(test_df.describe())
    print("\n")

def formatPrice(n) -> None:
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def df_to_tensor(df) -> torch.tensor:
	return torch.tensor(df.values).float()

def tensor_2d_to_3d(tensor) -> torch.tensor:
	'''
		input [n x m]
		output [1 x n x m]
	'''
	return tensor.unsqueeze(0)

def tensor_3d_to_1d(tensor: torch.tensor, num: int) -> torch.tensor:
	return tensor.reshape(-1, num)

def prepare_input_state(array: np.array, num: int) -> np.array:
	state_2d = df_to_tensor(array)
	state_3d = tensor_2d_to_3d(state_2d)
	state_1d = tensor_3d_to_1d(state_3d, num)
	return state_1d

def train(data_profile: dict, model_params: dict) -> None:
    train_data = data_profile["df"]
    min_values = {index: value for index, value in data_profile["min"].items()}
    max_values = {index: value for index, value in data_profile["max"].items()}
    window_size = model_params["window_size"]
    batch_size = model_params["batch_size"]
    episode_count = model_params["episode"]

    lenght = len(train_data)
    start_ind = window_size - 1
    num_features = train_data.shape[1] # #columns

    best_total_profit = 0
    best_model = None
    best_episode = None
    l = lenght - 1

    env = Environment(
                    data=train_data,
                    min_info=min_values,
                    max_info=max_values 
                )
    
    agent = Agent(
                    state_size=window_size, 
                    feature_size=num_features,
                    batch_size=batch_size
                )
    result = {
                "episode": [], 
                "total_profit": [],
                "best_episode": best_episode,
                "best_total_profit": best_total_profit,
                # "best_model": best_model
            }

    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = env.get_state(t=window_size-1, n=window_size)
        state = prepare_input_state(state, num=window_size*num_features)

        total_profit = 0
        agent.myport = []

        for t in range(window_size-1, l):
            action = agent.act(state)

            # sit
            next_state = env.get_state(t=t, n=window_size)
            reward = 0
            
            if action == 1 : # buy 
                agent.myport.append(train_data['close'].iloc[t])
                print("Buy: " + formatPrice(train_data['close'].iloc[t]))
            
            elif action == 2 and len(agent.myport) > 0: # sell
                # bought_price = agent.myport.pop(0)
                bought_price = np.mean(np.array(agent.myport)) # ! check
                agent.myport = [] # ! check
                reward = max(train_data['close'].iloc[t] - bought_price, 0)
                total_profit += train_data['close'].iloc[t] - bought_price
                print("Sell: " + formatPrice(train_data['close'].iloc[t]) + " | Profit: " + formatPrice(train_data['close'].iloc[t] - bought_price))

            done = True if t == l - 1 else False
            next_state = prepare_input_state(next_state, num=window_size*num_features)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")
                result["episode"].append(e)
                result["total_profit"].append(total_profit)

            if len(agent.memory) > batch_size:
                agent.exp_replay(batch_size)
        if best_total_profit < total_profit:
            print('best model is '+ str(e) + '\n')
            best_total_profit = total_profit
            best_model = deepcopy(agent._model)
            best_episode = e
    result["best_episode"] = [best_episode]
    result["best_total_profit"] = [best_total_profit]
    # result["best_model"] = best_model
    return result
    
def plot_train_result(result_data: dict) -> None:
    print(result_data)
    episode = result_data["episode"]
    total_profit = result_data["total_profit"]
    best_episode = result_data["best_episode"]
    best_total_profit = result_data["best_total_profit"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=episode, 
            y=total_profit, 
            mode='lines+markers', 
            name='Total Profit'
        ))
    fig.add_trace(go.Scatter(
            x=best_episode, 
            y=best_total_profit, 
            mode='markers', 
            name='Best Total Profit'
        ))
    fig.update_layout(
        title="Total Profit per Episode",
        xaxis_title="Episode",
        yaxis_title="Total Profit",
    )
    return fig
