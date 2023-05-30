"""
This is a boilerplate pipeline 'deep_q'
generated using Kedro 0.18.8
"""
from kedro.pipeline import Pipeline, node, pipeline
from kedro.io import *
from kedro.runner import *

from copy import deepcopy
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from .nodes import * # your node functions
from .environment import Environment
from .agent import Agent

import vectorbtpro as vbt

class Port:
    def __init__(
            self,
            cash :float
    ) -> None:
        self.port = vbt.pf_enums.ExecState(
            cash=cash,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=cash,
            val_price=0.0,
            value=0.0
        )

    def get_current_cash(self) -> float:
        return self.port.free_cash
    
    def get_current_position(self) -> float:
        return self.port.position
    
    def create_order(
        self,
        size: float,
        price: float,
        direction: float,
        fees : float
    ) -> vbt.portfolio.enums.Order:

        order = vbt.pf_nb.order_nb(
            size=size,
            price=price,
            direction=direction,
            size_granularity=1.0,
            log=True,
            fees=fees
        )
        return order

    def process_order(self,order) -> None:
        order_result, self.port = vbt.pf_nb.process_order_nb(
            0, 0, 0,
            exec_state=self.port,
            order=order,
            update_value=True,
        )
        

    def open_long(self,price : float,size : float,fees :float) -> None:
        order = self.create_order(
            size=size,
            price=price,
            direction=0,
            fees=fees
        )
        self.process_order(order)

    def open_short(self,price : float,size : float,fees :float) -> None:
        order = self.create_order(
            size=size,
            price=price,
            direction=1,
            fees=fees
        )
        self.process_order(order)

    def close_long(self,price : float,fees :float) -> None:
        order = self.create_order(
            size= -self.get_current_position(),
            price=price,
            direction=0,
            fees=fees
        )
        self.process_order(order)

    def close_short(self,price : float,fees :float) -> None:
        order = self.create_order(
            size=self.get_current_position(),
            price=price,
            direction=1,
            fees=fees
        )
        self.process_order(order)
    
    def get_profit(self) -> float:
        return self.port.value




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
    commission = model_params["commission"]
    budget = model_params["budget"]

    lenght = len(train_data)
    start_ind = window_size - 1
    num_features = train_data.shape[1] # #columns

    best_total_profit = 0
    best_model = None
    best_episode = None
    l = lenght - 1
    
    all_profit = []

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
                "buy_num": [],
                "sell_num": []
                # "best_model": best_model
            }

    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = env.get_state(t=window_size-1, n=window_size)
        state = prepare_input_state(state, num=window_size*num_features)

        total_profit = 0
        buy_count = 0
        sell_count = 0 
        agent.myport = []
        buy_num = 0
        sell_num = 0
        buy_list = []

        port = Port(float(budget))

        print(port.port)
        print("Total Profit: " + formatPrice(port.get_profit()))

        # print(f"Portfolio: $0.00 | Vol: {len(buy_list)} | Total: {formatPrice(sum(buy_list))}")
        # print(f"Cash Balance: {formatPrice(cash_balance)}\n")

        for t in range(start_ind, l):
            action = agent.act(state)
            close_price = float(train_data['close'].iloc[t])
            # sit
            next_state = env.get_state(t=t, n=window_size)
            reward = 0
            if action == 1 : # buy
                current_cash = port.get_current_cash()
                if current_cash > close_price:
                    buy_num += 1
                    print("----- Buy -----")
                    port.open_long(price=close_price,size=100.0,fees=commission)
                    buy_list.append(close_price)
                    print(f"price : {formatPrice(close_price)} ")
                    print(f"position : {port.port.position} ")
                    print(f"current cash : {formatPrice(port.get_current_cash())}")


                # print(f"close price : {formatPrice(close_price)} ")
                # profit = port.get_profit()
                # print(f"position : {port.port.position} ")
                # print(f"profit : {formatPrice(profit)} ")
                else:
                    pass

                
                # if cash_balance > close_price:
                #     buy_num += 1
                #     agent.myport.append(close_price)
                #     cash_balance = cash_balance - close_price
                #     buy_list.append(close_price)
                #     buy_count = buy_count + 1
                #     print(f"Buy: {formatPrice(close_price)} [{formatPrice(close_price)}] ")
                #     print(f"Portfolio:\n - AVG:{formatPrice(np.mean(np.array(buy_list)))}\n - Vol: {len(buy_list)}\n - Total: {formatPrice(sum(buy_list))}\n - Cash Balance: {formatPrice(cash_balance)}\n")
                
                # else:
                #     pass
                    # print(f"Buy: {formatPrice(close_price_com)} [{formatPrice(close_price)}]")
                    # print("Not enough money to buy \n")
                
            elif action == 2 :
            # and len(agent.myport) > 0: # sell
                bought_price = np.mean(np.array(buy_list))
                if port.port.position == 0.0:
                    pass
                else:
                    reward = max((close_price - bought_price), 0)
                    print("----- Sell -----")
                    port.close_long(price=close_price,fees=0.01)
                    sell_num += 1
                    print(f"price : {formatPrice(close_price)} ")
                    print(f"position : {port.port.position} ")
                
                # print(f"profit : {formatPrice(profit)} ")
                # bought_price = agent.myport.pop(0)
                # bought_price = np.mean(np.array(buy_list)) # ! check
                # if buy_list == []:
                #     pass
                # # agent.myport = [] # ! check
                # else:
                #     reward = max((close_price - bought_price), 0)
                #     # total_profit += close_price - bought_price
                #     total_profit += close_price*len(buy_list) - bought_price*len(buy_list)
                #     commission_fee = close_price*len(buy_list)*commission
                #     sell = close_price+commission_fee
                #     profit = sell*len(buy_list)-sum(buy_list) 
                #     cash_balance = cash_balance + sell
                #     sell_count = sell_count + 1
                #     sell_num += 1
                #     print(f"Sell: {formatPrice(sell*len(buy_list))} [{formatPrice(close_price*len(buy_list))}]| Profit: {formatPrice(profit)}")
                #     buy_list = []
                #     print(f"Portfolio:\n - AVG:{formatPrice(sum(buy_list))}\n - Vol: {len(buy_list)}\n - Total:{formatPrice(sum(buy_list))}\n - Cash Balance: {formatPrice(cash_balance)}\n ")
                    

                    

            done = True if t == l - 1 else False
            next_state = prepare_input_state(next_state, num=window_size*num_features)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                all_profit.append(formatPrice(total_profit))
                print("--------------------------------")
                print("Total Profit: " + formatPrice(port.get_profit()))
                print(f"Buy Count: {buy_num}")
                print(f"Sell Count: {sell_num}")
                print("--------------------------------")
                result["episode"].append(e)
                result["total_profit"].append(port.get_profit())
                result["buy_num"].append(buy_num)
                result["sell_num"].append(sell_num)

            if len(agent.memory) > batch_size:
                agent.exp_replay(batch_size)

                
        if best_total_profit < port.get_profit():
            print('best model is '+ str(e) + '\n')
            best_total_profit = port.get_profit()
            best_model = deepcopy(agent._model)
            best_episode = e
    result["best_episode"] = [best_episode]
    result["best_total_profit"] = [best_total_profit]
    return result
    
def plot_train_result(result_data: dict) -> None:
    print(result_data)
    episode = result_data["episode"]
    total_profit = result_data["total_profit"]
    best_episode = result_data["best_episode"]
    best_total_profit = result_data["best_total_profit"]
    buy_num = result_data["buy_num"]
    sell_num = result_data["sell_num"]
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
    fig.add_trace(go.Scatter(
            x=episode, 
            y=buy_num, 
            mode='lines+markers', 
            name='Buy Num'
        ))
    fig.add_trace(go.Scatter(
            x=episode,
            y=sell_num,
            mode='lines+markers',
            name='Sell Num'
        ))
    fig.update_layout(
        title="Total Profit per Episode",
        xaxis_title="Episode",
        yaxis_title="Total Profit",
    )
    return fig

def save_result_to_metrics(result_data: dict) -> dict:
    metrics = {
                "total_profit": result_data["best_total_profit"][0],
                "buy_num": result_data["buy_num"][result_data["best_episode"][0]],
                "sell_num": result_data["sell_num"][result_data["best_episode"][0]]
            }
    return metrics
