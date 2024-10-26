import numpy as np
import pandas as pd
from inputs import config_
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces



class EnvSetup:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        df
        feature_number : str
            start date of the data (modified from config_.py)
        use_technical_indicator : str
            end date of the data (modified from config_.py)
        use_turbulence : list
            a list of stock tickers (modified from config_.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """
    
    def __init__(self, 
                 stocks_dim:int,
                 state_space:int,
                 working_path:object,
                 hmax:float,
                 initial_amount:int,
                 transaction_cost_pct = 0.001,
                 reward_scaling = 1e-4):

        self.initial_amount = config_.initial_amount
        self.hmax = self.initial_amount/10000*config_.risk_level
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = config_.TECHNICAL_INDICATORS_LIST
        self.state_space = state_space
        self.stocks_dim = stocks_dim
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.stocks_dim,))  # normalized between 1 and -1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.state_space,))
        self.working_path = working_path
        self.memory_reset = config_.memory_reset

        
        

    def create_environment(self, data, env_class, model_name, initial_close = [], turbulence_threshold = 999999):
        
        environment = DummyVecEnv([lambda: env_class(df = data,
                                                    hmax = self.hmax,
                                                    initial_amount = self.initial_amount,
                                                    transaction_cost_pct = self.transaction_cost_pct,
                                                    reward_scaling = self.reward_scaling,
                                                    state_space = self.state_space,
                                                    stocks_dim = self.stocks_dim,
                                                    action_space = self.action_space,
                                                    observation_space = self.observation_space,
                                                    tech_indicator_list = self.tech_indicator_list,
                                                    initial_close = initial_close,
                                                    turbulence_threshold = turbulence_threshold,
                                                    model_name = model_name,
                                                    working_path = self.working_path,
                                                    memory_reset = self.memory_reset)])
        
        init_obs = environment.reset()
        print('init_obs reset executed')
        
        return environment, init_obs
