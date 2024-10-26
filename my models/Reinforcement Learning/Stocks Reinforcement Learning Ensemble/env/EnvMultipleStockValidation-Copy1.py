import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
import matplotlib
import matplotlib.pyplot as plt

import pickle

class StockEnvValidation(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                df,
                hmax,
                initial_amount, 
                transaction_cost_pct, 
                reward_scaling, 
                state_space,
                stocks_dim, 
                action_space,
                observation_space,
                tech_indicator_list, 
                initial_close_trade,
                turbulence_threshold, 
                model_name, 
                working_path):
        
        #super(StockEnv, self).__init__()
        self.df = df
        self.stocks_dim = stocks_dim
        self.state_space = state_space
        self.action_space = action_space
        self.observation_space = observation_space
        self.initial_day = self.df.index[0]
        self.day = self.df.index[0]
        self.data = self.df.loc[self.day,:]
        self.terminal = False  
        self.initial_amount = initial_amount
        self.tech_indicator_list = tech_indicator_list
        # initalize state
        self.state = [self.initial_amount] + \
                      self.data['Adj Close'].values.tolist() + \
                      [0]*self.stocks_dim + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        # initialize reward
        self.reward_scaling = reward_scaling
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.transaction_cost_pct =transaction_cost_pct
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.trades = 0
        self.model_name=model_name 
        self.working_path=working_path
        self.hmax = hmax
        self.turbulence = 0
        self.turbulence_threshold = turbulence_threshold
        self._seed()
        
#######################################################################################################################################################   

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence<self.turbulence_threshold:
            if self.state[index+self.stocks_dim+1] > 0:
                #update balance
                self.state[0] += self.state[index+1]*min(abs(action),self.state[index+self.stocks_dim+1]) * (1- self.transaction_cost_pct)
                self.state[index+self.stocks_dim+1] -= min(abs(action), self.state[index+self.stocks_dim+1])
                self.cost +=self.state[index+1]*min(abs(action),self.state[index+self.stocks_dim+1]) * self.transaction_cost_pct
                self.trades+=1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions 
            if self.state[index+self.stocks_dim+1] > 0:
                #update balance
                self.state[0] += self.state[index+1]*self.state[index+self.stocks_dim+1]* (1- self.transaction_cost_pct)
                self.state[index+self.stocks_dim+1] =0
                self.cost += self.state[index+1]*self.state[index+self.stocks_dim+1]* self.transaction_cost_pct
                self.trades+=1
            else:
                pass
            
#######################################################################################################################################################   
    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence< self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index+1]
            # print('available_amount:{}'.format(available_amount))
            #update balance
            self.state[0] -= self.state[index+1]*min(available_amount, action)* (1+ self.transaction_cost_pct)
            self.state[index+self.stocks_dim+1] += min(available_amount, action)
            self.cost+=self.state[index+1]*min(available_amount, action)* self.transaction_cost_pct
            self.trades+=1
        else:
            # if turbulence goes over threshold, just stop buying
            pass
        
#######################################################################################################################################################   
        
    def step(self, actions):

        self.terminal = self.day >= self.initial_day + len(self.df.index.unique())-1
        
        actions = actions * self.hmax
        #actions = (actions.astype(int))
        if self.turbulence>=self.turbulence_threshold:
            actions=np.array([-self.hmax]*self.stocks_dim)
        begin_total_asset = self.state[0]+ \
        sum(np.array(self.state[1:(self.stocks_dim+1)])*np.array(self.state[(self.stocks_dim+1):(self.stocks_dim*2+1)]))
        #print("begin_total_asset:{}".format(begin_total_asset))

        argsort_actions = np.argsort(actions)

        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            # print('take sell action'.format(actions[index]))
            self._sell_stock(index, actions[index])

        for index in buy_index:
            # print('take buy action: {}'.format(actions[index]))
            self._buy_stock(index, actions[index])

        self.data = self.df.loc[self.day,:]         
        self.turbulence = self.data['Turbulence'].values[0]
        #print(self.turbulence)
        #load next state
        self.state =  [self.state[0]] + \
                self.data['Adj Close'].values.tolist() + \
                list(self.state[(self.stocks_dim+1):(self.stocks_dim*2+1)]) + \
                sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])

        end_total_asset = self.state[0]+ \
        sum(np.array(self.state[1:(self.stocks_dim+1)])*np.array(self.state[(self.stocks_dim+1):(self.stocks_dim*2+1)]))
        self.asset_memory.append(end_total_asset)
        #print("end_total_asset:{}".format(end_total_asset))

        self.reward = end_total_asset - begin_total_asset            
        # print("step_reward:{}".format(self.reward))
        self.rewards_memory.append(self.reward)
        self.reward = self.reward*self.reward_scaling

        if not self.terminal:
            self.day += 1 
        
        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig(self.working_path+'/account_value_validation_{}.png'.format(self.model_name))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(self.working_path+'/account_value_validation_{}.csv'.format(self.model_name))
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (252**0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std() 
            print("Validation Sharpe Ratio: ",sharpe)

        return self.state, self.reward, self.terminal, {}
    
#######################################################################################################################################################   

    def reset(self):  
        self.asset_memory = [self.initial_amount]
        self.data = self.df.loc[self.initial_day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state      
        self.state = [self.initial_amount] + \
                      self.data['Adj Close'].values.tolist() + \
                      [0]*self.stocks_dim + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
                           
        return self.state
    
#######################################################################################################################################################   
    
    def render(self, mode='human',close=False):
        return self.state
    
#######################################################################################################################################################   

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]