import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
import matplotlib
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


"""A stock validation environment for OpenAI gym"""
class StockEnvTrain(gym.Env):
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
                initial_close,
                turbulence_threshold, 
                model_name, 
                working_path,
                memory_reset
                ):
        
        self.df = df
        self.stocks_dim = stocks_dim
        self.state_space = state_space
        self.action_space = action_space
        self.observation_space = observation_space
        self.initial = True
        self.initial_day = self.df.index[0]
        self.day = self.df.index[0]
        self.terminal = False  
        self.initial_amount = initial_amount
        self.tech_indicator_list = tech_indicator_list
        self.memory_reset = memory_reset
           
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
        self.hmax_init = hmax
        self.initial_close = initial_close
        self.steps_initial = 0
        self.steps_final = 0
        self.sharpe = 0
        self._seed()
        
#######################################################################################################################################################   
   

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index+self.stocks_dim+1] > 0:
            #update balance
            self.state[0] += self.state[index+1] * min(abs(action),self.state[index+self.stocks_dim+1]) * (1- self.transaction_cost_pct)
            self.state[index+self.stocks_dim+1] -= min(abs(action), self.state[index+self.stocks_dim+1])
            self.cost +=self.state[index+1] * min(abs(action),self.state[index+self.stocks_dim+1]) * self.transaction_cost_pct
            self.trades+=1
        else:
            pass

            
#######################################################################################################################################################   
   
    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        #update balance
        self.state[0] -= self.state[index+1] * min(available_amount, action) * (1+ self.transaction_cost_pct)
        self.state[index+self.stocks_dim+1] += min(available_amount, action)
        self.cost+=self.state[index+1] * min(available_amount, action)* self.transaction_cost_pct
        self.trades+=1
        
#######################################################################################################################################################   
   
        
    def step(self, actions):
        
        self.initial = False
        self.init_actions = actions
        self.terminal = self.day >= self.initial_day + len(self.df.index.unique())-1 # the reset function is automatically called when self.terminal == True
        self.previous_state = self.state        
        
        self.hmax = self.hmax_init * self.end_total_asset/self.initial_amount  
        
        # the initial actions is a vector of stocks_dim in the range [0,1]
        self.actions = actions * self.hmax # multiplies actions by maximum number of shares per trade        
  
        # calculates the total asset at the beginning of trading
        begin_total_asset = self.state[0] + \
                        sum(np.array(self.state[1:(self.stocks_dim+1)]) * np.array(self.state[(self.stocks_dim+1):(self.stocks_dim*2+1)])) 

        argsort_actions = np.argsort(self.actions) # sorts the actions to determine the sell and buy indices. The sorting of actions in descending order for buying and ascending order for selling is typically done to prioritize the actions with the highest absolute values
        sell_index = argsort_actions[:np.where(self.actions < 0)[0].shape[0]] #  most confident (lowest indexes, starting from -100) sales executed first
        buy_index = argsort_actions[::-1][:np.where(self.actions > 0)[0].shape[0]] #  most confident (highest indexes, starting from +100) purchases executed first ([::-1] reverts the indexes)

        for index in sell_index:
            self._sell_stock(index, self.actions[index])

        for index in buy_index:
            self._buy_stock(index, self.actions[index])       
        
        self.prev_data = self.df.loc[self.day,:] # to retrieve the previous close price
        
        # print('TRAIN day checked in step function', self.day, 'over', self.initial_day + len(self.df.index.unique())-1)
        
        if not self.terminal:
            self.day += 1 
            self.data = self.df.loc[self.day,:] # to retrieve the current tech indicators  
            self.steps_final +=1
                
        #load next state (after trading, once the sell and buy methods have been called and the self.state has been modified)
        self.state =  [self.state[0]] + \
                self.prev_data['Adj Close'].values.tolist() + \
                list(self.state[(self.stocks_dim+1):(self.stocks_dim*2+1)]) + \
                sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])     
                
        # calculates the total asset at the end of trading
        self.end_total_asset = self.state[0] + \
                sum(np.array(self.state[1:(self.stocks_dim+1)]) * np.array(self.state[(self.stocks_dim+1):(self.stocks_dim*2+1)]))
        self.asset_memory.append(self.end_total_asset) 

        # the reward is the total asset increase
        self.reward = self.end_total_asset - begin_total_asset            
        self.rewards_memory.append(self.reward)
        self.reward = self.reward*self.reward_scaling        
        
        if self.terminal: # If the trading period is over:
            print ('TRAIN STEP TERMINAL',self.model_name, 'over steps:', self.steps_initial, self.steps_final)
            plt.clf()
            plt.plot(self.asset_memory,'r')
            plt.savefig(self.working_path+'/account_value_train_{}_{}-{}.png'.format(self.model_name,self.steps_initial,self.steps_final))
            plt.close()
            
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            df_total_value.to_csv(self.working_path+'/account_value_train_{}_{}-{}.csv'.format(self.model_name,self.steps_initial,self.steps_final))

            self.sharpe = (252**0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std() 
            
            print('TRAIN initial asset', self.asset_memory[0])
            print('TRAIN end_total_asset', self.end_total_asset)
            print("TRAIN total_reward", self.end_total_asset - self.asset_memory[0], '==', np.sum(self.rewards_memory))
            print("TRAIN total_cost: ", self.cost)
            print("TRAIN total trades: ", self.trades)
            print("TRAIN Sharpe Ratio: ",self.sharpe, ',\n')
                        
            self.steps_initial = self.steps_final         
        
        return self.state, self.reward, self.terminal, {}

#######################################################################################################################################################   

    def reset(self):        
        if self.initial:    
            print('\n******************* reset initial TRAIN *******************')
            print('asset memory reset initial', self.asset_memory) ##########################################################33
            self.asset_memory = [self.initial_amount]
            self.prev_data = self.df.loc[self.initial_day,:]
            self.cost = 0
            self.trades = 0
            self.rewards_memory = []
            self.end_total_asset = self.initial_amount
                        
            self.state = [self.initial_amount] + \
                          self.initial_close + \
                          [0]*self.stocks_dim + \
                          sum([self.prev_data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            
        else:
            print('\n******************* reset NOT initial TRAIN *******************')  
            self.day = self.df.index[0]
            self.rewards_memory = []
            
            if self.memory_reset==True: # no memory
                self.asset_memory = [self.initial_amount]
                self.state = [self.initial_amount] + \
                              self.initial_close + \
                              [0]*self.stocks_dim + \
                              sum([self.prev_data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            else: #memory
                # simulating selling all stocks at the end of each training cycle
                self.asset_memory = [self.end_total_asset]
                self.state = [self.end_total_asset] + \
                              self.initial_close + \
                              [0]*self.stocks_dim + \
                              sum([self.prev_data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            
        return self.state
    
#######################################################################################################################################################   
    
    def render(self, mode='human',close=False):
        return self.state


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    