import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
import matplotlib
import matplotlib.pyplot as plt

import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


"""A stock trading environment for OpenAI gym"""
class StockEnvTrade(gym.Env):
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
        self.turbulence = 0
        self.turbulence_threshold = turbulence_threshold
        
        self.confusion_matrices = {i: [] for i in range(self.stocks_dim)} # initialize confusion matrices
        self.confusion_matrices_day = {i: [] for i in range(self.stocks_dim)} # initialize confusion matrices
        self.initial_close = initial_close
        self.sharpe = 0
        self._seed()
        
#######################################################################################################################################################   
   

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence<self.turbulence_threshold:
            if self.state[index+self.stocks_dim+1] > 0:
                #update balance
                self.state[0] += self.state[index+1] * min(abs(action),self.state[index+self.stocks_dim+1]) * (1- self.transaction_cost_pct)
                print ('--------------------------   SELL    ---------------------------')
                print ('asset (increment) after sell:', self.state[index+1] * min(abs(action),self.state[index+self.stocks_dim+1]) * (1- self.transaction_cost_pct))
                print('=')
                print('self.state[index+1]',self.state[index+1])
                print('*')
                print('min(abs(action),self.state[index+self.stocks_dim+1])',min(abs(action),self.state[index+self.stocks_dim+1]))
                print('*')
                print('(1- self.transaction_cost_pct)',(1- self.transaction_cost_pct))
                print('\nasset (without stock return) after sell:',self.state[0])
                print('------------------------------------------------------------------')
                self.state[index+self.stocks_dim+1] -= min(abs(action), self.state[index+self.stocks_dim+1])
                self.cost +=self.state[index+1] * min(abs(action),self.state[index+self.stocks_dim+1]) * self.transaction_cost_pct
                self.trades+=1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions 
            print('SELL TURB') #######
            if self.state[index+self.stocks_dim+1] > 0:
                #update balance
                self.state[0] += self.state[index+1] * self.state[index+self.stocks_dim+1] * (1- self.transaction_cost_pct)
                self.state[index+self.stocks_dim+1] =0
                self.cost += self.state[index+1] * self.state[index+self.stocks_dim+1] * self.transaction_cost_pct
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
            self.state[0] -= self.state[index+1] * min(available_amount, action) * (1+ self.transaction_cost_pct)
            print ('--------------------------   BUY    ---------------------------')
            print ('asset (decrement) after buy:', self.state[index+1] * min(available_amount, action) * (1+ self.transaction_cost_pct))
            print('=')
            print('self.state[index+1]',self.state[index+1])
            print('*')
            print('min(available_amount, action)',min(available_amount, action))
            print('*')
            print('(1+ self.transaction_cost_pct)',(1+ self.transaction_cost_pct))
            print('\nasset (without stock return) after buy:',self.state[0])
            print('------------------------------------------------------------------')
            self.state[index+self.stocks_dim+1] += min(available_amount, action)
            self.cost+=self.state[index+1] * min(available_amount, action)* self.transaction_cost_pct
            self.trades+=1
        else:
            print('BUY TURB') ########
            # if turbulence goes over threshold, just stop buying
            pass
        
#######################################################################################################################################################   
   
        
    '''
    The step function takes the predicted action, and outputs the next observation state (to be reused in the next prediction), the rewards, 'dones' if the trading is over, and optional additional info.
    The returned values are:
    self.state: The next state of the environment after the agent takes an action.
    self.reward: The reward the agent receives after taking an action.
    self.terminal: A boolean indicating whether the episode (e.g., a trading day) has ended.
    {}: An empty dictionary that is typically used for returning extra information that may be useful for debugging or logging purposes (just for consistency with the gym interface)
    '''
    def step(self, actions):
        
        self.initial = False
        self.init_actions = actions
        self.terminal = self.day >= self.initial_day + len(self.df.index.unique())-1 # the reset function is automatically called when self.terminal == True
        self.previous_state = self.state
        
        # print('\nself.initial_day',self.initial_day)
        print('\nself.day',self.day)
        
        self.hmax = self.hmax_init * self.end_total_asset/self.initial_amount  
        print('hmax:', self.hmax)

        # the initial actions is a vector of stocks_dim in the range [0,1]
        print('\ninitial actions',self.init_actions) 
        self.actions = actions * self.hmax # multiplies actions by maximum number of shares per trade
        
        self.turbulence = self.prev_data['Turbulence'].values[0]      
        
        if self.turbulence>=self.turbulence_threshold: #  checks if the turbulence is above a certain threshold (if so, it sells all stocks)
            self.actions = np.array([-self.hmax]*self.stocks_dim) # vector of stocks_dim with maximum negative number of shares (to sell)
        print('final actions',self.actions) 
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
        
        if not self.terminal:
            print('\n******************* step not terminal TRADE *******************')
            self.day += 1 ###########################################
            self.data = self.df.loc[self.day,:] # to retrieve the current tech indicators     
                        
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
            print('\n******************* step terminal TRADE *******************')    
            plt.plot(self.asset_memory,'r') # it plots the asset memory (the value of the portfolio over time)
            plt.savefig(self.working_path+'/account_value_trade_{}.png'.format(self.model_name)) # save the asset memory plot
            plt.close()
            
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            df_total_value.to_csv(self.working_path+'/account_value_trade_{}.csv'.format(self.model_name)) # save the asset memory as a csv file
            
            self.sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ df_total_value['daily_return'].std() 
            
            print('\nTRADE initial asset', self.asset_memory[0])
            print('TRADE end_total_asset', self.end_total_asset)
            print("TRADE total_reward", self.end_total_asset - self.asset_memory[0], '==', np.sum(self.rewards_memory))
            print("TRADE total_cost: ", self.cost)
            print("TRADE total trades: ", self.trades)
            print("TRADE Sharpe Ratio: ",self.sharpe, ',\n')
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(self.working_path+'/account_rewards_trade_{}.csv'.format(self.model_name))            
        
        print("\nself.end_total_asset:{}".format(self.end_total_asset))
        print('=')
        print('asset without stock return')
        print(self.state[0])
        print('+')
        print('STOCKS PRICE:')
        print(np.array(self.state[1:(self.stocks_dim+1)]))
        print('*')
        print('SHARES per STOCK:')
        print(np.array(self.state[(self.stocks_dim+1):(self.stocks_dim*2+1)]))
        print('=')
        print(self.state[0], '+', sum(np.array(self.state[1:(self.stocks_dim+1)]) * np.array(self.state[(self.stocks_dim+1):(self.stocks_dim*2+1)])))
        
        print('\nself.state\n', self.state)
        
        print ('\nCURRENT VS PREVIOUS STATES COMPARISON')
        print('self.previous_state stocks price', self.previous_state[1:(self.stocks_dim+1)])
        print('self.state stocks price', self.state[1:(self.stocks_dim+1)])
        self.stock_var = [self.state[1:(self.stocks_dim+1)][i] - self.previous_state[1:(self.stocks_dim+1)][i] for i in range(len(self.state[1:(self.stocks_dim+1)]))]
        print('difference:', self.stock_var, 'sign:', np.sign(self.stock_var))
        
        self.confusion_generation()
        
        return self.state, self.reward, self.terminal, {}


#######################################################################################################################################################   

    '''
    The reset function is a standard method in reinforcement learning environments. It’s used to reset the environment to its initial state at the start of each new episode. In this case, it’s resetting various variables related to the stock trading environment, such as the asset memory, day, data, turbulence, cost, trades, terminal status, rewards memory, and state.
    '''
    def reset(self):        
        if self.initial:    
            print('\n******************* reset initial TRADE *******************')
            self.asset_memory = [self.initial_amount]
            self.prev_data = self.df.loc[self.initial_day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.rewards_memory = []
            self.end_total_asset = self.initial_amount
                        
            self.state = [self.initial_amount] + \
                          self.initial_close + \
                          [0]*self.stocks_dim + \
                          sum([self.prev_data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            
            print('self.state\n', self.state)

        else:
            print('\n******************* reset NOT initial TRADE *******************')   
            
            # previous_total_asset = self.end_total_asset
            # print('previous_total_asset',previous_total_asset) 
            # self.asset_memory = [previous_total_asset]
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
            
            print('self.state\n', self.state)
            
            self.confusion_report()
 
        return self.state
    
#######################################################################################################################################################   
   
    
    def render(self, mode='human',close=False):
        return self.state

#######################################################################################################################################################   
   

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
#######################################################################################################################################################

    def confusion_generation(self):
        
        previous_state_prices = self.previous_state[1:(self.stocks_dim+1)]
        current_state_prices = self.state[1:(self.stocks_dim+1)]
        
        for i in range(self.stocks_dim):
            preds = np.sign(self.actions[i])
            actuals = np.sign(current_state_prices[i] - previous_state_prices[i])
            self.confusion_matrices[i].append((preds, actuals)) # update confusion matrix for each stock
            self.confusion_matrices_day[i].append((self.day - 1, ('Pred:', self.actions[i], '-->', preds), ('Act:', current_state_prices[i], previous_state_prices[i], '-->', actuals))) 

            
#######################################################################################################################################################            
    def confusion_report(self):
        
        tickers_list = self.prev_data.Ticker.unique()
        
        for i in range(len(tickers_list)):
            
            print(f"\nPred VS Actual values for Stock {tickers_list[i]}:")
            print(self.confusion_matrices_day[i])
            
            # Get the predicted and actual values
            preds, actuals = zip(*self.confusion_matrices[i])

            # Generate and print the classification report
            print(f"Classification Report for Stock {tickers_list[i]}:")
            print(classification_report(actuals, preds))

            # Generate the confusion matrix
            cm = confusion_matrix(actuals, preds)

            # Get unique labels
            labels = sorted(list(set(actuals + preds)))
            
            # Visualize the confusion matrix
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix for Stock {tickers_list[i]}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()
            
                    
        
            
        
        