#!/usr/bin/env python
# coding: utf-8

import numpy as np

import gymnasium as gym
from gymnasium import spaces
import logging

class StockTradingEnv(gym.Env):
    """
    A custom Stock Trading environment for reinforcement learning.
    This environment simulates stock trading for multiple tickers (stocks),
    where the agent interacts by buying or selling shares to maximize net worth.
    """
    metadata = {'render_modes': ['human']}  # Specify available render modes for visualization


    
    def __init__(self, stock_data, render_mode=False, log_file=False, transaction_cost_percent=0.005):
        """
        Initialize the Stock Trading Environment.

        Parameters:
        stock_data (dict): A dictionary where keys are stock tickers and values are DataFrames
                           with price and volume data for each stock.
        """
        super(StockTradingEnv, self).__init__()

        self.render_mode = render_mode
        self.log_file=log_file
        
        # Step 1: Remove any empty DataFrames (tickers with no data)
        self.stock_data = {ticker: df for ticker, df in stock_data.items() if not df.empty}
        self.tickers = list(self.stock_data.keys())  # List of tickers (stock symbols)

        if not self.tickers:
            raise ValueError("All provided stock data is empty")  # Raise an error if no valid data exists

        # Step 2: Calculate the number of features per stock (e.g., 'Open', 'Close', etc.)
        sample_df = next(iter(self.stock_data.values()))  # Take the first DataFrame as a sample
        self.n_features = len(sample_df.columns)  # Number of columns/features in the DataFrame

        # Step 3: Define the action space
        # Action space allows for an action per ticker, where values between -1 and 1 control:
        #   -1: Full sell
        #    0: Hold
        #   +1: Full buy
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32)

        # Step 4: Define the observation space
        # Observation space includes:
        # (n stocks * m columns) elements for price data
        # 1 elements for balance
        # (n stocks) elements for the shares values for each stock.
        # 3 elements for net_worth, max_net_worth (highest portfolio value (net worth) that the agent has achieved at any point during an episode) and current_step.
        self.obs_shape = self.n_features * len(self.tickers) + 1 + len(self.tickers) + 3 #  before [self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2] 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

        # Step 5: Initialize financial variables
        self.initial_balance = 100000  # Starting account balance
        self.balance = self.initial_balance  # Current account balance
        self.net_worth = self.initial_balance  # Current net worth of the portfolio
        self.max_net_worth = self.initial_balance  # Highest net worth achieved during the episode
        self.shares_held = {ticker: 0 for ticker in self.tickers}  # Shares held for each stock
        # self.total_shares_sold = {ticker: 0 for ticker in self.tickers}  # Total shares sold for each stock
        # self.total_sales_value = {ticker: 0 for ticker in self.tickers}  # Total revenue from sales per stock

        # Step 6: Initialize step tracking
        self.current_step = 0  # Start from the first step

        # Step 7: Determine the maximum number of steps (minimum length of data across all stocks)
        # minimum length across all the DataFrames (i.e., the stock with the shortest historical dataset).
        # The environment ensures that the agent doesn't exceed this limit, as it marks the end of the episode.
        self.max_steps = max(0, max(len(df) for df in self.stock_data.values()) - 1)
        
        # Transaction cost
        self.transaction_cost_percent = transaction_cost_percent


    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state at the start of each episode.
        """
        super().reset(seed=seed)  # Reset the RNG for reproducibility
        # The reset() method in the parent class is responsible for resetting the random number generator (RNG) to ensure reproducibility.
        self.balance = self.initial_balance  # Reset account balance
        self.net_worth = self.initial_balance  # Reset net worth
        self.max_net_worth = self.initial_balance  # Reset max net worth
        self.shares_held = {ticker: 0 for ticker in self.tickers}  # Reset shares held
        # self.total_shares_sold = {ticker: 0 for ticker in self.tickers}  # Reset total shares sold
        # self.total_sales_value = {ticker: 0 for ticker in self.tickers}  # Reset total sales value
        self.current_step = 0  # Reset step counter
        return self._next_observation(), {}  # Return initial observation


    
    def _next_observation(self):
        """
        Generate the next observation for the agent (frame is the real observation data, dynamically changing at each step)

        Observation includes:
        - Price data for all stocks at the current step
        - Balance
        - Shares held
        - Net worth
        - Maximum net worth
        """
        frame = np.zeros(self.obs_shape)  # Create a blank observation frame

        # Include price data for all stocks
        idx = 0

        # Populate stock feature data
        for ticker in self.tickers:
            df = self.stock_data[ticker]
            if self.current_step < len(df):
                # Use data from the current step
                frame[idx:idx+self.n_features] = df.iloc[self.current_step].values # the row at current step contains the feature values for the current timestep.
            else: # if self.current_step is beyond the length of the DataFrame (it can happen if some stocks have fewer rows than others)
                # Use the last available data if the current step exceeds the data length
                frame[idx:idx+self.n_features] = df.iloc[-1].values
            idx += self.n_features

        # Populate portfolio metrics
        frame[idx] = self.balance  # Add balance
        idx += 1
        
        frame[idx:idx+len(self.tickers)] = [self.shares_held[ticker] for ticker in self.tickers]  # Add shares held
        idx += len(self.tickers)
        
        frame[idx] = self.net_worth  # Add net worth
        idx += 1
        
        frame[idx] = self.max_net_worth  # Add maximum net worth
        idx += 1
        
        frame[idx] = self.current_step  # Add current step

        return frame


    
    def step(self, actions):
        """
        Execute the agent's actions (buy/sell/hold) and update the environment.
    
        Parameters:
            actions (array): An array of actions for each stock, ranging from -1 (sell) to +1 (buy).
    
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        info = {}
        truncated = False
    
        # Check if the episode has reached its maximum steps (truncation)
        if self.current_step >= self.max_steps:
            done = True
            truncated = True
            reward = 0
            obs = self._next_observation()
            return obs, reward, done, truncated, info  
    
        # Update step counter
        self.current_step += 1
    
        # Calculate the previous closing prices for all tickers
        previous_prices = {
            ticker: self.stock_data[ticker].iloc[self.current_step - 1]['Close']
            if self.current_step - 1 < len(self.stock_data[ticker])
            else self.stock_data[ticker].iloc[-1]['Close']  # Use the last known price if out-of-bounds (stocks with less data)
            for ticker in self.tickers
        }

    
        # Step 1: Execute all sales first
        for i, ticker in enumerate(self.tickers):
            action = actions[i]
            if action < 0:  # Sell
                # Calculate the amount to sell based on shares held. If shares to sell exceed shares held, sell all available shares
                shares_to_sell = min(self.shares_held[ticker] * abs(action), self.shares_held[ticker])  # Allow partial shares
                sale = shares_to_sell * previous_prices[ticker]
                transaction_cost = sale * self.transaction_cost_percent
                # Update balance and holdings
                self.balance += (sale - transaction_cost)
                self.shares_held[ticker] -= shares_to_sell
                # self.total_shares_sold[ticker] += shares_to_sell
                # self.total_sales_value[ticker] += sale
        
        # Calculate the remaining balance and set maximum purchase value per stock
        num_stocks_to_buy = sum(1 for action in actions if action > 0)  # Count stocks with buy actions
        max_purchase_value = self.balance / num_stocks_to_buy if num_stocks_to_buy > 0 else 0
        
        # Step 2: Execute purchases
        for i, ticker in enumerate(self.tickers):
            action = actions[i]
            if action > 0:  # Buy
                # Calculate the cost and the maximum number of shares to buy based on the capped purchase value
                cost = max_purchase_value * action
                transaction_cost = cost * self.transaction_cost_percent
                shares_to_buy = cost / previous_prices[ticker]  # Allow partial shares
                # Update balance and holdings
                self.balance -= (cost + transaction_cost)
                self.shares_held[ticker] += shares_to_buy

    
        # Get the current prices for portfolio valuation
        current_prices = {
            ticker: self.stock_data[ticker].iloc[self.current_step]['Close']
            if self.current_step < len(self.stock_data[ticker])
            else self.stock_data[ticker].iloc[-1]['Close'] # Use the last known price if out-of-bounds (stocks with less data)
            for ticker in self.tickers
        }


    
        # Update net worth and reward
        self.net_worth = self.balance + sum(
            self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers
        )
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        reward = self.net_worth - self.initial_balance
    
        # Determine if the episode is done
        done = self.net_worth <= 0 # or self.current_step >= self.max_steps
    
        # Generate the next observation
        obs = self._next_observation()
    
        return obs, reward, done, truncated, info  


    def render(self, agent_name):
        """
        Render the current state of the environment (useful for debugging).
        """
        if self.render_mode == 'human':
            profit = self.net_worth - self.initial_balance
            log_message = ("\n"+
                f"Agent: {agent_name}\n"
                f"Step: {self.current_step}\n"
                f"Steps max: {self.max_steps}\n"
                f"Balance: {self.balance:.2f}\n"
                f"Net worth: {self.net_worth:.2f}\n"
                f"Profit: {profit:.2f}\n"
                "Shares Price:\n" +
                "\n".join([f"  {ticker}: {self.stock_data[ticker].iloc[self.current_step]['Close']}" for ticker in self.tickers]) + "\n"
                "Shares Held:\n" +
                "\n".join([f"  {ticker}: {self.shares_held[ticker]}" for ticker in self.tickers]) + "\n"                           
            )

            # Setup logging
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format="%(asctime)s - %(message)s",
                filemode="w"  # Overwrite the file each time; use "a" to append
            )
            
            self.logger = logging.getLogger()
            
            self.logger.info(log_message)


    
    def close(self):
        """
        Close the environment (cleanup if necessary).
        """
        pass


    
    # def update_stock_data(self, new_stock_data):
    #     """
    #     Update the environment with new stock data.

    #     Parameters:
    #     new_stock_data (dict): Dictionary containing new stock data (ticker: DataFrame).
    #     """
    #     # Remove any empty DataFrames
    #     self.stock_data = {ticker: df for ticker, df in new_stock_data.items() if not df.empty}

    #     self.tickers = list(self.stock_data.keys())
    
    #     if not self.tickers:
    #         raise ValueError("All new stock data are empty")
    
    #     # Update the number of features if needed
    #     sample_df = next(iter(self.stock_data.values()))
    #     self.n_features = len(sample_df.columns)
    
    #     # Update observation space
    #     self.obs_shape = self.n_features * len(self.tickers) + 1 + len(self.tickers) + 3
    #     self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
    
    #     # Update maximum steps
    #     self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)
    
    #     # Update transaction cost if provided
    #     if transaction_cost_percent is not None:
    #         self.transaction_cost_percent = transaction_cost_percent
    
    #     # Reset the environment
    #     self.reset()
    
    #     print(f"The environment has been updated with {len(self.tickers)} new stocks.")




