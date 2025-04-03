 • State 𝒔 = [b, 𝒑, 𝒉] is a representation of a portfolio <-- check better State Space below
 where  𝑏 (real number) represents the remaining balance,
 𝒑 (vector price*D) represents the prices of D different stocks, 
 𝒉 (vector shares*D) represents the number of shares held for these D stocks.
 
 • Action 𝒂: a vector of actions over 𝐷 stocks. 
 The allowed actions on each stock include selling, buying, or holding an amount of shares 𝒌[𝑑]
 which result in decreasing, increasing, and no change of the stock shares 𝒉[𝑑], respectively.
 
 • Reward 𝑟 (𝑠,𝑎,𝑠′): the direct reward of taking action 𝑎 at state 𝑠 and arriving at the new state 𝑠′. Our aim is to maximize the reward.
 We define our reward function as the change of the portfolio value when action 𝑎 is taken at state 𝑠 and arriving at new state 𝑠 + 1 --> 𝑟(𝑠𝑡,𝑎𝑡,𝑠𝑡+1)
 
 • Policy 𝜋 (𝑠): the trading strategy at state 𝑠, which is the probability distribution of actions at state 𝑠.
 
 • Q-value 𝑄𝜋 (𝑠, 𝑎): the expected reward of taking action 𝑎 at state 𝑠 following policy 𝜋 .
 
 Environment for multiple stocks:
State Space (example with n_stocks=30): it's a 181-dimensional vector (1 + 30 stocks * 6) consists of seven parts of information to represent the state space of multiple stocks trading environment : Balance + 30 stocks * (Close Price, Shares owned, MACD, RSI, CCI, ADX)

 The action space is defined as {-k,…,-1, 0, 1, …, k}, where k and -k presents the number of shares we can buy and sell, and |k| ≤ h_max, being h_max a predefined parameter that sets as the maximum amount of shares for each buying action. So the size of the entire action space is (2k+1)^n_stocks.
The action space is then normalized to [-1, 1], since the RL algorithms A2C and PPO define the policy directly on a Gaussian distribution


--------------------------------------------------

The env folder likely contains different environment files for different stages of the reinforcement learning process. Each file probably sets up a specific environment for trading multiple stocks. Here’s what each one might be used for:

EnvMultipleStock_train.py: This file sets up the environment for the training phase of the reinforcement learning process. During training, the agent learns to interact with the environment by taking actions and receiving rewards, updating its policy based on the outcomes.

EnvMultipleStock_validation.py: This file is  used for the validation phase. During validation, the trained agent’s performance is evaluated on a separate validation dataset that was not used during training. This helps to ensure that the agent can generalize its learned policy to new, unseen data.

EnvMultipleStock_trade.py: This file is used for the trading (or testing) phase. After the agent has been trained and validated, it’s tested in a trading environment with real or more recent data. This is where you see how the agent performs in practice, making actual trading decisions.

Remember, this is a common practice in machine learning and reinforcement learning to have separate datasets or environments for training, validation, and testing (or in this case, trading). It helps to ensure that the model is learning effectively and can generalize its learning to new data.