# Q-LEARNING vs SARSA control - sensibility analysis
Simple python scripts used to compare the hyperparameters sensibility in two common reinforcement learning approaches for control (Q-learning & Sarsa) using OpenAI gym's CartPole-v0 environment.

## Modes
In these scripts you can find a variable called "mode", you can set it to:
* **EPS_SENSIBILITY** to analyze the sensibility to the exploration rate;
* **ALPHA_SENSIBILITY** to analyze the sensibility to the learning rate;
* **DISCOUNT_SENSIBILITY** to analyze the sensibility to the discount rate.

## Plots
All plots use the boxplot function from matplotlib library.
