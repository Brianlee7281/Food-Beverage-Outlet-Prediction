# Food-Beverage-Outlet-Prediction

1. Objective
 
This project is designed to forecast food and beverage sales columes across different store menus.

2. Key Features

The model groups data by Store Menu and trains a distinct model for each item to capture localized trends.
The time series sequencing implements a window approach, utilizing 28 days of historical data to predict 7 days into the future.
MinMaxScaler was applied to normaize sales volumes per menu item before training, ensuring stable gradient descent.

3. Model Architecture
 
This pipeline trains individualized sequence models for each specific store and menu combination to predict future demand.

The core prediction model is MultiOutputLSTM that reads temporal sequences and oututs a multi-layered forecast

4. Output
