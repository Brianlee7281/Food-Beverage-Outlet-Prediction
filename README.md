# 🍽️ F&B Store Sales Volume Prediction (LSTM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

## 📌 Overview
This repository contains a deep learning pipeline designed to forecast food and beverage sales volumes across different store menus.  

By framing the problem as a multivariate time-series forecasting task, the pipeline trains individualized sequence models for each specific store and menu combination to predict future demand accurately.

## ✨ Key Features
* **Individualized Modeling:** Groups data by `영업장명_메뉴명` (Store_Menu) and trains a distinct model for each item to capture localized trends.
* **Time-Series Sequencing:** Implements a sliding window approach, utilizing 28 days of historical data to predict exactly 7 days into the future.
* **Automated Scaling:** Applies `MinMaxScaler` to normalize sales volumes per menu item before training, ensuring stable gradient descent.
* **Batch Inference:** Automatically parses through multiple `TEST_*.csv` files, generates predictions, and maps them to a centralized `sample_submission.csv` format.

---

## 🧠 Model Architecture
The core predictive engine is the `MultiOutputLSTM`, a custom PyTorch module.  

It reads temporal sequences and directly outputs a multi-step forecast:

| Layer | Type | Configuration |
| :--- | :--- | :--- |
| **Input** | Time-Series Sequence | `input_dim = 1` (Sales Volume) |
| **Hidden** | LSTM | `hidden_dim = 64`, `num_layers = 2`, `batch_first=True` |
| **Output** | Linear (Fully Connected) | `output_dim = 7` (7-day prediction) |

---

## ⚙️ Hyperparameters & Configuration
The training pipeline is configured with the following global variables to ensure deterministic and reproducible results (via a strict seed of `42`):

* **LOOKBACK:** `28` days
* **PREDICT:** `7` days
* **BATCH_SIZE:** `16`
* **EPOCHS:** `50`
* **Optimizer:** Adam (`lr = 0.001`)
* **Loss Function:** Mean Squared Error (MSELoss)

## 🚀 How to Run
1. Place the training dataset (`train.csv`), test files (`TEST_*.csv`), and `sample_submission.csv` in the root directory.
2. Run the pipeline script. It will iterate through each menu item, scaling the data and training the LSTM models.
3. The inference function will process all test files and output the final results into `baseline_submission.csv`.
