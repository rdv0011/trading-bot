# Trading Bot

A modular crypto trading bot powered by **Lumibot**, **CatBoost**, **XGBoost**, and **pandas**, designed for flexible strategy development, backtesting, and live trading. This implementation is vibecoded based on ideas from:

* [Real-time-head-to-head-adaptive-modeling](https://emergentmethods.medium.com/real-time-head-to-head-adaptive-modeling-of-financial-market-data-using-xgboost-and-catboost-995a115a7495 )

* [Freqai-from-price-to-prediction](https://emergentmethods.medium.com/freqai-from-price-to-prediction-6fadac18b665)

---

## 🚀 Features

- Built on a **custom Lumibot fork** with Binance asset symbol fixes  
  ([rdv0011/lumibot@fix/binance-limit-assets-asset-symbol](https://github.com/rdv0011/lumibot/tree/fix/binance-limit-assets-asset-symbol))
- Integrated **machine learning models** using:
  - [CatBoost](https://catboost.ai/) for gradient boosting on categorical and numeric data  
  - [XGBoost](https://xgboost.readthedocs.io/) for fast tree-based modeling
- Uses `pandas_market_calendars` for exchange trading session logic
- Compatible with **Python 3.11** and **Conda environments**

---

## 🧩 Installation

Clone the repository and set up your environment:

```bash
gh repo clone rdv0011/trading-bot
cd trading-bot
```

## Then create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate tradingbot
```

This will:
* Install NumPy, Pandas, SciPy, CatBoost, and XGBoost from conda-forge
* Install Lumibot from your forked branch
* Install the latest pandas_market_calendars directly from GitHub

## 🧠 Usage
Run your trading strategy:
python main.py
