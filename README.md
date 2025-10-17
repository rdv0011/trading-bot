# Trading Bot

A modular crypto trading bot powered by **Lumibot**, **CatBoost**, **XGBoost**, and **pandas**, designed for flexible strategy development, backtesting, and live trading.

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
