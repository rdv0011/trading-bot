# Trading Bot

A modular crypto trading bot powered by **Lumibot**, **CatBoost**, **XGBoost**, and **pandas**, designed for flexible strategy development, backtesting, and live trading. This implementation is vibecoded based on ideas from:

* [Real-time-head-to-head-adaptive-modeling](https://emergentmethods.medium.com/real-time-head-to-head-adaptive-modeling-of-financial-market-data-using-xgboost-and-catboost-995a115a7495 )

* [Freqai-from-price-to-prediction](https://emergentmethods.medium.com/freqai-from-price-to-prediction-6fadac18b665)

---

## 🚀 Features

- Integrated **machine learning models** using:
  - [CatBoost](https://catboost.ai/) for gradient boosting on categorical and numeric data  
  - [XGBoost](https://xgboost.readthedocs.io/) for fast tree-based modeling
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

## 🧠 Usage
Run your trading strategy:
```python main.py```
Trading bot uses the latest by date model from the model/ folder

Run training:
```python xgcatboost.py```
The training script fill remove the oldets models to keep two model versions only to use a minimum file system storage.
> **_NOTE:_** The traning step shuold be repeated on a daily basic. To do this the crone job can be created to execute run traning script. Run the following ```crontab -e``` and put the following line 
```
0 2 * * * /usr/bin/python3 /fullpath_to_local_repo/xgcatboost.py >> /fullpath_to_local_repo/xgcatboost.log 2>&1
```
Breakdown:
```
0 2 * * * → Run at 2:00 AM daily
/usr/bin/python3 → Path to Python
/fullpath_to_local_repo/xgcatboost.py → Full path to your script
>> /fullpath_to_local_repo/xgcatboost.log 2>&1 → Logs output and errors to a file
```
Once trainng is complete and new models are generated the running trading bot will swap the model at runtime automatically.

## When running locally on Mac Mini
When local locally not in the Cloud the Mac Mini needs to be keep runnin 24/7. To achieve this run keepawake/keepawake.sh.
To disable this run keepawake/cancelkeepawake.sh

## Improvements
Optimize metaparameter: take profit, stop loss ...
