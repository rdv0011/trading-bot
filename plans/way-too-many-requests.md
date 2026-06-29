## The log:

2026-06-19 23:55:25,861 - INFO - 💤 Sleeping 280s until next 5m candle close                                             
2026-06-20 00:00:07,227 - INFO - Position | FLAT                                                                         
2026-06-20 00:00:30,477 - INFO - Tactical | signal=HOLD pred=-0.001981 min=-0.015887 max=0.004699                        
2026-06-20 00:00:30,479 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.6x exposure=1.00                 
2026-06-20 00:00:30,760 - INFO - 💤 Sleeping 275s until next 5m candle close                                             
2026-06-20 00:05:07,160 - ERROR - ❌ Error fetching leverage for BTCUSDT: APIError(code=-1003): Way too many requests; IP
(15.158.219.106) banned until 1781906804281. Please use the websocket for live updates to avoid bans.                    
2026-06-20 00:05:07,161 - INFO - Position | FLAT                                                                         
2026-06-20 00:05:27,751 - INFO - Tactical | signal=HOLD pred=-0.001728 min=-0.015887 max=0.004699
2026-06-20 00:05:27,753 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.7x exposure=0.99
2026-06-20 00:05:28,039 - INFO - 💤 Sleeping 277s until next 5m candle close
2026-06-20 00:10:06,420 - INFO - Position | FLAT                                                                         
2026-06-20 00:10:25,817 - INFO - Tactical | signal=HOLD pred=-0.003633 min=-0.015887 max=0.004699
2026-06-20 00:10:25,818 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.7x exposure=0.99
2026-06-20 00:10:26,103 - INFO - 💤 Sleeping 279s until next 5m candle close
2026-06-20 00:15:06,412 - INFO - Position | FLAT                                                                         
2026-06-20 00:15:06,678 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.105) banned until 1781907947627. Please use the websocket for live updates to avoid bans.           
2026-06-20 00:15:06,679 - INFO - ❌ Insufficient tactical data, skipping                                                 
2026-06-20 00:15:06,680 - INFO - 💤 Sleeping 299s until next 5m candle close
2026-06-20 00:20:07,084 - INFO - Position | FLAT                                                                         
2026-06-20 00:20:07,370 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.105) banned until 1781907947627. Please use the websocket for live updates to avoid bans.           
2026-06-20 00:20:07,377 - INFO - ❌ Insufficient tactical data, skipping                                                 
2026-06-20 00:20:07,378 - INFO - 💤 Sleeping 298s until next 5m candle close
2026-06-20 00:25:07,373 - INFO - Position | FLAT                                                                         
2026-06-20 00:25:07,648 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.105) banned until 1781908188094. Please use the websocket for live updates to avoid bans.
2026-06-20 00:25:07,649 - INFO - ❌ Insufficient tactical data, skipping                                                 
2026-06-20 00:25:07,650 - INFO - 💤 Sleeping 298s until next 5m candle close
2026-06-20 00:30:07,152 - INFO - Position | FLAT                                                                         
2026-06-20 00:30:14,956 - INFO - 🔄 StrategicML hot-swap: /home/armbian/trading-bot/model/strategic_meta_model_20260619T2
22041Z.pkl
[load_model] Loaded model: /home/armbian/trading-bot/model/strategic_meta_model_20260619T222041Z.pkl, meta info: /home/ar
mbian/trading-bot/model/strategic_meta_model_20260619T222041Z.meta.json
2026-06-20 00:30:15,725 - INFO - Tactical | signal=HOLD pred=-0.003277 min=-0.015887 max=0.004699
2026-06-20 00:30:15,727 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.2x exposure=1.00
2026-06-20 00:30:16,003 - INFO - 💤 Sleeping 289s until next 5m candle close
2026-06-20 00:35:05,989 - INFO - ❌ Error in trading iteration: APIError(code=-1003): Way too many requests; IP(15.158.21
9.71) banned until 1781909029251. Please use the websocket for live updates to avoid bans.
2026-06-20 00:35:05,992 - INFO - Traceback: Traceback (most recent call last):
  File "/home/armbian/trading-bot/basestrategy.py", line 99, in run                                                      
    self.on_trading_iteration()                             
  File "/home/armbian/trading-bot/dualmlstrategy.py", line 87, in on_trading_iteration
    current_equity = self.get_cash()                        
  File "/home/armbian/trading-bot/basestrategy.py", line 23, in get_cash                                                 
    return self._broker.get_cash(self.quote_asset_symbol)                                                                
  File "/home/armbian/trading-bot/binancefuturesbroker.py", line 21, in get_cash
    balances = self.client.futures_account_balance()                                                                     
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 8365, in futures_a
ccount_balance
    return self._request_futures_api("get", "balance", True, 3, data=params)
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 148, in _request_f
utures_api
    return self._request(method, uri, signed, force_params, **kwargs)                                                    
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 111, in _request
    return self._handle_response(self.response)                                                                          
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 120, in _handle_re
sponse
    raise BinanceAPIException(response, response.status_code, response.text)
binance.exceptions.BinanceAPIException: APIError(code=-1003): Way too many requests; IP(15.158.219.71) banned until 17819
2026-06-20 00:36:07,140 - ERROR - ❌ Error fetching position for BTCUSDT: APIError(code=-1003): Way too many requests; IP
(15.158.219.71) banned until 1781908909132. Please use the websocket for live updates to avoid bans.
2026-06-20 00:36:07,430 - INFO - Position | FLAT                                                                         
2026-06-20 00:36:15,081 - INFO - Tactical | signal=HOLD pred=-0.002048 min=-0.015887 max=0.004699
2026-06-20 00:36:15,082 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.3x exposure=1.00
2026-06-20 00:36:15,354 - INFO - 💤 Sleeping 230s until next 5m candle close
2026-06-20 00:40:05,993 - INFO - ❌ Error in trading iteration: APIError(code=-1003): Way too many requests; IP(15.158.21
9.71) banned until 1781908909132. Please use the websocket for live updates to avoid bans.
2026-06-20 00:40:05,998 - INFO - Traceback: Traceback (most recent call last):
  File "/home/armbian/trading-bot/basestrategy.py", line 99, in run                                                      
    self.on_trading_iteration()                             
  File "/home/armbian/trading-bot/dualmlstrategy.py", line 87, in on_trading_iteration
    current_equity = self.get_cash()                        
  File "/home/armbian/trading-bot/basestrategy.py", line 23, in get_cash                                                 
    return self._broker.get_cash(self.quote_asset_symbol)                                                                
  File "/home/armbian/trading-bot/binancefuturesbroker.py", line 21, in get_cash
    balances = self.client.futures_account_balance()                                                                     
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 8365, in futures_a
ccount_balance
    return self._request_futures_api("get", "balance", True, 3, data=params)
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 148, in _request_f
utures_api
    return self._request(method, uri, signed, force_params, **kwargs)                                                    
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 111, in _request
    return self._handle_response(self.response)                                                                          
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 120, in _handle_re
sponse
    raise BinanceAPIException(response, response.status_code, response.text)
binance.exceptions.BinanceAPIException: APIError(code=-1003): Way too many requests; IP(15.158.219.71) banned until 17819
08909132. Please use the websocket for live updates to avoid bans.

2026-06-20 00:41:06,923 - INFO - Position | FLAT                                                                         
2026-06-20 00:41:15,285 - INFO - Tactical | signal=HOLD pred=-0.001328 min=-0.015887 max=0.004699
2026-06-20 00:41:15,286 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.3x exposure=1.00
2026-06-20 00:41:15,559 - INFO - 💤 Sleeping 230s until next 5m candle close
2026-06-20 00:45:06,786 - INFO - Position | FLAT                                                                         
2026-06-20 00:45:14,414 - INFO - Tactical | signal=HOLD pred=-0.001580 min=-0.015887 max=0.004699
2026-06-20 00:45:14,416 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.2x exposure=1.00
2026-06-20 00:45:14,693 - INFO - 💤 Sleeping 291s until next 5m candle close
2026-06-20 00:50:07,299 - INFO - Position | FLAT                                                                         
2026-06-20 00:50:15,184 - INFO - Tactical | signal=HOLD pred=-0.002604 min=-0.015887 max=0.004699
2026-06-20 00:50:15,185 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.2x exposure=1.00
2026-06-20 00:50:15,460 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 00:55:06,935 - INFO - Position | FLAT                                                                         
2026-06-20 00:55:14,761 - INFO - Tactical | signal=HOLD pred=-0.000552 min=-0.015887 max=0.004699
2026-06-20 00:55:14,763 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.2x exposure=1.00
2026-06-20 00:55:15,039 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 01:00:06,600 - INFO - Position | FLAT                                                                         
2026-06-20 01:00:14,438 - INFO - Tactical | signal=HOLD pred=-0.001117 min=-0.015887 max=0.004699
2026-06-20 01:00:14,439 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.2x exposure=1.00
2026-06-20 01:00:14,716 - INFO - 💤 Sleeping 291s until next 5m candle close
2026-06-20 02:25:15,262 - INFO - Tactical | signal=HOLD pred=-0.000846 min=-0.015887 max=0.004699                        
2026-06-20 02:25:15,263 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00                 
2026-06-20 02:25:16,025 - INFO - 💤 Sleeping 289s until next 5m candle close                                             
2026-06-20 02:30:06,520 - INFO - Position | FLAT                                                                         
2026-06-20 02:30:07,279 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.106) banned until 1781915996152. Please use the websocket for live updates to avoid bans.           
2026-06-20 02:30:07,280 - INFO - ❌ Insufficient tactical data, skipping                                                 
2026-06-20 02:30:07,280 - INFO - 💤 Sleeping 298s until next 5m candle close
2026-06-20 02:35:06,827 - INFO - Position | FLAT                                                                         
2026-06-20 02:35:07,680 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.106) banned until 1781915996152. Please use the websocket for live updates to avoid bans.
2026-06-20 02:35:07,682 - INFO - ❌ Insufficient strategic data, skipping                                                
2026-06-20 02:35:07,683 - INFO - 💤 Sleeping 298s until next 5m candle close
2026-06-20 02:40:07,183 - INFO - Position | FLAT                                                                         
2026-06-20 02:40:15,053 - INFO - Tactical | signal=HOLD pred=-0.000160 min=-0.015887 max=0.004699
2026-06-20 02:40:15,054 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 02:40:15,327 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 02:45:06,795 - INFO - Position | FLAT                                                                         
2026-06-20 02:45:14,363 - INFO - Tactical | signal=HOLD pred=0.000124 min=-0.015887 max=0.004699
2026-06-20 02:45:14,364 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 02:45:14,634 - INFO - 💤 Sleeping 291s until next 5m candle close
2026-06-20 02:50:07,215 - INFO - Position | FLAT                                                                         
2026-06-20 02:50:14,753 - INFO - Tactical | signal=HOLD pred=0.000308 min=-0.015887 max=0.004699
2026-06-20 02:50:14,754 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 02:50:15,028 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 02:55:06,542 - INFO - Position | FLAT                                                                         
2026-06-20 02:55:14,118 - INFO - Tactical | signal=HOLD pred=-0.000575 min=-0.015887 max=0.004699
2026-06-20 02:55:14,120 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.4x exposure=1.00
2026-06-20 02:55:14,885 - INFO - 💤 Sleeping 291s until next 5m candle close
2026-06-20 03:00:07,095 - ERROR - ❌ Error fetching position for BTCUSDT: APIError(code=-1003): Way too many requests; IP
(15.158.219.71) banned until 1781917912912. Please use the websocket for live updates to avoid ban
2026-06-20 03:00:07,367 - INFO - Position | FLAT                                                                         
2026-06-20 03:00:15,242 - INFO - Tactical | signal=HOLD pred=-0.000623 min=-0.015887 max=0.004699
2026-06-20 03:00:15,243 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.4x exposure=1.00
2026-06-20 03:00:15,520 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 03:05:07,025 - INFO - Position | FLAT                                                                         
2026-06-20 03:05:14,847 - INFO - Tactical | signal=HOLD pred=-0.000204 min=-0.015887 max=0.004699
2026-06-20 03:05:14,849 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 03:05:15,124 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 03:10:06,150 - INFO - ❌ Error in trading iteration: APIError(code=-1003): Way too many requests; IP(15.158.21
9.71) banned until 1781917912912. Please use the websocket for live updates to avoid bans.
2026-06-20 03:10:06,152 - INFO - Traceback: Traceback (most recent call last):
  File "/home/armbian/trading-bot/basestrategy.py", line 99, in run                                                      
    self.on_trading_iteration()                             
  File "/home/armbian/trading-bot/dualmlstrategy.py", line 87, in on_trading_iteration
    current_equity = self.get_cash()                        
  File "/home/armbian/trading-bot/basestrategy.py", line 23, in get_cash                                                 
    return self._broker.get_cash(self.quote_asset_symbol)                                                                
  File "/home/armbian/trading-bot/binancefuturesbroker.py", line 21, in get_cash
    balances = self.client.futures_account_balance()                                                                     
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 8365, in futures_a
ccount_balance
    return self._request_futures_api("get", "balance", True, 3, data=params)
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 148, in _request_f
utures_api
    return self._request(method, uri, signed, force_params, **kwargs)                                                    
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 111, in _request
    return self._handle_response(self.response)                                                                          
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 120, in _handle_re
sponse
2026-06-20 03:11:06,510 - INFO - ❌ Error in trading iteration: APIError(code=-1003): Way too many requests; IP(15.158.21
9.71) banned until 1781917912912. Please use the websocket for live updates to avoid bans.
2026-06-20 03:11:06,516 - INFO - Traceback: Traceback (most recent call last):
  File "/home/armbian/trading-bot/basestrategy.py", line 99, in run                                                      
    self.on_trading_iteration()                             
  File "/home/armbian/trading-bot/dualmlstrategy.py", line 87, in on_trading_iteration
    current_equity = self.get_cash()                        
  File "/home/armbian/trading-bot/basestrategy.py", line 23, in get_cash                                                 
    return self._broker.get_cash(self.quote_asset_symbol)                                                                
  File "/home/armbian/trading-bot/binancefuturesbroker.py", line 21, in get_cash
    balances = self.client.futures_account_balance()                                                                     
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 8365, in futures_a
ccount_balance
    return self._request_futures_api("get", "balance", True, 3, data=params)
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 148, in _request_f
utures_api
    return self._request(method, uri, signed, force_params, **kwargs)                                                    
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 111, in _request
    return self._handle_response(self.response)                                                                          
  File "/home/armbian/miniconda3/envs/tradingbot/lib/python3.10/site-packages/binance/client.py", line 120, in _handle_re
sponse
    raise BinanceAPIException(response, response.status_code, response.text)
binance.exceptions.BinanceAPIException: APIError(code=-1003): Way too many requests; IP(15.158.219.71) banned until 17819
17912912. Please use the websocket for live updates to avoid bans.

2026-06-20 03:12:07,458 - INFO - Position | FLAT                                                                         
2026-06-20 03:12:08,129 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.66) banned until 1781918330696. Please use the websocket for live updates to avoid bans.
2026-06-20 03:12:08,130 - INFO - ❌ Insufficient strategic data, skipping                                                
2026-06-20 03:12:08,130 - INFO - 💤 Sleeping 177s until next 5m candle close
2026-06-20 03:15:06,293 - INFO - Position | FLAT                                                                         
2026-06-20 03:15:14,729 - INFO - Tactical | signal=HOLD pred=0.000723 min=-0.015887 max=0.004699
2026-06-20 03:15:14,730 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 03:15:15,004 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 03:20:06,566 - INFO - Position | FLAT                                                                         
2026-06-20 03:20:06,835 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.66) banned until 1781918451870. Please use the websocket for live updates to avoid bans.
2026-06-20 03:20:06,837 - INFO - ❌ Insufficient tactical data, skipping                                                 
2026-06-20 03:20:06,837 - INFO - 💤 Sleeping 299s until next 5m candle close
2026-06-20 03:25:07,314 - ERROR - ❌ Error fetching leverage for BTCUSDT: APIError(code=-1003): Way too many requests; IP
(15.158.219.69) banned until 1781919479716. Please use the websocket for live updates to avoid bans.
2026-06-20 03:25:07,316 - INFO - Position | FLAT                                                                         
2026-06-20 03:25:15,086 - INFO - Tactical | signal=HOLD pred=-0.000843 min=-0.015887 max=0.004699
2026-06-20 03:25:15,088 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 03:25:15,365 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 03:30:06,834 - INFO - Position | FLAT                                                                         
2026-06-20 03:30:07,105 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.66) banned until 1781920666680. Please use the websocket for live updates to avoid bans.
2026-06-20 03:30:07,106 - INFO - ❌ Insufficient tactical data, skipping                                                 
2026-06-20 03:30:07,107 - INFO - 💤 Sleeping 298s until next 5m candle close
2026-06-20 03:35:06,610 - ERROR - ❌ Error fetching leverage for BTCUSDT: APIError(code=-1003): Way too many requests; IP
(15.158.219.66) banned until 1781920178826. Please use the websocket for live updates to avoid bans.
2026-06-20 03:35:06,611 - INFO - Position | FLAT                                                                         
2026-06-20 03:35:14,198 - INFO - Tactical | signal=HOLD pred=-0.001188 min=-0.015887 max=0.004699
2026-06-20 03:35:14,199 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 03:35:14,954 - INFO - 💤 Sleeping 291s until next 5m candle close
2026-06-20 03:40:07,479 - INFO - Position | FLAT                                                                         
2026-06-20 03:40:08,097 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.66) banned until 1781920666680. Please use the websocket for live updates to avoid bans.
2026-06-20 03:40:08,098 - INFO - ❌ Insufficient strategic data, skipping                                                
2026-06-20 03:40:08,099 - INFO - 💤 Sleeping 297s until next 5m candle close
2026-06-20 03:45:06,587 - INFO - Position | FLAT              
2026-06-20 03:45:14,563 - INFO - Tactical | signal=HOLD pred=-0.001750 min=-0.015887 max=0.004699
2026-06-20 03:45:14,564 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.4x exposure=1.00
2026-06-20 03:45:14,841 - INFO - 💤 Sleeping 291s until next 5m candle close
2026-06-20 03:50:07,317 - INFO - Position | FLAT                                                                         
2026-06-20 03:50:15,382 - INFO - Tactical | signal=HOLD pred=-0.001998 min=-0.015887 max=0.004699
2026-06-20 03:50:15,384 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.4x exposure=1.00
2026-06-20 03:50:15,657 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 03:55:07,202 - INFO - Position | FLAT                                                                         
2026-06-20 03:55:07,484 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.66) banned until 1781920666680. Please use the websocket for live updates to avoid bans.
2026-06-20 03:55:07,485 - INFO - ❌ Insufficient tactical data, skipping                                                 
2026-06-20 03:55:07,486 - INFO - 💤 Sleeping 298s until next 5m candle close
2026-06-20 04:00:06,977 - INFO - Position | FLAT                                                                         
2026-06-20 04:00:15,304 - INFO - Tactical | signal=HOLD pred=-0.000993 min=-0.015887 max=0.004699
2026-06-20 04:00:15,305 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 04:00:15,578 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 04:05:07,086 - INFO - Position | FLAT                                                                         
2026-06-20 04:05:15,210 - INFO - Tactical | signal=HOLD pred=-0.000997 min=-0.015887 max=0.004699
2026-06-20 04:05:15,210 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.5x exposure=1.00
2026-06-20 04:05:15,477 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 04:10:06,995 - INFO - Position | FLAT                                                                         
2026-06-20 04:10:14,636 - INFO - Tactical | signal=HOLD pred=-0.001353 min=-0.015887 max=0.004699
2026-06-20 04:10:14,638 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.6x exposure=1.00
2026-06-20 04:10:14,914 - INFO - 💤 Sleeping 291s until next 5m candle close
2026-06-20 04:15:07,434 - INFO - Position | FLAT                                                                         
2026-06-20 04:15:08,059 - ERROR - ❌ Error fetching historical prices for BTCUSDT: APIError(code=-1003): Way too many req
uests; IP(15.158.219.74) banned until 1781922224652. Please use the websocket for live updates to avoid bans.
2026-06-20 04:15:08,060 - INFO - ❌ Insufficient strategic data, skipping                                                
2026-06-20 04:15:08,061 - INFO - 💤 Sleeping 297s until next 5m candle close
2026-06-20 04:20:06,594 - INFO - Position | FLAT                                                                         
2026-06-20 04:20:15,767 - INFO - Tactical | signal=HOLD pred=0.000800 min=-0.015887 max=0.004699
2026-06-20 04:20:15,769 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.7x exposure=1.00
2026-06-20 04:20:16,043 - INFO - 💤 Sleeping 289s until next 5m candle close
2026-06-20 04:25:06,563 - INFO - Position | FLAT                                                                         
2026-06-20 04:25:14,730 - INFO - Tactical | signal=HOLD pred=-0.001083 min=-0.015887 max=0.004699
2026-06-20 04:25:14,732 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.6x exposure=1.00
2026-06-20 04:25:15,020 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 04:30:06,493 - INFO - Position | FLAT                                                                         
2026-06-20 04:30:14,775 - INFO - Tactical | signal=HOLD pred=0.000594 min=-0.015887 max=0.004699
2026-06-20 04:30:14,776 - INFO - Strategic | allow=True regime=trend vol=low leverage=4.6x exposure=1.00
2026-06-20 04:30:15,045 - INFO - 💤 Sleeping 290s until next 5m candle close
2026-06-20 04:35:06,596 - INFO - Position | FLAT                          