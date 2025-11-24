import matplotlib.pyplot as plt

def _print_single_model_metrics(model_name, metrics, avg_train_time):
    """
    Print metrics for a single model with explanations and improvement suggestions.
    """
    print("\n" + "="*70)
    print(f"📊 {model_name} METRICS")
    print("="*70)
    
    # Final Wallet
    final_wallet = metrics['final_wallet']
    print(f"\n💰 Final Wallet: {final_wallet:.4f}")
    print(f"   └─ Explanation: Started with 1.0, ended with {final_wallet:.4f}")
    if final_wallet > 1.0:
        roi = (final_wallet - 1.0) * 100
        print(f"   └─ ✅ Profit: +{roi:.2f}% ROI")
        print(f"   └─ Suggestion: Consider increasing stake size or holding duration")
    elif final_wallet < 1.0:
        loss = (1.0 - final_wallet) * 100
        print(f"   └─ ❌ Loss: -{loss:.2f}%")
        print(f"   └─ Suggestion: Tighten stop-loss, adjust entry thresholds, or retrain more frequently")
    else:
        print(f"   └─ ⚠️  Break-even")
        print(f"   └─ Suggestion: Strategy needs significant adjustments")
    
    # Sharpe Ratio
    sharpe = metrics['sharpe_ratio_hourly']
    print(f"\n📈 Sharpe Ratio (Hourly): {sharpe:.4f}")
    print(f"   └─ Explanation: Risk-adjusted returns (higher is better)")
    if sharpe > 2.0:
        print(f"   └─ ✅ Excellent: Very strong risk-adjusted performance")
        print(f"   └─ Suggestion: Consider scaling up capital allocation")
    elif sharpe > 1.0:
        print(f"   └─ ✅ Good: Acceptable risk-adjusted returns")
        print(f"   └─ Suggestion: Optimize position sizing for better consistency")
    elif sharpe > 0:
        print(f"   └─ ⚠️  Fair: Barely beating risk-free rate")
        print(f"   └─ Suggestion: Improve entry/exit signals, reduce losing trades")
    else:
        print(f"   └─ ❌ Poor: Negative risk-adjusted returns")
        print(f"   └─ Suggestion: Strategy is taking too much risk for returns - major revision needed")
    
    # Max Drawdown
    max_dd = metrics['max_drawdown']
    print(f"\n📉 Max Drawdown: {max_dd:.2%}")
    print(f"   └─ Explanation: Largest peak-to-trough decline in wallet value")
    if max_dd > -0.05:
        print(f"   └─ ✅ Excellent: Very controlled risk (< 5%)")
        print(f"   └─ Suggestion: Strategy is conservative - could increase position size")
    elif max_dd > -0.15:
        print(f"   └─ ✅ Good: Acceptable drawdown (5-15%)")
        print(f"   └─ Suggestion: Monitor closely during volatile periods")
    elif max_dd > -0.30:
        print(f"   └─ ⚠️  Warning: High drawdown (15-30%)")
        print(f"   └─ Suggestion: Implement stricter stop-losses or reduce leverage")
    else:
        print(f"   └─ ❌ Critical: Severe drawdown (>30%)")
        print(f"   └─ Suggestion: Emergency - reduce position size immediately, review risk management")
    
    # Trade Counts
    num_long = metrics['num_long_trades']
    num_short = metrics['num_short_trades']
    total_trades = num_long + num_short
    print(f"\n🔢 Trade Statistics:")
    print(f"   ├─ Long Trades: {num_long}")
    print(f"   ├─ Short Trades: {num_short}")
    print(f"   └─ Total Trades: {total_trades}")
    if total_trades < 10:
        print(f"   └─ ⚠️  Very few trades - not enough data for statistical significance")
        print(f"   └─ Suggestion: Lower entry thresholds or increase data period")
    elif total_trades < 30:
        print(f"   └─ ⚠️  Low sample size - results may not be reliable")
        print(f"   └─ Suggestion: Collect more trades before drawing conclusions")
    else:
        print(f"   └─ ✅ Good sample size for analysis")
    
    # Short/Long Ratio
    sl_ratio = metrics['short_long_ratio']
    print(f"\n⚖️  Short/Long Ratio: {sl_ratio:.2f}")
    print(f"   └─ Explanation: Balance between short and long positions")
    if sl_ratio > 2.0:
        print(f"   └─ ⚠️  Heavily biased towards shorts")
        print(f"   └─ Suggestion: Check if model is overfitting to bearish trends")
    elif sl_ratio < 0.5:
        print(f"   └─ ⚠️  Heavily biased towards longs")
        print(f"   └─ Suggestion: Ensure model captures both market directions")
    else:
        print(f"   └─ ✅ Well-balanced strategy")
        print(f"   └─ Suggestion: Strategy adapts to market conditions effectively")
    
    # Average Returns per Trade Type
    avg_long_ret = metrics['avg_long_return_per_trade']
    avg_short_ret = metrics['avg_short_return_per_trade']
    print(f"\n💹 Average Returns per Trade:")
    print(f"   ├─ Long Trades: {avg_long_ret:.2%}")
    if avg_long_ret > 0.02:
        print(f"   │  └─ ✅ Strong long performance")
    elif avg_long_ret > 0:
        print(f"   │  └─ ⚠️  Marginal long performance - consider tighter entry criteria")
    else:
        print(f"   │  └─ ❌ Losing on longs - avoid long positions or fix entry logic")
    
    print(f"   └─ Short Trades: {avg_short_ret:.2%}")
    if avg_short_ret > 0.02:
        print(f"      └─ ✅ Strong short performance")
    elif avg_short_ret > 0:
        print(f"      └─ ⚠️  Marginal short performance - consider tighter entry criteria")
    else:
        print(f"      └─ ❌ Losing on shorts - avoid short positions or fix entry logic")
    
    # Percent Profitable Trades
    pct_profitable = metrics['percent_profitable_trades']
    print(f"\n🎯 Percent Profitable Trades: {pct_profitable:.2f}%")
    print(f"   └─ Explanation: Win rate of closed trades")
    if pct_profitable > 60:
        print(f"   └─ ✅ Excellent: High win rate")
        print(f"   └─ Suggestion: Consider increasing position size per trade")
    elif pct_profitable > 50:
        print(f"   └─ ✅ Good: Above 50% win rate")
        print(f"   └─ Suggestion: Focus on letting winners run longer")
    elif pct_profitable > 40:
        print(f"   └─ ⚠️  Below average: Strategy needs improvement")
        print(f"   └─ Suggestion: Improve entry signals or adjust stop-loss levels")
    else:
        print(f"   └─ ❌ Poor: Most trades are losing")
        print(f"   └─ Suggestion: Complete strategy overhaul needed - check feature engineering")
    
    # Training Time
    print(f"\n⏱️  Avg Train Time: {avg_train_time:.3f}s")
    print(f"   └─ Explanation: Average time to retrain model")
    if avg_train_time < 1.0:
        print(f"   └─ ✅ Fast: Good for high-frequency retraining")
    elif avg_train_time < 5.0:
        print(f"   └─ ✅ Acceptable: Reasonable for adaptive strategy")
    else:
        print(f"   └─ ⚠️  Slow: May impact real-time trading")
        print(f"   └─ Suggestion: Reduce n_estimators or training window size")

def print_metrics(metrics_xgb, metrics_cat, res_xgb, res_cat):
    """
    Display comprehensive metrics comparison for both models with insights.
    """
    _print_single_model_metrics("XGBOOST", metrics_xgb, res_xgb['train_time_s'].mean())
    _print_single_model_metrics("CATBOOST", metrics_cat, res_cat['train_time_s'].mean())
    
    # Comparative Summary
    print("\n" + "="*70)
    print("🏆 COMPARATIVE SUMMARY")
    print("="*70)
    
    # Determine winner
    xgb_wallet = metrics_xgb['final_wallet']
    cat_wallet = metrics_cat['final_wallet']
    
    if xgb_wallet > cat_wallet:
        diff = ((xgb_wallet - cat_wallet) / cat_wallet) * 100
        print(f"✅ Winner: XGBoost (outperformed by {diff:.2f}%)")
    elif cat_wallet > xgb_wallet:
        diff = ((cat_wallet - xgb_wallet) / xgb_wallet) * 100
        print(f"✅ Winner: CatBoost (outperformed by {diff:.2f}%)")
    else:
        print(f"⚖️  Tie: Both models performed equally")
    
    # Best metrics comparison
    print(f"\n📊 Best Metrics:")
    print(f"   ├─ Higher Sharpe: {'XGBoost' if metrics_xgb['sharpe_ratio_hourly'] > metrics_cat['sharpe_ratio_hourly'] else 'CatBoost'}")
    print(f"   ├─ Lower Drawdown: {'XGBoost' if metrics_xgb['max_drawdown'] > metrics_cat['max_drawdown'] else 'CatBoost'}")
    print(f"   ├─ Higher Win Rate: {'XGBoost' if metrics_xgb['percent_profitable_trades'] > metrics_cat['percent_profitable_trades'] else 'CatBoost'}")
    print(f"   └─ Faster Training: {'XGBoost' if res_xgb['train_time_s'].mean() < res_cat['train_time_s'].mean() else 'CatBoost'}")
    
    print("\n💡 General Recommendations:")
    if xgb_wallet > 1.0 or cat_wallet > 1.0:
        print("   ✅ Strategy shows promise - consider paper trading")
        print("   ✅ Optimize hyperparameters (stake size, stop-loss, max_hold)")
        print("   ✅ Test on different market conditions (bull/bear/sideways)")
    else:
        print("   ⚠️  Strategy needs major improvements before live trading")
        print("   ⚠️  Review feature engineering and label generation")
        print("   ⚠️  Consider different prediction horizons (H parameter)")
    
    print("="*70 + "\n")

def plot_results(df_xgb, df_cat):
    """
    Plot 2-panel comparison between XGBoost and CatBoost models.
    Top panel: Candlestick chart with trade entry/exit markers
    Bottom panel: Wallet performance comparison
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
    
    # ========================================
    # TOP PANEL: Candlestick chart with trade markers
    # ========================================
    df_plot = df_xgb[['open', 'high', 'low', 'close']].copy()
    
    # Draw candlesticks (sampled for performance)
    sample_rate = max(1, len(df_plot)//500)
    for idx in range(0, len(df_plot), sample_rate):
        date = df_plot.index[idx]
        open_price = df_plot['open'].iloc[idx]
        high_price = df_plot['high'].iloc[idx]
        low_price = df_plot['low'].iloc[idx]
        close_price = df_plot['close'].iloc[idx]
        
        color = 'green' if close_price >= open_price else 'red'
        
        # High-Low line (wick)
        ax1.plot([date, date], [low_price, high_price], color='black', linewidth=0.5, alpha=0.5)
        # Open-Close body
        ax1.plot([date, date], [open_price, close_price], color=color, linewidth=2, alpha=0.8)
    
    # Add trade markers for XGBoost
    markers_xgb = df_xgb.attrs.get('trade_markers', [])
    for marker in markers_xgb:
        if marker['type'] == 'entry':
            if marker['position'] == 'long':
                ax1.scatter(marker['timestamp'], marker['price'], marker='^', s=100, 
                           color='lime', edgecolors='darkgreen', linewidth=1.5, 
                           zorder=5, label='Long Entry' if marker == markers_xgb[0] else '')
            else:
                ax1.scatter(marker['timestamp'], marker['price'], marker='v', s=100, 
                           color='red', edgecolors='darkred', linewidth=1.5, 
                           zorder=5, label='Short Entry' if marker == markers_xgb[0] else '')
        elif marker['type'] == 'exit':
            color = 'cyan' if marker['profit'] else 'orange'
            ax1.scatter(marker['timestamp'], marker['price'], marker='x', s=100, 
                       color=color, linewidth=2, zorder=5, 
                       label='Exit (Profit)' if marker['profit'] and marker == markers_xgb[0] else 
                             ('Exit (Loss)' if not marker['profit'] and marker == markers_xgb[0] else ''))
    
    ax1.set_title('BTC/USDT Price with XGBoost Trade Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    
    # ========================================
    # BOTTOM PANEL: Wallet performance
    # ========================================
    ax2.plot(df_xgb.index, df_xgb['wallet'], label='XGBoost Wallet', 
             linewidth=2, color='blue', alpha=0.8)
    ax2.plot(df_cat.index, df_cat['wallet'], label='CatBoost Wallet', 
             linewidth=2, color='orange', alpha=0.8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Wallet')
    
    # Fill area between wallet and initial value to show profit/loss
    ax2.fill_between(df_xgb.index, df_xgb['wallet'], 1.0, 
                      where=(df_xgb['wallet'] >= 1.0), alpha=0.2, color='green', 
                      interpolate=True)
    ax2.fill_between(df_xgb.index, df_xgb['wallet'], 1.0, 
                      where=(df_xgb['wallet'] < 1.0), alpha=0.2, color='red', 
                      interpolate=True)

    ax2.set_title('Wallet Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Wallet Value', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()