"""
Simulation / Backtest engine for Dual-ML Bitcoin Trading Bot.
MockBroker executes trades on validation data with fees, slippage, and realistic constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from config import (
    INITIAL_EQUITY, FEE, SLIPPAGE,
    ABSOLUTE_THRESHOLD,
    STAKE_LONG_FRAC_DEFAULT, STAKE_SHORT_FRAC_DEFAULT,
    STOP_LOSS_FRAC_DEFAULT, TAKE_PROFIT_FRAC_DEFAULT,
    MAX_HOLD_HOURS_DEFAULT, LEVERAGE_DEFAULT,
)
from logger import log_trade_entry, log_trade_exit, log_equity, log_info, log_debug


# ── MockBroker ──────────────────────────────────────────────────────────
class MockBroker:
    """
    Simulated broker for backtesting.
    Executes trades at candle close prices with fee + slippage.
    Tracks equity, position, and trade history.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fee: float = FEE,
        slippage: float = SLIPPAGE,
        initial_equity: float = INITIAL_EQUITY,
    ):
        self.df = df
        self.fee = fee
        self.slippage = slippage
        self.initial_equity = initial_equity

        # State
        self.equity = initial_equity
        self.position = 0.0        # Base currency qty (positive=long, negative=short)
        self.entry_price = 0.0
        self.entry_time: Optional[datetime] = None
        self.entry_equity = initial_equity

        # Trade tracking
        self.current_trade: Optional[Dict] = None
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # Current meta-params (from strategic ML)
        self.current_meta = {
            "stake_long_frac": STAKE_LONG_FRAC_DEFAULT,
            "stake_short_frac": STAKE_SHORT_FRAC_DEFAULT,
            "stop_loss_frac": STOP_LOSS_FRAC_DEFAULT,
            "take_profit_frac": TAKE_PROFIT_FRAC_DEFAULT,
            "max_hold_hours": MAX_HOLD_HOURS_DEFAULT,
            "recommended_leverage": LEVERAGE_DEFAULT,
            "regime": "trend",
        }

    def _get_price(self, idx: int, side: str) -> float:
        """Get execution price with slippage."""
        close = self.df.iloc[idx]['close']
        if side == "long":
            # Buy at ask (higher)
            return close * (1 + self.slippage)
        else:
            # Sell at bid (lower)
            return close * (1 - self.slippage)

    def _calculate_qty(self, price: float, stake_frac: float, leverage: float) -> float:
        """Calculate position size in base currency."""
        stake = self.equity * stake_frac * leverage
        return stake / price

    def _check_exit_conditions(self, idx: int, current_price: float) -> Optional[str]:
        """Check if position should be closed. Returns exit reason or None."""
        if self.position == 0:
            return None

        # Time-based exit
        if self.entry_time is not None:
            hours_held = (self.df.index[idx] - self.entry_time).total_seconds() / 3600
            if hours_held >= self.current_meta["max_hold_hours"]:
                return "max_hold"

        # Stop loss / Take profit
        if self.position > 0:  # Long
            # SL: price dropped below entry * (1 - sl)
            sl_price = self.entry_price * (1 - self.current_meta["stop_loss_frac"])
            if current_price <= sl_price:
                return "sl"
            # TP: price rose above entry * (1 + tp)
            tp_price = self.entry_price * (1 + self.current_meta["take_profit_frac"])
            if current_price >= tp_price:
                return "tp"
        else:  # Short
            # SL: price rose above entry * (1 + sl)
            sl_price = self.entry_price * (1 + self.current_meta["stop_loss_frac"])
            if current_price >= sl_price:
                return "sl"
            # TP: price dropped below entry * (1 - tp)
            tp_price = self.entry_price * (1 - self.current_meta["take_profit_frac"])
            if current_price <= tp_price:
                return "tp"

        return None

    def update_meta(self, meta_params: Dict) -> None:
        """Update meta-parameters from strategic ML."""
        self.current_meta.update(meta_params)

    def step(
        self,
        idx: int,
        signal: str,          # "long", "short", "hold"
        meta_params: Dict = None,
    ) -> Dict[str, Any]:
        """
        Execute one simulation step at candle idx.
        Returns dict with step info.
        """
        timestamp = self.df.index[idx]
        close_price = self.df.iloc[idx]['close']

        # Update meta-params if provided
        if meta_params:
            self.update_meta(meta_params)

        # Log equity curve point
        unrealized_pnl = 0.0
        if self.position != 0:
            if self.position > 0:
                unrealized_pnl = (close_price - self.entry_price) * self.position
            else:
                unrealized_pnl = (self.entry_price - close_price) * abs(self.position)

        log_equity(timestamp, self.equity, self.position, unrealized_pnl)

        # Check exit conditions first
        exit_reason = self._check_exit_conditions(idx, close_price)

        # Execute exit if needed
        if exit_reason and self.position != 0:
            self._execute_exit(idx, exit_reason, close_price)

        # Execute entry if signal and no position
        if signal in ("long", "short") and self.position == 0:
            self._execute_entry(idx, signal, close_price)

        # Signal reversal: close and flip
        elif signal == "long" and self.position < 0:
            self._execute_exit(idx, "reversal", close_price)
            self._execute_entry(idx, "long", close_price)
        elif signal == "short" and self.position > 0:
            self._execute_exit(idx, "reversal", close_price)
            self._execute_entry(idx, "short", close_price)

        return {
            "timestamp": timestamp,
            "equity": self.equity,
            "position": self.position,
            "entry_price": self.entry_price if self.position != 0 else 0,
            "unrealized_pnl": unrealized_pnl,
        }

    def _execute_entry(self, idx: int, side: str, close_price: float) -> None:
        """Open a new position."""
        exec_price = self._get_price(idx, side)
        stake_frac = (
            self.current_meta["stake_long_frac"]
            if side == "long"
            else self.current_meta["stake_short_frac"]
        )
        leverage = self.current_meta["recommended_leverage"]

        qty = self._calculate_qty(exec_price, stake_frac, leverage)

        # Apply fee
        fee_paid = exec_price * qty * self.fee
        self.equity -= fee_paid

        # Slippage cost (already in exec_price, but track separately)
        slippage_paid = abs(exec_price - close_price) * qty

        # Update state
        self.position = qty if side == "long" else -qty
        self.entry_price = exec_price
        self.entry_time = self.df.index[idx]
        self.entry_equity = self.equity

        # Create trade record
        self.current_trade = log_trade_entry(
            timestamp=self.df.index[idx],
            symbol="BTCUSDT",
            side=side,
            entry_price=exec_price,
            qty=qty,
            stake_frac=stake_frac,
            leverage=leverage,
            stop_loss=self.current_meta["stop_loss_frac"],
            take_profit=self.current_meta["take_profit_frac"],
            max_hold_hours=self.current_meta["max_hold_hours"],
            regime=self.current_meta["regime"],
            tactical_pred=0.0,  # Will be filled by caller
            strategic_params=self.current_meta.copy(),
            equity_before=self.entry_equity,
        )

        log_debug(f"ENTRY {side.upper()} @ {exec_price:.2f} | Qty: {qty:.6f} | Equity: {self.equity:.6f}")

    def _execute_exit(self, idx: int, reason: str, close_price: float) -> None:
        """Close current position."""
        if self.position == 0 or self.current_trade is None:
            return

        side = "long" if self.position > 0 else "short"
        exec_price = self._get_price(idx, side)
        qty = abs(self.position)

        # Apply fee
        fee_paid = exec_price * qty * self.fee

        # Calculate PnL
        if side == "long":
            gross_pnl = (exec_price - self.entry_price) * qty
        else:
            gross_pnl = (self.entry_price - exec_price) * qty

        net_pnl = gross_pnl - fee_paid
        pnl_pct = net_pnl / self.entry_equity if self.entry_equity > 0 else 0

        self.equity += net_pnl

        # Slippage
        slippage_paid = abs(exec_price - close_price) * qty

        # Log exit
        log_trade_exit(
            trade=self.current_trade,
            exit_price=exec_price,
            exit_reason=reason,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            equity_after=self.equity,
            fee_paid=fee_paid,
            slippage_paid=slippage_paid,
        )

        # Archive trade
        self.trades.append(self.current_trade.copy())
        self.current_trade = None

        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = None

        log_debug(f"EXIT {reason.upper()} @ {exec_price:.2f} | PnL: {net_pnl:.6f} ({pnl_pct:.2%}) | Equity: {self.equity:.6f}")

    def close_all(self, idx: int = -1) -> None:
        """Force close any open position at end of simulation."""
        if self.position != 0:
            if idx == -1:
                idx = len(self.df) - 1
            self._execute_exit(idx, "end_of_data", self.df.iloc[idx]['close'])

    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "side", "entry_price", "exit_price",
                "qty", "pnl", "pnl_pct", "exit_reason", "regime"
            ])
        return pd.DataFrame(self.trades)

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity", "position", "unrealized_pnl"])
        return pd.DataFrame(self.equity_curve)

    def get_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "num_trades": 0,
                "avg_trade_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        trades_df = self.get_trades_df()
        equity_df = self.get_equity_curve_df()

        # Total return
        total_return = self.equity - self.initial_equity
        total_return_pct = total_return / self.initial_equity

        # Sharpe (assuming daily returns from equity curve)
        if len(equity_df) > 1:
            equity_df = equity_df.set_index('timestamp')
            daily_returns = equity_df['equity'].resample('D').last().pct_change().dropna()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        equity_series = equity_df['equity'] if len(equity_df) > 0 else pd.Series([self.initial_equity])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_dd = drawdown.min()
        max_dd_pct = abs(max_dd)

        # Win rate
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0

        # Profit factor
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            "total_return": round(total_return, 6),
            "total_return_pct": round(total_return_pct, 6),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 6),
            "max_drawdown_pct": round(max_dd_pct, 6),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "num_trades": len(trades_df),
            "avg_trade_pnl": round(trades_df['pnl'].mean(), 6),
            "avg_win": round(wins['pnl'].mean(), 6) if len(wins) > 0 else 0.0,
            "avg_loss": round(losses['pnl'].mean(), 6) if len(losses) > 0 else 0.0,
        }


# ── Run Simulation ──────────────────────────────────────────────────────
def run_simulation(
    df_val: pd.DataFrame,
    tactical_preds: pd.Series,
    strategic_meta_params: List[Dict],
    config: Any = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Run full simulation on validation data.

    Args:
        df_val: Validation DataFrame with features + prices
        tactical_preds: Tactical ML predictions (aligned with df_val)
        strategic_meta_params: List of meta-param dicts per row (from strategic ML)
        config: Optional config object for thresholds

    Returns:
        (trades_df, metrics_dict, equity_curve_df)
    """
    log_info("=" * 60)
    log_info("Starting simulation on validation data")
    log_info(f"  Period: {df_val.index[0]} to {df_val.index[-1]}")
    log_info(f"  Candles: {len(df_val)}")
    log_info("=" * 60)

    # Align predictions with validation data
    # tactical_preds may have NaN for warmup period
    min_len = min(len(df_val), len(tactical_preds), len(strategic_meta_params))
    df_val = df_val.iloc[-min_len:].copy() if min_len < len(df_val) else df_val
    tactical_preds = tactical_preds.iloc[-min_len:] if min_len < len(tactical_preds) else tactical_preds
    strategic_meta_params = strategic_meta_params[-min_len:] if min_len < len(strategic_meta_params) else strategic_meta_params

    # Initialize broker
    broker = MockBroker(
        df=df_val,
        fee=FEE,
        slippage=SLIPPAGE,
        initial_equity=INITIAL_EQUITY,
    )

    # Get threshold
    threshold = getattr(config, 'ABSOLUTE_THRESHOLD', ABSOLUTE_THRESHOLD) if config else ABSOLUTE_THRESHOLD

    # Run simulation
    for i in range(len(df_val)):
        # Get tactical prediction
        pred = tactical_preds.iloc[i] if i < len(tactical_preds) else 0.0

        # Convert prediction to signal
        if np.isnan(pred):
            signal = "hold"
        elif pred > threshold:
            signal = "long"
        elif pred < -threshold:
            signal = "short"
        else:
            signal = "hold"

        # Regime gate: skip entries in choppy markets (only trend/high_vol)
        row_regime = df_val.iloc[i].get("regime", "trend")
        if signal in ("long", "short") and row_regime == "chop":
            signal = "hold"

        # Get strategic meta-params
        meta = strategic_meta_params[i] if i < len(strategic_meta_params) else {}

        # Update current trade's tactical_pred if entering
        if broker.current_trade is not None and 'tactical_pred' in broker.current_trade:
            broker.current_trade['tactical_pred'] = round(float(pred), 6)

        # Step broker
        broker.step(i, signal, meta)

    # Close any open position at end
    broker.close_all()

    # Get results
    trades_df = broker.get_trades_df()
    equity_curve_df = broker.get_equity_curve_df()
    metrics = broker.get_metrics()

    # Log summary
    log_info("=" * 60)
    log_info("SIMULATION COMPLETE")
    log_info(f"  Trades:     {metrics['num_trades']}")
    log_info(f"  Return:     {metrics['total_return_pct']:.2%}")
    log_info(f"  Sharpe:     {metrics['sharpe']:.2f}")
    log_info(f"  Max DD:     {metrics['max_drawdown_pct']:.2%}")
    log_info(f"  Win Rate:   {metrics['win_rate']:.1%}")
    log_info(f"  Profit Fac: {metrics['profit_factor']:.2f}")
    log_info(f"  Final Eq:   {broker.equity:.6f}")
    log_info("=" * 60)

    return trades_df, metrics, equity_curve_df


# ── Quick Simulation (for testing) ─────────────────────────────────────
def quick_simulate(
    df_val: pd.DataFrame,
    predictions: pd.Series,
    threshold: float = ABSOLUTE_THRESHOLD,
) -> Dict[str, float]:
    """
    Quick simulation with fixed params (no strategic ML).
    Useful for testing tactical model alone.
    """
    broker = MockBroker(df=df_val)
    broker.current_meta = {
        "stake_long_frac": STAKE_LONG_FRAC_DEFAULT,
        "stake_short_frac": STAKE_SHORT_FRAC_DEFAULT,
        "stop_loss_frac": STOP_LOSS_FRAC_DEFAULT,
        "take_profit_frac": TAKE_PROFIT_FRAC_DEFAULT,
        "max_hold_hours": MAX_HOLD_HOURS_DEFAULT,
        "recommended_leverage": LEVERAGE_DEFAULT,
        "regime": "trend",
    }

    for i in range(len(df_val)):
        pred = predictions.iloc[i] if i < len(predictions) else 0.0
        if np.isnan(pred):
            signal = "hold"
        elif pred > threshold:
            signal = "long"
        elif pred < -threshold:
            signal = "short"
        else:
            signal = "hold"
        broker.step(i, signal)

    broker.close_all()
    return broker.get_metrics()