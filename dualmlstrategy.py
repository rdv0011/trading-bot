from basestrategy import BaseStrategy
import traceback
from dataclasses import replace
from binancebasebroker import SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_SHORT, MARKET_TYPE_SPOT
from mlio import MODEL_DIR
from mltrainingcore import make_features, make_labels, get_features
from timeframe_config import TIMEFRAMES
from tactical.tacticalml import TacticalML
from strategic.strategicml import StrategicML
from positionmanager import PositionManager
from riskguard import RiskGuard

STRATEGIC_HISTORY_CANDLES = 300
TACTICAL_HISTORY_CANDLES_MULTIPLIER = 2


class MissingHistoricalDataError(Exception):
    pass


class DualMLStrategy(BaseStrategy):

    def initialize(self):
        self.asset = self.parameters.get("asset_symbol", "BTC")
        self.market_type = self.parameters.get("market_type", "futures")

        self.tf_cfg_tactical = TIMEFRAMES[self.parameters.get("tactical_timeframe", "5m")]
        self.tf_cfg_strategic = TIMEFRAMES[self.parameters.get("strategic_timeframe", "1h")]

        tactical_init_length = self.compute_required_history(self.tf_cfg_tactical)

        self.log_message(
            f"⏱ DualMLStrategy | tactical={self.tf_cfg_tactical.name} "
            f"strategic={self.tf_cfg_strategic.name}"
        )

        df_hist = self.get_historical_prices(
            self.asset, tactical_init_length, self.tf_cfg_tactical.name
        )
        if df_hist is None or len(df_hist) == 0:
            raise MissingHistoricalDataError(
                f"No historical data for {self.asset} @ {self.tf_cfg_tactical.name}"
            )

        df_hist = make_features(df_hist, self.tf_cfg_tactical)
        df_hist = make_labels(df_hist, self.tf_cfg_tactical)
        features = get_features(df_hist)

        model_params = self.parameters.get("model_params", {"iterations": 300, "verbose": False})

        self.tactical_ml = TacticalML(
            model_params=model_params,
            tf_cfg=self.tf_cfg_tactical,
            logger=self.log_message,
        )
        self.tactical_ml.warmup(df_hist, features)

        model_dir = self.parameters.get("model_dir", MODEL_DIR)
        self.strategic_ml = StrategicML(
            model_dir=model_dir,
            tf_cfg=self.tf_cfg_strategic,
            logger=self.log_message,
        )

        if not self.strategic_ml.is_ready:
            self.log_message(
                "⚠️ StrategicML has no trained model yet. "
                "Run: python strategic/strategictraining.py"
            )

        self.position_manager = PositionManager(
            broker=self._broker,
            asset=self.asset,
            quote_symbol=self.quote_asset_symbol,
            logger=self.log_message,
        )

        self.risk_guard = RiskGuard(
            max_daily_loss_frac=self.parameters.get("max_daily_loss_frac", 0.05),
            max_drawdown_frac=self.parameters.get("max_drawdown_frac", 0.15),
            max_leverage=self.parameters.get("max_leverage", 10.0),
        )

        self.log_message("✅ DualMLStrategy initialized")

    def on_trading_iteration(self):
        current_equity = self.get_cash()
        if not self.risk_guard.update(current_equity):
            self.log_message(f"🛑 RiskGuard halted trading — equity={current_equity:.2f}")
            if self.position_manager.has_position:
                self.position_manager.emergency_close_live()
            return

        pos = self.get_position(self.asset)
        lev = self._broker.get_position_leverage(self._pair_asset_symbol(self.asset))
        lev_str = f" | leverage={lev}x" if lev is not None else ""
        if pos is not None and pos.amount is not None:
            direction = "LONG" if pos.amount > 0 else "SHORT"
            self.log_message(
                f"Position | {direction} {abs(pos.amount)} {self.asset} @ {pos.entry_price:.2f}{lev_str}"
            )
        else:
            self.log_message(f"Position | FLAT{lev_str}")

        df_tactical = self.get_historical_prices(
            self.asset,
            self.tf_cfg_tactical.max_history_candles,
            self.tf_cfg_tactical.name,
        )
        if df_tactical is None or len(df_tactical) < self.tf_cfg_tactical.min_feature_candles:
            self.log_message("❌ Insufficient tactical data, skipping")
            return

        df_strategic = self.get_historical_prices(
            self.asset,
            STRATEGIC_HISTORY_CANDLES,
            self.tf_cfg_strategic.name,
        )
        if df_strategic is None or len(df_strategic) < 50:
            self.log_message("❌ Insufficient strategic data, skipping")
            return

        df_tactical_raw = df_tactical
        df_tactical = make_features(df_tactical, self.tf_cfg_tactical)
        df_tactical = make_labels(df_tactical, self.tf_cfg_tactical)
        features = get_features(df_tactical)

        df_pred = make_features(df_tactical_raw, self.tf_cfg_tactical).iloc[[-1]]

        tactical_signal = self.tactical_ml.fit_and_predict(df_tactical, df_pred, features)

        strategic_decision = self.strategic_ml.predict(df_strategic)
        strategic_decision = replace(
            strategic_decision,
            recommended_leverage=self.risk_guard.clamp_leverage(strategic_decision.recommended_leverage),
        )

        if self.market_type.lower() == MARKET_TYPE_SPOT and tactical_signal.signal == SIGNAL_SHORT:
            tactical_signal = type(tactical_signal)(
                signal=SIGNAL_HOLD,
                prediction=tactical_signal.prediction,
                min_threshold=tactical_signal.min_threshold,
                max_threshold=tactical_signal.max_threshold,
            )

        self.log_message(
            f"Tactical | signal={tactical_signal.signal.upper()} "
            f"pred={tactical_signal.prediction:.6f} "
            f"min={tactical_signal.min_threshold:.6f} max={tactical_signal.max_threshold:.6f}"
        )
        self.log_message(
            f"Strategic | allow={strategic_decision.allow_trading} "
            f"regime={strategic_decision.market_regime} "
            f"vol={strategic_decision.volatility_state} "
            f"leverage={strategic_decision.recommended_leverage:.1f}x "
            f"exposure={strategic_decision.max_exposure_frac:.2f}"
        )

        current_price = self.get_last_price(self.asset)
        self.position_manager.on_signal(tactical_signal, strategic_decision, current_price)

    def on_abrupt_closing(self):
        try:
            self.log_message("⚠️ Abrupt closing — emergency position close")
            self.position_manager.emergency_close_live()
        except Exception as e:
            self.log_message(f"❌ Emergency close error: {e}")
            self.log_message(traceback.format_exc())
