import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def sample_ohlcv():
    idx = pd.date_range("2024-01-01", periods=300, freq="5min")
    rng = np.random.default_rng(42)
    close = 40000 + np.cumsum(rng.normal(0, 50, 300))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, 300),
        },
        index=idx,
    )


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_cash.return_value = 1000.0
    broker.get_position.return_value = None
    broker.open_position_with_bracket.return_value = MagicMock(
        success=True, data={"entry_price": 40000.0}
    )
    return broker
