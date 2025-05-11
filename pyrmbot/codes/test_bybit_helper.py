import pytest
from bybit_trading_enchanced import BybitHelper, load_config


@pytest.fixture
def helper():
    config = load_config()
    return BybitHelper(config)


def test_initialization(helper):
    assert helper.session is not None
    assert helper.exchange is not None
    assert helper.ws is not None
    assert helper.diagnose_connection()


def test_fetch_balance(helper):
    balance = helper.fetch_balance()
    assert isinstance(balance, dict)


def test_fetch_ohlcv(helper):
    ohlcv = helper.fetch_ohlcv("5m", limit=10)
    assert isinstance(ohlcv, list)
    assert len(ohlcv) <= 10


def test_indicators(helper):
    ohlcv = helper.fetch_ohlcv("5m", limit=50)
    if ohlcv:
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df_ind = helper.calculate_indicators(df)
        assert "evt_trend_7" in df_ind.columns
        assert "ATRr_14" in df_ind.columns
