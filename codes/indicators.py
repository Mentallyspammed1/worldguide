# indicators.py - Placeholder for Technical Indicator Functions
# Implement your indicator logic here.

import numpy as np  # Often needed for calculations
import pandas as pd


def rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    if not isinstance(close_prices, pd.Series):
        raise TypeError("close_prices must be a pandas Series")
    if close_prices.isnull().any():
        # Handle NaNs if necessary, e.g., dropna or fillna, or return NaNs
        # For simplicity, we'll proceed, but calculations might yield NaNs
        pass
    if len(close_prices) < period:
        return pd.Series([np.nan] * len(close_prices), index=close_prices.index)  # Not enough data

    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0.0).fillna(0)
    loss = -delta.where(delta < 0, 0.0).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use Exponential Moving Average (EMA) for subsequent calculations - more common for RSI
    # For the first value, use the simple mean calculated above
    # Subsequent values use EMA formula: EMA = (Current Value * alpha) + (Previous EMA * (1 - alpha))
    # alpha = 1 / period
    alpha = 1.0 / period
    for i in range(period, len(close_prices)):
        avg_gain[i] = (gain[i] * alpha) + (avg_gain[i - 1] * (1 - alpha))
        avg_loss[i] = (loss[i] * alpha) + (avg_loss[i - 1] * (1 - alpha))

    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    rsi_values.fillna(50, inplace=True)  # Fill initial NaNs, often RSI starts around 50 conceptually
    # Ensure the final series has the same index as the input
    return rsi_values.reindex(close_prices.index)


def ATR(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Average True Range (ATR)."""
    if not all(isinstance(s, pd.Series) for s in [high_prices, low_prices, close_prices]):
        raise TypeError("Inputs must be pandas Series")
    if high_prices.isnull().any() or low_prices.isnull().any() or close_prices.isnull().any():
        # Handle NaNs if necessary
        pass
    if len(high_prices) < period:
        return pd.Series([np.nan] * len(high_prices), index=high_prices.index)  # Not enough data

    high_low = high_prices - low_prices
    high_close_prev = abs(high_prices - close_prices.shift(1))
    low_close_prev = abs(low_prices - close_prices.shift(1))

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # Use Simple Moving Average (SMA) for ATR calculation typically
    atr_values = tr.rolling(window=period, min_periods=period).mean()

    # Ensure the final series has the same index as the input
    return atr_values.reindex(high_prices.index)


def FibonacciPivotPoints(high: float, low: float, close: float) -> dict:
    """Calculates Fibonacci Pivot Points."""
    if not all(isinstance(v, (int, float)) for v in [high, low, close]):
        raise TypeError("Inputs must be numeric (int or float)")

    pivot = (high + low + close) / 3.0
    fib_range = high - low

    pivots = {
        "Pivot": pivot,
        "R1": pivot + (0.382 * fib_range),
        "R2": pivot + (0.618 * fib_range),
        "R3": pivot + (1.000 * fib_range),
        "S1": pivot - (0.382 * fib_range),
        "S2": pivot - (0.618 * fib_range),
        "S3": pivot - (1.000 * fib_range),
    }
    return pivots
