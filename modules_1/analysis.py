# File: analysis.py
"""
Module for analyzing trading data, calculating technical indicators, and generating trading signals.

The TradingAnalyzer class takes historical OHLCV data, configuration, and market information
to compute various technical indicators. It then uses these indicators, along with configurable
weights, to generate BUY, SELL, or HOLD signals. It can also calculate Fibonacci levels and
suggest Take Profit/Stop Loss levels based on ATR.
"""

import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, getcontext
from typing import Any, Dict, Optional, Tuple, List, Union

# Import necessary libraries
import numpy as np
import pandas as pd

# Ensure pandas_ta is installed: pip install pandas_ta
try:
    import pandas_ta as ta
except ImportError:
    print(
        "ERROR: pandas_ta library not found. Please install it: pip install pandas_ta",
        file=sys.stderr,
    )

    # Define a dummy ta object to prevent NameError, although analysis will fail.
    class DummyTA:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                # Use standard logging here as utils might not be fully available
                logging.getLogger(__name__).error(
                    f"pandas_ta not installed. Cannot call function '{name}'."
                )
                return None  # Return None or appropriate default for the expected return type

            return method

    ta = DummyTA()


# Ensure Decimal context precision is sufficient
try:
    getcontext().prec = 38  # Set high precision for calculations
except Exception as e:
    logging.getLogger(__name__).error(f"Failed to set Decimal precision: {e}")

# Import constants and utility functions from utils
try:
    from utils import (
        CCXT_INTERVAL_MAP,
        DEFAULT_INDICATOR_PERIODS,
        FIB_LEVELS,
        get_min_tick_size,
        get_price_precision,
        NEON_RED,
        NEON_YELLOW,
        NEON_GREEN,
        RESET_ALL_STYLE,
        NEON_PURPLE,
        NEON_BLUE,
        NEON_CYAN,
        format_signal,  # Use the utility function for signal formatting
    )
except ImportError:
    print(
        "ERROR: Failed importing components from utils in analysis.py.", file=sys.stderr
    )
    # Define fallbacks for constants and functions if utils import fails
    NEON_RED = NEON_YELLOW = NEON_GREEN = RESET_ALL_STYLE = ""
    NEON_PURPLE = NEON_BLUE = NEON_CYAN = ""
    DEFAULT_INDICATOR_PERIODS = {}
    CCXT_INTERVAL_MAP = {}
    FIB_LEVELS = [
        Decimal(str(f)) for f in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    ]  # Basic Fib levels

    def get_price_precision(m, l):
        return 4  # Dummy function

    def get_min_tick_size(m, l):
        return Decimal("0.0001")  # Dummy function

    def format_signal(s, **kwargs):
        return str(s)  # Dummy function


# Define constants for frequently used indicator keys
ATR_KEY = "ATR"
EMA_SHORT_KEY = "EMA_Short"
EMA_LONG_KEY = "EMA_Long"
MOMENTUM_KEY = "Momentum"
CCI_KEY = "CCI"
WILLIAMS_R_KEY = "Williams_R"
MFI_KEY = "MFI"
VWAP_KEY = "VWAP"
PSAR_LONG_KEY = "PSAR_long"
PSAR_SHORT_KEY = "PSAR_short"
SMA10_KEY = "SMA10"
STOCHRSI_K_KEY = "StochRSI_K"
STOCHRSI_D_KEY = "StochRSI_D"
RSI_KEY = "RSI"
BB_LOWER_KEY = "BB_Lower"
BB_MIDDLE_KEY = "BB_Middle"
BB_UPPER_KEY = "BB_Upper"
VOLUME_MA_KEY = "Volume_MA"
OPEN_KEY = "Open"
HIGH_KEY = "High"
LOW_KEY = "Low"
CLOSE_KEY = "Close"
VOLUME_KEY = "Volume"

# Define which indicators should store their latest value as Decimal for precision
DECIMAL_INDICATOR_KEYS = {
    ATR_KEY,
    OPEN_KEY,
    HIGH_KEY,
    LOW_KEY,
    CLOSE_KEY,
    VOLUME_KEY,
    BB_LOWER_KEY,
    BB_MIDDLE_KEY,
    BB_UPPER_KEY,
    PSAR_LONG_KEY,
    PSAR_SHORT_KEY,
    EMA_SHORT_KEY,
    EMA_LONG_KEY,
    SMA10_KEY,
    VWAP_KEY,
    VOLUME_MA_KEY,
}


class TradingAnalyzer:
    """
    Analyzes trading data using technical indicators and generates signals.

    Takes historical OHLCV data, configuration, and market information to compute
    indicators via pandas-ta. Uses configurable weights to generate BUY/SELL/HOLD signals.
    """

    # --- Technical Indicator Configuration ---
    # Defines how each indicator is calculated, its parameters, expected output type,
    # data requirements, and how its results are integrated.
    INDICATOR_CONFIG: Dict[str, Dict[str, Any]] = {
        # --- Trend & Volatility ---
        "ATR": {
            "func_name": "atr",
            "params_map": {"length": "atr_period"},
            "main_col_pattern": "ATRr_{length}",
            "type": "decimal",
            "min_data_param_key": "length",
            "concat": False,
        },
        "EMA_Short": {
            "func_name": "ema",
            "params_map": {"length": "ema_short_period"},
            "main_col_pattern": "EMA_{length}",
            "type": "decimal",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        "EMA_Long": {
            "func_name": "ema",
            "params_map": {"length": "ema_long_period"},
            "main_col_pattern": "EMA_{length}",
            "type": "decimal",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        "PSAR": {
            "func_name": "psar",
            "params_map": {
                "initial": "psar_initial_af",
                "step": "psar_af_step",
                "max": "psar_max_af",
            },
            "multi_cols": {
                "PSAR_long": "PSARl_{initial}_{step}_{max}",
                "PSAR_short": "PSARs_{initial}_{step}_{max}",
            },
            "type": "decimal",
            "concat": True,
            "min_data": 2,
        },
        "SMA10": {
            "func_name": "sma",
            "params_map": {"length": "sma_10_window"},
            "main_col_pattern": "SMA_{length}",
            "type": "decimal",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        "Bollinger_Bands": {
            "func_name": "bbands",
            "params_map": {
                "length": "bollinger_bands_period",
                "std": "bollinger_bands_std_dev",
            },
            "multi_cols": {
                "BB_Lower": "BBL_{length}_{std:.1f}",
                "BB_Middle": "BBM_{length}_{std:.1f}",
                "BB_Upper": "BBU_{length}_{std:.1f}",
            },
            "type": "decimal",
            "concat": True,
            "min_data_param_key": "length",
        },
        "VWAP": {
            "func_name": "vwap",
            "params_map": {},
            "main_col_pattern": "VWAP_D",
            "type": "decimal",
            "concat": True,
            "min_data": 1,
        },
        # --- Momentum & Oscillators ---
        "Momentum": {
            "func_name": "mom",
            "params_map": {"length": "momentum_period"},
            "main_col_pattern": "MOM_{length}",
            "type": "float",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        "CCI": {
            "func_name": "cci",
            "params_map": {"length": "cci_window", "c": "cci_constant"},
            "main_col_pattern": "CCI_{length}_{c:.3f}",
            "type": "float",
            "min_data_param_key": "length",
            "concat": False,
        },
        "Williams_R": {
            "func_name": "willr",
            "params_map": {"length": "williams_r_window"},
            "main_col_pattern": "WILLR_{length}",
            "type": "float",
            "min_data_param_key": "length",
            "concat": False,
        },
        "StochRSI": {
            "func_name": "stochrsi",
            "params_map": {
                "length": "stoch_rsi_window",
                "rsi_length": "stoch_rsi_rsi_window",
                "k": "stoch_rsi_k",
                "d": "stoch_rsi_d",
            },
            "multi_cols": {
                "StochRSI_K": "STOCHRSIk_{length}_{rsi_length}_{k}_{d}",
                "StochRSI_D": "STOCHRSId_{length}_{rsi_length}_{k}_{d}",
            },
            "type": "float",
            "concat": True,
            "min_data_param_key": "length",
        },
        "RSI": {
            "func_name": "rsi",
            "params_map": {"length": "rsi_period"},
            "main_col_pattern": "RSI_{length}",
            "type": "float",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        # --- Volume ---
        "MFI": {
            "func_name": "mfi",
            "params_map": {"length": "mfi_window"},
            "main_col_pattern": "MFI_{length}",
            "type": "float",
            "concat": True,
            "min_data_param_key": "length",
        },
        "Volume_MA": {
            "func_name": "_calculate_volume_ma",
            "params_map": {"length": "volume_ma_period"},
            "main_col_pattern": "VOL_SMA_{length}",
            "type": "decimal",
            "min_data_param_key": "length",
            "concat": False,
        },
    }

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: Input DataFrame with OHLCV data (must contain 'open', 'high', 'low', 'close', 'volume').
                Timestamps should be in the index (timezone-naive UTC preferred).
            logger: Logger instance for logging messages.
            config: Configuration dictionary containing indicator settings, weights, etc.
            market_info: Market-specific information dictionary from CCXT.

        Raises:
            ValueError: If input DataFrame is invalid or missing required columns/config.
        """
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
        self.interval = str(config.get("interval", "5"))  # Ensure interval is string
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)

        # Initialize state variables
        self.indicator_values: Dict[
            str, Union[Decimal, float, None]
        ] = {}  # Stores latest indicator values
        self.signals: Dict[str, int] = {
            "BUY": 0,
            "SELL": 0,
            "HOLD": 1,
        }  # Default signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(
            self.active_weight_set_name, {}
        )
        self.fib_levels_data: Dict[str, Decimal] = {}  # Stores Fibonacci levels
        self.ta_column_names: Dict[
            str, str
        ] = {}  # Maps internal keys like 'EMA_Short' to actual df columns like 'EMA_9'
        self.df_calculated: pd.DataFrame = (
            pd.DataFrame()
        )  # Stores DataFrame with indicators

        # Pre-build map for indicator output types (Decimal/float)
        self.indicator_type_map: Dict[str, str] = {
            ind_key: details.get("type", "float")
            for ind_key, details in self.INDICATOR_CONFIG.items()
        }
        for details in self.INDICATOR_CONFIG.values():
            if "multi_cols" in details:
                sub_type = details.get(
                    "type", "float"
                )  # Assume multi-cols have same type as parent
                for sub_key in details["multi_cols"]:
                    self.indicator_type_map[sub_key] = sub_type

        # --- Input Validation ---
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"Input DataFrame {self.symbol} invalid/empty.")
        if not self.ccxt_interval:
            raise ValueError(f"Interval '{self.interval}' invalid.")
        if not self.weights:
            self.logger.warning(
                f"Weight set '{self.active_weight_set_name}' empty {self.symbol}."
            )
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        # --- Prepare DataFrames and Calculate ---
        self.df_original_ohlcv = df.copy()
        # Ensure index is timezone-naive (CCXT usually provides UTC timestamps)
        if self.df_original_ohlcv.index.tz is not None:
            self.df_original_ohlcv.index = self.df_original_ohlcv.index.tz_localize(
                None
            )

        self._validate_and_prepare_df_calculated()  # Prepare df_calculated with float64 types

        if (
            not self.df_calculated.empty and ta is not None
        ):  # Check if pandas_ta was imported
            self._calculate_all_indicators()
            self._update_latest_indicator_values()  # Get latest values from calculated df
            self.calculate_fibonacci_levels()  # Calculate Fib levels
        elif ta is None:
            self.logger.error(
                "pandas_ta not installed. Cannot calculate technical indicators."
            )
            self.indicator_values = {}
        else:
            self.logger.error(
                f"DataFrame prep failed {self.symbol}. No indicators calculated."
            )
            self.indicator_values = {}

    def _validate_and_prepare_df_calculated(self) -> None:
        """
        Validates OHLCV columns, converts them to float64 for TA calculations,
        and logs warnings about data length requirements.
        Sets self.df_calculated to empty DataFrame on critical failure.
        """
        self.df_calculated = self.df_original_ohlcv.copy()
        required_cols = ["open", "high", "low", "close", "volume"]
        min_required_rows = 1

        for col in required_cols:
            if col not in self.df_calculated.columns:
                self.logger.critical(
                    f"Column '{col}' missing for {self.symbol}. Invalidating DataFrame."
                )
                self.df_calculated = pd.DataFrame()
                return  # Cannot proceed

            try:
                # Convert to numeric first, coercing errors to NaN
                numeric_col = pd.to_numeric(self.df_calculated[col], errors="coerce")
                # Explicitly cast to float64 for pandas-ta compatibility
                self.df_calculated[col] = numeric_col.astype("float64")
            except Exception as e:
                self.logger.critical(
                    f"Failed convert column '{col}' to numeric/float64 {self.symbol}: {e}",
                    exc_info=True,
                )
                self.df_calculated = pd.DataFrame()
                return  # Invalidate on critical conversion error

            if self.df_calculated[col].isna().any():
                nan_count = self.df_calculated[col].isna().sum()
                self.logger.warning(
                    f"{nan_count} NaN(s) found in '{col}' after conversion {self.symbol}."
                )

        # Drop rows with NaNs in essential columns *after* conversion attempt
        self.df_calculated.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if self.df_calculated.empty:
            self.logger.error(
                f"DataFrame empty after dropping NaNs in OHLC {self.symbol}."
            )
            return  # Stop if no valid OHLC data

        # Determine minimum rows needed based on longest indicator period
        enabled_cfg = self.config.get("indicators", {})
        max_lookback = 1
        for ind_key, ind_details in self.INDICATOR_CONFIG.items():
            if enabled_cfg.get(ind_key.lower(), False):
                period_key = ind_details.get("min_data_param_key")
                config_key = (
                    ind_details.get("params_map", {}).get(period_key)
                    if period_key
                    else None
                )
                if config_key:
                    param_val = self.get_period(
                        config_key
                    )  # Use helper to get period/default
                    if (
                        isinstance(param_val, (int, float, Decimal))
                        and Decimal(str(param_val)) > 0
                    ):
                        max_lookback = max(max_lookback, int(Decimal(str(param_val))))
                elif isinstance(ind_details.get("min_data"), int):
                    max_lookback = max(max_lookback, ind_details["min_data"])

        buffer = self.config.get("indicator_buffer_candles", 20)  # Configurable buffer
        min_required_rows = max_lookback + buffer

        if len(self.df_calculated) < min_required_rows:
            self.logger.warning(
                f"Insufficient valid data rows ({len(self.df_calculated)}) for {self.symbol}. "
                f"Need approx. {min_required_rows} (Max Lookback: {max_lookback} + Buffer: {buffer}). "
                "Some indicators might be all NaN or inaccurate."
            )
        else:
            self.logger.debug(
                f"Data length ({len(self.df_calculated)}) sufficient for indicators (Min required: {min_required_rows})."
            )

    def get_period(self, key: str) -> Any:
        """
        Safely retrieves config value for indicator periods/params, using defaults if necessary.
        Ensures values are returned in their original type (int, float, Decimal) from defaults or config.
        """
        # Prioritize value from self.config if it exists and is not None
        config_val = self.config.get(key)
        if config_val is not None:
            # Attempt to return in the type suggested by the default value if possible
            default_val = DEFAULT_INDICATOR_PERIODS.get(key)
            if default_val is not None:
                try:
                    if isinstance(default_val, Decimal):
                        return Decimal(str(config_val))
                    if isinstance(default_val, float):
                        return float(config_val)
                    if isinstance(default_val, int):
                        return int(config_val)
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"Could not convert config value '{config_val}' for key '{key}' to type {type(default_val)}. Using raw config value."
                    )
            return (
                config_val  # Return raw config value if no default or conversion failed
            )

        # Fallback to default value if not in config or config value was None
        return DEFAULT_INDICATOR_PERIODS.get(key)

    def _format_ta_column_name(self, pattern: str, params: Dict[str, Any]) -> str:
        """Formats TA column names based on pattern and parameters, handling None and floats."""
        fmt_params = {}
        for k, v in params.items():
            if v is None:
                fmt_params[k] = "DEF"  # Placeholder for None
            elif isinstance(v, (float, Decimal)):
                # If format specifier exists in pattern (e.g., :.2f), pass as float for f-string formatting
                if f"{{{k}:." in pattern:
                    fmt_params[k] = float(v)
                # Otherwise, convert to string, replace '.' with '_' for safe column name part
                else:
                    fmt_params[k] = str(v).replace(".", "_")
            else:
                fmt_params[k] = v  # Use int/str directly
        try:
            return pattern.format(**fmt_params)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            self.logger.error(
                f"Error formatting TA column pattern '{pattern}' with params {fmt_params}: {e}"
            )
            # Create a fallback name
            base_name = (
                pattern.split("{")[0].rstrip("_")
                if "{" in pattern
                else pattern or "UNK_IND"
            )
            param_suffix = "_".join(map(str, params.values()))
            return f"{base_name}_{param_suffix}_FORMAT_ERROR"

    def _calculate_volume_ma(
        self, df: pd.DataFrame, length: int
    ) -> Optional[pd.Series]:
        """Calculates Simple Moving Average of volume."""
        if "volume" not in df.columns:
            self.logger.warning(
                f"Volume MA calc skipped {self.symbol}: 'volume' column missing."
            )
            return None
        if not (isinstance(length, int) and length > 0):
            self.logger.warning(
                f"Volume MA calc skipped {self.symbol}: Invalid length {length}."
            )
            return None
        # Ensure volume is float type for TA function
        volume_series = df["volume"].astype(float).fillna(0)
        if len(volume_series) < length:
            self.logger.debug(
                f"Insufficient data for Volume MA {length} on {self.symbol}."
            )
            return pd.Series(
                np.nan, index=df.index
            )  # Return Series of NaNs matching index
        try:
            # Ensure pandas_ta reference is valid
            if ta is None:
                raise ImportError("pandas_ta is not installed or failed to import.")
            return ta.sma(volume_series, length=length)
        except Exception as e:
            self.logger.error(
                f"Error calculating SMA for volume {self.symbol}: {e}", exc_info=True
            )
            return None

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators and stores results in df_calculated."""
        if self.df_calculated.empty:
            self.logger.warning(
                f"Skipping indicator calculation for {self.symbol}: DataFrame is empty/invalid."
            )
            return
        if ta is None:
            self.logger.error(
                "Skipping indicator calculation: pandas_ta library is not available."
            )
            return

        df_work = self.df_calculated  # Work directly on the prepared DataFrame
        enabled_cfg = self.config.get("indicators", {})
        self.ta_column_names = {}  # Reset mapping for this calculation run

        for ind_key, ind_details in self.INDICATOR_CONFIG.items():
            if not enabled_cfg.get(ind_key.lower(), False):
                continue  # Skip disabled

            params_for_ta = {}
            is_params_valid = True
            for ta_param, cfg_key in ind_details.get("params_map", {}).items():
                param_value = self.get_period(cfg_key)  # Uses updated get_period
                if param_value is None:
                    is_params_valid = False
                    break
                try:  # Convert Decimal/str to float/int for pandas-ta/TA-Lib
                    if isinstance(param_value, Decimal):
                        params_for_ta[ta_param] = float(param_value)
                    elif isinstance(param_value, str):
                        params_for_ta[ta_param] = (
                            float(param_value)
                            if "." in param_value
                            else int(param_value)
                        )
                    elif isinstance(param_value, (int, float)):
                        params_for_ta[ta_param] = param_value
                    else:
                        raise TypeError(f"Unsupported type {type(param_value)}")
                except (ValueError, TypeError) as e:
                    is_params_valid = False
                    self.logger.error(
                        f"Param conversion error for {cfg_key} ({param_value}): {e}"
                    )
                    break
            if not is_params_valid:
                self.logger.warning(
                    f"Skipping indicator {ind_key} for {self.symbol} due to invalid parameters."
                )
                continue

            try:
                func_name = ind_details["func_name"]
                # Check for custom methods first, then pandas_ta
                func_obj = (
                    getattr(self, func_name, None)
                    if func_name.startswith("_")
                    else getattr(ta, func_name, None)
                )

                if not func_obj or not callable(func_obj):
                    self.logger.error(
                        f"TA function '{func_name}' not found or not callable in pandas_ta or TradingAnalyzer."
                    )
                    continue

                # Basic data length check before calling TA function
                min_len_key = ind_details.get("min_data_param_key", "length")
                min_len = int(
                    params_for_ta.get(min_len_key, ind_details.get("min_data", 1))
                )
                if len(df_work) < min_len:
                    self.logger.debug(
                        f"Skip {ind_key}: data len {len(df_work)} < required {min_len}."
                    )
                    continue

                # Prepare inputs (OHLCV Series) required by the function
                required_ta_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]  # Potential cols needed by TA funcs
                input_df = df_work[
                    required_ta_cols
                ].copy()  # Pass relevant columns as DataFrame if func allows
                close_series = df_work["close"]  # Always have close series available

                # Execute the TA function
                result = None
                if func_name == "_calculate_volume_ma":
                    result = func_obj(df_work, **params_for_ta)  # Custom func takes df
                elif ind_details.get("pass_close_only", False):
                    result = func_obj(close=close_series, **params_for_ta)
                else:  # Standard pandas-ta call
                    # Inspect signature to pass only required arguments
                    import inspect

                    sig_params = inspect.signature(func_obj).parameters
                    func_args = {}
                    for col in ["open", "high", "low", "close", "volume"]:
                        if col in sig_params and col in input_df.columns:
                            func_args[col] = input_df[col]
                    # Fallback to 'close' if no other OHLCV is explicitly in signature but 'close' is
                    if not func_args and "close" in sig_params:
                        func_args["close"] = close_series

                    if func_args:  # If function signature expects specific columns
                        result = func_obj(**func_args, **params_for_ta)
                    else:  # If signature is generic (like *args, **kwargs), pass the DataFrame subset
                        # This is less common for pandas-ta functions but covers potential cases
                        self.logger.debug(
                            f"Passing DataFrame subset to TA func {func_name} as signature is generic or doesn't match OHLCV."
                        )
                        result = func_obj(
                            input_df, **params_for_ta
                        )  # Requires function to handle DataFrame input

                if result is None:
                    self.logger.debug(f"Indicator {ind_key} returned None.")
                    continue

                # --- Integrate results into the DataFrame ---
                should_concat = ind_details.get("concat", False)
                col_name = (
                    self._format_ta_column_name(
                        ind_details.get("main_col_pattern", ""), params_for_ta
                    )
                    if "main_col_pattern" in ind_details
                    else None
                )

                if isinstance(result, pd.Series):
                    if col_name:  # Needs a defined column name pattern
                        if col_name in df_work.columns:
                            df_work.drop(
                                columns=[col_name], inplace=True, errors="ignore"
                            )  # Avoid duplicate columns warning
                        if should_concat:
                            # Concat Series as a new column (ensure float type)
                            df_work = pd.concat(
                                [
                                    df_work,
                                    result.to_frame(name=col_name).astype("float64"),
                                ],
                                axis=1,
                                copy=False,
                            )
                        else:
                            # Assign Series as a new column (ensure float type)
                            df_work[col_name] = result.astype("float64")
                        self.ta_column_names[ind_key] = (
                            col_name  # Map internal key to the column name
                        )
                    else:
                        self.logger.error(
                            f"Missing 'main_col_pattern' for Series result of {ind_key}."
                        )
                elif isinstance(result, pd.DataFrame):
                    if should_concat:
                        # Prepare piece, handle type conversion and duplicates
                        try:
                            piece = result.astype("float64")
                        except Exception:  # Fallback: convert column by column
                            piece = pd.DataFrame(index=result.index)
                            for c in result.columns:
                                try:
                                    piece[c] = pd.to_numeric(
                                        result[c], errors="coerce"
                                    ).astype("float64")
                                except:
                                    self.logger.warning(
                                        f"Could not convert col {c} from {ind_key} result."
                                    )
                        # Drop columns that would be duplicated if they already exist in df_work
                        cols_to_drop = [
                            c for c in piece.columns if c in df_work.columns
                        ]
                        if cols_to_drop:
                            self.logger.debug(
                                f"Dropping duplicate columns before concat: {cols_to_drop}"
                            )
                            df_work.drop(
                                columns=cols_to_drop, inplace=True, errors="ignore"
                            )
                        df_work = pd.concat([df_work, piece], axis=1, copy=False)
                    else:  # concat=False, but got a DataFrame - this indicates a config mismatch
                        self.logger.error(
                            f"Indicator {ind_key} (concat=False) returned DataFrame."
                        )
                        continue
                else:  # Unexpected result type
                    self.logger.warning(
                        f"Indicator {ind_key} returned unexpected type: {type(result)}"
                    )
                    continue

                # --- Map internal keys for multi-column results ---
                if "multi_cols" in ind_details:
                    for internal_key, col_pattern in ind_details["multi_cols"].items():
                        actual_col_name = self._format_ta_column_name(
                            col_pattern, params_for_ta
                        )
                        if actual_col_name in df_work.columns:
                            self.ta_column_names[internal_key] = actual_col_name
                        else:  # This can happen if pandas-ta changes column naming conventions
                            self.logger.warning(
                                f"Mapped column '{actual_col_name}' for '{internal_key}' of {ind_key} not found in DataFrame. Check ta version/column names."
                            )
                            # Attempt partial match (useful if version slightly changed format)
                            partial_match = next(
                                (
                                    col
                                    for col in df_work.columns
                                    if col.startswith(actual_col_name.split("_")[0])
                                    and all(
                                        p in col
                                        for p in map(str, params_for_ta.values())
                                    )
                                ),
                                None,
                            )
                            if partial_match:
                                self.logger.warning(
                                    f"    --> Found partial match: '{partial_match}'. Using this column."
                                )
                                self.ta_column_names[internal_key] = partial_match
                            else:
                                self.logger.debug(
                                    f"Available columns after {ind_key}: {list(df_work.columns)}"
                                )

            except Exception as e:
                self.logger.error(
                    f"Error calculating indicator {ind_key} for {self.symbol}: {e}",
                    exc_info=True,
                )

        # Assign the updated DataFrame back
        self.df_calculated = df_work
        self.logger.debug(
            f"Indicator calculation complete {self.symbol}. Final columns: {list(self.df_calculated.columns)}"
        )
        self.logger.debug(f"TA Column Map: {self.ta_column_names}")

    def _update_latest_indicator_values(self) -> None:
        """Updates dict with latest indicator values, converting types based on config."""
        if self.df_calculated.empty:
            self.logger.warning(
                f"Cannot update latest values {self.symbol}: calculated DataFrame is empty."
            )
            self.indicator_values = {}
            return
        try:
            latest_ind_row = self.df_calculated.iloc[-1]
            latest_ohlcv_row = self.df_original_ohlcv.iloc[-1]

            ohlcv_map = {
                OPEN_KEY: "open",
                HIGH_KEY: "high",
                LOW_KEY: "low",
                CLOSE_KEY: "close",
                VOLUME_KEY: "volume",
            }
            latest_values: Dict[str, Union[Decimal, float, None]] = {}

            # Initialize all expected keys to None first
            all_keys = (
                set(self.indicator_type_map.keys())
                | set(ohlcv_map.keys())
                | set(self.ta_column_names.keys())
            )
            for key in all_keys:
                latest_values[key] = None

            # Process TA indicators from df_calculated using the mapped column names
            for int_key, actual_col in self.ta_column_names.items():
                if actual_col in latest_ind_row.index:
                    val = latest_ind_row[actual_col]
                    target_type = self.indicator_type_map.get(
                        int_key, "float"
                    )  # Default float
                    if pd.notna(val):
                        try:
                            latest_values[int_key] = (
                                Decimal(str(val))
                                if target_type == "decimal"
                                else float(val)
                            )
                        except (InvalidOperation, TypeError, ValueError) as conv_err:
                            self.logger.debug(
                                f"Conv err {int_key} ('{val}') to {target_type}: {conv_err}"
                            )
                            latest_values[int_key] = (
                                None  # Set to None on conversion error
                            )
                    # else: keep None if pd.isna(val)
                # else: keep None if column wasn't found/mapped

            # Process OHLCV from original df (prefer Decimal)
            for disp_key, src_col in ohlcv_map.items():
                val_ohlcv = latest_ohlcv_row.get(src_col)
                if pd.notna(val_ohlcv):
                    try:
                        latest_values[disp_key] = Decimal(str(val_ohlcv))
                    except (InvalidOperation, TypeError, ValueError) as conv_err:
                        self.logger.debug(
                            f"Conv err OHLCV {disp_key} ('{val_ohlcv}'): {conv_err}"
                        )
                        latest_values[disp_key] = (
                            None  # Set to None on conversion error
                        )
                # else keep None

            self.indicator_values = latest_values  # Assign the populated dictionary

            # Optional: Log summary of key values
            price_prec = get_price_precision(self.market_info, self.logger)
            log_items = []
            keys_to_log = [
                CLOSE_KEY,
                ATR_KEY,
                EMA_SHORT_KEY,
                EMA_LONG_KEY,
                RSI_KEY,
                STOCHRSI_K_KEY,
                STOCHRSI_D_KEY,
            ]
            for k in keys_to_log:
                v = self.indicator_values.get(k)
                if v is not None:
                    prec = (
                        price_prec + 2
                        if k == ATR_KEY
                        else price_prec
                        if isinstance(v, Decimal)
                        else 4
                    )
                    try:
                        log_items.append(f"{k}={v:.{prec}f}")
                    except:
                        log_items.append(f"{k}={v}")  # Fallback
                else:
                    log_items.append(f"{k}=N/A")
            self.logger.debug(f"Latest values {self.symbol}: {', '.join(log_items)}")

        except IndexError:
            self.logger.error(f"IndexError getting latest row {self.symbol}.")
        except Exception as e:
            self.logger.error(
                f"Error update latest vals {self.symbol}: {e}", exc_info=True
            )
        # Ensure indicator_values is empty if error occurs
        if not hasattr(self, "indicator_values") or not self.indicator_values:
            self.indicator_values = {}

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(
        self, window: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """Calculates Fibonacci levels based on high/low window."""
        window = window or self.get_period("fibonacci_window")
        if not (isinstance(window, int) and window > 0):
            return {}
        if len(self.df_original_ohlcv) < window:
            return {}

        df_slice = self.df_original_ohlcv.tail(window)
        try:
            high_series = pd.to_numeric(df_slice["high"], errors="coerce").dropna()
            low_series = pd.to_numeric(df_slice["low"], errors="coerce").dropna()
            if high_series.empty or low_series.empty:
                return {}

            period_high = Decimal(str(high_series.max()))
            period_low = Decimal(str(low_series.min()))
            diff = period_high - period_low
            levels = {}

            price_prec = get_price_precision(self.market_info, self.logger)
            min_tick = get_min_tick_size(self.market_info, self.logger)
            quant = (
                min_tick
                if min_tick and min_tick > Decimal(0)
                else Decimal(f"1e-{price_prec}")
            )

            if diff > Decimal(0):
                for level_pct_dec in FIB_LEVELS:
                    price_raw = period_high - (diff * level_pct_dec)
                    levels[f"Fib_{level_pct_dec * 100:.1f}%"] = (
                        price_raw / quant
                    ).quantize(Decimal("1"), rounding=ROUND_DOWN) * quant
            else:  # Flat price range
                level_q = (period_high / quant).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * quant
                for level_pct_dec in FIB_LEVELS:
                    levels[f"Fib_{level_pct_dec * 100:.1f}%"] = level_q

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated {len(levels)} Fib levels {self.symbol}.")
            return levels
        except Exception as e:
            self.logger.error(
                f"Fibonacci calculation error {self.symbol}: {e}", exc_info=False
            )  # Less verbose
            self.fib_levels_data = {}
            return {}

    # --- Nearest Fibonacci ---
    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> List[Tuple[str, Decimal]]:
        """Finds N nearest Fibonacci levels to the current price."""
        if (
            not self.fib_levels_data
            or not (isinstance(current_price, Decimal) and current_price > 0)
            or num_levels <= 0
        ):
            return []
        try:
            distances = [
                {"name": n, "level": p, "distance": abs(current_price - p)}
                for n, p in self.fib_levels_data.items()
                if isinstance(p, Decimal) and p > 0
            ]
            if not distances:
                return []
            distances.sort(key=lambda x: x["distance"])
            return [(item["name"], item["level"]) for item in distances[:num_levels]]
        except Exception as e:
            self.logger.error(
                f"Error find nearest Fib {self.symbol}: {e}", exc_info=False
            )
            return []

    # --- EMA Score ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculates score based on EMA alignment and price position."""
        ema_s = self.indicator_values.get(EMA_SHORT_KEY)
        ema_l = self.indicator_values.get(EMA_LONG_KEY)
        close = self.indicator_values.get(CLOSE_KEY)
        # Ensure all are valid Decimals before comparison
        if not all(isinstance(v, Decimal) for v in [ema_s, ema_l, close]):
            return np.nan
        ema_s: Decimal = ema_s
        ema_l: Decimal = ema_l
        close: Decimal = close  # Type hinting
        if close > ema_s > ema_l:
            return 1.0  # Bullish
        if close < ema_s < ema_l:
            return -1.0  # Bearish
        return 0.0  # Neutral

    # --- Signal Generation ---
    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict[str, Any]]
    ) -> str:
        """Generates BUY/SELL/HOLD signal based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}
        score, w_sum = Decimal("0"), Decimal("0")
        active_checks, nan_checks = 0, 0
        debug_scores = {}

        if not self.indicator_values:
            return "HOLD"  # Cannot generate signal without indicators

        # Min valid indicators check
        min_req = self.config.get("min_active_indicators_for_signal", 7)
        # Count valid (non-NaN, non-None) values for keys present in INDICATOR_CONFIG
        valid_core_count = sum(
            1
            for key in self.INDICATOR_CONFIG
            if pd.notna(self.indicator_values.get(key))
        )
        if valid_core_count < min_req:
            self.logger.warning(
                f"Signal {self.symbol}: Only {valid_core_count} valid indicators < req {min_req}. HOLD."
            )
            return "HOLD"
        if not (isinstance(current_price, Decimal) and current_price > 0):
            self.logger.warning(
                f"Invalid current price for signal gen {self.symbol}. HOLD."
            )
            return "HOLD"

        active_weights = self.weights
        if not active_weights:
            self.logger.warning(
                f"Weight set '{self.active_weight_set_name}' empty {self.symbol}. HOLD."
            )
            return "HOLD"

        # Calculate weighted score
        for ind_key_lower, weight_val in active_weights.items():
            if not self.config.get("indicators", {}).get(ind_key_lower, False):
                continue  # Skip disabled
            try:
                weight = Decimal(str(weight_val))
            except:
                continue  # Skip invalid weight
            if weight == 0:
                continue

            method_name = f"_check_{ind_key_lower}"
            method_obj = getattr(self, method_name, None)
            if not method_obj or not callable(method_obj):
                continue  # Skip if check method doesn't exist

            score_float = np.nan  # Default score if calculation fails
            try:
                if ind_key_lower == "orderbook":
                    score_float = method_obj(orderbook_data, current_price)
                else:
                    score_float = method_obj()
            except Exception as e:
                self.logger.error(
                    f"Error in check method {method_name}: {e}", exc_info=True
                )

            debug_scores[ind_key_lower] = (
                f"{score_float:.3f}" if pd.notna(score_float) else "NaN"
            )
            if pd.notna(score_float):
                try:
                    ind_score = Decimal(str(score_float))
                    clamped = max(
                        Decimal("-1"), min(Decimal("1"), ind_score)
                    )  # Clamp score
                    score += clamped * weight
                    w_sum += abs(weight)
                    active_checks += 1
                except (InvalidOperation, TypeError):
                    nan_checks += 1
                    self.logger.debug(
                        f"Score conversion failed for {ind_key_lower}: {score_float}"
                    )
            else:
                nan_checks += 1

        # Determine final signal
        final_signal = "HOLD"
        threshold = Decimal(str(self.get_period("signal_score_threshold") or "0.7"))
        if w_sum > 0:  # Only determine BUY/SELL if weights contributed
            if score >= threshold:
                final_signal = "BUY"
            elif score <= -threshold:
                final_signal = "SELL"
        elif active_checks > 0:
            self.logger.warning(f"Total weight sum is zero {self.symbol}. HOLD.")
        else:
            self.logger.warning(
                f"No indicators contributed to score {self.symbol}. HOLD."
            )

        # Log signal generation details
        price_prec = get_price_precision(self.market_info, self.logger)
        self.logger.info(
            f"Signal ({self.symbol} @ {_format_price_or_na(current_price, price_prec)}): "
            f"Set='{self.active_weight_set_name}', Checks[Act:{active_checks},NaN:{nan_checks}], "
            f"WeightSum={w_sum:.2f}, Score={score:.4f} (Th:{threshold:.2f}) "
            f"==> {format_signal(final_signal)}"
        )
        self.logger.debug(f"Individual Scores ({self.symbol}): {debug_scores}")
        self.signals = {
            "BUY": int(final_signal == "BUY"),
            "SELL": int(final_signal == "SELL"),
            "HOLD": int(final_signal == "HOLD"),
        }
        return final_signal

    # --- Individual Indicator Check Methods (_check_*) ---
    # (Implementations remain the same as previous version, using self.indicator_values)
    def _check_ema_alignment(self) -> float:
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:  # Simplified scaling
        mom = self.indicator_values.get(MOMENTUM_KEY)
        close = self.indicator_values.get(CLOSE_KEY)
        if pd.isna(mom) or not isinstance(close, Decimal) or close <= 0:
            return np.nan
        try:
            mom_dec = Decimal(str(mom))
            score = mom_dec / close
            return float(
                max(Decimal("-1"), min(Decimal("1"), score * 10))
            )  # Example scaling
        except:
            return 0.0

    def _check_volume_confirmation(self) -> float:
        vol = self.indicator_values.get(VOLUME_KEY)
        vol_ma = self.indicator_values.get(VOLUME_MA_KEY)
        mult = Decimal(str(self.get_period("volume_confirmation_multiplier") or "1.5"))
        if (
            not all(
                isinstance(v, Decimal) and pd.notna(v) and v >= 0 for v in [vol, vol_ma]
            )
            or mult <= 0
        ):
            return np.nan
        if vol_ma == 0:
            return 0.0  # Avoid division by zero
        ratio = vol / vol_ma
        return (
            1.0
            if ratio > mult
            else -0.4
            if ratio < (1 / mult if mult > 0 else 0)
            else 0.0
        )

    def _check_stoch_rsi(self) -> float:
        k, d = (
            self.indicator_values.get(STOCHRSI_K_KEY),
            self.indicator_values.get(STOCHRSI_D_KEY),
        )
        if pd.isna(k) or pd.isna(d):
            return np.nan
        k, d = float(k), float(d)
        os = float(self.get_period("stoch_rsi_oversold_threshold") or 20)
        ob = float(self.get_period("stoch_rsi_overbought_threshold") or 80)
        score = 0.0
        if k < os and d < os:
            score = 1.0  # Bullish (oversold)
        elif k > ob and d > ob:
            score = -1.0  # Bearish (overbought)
        if k > d and score >= 0:
            score = max(score, 0.4)  # Bullish cross / tendency
        if k < d and score <= 0:
            score = min(score, -0.4)  # Bearish cross / tendency
        if 40 < k < 60 and 40 < d < 60:
            score *= 0.5  # Reduce score in neutral zone
        return score

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get(RSI_KEY)
        if pd.isna(rsi):
            return np.nan
            rsi = float(rsi)
        os, ob = (
            float(self.get_period("rsi_oversold_threshold") or 30),
            float(self.get_period("rsi_overbought_threshold") or 70),
        )
        if rsi <= os:
            return 1.0
        if rsi >= ob:
            return -1.0
        if os < rsi < 50:
            return (50.0 - rsi) / (50.0 - os) * 0.8
        if 50 < rsi < ob:
            return (50.0 - rsi) / (ob - 50.0) * 0.8
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get(CCI_KEY)
        if pd.isna(cci):
            return np.nan
            cci = float(cci)
        sos, mob = (
            float(self.get_period("cci_strong_oversold") or -150),
            float(self.get_period("cci_moderate_overbought") or 100),
        )
        sob, mos = (
            float(self.get_period("cci_strong_overbought") or 150),
            float(self.get_period("cci_moderate_oversold") or -100),
        )
        if cci <= sos:
            return 1.0
        if cci >= sob:
            return -1.0
        if cci < mos:
            return 0.6
        if cci > mob:
            return -0.6
        return 0.0

    def _check_wr(self) -> float:
        wr = self.indicator_values.get(WILLIAMS_R_KEY)
        if pd.isna(wr):
            return np.nan
            wr = float(wr)
        os, ob = (
            float(self.get_period("wr_oversold_threshold") or -80),
            float(self.get_period("wr_overbought_threshold") or -20),
        )
        if wr <= os:
            return 1.0
        if wr >= ob:
            return -1.0
        mid = (os + ob) / 2.0
        if os < wr < mid:
            return (wr - mid) / (os - mid) * 0.7
        if mid <= wr < ob:
            return (wr - mid) / (ob - mid) * -0.7
        return 0.0

    def _check_psar(self) -> float:
        psar_l, psar_s, close = (
            self.indicator_values.get(PSAR_LONG_KEY),
            self.indicator_values.get(PSAR_SHORT_KEY),
            self.indicator_values.get(CLOSE_KEY),
        )
        if not isinstance(close, Decimal):
            return np.nan
        is_long = isinstance(psar_l, Decimal) and close > psar_l
        is_short = isinstance(psar_s, Decimal) and close < psar_s
        if is_long and not is_short:
            return 1.0
        if is_short and not is_long:
            return -1.0
        return 0.0

    def _check_sma_10(self) -> float:
        sma, close = (
            self.indicator_values.get(SMA10_KEY),
            self.indicator_values.get(CLOSE_KEY),
        )
        if not all(isinstance(v, Decimal) for v in [sma, close]):
            return np.nan
        diff = (close - sma) / sma if sma > 0 else Decimal(0)
        return float(max(Decimal("-1"), min(Decimal("1"), diff * 10)))

    def _check_vwap(self) -> float:
        vwap, close = (
            self.indicator_values.get(VWAP_KEY),
            self.indicator_values.get(CLOSE_KEY),
        )
        if not all(isinstance(v, Decimal) for v in [vwap, close]):
            return np.nan
        diff = (close - vwap) / vwap if vwap > 0 else Decimal(0)
        return float(max(Decimal("-1"), min(Decimal("1"), diff * 15)))

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get(MFI_KEY)
        if pd.isna(mfi):
            return np.nan
            mfi = float(mfi)
        os, ob = (
            float(self.get_period("mfi_oversold_threshold") or 20),
            float(self.get_period("mfi_overbought_threshold") or 80),
        )
        if mfi <= os:
            return 1.0
        if mfi >= ob:
            return -1.0
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bb_l, bb_m, bb_u, close = (
            self.indicator_values.get(BB_LOWER_KEY),
            self.indicator_values.get(BB_MIDDLE_KEY),
            self.indicator_values.get(BB_UPPER_KEY),
            self.indicator_values.get(CLOSE_KEY),
        )
        if not all(isinstance(v, Decimal) for v in [bb_l, bb_m, bb_u, close]):
            return np.nan
        if close <= bb_l:
            return 1.0
        if close >= bb_u:
            return -1.0
        width = bb_u - bb_l
        if width > 0:
            pos = (close - bb_m) / (width / 2)
            return float(
                max(Decimal("-0.7"), min(Decimal("0.7"), pos * Decimal("0.7")))
            )
        return 0.0

    def _check_orderbook(
        self, orderbook_data: Optional[Dict[str, Any]], current_price: Decimal
    ) -> float:  # Added type hint for clarity
        if not orderbook_data:
            return np.nan
        try:
            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])
            if not bids or not asks:
                return np.nan
            levels = self.config.get("orderbook_check_levels", 10)
            bid_q = sum(
                Decimal(str(b[1]))
                for b in bids[:levels]
                if len(b) == 2 and b[1] is not None
            )
            ask_q = sum(
                Decimal(str(a[1]))
                for a in asks[:levels]
                if len(a) == 2 and a[1] is not None
            )
            total_q = bid_q + ask_q
            if total_q == 0:
                return 0.0
            obi = (bid_q - ask_q) / total_q
            return float(max(Decimal("-1"), min(Decimal("1"), obi)))
        except Exception as e:
            self.logger.warning(f"OB analysis error: {e}")
            return np.nan

    # --- TP/SL Calculation ---
    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculates potential TP/SL based on ATR and config multipliers."""
        # Basic validation
        if (
            signal not in ["BUY", "SELL"]
            or not isinstance(entry_price_estimate, Decimal)
            or entry_price_estimate <= 0
        ):
            return entry_price_estimate, None, None

        atr = self.indicator_values.get(ATR_KEY)
        if not (isinstance(atr, Decimal) and atr > 0):
            self.logger.warning(
                f"TP/SL Calc {self.symbol}: ATR invalid ({atr}). Using default % if configured."
            )
            # Fallback to percentage-based ATR if ATR is invalid
            default_atr_pct_str = str(
                self.get_period("default_atr_percentage_of_price") or "0.0"
            )  # Default 0 if not set
            try:
                default_atr_pct = Decimal(default_atr_pct_str)
                if default_atr_pct > 0:
                    atr = entry_price_estimate * default_atr_pct
                else:
                    self.logger.error(
                        f"Cannot calculate TP/SL {self.symbol}: ATR invalid and no valid default ATR %."
                    )
                    return entry_price_estimate, None, None
            except:
                self.logger.error(
                    f"Cannot calculate TP/SL {self.symbol}: Invalid default ATR % '{default_atr_pct_str}'."
                )
                return entry_price_estimate, None, None

        try:
            # Get multipliers, ensuring they are positive Decimals
            tp_mult_cfg = self.get_period("take_profit_multiple") or "2.0"
            sl_mult_cfg = self.get_period("stop_loss_multiple") or "1.5"
            tp_mult = Decimal(str(tp_mult_cfg))
            sl_mult = Decimal(str(sl_mult_cfg))
            if tp_mult <= 0 or sl_mult <= 0:
                raise ValueError("Multipliers must be positive.")

            price_prec = get_price_precision(self.market_info, self.logger)
            min_tick = get_min_tick_size(self.market_info, self.logger)
            # Ensure quantize_unit is positive
            quant = (
                min_tick
                if min_tick and min_tick > Decimal(0)
                else Decimal(f"1e-{price_prec}")
            )
            if quant <= Decimal(0):
                quant = Decimal("1e-8")  # Absolute fallback

            tp_offset = atr * tp_mult
            sl_offset = atr * sl_mult

            if signal == "BUY":
                tp = ((entry_price_estimate + tp_offset) / quant).quantize(
                    Decimal("1"), rounding=ROUND_UP
                ) * quant
                sl = ((entry_price_estimate - sl_offset) / quant).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * quant
            else:  # SELL
                tp = ((entry_price_estimate - tp_offset) / quant).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * quant
                sl = ((entry_price_estimate + sl_offset) / quant).quantize(
                    Decimal("1"), rounding=ROUND_UP
                ) * quant

            # --- Final Validation of Calculated SL/TP ---
            # Ensure SL/TP are strictly better than entry price based on signal direction
            if sl is not None:
                if sl <= 0:
                    sl = None  # SL cannot be non-positive
                elif signal == "BUY" and sl >= entry_price_estimate:
                    sl = None
                elif signal == "SELL" and sl <= entry_price_estimate:
                    sl = None
            if tp is not None:
                if tp <= 0:
                    tp = None  # TP cannot be non-positive
                elif signal == "BUY" and tp <= entry_price_estimate:
                    tp = None
                elif signal == "SELL" and tp >= entry_price_estimate:
                    tp = None
            # Ensure SL and TP don't cross (e.g., SL higher than TP for a buy)
            if sl and tp:
                if (signal == "BUY" and sl >= tp) or (signal == "SELL" and sl <= tp):
                    self.logger.warning(
                        f"Invalid TP/SL range calculated for {self.symbol} {signal} (SL: {sl}, TP: {tp}). Setting both to None."
                    )
                    sl = tp = None

            self.logger.debug(
                f"Calc TP/SL ({signal}): Entry={_format_price_or_na(entry_price_estimate, price_prec)}, ATR={_format_price_or_na(atr, price_prec + 2)}, TP={_format_price_or_na(tp, price_prec)}, SL={_format_price_or_na(sl, price_prec)}"
            )
            return entry_price_estimate, tp, sl

        except (InvalidOperation, TypeError, ValueError) as e:
            self.logger.error(
                f"Error calculating TP/SL {self.symbol}: {e}", exc_info=False
            )  # Less verbose log
            return entry_price_estimate, None, None
        except Exception as e:  # Catch any other unexpected error
            self.logger.error(
                f"Unexpected error calculating TP/SL {self.symbol}: {e}", exc_info=True
            )
            return entry_price_estimate, None, None
