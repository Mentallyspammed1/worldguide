```python
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
import inspect # Used in _calculate_all_indicators

# Import necessary libraries
import numpy as np
import pandas as pd

# Module-level logger
MODULE_LOGGER = logging.getLogger(__name__)

# Ensure pandas_ta is installed: pip install pandas_ta
try:
    import pandas_ta as ta
except ImportError:
    MODULE_LOGGER.error("pandas_ta library not found. Please install it: pip install pandas_ta")

    # Define a dummy ta object to prevent NameError, although analysis will fail.
    class DummyTA:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                MODULE_LOGGER.error(f"pandas_ta not installed. Cannot call function '{name}'.")
                # Return None or an empty DataFrame/Series as TA functions often do on failure
                # Check typical return types of pandas_ta functions used. Many return Series.
                # If a DataFrame is expected (e.g., bbands, stochrsi), this might need adjustment
                # or the calling code must be robust to None.
                # For simplicity, returning None here.
                return None
            return method

    ta = DummyTA()


# Ensure Decimal context precision is sufficient
try:
    getcontext().prec = 38  # Set high precision for calculations
except Exception as e:
    MODULE_LOGGER.error(f"Failed to set Decimal precision: {e}")

# Import constants and utility functions from utils
try:
    from utils import (
        CCXT_INTERVAL_MAP,
        DEFAULT_INDICATOR_PERIODS,
        FIB_LEVELS,
        get_min_tick_size,
        get_price_precision,
        NEON_RED,  # Assuming these are color codes for logging/printing if used elsewhere
        NEON_YELLOW,
        NEON_GREEN,
        RESET_ALL_STYLE,
        NEON_PURPLE,
        NEON_BLUE,
        NEON_CYAN,
        format_signal,
    )
except ImportError:
    MODULE_LOGGER.error("Failed importing components from utils in analysis.py. Using fallbacks.")
    # Define fallbacks for constants and functions if utils import fails
    NEON_RED = NEON_YELLOW = NEON_GREEN = RESET_ALL_STYLE = ""
    NEON_PURPLE = NEON_BLUE = NEON_CYAN = ""
    DEFAULT_INDICATOR_PERIODS = {}
    CCXT_INTERVAL_MAP = {}
    FIB_LEVELS = [Decimal(str(f)) for f in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]]

    def get_price_precision(market_info: Dict[str, Any], logger: logging.Logger) -> int:
        logger.warning("Using fallback get_price_precision.")
        return 4

    def get_min_tick_size(market_info: Dict[str, Any], logger: logging.Logger) -> Decimal:
        logger.warning("Using fallback get_min_tick_size.")
        return Decimal("0.0001")

    def format_signal(signal_str: str, **kwargs) -> str: # Make fallback kwargs-compatible
        # Assuming this fallback should just return the signal string
        return signal_str


# Define constants for frequently used indicator keys (improves maintainability)
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

# Note: DECIMAL_INDICATOR_KEYS was defined in original but not used.
# The type handling is now done via INDICATOR_CONFIG and indicator_type_map.
# If it were to be used, its definition would be here.


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
        ATR_KEY: { # Using constant for key
            "func_name": "atr", "params_map": {"length": "atr_period"},
            "main_col_pattern": "ATRr_{length}", "type": "decimal",
            "min_data_param_key": "length", "concat": False,
        },
        EMA_SHORT_KEY: {
            "func_name": "ema", "params_map": {"length": "ema_short_period"},
            "main_col_pattern": "EMA_{length}", "type": "decimal",
            "pass_close_only": True, "min_data_param_key": "length", "concat": False,
        },
        EMA_LONG_KEY: {
            "func_name": "ema", "params_map": {"length": "ema_long_period"},
            "main_col_pattern": "EMA_{length}", "type": "decimal",
            "pass_close_only": True, "min_data_param_key": "length", "concat": False,
        },
        "PSAR": { # PSAR generates PSAR_long and PSAR_short
            "func_name": "psar",
            "params_map": {"initial": "psar_initial_af", "step": "psar_af_step", "max": "psar_max_af"},
            "multi_cols": {PSAR_LONG_KEY: "PSARl_{initial}_{step}_{max}", PSAR_SHORT_KEY: "PSARs_{initial}_{step}_{max}"},
            "type": "decimal", "concat": True, "min_data": 2,
        },
        SMA10_KEY: {
            "func_name": "sma", "params_map": {"length": "sma_10_window"},
            "main_col_pattern": "SMA_{length}", "type": "decimal",
            "pass_close_only": True, "min_data_param_key": "length", "concat": False,
        },
        "Bollinger_Bands": { # Generates BB_Lower, BB_Middle, BB_Upper
            "func_name": "bbands",
            "params_map": {"length": "bollinger_bands_period", "std": "bollinger_bands_std_dev"},
            "multi_cols": {
                BB_LOWER_KEY: "BBL_{length}_{std:.1f}",
                BB_MIDDLE_KEY: "BBM_{length}_{std:.1f}",
                BB_UPPER_KEY: "BBU_{length}_{std:.1f}",
            },
            "type": "decimal", "concat": True, "min_data_param_key": "length",
        },
        VWAP_KEY: {
            "func_name": "vwap", "params_map": {},
            "main_col_pattern": "VWAP_D", "type": "decimal",
            "concat": True, "min_data": 1,
        },
        # --- Momentum & Oscillators ---
        MOMENTUM_KEY: {
            "func_name": "mom", "params_map": {"length": "momentum_period"},
            "main_col_pattern": "MOM_{length}", "type": "float",
            "pass_close_only": True, "min_data_param_key": "length", "concat": False,
        },
        CCI_KEY: {
            "func_name": "cci", "params_map": {"length": "cci_window", "c": "cci_constant"},
            "main_col_pattern": "CCI_{length}_{c:.3f}", "type": "float",
            "min_data_param_key": "length", "concat": False,
        },
        WILLIAMS_R_KEY: {
            "func_name": "willr", "params_map": {"length": "williams_r_window"},
            "main_col_pattern": "WILLR_{length}", "type": "float",
            "min_data_param_key": "length", "concat": False,
        },
        "StochRSI": { # Generates StochRSI_K, StochRSI_D
            "func_name": "stochrsi",
            "params_map": {
                "length": "stoch_rsi_window", "rsi_length": "stoch_rsi_rsi_window",
                "k": "stoch_rsi_k", "d": "stoch_rsi_d",
            },
            "multi_cols": {
                STOCHRSI_K_KEY: "STOCHRSIk_{length}_{rsi_length}_{k}_{d}",
                STOCHRSI_D_KEY: "STOCHRSId_{length}_{rsi_length}_{k}_{d}",
            },
            "type": "float", "concat": True, "min_data_param_key": "length",
        },
        RSI_KEY: {
            "func_name": "rsi", "params_map": {"length": "rsi_period"},
            "main_col_pattern": "RSI_{length}", "type": "float",
            "pass_close_only": True, "min_data_param_key": "length", "concat": False,
        },
        # --- Volume ---
        MFI_KEY: {
            "func_name": "mfi", "params_map": {"length": "mfi_window"},
            "main_col_pattern": "MFI_{length}", "type": "float",
            "concat": True, "min_data_param_key": "length",
        },
        VOLUME_MA_KEY: {
            "func_name": "_calculate_volume_ma", "params_map": {"length": "volume_ma_period"},
            "main_col_pattern": "VOL_SMA_{length}", "type": "decimal",
            "min_data_param_key": "length", "concat": False,
        },
    }

    def __init__(
        self, df: pd.DataFrame, logger: logging.Logger, config: Dict[str, Any], market_info: Dict[str, Any]
    ) -> None:
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
        self.interval = str(config.get("interval", "5"))
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)

        self.indicator_values: Dict[str, Union[Decimal, float, None]] = {}
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {}
        self.ta_column_names: Dict[str, str] = {}
        self.df_calculated: pd.DataFrame = pd.DataFrame()

        self.indicator_type_map: Dict[str, str] = {}
        for ind_key, details in self.INDICATOR_CONFIG.items():
            self.indicator_type_map[ind_key] = details.get("type", "float")
            if "multi_cols" in details:
                sub_type = details.get("type", "float")
                for sub_key in details["multi_cols"]:
                    self.indicator_type_map[sub_key] = sub_type

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"Input DataFrame for {self.symbol} is invalid or empty.")
        if not self.ccxt_interval:
            raise ValueError(f"Interval '{self.interval}' is invalid for {self.symbol}.")
        if not self.weights:
            self.logger.warning(f"Weight set '{self.active_weight_set_name}' is empty for {self.symbol}.")
        
        required_cols = [OPEN_KEY.lower(), HIGH_KEY.lower(), LOW_KEY.lower(), CLOSE_KEY.lower(), VOLUME_KEY.lower()]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame for {self.symbol} missing required columns: {missing}")

        self.df_original_ohlcv = df.copy()
        if self.df_original_ohlcv.index.tz is not None:
            self.df_original_ohlcv.index = self.df_original_ohlcv.index.tz_localize(None)

        self._validate_and_prepare_df_calculated()

        if not self.df_calculated.empty and isinstance(ta, ta. আর্জেন্টিনা): # Check if pandas_ta was properly imported
            self._calculate_all_indicators()
            self._update_latest_indicator_values()
            self.calculate_fibonacci_levels()
        elif isinstance(ta, DummyTA): # pandas_ta import failed
            self.logger.error("pandas_ta not installed. Cannot calculate technical indicators for %s.", self.symbol)
            self.indicator_values = {} # Ensure it's empty
        else: # DataFrame preparation failed
            self.logger.error("DataFrame preparation failed for %s. No indicators calculated.", self.symbol)
            self.indicator_values = {} # Ensure it's empty

    def _validate_and_prepare_df_calculated(self) -> None:
        """Validates OHLCV columns, converts to float64, and checks data length."""
        self.df_calculated = self.df_original_ohlcv.copy()
        required_cols = [OPEN_KEY.lower(), HIGH_KEY.lower(), LOW_KEY.lower(), CLOSE_KEY.lower(), VOLUME_KEY.lower()]

        for col in required_cols:
            if col not in self.df_calculated.columns:
                self.logger.critical("Column '%s' missing for %s. Invalidating DataFrame.", col, self.symbol)
                self.df_calculated = pd.DataFrame()
                return

            try:
                numeric_col = pd.to_numeric(self.df_calculated[col], errors="coerce")
                self.df_calculated[col] = numeric_col.astype("float64")
            except Exception as e:
                self.logger.critical(
                    "Failed to convert column '%s' to numeric/float64 for %s: %s", col, self.symbol, e, exc_info=True
                )
                self.df_calculated = pd.DataFrame()
                return

            if self.df_calculated[col].isna().any():
                nan_count = self.df_calculated[col].isna().sum()
                self.logger.warning("%d NaN(s) found in '%s' after conversion for %s.", nan_count, col, self.symbol)
        
        essential_ohlc_cols = [OPEN_KEY.lower(), HIGH_KEY.lower(), LOW_KEY.lower(), CLOSE_KEY.lower()]
        self.df_calculated.dropna(subset=essential_ohlc_cols, inplace=True)
        if self.df_calculated.empty:
            self.logger.error("DataFrame empty after dropping NaNs in OHLC for %s.", self.symbol)
            return

        enabled_cfg = self.config.get("indicators", {})
        max_lookback = 1
        for ind_key_cfg, ind_details in self.INDICATOR_CONFIG.items():
            # Check if indicator is enabled by its primary key (e.g., "ATR", "EMA_Short")
            # The config "indicators" uses lowercase keys like "atr", "ema_short"
            # So, we need to map or check against a lowercase version of ind_key_cfg
            # Example: if ind_key_cfg is "EMA_Short", check enabled_cfg.get("ema_short")
            indicator_enabled_key = ind_key_cfg.lower() # Standardize to lower for checking config

            if enabled_cfg.get(indicator_enabled_key, False):
                period_key = ind_details.get("min_data_param_key")
                config_key_in_map = ind_details.get("params_map", {}).get(period_key) if period_key else None
                
                param_val_to_check = None
                if config_key_in_map:
                    param_val_to_check = self.get_period(config_key_in_map)
                elif isinstance(ind_details.get("min_data"), int): # Direct min_data in config
                    param_val_to_check = ind_details["min_data"]

                if param_val_to_check is not None:
                    try:
                        current_lookback = int(Decimal(str(param_val_to_check)))
                        if current_lookback > 0:
                             max_lookback = max(max_lookback, current_lookback)
                    except (ValueError, TypeError, InvalidOperation):
                        self.logger.warning("Invalid period value '%s' for %s, cannot determine lookback.",
                                            param_val_to_check, ind_key_cfg)
        
        buffer = self.config.get("indicator_buffer_candles", 20)
        min_required_rows = max_lookback + buffer

        if len(self.df_calculated) < min_required_rows:
            self.logger.warning(
                "Insufficient valid data rows (%d) for %s. Need approx. %d "
                "(Max Lookback: %d + Buffer: %d). Some indicators might be NaN or inaccurate.",
                len(self.df_calculated), self.symbol, min_required_rows, max_lookback, buffer
            )
        else:
            self.logger.debug(
                "Data length (%d) sufficient for indicators for %s (Min required: %d).",
                len(self.df_calculated), self.symbol, min_required_rows
            )

    def get_period(self, key: str) -> Any:
        """Safely retrieves config value, using defaults, preserving type from default if possible."""
        config_val = self.config.get(key)
        if config_val is not None:
            default_val = DEFAULT_INDICATOR_PERIODS.get(key)
            if default_val is not None:
                try:
                    if isinstance(default_val, Decimal): return Decimal(str(config_val))
                    if isinstance(default_val, float): return float(config_val)
                    if isinstance(default_val, int): return int(config_val)
                except (ValueError, TypeError, InvalidOperation):
                    self.logger.warning(
                        "Could not convert config value '%s' for key '%s' to type %s. Using raw config value.",
                        config_val, key, type(default_val)
                    )
            return config_val
        return DEFAULT_INDICATOR_PERIODS.get(key)

    def _format_ta_column_name(self, pattern: str, params: Dict[str, Any]) -> str:
        """Formats TA column names, handling None and float/Decimal types for f-string compatibility."""
        fmt_params = {}
        for k, v in params.items():
            if v is None:
                fmt_params[k] = "DEF"  # Placeholder for None values
            elif isinstance(v, (float, Decimal)):
                if f"{{{k}:." in pattern:  # Check if a float format specifier (e.g., :.1f) is used
                    fmt_params[k] = float(v)
                else: # No specifier, make it a safe string for column name part
                    fmt_params[k] = str(v).replace(".", "_")
            else: # int, str
                fmt_params[k] = v
        try:
            return pattern.format(**fmt_params)
        except Exception as e: # Catch broader errors during formatting
            self.logger.error("Error formatting TA column pattern '%s' with params %s: %s", pattern, fmt_params, e)
            base_name = pattern.split("{")[0].rstrip("_") if "{" in pattern else pattern or "UNK_IND"
            param_suffix = "_".join(str(p).replace(".", "_") for p in params.values())
            return f"{base_name}_{param_suffix}_FORMAT_ERROR"

    def _calculate_volume_ma(self, df: pd.DataFrame, length: int) -> Optional[pd.Series]:
        """Calculates Simple Moving Average of volume. Custom TA function example."""
        vol_col = VOLUME_KEY.lower()
        if vol_col not in df.columns:
            self.logger.warning("Volume MA calc skipped for %s: '%s' column missing.", self.symbol, vol_col)
            return None
        if not (isinstance(length, int) and length > 0):
            self.logger.warning("Volume MA calc skipped for %s: Invalid length %s.", self.symbol, length)
            return None
        
        volume_series = df[vol_col].astype(float).fillna(0)
        if len(volume_series) < length:
            self.logger.debug("Insufficient data for Volume MA %d on %s.", length, self.symbol)
            return pd.Series(np.nan, index=df.index, name=f"VOL_SMA_{length}") # Return Series of NaNs matching index
        
        try:
            if isinstance(ta, DummyTA): # pandas_ta not available
                 raise ImportError("pandas_ta is not installed or failed to import.")
            return ta.sma(volume_series, length=length)
        except Exception as e:
            self.logger.error("Error calculating SMA for volume on %s: %s", self.symbol, e, exc_info=True)
            return None # Return None or Series of NaNs

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators and stores results in df_calculated."""
        if self.df_calculated.empty:
            self.logger.warning("Skipping indicator calculation for %s: DataFrame is empty/invalid.", self.symbol)
            return
        if isinstance(ta, DummyTA):
            self.logger.error("Skipping indicator calculation for %s: pandas_ta library is not available.", self.symbol)
            return

        df_work = self.df_calculated # Operate on the prepared DataFrame copy
        enabled_cfg = self.config.get("indicators", {})
        self.ta_column_names = {} # Reset mapping for this run

        for ind_key_cfg, ind_details in self.INDICATOR_CONFIG.items():
            # Config keys are lowercase (e.g., "atr"), ind_key_cfg from INDICATOR_CONFIG might be "ATR"
            if not enabled_cfg.get(ind_key_cfg.lower(), False):
                continue

            params_for_ta = {}
            valid_params = True
            for ta_param_name, config_param_key in ind_details.get("params_map", {}).items():
                param_value = self.get_period(config_param_key)
                if param_value is None: # Critical parameter missing
                    valid_params = False
                    self.logger.warning("Missing critical param %s for %s on %s.", config_param_key, ind_key_cfg, self.symbol)
                    break
                try: # Convert to float/int for pandas-ta
                    if isinstance(param_value, Decimal): params_for_ta[ta_param_name] = float(param_value)
                    elif isinstance(param_value, str): # Attempt conversion if string
                        params_for_ta[ta_param_name] = float(param_value) if "." in param_value else int(param_value)
                    elif isinstance(param_value, (int, float)): params_for_ta[ta_param_name] = param_value
                    else: raise TypeError(f"Unsupported param type {type(param_value)}")
                except (ValueError, TypeError) as e:
                    valid_params = False
                    self.logger.error("Param conversion error for %s (%s) of %s: %s", config_param_key, param_value, ind_key_cfg, e)
                    break
            if not valid_params:
                self.logger.warning("Skipping indicator %s for %s due to invalid/missing parameters.", ind_key_cfg, self.symbol)
                continue

            try:
                func_name = ind_details["func_name"]
                func_obj = getattr(self, func_name, None) if func_name.startswith("_") else getattr(ta, func_name, None)

                if not callable(func_obj):
                    self.logger.error("TA function '%s' not found or not callable for %s.", func_name, self.symbol)
                    continue

                min_len_key = ind_details.get("min_data_param_key", "length") # Default to 'length' if specific key not given
                min_len_val_from_params = params_for_ta.get(min_len_key)
                min_len_val_from_details = ind_details.get("min_data")
                
                # Determine minimum data length required
                min_len_required = 1
                if min_len_val_from_params is not None:
                    min_len_required = int(min_len_val_from_params)
                elif isinstance(min_len_val_from_details, int):
                    min_len_required = min_len_val_from_details
                
                if len(df_work) < min_len_required:
                    self.logger.debug("Skipping %s for %s: data length %d < required %d.", ind_key_cfg, self.symbol, len(df_work), min_len_required)
                    continue

                # Prepare inputs for TA function
                # Standard OHLCV column names expected by pandas-ta
                ohlcv_cols_map = {
                    'open': OPEN_KEY.lower(), 'high': HIGH_KEY.lower(), 
                    'low': LOW_KEY.lower(), 'close': CLOSE_KEY.lower(), 
                    'volume': VOLUME_KEY.lower()
                }

                result = None
                if func_name.startswith("_"): # Custom method like _calculate_volume_ma
                    result = func_obj(df_work, **params_for_ta)
                elif ind_details.get("pass_close_only", False):
                    result = func_obj(close=df_work[ohlcv_cols_map['close']], **params_for_ta)
                else: # Standard pandas-ta function call
                    sig = inspect.signature(func_obj)
                    func_args_to_pass = {}
                    # Map standard TA arg names (open, high, low, close, volume) to df_work columns
                    for ta_arg_name, df_col_name in ohlcv_cols_map.items():
                        if ta_arg_name in sig.parameters and df_col_name in df_work.columns:
                            func_args_to_pass[ta_arg_name] = df_work[df_col_name]
                    
                    if not func_args_to_pass and 'close' in sig.parameters and ohlcv_cols_map['close'] in df_work.columns:
                        # Fallback if no OHLCV args matched, but 'close' is expected
                        func_args_to_pass['close'] = df_work[ohlcv_cols_map['close']]

                    if func_args_to_pass: # Call with specific series
                        result = func_obj(**func_args_to_pass, **params_for_ta)
                    else: # Fallback: Try passing the relevant subset of the DataFrame
                          # This assumes func_obj can handle a DataFrame input if specific series aren't in its signature.
                          # pandas-ta functions usually prefer series, or are called via df.ta accessor.
                        relevant_df_cols = [c for c in ohlcv_cols_map.values() if c in df_work.columns]
                        if relevant_df_cols:
                            result = func_obj(df_work[relevant_df_cols], **params_for_ta)
                        else:
                            self.logger.warning("Could not determine appropriate arguments for TA function %s for %s.", func_name, self.symbol)
                            continue # Skip this indicator if args cannot be prepared
                
                if result is None:
                    self.logger.debug("Indicator %s for %s returned None.", ind_key_cfg, self.symbol)
                    continue
                
                # Integrate results
                should_concat = ind_details.get("concat", False)
                is_multi_col_output = "multi_cols" in ind_details

                if isinstance(result, pd.Series):
                    col_name_pattern = ind_details.get("main_col_pattern")
                    if not col_name_pattern:
                        self.logger.error("Missing 'main_col_pattern' for Series result of %s for %s.", ind_key_cfg, self.symbol)
                        continue
                    actual_col_name = self._format_ta_column_name(col_name_pattern, params_for_ta)
                    df_work[actual_col_name] = result.astype("float64") # Assign or overwrite
                    self.ta_column_names[ind_key_cfg] = actual_col_name

                elif isinstance(result, pd.DataFrame):
                    if not should_concat and not is_multi_col_output : # Mismatch: expected Series, got DataFrame
                        self.logger.error("Indicator %s (concat=False, non-multi-col) for %s returned DataFrame. Check config.", ind_key_cfg, self.symbol)
                        continue
                    
                    # Ensure result columns are float64 and prepare for merging/concatenating
                    processed_result_df = pd.DataFrame(index=result.index)
                    for col in result.columns:
                        try:
                            processed_result_df[col] = pd.to_numeric(result[col], errors='coerce').astype('float64')
                        except Exception as e_conv:
                            self.logger.warning("Could not convert column %s from %s result to float64 for %s: %s",
                                                col, ind_key_cfg, self.symbol, e_conv)
                    
                    # Drop existing columns in df_work that would be duplicated by result_df columns
                    cols_to_drop_from_df_work = [c for c in processed_result_df.columns if c in df_work.columns]
                    if cols_to_drop_from_df_work:
                        df_work.drop(columns=cols_to_drop_from_df_work, inplace=True, errors='ignore')
                    
                    df_work = pd.concat([df_work, processed_result_df], axis=1, copy=False)

                    if is_multi_col_output: # Map internal keys if multi-column output defined
                        for internal_key, col_pattern in ind_details["multi_cols"].items():
                            actual_col_name = self._format_ta_column_name(col_pattern, params_for_ta)
                            if actual_col_name in df_work.columns:
                                self.ta_column_names[internal_key] = actual_col_name
                            else: # Attempt partial match or log warning
                                self.logger.warning(
                                    "Mapped column '%s' for '%s' of %s not found in DataFrame for %s. Check ta version/column names.",
                                    actual_col_name, internal_key, ind_key_cfg, self.symbol
                                )
                                # Simplified partial match logic can be added here if necessary
                else:
                    self.logger.warning("Indicator %s for %s returned unexpected type: %s", ind_key_cfg, self.symbol, type(result))

            except Exception as e:
                self.logger.error("Error calculating indicator %s for %s: %s", ind_key_cfg, self.symbol, e, exc_info=True)
        
        self.df_calculated = df_work
        self.logger.debug("Indicator calculation complete for %s. Final columns: %s", self.symbol, list(self.df_calculated.columns))
        self.logger.debug("TA Column Map for %s: %s", self.symbol, self.ta_column_names)

    def _update_latest_indicator_values(self) -> None:
        """Updates dict with latest indicator values, converting types based on config."""
        if self.df_calculated.empty:
            self.logger.warning("Cannot update latest values for %s: calculated DataFrame is empty.", self.symbol)
            self.indicator_values = {}
            return
        try:
            if len(self.df_calculated) == 0 or len(self.df_original_ohlcv) == 0: # Should be caught by .empty but defensive
                self.logger.error("DataFrame has zero length, cannot get latest row for %s.", self.symbol)
                self.indicator_values = {}
                return

            latest_ind_row = self.df_calculated.iloc[-1]
            latest_ohlcv_row = self.df_original_ohlcv.iloc[-1]

            latest_values: Dict[str, Union[Decimal, float, None]] = {}

            # Initialize all known indicator config keys and OHLCV keys to None
            for key in list(self.indicator_type_map.keys()) + [OPEN_KEY, HIGH_KEY, LOW_KEY, CLOSE_KEY, VOLUME_KEY]:
                latest_values[key] = None

            # Process TA indicators from df_calculated
            for internal_key, actual_col_name in self.ta_column_names.items():
                if actual_col_name in latest_ind_row.index:
                    val = latest_ind_row[actual_col_name]
                    target_type = self.indicator_type_map.get(internal_key, "float") # Default to float
                    if pd.notna(val):
                        try:
                            latest_values[internal_key] = Decimal(str(val)) if target_type == "decimal" else float(val)
                        except (InvalidOperation, TypeError, ValueError) as conv_err:
                            self.logger.debug("Conversion error for %s ('%s') to %s for %s: %s",
                                             internal_key, val, target_type, self.symbol, conv_err)
                            # latest_values[internal_key] remains None
                    # else: val is NaN, latest_values[internal_key] remains None
                # else: column not found, latest_values[internal_key] remains None

            # Process OHLCV from original df (store as Decimal for precision)
            ohlcv_map_to_df_cols = {
                OPEN_KEY: OPEN_KEY.lower(), HIGH_KEY: HIGH_KEY.lower(),
                LOW_KEY: LOW_KEY.lower(), CLOSE_KEY: CLOSE_KEY.lower(),
                VOLUME_KEY: VOLUME_KEY.lower()
            }
            for display_key, original_df_col_name in ohlcv_map_to_df_cols.items():
                if original_df_col_name in latest_ohlcv_row.index:
                    val_ohlcv = latest_ohlcv_row[original_df_col_name]
                    if pd.notna(val_ohlcv):
                        try:
                            latest_values[display_key] = Decimal(str(val_ohlcv))
                        except (InvalidOperation, TypeError, ValueError) as conv_err:
                            self.logger.debug("OHLCV conversion error for %s ('%s') for %s: %s",
                                             display_key, val_ohlcv, self.symbol, conv_err)
                            # latest_values[display_key] remains None
                    # else: val_ohlcv is NaN, latest_values[display_key] remains None

            self.indicator_values = latest_values

            # Log summary of key values for debugging
            price_prec = get_price_precision(self.market_info, self.logger)
            log_items = []
            keys_to_log = [CLOSE_KEY, ATR_KEY, EMA_SHORT_KEY, EMA_LONG_KEY, RSI_KEY, STOCHRSI_K_KEY]
            for k in keys_to_log:
                v = self.indicator_values.get(k)
                if v is not None:
                    prec_to_use = price_prec + 2 if k == ATR_KEY else \
                                  price_prec if isinstance(v, Decimal) else 4 # Default for floats
                    try:
                        log_items.append(f"{k}={v:.{prec_to_use}f}")
                    except TypeError: # Fallback if formatting fails (e.g. non-numeric string accidentally stored)
                        log_items.append(f"{k}={v}")
                else:
                    log_items.append(f"{k}=N/A")
            self.logger.debug("Latest indicator values for %s: %s", self.symbol, ", ".join(log_items))

        except IndexError:
            self.logger.error("IndexError while getting latest row for indicator values on %s.", self.symbol)
            self.indicator_values = {} # Ensure it's empty on error
        except Exception as e:
            self.logger.error("Error updating latest indicator values for %s: %s", self.symbol, e, exc_info=True)
            self.indicator_values = {} # Ensure it's empty on error

    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels based on high/low over a window."""
        window = window or self.get_period("fibonacci_window")
        if not (isinstance(window, int) and window > 0):
            self.logger.debug("Fibonacci calc skipped for %s: Invalid window %s.", self.symbol, window)
            return {}
        if len(self.df_original_ohlcv) < window:
            self.logger.debug("Fibonacci calc skipped for %s: Data length %d < window %d.",
                             self.symbol, len(self.df_original_ohlcv), window)
            return {}

        df_slice = self.df_original_ohlcv.tail(window)
        try:
            high_series = pd.to_numeric(df_slice[HIGH_KEY.lower()], errors="coerce").dropna()
            low_series = pd.to_numeric(df_slice[LOW_KEY.lower()], errors="coerce").dropna()

            if high_series.empty or low_series.empty:
                self.logger.debug("Fibonacci calc for %s: No valid high/low data in window.", self.symbol)
                return {}

            period_high = Decimal(str(high_series.max()))
            period_low = Decimal(str(low_series.min()))
            diff = period_high - period_low

            levels = {}
            price_prec = get_price_precision(self.market_info, self.logger)
            min_tick = get_min_tick_size(self.market_info, self.logger)
            # Quantization unit must be positive
            quantize_unit = min_tick if min_tick and min_tick > Decimal(0) else Decimal(f"1e-{max(0, price_prec)}") # Ensure positive precision

            if diff > Decimal(0):
                for level_pct_decimal in FIB_LEVELS: # Assuming FIB_LEVELS are Decimals
                    price_raw = period_high - (diff * level_pct_decimal)
                    # Quantize to the nearest valid tick size or precision
                    levels[f"Fib_{level_pct_decimal * 100:.1f}%"] = \
                        (price_raw / quantize_unit).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantize_unit
            else: # Flat price range, all levels are the same
                quantized_level = (period_high / quantize_unit).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantize_unit
                for level_pct_decimal in FIB_LEVELS:
                    levels[f"Fib_{level_pct_decimal * 100:.1f}%"] = quantized_level
            
            self.fib_levels_data = levels
            self.logger.debug("Calculated %d Fibonacci levels for %s.", len(levels), self.symbol)
            return levels
        except Exception as e:
            self.logger.error("Fibonacci calculation error for %s: %s", self.symbol, e, exc_info=False) # Less verbose for calc errors
            self.fib_levels_data = {}
            return {}

    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 5) -> List[Tuple[str, Decimal]]:
        """Finds N nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
            self.logger.debug("Cannot get nearest Fib levels for %s: No Fib data available.", self.symbol)
            return []
        if not (isinstance(current_price, Decimal) and current_price.is_finite() and current_price > 0):
            self.logger.debug("Cannot get nearest Fib levels for %s: Invalid current price %s.", self.symbol, current_price)
            return []
        if num_levels <= 0:
            return []

        try:
            # Ensure levels are valid Decimals and positive before calculating distance
            valid_fib_levels = {
                name: level_price for name, level_price in self.fib_levels_data.items()
                if isinstance(level_price, Decimal) and level_price.is_finite() and level_price > 0
            }
            if not valid_fib_levels:
                self.logger.debug("No valid positive Fibonacci levels to compare against for %s.", self.symbol)
                return []

            distances = [
                {"name": name, "level": price, "distance": abs(current_price - price)}
                for name, price in valid_fib_levels.items()
            ]
            
            distances.sort(key=lambda x: x["distance"])
            return [(item["name"], item["level"]) for item in distances[:num_levels]]
        except Exception as e:
            self.logger.error("Error finding nearest Fibonacci levels for %s: %s", self.symbol, e, exc_info=False)
            return []

    def calculate_ema_alignment_score(self) -> float:
        """Calculates score based on EMA alignment and price position relative to EMAs."""
        ema_s_val = self.indicator_values.get(EMA_SHORT_KEY)
        ema_l_val = self.indicator_values.get(EMA_LONG_KEY)
        close_val = self.indicator_values.get(CLOSE_KEY)

        # Ensure all values are valid Decimals for comparison
        if not all(isinstance(v, Decimal) and v.is_finite() for v in [ema_s_val, ema_l_val, close_val]):
            return np.nan # Not enough data or invalid data

        # Type hinting for clarity after check
        ema_s: Decimal = ema_s_val
        ema_l: Decimal = ema_l_val
        close: Decimal = close_val
        
        if close > ema_s > ema_l: return 1.0  # Strong bullish alignment
        if close < ema_s < ema_l: return -1.0 # Strong bearish alignment
        return 0.0  # Neutral or mixed alignment

    @staticmethod
    def _format_price_or_na(price: Optional[Decimal], precision: int) -> str:
        """Helper to format Decimal price for logging, or return 'N/A'."""
        if price is None or not isinstance(price, Decimal) or not price.is_finite():
            return "N/A"
        try:
            return f"{price:.{max(0, precision)}f}" # Ensure precision is not negative
        except (TypeError, ValueError): # Should not happen with Decimal if finite
            MODULE_LOGGER.warning("Failed to format price %s with precision %d.", price, precision)
            return "ErrorFmt"

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict[str, Any]]) -> str:
        """Generates BUY/SELL/HOLD signal based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1} # Default to HOLD
        final_score, total_weight_abs_sum = Decimal("0"), Decimal("0")
        active_checks_count, nan_checks_count = 0, 0
        debug_scores: Dict[str, str] = {}

        if not self.indicator_values:
            self.logger.warning("Cannot generate signal for %s: No indicator values available. Defaulting to HOLD.", self.symbol)
            return "HOLD"

        min_active_indicators = self.config.get("min_active_indicators_for_signal", 7)
        # Count valid (non-NaN, non-None) core indicator values
        valid_core_indicators_count = sum(
            1 for key_const in self.INDICATOR_CONFIG # Iterate over defined indicators in INDICATOR_CONFIG
            if pd.notna(self.indicator_values.get(key_const)) # Check if this key has a valid value
        )
        
        if valid_core_indicators_count < min_active_indicators:
            self.logger.warning(
                "Signal for %s: Only %d valid core indicators, less than required %d. Defaulting to HOLD.",
                self.symbol, valid_core_indicators_count, min_active_indicators
            )
            return "HOLD"

        if not (isinstance(current_price, Decimal) and current_price.is_finite() and current_price > 0):
            self.logger.warning("Invalid current price (%s) for signal generation for %s. Defaulting to HOLD.",
                                current_price, self.symbol)
            return "HOLD"

        active_weights_map = self.weights
        if not active_weights_map:
            self.logger.warning("Weight set '%s' is empty for %s. Defaulting to HOLD.",
                                self.active_weight_set_name, self.symbol)
            return "HOLD"

        # Calculate weighted score
        for ind_key_lower, weight_val_from_config in active_weights_map.items():
            # Ensure indicator is enabled in main config (indicator key in config.indicators is lowercase)
            if not self.config.get("indicators", {}).get(ind_key_lower, False):
                continue

            try:
                weight = Decimal(str(weight_val_from_config))
            except (InvalidOperation, TypeError):
                self.logger.warning("Invalid weight '%s' for indicator '%s' for %s. Skipping.",
                                    weight_val_from_config, ind_key_lower, self.symbol)
                continue
            
            if weight == Decimal(0): # Zero weight, no contribution
                continue

            check_method_name = f"_check_{ind_key_lower}"
            check_method_obj = getattr(self, check_method_name, None)
            if not callable(check_method_obj):
                self.logger.debug("Check method %s not found or not callable for %s.", check_method_name, self.symbol)
                continue

            indicator_score_float = np.nan # Default if method fails or returns NaN
            try:
                if ind_key_lower == "orderbook": # Special case for orderbook
                    indicator_score_float = check_method_obj(orderbook_data, current_price)
                else:
                    indicator_score_float = check_method_obj()
            except Exception as e:
                self.logger.error("Error in check method %s for %s: %s", check_method_name, self.symbol, e, exc_info=True)

            debug_scores[ind_key_lower] = f"{indicator_score_float:.3f}" if pd.notna(indicator_score_float) else "NaN"
            
            if pd.notna(indicator_score_float):
                try:
                    # Convert float score to Decimal for precision in weighted sum
                    indicator_score_decimal = Decimal(str(indicator_score_float))
                    # Clamp score to [-1, 1] range
                    clamped_score = max(Decimal("-1"), min(Decimal("1"), indicator_score_decimal))
                    
                    final_score += clamped_score * weight
                    total_weight_abs_sum += abs(weight) # Sum of absolute weights for normalization or thresholding
                    active_checks_count += 1
                except (InvalidOperation, TypeError):
                    nan_checks_count += 1 # Failed to convert score to Decimal
                    self.logger.debug("Score conversion to Decimal failed for %s: %s on %s.",
                                      ind_key_lower, indicator_score_float, self.symbol)
            else: # Score was NaN
                nan_checks_count += 1
        
        # Determine final signal
        final_signal_str = "HOLD"
        # Use Decimal for threshold comparison
        signal_threshold = Decimal(str(self.get_period("signal_score_threshold") or "0.7"))

        if total_weight_abs_sum > Decimal(0): # Only determine BUY/SELL if weights contributed
            # Normalized score could be used: normalized_score = final_score / total_weight_abs_sum
            # Here, final_score is used directly against the threshold
            if final_score >= signal_threshold:
                final_signal_str = "BUY"
            elif final_score <= -signal_threshold:
                final_signal_str = "SELL"
        elif active_checks_count > 0 : # Checks were made, but total weight sum was zero
            self.logger.warning("Total weight sum is zero for %s, though %d checks were active. Defaulting to HOLD.",
                                self.symbol, active_checks_count)
        else: # No indicators contributed (e.g. all disabled, all had NaN scores, or no check methods)
            self.logger.warning("No indicators contributed to the score for %s. Defaulting to HOLD.", self.symbol)

        price_prec = get_price_precision(self.market_info, self.logger)
        self.logger.info(
            "Signal (%s @ %s): Set='%s', Checks[Act:%d,NaN:%d], WeightSum=%.2f, Score=%.4f (Th:%.2f) ==> %s",
            self.symbol, self._format_price_or_na(current_price, price_prec),
            self.active_weight_set_name, active_checks_count, nan_checks_count,
            total_weight_abs_sum, final_score, signal_threshold,
            format_signal(final_signal_str) # Use utility for colored/formatted output
        )
        self.logger.debug("Individual Scores for %s: %s", self.symbol, debug_scores)

        self.signals = {
            "BUY": int(final_signal_str == "BUY"),
            "SELL": int(final_signal_str == "SELL"),
            "HOLD": int(final_signal_str == "HOLD"),
        }
        return final_signal_str

    # --- Individual Indicator Check Methods (_check_*) ---
    # These methods return a float score, typically between -1.0 and 1.0.
    # np.nan should be returned if a score cannot be calculated.

    def _check_ema_alignment(self) -> float:
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        mom_val = self.indicator_values.get(MOMENTUM_KEY) # float or None
        close_val = self.indicator_values.get(CLOSE_KEY)   # Decimal or None

        if pd.isna(mom_val) or not (isinstance(close_val, Decimal) and close_val.is_finite() and close_val > 0):
            return np.nan
        
        try:
            # mom_val is float, convert to Decimal for precise division if needed, or use float arithmetic
            mom_decimal = Decimal(str(mom_val))
            # Score relative to price; needs scaling.
            # Example: if momentum is 1% of price, score is 0.01. Scale by 10 => 0.1.
            relative_momentum = mom_decimal / close_val 
            score = float(max(Decimal("-1"), min(Decimal("1"), relative_momentum * 10))) # Scaled and clamped
            return score
        except (InvalidOperation, TypeError):
            return 0.0 # Fallback on conversion error

    def _check_volume_confirmation(self) -> float:
        vol_val = self.indicator_values.get(VOLUME_KEY)       # Decimal or None
        vol_ma_val = self.indicator_values.get(VOLUME_MA_KEY) # Decimal or None
        
        multiplier_str = str(self.get_period("volume_confirmation_multiplier") or "1.5")
        try:
            confirmation_multiplier = Decimal(multiplier_str)
        except InvalidOperation:
            self.logger.warning("Invalid volume_confirmation_multiplier: %s for %s.", multiplier_str, self.symbol)
            return np.nan

        if not all(isinstance(v, Decimal) and v.is_finite() and v >= 0 for v in [vol_val, vol_ma_val]) \
           or confirmation_multiplier <= 0:
            return np.nan

        vol: Decimal = vol_val
        vol_ma: Decimal = vol_ma_val

        if vol_ma == Decimal(0): # Avoid division by zero; treat as neutral or no signal
            return 0.0 
        
        volume_ratio = vol / vol_ma
        # Score based on volume being significantly above or below its MA
        if volume_ratio > confirmation_multiplier: return 1.0 # Strong volume confirmation
        # Inverse multiplier for low volume check
        low_volume_threshold = Decimal(1) / confirmation_multiplier if confirmation_multiplier > 0 else Decimal(0)
        if volume_ratio < low_volume_threshold : return -0.4 # Low volume, potential lack of conviction
        return 0.0 # Neutral volume

    def _check_stoch_rsi(self) -> float:
        k_val = self.indicator_values.get(STOCHRSI_K_KEY) # float or None
        d_val = self.indicator_values.get(STOCHRSI_D_KEY) # float or None

        if pd.isna(k_val) or pd.isna(d_val):
            return np.nan
        
        # Values are already floats from indicator_values
        k, d = k_val, d_val

        oversold_thresh = float(self.get_period("stoch_rsi_oversold_threshold") or 20)
        overbought_thresh = float(self.get_period("stoch_rsi_overbought_threshold") or 80)
        
        score = 0.0
        if k < oversold_thresh and d < oversold_thresh: score = 1.0  # Oversold condition
        elif k > overbought_thresh and d > overbought_thresh: score = -1.0 # Overbought condition
        
        # Consider K/D crossover or relative position
        if k > d and score >= 0: # Bullish crossover or K above D in non-overbought
            score = max(score, 0.4) # Reinforce bullish or assign mild bullish
        if k < d and score <= 0: # Bearish crossover or K below D in non-oversold
            score = min(score, -0.4) # Reinforce bearish or assign mild bearish
            
        # Dampen score if in neutral zone (e.g., between 40-60)
        if 40 < k < 60 and 40 < d < 60:
            score *= 0.5
        return score

    def _check_rsi(self) -> float:
        rsi_val = self.indicator_values.get(RSI_KEY) # float or None
        if pd.isna(rsi_val):
            return np.nan
        # rsi_val is already float if not NaN

        oversold_thresh = float(self.get_period("rsi_oversold_threshold") or 30)
        overbought_thresh = float(self.get_period("rsi_overbought_threshold") or 70)
        
        if rsi_val <= oversold_thresh: return 1.0
        if rsi_val >= overbought_thresh: return -1.0
        
        # Linear interpolation for scores between thresholds and neutral (50)
        # Scale: 0 at 50, 0.8 at threshold (e.g. 30 or 70)
        if oversold_thresh < rsi_val < 50:
            return (50.0 - rsi_val) / (50.0 - oversold_thresh) * 0.8
        if 50 <= rsi_val < overbought_thresh: # RSI at 50 is 0.0
             # (rsi_val - 50.0) / (overbought_thresh - 50.0) would be positive, so multiply by -0.8
            return (rsi_val - 50.0) / (overbought_thresh - 50.0) * -0.8
        return 0.0 # Should be covered by conditions above, but as a fallback

    def _check_cci(self) -> float:
        cci_val = self.indicator_values.get(CCI_KEY) # float or None
        if pd.isna(cci_val):
            return np.nan
        # cci_val is already float

        strong_oversold = float(self.get_period("cci_strong_oversold") or -150)
        moderate_overbought = float(self.get_period("cci_moderate_overbought") or 100)
        strong_overbought = float(self.get_period("cci_strong_overbought") or 150)
        moderate_oversold = float(self.get_period("cci_moderate_oversold") or -100)

        if cci_val <= strong_oversold: return 1.0
        if cci_val >= strong_overbought: return -1.0
        if cci_val <= moderate_oversold: return 0.6 # Using cci_val < moderate_oversold (e.g. < -100)
        if cci_val >= moderate_overbought: return -0.6 # Using cci_val > moderate_overbought (e.g. > 100)
        return 0.0 # CCI between -100 and 100 (or configured moderate levels)

    def _check_wr(self) -> float: # Williams %R
        wr_val = self.indicator_values.get(WILLIAMS_R_KEY) # float or None
        if pd.isna(wr_val):
            return np.nan
        # wr_val is already float

        oversold_thresh = float(self.get_period("wr_oversold_threshold") or -80) # e.g., -80
        overbought_thresh = float(self.get_period("wr_overbought_threshold") or -20) # e.g., -20
        
        if wr_val <= oversold_thresh: return 1.0  # e.g., WR <= -80 is oversold
        if wr_val >= overbought_thresh: return -1.0 # e.g., WR >= -20 is overbought
        
        # Midpoint for Williams %R is typically -50
        mid_point = (oversold_thresh + overbought_thresh) / 2.0 # e.g., (-80 + -20)/2 = -50
        
        # Score interpolation (0.7 strength)
        if oversold_thresh < wr_val < mid_point: # e.g., -80 < WR < -50
            # As WR moves from oversold_thresh towards mid_point, score decreases from 0.7 to 0
            return (wr_val - mid_point) / (oversold_thresh - mid_point) * 0.7
        if mid_point <= wr_val < overbought_thresh: # e.g., -50 <= WR < -20
            # As WR moves from mid_point towards overbought_thresh, score decreases from 0 to -0.7
            return (wr_val - mid_point) / (overbought_thresh - mid_point) * -0.7
        return 0.0

    def _check_psar(self) -> float:
        psar_l_val = self.indicator_values.get(PSAR_LONG_KEY)   # Decimal or None
        psar_s_val = self.indicator_values.get(PSAR_SHORT_KEY)  # Decimal or None
        close_val = self.indicator_values.get(CLOSE_KEY)      # Decimal or None

        if not isinstance(close_val, Decimal) or not close_val.is_finite():
            return np.nan # Close price is essential

        # PSAR gives a long value XOR a short value. One is NaN, the other is the PSAR level.
        # If psar_l_val is a valid Decimal, we are in an uptrend (PSAR is below price).
        # If psar_s_val is a valid Decimal, we are in a downtrend (PSAR is above price).
        
        is_uptrend_signal = isinstance(psar_l_val, Decimal) and psar_l_val.is_finite() # and close_val > psar_l_val
        is_downtrend_signal = isinstance(psar_s_val, Decimal) and psar_s_val.is_finite() # and close_val < psar_s_val

        if is_uptrend_signal and not is_downtrend_signal : return 1.0
        if is_downtrend_signal and not is_uptrend_signal : return -1.0
        return 0.0 # Ambiguous or no clear PSAR signal (e.g. both NaN, or somehow both valid)

    def _check_sma_10(self) -> float:
        sma_val = self.indicator_values.get(SMA10_KEY)     # Decimal or None
        close_val = self.indicator_values.get(CLOSE_KEY)   # Decimal or None

        if not all(isinstance(v, Decimal) and v.is_finite() for v in [sma_val, close_val]):
            return np.nan
        
        sma: Decimal = sma_val
        close: Decimal = close_val

        if sma == Decimal(0): return 0.0 # Avoid division by zero if SMA is zero
        
        # Percentage difference from SMA, scaled.
        # (close - sma) / sma gives fractional difference. Scale by 10.
        difference_scaled = (close - sma) / sma * 10
        return float(max(Decimal("-1"), min(Decimal("1"), difference_scaled))) # Clamp to [-1, 1]

    def _check_vwap(self) -> float:
        vwap_val = self.indicator_values.get(VWAP_KEY)    # Decimal or None
        close_val = self.indicator_values.get(CLOSE_KEY)  # Decimal or None

        if not all(isinstance(v, Decimal) and v.is_finite() for v in [vwap_val, close_val]):
            return np.nan

        vwap: Decimal = vwap_val
        close: Decimal = close_val

        if vwap == Decimal(0): return 0.0
        
        # Percentage difference from VWAP, scaled more aggressively (e.g. by 15).
        difference_scaled = (close - vwap) / vwap * 15
        return float(max(Decimal("-1"), min(Decimal("1"), difference_scaled)))

    def _check_mfi(self) -> float: # Money Flow Index
        mfi_val = self.indicator_values.get(MFI_KEY) # float or None
        if pd.isna(mfi_val):
            return np.nan
        # mfi_val is already float

        oversold_thresh = float(self.get_period("mfi_oversold_threshold") or 20)
        overbought_thresh = float(self.get_period("mfi_overbought_threshold") or 80)
        
        if mfi_val <= oversold_thresh: return 1.0
        if mfi_val >= overbought_thresh: return -1.0
        # Could add interpolated scores for MFI as well, similar to RSI, if desired.
        # For now, simple thresholding.
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bb_l_val = self.indicator_values.get(BB_LOWER_KEY)    # Decimal or None
        bb_m_val = self.indicator_values.get(BB_MIDDLE_KEY)   # Decimal or None
        bb_u_val = self.indicator_values.get(BB_UPPER_KEY)    # Decimal or None
        close_val = self.indicator_values.get(CLOSE_KEY)     # Decimal or None

        if not all(isinstance(v, Decimal) and v.is_finite() for v in [bb_l_val, bb_m_val, bb_u_val, close_val]):
            return np.nan

        bb_l: Decimal = bb_l_val
        bb_m: Decimal = bb_m_val
        bb_u: Decimal = bb_u_val
        close: Decimal = close_val

        if close <= bb_l: return 1.0 # Price at or below lower band (potential buy/reversal)
        if close >= bb_u: return -1.0 # Price at or above upper band (potential sell/reversal)
        
        band_width = bb_u - bb_l
        if band_width > Decimal(0):
            # Position within the bands, relative to the middle band.
            # Score is scaled by 0.7; 0 at middle band, +/-0.7 at bands.
            # (close - bb_m) is distance from middle. (band_width / 2) is distance from middle to a band.
            relative_position = (close - bb_m) / (band_width / Decimal(2))
            # Clamp to ensure score is within [-0.7, 0.7] if price is within bands.
            score = max(Decimal("-0.7"), min(Decimal("0.7"), relative_position * Decimal("0.7")))
            return float(score)
        return 0.0 # Undefined or zero width (e.g. flat price)

    def _check_orderbook(self, orderbook_data: Optional[Dict[str, Any]], current_price: Decimal) -> float:
        if not orderbook_data:
            return np.nan
        
        try:
            bids = orderbook_data.get("bids", []) # List of [price, quantity]
            asks = orderbook_data.get("asks", []) # List of [price, quantity]

            if not bids or not asks: return np.nan

            num_levels_to_check = self.config.get("orderbook_check_levels", 10)
            
            # Sum quantities from top N levels, ensuring data is valid
            total_bid_quantity = sum(Decimal(str(b[1])) for b in bids[:num_levels_to_check]
                                     if len(b) == 2 and b[1] is not None and pd.notna(b[1]))
            total_ask_quantity = sum(Decimal(str(a[1])) for a in asks[:num_levels_to_check]
                                     if len(a) == 2 and a[1] is not None and pd.notna(a[1]))

            total_quantity_at_levels = total_bid_quantity + total_ask_quantity
            if total_quantity_at_levels == Decimal(0):
                return 0.0 # No liquidity or balanced at zero
            
            # Order Book Imbalance (OBI)
            obi = (total_bid_quantity - total_ask_quantity) / total_quantity_at_levels
            # Clamp OBI to [-1, 1] (should inherently be, but good practice)
            return float(max(Decimal("-1"), min(Decimal("1"), obi)))
        except (InvalidOperation, TypeError, ValueError, IndexError) as e:
            self.logger.warning("Order book analysis error for %s: %s", self.symbol, e)
            return np.nan # Error during processing

    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculates potential Take Profit (TP) and Stop Loss (SL) levels."""
        if signal not in ["BUY", "SELL"] or \
           not (isinstance(entry_price_estimate, Decimal) and 
                entry_price_estimate.is_finite() and entry_price_estimate > 0):
            self.logger.warning("TP/SL calc for %s: Invalid entry price or signal. Entry: %s, Signal: %s",
                                self.symbol, entry_price_estimate, signal)
            return entry_price_estimate, None, None # Return original entry, no TP/SL

        atr_val = self.indicator_values.get(ATR_KEY) # Decimal or None
        
        # Validate or fallback ATR
        if not (isinstance(atr_val, Decimal) and atr_val.is_finite() and atr_val > 0):
            self.logger.warning("TP/SL Calc for %s: ATR invalid (%s). Attempting default ATR%%.", self.symbol, atr_val)
            default_atr_pct_str = str(self.get_period("default_atr_percentage_of_price") or "0.0")
            try:
                default_atr_pct = Decimal(default_atr_pct_str)
                if default_atr_pct > 0:
                    atr_val = entry_price_estimate * default_atr_pct # Calculate ATR based on percentage
                    self.logger.info("Used default ATR of %.2f%% (%.4f) for TP/SL calc on %s.", 
                                     default_atr_pct * 100, atr_val, self.symbol)
                else: # Default ATR percentage is not positive or not configured
                    self.logger.error("Cannot calculate TP/SL for %s: ATR invalid and no valid default ATR%% configured.", self.symbol)
                    return entry_price_estimate, None, None
            except (InvalidOperation, TypeError):
                self.logger.error("Cannot calculate TP/SL for %s: Invalid default ATR%% format '%s'.",
                                  self.symbol, default_atr_pct_str)
                return entry_price_estimate, None, None
        
        # atr should now be a valid positive Decimal
        atr: Decimal = atr_val

        try:
            tp_multiplier_str = str(self.get_period("take_profit_multiple") or "2.0")
            sl_multiplier_str = str(self.get_period("stop_loss_multiple") or "1.5")
            tp_multiplier = Decimal(tp_multiplier_str)
            sl_multiplier = Decimal(sl_multiplier_str)

            if not (tp_multiplier > 0 and sl_multiplier > 0):
                self.logger.error("TP/SL multipliers must be positive for %s. TP: %s, SL: %s",
                                  self.symbol, tp_multiplier, sl_multiplier)
                raise ValueError("Multipliers must be positive.")

            price_prec = get_price_precision(self.market_info, self.logger)
            min_tick = get_min_tick_size(self.market_info, self.logger)
            quantize_unit = min_tick if min_tick and min_tick > Decimal(0) else Decimal(f"1e-{max(0, price_prec)}")
            if quantize_unit <= Decimal(0): quantize_unit = Decimal("1e-8") # Absolute fallback

            tp_offset = atr * tp_multiplier
            sl_offset = atr * sl_multiplier

            tp_level: Optional[Decimal] = None
            sl_level: Optional[Decimal] = None

            if signal == "BUY":
                tp_raw = entry_price_estimate + tp_offset
                sl_raw = entry_price_estimate - sl_offset
                tp_level = (tp_raw / quantize_unit).quantize(Decimal("1"), rounding=ROUND_UP) * quantize_unit
                sl_level = (sl_raw / quantize_unit).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantize_unit
            else:  # SELL
                tp_raw = entry_price_estimate - tp_offset
                sl_raw = entry_price_estimate + sl_offset
                tp_level = (tp_raw / quantize_unit).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantize_unit
                sl_level = (sl_raw / quantize_unit).quantize(Decimal("1"), rounding=ROUND_UP) * quantize_unit

            # --- Final Validation of Calculated SL/TP ---
            if sl_level is not None:
                if sl_level <= 0: sl_level = None # SL cannot be zero or negative
                elif signal == "BUY" and sl_level >= entry_price_estimate: sl_level = None # SL must be below entry for BUY
                elif signal == "SELL" and sl_level <= entry_price_estimate: sl_level = None # SL must be above entry for SELL
            
            if tp_level is not None:
                if tp_level <= 0: tp_level = None # TP cannot be zero or negative
                elif signal == "BUY" and tp_level <= entry_price_estimate: tp_level = None # TP must be above entry for BUY
                elif signal == "SELL" and tp_level >= entry_price_estimate: tp_level = None # TP must be below entry for SELL

            # Ensure SL and TP don't cross invalidly (e.g., SL > TP for BUY)
            if sl_level and tp_level:
                if (signal == "BUY" and sl_level >= tp_level) or \
                   (signal == "SELL" and sl_level <= tp_level):
                    self.logger.warning(
                        "Invalid TP/SL range for %s %s (Entry:%s, SL:%s, TP:%s). Setting both to None.",
                        self.symbol, signal, 
                        self._format_price_or_na(entry_price_estimate, price_prec),
                        self._format_price_or_na(sl_level, price_prec),
                        self._format_price_or_na(tp_level, price_prec)
                    )
                    sl_level = tp_level = None
            
            self.logger.debug(
                "Calculated TP/SL for %s (%s): Entry=%s, ATR=%s (xTP:%.1f, xSL:%.1f), TP=%s, SL=%s",
                self.symbol, signal,
                self._format_price_or_na(entry_price_estimate, price_prec),
                self._format_price_or_na(atr, price_prec + 2), # ATR often has more precision
                tp_multiplier, sl_multiplier,
                self._format_price_or_na(tp_level, price_prec),
                self._format_price_or_na(sl_level, price_prec)
            )
            return entry_price_estimate, tp_level, sl_level

        except (InvalidOperation, TypeError, ValueError) as e: # Expected errors related to Decimal/config
            self.logger.error("Error calculating TP/SL for %s: %s", self.symbol, e, exc_info=False)
            return entry_price_estimate, None, None
        except Exception as e: # Catch any other unexpected error
            self.logger.error("Unexpected error calculating TP/SL for %s: %s", self.symbol, e, exc_info=True)
            return entry_price_estimate, None, None
```