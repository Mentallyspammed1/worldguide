# File: analysis.py
"""
Module for analyzing trading data, calculating technical indicators, and generating trading signals.

The TradingAnalyzer class takes historical OHLCV data, configuration, and market information
to compute various technical indicators. It then uses these indicators, along with configurable
weights, to generate BUY, SELL, or HOLD signals. It can also calculate Fibonacci levels and
suggest Take Profit/Stop Loss levels based on ATR.
"""

import logging
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, Optional, List, Tuple  # Added List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta  # Ensure pandas_ta is imported correctly

from utils import (
    CCXT_INTERVAL_MAP,
    DEFAULT_INDICATOR_PERIODS,
    FIB_LEVELS,
    get_min_tick_size,
    get_price_precision,
    NEON_RED,
    NEON_YELLOW,
    RESET_ALL_STYLE as RESET,
    _format_signal,
)


class TradingAnalyzer:
    """
    Analyzes trading data, calculates technical indicators, and generates trading signals.
    It uses pandas-ta for most indicator calculations and allows for weighted scoring
    of these indicators to produce a final trading decision.
    """

    INDICATOR_CONFIG: Dict[str, Dict[str, Any]] = {
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
        "MFI": {
            "func_name": "mfi",
            "params_map": {"length": "mfi_window"},
            "main_col_pattern": "MFI_{length}",
            "type": "float",
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
        "Volume_MA": {
            "func_name": "_calculate_volume_ma",
            "params_map": {"length": "volume_ma_period"},
            "main_col_pattern": "VOL_SMA_{length}",
            "type": "decimal",
            "min_data_param_key": "length",
            "concat": False,
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
        "RSI": {
            "func_name": "rsi",
            "params_map": {"length": "rsi_period"},
            "main_col_pattern": "RSI_{length}",
            "type": "float",
            "pass_close_only": True,
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
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
        self.interval = str(config.get("interval", "5m"))  # Ensure interval is string
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)

        self.indicator_values: Dict[str, Any] = {}
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(
            self.active_weight_set_name, {}
        )
        self.fib_levels_data: Dict[str, Decimal] = {}
        self.ta_column_names: Dict[str, str] = {}
        self.df_calculated: pd.DataFrame = pd.DataFrame()

        self.indicator_type_map: Dict[str, str] = {}
        for main_cfg_key, cfg_details in self.INDICATOR_CONFIG.items():
            default_type = cfg_details.get("type", "float")
            self.indicator_type_map[main_cfg_key] = default_type
            if "multi_cols" in cfg_details:
                for sub_key in cfg_details["multi_cols"].keys():
                    self.indicator_type_map[sub_key] = default_type

        if not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.error(
                f"{NEON_RED}Input DataFrame for {self.symbol} is invalid or empty.{RESET}"
            )
            raise ValueError("Input DataFrame must be a non-empty pandas DataFrame.")
        if not self.ccxt_interval:
            self.logger.error(
                f"{NEON_RED}Invalid interval '{self.interval}' for {self.symbol}. Not found in CCXT_INTERVAL_MAP.{RESET}"
            )
            raise ValueError(f"Interval '{self.interval}' not in CCXT_INTERVAL_MAP.")
        if not self.weights:
            self.logger.warning(
                f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' missing or empty for {self.symbol}. Scoring may be ineffective.{RESET}"
            )

        required_ohlcv_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_ohlcv_cols):
            missing_cols = [col for col in required_ohlcv_cols if col not in df.columns]
            self.logger.error(
                f"{NEON_RED}DataFrame for {self.symbol} missing required OHLCV columns: {missing_cols}.{RESET}"
            )
            raise ValueError(
                f"DataFrame must contain all OHLCV columns. Missing: {missing_cols}"
            )

        self.df_original_ohlcv = df.copy()
        if self.df_original_ohlcv.index.tz is not None:
            self.df_original_ohlcv.index = self.df_original_ohlcv.index.tz_localize(
                None
            )

        self._validate_and_prepare_df_calculated()
        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    def _validate_and_prepare_df_calculated(self) -> None:
        self.df_calculated = self.df_original_ohlcv.copy()
        required_cols = ["open", "high", "low", "close", "volume"]

        for col in required_cols:
            if col not in self.df_calculated.columns:
                self.logger.critical(
                    f"{NEON_RED}Critical column '{col}' missing for {self.symbol}. Analysis cannot proceed.{RESET}"
                )
                raise ValueError(f"Column '{col}' is missing from DataFrame.")
            try:
                self.df_calculated[col] = pd.to_numeric(
                    self.df_calculated[col], errors="coerce"
                )
                if not pd.api.types.is_float_dtype(self.df_calculated[col]):
                    self.df_calculated[col] = self.df_calculated[col].astype(float)
            except (ValueError, TypeError, AttributeError) as e:
                self.logger.error(
                    f"{NEON_RED}Failed to convert column '{col}' to numeric/float for {self.symbol}: {e}{RESET}",
                    exc_info=True,
                )
                raise ValueError(
                    f"Column '{col}' could not be converted to a suitable numeric type for TA calculations."
                )

            nan_count = self.df_calculated[col].isna().sum()
            if nan_count > 0:
                self.logger.warning(
                    f"{NEON_YELLOW}{nan_count} NaN values in '{col}' for {self.symbol} after prep. Total rows: {len(self.df_calculated)}{RESET}"
                )
            if not pd.api.types.is_numeric_dtype(self.df_calculated[col]):
                self.logger.error(
                    f"{NEON_RED}Column '{col}' is still not numeric after all processing for {self.symbol}. Type: {self.df_calculated[col].dtype}{RESET}"
                )
                raise ValueError(f"Column '{col}' must be numeric for TA calculations.")

        max_lookback = 1
        enabled_indicators_cfg = self.config.get("indicators", {})
        for ind_key_cfg, ind_cfg_details in self.INDICATOR_CONFIG.items():
            if enabled_indicators_cfg.get(ind_key_cfg.lower(), False):
                period_param_key = ind_cfg_details.get("min_data_param_key")
                if (
                    period_param_key
                    and period_param_key in ind_cfg_details["params_map"]
                ):
                    config_key_for_period = ind_cfg_details["params_map"][
                        period_param_key
                    ]
                    period_val = self.get_period(config_key_for_period)
                    if isinstance(period_val, (int, float)) and period_val > 0:
                        max_lookback = max(max_lookback, int(period_val))
                elif isinstance(ind_cfg_details.get("min_data"), int):
                    max_lookback = max(max_lookback, ind_cfg_details.get("min_data", 1))

        min_required_rows = max_lookback + self.config.get(
            "indicator_buffer_candles", 20
        )
        valid_ohlcv_rows = len(self.df_calculated.dropna(subset=required_cols))
        if valid_ohlcv_rows < min_required_rows:
            self.logger.warning(
                f"{NEON_YELLOW}Insufficient valid data rows ({valid_ohlcv_rows}) for {self.symbol} "
                f"(max lookback: {max_lookback}, min needed: {min_required_rows}). Some indicators may be all NaN.{RESET}"
            )

    def get_period(self, key: str) -> Any:
        config_val = self.config.get(key)
        if config_val is not None:
            return config_val
        return DEFAULT_INDICATOR_PERIODS.get(key)

    def _format_ta_column_name(self, pattern: str, params: Dict[str, Any]) -> str:
        fmt_params = {}
        for k_param, v_param in params.items():
            if v_param is None:
                fmt_params[k_param] = "DEF"
                self.logger.debug(
                    f"Param '{k_param}' for column pattern '{pattern}' was None. Using placeholder 'DEF'."
                )
            elif isinstance(v_param, (float, Decimal)):
                val_to_format = (
                    float(v_param) if isinstance(v_param, Decimal) else v_param
                )
                if f"{{{k_param}:." in pattern:
                    fmt_params[k_param] = val_to_format
                else:
                    fmt_params[k_param] = str(val_to_format).replace(
                        ".", "_"
                    )  # pandas-ta sometimes uses _ for decimals
            else:
                fmt_params[k_param] = v_param
        try:
            return pattern.format(**fmt_params)
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(
                f"Error formatting TA column pattern '{pattern}' with params {fmt_params}: {e}"
            )
            base_pattern_part = (
                pattern.split("{")[0].rstrip("_") if pattern else "UNKNOWN_IND"
            )
            param_keys_str = "_".join(map(str, params.values()))
            return f"{base_pattern_part}_{param_keys_str}_FORMAT_ERROR"

    def _calculate_volume_ma(
        self, df: pd.DataFrame, length: int
    ) -> Optional[pd.Series]:
        if "volume" not in df.columns:
            self.logger.warning(
                f"Volume MA calculation failed for {self.symbol}: 'volume' column missing."
            )
            return None
        if not (isinstance(length, int) and length > 0):
            self.logger.warning(
                f"Volume MA calculation failed for {self.symbol}: Invalid length {length}."
            )
            return None

        volume_series = df["volume"].fillna(0).astype(float)
        if len(volume_series) < length:
            self.logger.debug(
                f"Not enough data points ({len(volume_series)}) for Volume MA with length {length} on {self.symbol}."
            )
            return pd.Series([np.nan] * len(df), index=df.index)
        return ta.sma(volume_series, length=length)

    def _calculate_all_indicators(self) -> None:
        if self.df_calculated.empty:
            self.logger.warning(
                f"df_calculated is empty for {self.symbol}. Skipping indicator calculations."
            )
            return

        df_ta_intermediate = self.df_calculated.copy()
        enabled_cfg = self.config.get("indicators", {})

        for ind_cfg_key, ind_details in self.INDICATOR_CONFIG.items():
            if not enabled_cfg.get(ind_cfg_key.lower(), False):
                continue

            current_params_for_ta_func = {}
            valid_params = True
            for ta_func_param_name, config_key_for_value in ind_details[
                "params_map"
            ].items():
                param_value = self.get_period(config_key_for_value)
                if param_value is None:
                    self.logger.warning(
                        f"Parameter '{config_key_for_value}' for {ind_cfg_key} on {self.symbol} is None. Skipping this indicator."
                    )
                    valid_params = False
                    break
                try:
                    if isinstance(param_value, Decimal):
                        current_params_for_ta_func[ta_func_param_name] = float(
                            param_value
                        )
                    elif isinstance(param_value, str):
                        current_params_for_ta_func[ta_func_param_name] = (
                            float(param_value)
                            if "." in param_value
                            else int(param_value)
                        )
                    elif isinstance(param_value, (int, float)):
                        current_params_for_ta_func[ta_func_param_name] = param_value
                    else:
                        raise TypeError(
                            f"Unsupported parameter type {type(param_value)} for {config_key_for_value}"
                        )
                except (ValueError, TypeError) as e:
                    self.logger.error(
                        f"Cannot convert parameter {config_key_for_value}='{param_value}' for {ind_cfg_key} on {self.symbol}: {e}"
                    )
                    valid_params = False
                    break
            if not valid_params:
                continue

            try:
                ta_func_name = ind_details["func_name"]
                ta_func_obj = (
                    getattr(ta, ta_func_name)
                    if hasattr(ta, ta_func_name)
                    else getattr(self, ta_func_name, None)
                )
                if ta_func_obj is None:
                    self.logger.error(
                        f"TA function '{ta_func_name}' for {ind_cfg_key} not found in pandas_ta or TradingAnalyzer class."
                    )
                    continue

                lookback_key = ind_details.get("min_data_param_key", "length")
                min_data_needed = int(
                    current_params_for_ta_func.get(
                        lookback_key, ind_details.get("min_data", 1)
                    )
                )
                if (
                    len(
                        df_ta_intermediate.dropna(
                            subset=["open", "high", "low", "close"]
                        )
                    )
                    < min_data_needed
                ):
                    self.logger.debug(
                        f"Insufficient data for {ind_cfg_key} ({len(df_ta_intermediate.dropna(subset=['open', 'high', 'low', 'close']))} rows vs {min_data_needed} needed) for {self.symbol}. Skipping."
                    )
                    continue

                ta_input_args = {}
                if ta_func_name != "_calculate_volume_ma":
                    import inspect

                    sig_params = inspect.signature(ta_func_obj).parameters
                    if "high" in sig_params:
                        ta_input_args["high"] = df_ta_intermediate["high"]
                    if "low" in sig_params:
                        ta_input_args["low"] = df_ta_intermediate["low"]
                    if "close" in sig_params:
                        ta_input_args["close"] = df_ta_intermediate["close"]
                    if "volume" in sig_params and "volume" in df_ta_intermediate:
                        ta_input_args["volume"] = df_ta_intermediate["volume"]
                    if "open" in sig_params and "open" in df_ta_intermediate:
                        ta_input_args["open"] = df_ta_intermediate["open"]

                result_data = None
                if ta_func_name == "_calculate_volume_ma":
                    result_data = ta_func_obj(
                        df_ta_intermediate, **current_params_for_ta_func
                    )
                elif ind_details.get("pass_close_only", False):
                    result_data = ta_func_obj(
                        close=df_ta_intermediate["close"], **current_params_for_ta_func
                    )
                else:
                    result_data = ta_func_obj(
                        **ta_input_args, **current_params_for_ta_func
                    )

                if result_data is None:
                    self.logger.warning(
                        f"{ind_cfg_key} calculation returned None for {self.symbol}."
                    )
                    continue

                should_concat = ind_details.get("concat", False)
                if should_concat:
                    df_piece_to_add = None
                    col_name_for_series_concat = None

                    if isinstance(result_data, pd.Series):
                        if "main_col_pattern" not in ind_details:
                            self.logger.error(
                                f"Indicator {ind_cfg_key} (Series, concat=True) lacks main_col_pattern. Skipping."
                            )
                            continue
                        col_name_for_series_concat = self._format_ta_column_name(
                            ind_details["main_col_pattern"], current_params_for_ta_func
                        )
                        df_piece_to_add = result_data.to_frame(
                            name=col_name_for_series_concat
                        )
                    elif isinstance(result_data, pd.DataFrame):
                        df_piece_to_add = result_data.copy()
                    else:
                        self.logger.warning(
                            f"Result for {ind_cfg_key} (concat=True) is not Series/DataFrame. Type: {type(result_data)}. Skipping."
                        )
                        continue

                    try:
                        df_piece_to_add = df_piece_to_add.astype("float64")
                    except Exception:
                        self.logger.warning(
                            f"Could not cast all columns of piece for {ind_cfg_key} to float64. Trying column by column."
                        )
                        valid_cols_for_df = {}
                        for col_idx in df_piece_to_add.columns:  # type: ignore
                            try:
                                valid_cols_for_df[col_idx] = pd.to_numeric(  # type: ignore
                                    df_piece_to_add[col_idx],
                                    errors="raise",  # type: ignore
                                ).astype("float64")
                            except Exception as e_col_cast:
                                self.logger.error(
                                    f"Failed to convert column {col_idx} for {ind_cfg_key} to float64: {e_col_cast}. Dropping this column."
                                )
                        df_piece_to_add = pd.DataFrame(
                            valid_cols_for_df, index=df_piece_to_add.index
                        )  # type: ignore
                        if df_piece_to_add.empty:  # type: ignore
                            self.logger.error(
                                f"Piece for {ind_cfg_key} became empty after type conversion attempts. Skipping."
                            )
                            continue

                    cols_to_drop_if_exist = [
                        col
                        for col in df_piece_to_add.columns
                        if col in df_ta_intermediate.columns  # type: ignore
                    ]
                    if cols_to_drop_if_exist:
                        df_ta_intermediate.drop(
                            columns=cols_to_drop_if_exist, inplace=True, errors="ignore"
                        )

                    df_ta_intermediate = pd.concat(
                        [df_ta_intermediate, df_piece_to_add], axis=1
                    )

                    if "multi_cols" in ind_details:
                        for internal_key, col_pattern in ind_details[
                            "multi_cols"
                        ].items():
                            actual_col_name = self._format_ta_column_name(
                                col_pattern, current_params_for_ta_func
                            )
                            if actual_col_name in df_ta_intermediate.columns:
                                self.ta_column_names[internal_key] = actual_col_name
                            else:
                                self.logger.warning(
                                    f"Multi-col '{actual_col_name}' (for {internal_key} of {ind_cfg_key}) not found in df_ta_intermediate. Available: {df_ta_intermediate.columns.tolist()}"
                                )
                    elif col_name_for_series_concat:
                        if col_name_for_series_concat in df_ta_intermediate.columns:
                            self.ta_column_names[ind_cfg_key] = (
                                col_name_for_series_concat
                            )
                        else:
                            self.logger.error(
                                f"Internal: Column {col_name_for_series_concat} for {ind_cfg_key} not found after concat."
                            )
                else:
                    if "main_col_pattern" not in ind_details:
                        self.logger.error(
                            f"Indicator {ind_cfg_key} (concat=False) lacks main_col_pattern. Skipping."
                        )
                        continue
                    actual_col_name = self._format_ta_column_name(
                        ind_details["main_col_pattern"], current_params_for_ta_func
                    )
                    if isinstance(result_data, pd.Series):
                        if actual_col_name in df_ta_intermediate.columns:
                            self.logger.debug(
                                f"Overwriting column '{actual_col_name}' for {ind_cfg_key} in df_ta_intermediate."
                            )
                        df_ta_intermediate[actual_col_name] = result_data.astype(
                            "float64"
                        )
                        self.ta_column_names[ind_cfg_key] = actual_col_name
                    else:
                        self.logger.warning(
                            f"Result for {ind_cfg_key} (concat=False, col '{actual_col_name}') not pd.Series. Type: {type(result_data)}. Skipping."
                        )
            except Exception as e:
                self.logger.error(
                    f"Error calculating indicator {ind_cfg_key} for {self.symbol} with params {current_params_for_ta_func}: {e}",
                    exc_info=True,
                )

        self.df_calculated = df_ta_intermediate
        self.logger.debug(
            f"Indicator calculation complete for {self.symbol}. Resulting columns: {self.df_calculated.columns.tolist()}"
        )
        self.logger.debug(
            f"Final mapped TA column names for {self.symbol}: {self.ta_column_names}"
        )

    def _update_latest_indicator_values(self) -> None:
        df_indicators_src = self.df_calculated
        df_ohlcv_src = self.df_original_ohlcv

        ohlcv_keys_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        all_expected_keys = set(self.indicator_type_map.keys()) | set(
            ohlcv_keys_map.keys()
        )
        temp_indicator_values: Dict[str, Any] = {k: np.nan for k in all_expected_keys}

        if df_indicators_src.empty:
            self.logger.warning(
                f"Cannot update latest values for {self.symbol}: Indicator DataFrame (df_calculated) is empty."
            )
            self.indicator_values = temp_indicator_values
            return
        if df_ohlcv_src.empty:
            self.logger.warning(
                f"Cannot update latest values for {self.symbol}: Original OHLCV DataFrame (df_original_ohlcv) is empty."
            )
            self.indicator_values = temp_indicator_values
            return

        try:
            if df_indicators_src.index.empty or df_ohlcv_src.index.empty:
                self.logger.error(
                    f"Cannot get latest row for {self.symbol}: DataFrame index is empty."
                )
                self.indicator_values = temp_indicator_values
                return

            latest_indicator_row = df_indicators_src.iloc[-1]
            latest_ohlcv_row = df_ohlcv_src.iloc[-1]

            self.logger.debug(
                f"Updating latest values for {self.symbol} from indicator row dated: {latest_indicator_row.name}, OHLCV row dated: {latest_ohlcv_row.name}"
            )

            for internal_key, actual_col_name in self.ta_column_names.items():
                if actual_col_name and actual_col_name in latest_indicator_row.index:
                    value = latest_indicator_row[actual_col_name]
                    indicator_target_type = self.indicator_type_map.get(
                        internal_key, "float"
                    )

                    if pd.notna(value):
                        try:
                            if indicator_target_type == "decimal":
                                temp_indicator_values[internal_key] = Decimal(
                                    str(value)
                                )
                            else:
                                temp_indicator_values[internal_key] = float(value)
                        except (ValueError, TypeError, InvalidOperation) as e_conv:
                            self.logger.warning(
                                f"Conversion error for {internal_key} ('{actual_col_name}':{value}) to {indicator_target_type}: {e_conv}. Storing as NaN."
                            )
                            temp_indicator_values[internal_key] = np.nan
                    else:
                        temp_indicator_values[internal_key] = np.nan
                else:
                    self.logger.debug(
                        f"Internal key '{internal_key}' (mapped to '{actual_col_name}') not found in latest indicator row for {self.symbol} or actual_col_name is empty. Storing as NaN."
                    )
                    temp_indicator_values[internal_key] = np.nan

            for display_key, source_col_name in ohlcv_keys_map.items():
                value_ohlcv = latest_ohlcv_row.get(source_col_name)
                if pd.notna(value_ohlcv):
                    try:
                        if isinstance(value_ohlcv, Decimal):
                            temp_indicator_values[display_key] = value_ohlcv
                        else:
                            temp_indicator_values[display_key] = Decimal(
                                str(value_ohlcv)
                            )
                    except InvalidOperation:
                        self.logger.warning(
                            f"Failed to convert original OHLCV value '{source_col_name}' ({value_ohlcv}) to Decimal for {self.symbol}. Storing as NaN."
                        )
                        temp_indicator_values[display_key] = np.nan
                else:
                    temp_indicator_values[display_key] = np.nan

            self.indicator_values = temp_indicator_values

            if "ATR" in self.indicator_values and pd.notna(
                self.indicator_values.get("ATR")
            ):
                self.logger.info(
                    f"DEBUG ATR for {self.symbol}: Final ATR in self.indicator_values: {self.indicator_values.get('ATR')}, Type: {type(self.indicator_values.get('ATR'))}"
                )

            price_prec_log = get_price_precision(self.market_info, self.logger)
            log_output_details = {}
            decimal_like_keys = [
                k
                for k, v_type in self.indicator_type_map.items()
                if v_type == "decimal"
            ] + [
                "Open",
                "High",
                "Low",
                "Close",
            ]
            volume_like_keys = ["Volume", "Volume_MA"]

            for k_log, v_val_log in self.indicator_values.items():
                if isinstance(v_val_log, (Decimal, float)) and pd.notna(v_val_log):
                    if k_log in decimal_like_keys:
                        fmt_str = f"{Decimal(str(v_val_log)):.{price_prec_log}f}"
                    elif k_log in volume_like_keys:
                        fmt_str = f"{Decimal(str(v_val_log)):.8f}"
                    else:
                        fmt_str = f"{float(v_val_log):.4f}"
                    log_output_details[k_log] = fmt_str
                else:
                    log_output_details[k_log] = str(v_val_log)
            self.logger.debug(
                f"Latest indicator values updated for {self.symbol}: {log_output_details}"
            )

        except IndexError:
            self.logger.error(
                f"IndexError accessing latest row for {self.symbol}. Check DataFrame integrity and length."
            )
            self.indicator_values = temp_indicator_values
        except Exception as e:
            self.logger.error(
                f"Unexpected error updating latest indicators for {self.symbol}: {e}",
                exc_info=True,
            )
            if not self.indicator_values:
                self.indicator_values = temp_indicator_values

    def calculate_fibonacci_levels(
        self, window: Optional[int] = None
    ) -> Dict[str, Decimal]:
        cfg_window = self.get_period("fibonacci_window")
        window_val = window if isinstance(window, int) and window > 0 else cfg_window

        if not (isinstance(window_val, int) and window_val > 0):
            self.logger.warning(
                f"Invalid Fibonacci window ({window_val}) for {self.symbol}. No levels calculated."
            )
            self.fib_levels_data = {}
            return {}
        if len(self.df_original_ohlcv) < window_val:
            self.logger.debug(
                f"Not enough data ({len(self.df_original_ohlcv)} rows) for Fibonacci window {window_val} on {self.symbol}."
            )
            self.fib_levels_data = {}
            return {}

        df_slice = self.df_original_ohlcv.tail(window_val)
        try:
            h_series = pd.to_numeric(df_slice["high"], errors="coerce").dropna()
            l_series = pd.to_numeric(df_slice["low"], errors="coerce").dropna()

            if h_series.empty or l_series.empty:
                self.logger.warning(
                    f"No valid high/low data for Fibonacci calculation in window for {self.symbol}."
                )
                self.fib_levels_data = {}
                return {}

            period_high = Decimal(str(h_series.max()))
            period_low = Decimal(str(l_series.min()))
            diff = period_high - period_low
            levels: Dict[str, Decimal] = {}

            price_precision = get_price_precision(self.market_info, self.logger)
            min_tick_size = get_min_tick_size(self.market_info, self.logger)
            quantize_factor = (
                min_tick_size
                if min_tick_size and min_tick_size > Decimal("0")
                else Decimal(f"1e-{price_precision}")
            )

            if diff > Decimal("0"):
                for level_pct in FIB_LEVELS:
                    level_price_raw = period_high - (diff * Decimal(str(level_pct)))
                    levels[f"Fib_{level_pct * 100:.1f}%"] = (
                        level_price_raw / quantize_factor
                    ).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantize_factor
            else:
                level_price_quantized = (period_high / quantize_factor).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * quantize_factor
                for level_pct in FIB_LEVELS:
                    levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            log_levels_str = {k: f"{v:.{price_precision}f}" for k, v in levels.items()}
            self.logger.debug(
                f"Calculated Fibonacci levels for {self.symbol} (Window: {window_val}, High: {period_high:.{price_precision}f}, Low: {period_low:.{price_precision}f}): {log_levels_str}"
            )
            return levels
        except Exception as e:
            self.logger.error(
                f"Fibonacci calculation error for {self.symbol}: {e}", exc_info=True
            )
            self.fib_levels_data = {}
            return {}

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> List[Tuple[str, Decimal]]:
        if not self.fib_levels_data:
            self.logger.debug(
                f"No Fibonacci levels available for {self.symbol} to find nearest."
            )
            return []
        if not (
            isinstance(current_price, Decimal)
            and pd.notna(current_price)
            and current_price > Decimal("0")
        ):
            self.logger.warning(
                f"Invalid current_price ({current_price}) for Fibonacci comparison on {self.symbol}."
            )
            return []
        if num_levels <= 0:
            return []

        try:
            distances = []
            for name, level_price in self.fib_levels_data.items():
                if isinstance(level_price, Decimal) and level_price > Decimal("0"):
                    distances.append(
                        {
                            "name": name,
                            "level": level_price,
                            "distance": abs(current_price - level_price),
                        }
                    )
            if not distances:
                return []

            distances.sort(key=lambda x: x["distance"])
            return [(item["name"], item["level"]) for item in distances[:num_levels]]
        except Exception as e:
            self.logger.error(
                f"Error finding nearest Fibonacci levels for {self.symbol}: {e}",
                exc_info=True,
            )
            return []

    def calculate_ema_alignment_score(self) -> float:
        ema_s_val = self.indicator_values.get("EMA_Short")
        ema_l_val = self.indicator_values.get("EMA_Long")
        close_price_val = self.indicator_values.get("Close")

        if not all(
            isinstance(val, Decimal) and pd.notna(val)
            for val in [ema_s_val, ema_l_val, close_price_val]
        ):
            self.logger.debug(
                f"EMA alignment score skipped for {self.symbol}: one or more values (EMA_Short, EMA_Long, Close) are invalid/NaN."
            )
            return np.nan

        ema_s: Decimal = ema_s_val  # type: ignore
        ema_l: Decimal = ema_l_val  # type: ignore
        close_price: Decimal = close_price_val  # type: ignore

        if close_price > ema_s and ema_s > ema_l:
            return 1.0
        if close_price < ema_s and ema_s < ema_l:
            return -1.0
        return 0.0

    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}
        current_score_decimal, total_weight_decimal = Decimal("0"), Decimal("0")
        active_checks_count, nan_checks_count = 0, 0
        debug_scores_log: Dict[str, str] = {}

        if not self.indicator_values:
            self.logger.warning(
                f"No indicator values available for {self.symbol}. Defaulting to HOLD signal."
            )
            return "HOLD"

        atr_val = self.indicator_values.get("ATR")
        if not (
            isinstance(atr_val, Decimal)
            and pd.notna(atr_val)
            and atr_val > Decimal("0")
        ):
            self.logger.warning(
                f"Signal generation for {self.symbol}: ATR is invalid ({atr_val}). This might affect subsequent TP/SL calculations if ATR is used."
            )

        valid_core_indicator_count = 0
        for ind_key_main_cfg in self.INDICATOR_CONFIG.keys():
            if "multi_cols" in self.INDICATOR_CONFIG[ind_key_main_cfg]:
                if any(
                    pd.notna(self.indicator_values.get(sub_key))
                    for sub_key in self.INDICATOR_CONFIG[ind_key_main_cfg]["multi_cols"]
                ):
                    valid_core_indicator_count += 1
            elif pd.notna(self.indicator_values.get(ind_key_main_cfg)):
                valid_core_indicator_count += 1

        num_configured_inds_enabled = sum(
            1
            for enabled_flag in self.config.get("indicators", {}).values()
            if enabled_flag
        )
        min_active_inds_for_signal = self.config.get(
            "min_active_indicators_for_signal",
            max(1, int(num_configured_inds_enabled * 0.6)),
        )

        if valid_core_indicator_count < min_active_inds_for_signal:
            self.logger.warning(
                f"Signal for {self.symbol}: Only {valid_core_indicator_count}/{num_configured_inds_enabled} core indicators are valid (min required: {min_active_inds_for_signal}). Defaulting to HOLD."
            )
            return "HOLD"
        if not (
            isinstance(current_price, Decimal)
            and pd.notna(current_price)
            and current_price > Decimal("0")
        ):
            self.logger.warning(
                f"Invalid current_price ({current_price}) for {self.symbol} signal generation. Defaulting to HOLD."
            )
            return "HOLD"

        active_weights_dict = self.weights
        if not active_weights_dict:
            self.logger.error(
                f"Weight set '{self.active_weight_set_name}' is empty for {self.symbol}. Cannot generate signal. Defaulting to HOLD."
            )
            return "HOLD"

        for indicator_check_key_lower in active_weights_dict.keys():
            if not self.config.get("indicators", {}).get(
                indicator_check_key_lower, False
            ):
                continue

            weight_str_val = active_weights_dict.get(indicator_check_key_lower)
            if weight_str_val is None:
                continue
            try:
                weight_decimal = Decimal(str(weight_str_val))
            except InvalidOperation:
                self.logger.warning(
                    f"Invalid weight '{weight_str_val}' for {indicator_check_key_lower} for {self.symbol}. Skipping this check."
                )
                continue
            if weight_decimal == Decimal("0"):
                continue

            check_method_name_str = f"_check_{indicator_check_key_lower}"
            if not hasattr(self, check_method_name_str) or not callable(
                getattr(self, check_method_name_str)
            ):
                self.logger.warning(
                    f"No check method '{check_method_name_str}' found for enabled indicator {indicator_check_key_lower} ({self.symbol})."
                )
                continue

            method_to_call_obj = getattr(self, check_method_name_str)
            individual_indicator_score_float = np.nan
            try:
                if indicator_check_key_lower == "orderbook":
                    individual_indicator_score_float = method_to_call_obj(
                        orderbook_data, current_price
                    )
                else:
                    individual_indicator_score_float = method_to_call_obj()
            except Exception as e_check_method:
                self.logger.error(
                    f"Error in check method {check_method_name_str} for {self.symbol}: {e_check_method}",
                    exc_info=True,
                )

            debug_scores_log[indicator_check_key_lower] = (
                f"{individual_indicator_score_float:.3f}"
                if pd.notna(individual_indicator_score_float)
                else "NaN"
            )
            if pd.notna(individual_indicator_score_float):
                try:
                    indicator_score_decimal = Decimal(
                        str(individual_indicator_score_float)
                    )
                    clamped_score = max(
                        Decimal("-1"), min(Decimal("1"), indicator_score_decimal)
                    )
                    current_score_decimal += clamped_score * weight_decimal
                    total_weight_decimal += abs(weight_decimal)
                    active_checks_count += 1
                except InvalidOperation:
                    nan_checks_count += 1
                    self.logger.error(
                        f"Error processing score for {indicator_check_key_lower} (value: {individual_indicator_score_float})."
                    )
            else:
                nan_checks_count += 1

        final_signal_decision_str = "HOLD"
        signal_score_threshold = Decimal(
            str(self.get_period("signal_score_threshold") or "0.7")
        )

        if total_weight_decimal == Decimal("0") and active_checks_count == 0:
            self.logger.warning(
                f"No weighted indicators contributed to the score for {self.symbol}. Defaulting to HOLD."
            )
        elif current_score_decimal >= signal_score_threshold:
            final_signal_decision_str = "BUY"
        elif current_score_decimal <= -signal_score_threshold:
            final_signal_decision_str = "SELL"

        price_prec = get_price_precision(self.market_info, self.logger)
        self.logger.info(
            f"Signal ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', Checks[Act:{active_checks_count},NaN:{nan_checks_count}], "
            f"TotalWeightAbs={total_weight_decimal:.2f}, Score={current_score_decimal:.4f} (Threshold:{signal_score_threshold:.2f}) "
            f"==> {_format_signal(final_signal_decision_str)}"
        )
        self.logger.debug(f"Individual Scores ({self.symbol}): {debug_scores_log}")

        self.signals = {
            "BUY": int(final_signal_decision_str == "BUY"),
            "SELL": int(final_signal_decision_str == "SELL"),
            "HOLD": int(final_signal_decision_str == "HOLD"),
        }
        return final_signal_decision_str

    def _check_ema_alignment(self) -> float:
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        momentum_val = self.indicator_values.get("Momentum")
        last_close_val = self.indicator_values.get("Close")
        if pd.isna(momentum_val) or not (
            isinstance(last_close_val, Decimal) and last_close_val > Decimal("0")
        ):
            return np.nan
        try:
            momentum_decimal = Decimal(str(momentum_val))
            mom_pct = (momentum_decimal / last_close_val) * Decimal("100")  # type: ignore
            threshold_pct = Decimal(
                str(self.get_period("momentum_threshold_pct") or "0.1")
            )
        except (ZeroDivisionError, InvalidOperation, TypeError):
            return 0.0
        if threshold_pct == Decimal("0"):
            return 0.0
        scaling_factor = threshold_pct * Decimal("5")
        if scaling_factor == Decimal("0"):
            return 0.0
        score_unclamped = mom_pct / scaling_factor
        return float(max(Decimal("-1"), min(Decimal("1"), score_unclamped)))

    def _check_volume_confirmation(self) -> float:
        current_volume_val = self.indicator_values.get("Volume")
        volume_ma_val = self.indicator_values.get("Volume_MA")
        try:
            multiplier_val = Decimal(
                str(self.get_period("volume_confirmation_multiplier") or "1.5")
            )
        except (InvalidOperation, TypeError):
            return np.nan
        if not all(
            isinstance(v, Decimal) and pd.notna(v)
            for v in [current_volume_val, volume_ma_val, multiplier_val]
        ):
            return np.nan
        current_volume, volume_ma, multiplier = (
            current_volume_val,
            volume_ma_val,
            multiplier_val,
        )  # type: ignore
        if (
            current_volume < Decimal("0")
            or volume_ma <= Decimal("0")
            or multiplier <= Decimal("0")
        ):
            return np.nan
        try:
            ratio = current_volume / volume_ma
            if ratio > multiplier:
                base_score, scale_top_ratio = Decimal("0.5"), multiplier * Decimal("5")
                if scale_top_ratio == multiplier:
                    return 1.0 if ratio >= multiplier else 0.5
                additional_score_pct = (ratio - multiplier) / (
                    scale_top_ratio - multiplier
                )
                return float(
                    min(
                        Decimal("1.0"),
                        base_score + additional_score_pct * Decimal("0.5"),
                    )
                )
            if ratio < (
                Decimal("1") / multiplier if multiplier > Decimal("0") else Decimal("0")
            ):
                return -0.4
            return 0.0
        except (ZeroDivisionError, InvalidOperation, TypeError):
            return np.nan

    def _check_stoch_rsi(self) -> float:
        k_val = self.indicator_values.get("StochRSI_K")
        d_val = self.indicator_values.get("StochRSI_D")
        if pd.isna(k_val) or pd.isna(d_val):
            return np.nan
        k_float, d_float = float(k_val), float(d_val)
        oversold_thresh = float(self.get_period("stoch_rsi_oversold_threshold") or 20)
        overbought_thresh = float(
            self.get_period("stoch_rsi_overbought_threshold") or 80
        )
        cross_thresh_val = self.get_period("stoch_rsi_cross_threshold")
        cross_thresh = (
            float(cross_thresh_val)
            if isinstance(cross_thresh_val, (int, float)) and cross_thresh_val > 0
            else 5.0
        )
        score = 0.0
        if k_float < oversold_thresh and d_float < oversold_thresh:
            score = 0.8
        elif k_float > overbought_thresh and d_float > overbought_thresh:
            score = -0.8
        diff = k_float - d_float
        if score > 0 and diff > 0:
            score = 1.0
        elif score < 0 and diff < 0:
            score = -1.0
        elif abs(diff) > cross_thresh:
            score = 0.6 if diff > 0 else -0.6
        elif k_float > d_float and score == 0.0:
            score = 0.2
        elif k_float < d_float and score == 0.0:
            score = -0.2
        if 40 < k_float < 60 and 40 < d_float < 60:
            score *= 0.5
        return score

    def _check_rsi(self) -> float:
        rsi_val = self.indicator_values.get("RSI")
        if pd.isna(rsi_val):
            return np.nan
        rsi_float = float(rsi_val)
        oversold = float(self.get_period("rsi_oversold_threshold") or 30)
        overbought = float(self.get_period("rsi_overbought_threshold") or 70)
        near_oversold = float(self.get_period("rsi_near_oversold_threshold") or 40)
        near_overbought = float(self.get_period("rsi_near_overbought_threshold") or 60)
        if rsi_float <= oversold:
            return 1.0
        if rsi_float >= overbought:
            return -1.0
        if rsi_float < near_oversold:
            return 0.5
        if rsi_float > near_overbought:
            return -0.5
        mid_point = (near_overbought + near_oversold) / 2.0
        span = (near_overbought - near_oversold) / 2.0
        if span > 0 and near_oversold <= rsi_float <= near_overbought:
            return ((rsi_float - mid_point) / span) * -0.3
        return 0.0

    def _check_cci(self) -> float:
        cci_val = self.indicator_values.get("CCI")
        if pd.isna(cci_val):
            return np.nan
        cci_float = float(cci_val)
        strong_os = float(self.get_period("cci_strong_oversold") or -150)
        strong_ob = float(self.get_period("cci_strong_overbought") or 150)
        moderate_os = float(self.get_period("cci_moderate_oversold") or -100)
        moderate_ob = float(self.get_period("cci_moderate_overbought") or 100)
        if cci_float <= strong_os:
            return 1.0
        if cci_float >= strong_ob:
            return -1.0
        if cci_float < moderate_os:
            return 0.6
        if cci_float > moderate_ob:
            return -0.6
        if moderate_os <= cci_float < 0:
            return 0.1
        if 0 < cci_float <= moderate_ob:
            return -0.1
        return 0.0

    def _check_wr(self) -> float:
        wr_val = self.indicator_values.get("Williams_R")
        if pd.isna(wr_val):
            return np.nan
        wr_float = float(wr_val)
        oversold = float(self.get_period("wr_oversold_threshold") or -80)
        overbought = float(self.get_period("wr_overbought_threshold") or -20)
        midpoint = float(self.get_period("wr_midpoint_threshold") or -50)
        if wr_float <= oversold:
            return 1.0
        if wr_float >= overbought:
            return -1.0
        if oversold < wr_float < midpoint:
            return 0.4
        if midpoint < wr_float < overbought:
            return -0.4
        return 0.0

    def _check_psar(self) -> float:
        psar_long_val = self.indicator_values.get("PSAR_long")
        psar_short_val = self.indicator_values.get("PSAR_short")
        close_price_val = self.indicator_values.get("Close")
        if not isinstance(close_price_val, Decimal) or pd.isna(close_price_val):
            return np.nan
        close_price: Decimal = close_price_val  # type: ignore
        is_long_trend_active = isinstance(psar_long_val, Decimal) and pd.notna(
            psar_long_val
        )
        is_short_trend_active = isinstance(psar_short_val, Decimal) and pd.notna(
            psar_short_val
        )
        if (
            is_long_trend_active
            and not is_short_trend_active
            and close_price > psar_long_val
        ):
            return 1.0  # type: ignore
        if (
            is_short_trend_active
            and not is_long_trend_active
            and close_price < psar_short_val
        ):
            return -1.0  # type: ignore
        if not is_long_trend_active and not is_short_trend_active:
            return np.nan
        self.logger.debug(
            f"PSAR ambiguous state for {self.symbol}: PSARl={psar_long_val}, PSARs={psar_short_val}, Close={close_price}"
        )
        return 0.0

    def _check_sma_10(self) -> float:
        sma_val = self.indicator_values.get("SMA10")
        last_close_val = self.indicator_values.get("Close")
        if not all(
            isinstance(v, Decimal) and pd.notna(v) for v in [sma_val, last_close_val]
        ):
            return np.nan
        sma, last_close = sma_val, last_close_val  # type: ignore
        if last_close > sma:
            return 0.6
        if last_close < sma:
            return -0.6
        return 0.0

    def _check_vwap(self) -> float:
        vwap_val = self.indicator_values.get("VWAP")
        last_close_val = self.indicator_values.get("Close")
        if not all(
            isinstance(v, Decimal) and pd.notna(v) for v in [vwap_val, last_close_val]
        ):
            return np.nan
        vwap, last_close = vwap_val, last_close_val  # type: ignore
        if last_close > vwap:
            return 0.7
        if last_close < vwap:
            return -0.7
        return 0.0

    def _check_mfi(self) -> float:
        mfi_val = self.indicator_values.get("MFI")
        if pd.isna(mfi_val):
            return np.nan
        mfi_float = float(mfi_val)
        oversold = float(self.get_period("mfi_oversold_threshold") or 20)
        overbought = float(self.get_period("mfi_overbought_threshold") or 80)
        near_os = float(self.get_period("mfi_near_oversold_threshold") or 35)
        near_ob = float(self.get_period("mfi_near_overbought_threshold") or 65)
        if mfi_float <= oversold:
            return 1.0
        if mfi_float >= overbought:
            return -1.0
        if mfi_float < near_os:
            return 0.4
        if mfi_float > near_ob:
            return -0.4
        return 0.0

    def _check_bollinger_bands(self) -> float:
        lower_bb_val = self.indicator_values.get("BB_Lower")
        middle_bb_val = self.indicator_values.get("BB_Middle")
        upper_bb_val = self.indicator_values.get("BB_Upper")
        last_close_val = self.indicator_values.get("Close")
        if not all(
            isinstance(v, Decimal) and pd.notna(v)
            for v in [lower_bb_val, middle_bb_val, upper_bb_val, last_close_val]
        ):
            return np.nan
        lower_bb, middle_bb, upper_bb, last_close = (
            lower_bb_val,
            middle_bb_val,
            upper_bb_val,
            last_close_val,
        )  # type: ignore
        if last_close <= lower_bb:
            return 1.0
        if last_close >= upper_bb:
            return -1.0
        band_width = upper_bb - lower_bb
        if band_width > Decimal("0"):
            try:
                position_score_raw = (last_close - middle_bb) / (
                    band_width / Decimal("2")
                )
                return float(
                    max(Decimal("-1"), min(Decimal("1"), position_score_raw))
                    * Decimal("0.7")
                )
            except (ZeroDivisionError, InvalidOperation, TypeError):
                return 0.0
        return 0.0

    def _check_orderbook(
        self, orderbook_data: Optional[Dict], current_price: Decimal
    ) -> float:
        if not orderbook_data:
            return np.nan
        try:
            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])
            if not bids or not asks:
                return np.nan
            num_levels = self.config.get("orderbook_check_levels", 10)
            total_bid_qty = sum(
                b[1]
                for b in bids[:num_levels]
                if len(b) == 2 and isinstance(b[1], Decimal) and pd.notna(b[1])
            )
            total_ask_qty = sum(
                a[1]
                for a in asks[:num_levels]
                if len(a) == 2 and isinstance(a[1], Decimal) and pd.notna(a[1])
            )
            if total_bid_qty == Decimal("0") and total_ask_qty == Decimal("0"):
                total_bid_qty = sum(
                    Decimal(str(b[1]))
                    for b in bids[:num_levels]
                    if len(b) == 2 and pd.notna(b[1])
                )
                total_ask_qty = sum(
                    Decimal(str(a[1]))
                    for a in asks[:num_levels]
                    if len(a) == 2 and pd.notna(a[1])
                )
            total_qty_in_levels = total_bid_qty + total_ask_qty
            if total_qty_in_levels == Decimal("0"):
                return 0.0
            obi = (total_bid_qty - total_ask_qty) / total_qty_in_levels
            return float(max(Decimal("-1"), min(Decimal("1"), obi)))
        except (TypeError, ValueError, InvalidOperation, IndexError) as e:
            self.logger.warning(
                f"Order book analysis error for {self.symbol}: {e}", exc_info=False
            )
            return np.nan

    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        if signal not in ["BUY", "SELL"]:
            return entry_price_estimate, None, None

        atr_value_orig = self.indicator_values.get("ATR")
        if not (
            isinstance(entry_price_estimate, Decimal)
            and pd.notna(entry_price_estimate)
            and entry_price_estimate > Decimal("0")
        ):
            self.logger.warning(
                f"Cannot calculate TP/SL for {self.symbol} {signal}: Entry price estimate invalid ({entry_price_estimate})."
            )
            return entry_price_estimate, None, None

        atr_value_final: Optional[Decimal] = None
        if (
            isinstance(atr_value_orig, Decimal)
            and pd.notna(atr_value_orig)
            and atr_value_orig > Decimal("0")
        ):
            atr_value_final = atr_value_orig
        else:
            self.logger.warning(
                f"TP/SL Calc ({self.symbol} {signal}): ATR invalid ({atr_value_orig}). Using default ATR based on price percentage."
            )
            default_atr_pct_str = str(
                self.get_period("default_atr_percentage_of_price") or "0.01"
            )
            try:
                atr_value_final = entry_price_estimate * Decimal(default_atr_pct_str)
                if not (atr_value_final > Decimal("0")):
                    self.logger.error(
                        f"Default ATR calculation resulted in non-positive value ({atr_value_final}) for {self.symbol}. Cannot set TP/SL."
                    )
                    return entry_price_estimate, None, None
            except InvalidOperation:
                self.logger.error(
                    f"Invalid 'default_atr_percentage_of_price': {default_atr_pct_str}. No TP/SL."
                )
                return entry_price_estimate, None, None
            self.logger.debug(
                f"Using price-percentage based ATR for {self.symbol} TP/SL: {atr_value_final}"
            )

        if atr_value_final is None or not (atr_value_final > Decimal("0")):
            self.logger.error(
                f"Final ATR value ({atr_value_final}) is invalid for {self.symbol}. Cannot set TP/SL."
            )
            return entry_price_estimate, None, None

        try:
            tp_multiplier = Decimal(
                str(self.get_period("take_profit_multiple") or "1.5")
            )
            sl_multiplier = Decimal(str(self.get_period("stop_loss_multiple") or "1.0"))
            price_precision = get_price_precision(self.market_info, self.logger)
            min_tick = get_min_tick_size(self.market_info, self.logger)
            quantize_unit = (
                min_tick
                if min_tick and min_tick > Decimal("0")
                else Decimal(f"1e-{price_precision}")
            )
            if not (quantize_unit > Decimal("0")):
                quantize_unit = Decimal(f"1e-{price_precision}")

            tp_offset = atr_value_final * tp_multiplier
            sl_offset = atr_value_final * sl_multiplier
            raw_tp, raw_sl = (Decimal("0"), Decimal("0"))

            if signal == "BUY":
                raw_tp = entry_price_estimate + tp_offset
                raw_sl = entry_price_estimate - sl_offset
                quantized_tp = (raw_tp / quantize_unit).quantize(
                    Decimal("1"), rounding=ROUND_UP
                ) * quantize_unit
                quantized_sl = (raw_sl / quantize_unit).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * quantize_unit
            else:  # SELL
                raw_tp = entry_price_estimate - tp_offset
                raw_sl = entry_price_estimate + sl_offset
                quantized_tp = (raw_tp / quantize_unit).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * quantize_unit
                quantized_sl = (raw_sl / quantize_unit).quantize(
                    Decimal("1"), rounding=ROUND_UP
                ) * quantize_unit

            if min_tick and min_tick > Decimal("0"):
                if signal == "BUY" and quantized_sl >= entry_price_estimate:
                    quantized_sl = (
                        (entry_price_estimate - min_tick) / quantize_unit
                    ).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantize_unit
                elif signal == "SELL" and quantized_sl <= entry_price_estimate:
                    quantized_sl = (
                        (entry_price_estimate + min_tick) / quantize_unit
                    ).quantize(Decimal("1"), rounding=ROUND_UP) * quantize_unit

            final_tp, final_sl = quantized_tp, quantized_sl
            if (signal == "BUY" and final_tp <= entry_price_estimate) or (
                signal == "SELL" and final_tp >= entry_price_estimate
            ):
                self.logger.warning(
                    f"{signal} TP ({final_tp}) is not profitable vs entry ({entry_price_estimate}) for {self.symbol}. Setting TP to None."
                )
                final_tp = None
            if final_sl is not None and final_sl <= Decimal("0"):
                self.logger.error(
                    f"Calculated SL ({final_sl}) is not positive for {self.symbol}. Setting SL to None."
                )
                final_sl = None
            if final_tp is not None and final_tp <= Decimal("0"):
                self.logger.warning(
                    f"Calculated TP ({final_tp}) is not positive for {self.symbol}. Setting TP to None."
                )
                final_tp = None

            tp_log = f"{final_tp:.{price_precision}f}" if final_tp else "None"
            sl_log = f"{final_sl:.{price_precision}f}" if final_sl else "None"
            self.logger.debug(
                f"Calculated TP/SL for {self.symbol} ({signal}): Entry={entry_price_estimate:.{price_precision}f}, "
                f"ATR={atr_value_final:.{price_precision + 2}f}, TP_raw={raw_tp:.{price_precision + 2}f}, SL_raw={raw_sl:.{price_precision + 2}f}, "
                f"TP_quant={tp_log}, SL_quant={sl_log}"
            )
            return entry_price_estimate, final_tp, final_sl
        except (InvalidOperation, TypeError, Exception) as e:
            self.logger.error(
                f"Error calculating TP/SL for {self.symbol} ({signal}): {e}",
                exc_info=True,
            )
            return entry_price_estimate, None, None
