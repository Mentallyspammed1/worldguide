Ah, seeker! Your quest for refined strategic insight within Pyrmethus, coupled with a richer, more luminous neon interface to articulate its arcane maneuvers, is a commendable endeavor. This task is substantial, mirroring the intricate processes of re-aligning a potent magical core and meticulously re-inscribing the very runes that channel its visual output.

For this **Unified Scalping Spell v2.8.0 (Strategic Illumination)**, the following enhancements have been woven into its fabric:

1.  **Refined Strategy Logic (Dual Supertrend with Momentum Confirmation):**
    *   The previous placeholder logic has been supplanted with a fully implemented `DualSupertrendMomentumStrategy`. This strategy now mandates confirmation from a Momentum indicator for its entry signals, adding a layer of analytical depth. The underlying `calculate_supertrend` and `calculate_momentum` functions have also been concretely implemented using `pandas_ta` for more realistic signal generation.
    *   For illustrative clarity and simplicity within this version, exit signals remain primarily governed by the primary SuperTrend's directional flip, though this foundation allows for future expansion with more sophisticated exit criteria.

2.  **Vibrant Neon Colorization Overhaul:**
    *   **Expanded & Distinct State Colors:** The `NEON` color palette has been adopted as specified, ensuring a comprehensive and consistent color scheme clearly distinguishes various operational states (e.g., long, short, holding, error notifications, successful operations).
    *   **Enhanced Key Value Highlighting:** Critical numerical data—such as prices, quantities, and Profit/Loss figures—are now rendered with greater prominence for immediate visual impact using the defined `NEON` values.
    *   **Intuitive Thematic Color Grouping:** Colors are grouped thematically to enhance readability and cognitive association (e.g., varying shades of blue for informational messages, greens for successful outcomes and long positions, reds for errors and short positions, and yellows for warnings or items requiring attention).
    *   **Strategic Use of Brighter Accents:** The `Style.BRIGHT` attribute is employed judiciously, as defined in the `NEON` palette, to emphasize key information and draw attention where it's most needed.

**Key Considerations for the Adept:**

*   **Strategy Nuance:** While the `DualSupertrendMomentumStrategy` introduces more sophistication, it remains a foundational example. Production-grade strategies typically incorporate a multitude of conditions, parameters, and adaptive logic.
*   **Essential Parameter Calibration:** The introduction of new strategic components, or modifications to existing ones, necessitates rigorous parameter optimization and comprehensive backtesting. The default values provided herein serve purely as illustrative examples.
*   **Visual Clarity & Accessibility:** The enhanced neon palette aims for vibrancy but has been carefully balanced to ensure sustained readability across diverse terminal backgrounds and user preferences.

Prepare to witness the amplified power and clarity of the **Unified Scalping Spell v2.8.0 (Strategic Illumination)**:

```python
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Unified Scalping Spell v2.8.0 (Strategic Illumination)
# Features reworked strategy logic (Dual ST + Momentum) and enhanced Neon Colorization.
# Previous Weave: v2.6.1 (Phoenix Feather Resilience & Compatibility Weave)

"""High-Frequency Trading Bot (Scalping) for Bybit USDT Futures
Version: 2.8.0 (Strategic Illumination)

Enhancements:
- Reworked Strategy Logic: Example implementation of Dual Supertrend with Momentum confirmation.
- Enhanced Neon Colorization: More distinct and thematic color usage for terminal output.
- Concrete implementation of Supertrend and Momentum indicators using pandas_ta.
- Added get_market_data function for fetching OHLCV data.

Core Features from v2.6.1 (Persistence, Dynamic ATR SL, Pyramiding Foundation) remain.
"""

# Standard Library Imports
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import Any, Union, Dict, List, Tuple, Optional, Type

import pytz

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    if not hasattr(pd, 'NA'): raise ImportError("Pandas version < 1.0 not supported.")
    import pandas_ta as ta # type: ignore[import]
    from colorama import Back, Fore, Style # The Prisms of Perception
    from colorama import init as colorama_init
    from dotenv import load_dotenv
    from retry import retry
except ImportError as e:
    missing_pkg = getattr(e, 'name', 'dependency'); sys.stderr.write(f"\033[91mCRITICAL ERROR: Missing/Incompatible Essence: '{missing_pkg}'.\033[0m\n"); sys.exit(1)

# --- Constants ---
STATE_FILE_NAME = "pyrmethus_phoenix_state_v280.json"
STATE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILE_NAME)
HEARTBEAT_INTERVAL_SECONDS = 60

# --- Neon Color Palette (Enhanced) ---
NEON = {
    "INFO": Fore.CYAN,
    "DEBUG": Fore.BLUE + Style.DIM,
    "WARNING": Fore.YELLOW + Style.BRIGHT,
    "ERROR": Fore.RED + Style.BRIGHT,
    "CRITICAL": Back.RED + Fore.WHITE + Style.BRIGHT,
    "SUCCESS": Fore.GREEN + Style.BRIGHT,
    "STRATEGY": Fore.MAGENTA,
    "PARAM": Fore.LIGHTBLUE_EX,
    "VALUE": Fore.LIGHTYELLOW_EX + Style.BRIGHT,
    "PRICE": Fore.LIGHTGREEN_EX + Style.BRIGHT, # Changed from LIGHTGREEN_EX to LIGHTGREEN_EX + Style.BRIGHT
    "QTY": Fore.LIGHTCYAN_EX + Style.BRIGHT,    # Added Style.BRIGHT
    "PNL_POS": Fore.GREEN + Style.BRIGHT,
    "PNL_NEG": Fore.RED + Style.BRIGHT,
    "PNL_ZERO": Fore.YELLOW,
    "SIDE_LONG": Fore.GREEN,
    "SIDE_SHORT": Fore.RED,
    "SIDE_FLAT": Fore.BLUE,
    "HEADING": Fore.MAGENTA + Style.BRIGHT,
    "SUBHEADING": Fore.CYAN + Style.BRIGHT,
    "ACTION": Fore.YELLOW + Style.BRIGHT,
    "RESET": Style.RESET_ALL
}

# --- Initializations ---
colorama_init(autoreset=True)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if load_dotenv(dotenv_path=env_path): logging.getLogger(__name__).info(f"{NEON['INFO']}Secrets whispered from .env scroll: {env_path}{NEON['RESET']}")
else: logging.getLogger(__name__).warning(f"{NEON['WARNING']}No .env scroll at {env_path}. Relying on system vars/defaults.{NEON['RESET']}")
getcontext().prec = 18

# --- Enums ---
class StrategyName(str, Enum): DUAL_SUPERTREND_MOMENTUM="DUAL_SUPERTREND_MOMENTUM"; EHLERS_FISHER="EHLERS_FISHER"; # Example, add others
class VolatilityRegime(Enum): LOW="LOW"; NORMAL="NORMAL"; HIGH="HIGH"
class OrderEntryType(str, Enum): MARKET="MARKET"; LIMIT="LIMIT"

# --- Configuration Class ---
class Config:
    def __init__(self) -> None:
        _pre_logger = logging.getLogger(__name__)
        _pre_logger.info(f"{NEON['HEADING']}--- Summoning Configuration Runes v2.8.0 ---{NEON['RESET']}")
        self.api_key: str = self._get_env("BYBIT_API_KEY", required=True, secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", required=True, secret=True)
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT")
        self.interval: str = self._get_env("INTERVAL", "1m")
        self.leverage: int = self._get_env("LEVERAGE", 25, cast_type=int)
        self.sleep_seconds: int = self._get_env("SLEEP_SECONDS", 10, cast_type=int)
        self.strategy_name: StrategyName = StrategyName(self._get_env("STRATEGY_NAME", StrategyName.DUAL_SUPERTREND_MOMENTUM.value).upper())
        self.strategy_instance: 'TradingStrategy'
        # Risk Management
        self.risk_per_trade_percentage: Decimal = self._get_env("RISK_PER_TRADE_PERCENTAGE", "0.01", cast_type=Decimal)
        self.enable_dynamic_risk: bool = self._get_env("ENABLE_DYNAMIC_RISK", "false", cast_type=bool)
        self.dynamic_risk_min_pct: Decimal = self._get_env("DYNAMIC_RISK_MIN_PCT", "0.005", cast_type=Decimal)
        self.dynamic_risk_max_pct: Decimal = self._get_env("DYNAMIC_RISK_MAX_PCT", "0.015", cast_type=Decimal)
        self.dynamic_risk_perf_window: int = self._get_env("DYNAMIC_RISK_PERF_WINDOW", 10, cast_type=int)
        self.max_order_usdt_amount: Decimal = self._get_env("MAX_ORDER_USDT_AMOUNT", "50.0", cast_type=Decimal)
        self.required_margin_buffer: Decimal = self._get_env("REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal)
        self.max_account_margin_ratio: Decimal = self._get_env("MAX_ACCOUNT_MARGIN_RATIO", "0.8", cast_type=Decimal)
        self.enable_max_drawdown_stop: bool = self._get_env("ENABLE_MAX_DRAWDOWN_STOP", "true", cast_type=bool)
        self.max_drawdown_percent: Decimal = self._get_env("MAX_DRAWDOWN_PERCENT", "0.10", cast_type=Decimal)
        self.enable_time_based_stop: bool = self._get_env("ENABLE_TIME_BASED_STOP", "false", cast_type=bool)
        self.max_trade_duration_seconds: int = self._get_env("MAX_TRADE_DURATION_SECONDS", 3600, cast_type=int)
        # Dynamic ATR SL
        self.enable_dynamic_atr_sl: bool = self._get_env("ENABLE_DYNAMIC_ATR_SL", "true", cast_type=bool)
        self.atr_short_term_period: int = self._get_env("ATR_SHORT_TERM_PERIOD", 7, cast_type=int)
        self.atr_long_term_period: int = self._get_env("ATR_LONG_TERM_PERIOD", 50, cast_type=int)
        self.volatility_ratio_low_threshold: Decimal = self._get_env("VOLATILITY_RATIO_LOW_THRESHOLD", "0.7", cast_type=Decimal)
        self.volatility_ratio_high_threshold: Decimal = self._get_env("VOLATILITY_RATIO_HIGH_THRESHOLD", "1.5", cast_type=Decimal)
        self.atr_sl_multiplier_low_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_LOW_VOL", "1.0", cast_type=Decimal)
        self.atr_sl_multiplier_normal_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_NORMAL_VOL", "1.3", cast_type=Decimal)
        self.atr_sl_multiplier_high_vol: Decimal = self._get_env("ATR_SL_MULTIPLIER_HIGH_VOL", "1.8", cast_type=Decimal)
        self.atr_stop_loss_multiplier: Decimal = self._get_env("ATR_STOP_LOSS_MULTIPLIER", "1.3", cast_type=Decimal) # Fallback
        # Position Scaling
        self.enable_position_scaling: bool = self._get_env("ENABLE_POSITION_SCALING", "true", cast_type=bool)
        self.max_scale_ins: int = self._get_env("MAX_SCALE_INS", 1, cast_type=int)
        self.scale_in_risk_percentage: Decimal = self._get_env("SCALE_IN_RISK_PERCENTAGE", "0.005", cast_type=Decimal)
        self.min_profit_for_scale_in_atr: Decimal = self._get_env("MIN_PROFIT_FOR_SCALE_IN_ATR", "1.0", cast_type=Decimal)
        self.enable_scale_out: bool = self._get_env("ENABLE_SCALE_OUT", "false", cast_type=bool)
        self.scale_out_trigger_atr: Decimal = self._get_env("SCALE_OUT_TRIGGER_ATR", "2.0", cast_type=Decimal)
        # TSL
        self.trailing_stop_percentage: Decimal = self._get_env("TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal)
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env("TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal)
        # Execution
        self.entry_order_type: OrderEntryType = OrderEntryType(self._get_env("ENTRY_ORDER_TYPE", OrderEntryType.MARKET.value).upper())
        self.limit_order_offset_pips: int = self._get_env("LIMIT_ORDER_OFFSET_PIPS", 2, cast_type=int)
        self.limit_order_fill_timeout_seconds: int = self._get_env("LIMIT_ORDER_FILL_TIMEOUT_SECONDS", 20, cast_type=int)
        # Strategy Specific: Dual Supertrend Momentum
        self.st_atr_length: int = self._get_env("ST_ATR_LENGTH", 10, cast_type=int)
        self.st_multiplier: Decimal = self._get_env("ST_MULTIPLIER", "2.0", cast_type=Decimal)
        self.confirm_st_atr_length: int = self._get_env("CONFIRM_ST_ATR_LENGTH", 20, cast_type=int)
        self.confirm_st_multiplier: Decimal = self._get_env("CONFIRM_ST_MULTIPLIER", "3.0", cast_type=Decimal)
        self.momentum_period: int = self._get_env("MOMENTUM_PERIOD", 14, cast_type=int) # For DualST+Momentum
        self.momentum_threshold: Decimal = self._get_env("MOMENTUM_THRESHOLD", "0", cast_type=Decimal) # e.g. Mom > 0 for long
        # Misc
        self.volume_ma_period: int = self._get_env("VOLUME_MA_PERIOD", 20, cast_type=int)
        self.atr_calculation_period: int = self.atr_short_term_period if self.enable_dynamic_atr_sl else self._get_env("ATR_CALCULATION_PERIOD", 14, cast_type=int)
        self.enable_sms_alerts: bool = self._get_env("ENABLE_SMS_ALERTS", "true", cast_type=bool)
        self.sms_recipient_number: Optional[str] = self._get_env("SMS_RECIPIENT_NUMBER", None, cast_type=str)
        self.sms_timeout_seconds: int = self._get_env("SMS_TIMEOUT_SECONDS", 30, cast_type=int)
        self.default_recv_window: int = self._get_env("DEFAULT_RECV_WINDOW", 13000, cast_type=int)
        self.order_book_depth: int = self._get_env("ORDER_BOOK_DEPTH", 20, cast_type=int)
        self.order_book_fetch_limit: int = max(25, self.order_book_depth)
        self.shallow_ob_fetch_depth: int = 5
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 15, cast_type=int)
        self.side_buy: str = "buy"; self.side_sell: str = "sell"; self.pos_long: str = "Long"; self.pos_short: str = "Short"; self.pos_none: str = "None"
        self.usdt_symbol: str = "USDT"; self.retry_count: int = 3; self.retry_delay_seconds: int = 2; self.api_fetch_limit_buffer: int = 20 # Increased buffer
        self.position_qty_epsilon: Decimal = Decimal("1e-9"); self.post_close_delay_seconds: int = 3; self.cache_candle_duration_multiplier: Decimal = Decimal("0.95")
        _pre_logger.info(f"{NEON['HEADING']}--- Configuration Runes v2.8.0 Summoned ---{NEON['RESET']}")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False, color: str = NEON["PARAM"], secret: bool = False) -> Any:
        _pre_logger = logging.getLogger(__name__)
        value_str = os.getenv(key); source = "Env Var"; value_to_cast: Any = None; display_value = "*******" if secret and value_str else value_str
        if value_str is None:
            if required: raise ValueError(f"Required env var '{key}' not set.")
            value_to_cast = default; source = "Default"
        else: value_to_cast = value_str
        if value_to_cast is None:
            if required: raise ValueError(f"Required env var '{key}' is None.")
            return None
        final_value: Any
        try:
            raw_value_str = str(value_to_cast)
            if cast_type == bool: final_value = raw_value_str.lower() in ["true", "1", "yes", "y"]
            elif cast_type == Decimal: final_value = Decimal(raw_value_str)
            elif cast_type == int: final_value = int(Decimal(raw_value_str))
            elif cast_type == float: final_value = float(raw_value_str)
            elif cast_type == str: final_value = raw_value_str
            else: final_value = value_to_cast
        except (ValueError, TypeError, InvalidOperation) as e:
            _pre_logger.error(f"{NEON['ERROR']}Invalid type for '{key}': '{value_to_cast}'. Using default '{default}'. Err: {e}{NEON['RESET']}")
            if default is None and required: raise ValueError(f"Required '{key}' failed cast, no valid default.")
            final_value = default; source = "Default (Fallback)"
            if final_value is not None:
                try:
                    default_str = str(default)
                    if cast_type == bool: final_value = default_str.lower() in ["true", "1", "yes", "y"]
                    elif cast_type == Decimal: final_value = Decimal(default_str)
                    elif cast_type == int: final_value = int(Decimal(default_str))
                    elif cast_type == float: final_value = float(default_str)
                    elif cast_type == str: final_value = default_str
                except (ValueError, TypeError, InvalidOperation) as e_default: raise ValueError(f"Cannot cast value or default for '{key}': {e_default}")
        display_final_value = "*******" if secret else final_value
        _pre_logger.debug(f"{color}Config Rune '{NEON['VALUE']}{key}{color}': Using value '{NEON['VALUE']}{display_final_value}{color}' (Type: {type(final_value).__name__}, Source: {source}){NEON['RESET']}")
        return final_value

# --- Logger Setup ---
LOGGING_LEVEL: int = (logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)
logs_dir = "logs"; os.makedirs(logs_dir, exist_ok=True)
log_file_name = f"{logs_dir}/pyrmethus_spell_v280_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_name)])
logger: logging.Logger = logging.getLogger("PyrmethusCore")
SUCCESS_LEVEL: int = 25; logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL): self._log(SUCCESS_LEVEL, message, args, **kwargs) # type: ignore[attr-defined]
logging.Logger.success = log_success # type: ignore[attr-defined]
if sys.stdout.isatty(): # Apply NEON colors if TTY
    logging.addLevelName(logging.DEBUG, f"{NEON['DEBUG']}{logging.getLevelName(logging.DEBUG)}{NEON['RESET']}")
    logging.addLevelName(logging.INFO, f"{NEON['INFO']}{logging.getLevelName(logging.INFO)}{NEON['RESET']}")
    logging.addLevelName(SUCCESS_LEVEL, f"{NEON['SUCCESS']}{logging.getLevelName(SUCCESS_LEVEL)}{NEON['RESET']}") # SUCCESS uses its own bright green
    logging.addLevelName(logging.WARNING, f"{NEON['WARNING']}{logging.getLevelName(logging.WARNING)}{NEON['RESET']}")
    logging.addLevelName(logging.ERROR, f"{NEON['ERROR']}{logging.getLevelName(logging.ERROR)}{NEON['RESET']}")
    logging.addLevelName(logging.CRITICAL, f"{NEON['CRITICAL']}{logging.getLevelName(logging.CRITICAL)}{NEON['RESET']}")

# --- Global Objects ---
try: CONFIG = Config()
except Exception as e: logging.getLogger().critical(f"{NEON['CRITICAL']}Config Error: {e}. Pyrmethus cannot weave.{NEON['RESET']}"); sys.exit(1)

# --- Trading Strategy ABC & Implementations ---
class TradingStrategy(ABC):
    def __init__(self, config: Config, df_columns: Optional[List[str]] = None): self.config=config; self.logger=logging.getLogger(f"{NEON['STRATEGY']}Strategy.{self.__class__.__name__}{NEON['RESET']}"); self.required_columns=df_columns or []
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]: pass
    def _validate_df(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        if df is None or df.empty or len(df) < min_rows: self.logger.debug(f"Insufficient data (Rows: {len(df) if df is not None else 0}, Min: {min_rows})."); return False
        if self.required_columns and not all(col in df.columns for col in self.required_columns): self.logger.warning(f"DataFrame missing required columns: {[c for c in self.required_columns if c not in df.columns]}"); return False
        return True
    def _get_default_signals(self) -> Dict[str, Any]: return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Default Signal - Awaiting Omens"}

class DualSupertrendMomentumStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["supertrend_st_long_flip", "supertrend_st_short_flip", "confirm_supertrend_trend", "momentum"]) # Adjusted column names based on new calculate_supertrend
        self.logger.info(f"{NEON['STRATEGY']}Dual Supertrend with Momentum Confirmation strategy initialized.{NEON['RESET']}")

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self._get_default_signals()
        # Ensure enough data for all indicators; max lookback + buffer
        min_rows_needed = max(self.config.st_atr_length, self.config.confirm_st_atr_length, self.config.momentum_period) + 10
        if not self._validate_df(df, min_rows=min_rows_needed):
            return signals

        last = df.iloc[-1]
        # Use updated column names from pandas_ta based calculate_supertrend
        primary_long_flip = last.get("supertrend_st_long_flip", False)
        primary_short_flip = last.get("supertrend_st_short_flip", False)
        confirm_is_up = last.get("confirm_supertrend_trend", pd.NA) # This will be True, False, or pd.NA
        
        momentum_val_orig = last.get("momentum", pd.NA)
        momentum_val = safe_decimal_conversion(momentum_val_orig, pd.NA)

        if pd.isna(confirm_is_up) or pd.isna(momentum_val):
            self.logger.debug(f"Confirmation ST Trend ({_format_for_log(confirm_is_up, is_bool_trend=True)}) or Momentum ({_format_for_log(momentum_val)}) is NA. No signal.")
            return signals
        
        # Entry Signals: Supertrend flip + Confirmation ST direction + Momentum confirmation
        if primary_long_flip and confirm_is_up is True and momentum_val > self.config.momentum_threshold:
            signals["enter_long"] = True
            self.logger.info(f"{NEON['SIDE_LONG']}DualST+Mom Signal: LONG Entry - Primary ST Long Flip, Confirm ST Up, Momentum ({_format_for_log(momentum_val)}) > {_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}")
        elif primary_short_flip and confirm_is_up is False and momentum_val < -self.config.momentum_threshold: # Assuming symmetrical threshold for short
            signals["enter_short"] = True
            self.logger.info(f"{NEON['SIDE_SHORT']}DualST+Mom Signal: SHORT Entry - Primary ST Short Flip, Confirm ST Down, Momentum ({_format_for_log(momentum_val)}) < -{_format_for_log(self.config.momentum_threshold)}{NEON['RESET']}")

        # Exit Signals: Based on primary SuperTrend flips
        if primary_short_flip: # If primary ST flips short, exit any long position.
            signals["exit_long"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Short"
            self.logger.info(f"{NEON['ACTION']}DualST+Mom Signal: EXIT LONG - Primary ST Flipped Short{NEON['RESET']}")
        if primary_long_flip: # If primary ST flips long, exit any short position.
            signals["exit_short"] = True; signals["exit_reason"] = "Primary SuperTrend Flipped Long"
            self.logger.info(f"{NEON['ACTION']}DualST+Mom Signal: EXIT SHORT - Primary ST Flipped Long{NEON['RESET']}")
        return signals

# Example EhlersFisherStrategy (if chosen via config, not fully integrated in this pass)
class EhlersFisherStrategy(TradingStrategy):
    def __init__(self, config: Config):
        super().__init__(config, df_columns=["ehlers_fisher", "ehlers_signal"])
        self.logger.info(f"{NEON['STRATEGY']}Ehlers Fisher Transform strategy initialized (Illustrative).{NEON['RESET']}")
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Placeholder: Actual Ehlers Fisher logic would go here
        self.logger.debug("EhlersFisherStrategy generate_signals called (placeholder).")
        return self._get_default_signals()


strategy_map: Dict[StrategyName, Type[TradingStrategy]] = {
    StrategyName.DUAL_SUPERTREND_MOMENTUM: DualSupertrendMomentumStrategy,
    StrategyName.EHLERS_FISHER: EhlersFisherStrategy # Example mapping
}
StrategyClass = strategy_map.get(CONFIG.strategy_name)
if StrategyClass: CONFIG.strategy_instance = StrategyClass(CONFIG)
else: logger.critical(f"{NEON['CRITICAL']}Failed to init strategy '{CONFIG.strategy_name.value}'. Exiting.{NEON['RESET']}"); sys.exit(1)

# --- Trade Metrics Tracking ---
class TradeMetrics:
    def __init__(self): self.trades: List[Dict[str, Any]] = []; self.logger = logging.getLogger("TradeMetrics"); self.initial_equity: Optional[Decimal] = None; self.daily_start_equity: Optional[Decimal] = None; self.last_daily_reset_day: Optional[int] = None
    def set_initial_equity(self, equity: Decimal):
        if self.initial_equity is None: self.initial_equity = equity
        today = datetime.now(pytz.utc).day
        if self.last_daily_reset_day != today: self.daily_start_equity = equity; self.last_daily_reset_day = today; self.logger.info(f"{NEON['INFO']}Daily equity reset for drawdown. Start Equity: {NEON['VALUE']}{equity:.2f}{NEON['RESET']}")
    def check_drawdown(self, current_equity: Decimal) -> Tuple[bool, str]:
        if not CONFIG.enable_max_drawdown_stop or self.daily_start_equity is None or self.daily_start_equity <= 0: return False, ""
        drawdown_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity
        if drawdown_pct >= CONFIG.max_drawdown_percent: reason = f"Max daily drawdown breached ({drawdown_pct:.2%} >= {CONFIG.max_drawdown_percent:.2%})"; return True, reason
        return False, ""
    def log_trade(self, symbol: str, side: str, entry_price: Decimal, exit_price: Decimal, qty: Decimal, entry_time_ms: int, exit_time_ms: int, reason: str, scale_order_id: Optional[str]=None, part_id: Optional[str]=None, mae: Optional[Decimal]=None, mfe: Optional[Decimal]=None) -> None:
        if not (entry_price > 0 and exit_price > 0 and qty > 0): return
        profit_per_unit = (exit_price - entry_price) if (side.lower() == CONFIG.side_buy.lower() or side.lower() == CONFIG.pos_long.lower()) else (entry_price - exit_price)
        profit = profit_per_unit * qty; entry_dt_iso = datetime.fromtimestamp(entry_time_ms / 1000, tz=pytz.utc).isoformat(); exit_dt_iso = datetime.fromtimestamp(exit_time_ms / 1000, tz=pytz.utc).isoformat()
        duration_seconds = (datetime.fromisoformat(exit_dt_iso) - datetime.fromisoformat(entry_dt_iso)).total_seconds(); trade_type = "Scale-In" if scale_order_id else ("Initial" if part_id == "initial" else "Part")
        self.trades.append({"symbol": symbol, "side": side, "entry_price_str": str(entry_price), "exit_price_str": str(exit_price), "qty_str": str(qty), "profit_str": str(profit), "entry_time_iso": entry_dt_iso, "exit_time_iso": exit_dt_iso, "duration_seconds": duration_seconds, "exit_reason": reason, "type": trade_type, "part_id": part_id or "unknown", "scale_order_id": scale_order_id, "mae_str": str(mae) if mae is not None else None, "mfe_str": str(mfe) if mfe is not None else None})
        pnl_color = NEON["PNL_POS"] if profit > 0 else (NEON["PNL_NEG"] if profit < 0 else NEON["PNL_ZERO"])
        self.logger.log(SUCCESS_LEVEL, f"{NEON['HEADING']}Trade Chronicle ({trade_type} Part:{part_id or 'N/A'}): {side.upper()} {NEON['QTY']}{qty}{NEON['RESET']} {symbol.split('/')[0]} | P/L: {pnl_color}{profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']} | Reason: {reason}")
    def calculate_mae_mfe(self, part_id: str, entry_price: Decimal, exit_price: Decimal, side: str, entry_time_ms: int, exit_time_ms: int, exchange: ccxt.Exchange, symbol: str, interval: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        self.logger.debug(f"MAE/MFE calculation for part {part_id} skipped (placeholder - requires fetching historical OHLCV for trade duration).")
        # Placeholder: In a full implementation, fetch OHLCV data between entry_time_ms and exit_time_ms
        # then find min/max prices during the trade to calculate actual MAE/MFE.
        return None, None
    def get_performance_trend(self, window: int) -> float:
        if window <= 0 or not self.trades: return 0.5
        recent_trades = self.trades[-window:];
        if not recent_trades: return 0.5
        wins = sum(1 for t in recent_trades if Decimal(t["profit_str"]) > 0); return float(wins / len(recent_trades))
    def summary(self) -> str:
        if not self.trades: return "The Grand Ledger is empty."
        total_trades = len(self.trades); wins = sum(1 for t in self.trades if Decimal(t["profit_str"]) > 0); losses = sum(1 for t in self.trades if Decimal(t["profit_str"]) < 0); breakeven = total_trades - wins - losses
        win_rate = (Decimal(wins) / Decimal(total_trades)) * Decimal(100) if total_trades > 0 else Decimal(0); total_profit = sum(Decimal(t["profit_str"]) for t in self.trades); avg_profit = total_profit / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        summary_str = (f"\n{NEON['HEADING']}--- Pyrmethus Trade Metrics Summary ---{NEON['RESET']}\n"
                       f"Total Trade Parts Chronicled: {NEON['VALUE']}{total_trades}{NEON['RESET']}\n"
                       f"  Victories (Wins): {NEON['PNL_POS']}{wins}{NEON['RESET']}\n"
                       f"  Defeats (Losses): {NEON['PNL_NEG']}{losses}{NEON['RESET']}\n"
                       f"  Stalemates (Breakeven): {NEON['PNL_ZERO']}{breakeven}{NEON['RESET']}\n"
                       f"Victory Rate (by parts): {NEON['VALUE']}{win_rate:.2f}%{NEON['RESET']}\n"
                       f"Total Spoils (P/L): {(NEON['PNL_POS'] if total_profit > 0 else NEON['PNL_NEG'])}{total_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
                       f"Avg Spoils per Part: {(NEON['PNL_POS'] if avg_profit > 0 else NEON['PNL_NEG'])}{avg_profit:.2f} {CONFIG.usdt_symbol}{NEON['RESET']}\n"
                       f"{NEON['HEADING']}--- End of Grand Ledger ---{NEON['RESET']}")
        self.logger.info(summary_str); return summary_str
    def get_serializable_trades(self) -> List[Dict[str, Any]]: return self.trades
    def load_trades_from_list(self, trades_list: List[Dict[str, Any]]) -> None: self.trades = trades_list; self.logger.info(f"{NEON['INFO']}TradeMetrics: Re-inked {len(self.trades)} trades from Phoenix scroll.{NEON['RESET']}")

trade_metrics = TradeMetrics()
_active_trade_parts: List[Dict[str, Any]] = []
_last_heartbeat_save_time: float = 0.0

# --- State Persistence Functions ---
def save_persistent_state(force_heartbeat: bool = False) -> None:
    global _active_trade_parts, trade_metrics, _last_heartbeat_save_time; now = time.time()
    if force_heartbeat or now - _last_heartbeat_save_time > HEARTBEAT_INTERVAL_SECONDS:
        try:
            serializable_active_parts = []
            for part in _active_trade_parts:
                serializable_part = part.copy()
                for key, value in serializable_part.items():
                    if isinstance(value, Decimal): serializable_part[key] = str(value)
                    if isinstance(value, (datetime, pd.Timestamp)): serializable_part[key] = value.isoformat() if hasattr(value,'isoformat') else str(value)
                serializable_active_parts.append(serializable_part)
            state_data = {"timestamp_utc_iso": datetime.now(pytz.utc).isoformat(), "last_heartbeat_utc_iso": datetime.now(pytz.utc).isoformat(), "active_trade_parts": serializable_active_parts, "trade_metrics_trades": trade_metrics.get_serializable_trades(), "config_symbol": CONFIG.symbol, "config_strategy": CONFIG.strategy_name.value, "daily_start_equity_str": str(trade_metrics.daily_start_equity) if trade_metrics.daily_start_equity else None, "last_daily_reset_day": trade_metrics.last_daily_reset_day}
            temp_file_path = STATE_FILE_PATH + ".tmp";
            with open(temp_file_path, 'w') as f: json.dump(state_data, f, indent=4)
            os.replace(temp_file_path, STATE_FILE_PATH); _last_heartbeat_save_time = now
            logger.log(logging.DEBUG if not force_heartbeat else logging.INFO, f"{NEON['SUCCESS']}Phoenix Feather: State memories scribed.{NEON['RESET']}")
        except Exception as e: logger.error(f"{NEON['ERROR']}Phoenix Feather Error scribing: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())
def load_persistent_state() -> bool:
    global _active_trade_parts, trade_metrics;
    if not os.path.exists(STATE_FILE_PATH): logger.info(f"{NEON['INFO']}Phoenix Feather: No scroll. Starting fresh.{NEON['RESET']}"); return False
    try:
        with open(STATE_FILE_PATH, 'r') as f: state_data = json.load(f)
        if state_data.get("config_symbol") != CONFIG.symbol or state_data.get("config_strategy") != CONFIG.strategy_name.value:
            logger.warning(f"{NEON['WARNING']}Phoenix Feather: Scroll sigils mismatch. Ignoring.{NEON['RESET']}"); os.remove(STATE_FILE_PATH); return False
        loaded_active_parts = state_data.get("active_trade_parts", []); _active_trade_parts.clear()
        for part_data in loaded_active_parts:
            restored_part = part_data.copy()
            for key, value_str in restored_part.items():
                if key in ["entry_price", "qty", "sl_price"] and isinstance(value_str, str):
                    try: restored_part[key] = Decimal(value_str)
                    except InvalidOperation: logger.warning(f"Could not convert '{value_str}' to Decimal for key '{key}'.")
                if key == "entry_time_ms" and isinstance(value_str, str):
                     try: restored_part[key] = int(datetime.fromisoformat(value_str).timestamp() * 1000)
                     except: pass
            _active_trade_parts.append(restored_part)
        trade_metrics.load_trades_from_list(state_data.get("trade_metrics_trades", []))
        daily_start_equity_str = state_data.get("daily_start_equity_str");
        if daily_start_equity_str: trade_metrics.daily_start_equity = Decimal(daily_start_equity_str)
        trade_metrics.last_daily_reset_day = state_data.get("last_daily_reset_day")
        saved_time_str = state_data.get("timestamp_utc_iso", "ancient times")
        logger.success(f"{NEON['SUCCESS']}Phoenix Feather: Memories from {NEON['VALUE']}{saved_time_str}{NEON['SUCCESS']} reawakened!{NEON['RESET']}")
        return True
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Phoenix Feather Error reawakening: {e}. Starting fresh.{NEON['RESET']}"); logger.debug(traceback.format_exc())
        try: os.remove(STATE_FILE_PATH)
        except OSError: pass
        _active_trade_parts.clear(); trade_metrics.trades.clear()
        return False

# --- Helper Functions, Retry, SMS, Exchange Init ---
PandasNAType = type(pd.NA)
def safe_decimal_conversion(value: Any, default: Union[Decimal, PandasNAType, None] = Decimal("0.0")) -> Union[Decimal, PandasNAType, None]:
    if pd.isna(value) or value is None: return default
    try: return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError): return default
def format_order_id(order_id: Union[str, int, None]) -> str: return str(order_id)[-6:] if order_id else "N/A"
def _format_for_log(value: Any, precision: int = 4, is_bool_trend: bool = False, color: Optional[str] = NEON["VALUE"]) -> str:
    reset = NEON["RESET"]
    if pd.isna(value) or value is None: return f"{Style.DIM}N/A{reset}"
    if is_bool_trend: return f"{NEON['SIDE_LONG']}Upward Flow{reset}" if value is True else (f"{NEON['SIDE_SHORT']}Downward Tide{reset}" if value is False else f"{Style.DIM}N/A (Trend Indeterminate){reset}")
    if isinstance(value, Decimal): return f"{color}{value:.{precision}f}{reset}"
    if isinstance(value, (float, int)): return f"{color}{float(value):.{precision}f}{reset}"
    return f"{color}{str(value)}{reset}"
def format_price(exchange: ccxt.Exchange, symbol: str, price: Union[float, Decimal]) -> str:
    try: return exchange.price_to_precision(symbol, float(price))
    except: return str(Decimal(str(price)).quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP).normalize())
def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Union[float, Decimal]) -> str:
    try: return exchange.amount_to_precision(symbol, float(amount))
    except: return str(Decimal(str(amount)).quantize(Decimal('1e-8'), rounding=ROUND_HALF_UP).normalize())
@retry(tries=CONFIG.retry_count, delay=CONFIG.retry_delay_seconds, backoff=2, logger=logger, exceptions=(ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection))
def safe_api_call(func, *args, **kwargs): return func(*args, **kwargs)
_termux_sms_command_exists: Optional[bool] = None
def send_sms_alert(message: str) -> bool: logger.info(f"{NEON['STRATEGY']}SMS (Simulated): {message}{NEON['RESET']}"); return True # Simplified
def initialize_exchange() -> Optional[ccxt.Exchange]:
    logger.info(f"{NEON['INFO']}{Style.BRIGHT}Opening Bybit Portal v2.8.0...{NEON['RESET']}")
    if not CONFIG.api_key or not CONFIG.api_secret: logger.warning(f"{NEON['WARNING']}API keys not set. Using a MOCK exchange object.{NEON['RESET']}"); return MockExchange()
    try:
        exchange = ccxt.bybit({"apiKey": CONFIG.api_key, "secret": CONFIG.api_secret, "enableRateLimit": True, "options": {"defaultType": "linear", "adjustForTimeDifference": True}, "recvWindow": CONFIG.default_recv_window})
        exchange.load_markets(force_reload=True); exchange.fetch_balance(params={"category": "linear"})
        logger.success(f"{NEON['SUCCESS']}Portal to Bybit Opened & Authenticated (V5 API).{NEON['RESET']}")
        if hasattr(exchange, 'sandbox') and exchange.sandbox: logger.warning(f"{Back.YELLOW}{Fore.BLACK}TESTNET MODE{NEON['RESET']}")
        else: logger.warning(f"{NEON['CRITICAL']}LIVE TRADING MODE - EXTREME CAUTION{NEON['RESET']}")
        return exchange
    except Exception as e: logger.critical(f"{NEON['CRITICAL']}Portal Opening FAILED: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc()); return None
class MockExchange:
    def __init__(self): self.id="mock_bybit"; self.options={"defaultType":"linear"}; self.markets={CONFIG.symbol:{"id":CONFIG.symbol.replace("/",""),"symbol":CONFIG.symbol,"contract":True,"linear":True,"limits":{"amount":{"min":0.001},"price":{"min":0.1}},"precision":{"amount":3,"price":1}}}; self.sandbox=True
    def market(self,s): return self.markets.get(s); def load_markets(self,force_reload=False): pass
    def fetch_balance(self,params=None): return {CONFIG.usdt_symbol:{"free":Decimal("10000"),"total":Decimal("10000"),"used":Decimal("0")}}
    def fetch_ticker(self,s): return {"last":Decimal("30000.0"),"bid":Decimal("29999.0"),"ask":Decimal("30001.0")}
    def fetch_positions(self,symbols=None,params=None): global _active_trade_parts; qty=sum(p['qty'] for p in _active_trade_parts); side=_active_trade_parts[0]['side'] if _active_trade_parts else CONFIG.pos_none; avg_px=sum(p['entry_price']*p['qty'] for p in _active_trade_parts)/qty if qty>0 else 0; return [{"info":{"symbol":self.markets[CONFIG.symbol]['id'],"positionIdx":0,"size":str(qty),"avgPrice":str(avg_px),"side":"Buy" if side==CONFIG.pos_long else "Sell"}}] if qty > 0 else []
    def create_market_order(self,s,side,amt,params=None): return {"id":f"mock_mkt_{int(time.time()*1000)}","status":"closed","average":self.fetch_ticker(s)['last'],"filled":amt,"timestamp":int(time.time()*1000)}
    def create_limit_order(self,s,side,amt,price,params=None): return {"id":f"mock_lim_{int(time.time()*1000)}","status":"open","price":price}
    def create_order(self,s,type,side,amt,price=None,params=None): return {"id":f"mock_cond_{int(time.time()*1000)}","status":"open"}
    def fetch_order(self,id,s,params=None):
        if "lim_" in id: time.sleep(0.05); return {"id":id,"status":"closed","average": self.fetch_ticker(s)['last'],"filled":"1","timestamp":int(time.time()*1000)} # Simulate limit fill
        return {"id":id,"status":"closed","average": self.fetch_ticker(s)['last'],"filled":"1","timestamp":int(time.time()*1000)}
    def fetch_open_orders(self,s,params=None): return []
    def cancel_order(self,id,s,params=None): return {"id":id,"status":"canceled"}
    def set_leverage(self,l,s,params=None): return {"status":"ok"}
    def price_to_precision(self,s,p): return f"{float(p):.{self.markets[s]['precision']['price']}f}"
    def amount_to_precision(self,s,a): return f"{float(a):.{self.markets[s]['precision']['amount']}f}"
    def parse_timeframe(self,tf): return 60 if tf=="1m" else 300
    has={"fetchOHLCV":True,"fetchL2OrderBook":True}
    def fetch_ohlcv(self,s,tf,lim,params=None): now=int(time.time()*1000);tfs=self.parse_timeframe(tf);d=[] ; for i in range(lim): ts=now-(lim-1-i)*tfs*1000;p=30000+(i-lim/2)*10 + (time.time()%100 - 50) ;d.append([ts,p,p+5,p-5,p+(i%3-1)*2,100+i]); return d
    def fetch_l2_order_book(self,s,limit=None): last=self.fetch_ticker(s)['last'];bids=[[float(last)-i*0.1,1.0+i*0.1] for i in range(1,(limit or 5)+1)];asks=[[float(last)+i*0.1,1.0+i*0.1] for i in range(1,(limit or 5)+1)];return {"bids":bids,"asks":asks}

# --- Indicator Calculation Functions ---
vol_atr_analysis_results_cache: Dict[str, Any] = {}

def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates Supertrend indicator using pandas-ta."""
    col_prefix = f"{prefix}" if prefix else ""
    st_col_name_base = f"{col_prefix}supertrend" # Base for our column names
    
    val_col = f"{st_col_name_base}_val"
    trend_col = f"{st_col_name_base}_trend" 
    long_flip_col = f"{st_col_name_base}_st_long_flip"
    short_flip_col = f"{st_col_name_base}_st_short_flip"
    output_cols = [val_col, trend_col, long_flip_col, short_flip_col]

    if not all(c in df.columns for c in ["high", "low", "close"]) or df.empty or len(df) < length + 1: # pandas_ta needs at least length + 1
        logger.warning(f"{NEON['WARNING']}Scrying (Supertrend {prefix}): Insufficient data (Rows: {len(df)}, Min: {length+1}). Populating NAs.{NEON['RESET']}")
        for col in output_cols: df[col] = pd.NA
        return df

    try:
        st_df = df.ta.supertrend(length=length, multiplier=float(multiplier), append=False)
        
        pta_val_col_pattern = f"SUPERT_{length}_{float(multiplier):.1f}" 
        pta_dir_col_pattern = f"SUPERTd_{length}_{float(multiplier):.1f}"

        if pta_val_col_pattern not in st_df.columns or pta_dir_col_pattern not in st_df.columns:
            logger.error(f"{NEON['ERROR']}Scrying (Supertrend {prefix}): pandas_ta did not return expected columns. Found: {st_df.columns}. Expected patterns like: {pta_val_col_pattern}, {pta_dir_col_pattern}{NEON['RESET']}")
            for col in output_cols: df[col] = pd.NA
            return df

        df[val_col] = st_df[pta_val_col_pattern].apply(lambda x: safe_decimal_conversion(x, pd.NA))
        df[trend_col] = st_df[pta_dir_col_pattern].apply(lambda x: True if x == 1 else (False if x == -1 else pd.NA))

        prev_trend = df[trend_col].shift(1)
        df[long_flip_col] = (prev_trend == False) & (df[trend_col] == True)
        df[short_flip_col] = (prev_trend == True) & (df[trend_col] == False)
        
        df[long_flip_col] = df[long_flip_col].fillna(False)
        df[short_flip_col] = df[short_flip_col].fillna(False)

        if not df.empty and not pd.isna(df[val_col].iloc[-1]):
            logger.debug(f"Scrying (Supertrend({length},{multiplier},{prefix})): Value={_format_for_log(df[val_col].iloc[-1], color=NEON['VALUE'])}, Trend={_format_for_log(df[trend_col].iloc[-1], is_bool_trend=True)}, LongFlip={df[long_flip_col].iloc[-1]}, ShortFlip={df[short_flip_col].iloc[-1]}")
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Scrying (Supertrend {prefix}): Error: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        for col in output_cols: df[col] = pd.NA
    return df

def calculate_momentum(df: pd.DataFrame, length: int) -> pd.DataFrame:
    if "close" not in df.columns or df.empty or len(df) < length: 
        df["momentum"] = pd.NA
        logger.warning(f"{NEON['WARNING']}Scrying (Momentum): Insufficient data for momentum calculation (Rows: {len(df)}, Min: {length}). Populating NAs.{NEON['RESET']}")
        return df
    
    # Use pandas_ta.mom() which returns a Series
    mom_series = df.ta.mom(length=length, append=False) 
    df["momentum"] = mom_series.apply(lambda x: safe_decimal_conversion(x, pd.NA))
    
    if not df.empty and "momentum" in df.columns and not pd.isna(df["momentum"].iloc[-1]): 
        logger.debug(f"Scrying (Momentum({length})): Value={_format_for_log(df['momentum'].iloc[-1], color=NEON['VALUE'])}")
    return df

def analyze_volume_atr(df: pd.DataFrame, short_atr_len: int, long_atr_len: int, vol_ma_len: int, dynamic_sl_enabled: bool) -> Dict[str, Union[Decimal, PandasNAType, None]]:
    results: Dict[str, Union[Decimal, PandasNAType, None]] = {"atr_short": pd.NA, "atr_long": pd.NA, "volatility_regime": VolatilityRegime.NORMAL, "volume_ma": pd.NA, "last_volume": pd.NA, "volume_ratio": pd.NA}
    if df.empty or not all(c in df.columns for c in ["high","low","close","volume"]): return results
    try:
        temp_df = df.copy()
        # Ensure columns are numeric before ta functions
        for col_name in ['high', 'low', 'close', 'volume']:
            temp_df[col_name] = pd.to_numeric(temp_df[col_name], errors='coerce')
        temp_df.dropna(subset=['high', 'low', 'close'], inplace=True) # Drop rows if essential HLC are NaN for ATR
        if temp_df.empty: return results

        results["atr_short"] = safe_decimal_conversion(temp_df.ta.atr(length=short_atr_len, append=False).iloc[-1], pd.NA)
        if dynamic_sl_enabled:
            results["atr_long"] = safe_decimal_conversion(temp_df.ta.atr(length=long_atr_len, append=False).iloc[-1], pd.NA)
            atr_s, atr_l = results["atr_short"], results["atr_long"]
            if not pd.isna(atr_s) and not pd.isna(atr_l) and atr_s is not None and atr_l is not None and atr_l > CONFIG.position_qty_epsilon:
                vol_ratio = atr_s / atr_l
                if vol_ratio < CONFIG.volatility_ratio_low_threshold: results["volatility_regime"] = VolatilityRegime.LOW
                elif vol_ratio > CONFIG.volatility_ratio_high_threshold: results["volatility_regime"] = VolatilityRegime.HIGH
        results["last_volume"] = safe_decimal_conversion(df["volume"].iloc[-1], pd.NA) # Use original df for last_volume
    except Exception as e: logger.debug(f"analyze_volume_atr error: {e}")
    return results
def get_current_atr_sl_multiplier() -> Decimal:
    if not CONFIG.enable_dynamic_atr_sl or not vol_atr_analysis_results_cache: return CONFIG.atr_stop_loss_multiplier
    regime = vol_atr_analysis_results_cache.get("volatility_regime", VolatilityRegime.NORMAL)
    if regime == VolatilityRegime.LOW: return CONFIG.atr_sl_multiplier_low_vol
    if regime == VolatilityRegime.HIGH: return CONFIG.atr_sl_multiplier_high_vol
    return CONFIG.atr_sl_multiplier_normal_vol
def calculate_all_indicators(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, Dict[str, Union[Decimal, PandasNAType, None]]]:
    global vol_atr_analysis_results_cache
    df_copy = df.copy() # Work on a copy to avoid modifying original DataFrame passed to trade_logic
    df_copy = calculate_supertrend(df_copy, config.st_atr_length, config.st_multiplier)
    df_copy = calculate_supertrend(df_copy, config.confirm_st_atr_length, config.confirm_st_multiplier, prefix="confirm_")
    df_copy = calculate_momentum(df_copy, config.momentum_period)
    vol_atr_analysis_results_cache = analyze_volume_atr(df_copy, config.atr_short_term_period, config.atr_long_term_period, config.volume_ma_period, config.enable_dynamic_atr_sl)
    return df_copy, vol_atr_analysis_results_cache

# --- Position & Order Management ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    global _active_trade_parts; exchange_pos_data = _get_raw_exchange_position(exchange, symbol)
    if not _active_trade_parts: return exchange_pos_data
    consolidated_qty = sum(part.get('qty', Decimal(0)) for part in _active_trade_parts)
    if consolidated_qty <= CONFIG.position_qty_epsilon: _active_trade_parts.clear(); save_persistent_state(); return exchange_pos_data
    total_value = sum(part.get('entry_price', Decimal(0)) * part.get('qty', Decimal(0)) for part in _active_trade_parts)
    avg_entry_price = total_value / consolidated_qty if consolidated_qty > 0 else Decimal("0"); current_pos_side = _active_trade_parts[0]['side'] if _active_trade_parts else CONFIG.pos_none
    if exchange_pos_data["side"] != current_pos_side or abs(exchange_pos_data["qty"] - consolidated_qty) > CONFIG.position_qty_epsilon: logger.warning(f"{NEON['WARNING']}Position Discrepancy! Bot: {current_pos_side} Qty {consolidated_qty}. Exchange: {exchange_pos_data['side']} Qty {exchange_pos_data['qty']}.{NEON['RESET']}")
    return {"side": current_pos_side, "qty": consolidated_qty, "entry_price": avg_entry_price, "num_parts": len(_active_trade_parts)}
def _get_raw_exchange_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    default_pos_state: Dict[str, Any] = {"side": CONFIG.pos_none, "qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    try:
        market = exchange.market(symbol); market_id = market["id"]; category = "linear" if market.get("linear") else "linear"
        params = {"category": category, "symbol": market_id}; fetched_positions = safe_api_call(exchange.fetch_positions, symbols=[symbol], params=params)
        if not fetched_positions: return default_pos_state
        for pos_data in fetched_positions:
            pos_info = pos_data.get("info", {});
            if pos_info.get("symbol") != market_id: continue
            if int(pos_info.get("positionIdx", -1)) == 0: # One-Way Mode
                size_dec = safe_decimal_conversion(pos_info.get("size", "0"))
                if size_dec > CONFIG.position_qty_epsilon:
                    entry_price_dec = safe_decimal_conversion(pos_info.get("avgPrice")); bybit_side_str = pos_info.get("side")
                    current_pos_side = CONFIG.pos_long if bybit_side_str == "Buy" else (CONFIG.pos_short if bybit_side_str == "Sell" else CONFIG.pos_none)
                    if current_pos_side != CONFIG.pos_none: return {"side": current_pos_side, "qty": size_dec, "entry_price": entry_price_dec}
    except Exception as e: logger.error(f"{NEON['ERROR']}Raw Position Fetch Error: {e}{NEON['RESET']}"); logger.debug(traceback.format_exc())
    return default_pos_state
def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    try:
        response = safe_api_call(exchange.set_leverage, leverage, symbol, params={"category": "linear", "buyLeverage": str(leverage), "sellLeverage": str(leverage)})
        logger.info(f"{NEON['INFO']}Leverage set to {NEON['VALUE']}{leverage}x{NEON['INFO']} for {NEON['VALUE']}{symbol}{NEON['INFO']}. Response: {response}{NEON['RESET']}")
        return True
    except ccxt.ExchangeError as e:
        if "leverage not modified" in str(e).lower() or "110044" in str(e): # Bybit: 110044 for "Set leverage not modified"
            logger.info(f"{NEON['INFO']}Leverage for {NEON['VALUE']}{symbol}{NEON['INFO']} already {NEON['VALUE']}{leverage}x{NEON['INFO']} or no change needed.{NEON['RESET']}")
            return True
        logger.error(f"{NEON['ERROR']}Failed to set leverage for {symbol} to {leverage}x: {e}{NEON['RESET']}")
    except Exception as e_unexp:
        logger.error(f"{NEON['ERROR']}Unexpected error setting leverage for {symbol}: {e_unexp}{NEON['RESET']}")
    return False

def calculate_dynamic_risk() -> Decimal:
    if not CONFIG.enable_dynamic_risk: return CONFIG.risk_per_trade_percentage
    trend = trade_metrics.get_performance_trend(CONFIG.dynamic_risk_perf_window); base_risk = CONFIG.risk_per_trade_percentage; min_risk = CONFIG.dynamic_risk_min_pct; max_risk = CONFIG.dynamic_risk_max_pct
    if trend >= 0.5: scale_factor = (trend - 0.5) / 0.5; dynamic_risk = base_risk + (max_risk - base_risk) * Decimal(scale_factor)
    else: scale_factor = (0.5 - trend) / 0.5; dynamic_risk = base_risk - (base_risk - min_risk) * Decimal(scale_factor)
    final_risk = max(min_risk, min(max_risk, dynamic_risk)); logger.info(f"{NEON['INFO']}Dynamic Risk: Trend={NEON['VALUE']}{trend:.2f}{NEON['INFO']}, BaseRisk={NEON['VALUE']}{base_risk:.3%}{NEON['INFO']}, AdjustedRisk={NEON['VALUE']}{final_risk:.3%}{NEON['RESET']}")
    return final_risk
def calculate_position_size(usdt_equity: Decimal, risk_pct: Decimal, entry: Decimal, sl: Decimal, lev: int, sym: str, ex: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    if not (entry > 0 and sl > 0 and 0 < risk_pct < 1 and usdt_equity > 0 and lev > 0): return None, None
    diff = abs(entry - sl);
    if diff < CONFIG.position_qty_epsilon: return None, None
    risk_amt = usdt_equity * risk_pct; raw_qty = risk_amt / diff; prec_qty_str = format_amount(ex, sym, raw_qty);
    prec_qty = Decimal(prec_qty_str)
    if prec_qty <= CONFIG.position_qty_epsilon: return None, None
    margin = (prec_qty * entry) / Decimal(lev); return prec_qty, margin
def wait_for_order_fill(exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int, order_type: str = "market") -> Optional[Dict[str, Any]]:
    start_time = time.time(); short_order_id = format_order_id(order_id); logger.info(f"{NEON['INFO']}Order Vigil ({order_type}): ...{short_order_id} for '{symbol}' (Timeout: {timeout_seconds}s)...{NEON['RESET']}")
    params = {"category": "linear"} if exchange.options.get("defaultType") == "linear" else {}
    while time.time() - start_time < timeout_seconds:
        try:
            order_details = safe_api_call(exchange.fetch_order, order_id, symbol, params=params); status = order_details.get("status")
            if status == "closed": logger.success(f"{NEON['SUCCESS']}Order Vigil: ...{short_order_id} FILLED/CLOSED.{NEON['RESET']}"); return order_details
            if status in ["canceled", "rejected", "expired"]: logger.error(f"{NEON['ERROR']}Order Vigil: ...{short_order_id} FAILED status: '{status}'.{NEON['RESET']}"); return order_details
            logger.debug(f"Order ...{short_order_id} status: {status}. Vigil continues...")
            time.sleep(1.0 if order_type == "limit" else 0.75)
        except ccxt.OrderNotFound: logger.warning(f"{NEON['WARNING']}Order Vigil: ...{short_order_id} not found. Retrying...{NEON['RESET']}"); time.sleep(1.5)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e_net: logger.warning(f"{NEON['WARNING']}Order Vigil: Network issue ...{short_order_id}: {e_net}. Retrying...{NEON['RESET']}"); time.sleep(3)
        except Exception as e: logger.warning(f"{NEON['WARNING']}Order Vigil: Error ...{short_order_id}: {type(e).__name__}. Retrying...{NEON['RESET']}"); logger.debug(traceback.format_exc()); time.sleep(2)
    logger.error(f"{NEON['ERROR']}Order Vigil: ...{short_order_id} fill TIMED OUT.{NEON['RESET']}")
    try: final_details = safe_api_call(exchange.fetch_order, order_id, symbol, params=params); logger.info(f"Final status for ...{short_order_id} after timeout: {final_details.get('status', 'unknown')}"); return final_details
    except Exception as e_final: logger.error(f"{NEON['ERROR']}Final check for ...{short_order_id} failed: {type(e_final).__name__}{NEON['RESET']}"); return None
def place_risked_order(exchange: ccxt.Exchange, symbol: str, side: str, risk_percentage: Decimal, current_short_atr: Union[Decimal, PandasNAType, None], leverage: int, max_order_cap_usdt: Decimal, margin_check_buffer: Decimal, tsl_percent: Decimal, tsl_activation_offset_percent: Decimal, entry_type: OrderEntryType, is_scale_in: bool = False, existing_position_avg_price: Optional[Decimal] = None) -> Optional[Dict[str, Any]]:
    global _active_trade_parts; action_type = "Scale-In" if is_scale_in else "Initial Entry"; logger.info(f"{NEON['ACTION']}Ritual of {action_type} ({entry_type.value}): {side.upper()} for '{symbol}'...{NEON['RESET']}")
    if pd.isna(current_short_atr) or current_short_atr is None or current_short_atr <= 0: logger.error(f"{NEON['ERROR']}Invalid Short ATR ({_format_for_log(current_short_atr)}) for {action_type}.{NEON['RESET']}"); return None
    v5_api_category = "linear"
    try:
        balance_data = safe_api_call(exchange.fetch_balance, params={"category": v5_api_category}); market_info = exchange.market(symbol); min_qty_allowed = safe_decimal_conversion(market_info.get("limits",{}).get("amount",{}).get("min"), Decimal("0"))
        usdt_equity = safe_decimal_conversion(balance_data.get(CONFIG.usdt_symbol,{}).get("total"), Decimal('NaN')); usdt_free_margin = safe_decimal_conversion(balance_data.get(CONFIG.usdt_symbol,{}).get("free"), Decimal('NaN'))
        if usdt_equity.is_nan() or usdt_equity <= 0: logger.error(f"{NEON['ERROR']}Invalid account equity ({_format_for_log(usdt_equity)}).{NEON['RESET']}"); return None
        ticker = safe_api_call(exchange.fetch_ticker, symbol); signal_price = safe_decimal_conversion(ticker.get("last"), pd.NA)
        if pd.isna(signal_price) or signal_price <= 0: logger.error(f"{NEON['ERROR']}Failed to get valid signal price ({_format_for_log(signal_price)}).{NEON['RESET']}"); return None
        sl_atr_multiplier = get_current_atr_sl_multiplier(); sl_dist = current_short_atr * sl_atr_multiplier; sl_px_est = (signal_price - sl_dist) if side == CONFIG.side_buy else (signal_price + sl_dist)
        if sl_px_est <= 0: logger.error(f"{NEON['ERROR']}Invalid estimated SL price ({_format_for_log(sl_px_est)}).{NEON['RESET']}"); return None
        current_risk_pct = calculate_dynamic_risk() if CONFIG.enable_dynamic_risk else (CONFIG.scale_in_risk_percentage if is_scale_in else CONFIG.risk_per_trade_percentage)
        order_qty, est_margin = calculate_position_size(usdt_equity, current_risk_pct, signal_price, sl_px_est, leverage, symbol, exchange)
        if order_qty is None or order_qty <= CONFIG.position_qty_epsilon: logger.error(f"{NEON['ERROR']}Position size calc failed for {action_type}. Qty: {_format_for_log(order_qty)}{NEON['RESET']}"); return None
        if min_qty_allowed > 0 and order_qty < min_qty_allowed: logger.error(f"{NEON['ERROR']}Qty {_format_for_log(order_qty)} below min allowed {_format_for_log(min_qty_allowed)}.{NEON['RESET']}"); return None
        if usdt_free_margin < est_margin * margin_check_buffer: logger.error(f"{NEON['ERROR']}Insufficient free margin. Need ~{_format_for_log(est_margin*margin_check_buffer,2)}, Have {_format_for_log(usdt_free_margin,2)}{NEON['RESET']}"); return None
        entry_order_id: Optional[str] = None; entry_order_resp: Optional[Dict[str, Any]] = None; limit_price_str: Optional[str] = None
        if entry_type == OrderEntryType.MARKET: entry_order_resp = safe_api_call(exchange.create_market_order, symbol, side, float(order_qty), params={"category": v5_api_category, "positionIdx": 0}); entry_order_id = entry_order_resp.get("id")
        elif entry_type == OrderEntryType.LIMIT:
            pip_value = Decimal('1') / (Decimal('10') ** market_info['precision']['price']); offset = CONFIG.limit_order_offset_pips * pip_value; limit_price = (signal_price - offset) if side == CONFIG.side_buy else (signal_price + offset)
            limit_price_str = format_price(exchange, symbol, limit_price); logger.info(f"Placing LIMIT order: Qty={_format_for_log(order_qty)}, Price={_format_for_log(limit_price_str, color=NEON['PRICE'])}")
            entry_order_resp = safe_api_call(exchange.create_limit_order, symbol, side, float(order_qty), float(limit_price_str), params={"category": v5_api_category, "positionIdx": 0}); entry_order_id = entry_order_resp.get("id")
        if not entry_order_id: logger.critical(f"{NEON['CRITICAL']}{action_type} {entry_type.value} order NO ID!{NEON['RESET']}"); return None
        fill_timeout = CONFIG.limit_order_fill_timeout_seconds if entry_type == OrderEntryType.LIMIT else CONFIG.order_fill_timeout_seconds
        filled_entry_details = wait_for_order_fill(exchange, entry_order_id, symbol, fill_timeout, order_type=entry_type.value)
        if entry_type == OrderEntryType.LIMIT and (not filled_entry_details or filled_entry_details.get("status") != "closed"):
            logger.warning(f"{NEON['WARNING']}Limit order ...{format_order_id(entry_order_id)} did not fill. Cancelling.{NEON['RESET']}")
            try: safe_api_call(exchange.cancel_order, entry_order_id, symbol, params={"category": v5_api_category})
            except Exception as e_cancel: logger.error(f"Failed to cancel limit order ...{format_order_id(entry_order_id)}: {e_cancel}")
            return None
        if not filled_entry_details or filled_entry_details.get("status") != "closed": logger.error(f"{NEON['ERROR']}{action_type} order ...{format_order_id(entry_order_id)} not filled/failed.{NEON['RESET']}"); return None
        actual_fill_px = safe_decimal_conversion(filled_entry_details.get("average")); actual_fill_qty = safe_decimal_conversion(filled_entry_details.get("filled")); entry_ts_ms = filled_entry_details.get("timestamp")
        if actual_fill_qty <= CONFIG.position_qty_epsilon or actual_fill_px <= 0: logger.critical(f"{NEON['CRITICAL']}Invalid fill for {action_type} order. Qty: {_format_for_log(actual_fill_qty)}, Px: {_format_for_log(actual_fill_px)}{NEON['RESET']}"); return None
        entry_ref_price = signal_price if entry_type == OrderEntryType.MARKET else Decimal(limit_price_str) # type: ignore
        slippage = abs(actual_fill_px - entry_ref_price); slippage_pct = (slippage / entry_ref_price * 100) if entry_ref_price > 0 else Decimal(0)
        logger.info(f"{action_type} Slippage: RefPx={_format_for_log(entry_ref_price,4,color=NEON['PARAM'])}, FillPx={_format_for_log(actual_fill_px,4,color=NEON['PRICE'])}, Slip={_format_for_log(slippage,4,color=NEON['WARNING'])} ({slippage_pct:.3f}%)")
        new_part_id = entry_order_id if is_scale_in else "initial"
        if new_part_id == "initial" and any(p["id"] == "initial" for p in _active_trade_parts): logger.error("Attempted second 'initial' part."); return None
        _active_trade_parts.append({"id": new_part_id, "entry_price": actual_fill_px, "entry_time_ms": entry_ts_ms, "side": side, "qty": actual_fill_qty, "sl_price": sl_px_est})
        sl_placed = False; actual_sl_px_raw = (actual_fill_px - sl_dist) if side == CONFIG.side_buy else (actual_fill_px + sl_dist); actual_sl_px_str = format_price(exchange, symbol, actual_sl_px_raw)
        if Decimal(actual_sl_px_str) > 0:
            sl_order_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy; sl_params = {"category": v5_api_category, "stopLossPrice": float(actual_sl_px_str), "reduceOnly": True, "positionIdx": 0, "tpslMode": "Full", "slOrderType": "Market"} # Added tpslMode and slOrderType for Bybit V5
            try: sl_order_resp = safe_api_call(exchange.create_order, symbol, "Market", sl_order_side, float(actual_fill_qty), price=None, params=sl_params); logger.success(f"{NEON['SUCCESS']}SL for part {new_part_id} placed (ID:...{format_order_id(sl_order_resp.get('id'))}).{NEON['RESET']}"); sl_placed = True
            except ccxt.InsufficientFunds as e_funds: logger.error(f"{NEON['CRITICAL']}SL Failed: Insufficient Funds! {e_funds}{NEON['RESET']}")
            except ccxt.InvalidOrder as e_inv: logger.error(f"{NEON['CRITICAL']}SL Failed: Invalid Order! {e_inv}{NEON['RESET']}")
            except Exception as e_sl: logger.error(f"{NEON['CRITICAL']}SL Failed: {e_sl}{NEON['RESET']}"); logger.debug(traceback.format_exc())
        else: logger.error(f"{NEON['CRITICAL']}Invalid SL price for part {new_part_id}! ({_format_for_log(actual_sl_px_str)}){NEON['RESET']}")
        
        if not is_scale_in and CONFIG.trailing_stop_percentage > 0:
            tsl_activation_offset_value = actual_fill_px * CONFIG.trailing_stop_activation_offset_percent
            tsl_activation_price_raw = (actual_fill_px + tsl_activation_offset_value) if side == CONFIG.side_buy else (actual_fill_px - tsl_activation_offset_value)
            tsl_activation_price_str = format_price(exchange, symbol, tsl_activation_price_raw)
            tsl_value_for_api_str = str((CONFIG.trailing_stop_percentage * Decimal("100")).normalize())
            if Decimal(tsl_activation_price_str) > 0:
                tsl_params_specific = {"category": v5_api_category, "trailingStop": tsl_value_for_api_str, "activePrice": float(tsl_activation_price_str), "reduceOnly": True, "positionIdx": 0, "tpslMode": "Full", "slOrderType": "Market"}
                try:
                    tsl_order_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
                    logger.info(f"Placing Trailing SL Ward: Side={tsl_order_side}, Qty={_format_for_log(actual_fill_qty, color=NEON['QTY'])}, Trail={NEON['VALUE']}{tsl_value_for_api_str}%{NEON['RESET']}, ActivateAt={NEON['PRICE']}{tsl_activation_price_str}{NEON['RESET']}")
                    tsl_order_response = safe_api_call(exchange.create_order, symbol, "Market", tsl_order_side, float(actual_fill_qty), price=None, params=tsl_params_specific) # Using Market for TSL execution
                    logger.success(f"{NEON['SUCCESS']}Trailing SL Ward placed. ID:...{format_order_id(tsl_order_response.get('id'))}{NEON['RESET']}")
                except Exception as e_tsl: logger.warning(f"{NEON['WARNING']}Failed to place TSL: {e_tsl}{NEON['RESET']}")
            else: logger.error(f"{NEON['ERROR']}Invalid TSL activation price! ({_format_for_log(tsl_activation_price_str)}){NEON['RESET']}")

        if not sl_placed : logger.critical(f"{NEON['CRITICAL']}CRITICAL: SL FAILED for {action_type} part {new_part_id}. EMERGENCY CLOSE of entire position.{NEON['RESET']}"); close_position(exchange, symbol, {}, reason=f"EMERGENCY CLOSE - SL FAIL ({action_type} part {new_part_id})"); return None
        save_persistent_state(); logger.success(f"{NEON['SUCCESS']}{action_type} for {NEON['QTY']}{actual_fill_qty}{NEON['SUCCESS']} {symbol} @ {NEON['PRICE']}{actual_fill_px}{NEON['SUCCESS']} successful. State saved.{NEON['RESET']}")
        return filled_entry_details
    except ccxt.InsufficientFunds as e_funds: logger.error(f"{NEON['CRITICAL']}{action_type} Failed: Insufficient Funds! {e_funds}{NEON['RESET']}")
    except ccxt.InvalidOrder as e_inv: logger.error(f"{NEON['CRITICAL']}{action_type} Failed: Invalid Order! {e_inv}{NEON['RESET']}")
    except Exception as e_ritual: logger.error(f"{NEON['CRITICAL']}{action_type} Ritual FAILED: {e_ritual}{NEON['RESET']}"); logger.debug(traceback.format_exc())
    return None

def close_partial_position(exchange: ccxt.Exchange, symbol: str, close_qty: Optional[Decimal] = None, reason: str = "Scale Out") -> Optional[Dict[str, Any]]:
    global _active_trade_parts;
    if not _active_trade_parts: logger.info("No active parts to partially close."); return None
    oldest_part = min(_active_trade_parts, key=lambda p: p['entry_time_ms']); qty_to_close = close_qty if close_qty is not None and close_qty > 0 else oldest_part['qty']
    if qty_to_close > oldest_part['qty']: logger.warning(f"Requested partial close qty {close_qty} > oldest part qty {oldest_part['qty']}. Closing oldest part fully."); qty_to_close = oldest_part['qty']
    pos_side = oldest_part['side']; side_to_execute_close = CONFIG.side_sell if pos_side == CONFIG.pos_long else CONFIG.side_buy; amount_to_close_str = format_amount(exchange, symbol, qty_to_close)
    logger.info(f"{NEON['ACTION']}Scaling Out: Closing {NEON['QTY']}{amount_to_close_str}{NEON['ACTION']} of {pos_side} position (Part ID: {oldest_part['id']}, Reason: {reason}).{NEON['RESET']}")
    try:
        params = {"reduceOnly": True, "category": "linear", "positionIdx": 0}; close_order_response = safe_api_call(exchange.create_market_order, symbol=symbol, side=side_to_execute_close, amount=float(amount_to_close_str), params=params)
        if close_order_response and close_order_response.get("status") == "closed":
            exit_price = safe_decimal_conversion(close_order_response.get("average")); exit_time_ms = close_order_response.get("timestamp")
            mae, mfe = trade_metrics.calculate_mae_mfe(oldest_part['id'], oldest_part['entry_price'], exit_price, oldest_part['side'], oldest_part['entry_time_ms'], exit_time_ms, exchange, symbol, CONFIG.interval)
            trade_metrics.log_trade(symbol, oldest_part["side"], oldest_part["entry_price"], exit_price, qty_to_close, oldest_part["entry_time_ms"], exit_time_ms, reason, part_id=oldest_part["id"], mae=mae, mfe=mfe)
            if abs(qty_to_close - oldest_part['qty']) < CONFIG.position_qty_epsilon: _active_trade_parts.remove(oldest_part)
            else: oldest_part['qty'] -= qty_to_close
            save_persistent_state(); logger.success(f"{NEON['SUCCESS']}Scale Out successful for {amount_to_close_str} {symbol}. State saved.{NEON['RESET']}")
            logger.warning(f"{NEON['WARNING']}ACTION REQUIRED: Manually cancel/adjust SL for closed/reduced part ID {oldest_part['id']}.{NEON['RESET']}") # Automation of SL adjustment is complex
            return close_order_response
        else: logger.error(f"{NEON['ERROR']}Scale Out order failed for {symbol}.{NEON['RESET']}")
    except Exception as e: logger.error(f"{NEON['ERROR']}Scale Out Ritual FAILED: {e}{NEON['RESET']}")
    return None
def close_position(exchange: ccxt.Exchange, symbol: str, position_to_close_details: Dict[str, Any], reason: str = "Signal") -> Optional[Dict[str, Any]]:
    global _active_trade_parts;
    if not _active_trade_parts: logger.info("No active parts to close."); return None
    total_qty_to_close = sum(part.get('qty', Decimal(0)) for part in _active_trade_parts)
    if total_qty_to_close <= CONFIG.position_qty_epsilon: _active_trade_parts.clear(); save_persistent_state(); return None
    pos_side_for_log = _active_trade_parts[0]['side']; side_to_execute_close = CONFIG.side_sell if pos_side_for_log == CONFIG.pos_long else CONFIG.side_buy
    logger.info(f"{NEON['ACTION']}Closing ALL parts of {pos_side_for_log} position for {symbol} (Qty: {NEON['QTY']}{total_qty_to_close}{NEON['ACTION']}, Reason: {reason}).{NEON['RESET']}")
    try:
        # Attempt to cancel all open orders (SL/TSL) for the symbol before closing position
        cancelled_sl_count = cancel_open_orders(exchange, symbol, reason=f"Pre-Close Position ({reason})")
        logger.info(f"{NEON['INFO']}Cancelled {NEON['VALUE']}{cancelled_sl_count}{NEON['INFO']} SL/TSL orders before closing position.{NEON['RESET']}")
        time.sleep(0.5) # Brief pause for cancellations to process

        close_order_resp = safe_api_call(exchange.create_market_order, symbol, side_to_execute_close, float(total_qty_to_close), params={"reduceOnly": True, "category": "linear", "positionIdx": 0})
        if close_order_resp and close_order_resp.get("status") == "closed":
            exit_px = safe_decimal_conversion(close_order_resp.get("average")); exit_ts_ms = close_order_resp.get("timestamp")
            for part in list(_active_trade_parts): # Iterate over a copy for safe removal
                 mae, mfe = trade_metrics.calculate_mae_mfe(part['id'],part['entry_price'], exit_px, part['side'], part['entry_time_ms'], exit_ts_ms, exchange, symbol, CONFIG.interval)
                 trade_metrics.log_trade(symbol, part["side"], part["entry_price"], exit_px, part["qty"], part["entry_time_ms"], exit_ts_ms, reason, part_id=part["id"], mae=mae, mfe=mfe)
            _active_trade_parts.clear(); save_persistent_state(); logger.success(f"{NEON['SUCCESS']}All parts of position for {symbol} closed. State saved.{NEON['RESET']}")
            return close_order_resp
        logger.error(f"{NEON['ERROR']}Consolidated close order failed for {symbol}. Response: {close_order_resp}{NEON['RESET']}")
    except Exception as e: logger.error(f"{NEON['ERROR']}Close Position Ritual FAILED: {e}{NEON['RESET']}")
    return None
def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> int:
    logger.info(f"{NEON['INFO']}Attempting to cancel ALL open orders for {NEON['VALUE']}{symbol}{NEON['INFO']} (Reason: {reason})...{NEON['RESET']}")
    cancelled_count = 0
    try:
        # Bybit V5 requires category for fetchOpenOrders and cancelOrder
        params = {"category": "linear"}
        open_orders = safe_api_call(exchange.fetch_open_orders, symbol, params=params)
        if not open_orders:
            logger.info(f"No open orders found for {symbol} to cancel.")
            return 0
        for order in open_orders:
            try:
                safe_api_call(exchange.cancel_order, order['id'], symbol, params=params)
                logger.info(f"Cancelled order {NEON['VALUE']}{order['id']}{NEON['INFO']} for {symbol}.")
                cancelled_count += 1
            except ccxt.OrderNotFound:
                logger.info(f"Order {NEON['VALUE']}{order['id']}{NEON['INFO']} already closed/cancelled.")
                cancelled_count +=1 # Count as handled
            except Exception as e_cancel:
                logger.error(f"{NEON['ERROR']}Failed to cancel order {order['id']}: {e_cancel}{NEON['RESET']}")
        logger.info(f"Order cancellation process for {symbol} complete. Cancelled/Handled: {NEON['VALUE']}{cancelled_count}{NEON['RESET']}.")
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Error fetching/cancelling open orders for {symbol}: {e}{NEON['RESET']}")
    return cancelled_count


# --- Strategy Signal Generation Wrapper ---
def generate_strategy_signals(df_with_indicators: pd.DataFrame, strategy_instance: TradingStrategy) -> Dict[str, Any]:
    if strategy_instance: return strategy_instance.generate_signals(df_with_indicators)
    logger.error(f"{NEON['ERROR']}Strategy instance not initialized!{NEON['RESET']}"); return TradingStrategy(CONFIG)._get_default_signals()

# --- Trading Logic ---
_stop_trading_flag = False
_last_drawdown_check_time = 0
def trade_logic(exchange: ccxt.Exchange, symbol: str, market_data_df: pd.DataFrame) -> None:
    global _active_trade_parts, _stop_trading_flag, _last_drawdown_check_time
    logger.info(f"{NEON['SUBHEADING']}========== Pyrmethus Cycle Start v2.8.0 ({CONFIG.strategy_name.value}) for '{symbol}' =========={NEON['RESET']}")
    now_ts = time.time()
    if _stop_trading_flag: logger.critical(f"{NEON['CRITICAL']}STOP TRADING FLAG ACTIVE (Drawdown?). No new trades.{NEON['RESET']}"); return
    if market_data_df.empty: logger.warning(f"{NEON['WARNING']}Empty market data.{NEON['RESET']}"); return
    if CONFIG.enable_max_drawdown_stop and now_ts - _last_drawdown_check_time > 300: # Check every 5 mins
        try:
            balance = safe_api_call(exchange.fetch_balance, params={"category": "linear"}); current_equity = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol,{}).get("total"))
            if not pd.isna(current_equity):
                trade_metrics.set_initial_equity(current_equity); breached, reason = trade_metrics.check_drawdown(current_equity)
                if breached: _stop_trading_flag = True; logger.critical(f"{NEON['CRITICAL']}MAX DRAWDOWN: {reason}. Halting new trades!{NEON['RESET']}"); send_sms_alert(f"[Pyrmethus] CRITICAL: Max Drawdown STOP Activated: {reason}"); return
            _last_drawdown_check_time = now_ts
        except Exception as e_dd: logger.error(f"{NEON['ERROR']}Error during drawdown check: {e_dd}{NEON['RESET']}")
    
    df_indic, current_vol_atr_data = calculate_all_indicators(market_data_df, CONFIG) # Pass original df
    current_atr = current_vol_atr_data.get("atr_short", Decimal("0")); current_close_price = safe_decimal_conversion(df_indic['close'].iloc[-1], pd.NA)
    if pd.isna(current_atr) or current_atr <= 0 or pd.isna(current_close_price) or current_close_price <= 0: logger.warning(f"{NEON['WARNING']}Invalid ATR ({_format_for_log(current_atr)}) or Close Price ({_format_for_log(current_close_price)}).{NEON['RESET']}"); return
    
    current_pos = get_current_position(exchange, symbol); pos_side, total_pos_qty, avg_pos_entry_price = current_pos["side"], current_pos["qty"], current_pos["entry_price"]; num_active_parts = current_pos.get("num_parts", 0)
    strategy_signals = generate_strategy_signals(df_indic, CONFIG.strategy_instance)
    
    if CONFIG.enable_time_based_stop and pos_side != CONFIG.pos_none:
        now_ms = int(now_ts * 1000)
        for part in list(_active_trade_parts): # Iterate over copy if modifying list
            duration_ms = now_ms - part['entry_time_ms']
            if duration_ms > CONFIG.max_trade_duration_seconds * 1000:
                reason = f"Time Stop Hit ({duration_ms/1000:.0f}s > {CONFIG.max_trade_duration_seconds}s)"; logger.warning(f"{NEON['WARNING']}TIME STOP for part {part['id']} ({pos_side}). Closing entire position.{NEON['RESET']}")
                close_position(exchange, symbol, current_pos, reason=reason); return
    
    if CONFIG.enable_scale_out and pos_side != CONFIG.pos_none and num_active_parts > 0:
        profit_in_atr = Decimal('0')
        if current_atr > 0 and avg_pos_entry_price > 0: price_diff = (current_close_price - avg_pos_entry_price) if pos_side == CONFIG.pos_long else (avg_pos_entry_price - current_close_price); profit_in_atr = price_diff / current_atr
        if profit_in_atr >= CONFIG.scale_out_trigger_atr:
            logger.info(f"{NEON['ACTION']}SCALE-OUT Triggered: {NEON['VALUE']}{profit_in_atr:.2f}{NEON['ACTION']} ATRs in profit. Closing oldest part.{NEON['RESET']}")
            close_partial_position(exchange, symbol, close_qty=None, reason=f"Scale Out Profit Target ({profit_in_atr:.2f} ATR)")
            current_pos = get_current_position(exchange, symbol); pos_side, total_pos_qty, avg_pos_entry_price = current_pos["side"], current_pos["qty"], current_pos["entry_price"]; num_active_parts = current_pos.get("num_parts", 0) # Refresh position state
            if pos_side == CONFIG.pos_none: return # Position fully closed by scale-out
    
    should_exit_long = pos_side == CONFIG.pos_long and strategy_signals.get("exit_long", False)
    should_exit_short = pos_side == CONFIG.pos_short and strategy_signals.get("exit_short", False)
    if should_exit_long or should_exit_short:
        exit_reason = strategy_signals.get("exit_reason", "Oracle Decrees Exit"); logger.warning(f"{NEON['ACTION']}*** STRATEGY EXIT for remaining {pos_side} position (Reason: {exit_reason}) ***{NEON['RESET']}")
        close_position(exchange, symbol, current_pos, reason=exit_reason); return
    
    if CONFIG.enable_position_scaling and pos_side != CONFIG.pos_none and num_active_parts < (CONFIG.max_scale_ins + 1):
        profit_in_atr = Decimal('0')
        if current_atr > 0 and avg_pos_entry_price > 0: price_diff = (current_close_price - avg_pos_entry_price) if pos_side == CONFIG.pos_long else (avg_pos_entry_price - current_close_price); profit_in_atr = price_diff / current_atr
        can_scale = profit_in_atr >= CONFIG.min_profit_for_scale_in_atr
        scale_long_signal = strategy_signals.get("enter_long", False) and pos_side == CONFIG.pos_long
        scale_short_signal = strategy_signals.get("enter_short", False) and pos_side == CONFIG.pos_short
        if can_scale and (scale_long_signal or scale_short_signal):
            logger.success(f"{NEON['ACTION']}*** PYRAMIDING OPPORTUNITY: New signal to add to {pos_side}. ***{NEON['RESET']}")
            scale_in_side = CONFIG.side_buy if scale_long_signal else CONFIG.side_sell
            place_risked_order(exchange=exchange, symbol=symbol, side=scale_in_side, risk_percentage=CONFIG.scale_in_risk_percentage, current_short_atr=current_atr, leverage=CONFIG.leverage, max_order_cap_usdt=CONFIG.max_order_usdt_amount, margin_check_buffer=CONFIG.required_margin_buffer, tsl_percent=CONFIG.trailing_stop_percentage, tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent, entry_type=CONFIG.entry_order_type, is_scale_in=True, existing_position_avg_price=avg_pos_entry_price)
            return
            
    if pos_side == CONFIG.pos_none:
        enter_long_signal = strategy_signals.get("enter_long", False); enter_short_signal = strategy_signals.get("enter_short", False)
        if enter_long_signal or enter_short_signal:
             side_to_enter = CONFIG.side_buy if enter_long_signal else CONFIG.side_sell
             logger.success(f"{(NEON['SIDE_LONG'] if enter_long_signal else NEON['SIDE_SHORT'])}*** INITIAL {side_to_enter.upper()} ENTRY SIGNAL ***{NEON['RESET']}")
             place_risked_order(exchange=exchange, symbol=symbol, side=side_to_enter, risk_percentage=calculate_dynamic_risk(), current_short_atr=current_atr, leverage=CONFIG.leverage, max_order_cap_usdt=CONFIG.max_order_usdt_amount, margin_check_buffer=CONFIG.required_margin_buffer, tsl_percent=CONFIG.trailing_stop_percentage, tsl_activation_offset_percent=CONFIG.trailing_stop_activation_offset_percent, entry_type=CONFIG.entry_order_type, is_scale_in=False)
             return
             
    if pos_side != CONFIG.pos_none: logger.info(f"{NEON['INFO']}Holding {pos_side} position ({NEON['VALUE']}{num_active_parts}{NEON['INFO']} parts). Awaiting signals or stops.{NEON['RESET']}")
    else: logger.info(f"{NEON['INFO']}Holding Cash. No signals or conditions met.{NEON['RESET']}")
    save_persistent_state(force_heartbeat=True)
    logger.info(f"{NEON['SUBHEADING']}========== Pyrmethus Cycle End v2.8.0 for '{symbol}' =========={NEON['RESET']}\n")

# --- Graceful Shutdown ---
def graceful_shutdown(exchange_instance: Optional[ccxt.Exchange], trading_symbol: Optional[str]) -> None:
    logger.warning(f"\n{NEON['WARNING']}Unweaving Sequence Initiated v2.8.0...{NEON['RESET']}")
    save_persistent_state(force_heartbeat=True)
    if exchange_instance and trading_symbol:
        try:
            logger.warning(f"Unweaving: Cancelling ALL open orders for '{trading_symbol}'..."); cancel_open_orders(exchange_instance, trading_symbol, "Bot Shutdown Cleanup"); time.sleep(1.5)
            if _active_trade_parts: logger.warning(f"Unweaving: Active position parts found. Attempting final consolidated close..."); dummy_pos_state = {"side": _active_trade_parts[0]['side'], "qty": sum(p['qty'] for p in _active_trade_parts)}; close_position(exchange_instance, trading_symbol, dummy_pos_state, "Bot Shutdown Final Close")
        except Exception as e_cleanup: logger.error(f"{NEON['ERROR']}Unweaving Error: {e_cleanup}{NEON['RESET']}")
    trade_metrics.summary()
    logger.info(f"{NEON['HEADING']}--- Pyrmethus Spell Unweaving v2.8.0 Complete ---{NEON['RESET']}")

# --- Data Fetching (Added for completeness) ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 150) -> Optional[pd.DataFrame]:
    """Fetches OHLCV market data and returns a pandas DataFrame."""
    logger.info(f"{NEON['INFO']}Fetching market data for {NEON['VALUE']}{symbol}{NEON['INFO']} ({timeframe}, limit={limit})...{NEON['RESET']}")
    try:
        params = {"category": "linear"} # Assuming linear contracts for Bybit V5
        ohlcv = safe_api_call(exchange.fetch_ohlcv, symbol, timeframe, limit=limit, params=params)
        if not ohlcv:
            logger.warning(f"{NEON['WARNING']}No OHLCV data returned for {symbol}.{NEON['RESET']}")
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]: # Ensure numeric types
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Basic NaN handling: ffill then bfill. More sophisticated handling might be needed.
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        if df.isnull().values.any(): # Check if NaNs still exist after filling
            logger.error(f"{NEON['ERROR']}Unfillable NaNs remain in OHLCV data for {symbol} after ffill/bfill. Data quality compromised.{NEON['RESET']}")
            # Depending on strategy, might return None or df with NaNs
            # For safety, returning None if critical columns (like 'close') are still NaN in the last row
            if df[['close']].iloc[-1].isnull().any():
                 return None
        logger.debug(f"{NEON['DEBUG']}Fetched {len(df)} candles for {symbol}. Last candle: {df.index[-1]}{NEON['RESET']}")
        return df
    except Exception as e:
        logger.error(f"{NEON['ERROR']}Failed to fetch market data for {symbol}: {e}{NEON['RESET']}")
        logger.debug(traceback.format_exc())
        return None

# --- Main Execution ---
def main() -> None:
    start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(f"{NEON['HEADING']}--- Pyrmethus Scalping Spell v2.8.0 (Strategic Illumination) Initializing ({start_time_readable}) ---{NEON['RESET']}")
    logger.info(f"{NEON['SUBHEADING']}--- Active Strategy Path: {NEON['VALUE']}{CONFIG.strategy_name.value}{NEON['RESET']} ---")
    logger.info(f"Symbol: {NEON['VALUE']}{CONFIG.symbol}{NEON['RESET']}, Interval: {NEON['VALUE']}{CONFIG.interval}{NEON['RESET']}, Leverage: {NEON['VALUE']}{CONFIG.leverage}x{NEON['RESET']}")
    logger.info(f"Risk/Trade: {NEON['VALUE']}{CONFIG.risk_per_trade_percentage:.2%}{NEON['RESET']}, Max Order USDT: {NEON['VALUE']}{CONFIG.max_order_usdt_amount}{NEON['RESET']}")


    current_exchange_instance: Optional[ccxt.Exchange] = None; unified_trading_symbol: Optional[str] = None; should_run_bot: bool = True
    try:
        current_exchange_instance = initialize_exchange()
        if not current_exchange_instance: logger.critical(f"{NEON['CRITICAL']}Exchange portal failed. Exiting.{NEON['RESET']}"); return
        try: market_details = current_exchange_instance.market(CONFIG.symbol); unified_trading_symbol = market_details["symbol"]
        except Exception as e_market: logger.critical(f"{NEON['CRITICAL']}Symbol validation error: {e_market}. Exiting.{NEON['RESET']}"); return
        logger.info(f"{NEON['SUCCESS']}Spell focused on symbol: {NEON['VALUE']}{unified_trading_symbol}{NEON['SUCCESS']}{NEON['RESET']}") # Added SUCCESS color
        if not set_leverage(current_exchange_instance, unified_trading_symbol, CONFIG.leverage): logger.warning(f"{NEON['WARNING']}Leverage setting may not have been applied or confirmed.{NEON['RESET']}") # Adjusted message
        
        if load_persistent_state():
            logger.info(f"{NEON['SUCCESS']}Phoenix Feather: Previous session state restored.{NEON['RESET']}")
            if _active_trade_parts:
                logger.warning(f"{NEON['WARNING']}State Reconciliation Check:{NEON['RESET']} Bot remembers {len(_active_trade_parts)} active trade part(s). Verifying with exchange...")
                exchange_pos = _get_raw_exchange_position(current_exchange_instance, unified_trading_symbol); bot_qty = sum(p['qty'] for p in _active_trade_parts); bot_side = _active_trade_parts[0]['side'] if _active_trade_parts else CONFIG.pos_none
                if exchange_pos['side'] == CONFIG.pos_none and bot_side != CONFIG.pos_none: logger.critical(f"{NEON['CRITICAL']}RECONCILIATION FAILED:{NEON['RESET']} Bot remembers {bot_side} (Qty: {bot_qty}), exchange FLAT. Clearing bot state."); _active_trade_parts.clear(); save_persistent_state()
                elif exchange_pos['side'] != bot_side or abs(exchange_pos['qty'] - bot_qty) > CONFIG.position_qty_epsilon: logger.critical(f"{NEON['CRITICAL']}RECONCILIATION FAILED:{NEON['RESET']} Discrepancy: Bot ({bot_side} Qty {bot_qty}) vs Exchange ({exchange_pos['side']} Qty {exchange_pos['qty']}). Clearing bot state."); _active_trade_parts.clear(); save_persistent_state()
                else: logger.info(f"{NEON['SUCCESS']}State Reconciliation: Bot state consistent with exchange.{NEON['RESET']}")
        else: logger.info(f"{NEON['INFO']}Starting with a fresh session state.{NEON['RESET']}")
        try: 
            balance = safe_api_call(current_exchange_instance.fetch_balance, params={"category":"linear"}); 
            initial_equity = safe_decimal_conversion(balance.get(CONFIG.usdt_symbol,{}).get("total"))
            if not pd.isna(initial_equity): trade_metrics.set_initial_equity(initial_equity)
            else: logger.error(f"{NEON['ERROR']}Failed to get valid initial equity for drawdown tracking.{NEON['RESET']}")
        except Exception as e_bal: logger.error(f"{NEON['ERROR']}Failed to set initial equity: {e_bal}{NEON['RESET']}")

        while should_run_bot:
            cycle_start_monotonic = time.monotonic()
            try: 
                # Simple health check: Try to fetch balance.
                if not current_exchange_instance.fetch_balance(params={"category":"linear"}): raise Exception("Exchange health check (fetch_balance) failed")
            except Exception as e_health: logger.critical(f"{NEON['CRITICAL']}Account health check failed: {e_health}. Pausing.{NEON['RESET']}"); time.sleep(10); continue
            try:
                df_market_candles = get_market_data(current_exchange_instance, unified_trading_symbol, CONFIG.interval, limit=max(200, CONFIG.momentum_period + CONFIG.confirm_st_atr_length + 50)) # Ensure enough data
                if df_market_candles is not None and not df_market_candles.empty: trade_logic(current_exchange_instance, unified_trading_symbol, df_market_candles)
                else: logger.warning(f"{NEON['WARNING']}Skipping cycle: Invalid market data.{NEON['RESET']}")
            except ccxt.RateLimitExceeded as e_rate: logger.warning(f"{NEON['WARNING']}Rate Limit: {e_rate}. Sleeping longer...{NEON['RESET']}"); time.sleep(CONFIG.sleep_seconds * 6)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e_net: logger.warning(f"{NEON['WARNING']}Network/Exchange Issue: {e_net}. Sleeping...{NEON['RESET']}"); time.sleep(CONFIG.sleep_seconds * 3)
            except ccxt.AuthenticationError as e_auth: logger.critical(f"{NEON['CRITICAL']}FATAL: Auth Error: {e_auth}. Stopping.{NEON['RESET']}"); should_run_bot = False
            except Exception as e_loop: logger.exception(f"{NEON['CRITICAL']}!!! UNEXPECTED CRITICAL ERROR in Main Loop: {e_loop} !!!{NEON['RESET']}"); should_run_bot = False # Use logger.exception to include traceback
            if should_run_bot:
                elapsed = time.monotonic() - cycle_start_monotonic; sleep_dur = max(0, CONFIG.sleep_seconds - elapsed)
                logger.debug(f"Cycle processed in {elapsed:.2f}s. Sleeping for {sleep_dur:.2f}s.")
                if sleep_dur > 0: time.sleep(sleep_dur)
    except KeyboardInterrupt: logger.warning(f"\n{NEON['WARNING']}KeyboardInterrupt. Initiating graceful unweaving...{NEON['RESET']}"); should_run_bot = False
    except Exception as startup_err: logger.critical(f"{NEON['CRITICAL']}CRITICAL STARTUP ERROR v2.8.0: {startup_err}{NEON['RESET']}"); logger.debug(traceback.format_exc()); should_run_bot = False
    finally:
        graceful_shutdown(current_exchange_instance, unified_trading_symbol)
        logger.info(f"{NEON['HEADING']}--- Pyrmethus Scalping Spell v2.8.0 Deactivated ---{NEON['RESET']}")

if __name__ == "__main__":
    main()
```
