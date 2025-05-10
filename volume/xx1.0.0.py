import json
import logging
import math
import os
import signal
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, TypedDict

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:

    class ZoneInfo:  # type: ignore [no-redef]
        def __init__(self, key: str) -> None:
            self._key = "UTC"

        def __call__(self, dt=None):
            return dt.replace(tzinfo=UTC) if dt else None

        def fromutc(self, dt):
            return dt.replace(tzinfo=UTC)

        def utcoffset(self, dt):
            return timedelta(0)

        def dst(self, dt):
            return timedelta(0)

        def tzname(self, dt) -> str:
            return "UTC"

    class ZoneInfoNotFoundError(Exception):
        pass  # type: ignore [no-redef]


import ccxt
import numpy as np
import pandas as pd
import requests
from colorama import Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv

getcontext().prec = 28
colorama_init(autoreset=True)
load_dotenv()
BOT_VERSION = "1.4.0+"
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    sys.exit(1)
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DEFAULT_TIMEZONE_STR = "America/Chicago"
TIMEZONE_STR = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"
except Exception:
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
POSITION_CONFIRM_DELAY_SECONDS = 8
LOOP_DELAY_SECONDS = 15
BYBIT_API_KLINE_LIMIT = 1000
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
}
DEFAULT_FETCH_LIMIT = 750
MAX_DF_LEN = 2000
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 950
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0
DEFAULT_OB_SOURCE = "Wicks"
DEFAULT_PH_LEFT = 10
DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10
DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True
DEFAULT_OB_MAX_BOXES = 50
QUOTE_CURRENCY = "USDT"
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT
DIM = Style.DIM
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError:
    sys.exit(1)
_shutdown_requested = False


class OrderBlock(TypedDict):
    id: str
    type: str
    timestamp: pd.Timestamp
    top: Decimal
    bottom: Decimal
    active: bool
    violated: bool
    violation_ts: pd.Timestamp | None
    extended_to_ts: pd.Timestamp | None


class StrategyAnalysisResults(TypedDict):
    dataframe: pd.DataFrame
    last_close: Decimal
    current_trend_up: bool | None
    trend_just_changed: bool
    active_bull_boxes: list[OrderBlock]
    active_bear_boxes: list[OrderBlock]
    vol_norm_int: int | None
    atr: Decimal | None
    upper_band: Decimal | None
    lower_band: Decimal | None


class MarketInfo(TypedDict):
    id: str
    symbol: str
    base: str
    quote: str
    settle: str | None
    baseId: str
    quoteId: str
    settleId: str | None
    type: str
    spot: bool
    margin: bool
    swap: bool
    future: bool
    option: bool
    active: bool
    contract: bool
    linear: bool | None
    inverse: bool | None
    quanto: bool | None
    taker: float
    maker: float
    contractSize: Any | None
    expiry: int | None
    expiryDatetime: str | None
    strike: float | None
    optionType: str | None
    precision: dict[str, Any]
    limits: dict[str, Any]
    info: dict[str, Any]
    is_contract: bool
    is_linear: bool
    is_inverse: bool
    contract_type_str: str
    min_amount_decimal: Decimal | None
    max_amount_decimal: Decimal | None
    min_cost_decimal: Decimal | None
    max_cost_decimal: Decimal | None
    amount_precision_step_decimal: Decimal | None
    price_precision_step_decimal: Decimal | None
    contract_size_decimal: Decimal


class PositionInfo(TypedDict):
    id: str | None
    symbol: str
    timestamp: int | None
    datetime: str | None
    contracts: float | None
    contractSize: Any | None
    side: str | None
    notional: Any | None
    leverage: Any | None
    unrealizedPnl: Any | None
    realizedPnl: Any | None
    collateral: Any | None
    entryPrice: Any | None
    markPrice: Any | None
    liquidationPrice: Any | None
    marginMode: str | None
    hedged: bool | None
    maintenanceMargin: Any | None
    maintenanceMarginPercentage: float | None
    initialMargin: Any | None
    initialMarginPercentage: float | None
    marginRatio: float | None
    lastUpdateTimestamp: int | None
    info: dict[str, Any]
    size_decimal: Decimal
    stopLossPrice: str | None
    takeProfitPrice: str | None
    trailingStopLoss: str | None
    tslActivationPrice: str | None
    be_activated: bool
    tsl_activated: bool


class SignalResult(TypedDict):
    signal: str
    reason: str
    initial_sl: Decimal | None
    initial_tp: Decimal | None


class SensitiveFormatter(logging.Formatter):
    _api_key_placeholder = "***BYBIT_API_KEY***"
    _api_secret_placeholder = "***BYBIT_API_SECRET***"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        key = API_KEY
        secret = API_SECRET
        try:
            if key and isinstance(key, str) and key in msg:
                msg = msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in msg:
                msg = msg.replace(secret, self._api_secret_placeholder)
        except Exception:
            pass
        return msg


def setup_logger(name: str) -> logging.Logger:
    safe_name = name.replace("/", "_").replace(":", "-")
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)
    try:
        fh = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        ff = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ff.converter = time.gmtime  # type: ignore
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    except Exception:
        pass
    try:
        sh = logging.StreamHandler(sys.stdout)
        level_colors = {
            logging.DEBUG: NEON_CYAN + DIM,
            logging.INFO: NEON_BLUE,
            logging.WARNING: NEON_YELLOW,
            logging.ERROR: NEON_RED,
            logging.CRITICAL: NEON_RED + BRIGHT,
        }

        class NeonConsoleFormatter(SensitiveFormatter):
            _level_colors = level_colors
            _tz = TIMEZONE

            def format(self, record: logging.LogRecord) -> str:
                level_color = self._level_colors.get(record.levelno, NEON_BLUE)
                log_fmt = f"{NEON_BLUE}%(asctime)s{RESET} - {level_color}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
                formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
                formatter.converter = lambda *args: datetime.now(self._tz).timetuple()  # type: ignore
                return super().format(record)

        sh.setFormatter(NeonConsoleFormatter())
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception:
        pass
    logger.propagate = False
    return logger


init_logger = setup_logger("init")
init_logger.info(
    f"{Fore.MAGENTA}{BRIGHT}Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing...{Style.RESET_ALL}"
)
init_logger.info(f"Using Timezone: {TIMEZONE_STR}")


def _ensure_config_keys(
    config: dict[str, Any], default_config: dict[str, Any], parent_key: str = ""
) -> tuple[dict[str, Any], bool]:
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            updated_config[key] = default_value
            changed = True
            init_logger.info(
                f"{NEON_YELLOW}Config Update: Added '{full_key_path}' = {repr(default_value)}{RESET}"
            )
        elif isinstance(default_value, dict) and isinstance(
            updated_config.get(key), dict
        ):
            nested_config, nested_changed = _ensure_config_keys(
                updated_config[key], default_value, full_key_path
            )
            if nested_changed:
                updated_config[key] = nested_config
                changed = True
    return updated_config, changed


def load_config(filepath: str) -> dict[str, Any]:
    init_logger.info(
        f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}"
    )
    default_config = {
        "trading_pairs": ["BTC/USDT"],
        "interval": "5",
        "retry_delay": RETRY_DELAY_SECONDS,
        "fetch_limit": DEFAULT_FETCH_LIMIT,
        "orderbook_limit": 25,
        "enable_trading": False,
        "use_sandbox": True,
        "risk_per_trade": 0.01,
        "leverage": 20,
        "max_concurrent_positions": 1,
        "quote_currency": "USDT",
        "loop_delay_seconds": LOOP_DELAY_SECONDS,
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER),
            "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER),
            "ob_source": DEFAULT_OB_SOURCE,
            "ph_left": DEFAULT_PH_LEFT,
            "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT,
            "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005,
            "ob_exit_proximity_factor": 1.001,
        },
        "protection": {
            "enable_trailing_stop": True,
            "trailing_stop_callback_rate": 0.005,
            "trailing_stop_activation_percentage": 0.003,
            "enable_break_even": True,
            "break_even_trigger_atr_multiple": 1.0,
            "break_even_offset_ticks": 2,
            "initial_stop_loss_atr_multiple": 1.8,
            "initial_take_profit_atr_multiple": 0.7,
        },
    }
    config_needs_saving: bool = False
    loaded_config: dict[str, Any] = {}
    if not os.path.exists(filepath):
        init_logger.warning(
            f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default.{RESET}"
        )
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Created default config: {filepath}{RESET}")
            global QUOTE_CURRENCY
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
        except OSError as e:
            init_logger.critical(
                f"{NEON_RED}FATAL: Error creating config '{filepath}': {e}{RESET}"
            )
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
    try:
        with open(filepath, encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
            raise TypeError("Config is not a JSON object.")
    except json.JSONDecodeError as e:
        init_logger.error(
            f"{NEON_RED}Error decoding JSON from '{filepath}': {e}. Recreating default.{RESET}"
        )
        try:
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Recreated default config: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
        except OSError as e_create:
            init_logger.critical(
                f"{NEON_RED}FATAL: Error recreating config: {e_create}. Using internal defaults.{RESET}"
            )
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
    except Exception as e:
        init_logger.critical(
            f"{NEON_RED}FATAL: Error loading config '{filepath}': {e}{RESET}",
            exc_info=True,
        )
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        return default_config
    try:
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True

        def validate_numeric(
            cfg: dict,
            key_path: str,
            min_val,
            max_val,
            is_strict_min=False,
            is_int=False,
            allow_zero=False,
        ) -> bool:
            nonlocal config_needs_saving
            keys = key_path.split(".")
            current_level = cfg
            default_level = default_config
            try:
                for key in keys[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                leaf_key = keys[-1]
                original_val = current_level.get(leaf_key)
                default_val = default_level.get(leaf_key)
            except (KeyError, TypeError):
                init_logger.error(
                    f"Config validation error: Invalid path '{key_path}'."
                )
                return False
            if original_val is None:
                init_logger.warning(
                    f"Config validation: Value missing at '{key_path}'. Using default: {repr(default_val)}"
                )
                current_level[leaf_key] = default_val
                config_needs_saving = True
                return True
            corrected = False
            final_val = original_val
            try:
                num_val = Decimal(str(original_val))
                min_dec = Decimal(str(min_val))
                max_dec = Decimal(str(max_val))
                min_check = num_val > min_dec if is_strict_min else num_val >= min_dec
                range_check = min_check and num_val <= max_dec
                zero_ok = allow_zero and num_val.is_zero()
                if not range_check and not zero_ok:
                    raise ValueError("Value outside allowed range.")
                target_type = int if is_int else float
                converted_val = target_type(num_val)
                needs_correction = False
                if isinstance(original_val, bool):
                    raise TypeError("Boolean found where numeric expected.")
                elif is_int and not isinstance(original_val, int):
                    needs_correction = True
                elif not is_int and not isinstance(original_val, float):
                    needs_correction = (
                        True
                        if not isinstance(original_val, int)
                        else bool(converted_val := float(original_val)) or True
                    )
                elif (
                    isinstance(original_val, float)
                    and abs(original_val - converted_val) > 1e-9
                    or isinstance(original_val, int)
                    and original_val != converted_val
                ):
                    needs_correction = True
                if needs_correction:
                    init_logger.info(
                        f"{NEON_YELLOW}Config Update: Corrected '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}"
                    )
                    final_val = converted_val
                    corrected = True
            except (ValueError, InvalidOperation, TypeError) as e:
                range_str = (
                    f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                    + (" or 0" if allow_zero else "")
                )
                init_logger.warning(
                    f"{NEON_YELLOW}Config Validation: Invalid '{key_path}'='{repr(original_val)}'. Using default: {repr(default_val)}. Err: {e}. Expected: {'int' if is_int else 'float'}, Range: {range_str}{RESET}"
                )
                final_val = default_val
                corrected = True
            if corrected:
                current_level[leaf_key] = final_val
                config_needs_saving = True
            return corrected

        init_logger.debug("# Validating configuration parameters...")
        if not isinstance(updated_config.get("trading_pairs"), list) or not all(
            isinstance(s, str) and s and "/" in s
            for s in updated_config.get("trading_pairs", [])
        ):
            init_logger.warning(
                f"Invalid 'trading_pairs'. Using default {default_config['trading_pairs']}."
            )
            updated_config["trading_pairs"] = default_config["trading_pairs"]
            config_needs_saving = True
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.warning(
                f"Invalid 'interval'. Using default '{default_config['interval']}'."
            )
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True)
        validate_numeric(
            updated_config,
            "risk_per_trade",
            Decimal("0"),
            Decimal("0.5"),
            is_strict_min=True,
        )
        validate_numeric(
            updated_config, "leverage", 0, 200, is_int=True, allow_zero=True
        )
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(
            updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True
        )
        if not isinstance(
            updated_config.get("quote_currency"), str
        ) or not updated_config.get("quote_currency"):
            init_logger.warning(
                f"Invalid 'quote_currency'. Using default '{default_config['quote_currency']}'."
            )
            updated_config["quote_currency"] = default_config["quote_currency"]
            config_needs_saving = True
        if not isinstance(updated_config.get("enable_trading"), bool):
            init_logger.warning(
                f"Invalid 'enable_trading'. Using default '{default_config['enable_trading']}'."
            )
            updated_config["enable_trading"] = default_config["enable_trading"]
            config_needs_saving = True
        if not isinstance(updated_config.get("use_sandbox"), bool):
            init_logger.warning(
                f"Invalid 'use_sandbox'. Using default '{default_config['use_sandbox']}'."
            )
            updated_config["use_sandbox"] = default_config["use_sandbox"]
            config_needs_saving = True
        validate_numeric(
            updated_config, "strategy_params.vt_length", 1, 1000, is_int=True
        )
        validate_numeric(
            updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True
        )
        validate_numeric(
            updated_config,
            "strategy_params.vt_vol_ema_length",
            1,
            MAX_DF_LEN,
            is_int=True,
        )
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(
            updated_config, "strategy_params.ph_right", 1, 100, is_int=True
        )
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(
            updated_config, "strategy_params.pl_right", 1, 100, is_int=True
        )
        validate_numeric(
            updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True
        )
        validate_numeric(
            updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1
        )
        validate_numeric(
            updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1
        )
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
            init_logger.warning(
                f"Invalid ob_source. Using default '{DEFAULT_OB_SOURCE}'."
            )
            updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE
            config_needs_saving = True
        if not isinstance(updated_config["strategy_params"].get("ob_extend"), bool):
            init_logger.warning(
                f"Invalid ob_extend. Using default '{DEFAULT_OB_EXTEND}'."
            )
            updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND
            config_needs_saving = True
        if not isinstance(
            updated_config["protection"].get("enable_trailing_stop"), bool
        ):
            init_logger.warning("Invalid enable_trailing_stop. Using default.")
            updated_config["protection"]["enable_trailing_stop"] = default_config[
                "protection"
            ]["enable_trailing_stop"]
            config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_break_even"), bool):
            init_logger.warning("Invalid enable_break_even. Using default.")
            updated_config["protection"]["enable_break_even"] = default_config[
                "protection"
            ]["enable_break_even"]
            config_needs_saving = True
        validate_numeric(
            updated_config,
            "protection.trailing_stop_callback_rate",
            Decimal("0.0001"),
            Decimal("0.1"),
            is_strict_min=True,
        )
        validate_numeric(
            updated_config,
            "protection.trailing_stop_activation_percentage",
            Decimal("0"),
            Decimal("0.1"),
            allow_zero=True,
        )
        validate_numeric(
            updated_config,
            "protection.break_even_trigger_atr_multiple",
            Decimal("0.1"),
            Decimal("10.0"),
        )
        validate_numeric(
            updated_config,
            "protection.break_even_offset_ticks",
            0,
            1000,
            is_int=True,
            allow_zero=True,
        )
        validate_numeric(
            updated_config,
            "protection.initial_stop_loss_atr_multiple",
            Decimal("0.1"),
            Decimal("20.0"),
            is_strict_min=True,
        )
        validate_numeric(
            updated_config,
            "protection.initial_take_profit_atr_multiple",
            Decimal("0"),
            Decimal("20.0"),
            allow_zero=True,
        )
        if config_needs_saving:
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                init_logger.info(
                    f"{NEON_GREEN}Config file '{filepath}' updated.{RESET}"
                )
            except Exception as save_err:
                init_logger.error(
                    f"{NEON_RED}Error saving updated config to '{filepath}': {save_err}{RESET}",
                    exc_info=True,
                )
        global QUOTE_CURRENCY
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.info(f"Quote currency set: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        init_logger.info(
            f"{Fore.CYAN}# Configuration loading/validation complete.{Style.RESET_ALL}"
        )
        return updated_config
    except Exception as e:
        init_logger.critical(
            f"{NEON_RED}FATAL: Error processing config: {e}. Using defaults.{RESET}",
            exc_info=True,
        )
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        return default_config


CONFIG = load_config(CONFIG_FILE)


def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    lg = logger
    lg.info(f"{Fore.CYAN}# Initializing Bybit exchange connection...{Style.RESET_ALL}")
    try:
        exchange_options = {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",
                "adjustForTimeDifference": True,
                "fetchTickerTimeout": 15000,
                "fetchBalanceTimeout": 20000,
                "createOrderTimeout": 30000,
                "cancelOrderTimeout": 20000,
                "fetchPositionsTimeout": 20000,
                "fetchOHLCVTimeout": 60000,
            },
        }
        exchange = ccxt.bybit(exchange_options)
        is_sandbox = CONFIG.get("use_sandbox", True)
        exchange.set_sandbox_mode(is_sandbox)
        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}<<< SANDBOX MODE ACTIVE >>>{RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< LIVE TRADING ACTIVE >>> !!!{RESET}")
        lg.info(f"Loading market data for {exchange.id}...")
        markets_loaded = False
        last_market_error = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market load attempt {attempt + 1}...")
                exchange.load_markets(reload=(attempt > 0))
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(
                        f"{NEON_GREEN}Market data loaded ({len(exchange.markets)} symbols).{RESET}"
                    )
                    markets_loaded = True
                    break
                else:
                    last_market_error = ValueError("Market data empty.")
                    lg.warning(
                        f"Market data empty (Attempt {attempt + 1}). Retrying..."
                    )
            except (
                ccxt.NetworkError,
                ccxt.RequestTimeout,
                requests.exceptions.RequestException,
            ) as e:
                last_market_error = e
                lg.warning(
                    f"Network error loading markets (Attempt {attempt + 1}): {e}."
                )
            except ccxt.AuthenticationError as e:
                last_market_error = e
                lg.critical(
                    f"{NEON_RED}Auth error loading markets: {e}. Exiting.{RESET}"
                )
                return None
            except Exception as e:
                last_market_error = e
                lg.critical(
                    f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}",
                    exc_info=True,
                )
                return None
            if not markets_loaded and attempt < MAX_API_RETRIES:
                delay = RETRY_DELAY_SECONDS * (attempt + 1)
                lg.warning(f"Retrying market load in {delay}s...")
                time.sleep(delay)
        if not markets_loaded:
            lg.critical(
                f"{NEON_RED}Failed to load markets. Last error: {last_market_error}. Exiting.{RESET}"
            )
            return None
        lg.info(f"Exchange initialized: {exchange.id} | Sandbox: {is_sandbox}")
        lg.info(f"Checking initial balance ({QUOTE_CURRENCY})...")
        initial_balance: Decimal | None = None
        try:
            initial_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        except ccxt.AuthenticationError as auth_err:
            lg.critical(
                f"{NEON_RED}Auth error during balance check: {auth_err}. Exiting.{RESET}"
            )
            return None
        except Exception as balance_err:
            lg.warning(
                f"{NEON_YELLOW}Initial balance check error: {balance_err}.{RESET}",
                exc_info=False,
            )
        if initial_balance is not None:
            lg.info(
                f"{NEON_GREEN}Initial balance: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}"
            )
            lg.info(f"{Fore.CYAN}# Exchange init complete.{Style.RESET_ALL}")
            return exchange
        else:
            lg.error(
                f"{NEON_RED}Initial balance check FAILED ({QUOTE_CURRENCY}).{RESET}"
            )
            if CONFIG.get("enable_trading", False):
                lg.critical(
                    f"{NEON_RED}Trading enabled, but balance check failed. Exiting.{RESET}"
                )
                return None
            else:
                lg.warning(
                    f"{NEON_YELLOW}Trading disabled. Proceeding without balance check.{RESET}"
                )
                lg.info(
                    f"{Fore.CYAN}# Exchange init complete (no balance confirmation).{Style.RESET_ALL}"
                )
                return exchange
    except Exception as e:
        lg.critical(f"{NEON_RED}Exchange init failed: {e}{RESET}", exc_info=True)
        return None


def _safe_market_decimal(
    value: Any | None,
    field_name: str,
    allow_zero: bool = True,
    allow_negative: bool = False,
) -> Decimal | None:
    if value is None:
        return None
    try:
        s_val = str(value).strip()
        if not s_val:
            return None
        d_val = Decimal(s_val)
        if not allow_zero and d_val.is_zero():
            return None
        if not allow_negative and d_val < Decimal("0"):
            return None
        return d_val
    except (InvalidOperation, TypeError, ValueError):
        return None


def _format_price(
    exchange: ccxt.Exchange, symbol: str, price: Decimal | float | str
) -> str | None:
    try:
        price_decimal = Decimal(str(price))
        if price_decimal <= Decimal("0"):
            return None
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))
        return formatted_str if Decimal(formatted_str) > Decimal("0") else None
    except Exception as e:
        init_logger.warning(f"Error formatting price '{price}' for {symbol}: {e}")
        return None


def fetch_current_price_ccxt(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger
) -> Decimal | None:
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching price ({symbol}, Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price: Decimal | None = None
            source = "N/A"

            def safe_decimal_from_ticker(val: Any | None, name: str) -> Decimal | None:
                return _safe_market_decimal(
                    val, f"ticker.{name}", allow_zero=False, allow_negative=False
                )

            price = safe_decimal_from_ticker(ticker.get("last"), "last")
            source = "'last'" if price else source
            if price is None:
                bid = safe_decimal_from_ticker(ticker.get("bid"), "bid")
                ask = safe_decimal_from_ticker(ticker.get("ask"), "ask")
                if bid and ask:
                    price = (bid + ask) / Decimal("2")
                    source = f"mid (B:{bid.normalize()}, A:{ask.normalize()})"
                elif ask:
                    price = ask
                    source = f"'ask' ({ask.normalize()})"
                elif bid:
                    price = bid
                    source = f"'bid' ({bid.normalize()})"
            if price:
                normalized_price = price.normalize()
                lg.debug(f"Price ({symbol}) from {source}: {normalized_price}")
                return normalized_price
            else:
                last_exception = ValueError("No valid price source in ticker.")
                lg.warning(
                    f"No valid price ({symbol}, Attempt {attempts + 1}). Ticker: {ticker}. Retrying..."
                )
        except (
            ccxt.NetworkError,
            ccxt.RequestTimeout,
            requests.exceptions.RequestException,
        ) as e:
            last_exception = e
            lg.warning(
                f"{NEON_YELLOW}Net error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}"
            )
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait = RETRY_DELAY_SECONDS * 3
            lg.warning(
                f"{NEON_YELLOW}Rate limit price ({symbol}): {e}. Wait {wait}s...{RESET}"
            )
            time.sleep(wait)
            continue
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Auth error price: {e}. Stop.{RESET}")
            return None
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exch error price ({symbol}): {e}{RESET}")
        except Exception as e:
            last_exception = e
            lg.error(
                f"{NEON_RED}Unexpected error price ({symbol}): {e}{RESET}",
                exc_info=True,
            )
            return None
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(
        f"{NEON_RED}Failed fetch price ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last: {last_exception}{RESET}"
    )
    return None


def fetch_klines_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    lg = logger
    lg.info(
        f"{Fore.CYAN}# Fetching klines for {symbol} | TF: {timeframe} | Target: {limit}...{Style.RESET_ALL}"
    )
    if not hasattr(exchange, "fetch_ohlcv") or not exchange.has.get("fetchOHLCV"):
        lg.error(f"Exchange {exchange.id} lacks fetchOHLCV.")
        return pd.DataFrame()
    min_required = 0
    try:
        sp = CONFIG.get("strategy_params", {})
        min_required = (
            max(
                sp.get("vt_length", 0) * 2,
                sp.get("vt_atr_period", 0),
                sp.get("vt_vol_ema_length", 0),
                sp.get("ph_left", 0) + sp.get("ph_right", 0) + 1,
                sp.get("pl_left", 0) + sp.get("pl_right", 0) + 1,
            )
            + 50
        )
        lg.debug(f"Min candles needed: ~{min_required}")
    except Exception as e:
        lg.warning(f"Could not estimate min candles: {e}")
    if limit < min_required:
        lg.warning(
            f"{NEON_YELLOW}Req limit ({limit}) < est strategy need ({min_required}). Accuracy risk.{RESET}"
        )
    category = "spot"
    market_id = symbol
    try:
        market = exchange.market(symbol)
        market_id = market["id"]
        category = (
            "linear"
            if market.get("linear")
            else "inverse"
            if market.get("inverse")
            else "spot"
        )
        lg.debug(f"Using category '{category}', market ID '{market_id}'.")
    except Exception as e:
        lg.warning(f"Could not get market category/ID ({symbol}): {e}.")
    all_ohlcv_data: list[list] = []
    remaining_limit = limit
    end_timestamp_ms: int | None = None
    max_chunks = math.ceil(limit / BYBIT_API_KLINE_LIMIT) + 2
    chunk_num = 0
    total_fetched = 0
    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1
        fetch_size = min(remaining_limit, BYBIT_API_KLINE_LIMIT)
        lg.debug(
            f"Fetching chunk {chunk_num}/{max_chunks} ({fetch_size} candles) for {symbol}. End TS: {end_timestamp_ms}"
        )
        attempts = 0
        last_exception = None
        chunk_data = None
        while attempts <= MAX_API_RETRIES:
            try:
                params = (
                    {"category": category} if "bybit" in exchange.id.lower() else {}
                )
                fetch_args: dict[str, Any] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": fetch_size,
                    "params": params,
                }
                if end_timestamp_ms:
                    fetch_args["until"] = end_timestamp_ms
                chunk_data = exchange.fetch_ohlcv(**fetch_args)
                fetched_count_chunk = len(chunk_data) if chunk_data else 0
                lg.debug(f"API returned {fetched_count_chunk} for chunk {chunk_num}.")
                if chunk_data:
                    if chunk_num == 1:
                        try:
                            last_ts = pd.to_datetime(
                                chunk_data[-1][0], unit="ms", utc=True
                            )
                            interval_seconds = exchange.parse_timeframe(timeframe)
                            if interval_seconds:
                                max_lag = interval_seconds * 2.5
                                actual_lag = (
                                    pd.Timestamp.utcnow() - last_ts
                                ).total_seconds()
                                if actual_lag > max_lag:
                                    last_exception = ValueError(
                                        f"Stale data? Lag {actual_lag:.1f}s > Max {max_lag:.1f}s"
                                    )
                                    lg.warning(
                                        f"{NEON_YELLOW}Lag detected ({symbol}): {last_exception}. Retrying...{RESET}"
                                    )
                                    chunk_data = None
                                else:
                                    break
                            else:
                                break
                        except Exception as ts_err:
                            lg.warning(
                                f"Lag check failed ({symbol}): {ts_err}. Proceeding."
                            )
                            break
                    else:
                        break
                else:
                    lg.debug(
                        f"API returned no data for chunk {chunk_num}. End of history?"
                    )
                    remaining_limit = 0
                    break
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                last_exception = e
                lg.warning(
                    f"{NEON_YELLOW}Net error klines chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}"
                )
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                wait = RETRY_DELAY_SECONDS * 3
                lg.warning(
                    f"{NEON_YELLOW}Rate limit klines chunk {chunk_num} ({symbol}): {e}. Wait {wait}s...{RESET}"
                )
                time.sleep(wait)
                continue
            except ccxt.AuthenticationError as e:
                last_exception = e
                lg.critical(f"{NEON_RED}Auth error klines: {e}. Stop.{RESET}")
                return pd.DataFrame()
            except ccxt.ExchangeError as e:
                last_exception = e
                lg.error(
                    f"{NEON_RED}Exch error klines chunk {chunk_num} ({symbol}): {e}{RESET}"
                )
                err_str = str(e).lower()
                if (
                    "invalid timeframe" in err_str
                    or "interval not supported" in err_str
                    or "symbol invalid" in err_str
                ):
                    lg.critical(
                        f"{NEON_RED}Non-retryable kline error: {e}. Stop.{RESET}"
                    )
                    return pd.DataFrame()
            except Exception as e:
                last_exception = e
                lg.error(
                    f"{NEON_RED}Unexpected error klines chunk {chunk_num} ({symbol}): {e}{RESET}",
                    exc_info=True,
                )
                return pd.DataFrame()
            attempts += 1
            if attempts <= MAX_API_RETRIES and chunk_data is None:
                time.sleep(RETRY_DELAY_SECONDS * attempts)
        if chunk_data:
            all_ohlcv_data = chunk_data + all_ohlcv_data
            chunk_len = len(chunk_data)
            remaining_limit -= chunk_len
            total_fetched += chunk_len
            end_timestamp_ms = chunk_data[0][0] - 1
            if chunk_len < fetch_size:
                lg.debug(
                    "Received fewer candles than requested. Assuming end of history."
                )
                remaining_limit = 0
        else:
            lg.error(
                f"{NEON_RED}Failed fetch kline chunk {chunk_num} ({symbol}) after retries. Last: {last_exception}{RESET}"
            )
            if not all_ohlcv_data:
                lg.error(f"Failed first chunk ({symbol}). Cannot proceed.")
                return pd.DataFrame()
            else:
                lg.warning(
                    f"Proceeding with {total_fetched} candles fetched before error."
                )
                break
        if remaining_limit > 0:
            time.sleep(0.5)
    if chunk_num >= max_chunks and remaining_limit > 0:
        lg.warning(f"Stopped fetching klines ({symbol}) at max chunks ({max_chunks}).")
    if not all_ohlcv_data:
        lg.error(f"No kline data fetched ({symbol} {timeframe}).")
        return pd.DataFrame()
    lg.info(f"Total klines fetched across requests: {len(all_ohlcv_data)}")
    seen_timestamps = set()
    unique_data = []
    for candle in reversed(all_ohlcv_data):
        if candle[0] not in seen_timestamps:
            unique_data.append(candle)
            seen_timestamps.add(candle[0])
    if len(unique_data) != len(all_ohlcv_data):
        lg.warning(
            f"Removed {len(all_ohlcv_data) - len(unique_data)} duplicate candles ({symbol})."
        )
    unique_data.sort(key=lambda x: x[0])
    if len(unique_data) > limit:
        lg.debug(f"Fetched {len(unique_data)}, trimming to {limit}.")
        unique_data = unique_data[-limit:]
    try:
        lg.debug(f"Processing {len(unique_data)} final candles ({symbol})...")
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(unique_data, columns=cols[: len(unique_data[0])])
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True, errors="coerce"
        )
        df.dropna(subset=["timestamp"], inplace=True)
        if df.empty:
            lg.error(f"DF empty after timestamp conv ({symbol}).")
            return pd.DataFrame()
        df.set_index("timestamp", inplace=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                df[col] = numeric_series.apply(
                    lambda x: Decimal(str(x))
                    if pd.notna(x) and np.isfinite(x)
                    else Decimal("NaN")
                )
            else:
                lg.warning(f"Missing col '{col}' ({symbol}).")
        initial_len = len(df)
        essential = ["open", "high", "low", "close"]
        df.dropna(subset=essential, inplace=True)
        df = df[df["close"] > Decimal("0")]
        if "volume" in df.columns:
            df.dropna(subset=["volume"], inplace=True)
            df = df[df["volume"] >= Decimal("0")]
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows ({symbol}) during cleaning.")
        if df.empty:
            lg.warning(f"DF empty after cleaning ({symbol}).")
            return pd.DataFrame()
        if not df.index.is_monotonic_increasing:
            lg.warning(f"Index not monotonic ({symbol}), sorting...")
            df.sort_index(inplace=True)
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DF length {len(df)} > max {MAX_DF_LEN}. Trimming ({symbol}).")
            df = df.iloc[-MAX_DF_LEN:].copy()
        lg.info(f"{NEON_GREEN}Processed {len(df)} klines ({symbol} {timeframe}){RESET}")
        return df
    except Exception as e:
        lg.error(
            f"{NEON_RED}Error processing klines ({symbol}): {e}{RESET}", exc_info=True
        )
        return pd.DataFrame()


# ... (Rest of the functions and classes: get_market_info, fetch_balance, get_open_position, set_leverage_ccxt, calculate_position_size, cancel_order, place_trade, _set_position_protection, set_trailing_stop_loss, VolumaticOBStrategy, SignalGenerator, analyze_and_trade_symbol, _handle_shutdown_signal, main)
# Placeholder for the rest of the functions and classes (from the previous enhanced version)
# For a runnable script, paste the full function/class definitions here, then minify them.
# Example placeholders (replace with actual minified code):
def get_market_info(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger
) -> MarketInfo | None:
    return None


# def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]: print("fetch_balance placeholder"); return None
# def get_open_position(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, logger: logging.Logger) -> Optional[PositionInfo]: print("get_open_position placeholder"); return None
# def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool: print("set_leverage_ccxt placeholder"); return False
# def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal, market_info: MarketInfo, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]: print("calculate_position_size placeholder"); return None
# def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool: print("cancel_order placeholder"); return False
# def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: MarketInfo, logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]: print("place_trade placeholder"); return None
# def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, logger: logging.Logger, stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None, trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool: print("_set_position_protection placeholder"); return False
# def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, config: Dict[str, Any], logger: logging.Logger, take_profit_price: Optional[Decimal] = None) -> bool: print("set_trailing_stop_loss placeholder"); return False
# class VolumaticOBStrategy: def __init__(self, config, market_info, logger): print("VolumaticOBStrategy init placeholder"); self.market_info=market_info; self.min_data_len=100; self.lg=logger; def update(self, df): print("Strategy update placeholder"); return StrategyAnalysisResults(dataframe=df, last_close=Decimal('1'), current_trend_up=True, trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=50, atr=Decimal('1'), upper_band=Decimal('2'), lower_band=Decimal('0'))
# class SignalGenerator: def __init__(self, config, market_info, logger): print("SignalGenerator init placeholder"); self.lg=logger; def generate_signal(self, analysis, pos, sym): print("Signal generate placeholder"); return SignalResult(signal="HOLD", reason="Placeholder", initial_sl=None, initial_tp=None)
# def analyze_and_trade_symbol(exchange, symbol, config, logger, strategy_engine, signal_generator, market_info): print(f"analyze_and_trade_symbol placeholder for {symbol}")
# def manage_existing_position(exchange, symbol, market_info, position_info, analysis_results, position_state, logger): print(f"manage_existing_position placeholder for {symbol}")
# def execute_trade_action(exchange, symbol, market_info, current_position, signal_info, analysis_results, logger): print(f"execute_trade_action placeholder for {symbol}")
def _handle_shutdown_signal(signum, frame) -> None:
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    init_logger.warning(
        f"\n{NEON_RED}{BRIGHT}Shutdown signal ({signal_name}) received! Exiting...{RESET}"
    )
    _shutdown_requested = True


def main() -> None:
    global CONFIG, _shutdown_requested
    main_logger = setup_logger("main")
    main_logger.info(
        f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Bot v{BOT_VERSION} Starting ---{Style.RESET_ALL}"
    )
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    exchange = initialize_exchange(main_logger)
    if not exchange:
        main_logger.critical("Exchange init failed. Shutting down.")
        sys.exit(1)
    trading_pairs = CONFIG.get("trading_pairs", [])
    valid_pairs: list[str] = []
    all_valid = True
    main_logger.info(f"Validating trading pairs: {trading_pairs}")
    market_infos: dict[str, MarketInfo] = {}
    strategy_engines: dict[str, VolumaticOBStrategy] = {}
    signal_generators: dict[str, SignalGenerator] = {}
    for pair in trading_pairs:
        market_info = get_market_info(exchange, pair, main_logger)
        if market_info and market_info.get("active"):
            valid_pairs.append(pair)
            market_infos[pair] = market_info
            main_logger.info(f" -> {NEON_GREEN}{pair} valid.{RESET}")
            try:
                strategy_engines[pair] = VolumaticOBStrategy(
                    CONFIG, market_info, setup_logger(pair)
                )
                signal_generators[pair] = SignalGenerator(
                    CONFIG, market_info, setup_logger(pair)
                )
            except ValueError as init_err:
                main_logger.error(
                    f" -> {NEON_RED}Failed init for {pair}: {init_err}. Skipping.{RESET}"
                )
                all_valid = False
                valid_pairs.remove(pair)
                market_infos.pop(pair, None)
        else:
            main_logger.error(
                f" -> {NEON_RED}{pair} invalid/inactive. Skipping.{RESET}"
            )
            all_valid = False
    if not valid_pairs:
        main_logger.critical("No valid pairs. Shutting down.")
        sys.exit(1)
    if not all_valid:
        main_logger.warning(f"Proceeding with valid pairs: {valid_pairs}")
    if not CONFIG.get("enable_trading", False):
        main_logger.warning(f"{NEON_YELLOW}--- TRADING DISABLED ---{RESET}")
    main_logger.info(f"{Fore.CYAN}### Starting Main Loop ###{Style.RESET_ALL}")
    loop_count = 0
    position_states: dict[str, dict[str, bool]] = {
        sym: {"be_activated": False, "tsl_activated": False} for sym in valid_pairs
    }
    while not _shutdown_requested:
        loop_count += 1
        main_logger.debug(f"--- Loop Cycle #{loop_count} ---")
        start_time = time.monotonic()
        for symbol in valid_pairs:
            if _shutdown_requested:
                break
            symbol_logger = get_logger_for_symbol(symbol)
            symbol_logger.info(f"--- Processing: {symbol} (Cycle #{loop_count}) ---")
            try:
                market_info = market_infos[symbol]  # Use cached info for loop speed
                analyze_and_trade_symbol(
                    exchange,
                    symbol,
                    CONFIG,
                    symbol_logger,
                    strategy_engines[symbol],
                    signal_generators[symbol],
                    market_info,
                    position_states,
                )
            except ccxt.AuthenticationError as e:
                symbol_logger.critical(
                    f"{NEON_RED}Auth Error ({symbol}): {e}. Stopping bot.{RESET}"
                )
                _shutdown_requested = True
                break
            except Exception as symbol_err:
                symbol_logger.error(
                    f"{NEON_RED}!! Unhandled error ({symbol}): {symbol_err} !!{RESET}",
                    exc_info=True,
                )
            finally:
                symbol_logger.info(f"--- Finished: {symbol} ---")
            if _shutdown_requested:
                break
            time.sleep(0.2)  # Small delay between symbols
        if _shutdown_requested:
            break
        end_time = time.monotonic()
        cycle_dur = end_time - start_time
        loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
        wait_time = max(0, loop_delay - cycle_dur)
        main_logger.info(
            f"Cycle {loop_count} duration: {cycle_dur:.2f}s. Waiting {wait_time:.2f}s..."
        )
        for _ in range(int(wait_time)):
            if _shutdown_requested:
                break
            time.sleep(1)
        if not _shutdown_requested and wait_time % 1 > 0:
            time.sleep(wait_time % 1)
    main_logger.info(
        f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Bot Shutting Down ---{Style.RESET_ALL}"
    )
    main_logger.info("Shutdown complete.")
    sys.exit(0)


def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: dict[str, Any],
    logger: logging.Logger,
    strategy_engine: "VolumaticOBStrategy",
    signal_generator: "SignalGenerator",
    market_info: MarketInfo,
    position_states: dict[str, dict[str, bool]],
) -> None:
    lg = logger
    lg.debug(f"analyze_and_trade_symbol started for {symbol}")
    timeframe_key = config.get("interval", "5")
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe_key, "5m")
    fetch_limit = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    df_raw = fetch_klines_ccxt(exchange, symbol, ccxt_timeframe, fetch_limit, lg)
    if df_raw.empty:
        lg.warning(f"Kline data empty for {symbol}. Skipping cycle.")
        return
    try:
        analysis = strategy_engine.update(df_raw)
    except Exception as analysis_err:
        lg.error(f"Strategy analysis failed ({symbol}): {analysis_err}", exc_info=True)
        return
    if not analysis or analysis["dataframe"].empty or analysis["last_close"].is_nan():
        lg.warning(f"Invalid strategy analysis ({symbol}). Skipping cycle.")
        return
    current_position = get_open_position(exchange, symbol, market_info, lg)
    symbol_state = position_states.setdefault(
        symbol, {"be_activated": False, "tsl_activated": False}
    )
    if current_position:
        symbol_state["tsl_activated"] = bool(
            current_position.get("trailingStopLoss")
        )  # Update TSL state from API
        manage_existing_position(
            exchange, symbol, market_info, current_position, analysis, symbol_state, lg
        )
    else:
        if symbol_state["be_activated"] or symbol_state["tsl_activated"]:
            lg.debug(f"Resetting BE/TSL state for {symbol} (no position).")
        position_states[symbol] = {"be_activated": False, "tsl_activated": False}
    signal_info = signal_generator.generate_signal(analysis, current_position, symbol)
    execute_trade_action(
        exchange,
        symbol,
        market_info,
        current_position,
        signal_info,
        analysis,
        symbol_state,
        lg,
    )  # Pass state to execution


def manage_existing_position(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    position_info: PositionInfo,
    analysis_results: StrategyAnalysisResults,
    position_state: dict[str, bool],
    logger: logging.Logger,
) -> None:
    lg = logger
    lg.debug(f"Managing existing {position_info['side']} position for {symbol}...")
    pos_side = position_info["side"]
    entry_price_any = position_info.get("entryPrice")
    atr = analysis_results["atr"]
    last_close = analysis_results["last_close"]
    be_activated = position_state["be_activated"]
    position_state["tsl_activated"]
    tsl_active_on_exchange = bool(
        position_info.get("trailingStopLoss")
    )  # Check current API state
    if (
        not isinstance(entry_price_any, (Decimal, int, float))
        or Decimal(str(entry_price_any)) <= 0
    ):
        lg.warning(f"Pos Mgmt ({symbol}): Invalid entry price ({entry_price_any}).")
        return
    entry_price = Decimal(str(entry_price_any))
    if not isinstance(atr, Decimal) or not atr.is_finite() or atr <= 0:
        lg.warning(f"Pos Mgmt ({symbol}): Invalid ATR ({atr}). Skipping BE/TSL.")
        return
    if (
        not isinstance(last_close, Decimal)
        or not last_close.is_finite()
        or last_close <= 0
    ):
        lg.warning(
            f"Pos Mgmt ({symbol}): Invalid close price ({last_close}). Skipping BE/TSL."
        )
        return
    protection_cfg = CONFIG.get("protection", {})
    price_tick = market_info["price_precision_step_decimal"]
    if price_tick is None or price_tick <= 0:
        lg.warning(f"Pos Mgmt ({symbol}): Invalid price tick. Cannot manage BE.")
        return
    enable_be = protection_cfg.get("enable_break_even", False) and not be_activated
    if enable_be and not tsl_active_on_exchange:
        be_trigger_mult = Decimal(
            str(protection_cfg.get("break_even_trigger_atr_multiple", 1.0))
        )
        be_offset_ticks = int(protection_cfg.get("break_even_offset_ticks", 2))
        profit_distance = abs(last_close - entry_price)
        profit_in_atr = profit_distance / atr if atr > 0 else Decimal(0)
        lg.debug(
            f"  BE Check ({symbol}): ProfitATR={profit_in_atr.normalize()} vs Trigger={be_trigger_mult.normalize()}, Offset={be_offset_ticks} ticks"
        )
        if profit_in_atr >= be_trigger_mult:
            lg.info(
                f"{NEON_YELLOW}BE Triggered ({symbol}): Profit ATR >= Trigger.{RESET}"
            )
            offset_amount = price_tick * be_offset_ticks
            be_sl_price = (
                entry_price + offset_amount
                if pos_side == "long"
                else entry_price - offset_amount
            )
            be_sl_price = quantize_price(be_sl_price, price_tick, pos_side, is_tp=False)
            valid_be_sl = (
                pos_side == "long" and be_sl_price > entry_price - price_tick
            ) or (pos_side == "short" and be_sl_price < entry_price + price_tick)
            if valid_be_sl and be_sl_price > 0:
                current_sl_str = position_info.get("stopLossPrice")
                current_sl = (
                    _safe_market_decimal(current_sl_str, "current_sl", False)
                    if current_sl_str
                    else None
                )
                needs_update = (
                    current_sl is None
                    or (pos_side == "long" and be_sl_price > current_sl)
                    or (pos_side == "short" and be_sl_price < current_sl)
                )
                if needs_update:
                    lg.warning(
                        f"{BRIGHT}---> Setting BE SL for {symbol} at {be_sl_price.normalize()} (Old SL: {current_sl.normalize() if current_sl else 'None'}) <---{RESET}"
                    )
                    if CONFIG.get("enable_trading", False):
                        current_tp_str = position_info.get("takeProfitPrice")
                        current_tp_dec = (
                            Decimal(current_tp_str)
                            if current_tp_str
                            and _safe_market_decimal(current_tp_str, "current_tp")
                            else None
                        )
                        success = _set_position_protection(
                            exchange,
                            symbol,
                            market_info,
                            position_info,
                            lg,
                            stop_loss_price=be_sl_price,
                            take_profit_price=current_tp_dec,
                            trailing_stop_distance=Decimal("0"),
                            tsl_activation_price=Decimal("0"),
                        )
                        if success:
                            position_state["be_activated"] = True
                            lg.info(f"{NEON_GREEN}BE SL set ({symbol}).{RESET}")
                        else:
                            lg.error(
                                f"{NEON_RED}Failed to set BE SL ({symbol})!{RESET}"
                            )
                    else:
                        lg.warning(f"Trading disabled: Would set BE SL ({symbol}).")
                else:
                    lg.info(
                        f"BE ({symbol}): Current SL ({current_sl.normalize() if current_sl else 'N/A'}) already better than BE ({be_sl_price.normalize()}). No update."
                    )
            else:
                lg.error(
                    f"BE triggered but calc BE SL invalid ({symbol}): {be_sl_price.normalize()}."
                )
    enable_tsl = (
        protection_cfg.get("enable_trailing_stop", False)
        and not tsl_active_on_exchange
        and not be_activated
    )  # Don't activate TSL if BE already activated by bot
    if enable_tsl:
        tsl_act_perc = Decimal(
            str(protection_cfg.get("trailing_stop_activation_percentage", 0.003))
        )
        tsl_cb_rate = Decimal(
            str(protection_cfg.get("trailing_stop_callback_rate", 0.005))
        )
        profit_perc = (
            (last_close - entry_price) / entry_price
            if pos_side == "long"
            else (entry_price - last_close) / entry_price
        )
        lg.debug(
            f"  TSL Check ({symbol}): Profit%={profit_perc:.4%} vs Activation%={tsl_act_perc:.4%}"
        )
        if profit_perc >= tsl_act_perc:
            lg.info(
                f"{NEON_YELLOW}TSL Activation Triggered ({symbol}): Profit >= Activation%.{RESET}"
            )
            activation_price = last_close  # Use current price to activate
            tsl_distance = activation_price * tsl_cb_rate
            tsl_distance = max(tsl_distance, price_tick)
            q_tsl_dist = quantize_price(tsl_distance, price_tick, pos_side, is_tp=False)
            q_act_price = quantize_price(
                activation_price, price_tick, pos_side, is_tp=True
            )  # Activation price should be 'good' side
            if q_tsl_dist > 0 and q_act_price > 0:
                lg.warning(
                    f"{BRIGHT}---> Activating TSL for {symbol} | Dist: {q_tsl_dist.normalize()}, ActPrice: {q_act_price.normalize()} <---{RESET}"
                )
                if CONFIG.get("enable_trading", False):
                    current_tp_str = position_info.get("takeProfitPrice")
                    current_tp_dec = (
                        Decimal(current_tp_str)
                        if current_tp_str
                        and _safe_market_decimal(current_tp_str, "current_tp")
                        else None
                    )
                    success = _set_position_protection(
                        exchange,
                        symbol,
                        market_info,
                        position_info,
                        lg,
                        trailing_stop_distance=q_tsl_dist,
                        tsl_activation_price=q_act_price,
                        take_profit_price=current_tp_dec,
                        stop_loss_price=None,
                    )
                    if success:
                        position_state["tsl_activated"] = True
                        lg.info(f"{NEON_GREEN}TSL activated ({symbol}).{RESET}")
                    else:
                        lg.error(f"{NEON_RED}Failed to activate TSL ({symbol})!{RESET}")
                else:
                    lg.warning(f"Trading disabled: Would activate TSL ({symbol}).")
            else:
                lg.error(
                    f"TSL calculation invalid ({symbol}): Dist={q_tsl_dist}, Act={q_act_price}."
                )


def execute_trade_action(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: MarketInfo,
    current_position: PositionInfo | None,
    signal_info: SignalResult,
    analysis_results: StrategyAnalysisResults,
    position_state: dict[str, bool],
    logger: logging.Logger,
) -> None:
    lg = logger
    signal = signal_info["signal"]
    trading_enabled = CONFIG.get("enable_trading", False)
    if not trading_enabled:
        if signal != "HOLD":
            lg.warning(
                f"{NEON_YELLOW}TRADING DISABLED:{RESET} Signal '{signal}' ({symbol}) not executed. Reason: {signal_info['reason']}"
            )
        return
    if signal in ["EXIT_LONG", "EXIT_SHORT"]:
        if not current_position:
            lg.warning(
                f"Received {signal} signal ({symbol}), but no position found. Ignoring."
            )
            return
        pos_side = current_position["side"]
        pos_size_dec = current_position["size_decimal"]
        if (signal == "EXIT_LONG" and pos_side != "long") or (
            signal == "EXIT_SHORT" and pos_side != "short"
        ):
            lg.error(
                f"Signal mismatch: {signal} vs {pos_side} pos ({symbol}). Ignoring."
            )
            return
        exit_size = abs(pos_size_dec)
        if exit_size <= market_info.get(
            "amount_precision_step_decimal", Decimal("1e-9")
        ):
            lg.error(
                f"Cannot {signal} ({symbol}): Pos size near zero ({pos_size_dec})."
            )
            return
        lg.warning(
            f"{BRIGHT}>>> Executing {signal} for {symbol} <<< Reason: {signal_info['reason']}"
        )
        lg.info(f"  Closing {pos_side} size: {exit_size.normalize()}")
        order_result = place_trade(
            exchange, symbol, signal, exit_size, market_info, lg, reduce_only=True
        )
        if order_result:
            lg.info(
                f"Exit order placed ({symbol}). ID: {order_result.get('id', 'N/A')}"
            )
            position_state["be_activated"] = False
            position_state["tsl_activated"] = False  # Reset state on exit attempt
        else:
            lg.error(f"{NEON_RED}Failed to place {signal} order ({symbol}).{RESET}")
    elif signal in ["BUY", "SELL"]:
        if current_position:
            lg.warning(
                f"Received {signal} ({symbol}), but position exists. Ignoring entry."
            )
            return
        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None or balance <= 0:
            lg.error(f"Cannot place {signal} ({symbol}): Invalid balance {balance}.")
            return
        initial_sl = signal_info["initial_sl"]
        initial_tp = signal_info["initial_tp"]
        entry_price = analysis_results["last_close"]
        if not initial_sl:
            lg.error(f"Cannot place {signal} ({symbol}): Missing initial SL.")
            return
        position_size = calculate_position_size(
            balance,
            CONFIG["risk_per_trade"],
            initial_sl,
            entry_price,
            market_info,
            exchange,
            lg,
        )
        if position_size is None or position_size <= 0:
            lg.error(f"Cannot place {signal} ({symbol}): Position sizing failed.")
            return
        leverage = CONFIG.get("leverage", 0)
        if market_info.get("is_contract") and leverage > 0:
            if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                lg.warning(
                    f"Leverage set failed ({symbol}). Continuing entry attempt..."
                )
        lg.warning(
            f"{BRIGHT}>>> Initiating {signal} Entry for {symbol} | Size: {position_size.normalize()} <<< Reason: {signal_info['reason']}"
        )
        lg.info(
            f"  Initial SL: {initial_sl.normalize()}, Initial TP: {initial_tp.normalize() if initial_tp else 'Disabled'}"
        )
        order_result = place_trade(
            exchange, symbol, signal, position_size, market_info, lg, reduce_only=False
        )
        if order_result:
            lg.info(
                f"Entry order placed ({symbol}). ID: {order_result.get('id', 'N/A')}"
            )
            confirm_delay = CONFIG.get(
                "position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS
            )
            lg.debug(f"Waiting {confirm_delay}s to confirm {symbol} position...")
            time.sleep(confirm_delay)
            new_position = None
            for attempt in range(2):
                temp_pos = get_open_position(exchange, symbol, market_info, lg)
                if temp_pos and temp_pos.get("side") == (
                    "long" if signal == "BUY" else "short"
                ):
                    new_position = temp_pos
                    break
                if attempt == 0:
                    time.sleep(3)
            if new_position:
                lg.info(
                    f"{NEON_GREEN}Confirmed {new_position['side']} position opened ({symbol}). Size: {new_position['size_decimal'].normalize()}, Entry: {new_position.get('entryPrice', 'N/A')}{RESET}"
                )
                lg.info(
                    f"Setting initial protection for {symbol}: SL={initial_sl.normalize()}, TP={initial_tp.normalize() if initial_tp else 'None'}"
                )
                success = _set_position_protection(
                    exchange,
                    symbol,
                    market_info,
                    new_position,
                    lg,
                    stop_loss_price=initial_sl,
                    take_profit_price=initial_tp,
                )
                if success:
                    lg.info(f"Initial SL/TP set successfully ({symbol}).")
                    position_state["be_activated"] = False
                    position_state["tsl_activated"] = False  # Reset state for new pos
                else:
                    lg.error(
                        f"{NEON_RED}Failed to set initial SL/TP for {symbol}!{RESET}"
                    )
            else:
                lg.error(
                    f"{NEON_RED}Failed to confirm open position ({symbol}) after entry order {order_result.get('id', 'N/A')}. Manual check required!{RESET}"
                )
        else:
            lg.error(f"{NEON_RED}Failed {signal} entry order ({symbol}).{RESET}")
    elif signal == "HOLD":
        lg.info(f"Signal ({symbol}): HOLD. Reason: {signal_info['reason']}")
    else:
        lg.error(f"Unknown signal '{signal}' ({symbol}). Ignoring.")


def _handle_shutdown_signal(signum, frame) -> None:
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    init_logger.warning(
        f"\n{NEON_RED}{BRIGHT}Shutdown signal ({signal_name}) received! Requesting graceful exit...{RESET}"
    )
    _shutdown_requested = True


def main() -> None:
    global CONFIG, _shutdown_requested
    main_logger = setup_logger("main")
    main_logger.info(
        f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Volumatic Bot v{BOT_VERSION} Starting ---{Style.RESET_ALL}"
    )
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    exchange = initialize_exchange(main_logger)
    if not exchange:
        main_logger.critical("Exchange init failed. Shutting down.")
        sys.exit(1)
    trading_pairs = CONFIG.get("trading_pairs", [])
    valid_pairs: list[str] = []
    all_valid = True
    main_logger.info(f"Validating trading pairs: {trading_pairs}")
    market_infos: dict[str, MarketInfo] = {}
    strategy_engines: dict[str, VolumaticOBStrategy] = {}
    signal_generators: dict[str, SignalGenerator] = {}
    for pair in trading_pairs:
        market_info = get_market_info(exchange, pair, main_logger)
        if market_info and market_info.get("active"):
            valid_pairs.append(pair)
            market_infos[pair] = market_info
            main_logger.info(f" -> {NEON_GREEN}{pair} valid.{RESET}")
            try:
                strategy_engines[pair] = VolumaticOBStrategy(
                    CONFIG, market_info, setup_logger(pair)
                )
                signal_generators[pair] = SignalGenerator(
                    CONFIG, market_info, setup_logger(pair)
                )
            except ValueError as init_err:
                main_logger.error(
                    f" -> {NEON_RED}Failed init for {pair}: {init_err}. Skipping.{RESET}"
                )
                all_valid = False
                valid_pairs.remove(pair)
                market_infos.pop(pair, None)
        else:
            main_logger.error(
                f" -> {NEON_RED}{pair} invalid/inactive. Skipping.{RESET}"
            )
            all_valid = False
    if not valid_pairs:
        main_logger.critical("No valid pairs. Shutting down.")
        sys.exit(1)
    if not all_valid:
        main_logger.warning(f"Proceeding with valid pairs: {valid_pairs}")
    if not CONFIG.get("enable_trading", False):
        main_logger.warning(f"{NEON_YELLOW}--- TRADING DISABLED ---{RESET}")
    main_logger.info(f"{Fore.CYAN}### Starting Main Loop ###{Style.RESET_ALL}")
    loop_count = 0
    position_states: dict[str, dict[str, bool]] = {
        sym: {"be_activated": False, "tsl_activated": False} for sym in valid_pairs
    }
    while not _shutdown_requested:
        loop_count += 1
        main_logger.debug(f"--- Loop Cycle #{loop_count} ---")
        start_time = time.monotonic()
        for symbol in valid_pairs:
            if _shutdown_requested:
                break
            symbol_logger = get_logger_for_symbol(symbol)
            symbol_logger.info(f"--- Processing: {symbol} (Cycle #{loop_count}) ---")
            try:
                market_info = market_infos[symbol]  # Use cached info
                analyze_and_trade_symbol(
                    exchange,
                    symbol,
                    CONFIG,
                    symbol_logger,
                    strategy_engines[symbol],
                    signal_generators[symbol],
                    market_info,
                    position_states,
                )
            except ccxt.AuthenticationError as e:
                symbol_logger.critical(
                    f"{NEON_RED}Auth Error ({symbol}): {e}. Stopping bot.{RESET}"
                )
                _shutdown_requested = True
                break
            except Exception as symbol_err:
                symbol_logger.error(
                    f"{NEON_RED}!! Unhandled error ({symbol}): {symbol_err} !!{RESET}",
                    exc_info=True,
                )
            finally:
                symbol_logger.info(f"--- Finished: {symbol} ---")
            if _shutdown_requested:
                break
            time.sleep(0.2)  # Short delay between symbols
        if _shutdown_requested:
            break
        end_time = time.monotonic()
        cycle_dur = end_time - start_time
        loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
        wait_time = max(0, loop_delay - cycle_dur)
        main_logger.info(
            f"Cycle {loop_count} duration: {cycle_dur:.2f}s. Waiting {wait_time:.2f}s..."
        )
        for _ in range(int(wait_time)):
            if _shutdown_requested:
                break
            time.sleep(1)
        if not _shutdown_requested and wait_time % 1 > 0:
            time.sleep(wait_time % 1)
    main_logger.info(
        f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Bot Shutting Down ---{Style.RESET_ALL}"
    )
    logging.shutdown()  # Flush and close handlers
    main_logger.info("Shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        init_logger.info("KeyboardInterrupt caught in __main__. Exiting.")
        sys.exit(0)
    except Exception as global_err:
        init_logger.critical(
            f"{NEON_RED}{BRIGHT}FATAL UNHANDLED EXCEPTION:{RESET} {global_err}",
            exc_info=True,
        )
        sys.exit(1)
