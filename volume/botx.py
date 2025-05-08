# pyrmethus_volumatic_bot.py - Merged & Minified v1.4.1 (Fixed global scope error)
import hashlib, hmac, json, logging, math, os, sys, time, signal, re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
try: from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    print("Warning: 'zoneinfo' module not found. Falling back to UTC. Ensure Python 3.9+ and install 'tzdata'.")
    class ZoneInfo: # type: ignore [no-redef]
        def __init__(self, key: str): self._key = "UTC"
        def __call__(self, dt=None): return dt.replace(tzinfo=timezone.utc) if dt else None
        def fromutc(self, dt): return dt.replace(tzinfo=timezone.utc)
        def utcoffset(self, dt): return timedelta(0)
        def dst(self, dt): return timedelta(0)
        def tzname(self, dt): return "UTC"
    class ZoneInfoNotFoundError(Exception): pass # type: ignore [no-redef]
import numpy as np, pandas as pd, pandas_ta as ta, requests, ccxt
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
getcontext().prec = 28; colorama_init(autoreset=True); load_dotenv()
BOT_VERSION = "1.4.1"
API_KEY = os.getenv("BYBIT_API_KEY"); API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET: print(f"{Fore.RED}{Style.BRIGHT}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET missing in .env. Exiting.{Style.RESET_ALL}"); sys.exit(1)
CONFIG_FILE = "config.json"; LOG_DIRECTORY = "bot_logs"; DEFAULT_TIMEZONE_STR = "America/Chicago"
TIMEZONE_STR = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try: TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError: print(f"{Fore.RED}Timezone '{TIMEZONE_STR}' not found. Using UTC.{Style.RESET_ALL}"); TIMEZONE = ZoneInfo("UTC"); TIMEZONE_STR = "UTC"
except Exception as tz_err: print(f"{Fore.RED}Timezone init error for '{TIMEZONE_STR}': {tz_err}. Using UTC.{Style.RESET_ALL}"); TIMEZONE = ZoneInfo("UTC"); TIMEZONE_STR = "UTC"
MAX_API_RETRIES = 3; RETRY_DELAY_SECONDS = 5; POSITION_CONFIRM_DELAY_SECONDS = 8; LOOP_DELAY_SECONDS = 15; BYBIT_API_KLINE_LIMIT = 1000
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {"1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"}
DEFAULT_FETCH_LIMIT = 750; MAX_DF_LEN = 2000
DEFAULT_VT_LENGTH = 40; DEFAULT_VT_ATR_PERIOD = 200; DEFAULT_VT_VOL_EMA_LENGTH = 950; DEFAULT_VT_ATR_MULTIPLIER = 3.0; DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0
DEFAULT_OB_SOURCE = "Wicks"; DEFAULT_PH_LEFT = 10; DEFAULT_PH_RIGHT = 10; DEFAULT_PL_LEFT = 10; DEFAULT_PL_RIGHT = 10; DEFAULT_OB_EXTEND = True; DEFAULT_OB_MAX_BOXES = 50
QUOTE_CURRENCY = "USDT"
NEON_GREEN = Fore.LIGHTGREEN_EX; NEON_BLUE = Fore.CYAN; NEON_PURPLE = Fore.MAGENTA; NEON_YELLOW = Fore.YELLOW; NEON_RED = Fore.LIGHTRED_EX; NEON_CYAN = Fore.CYAN; RESET = Style.RESET_ALL; BRIGHT = Style.BRIGHT; DIM = Style.DIM
try: os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError as e: print(f"{NEON_RED}{BRIGHT}FATAL: Could not create log directory '{LOG_DIRECTORY}': {e}. Exiting.{RESET}"); sys.exit(1)
_shutdown_requested = False
class OrderBlock(TypedDict): id: str; type: str; timestamp: pd.Timestamp; top: Decimal; bottom: Decimal; active: bool; violated: bool; violation_ts: Optional[pd.Timestamp]; extended_to_ts: Optional[pd.Timestamp]
class StrategyAnalysisResults(TypedDict): dataframe: pd.DataFrame; last_close: Decimal; current_trend_up: Optional[bool]; trend_just_changed: bool; active_bull_boxes: List[OrderBlock]; active_bear_boxes: List[OrderBlock]; vol_norm_int: Optional[int]; atr: Optional[Decimal]; upper_band: Optional[Decimal]; lower_band: Optional[Decimal]
class MarketInfo(TypedDict): id: str; symbol: str; base: str; quote: str; settle: Optional[str]; baseId: str; quoteId: str; settleId: Optional[str]; type: str; spot: bool; margin: bool; swap: bool; future: bool; option: bool; active: bool; contract: bool; linear: Optional[bool]; inverse: Optional[bool]; quanto: Optional[bool]; taker: float; maker: float; contractSize: Optional[Any]; expiry: Optional[int]; expiryDatetime: Optional[str]; strike: Optional[float]; optionType: Optional[str]; precision: Dict[str, Any]; limits: Dict[str, Any]; info: Dict[str, Any]; is_contract: bool; is_linear: bool; is_inverse: bool; contract_type_str: str; min_amount_decimal: Optional[Decimal]; max_amount_decimal: Optional[Decimal]; min_cost_decimal: Optional[Decimal]; max_cost_decimal: Optional[Decimal]; amount_precision_step_decimal: Optional[Decimal]; price_precision_step_decimal: Optional[Decimal]; contract_size_decimal: Decimal
class PositionInfo(TypedDict): id: Optional[str]; symbol: str; timestamp: Optional[int]; datetime: Optional[str]; contracts: Optional[float]; contractSize: Optional[Any]; side: Optional[str]; notional: Optional[Any]; leverage: Optional[Any]; unrealizedPnl: Optional[Any]; realizedPnl: Optional[Any]; collateral: Optional[Any]; entryPrice: Optional[Any]; markPrice: Optional[Any]; liquidationPrice: Optional[Any]; marginMode: Optional[str]; hedged: Optional[bool]; maintenanceMargin: Optional[Any]; maintenanceMarginPercentage: Optional[float]; initialMargin: Optional[Any]; initialMarginPercentage: Optional[float]; marginRatio: Optional[float]; lastUpdateTimestamp: Optional[int]; info: Dict[str, Any]; size_decimal: Decimal; stopLossPrice: Optional[str]; takeProfitPrice: Optional[str]; trailingStopLoss: Optional[str]; tslActivationPrice: Optional[str]; be_activated: bool; tsl_activated: bool
class SignalResult(TypedDict): signal: str; reason: str; initial_sl: Optional[Decimal]; initial_tp: Optional[Decimal]
class SensitiveFormatter(logging.Formatter):
    _api_key_placeholder = "***BYBIT_API_KEY***"; _api_secret_placeholder = "***BYBIT_API_SECRET***"
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record); key = API_KEY; secret = API_SECRET
        try:
            if key and isinstance(key, str) and key in msg: msg = msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in msg: msg = msg.replace(secret, self._api_secret_placeholder)
        except Exception as e: print(f"WARNING: Error during log message redaction: {e}", file=sys.stderr)
        return msg
def setup_logger(name: str) -> logging.Logger:
    safe_name = name.replace('/', '_').replace(':', '-'); logger_name = f"pyrmethus_bot_{safe_name}"; log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name);
    if logger.hasHandlers(): return logger
    logger.setLevel(logging.DEBUG)
    try:
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        ff = SensitiveFormatter("%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s", datefmt='%Y-%m-%d %H:%M:%S'); ff.converter = time.gmtime # type: ignore
        fh.setFormatter(ff); fh.setLevel(logging.DEBUG); logger.addHandler(fh)
    except Exception as e: print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")
    try:
        sh = logging.StreamHandler(sys.stdout)
        level_colors = {logging.DEBUG: NEON_CYAN + DIM, logging.INFO: NEON_BLUE, logging.WARNING: NEON_YELLOW, logging.ERROR: NEON_RED, logging.CRITICAL: NEON_RED + BRIGHT}
        class NeonConsoleFormatter(SensitiveFormatter):
            _level_colors = level_colors; _tz = TIMEZONE
            def format(self, record: logging.LogRecord) -> str:
                level_color = self._level_colors.get(record.levelno, NEON_BLUE)
                log_fmt = f"{NEON_BLUE}%(asctime)s{RESET} - {level_color}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
                formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S'); formatter.converter = lambda *args: datetime.now(self._tz).timetuple() # type: ignore
                return super().format(record)
        sh.setFormatter(NeonConsoleFormatter())
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper(); log_level = getattr(logging, log_level_str, logging.INFO)
        sh.setLevel(log_level); logger.addHandler(sh)
    except Exception as e: print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")
    logger.propagate = False; return logger
init_logger = setup_logger("init"); init_logger.info(f"{Fore.MAGENTA}{BRIGHT}Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing...{Style.RESET_ALL}"); init_logger.info(f"Using Timezone: {TIMEZONE_STR}")
def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    updated_config = config.copy(); changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config: updated_config[key] = default_value; changed = True; init_logger.info(f"{NEON_YELLOW}Config Update: Added '{full_key_path}' = {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed: updated_config[key] = nested_config; changed = True
    return updated_config, changed
def load_config(filepath: str) -> Dict[str, Any]:
    global QUOTE_CURRENCY # Declare global at the beginning
    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")
    default_config = {
        "trading_pairs": ["BTC/USDT"], "interval": "5", "retry_delay": RETRY_DELAY_SECONDS, "fetch_limit": DEFAULT_FETCH_LIMIT, "orderbook_limit": 25,
        "enable_trading": False, "use_sandbox": True, "risk_per_trade": 0.01, "leverage": 20, "max_concurrent_positions": 1, "quote_currency": "USDT",
        "loop_delay_seconds": LOOP_DELAY_SECONDS, "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "strategy_params": {"vt_length": DEFAULT_VT_LENGTH, "vt_atr_period": DEFAULT_VT_ATR_PERIOD, "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER), "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER), "ob_source": DEFAULT_OB_SOURCE, "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT, "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT, "ob_extend": DEFAULT_OB_EXTEND, "ob_max_boxes": DEFAULT_OB_MAX_BOXES, "ob_entry_proximity_factor": 1.005, "ob_exit_proximity_factor": 1.001},
        "protection": {"enable_trailing_stop": True, "trailing_stop_callback_rate": 0.005, "trailing_stop_activation_percentage": 0.003, "enable_break_even": True, "break_even_trigger_atr_multiple": 1.0, "break_even_offset_ticks": 2, "initial_stop_loss_atr_multiple": 1.8, "initial_take_profit_atr_multiple": 0.7}}
    config_needs_saving: bool = False; loaded_config: Dict[str, Any] = {}
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f: json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Created default config: {filepath}{RESET}"); QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config
        except IOError as e: init_logger.critical(f"{NEON_RED}FATAL: Error creating config '{filepath}': {e}{RESET}"); QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config
    try:
        with open(filepath, "r", encoding="utf-8") as f: loaded_config = json.load(f)
        if not isinstance(loaded_config, dict): raise TypeError("Config is not a JSON object.")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from '{filepath}': {e}. Recreating default.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f_create: json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Recreated default config: {filepath}{RESET}"); QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config
        except IOError as e_create: init_logger.critical(f"{NEON_RED}FATAL: Error recreating config: {e_create}. Using internal defaults.{RESET}"); QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config
    except Exception as e: init_logger.critical(f"{NEON_RED}FATAL: Error loading config '{filepath}': {e}{RESET}", exc_info=True); QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config
    try:
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config);
        if added_keys: config_needs_saving = True
        def validate_numeric(cfg: Dict, key_path: str, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False) -> bool:
            nonlocal config_needs_saving; keys = key_path.split('.'); current_level = cfg; default_level = default_config
            try:
                for key in keys[:-1]: current_level = current_level[key]; default_level = default_level[key]
                leaf_key = keys[-1]; original_val = current_level.get(leaf_key); default_val = default_level.get(leaf_key)
            except (KeyError, TypeError): init_logger.error(f"Config validation error: Invalid path '{key_path}'."); return False
            if original_val is None: init_logger.warning(f"Config validation: Value missing at '{key_path}'. Using default: {repr(default_val)}"); current_level[leaf_key] = default_val; config_needs_saving = True; return True
            corrected = False; final_val = original_val
            try:
                num_val = Decimal(str(original_val)); min_dec = Decimal(str(min_val)); max_dec = Decimal(str(max_val))
                min_check = num_val > min_dec if is_strict_min else num_val >= min_dec; range_check = min_check and num_val <= max_dec; zero_ok = allow_zero and num_val.is_zero()
                if not range_check and not zero_ok: raise ValueError("Value outside allowed range.")
                target_type = int if is_int else float; converted_val = target_type(num_val); needs_correction = False
                if isinstance(original_val, bool): raise TypeError("Boolean found where numeric expected.")
                elif is_int and not isinstance(original_val, int): needs_correction = True
                elif not is_int and not isinstance(original_val, float): needs_correction = True if not isinstance(original_val, int) else bool(converted_val := float(original_val)) or True
                elif isinstance(original_val, float) and abs(original_val - converted_val) > 1e-9: needs_correction = True
                elif isinstance(original_val, int) and original_val != converted_val: needs_correction = True
                if needs_correction: init_logger.info(f"{NEON_YELLOW}Config Update: Corrected '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}"); final_val = converted_val; corrected = True
            except (ValueError, InvalidOperation, TypeError) as e:
                range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}" + (" or 0" if allow_zero else ""); init_logger.warning(f"{NEON_YELLOW}Config Validation: Invalid '{key_path}'='{repr(original_val)}'. Using default: {repr(default_val)}. Err: {e}. Expected: {'int' if is_int else 'float'}, Range: {range_str}{RESET}"); final_val = default_val; corrected = True
            if corrected: current_level[leaf_key] = final_val; config_needs_saving = True
            return corrected
        init_logger.debug("# Validating configuration parameters...");
        if not isinstance(updated_config.get("trading_pairs"), list) or not all(isinstance(s, str) and s and '/' in s for s in updated_config.get("trading_pairs", [])): init_logger.warning(f"Invalid 'trading_pairs'. Using default {default_config['trading_pairs']}."); updated_config["trading_pairs"] = default_config["trading_pairs"]; config_needs_saving = True
        if updated_config.get("interval") not in VALID_INTERVALS: init_logger.warning(f"Invalid 'interval'. Using default '{default_config['interval']}'."); updated_config["interval"] = default_config["interval"]; config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True); validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True); validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('0.5'), is_strict_min=True); validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True); validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True); validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)
        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"): init_logger.warning(f"Invalid 'quote_currency'. Using default '{default_config['quote_currency']}'."); updated_config["quote_currency"] = default_config["quote_currency"]; config_needs_saving = True
        if not isinstance(updated_config.get("enable_trading"), bool): init_logger.warning(f"Invalid 'enable_trading'. Using default '{default_config['enable_trading']}'."); updated_config["enable_trading"] = default_config["enable_trading"]; config_needs_saving = True
        if not isinstance(updated_config.get("use_sandbox"), bool): init_logger.warning(f"Invalid 'use_sandbox'. Using default '{default_config['use_sandbox']}'."); updated_config["use_sandbox"] = default_config["use_sandbox"]; config_needs_saving = True
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 1000, is_int=True); validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True); validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True); validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0); validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True); validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True); validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True); validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True); validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True); validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1); validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]: init_logger.warning(f"Invalid ob_source. Using default '{DEFAULT_OB_SOURCE}'."); updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE; config_needs_saving = True
        if not isinstance(updated_config["strategy_params"].get("ob_extend"), bool): init_logger.warning(f"Invalid ob_extend. Using default '{DEFAULT_OB_EXTEND}'."); updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND; config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_trailing_stop"), bool): init_logger.warning(f"Invalid enable_trailing_stop. Using default."); updated_config["protection"]["enable_trailing_stop"] = default_config["protection"]["enable_trailing_stop"]; config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_break_even"), bool): init_logger.warning(f"Invalid enable_break_even. Using default."); updated_config["protection"]["enable_break_even"] = default_config["protection"]["enable_break_even"]; config_needs_saving = True
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.1'), is_strict_min=True); validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.1'), allow_zero=True); validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0')); validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True); validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('20.0'), is_strict_min=True); validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('20.0'), allow_zero=True)
        if config_needs_saving:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write: json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Config file '{filepath}' updated.{RESET}")
             except Exception as save_err: init_logger.error(f"{NEON_RED}Error saving updated config to '{filepath}': {save_err}{RESET}", exc_info=True)
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT"); init_logger.info(f"Quote currency set: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        init_logger.info(f"{Fore.CYAN}# Configuration loading/validation complete.{Style.RESET_ALL}"); return updated_config
    except Exception as e: init_logger.critical(f"{NEON_RED}FATAL: Error processing config: {e}. Using defaults.{RESET}", exc_info=True); QUOTE_CURRENCY = default_config.get("quote_currency", "USDT"); return default_config
CONFIG = load_config(CONFIG_FILE);
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    lg = logger; lg.info(f"{Fore.CYAN}# Initializing Bybit exchange connection...{Style.RESET_ALL}")
    try:
        exchange_options = {'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'linear', 'adjustForTimeDifference': True, 'fetchTickerTimeout': 15000, 'fetchBalanceTimeout': 20000, 'createOrderTimeout': 30000, 'cancelOrderTimeout': 20000, 'fetchPositionsTimeout': 20000, 'fetchOHLCVTimeout': 60000}}
        exchange = ccxt.bybit(exchange_options); is_sandbox = CONFIG.get('use_sandbox', True); exchange.set_sandbox_mode(is_sandbox)
        if is_sandbox: lg.warning(f"{NEON_YELLOW}<<< SANDBOX MODE ACTIVE >>>{RESET}")
        else: lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< LIVE TRADING ACTIVE >>> !!!{RESET}")
        lg.info(f"Loading market data for {exchange.id}..."); markets_loaded = False; last_market_error = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market load attempt {attempt + 1}..."); exchange.load_markets(reload=(attempt > 0))
                if exchange.markets and len(exchange.markets) > 0: lg.info(f"{NEON_GREEN}Market data loaded ({len(exchange.markets)} symbols).{RESET}"); markets_loaded = True; break
                else: last_market_error = ValueError("Market data empty."); lg.warning(f"Market data empty (Attempt {attempt + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_market_error = e; lg.warning(f"Network error loading markets (Attempt {attempt + 1}): {e}.")
            except ccxt.AuthenticationError as e: last_market_error = e; lg.critical(f"{NEON_RED}Auth error loading markets: {e}. Exiting.{RESET}"); return None
            except Exception as e: last_market_error = e; lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True); return None
            if not markets_loaded and attempt < MAX_API_RETRIES: delay = RETRY_DELAY_SECONDS * (attempt + 1); lg.warning(f"Retrying market load in {delay}s..."); time.sleep(delay)
        if not markets_loaded: lg.critical(f"{NEON_RED}Failed to load markets. Last error: {last_market_error}. Exiting.{RESET}"); return None
        lg.info(f"Exchange initialized: {exchange.id} | Sandbox: {is_sandbox}")
        lg.info(f"Checking initial balance ({QUOTE_CURRENCY})..."); initial_balance: Optional[Decimal] = None
        try: initial_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        except ccxt.AuthenticationError as auth_err: lg.critical(f"{NEON_RED}Auth error during balance check: {auth_err}. Exiting.{RESET}"); return None
        except Exception as balance_err: lg.warning(f"{NEON_YELLOW}Initial balance check error: {balance_err}.{RESET}", exc_info=False)
        if initial_balance is not None: lg.info(f"{NEON_GREEN}Initial balance: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}"); lg.info(f"{Fore.CYAN}# Exchange init complete.{Style.RESET_ALL}"); return exchange
        else:
            lg.error(f"{NEON_RED}Initial balance check FAILED ({QUOTE_CURRENCY}).{RESET}")
            if CONFIG.get('enable_trading', False): lg.critical(f"{NEON_RED}Trading enabled, but balance check failed. Exiting.{RESET}"); return None
            else: lg.warning(f"{NEON_YELLOW}Trading disabled. Proceeding without balance check.{RESET}"); lg.info(f"{Fore.CYAN}# Exchange init complete (no balance confirmation).{Style.RESET_ALL}"); return exchange
    except Exception as e: lg.critical(f"{NEON_RED}Exchange init failed: {e}{RESET}", exc_info=True); return None
def _safe_market_decimal(value: Optional[Any], field_name: str, allow_zero: bool = True, allow_negative: bool = False) -> Optional[Decimal]:
    if value is None: return None
    try: s_val = str(value).strip();
    if not s_val: return None
    d_val = Decimal(s_val);
    if not allow_zero and d_val.is_zero(): return None
    if not allow_negative and d_val < Decimal('0'): return None
    return d_val
    except (InvalidOperation, TypeError, ValueError): return None
def _format_price(exchange: ccxt.Exchange, symbol: str, price: Union[Decimal, float, str]) -> Optional[str]:
    try: price_decimal = Decimal(str(price));
    if price_decimal <= Decimal('0'): return None
    formatted_str = exchange.price_to_precision(symbol, float(price_decimal)); return formatted_str if Decimal(formatted_str) > Decimal('0') else None
    except Exception as e: init_logger.warning(f"Error formatting price '{price}' for {symbol}: {e}"); return None
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    lg = logger; attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching price ({symbol}, Attempt {attempts + 1})"); ticker = exchange.fetch_ticker(symbol); price: Optional[Decimal] = None; source = "N/A"
            def safe_decimal_from_ticker(val: Optional[Any], name: str) -> Optional[Decimal]: return _safe_market_decimal(val, f"ticker.{name}", allow_zero=False, allow_negative=False)
            price = safe_decimal_from_ticker(ticker.get('last'), 'last'); source = "'last'" if price else source
            if price is None:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid'); ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask: price = (bid + ask) / Decimal('2'); source = f"mid (B:{bid.normalize()}, A:{ask.normalize()})"
                elif ask: price = ask; source = f"'ask' ({ask.normalize()})"
                elif bid: price = bid; source = f"'bid' ({bid.normalize()})"
            if price: normalized_price = price.normalize(); lg.debug(f"Price ({symbol}) from {source}: {normalized_price}"); return normalized_price
            else: last_exception = ValueError("No valid price source in ticker."); lg.warning(f"No valid price ({symbol}, Attempt {attempts + 1}). Ticker: {ticker}. Retrying...")
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Net error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit price ({symbol}): {e}. Wait {wait}s...{RESET}"); time.sleep(wait); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error price: {e}. Stop.{RESET}"); return None
        except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exch error price ({symbol}): {e}{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error price ({symbol}): {e}{RESET}", exc_info=True); return None
        attempts += 1;
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed fetch price ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last: {last_exception}{RESET}"); return None
def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    lg = logger; lg.info(f"{Fore.CYAN}# Fetching klines for {symbol} | TF: {timeframe} | Target: {limit}...{Style.RESET_ALL}")
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'): lg.error(f"Exchange {exchange.id} lacks fetchOHLCV."); return pd.DataFrame()
    min_required = 0;
    try: sp = CONFIG.get('strategy_params', {}); min_required = max(sp.get('vt_length', 0)*2, sp.get('vt_atr_period', 0), sp.get('vt_vol_ema_length', 0), sp.get('ph_left', 0)+sp.get('ph_right', 0)+1, sp.get('pl_left', 0)+sp.get('pl_right', 0)+1) + 50; lg.debug(f"Min candles needed: ~{min_required}");
    except Exception as e: lg.warning(f"Could not estimate min candles: {e}")
    if limit < min_required: lg.warning(f"{NEON_YELLOW}Req limit ({limit}) < est strategy need ({min_required}). Accuracy risk.{RESET}")
    category = 'spot'; market_id = symbol
    try: market = exchange.market(symbol); market_id = market['id']; category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'; lg.debug(f"Using category '{category}', market ID '{market_id}'.")
    except Exception as e: lg.warning(f"Could not get market category/ID ({symbol}): {e}.")
    all_ohlcv_data: List[List] = []; remaining_limit = limit; end_timestamp_ms: Optional[int] = None; max_chunks = math.ceil(limit / BYBIT_API_KLINE_LIMIT) + 2; chunk_num = 0; total_fetched = 0
    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1; fetch_size = min(remaining_limit, BYBIT_API_KLINE_LIMIT); lg.debug(f"Fetching chunk {chunk_num}/{max_chunks} ({fetch_size} candles) for {symbol}. End TS: {end_timestamp_ms}")
        attempts = 0; last_exception = None; chunk_data = None
        while attempts <= MAX_API_RETRIES:
            try:
                params = {'category': category} if 'bybit' in exchange.id.lower() else {}; fetch_args: Dict[str, Any] = {'symbol': symbol, 'timeframe': timeframe, 'limit': fetch_size, 'params': params}
                if end_timestamp_ms: fetch_args['until'] = end_timestamp_ms
                chunk_data = exchange.fetch_ohlcv(**fetch_args); fetched_count_chunk = len(chunk_data) if chunk_data else 0; lg.debug(f"API returned {fetched_count_chunk} for chunk {chunk_num}.")
                if chunk_data:
                    if chunk_num == 1:
                        try:
                            last_ts = pd.to_datetime(chunk_data[-1][0], unit='ms', utc=True); interval_seconds = exchange.parse_timeframe(timeframe)
                            if interval_seconds:
                                max_lag = interval_seconds * 2.5; actual_lag = (pd.Timestamp.utcnow() - last_ts).total_seconds()
                                if actual_lag > max_lag: last_exception = ValueError(f"Stale data? Lag {actual_lag:.1f}s > Max {max_lag:.1f}s"); lg.warning(f"{NEON_YELLOW}Lag detected ({symbol}): {last_exception}. Retrying...{RESET}"); chunk_data = None
                                else: break
                            else: break
                        except Exception as ts_err: lg.warning(f"Lag check failed ({symbol}): {ts_err}. Proceeding."); break
                    else: break
                else: lg.debug(f"API returned no data for chunk {chunk_num}. End of history?"); remaining_limit = 0; break
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Net error klines chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e: last_exception = e; wait = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit klines chunk {chunk_num} ({symbol}): {e}. Wait {wait}s...{RESET}"); time.sleep(wait); continue
            except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error klines: {e}. Stop.{RESET}"); return pd.DataFrame()
            except ccxt.ExchangeError as e:
                last_exception = e; lg.error(f"{NEON_RED}Exch error klines chunk {chunk_num} ({symbol}): {e}{RESET}"); err_str = str(e).lower()
                if "invalid timeframe" in err_str or "interval not supported" in err_str or "symbol invalid" in err_str: lg.critical(f"{NEON_RED}Non-retryable kline error: {e}. Stop.{RESET}"); return pd.DataFrame()
            except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error klines chunk {chunk_num} ({symbol}): {e}{RESET}", exc_info=True); return pd.DataFrame()
            attempts += 1
            if attempts <= MAX_API_RETRIES and chunk_data is None: time.sleep(RETRY_DELAY_SECONDS * attempts)
        if chunk_data:
            all_ohlcv_data = chunk_data + all_ohlcv_data; chunk_len = len(chunk_data); remaining_limit -= chunk_len; total_fetched += chunk_len; end_timestamp_ms = chunk_data[0][0] - 1
            if chunk_len < fetch_size: lg.debug(f"Received fewer candles than requested. Assuming end of history."); remaining_limit = 0
        else:
            lg.error(f"{NEON_RED}Failed fetch kline chunk {chunk_num} ({symbol}) after retries. Last: {last_exception}{RESET}")
            if not all_ohlcv_data: lg.error(f"Failed first chunk ({symbol}). Cannot proceed."); return pd.DataFrame()
            else: lg.warning(f"Proceeding with {total_fetched} candles fetched before error."); break
        if remaining_limit > 0: time.sleep(0.5)
    if chunk_num >= max_chunks and remaining_limit > 0: lg.warning(f"Stopped fetching klines ({symbol}) at max chunks ({max_chunks}).")
    if not all_ohlcv_data: lg.error(f"No kline data fetched ({symbol} {timeframe})."); return pd.DataFrame()
    lg.info(f"Total klines fetched: {len(all_ohlcv_data)}")
    seen_timestamps = set(); unique_data = [];
    for candle in reversed(all_ohlcv_data):
        if candle[0] not in seen_timestamps: unique_data.append(candle); seen_timestamps.add(candle[0])
    if len(unique_data) != len(all_ohlcv_data): lg.warning(f"Removed {len(all_ohlcv_data) - len(unique_data)} duplicate candles ({symbol}).")
    unique_data.sort(key=lambda x: x[0])
    if len(unique_data) > limit: lg.debug(f"Fetched {len(unique_data)}, trimming to {limit}."); unique_data = unique_data[-limit:]
    try:
        lg.debug(f"Processing {len(unique_data)} final candles ({symbol})...")
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']; df = pd.DataFrame(unique_data, columns=cols[:len(unique_data[0])])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce'); df.dropna(subset=['timestamp'], inplace=True)
        if df.empty: lg.error(f"DF empty after timestamp conv ({symbol})."); return pd.DataFrame()
        df.set_index('timestamp', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume'];
        for col in numeric_cols:
            if col in df.columns: numeric_series = pd.to_numeric(df[col], errors='coerce'); df[col] = numeric_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            else: lg.warning(f"Missing col '{col}' ({symbol}).")
        initial_len = len(df); essential = ['open', 'high', 'low', 'close']; df.dropna(subset=essential, inplace=True); df = df[df['close'] > Decimal('0')]
        if 'volume' in df.columns: df.dropna(subset=['volume'], inplace=True); df = df[df['volume'] >= Decimal('0')]
        rows_dropped = initial_len - len(df);
        if rows_dropped > 0: lg.debug(f"Dropped {rows_dropped} rows ({symbol}) during cleaning.")
        if df.empty: lg.warning(f"DF empty after cleaning ({symbol})."); return pd.DataFrame()
        if not df.index.is_monotonic_increasing: lg.warning(f"Index not monotonic ({symbol}), sorting..."); df.sort_index(inplace=True)
        if len(df) > MAX_DF_LEN: lg.debug(f"DF length {len(df)} > max {MAX_DF_LEN}. Trimming ({symbol})."); df = df.iloc[-MAX_DF_LEN:].copy()
        lg.info(f"{NEON_GREEN}Processed {len(df)} klines ({symbol} {timeframe}){RESET}"); return df
    except Exception as e: lg.error(f"{NEON_RED}Error processing klines ({symbol}): {e}{RESET}", exc_info=True); return pd.DataFrame()
def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    lg = logger; lg.debug(f"Seeking market details for: {symbol}...")
    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            market: Optional[Dict] = None
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market details for '{symbol}' missing. Refreshing...");
                try: exchange.load_markets(reload=True); lg.info("Market map refreshed.")
                except Exception as reload_err: last_exception = reload_err; lg.error(f"Failed market refresh ({symbol}): {reload_err}")
            try: market = exchange.market(symbol)
            except ccxt.BadSymbol: lg.error(f"{NEON_RED}Symbol '{symbol}' invalid on {exchange.id}.{RESET}"); return None
            except Exception as fetch_err: last_exception = fetch_err; lg.warning(f"Error fetching market dict '{symbol}': {fetch_err}. Retry {attempts + 1}..."); market = None
            if market:
                lg.debug(f"Market found ({symbol}). Parsing..."); std_market = market.copy()
                is_spot = std_market.get('spot', False); is_swap = std_market.get('swap', False); is_future = std_market.get('future', False); is_linear = std_market.get('linear'); is_inverse = std_market.get('inverse')
                std_market['is_contract'] = is_swap or is_future or std_market.get('contract', False); std_market['is_linear'] = bool(is_linear) and std_market['is_contract']; std_market['is_inverse'] = bool(is_inverse) and std_market['is_contract']
                std_market['contract_type_str'] = "Linear" if std_market['is_linear'] else "Inverse" if std_market['is_inverse'] else "Spot" if is_spot else "Unknown"
                precision = std_market.get('precision', {}); limits = std_market.get('limits', {}); amt_limits = limits.get('amount', {}); cost_limits = limits.get('cost', {})
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision.get('amount'), 'prec.amount', False); std_market['price_precision_step_decimal'] = _safe_market_decimal(precision.get('price'), 'prec.price', False)
                std_market['min_amount_decimal'] = _safe_market_decimal(amt_limits.get('min'), 'lim.amt.min'); std_market['max_amount_decimal'] = _safe_market_decimal(amt_limits.get('max'), 'lim.amt.max', False)
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), 'lim.cost.min'); std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), 'lim.cost.max', False)
                contract_size_val = std_market.get('contractSize', '1'); std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, 'contractSize', False) or Decimal('1')
                if std_market['amount_precision_step_decimal'] is None or std_market['price_precision_step_decimal'] is None: lg.critical(f"{NEON_RED}CRITICAL ({symbol}): Missing essential precision data! AmountStep={std_market['amount_precision_step_decimal']}, PriceStep={std_market['price_precision_step_decimal']}"); return None
                amt_s = std_market['amount_precision_step_decimal'].normalize(); price_s = std_market['price_precision_step_decimal'].normalize(); min_a = std_market['min_amount_decimal'].normalize() if std_market['min_amount_decimal'] else 'N/A'; max_a = std_market['max_amount_decimal'].normalize() if std_market['max_amount_decimal'] else 'N/A'; min_c = std_market['min_cost_decimal'].normalize() if std_market['min_cost_decimal'] else 'N/A'; max_c = std_market['max_cost_decimal'].normalize() if std_market['max_cost_decimal'] else 'N/A'; contr_s = std_market['contract_size_decimal'].normalize()
                log_msg = (f"Market Details ({symbol}): Type={std_market['contract_type_str']}, Active={std_market.get('active')}\n  Precision(Amt/Price): {amt_s}/{price_s}\n  Limits(Amt Min/Max): {min_a}/{max_a}\n  Limits(Cost Min/Max): {min_c}/{max_c}\n  Contract Size: {contr_s}"); lg.debug(log_msg)
                try: final_market_info: MarketInfo = std_market; return final_market_info # type: ignore
                except Exception as cast_err: lg.error(f"Error casting market dict ({symbol}): {cast_err}"); return std_market # type: ignore
            else:
                if attempts < MAX_API_RETRIES: lg.warning(f"Symbol '{symbol}' not found/fetch failed (Attempt {attempts + 1}). Retrying...")
                else: lg.error(f"{NEON_RED}Market '{symbol}' not found after retries. Last: {last_exception}{RESET}"); return None
        except ccxt.BadSymbol as e: lg.error(f"Symbol '{symbol}' invalid on {exchange.id}: {e}"); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Net error market info ({symbol}): {e}. Retry {attempts + 1}...{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries NetError market info ({symbol}).{RESET}"); return None
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error market info: {e}. Stop.{RESET}"); return None
        except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exch error market info ({symbol}): {e}{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries ExchError market info ({symbol}).{RESET}"); return None
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error market info ({symbol}): {e}{RESET}", exc_info=True); return None
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed market info ({symbol}) after attempts. Last: {last_exception}{RESET}"); return None
def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    lg = logger; lg.debug(f"Fetching balance for: {currency}..."); attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None; balance_source: str = "N/A"; found: bool = False; balance_info: Optional[Dict] = None
            types_to_check = ['UNIFIED', 'CONTRACT', ''] if 'bybit' in exchange.id.lower() else ['']
            for acc_type in types_to_check:
                try:
                    params = {'accountType': acc_type} if acc_type else {}; type_desc = f"Type: {acc_type}" if acc_type else "Default"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1})...")
                    balance_info = exchange.fetch_balance(params=params)
                    if currency in balance_info and balance_info[currency].get('free') is not None: balance_str = str(balance_info[currency]['free']); balance_source = f"{type_desc} ('free')"; found = True; break
                    elif 'info' in balance_info and isinstance(balance_info.get('info'), dict) and isinstance(balance_info['info'].get('result'), dict) and isinstance(balance_info['info']['result'].get('list'), list):
                        for acc_details in balance_info['info']['result']['list']:
                            if (not acc_type or acc_details.get('accountType') == acc_type) and isinstance(acc_details.get('coin'), list):
                                for coin_data in acc_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        val = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                        src = 'availableToWithdraw' if coin_data.get('availableToWithdraw') else 'availableBalance' if coin_data.get('availableBalance') else 'walletBalance'
                                        if val is not None: balance_str = str(val); balance_source = f"V5 ({acc_details.get('accountType')}, '{src}')"; found = True; break
                                if found: break
                        if found: break
                except ccxt.ExchangeError as e: err_str = str(e).lower(); if acc_type and ("account type does not exist" in err_str or "invalid account type" in err_str): lg.debug(f"Account type '{acc_type}' not found. Trying next..."); continue; elif acc_type: lg.debug(f"Exch err bal ({acc_type}): {e}. Trying next..."); last_exception = e; continue; else: raise e
                except Exception as e: lg.warning(f"Unexpected err bal ({acc_type or 'Default'}): {e}. Trying next..."); last_exception = e; continue
            if found and balance_str is not None:
                try: bal_dec = Decimal(balance_str); final_bal = max(bal_dec, Decimal('0')); lg.debug(f"Parsed balance ({currency}) from {balance_source}: {final_bal.normalize()}"); return final_bal
                except (ValueError, InvalidOperation, TypeError) as e: raise ccxt.ExchangeError(f"Failed convert balance str '{balance_str}' ({currency}): {e}")
            elif not found: raise ccxt.ExchangeError(f"Balance for '{currency}' not found. Last response: {balance_info}")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Net error balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit balance ({currency}): {e}. Wait {wait}s...{RESET}"); time.sleep(wait); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error balance: {e}. Stop.{RESET}"); raise e
        except ccxt.ExchangeError as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Exch error balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error balance ({currency}): {e}{RESET}", exc_info=True); return None
        attempts += 1;
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed fetch balance ({currency}) after {MAX_API_RETRIES + 1} attempts. Last: {last_exception}{RESET}"); return None
def get_open_position(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, logger: logging.Logger) -> Optional[PositionInfo]:
    lg = logger; attempts = 0; last_exception = None; market_id = market_info.get('id'); category = market_info.get('contract_type_str', 'Spot').lower()
    if not market_info.get('is_contract'): lg.debug(f"Position check skipped ({symbol}): Spot."); return None
    if not market_id or category not in ['linear', 'inverse']: lg.error(f"Cannot check position ({symbol}): Invalid market ID/category."); return None
    lg.debug(f"Using Market ID: '{market_id}', Category: '{category}' for position check.")
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions ({symbol}, Attempt {attempts + 1})...")
            positions: List[Dict] = [];
            try:
                params = {'category': category, 'symbol': market_id}; lg.debug(f"Fetching positions with params: {params}")
                if exchange.has.get('fetchPositions'): all_pos = exchange.fetch_positions(params=params); positions = [p for p in all_pos if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]; lg.debug(f"Fetched {len(all_pos)} ({category}), filtered to {len(positions)} for {symbol}.")
                else: raise ccxt.NotSupported(f"{exchange.id} lacks fetchPositions.")
            except ccxt.ExchangeError as e:
                 codes = [110025]; msgs = ["position not found", "no position", "position does not exist", "order not found"]; err_str = str(e).lower(); code_str = ""; match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE); code_str = match.group(2) if match else str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
                 if (any(str(c) == code_str for c in codes) if code_str else False) or any(msg in err_str for msg in msgs): lg.info(f"No open position found ({symbol})."); return None
                 else: raise e
            active_raw: Optional[Dict] = None; size_thresh = Decimal('1e-9');
            try: amt_step = market_info.get('amount_precision_step_decimal'); size_thresh = amt_step * Decimal('0.01') if amt_step and amt_step > 0 else size_thresh
            except Exception: pass
            lg.debug(f"Pos size threshold > {size_thresh.normalize()} ({symbol}).")
            for pos in positions:
                size_str = str(pos.get('info', {}).get('size', pos.get('contracts', ''))).strip()
                if not size_str: continue
                try: size_dec = Decimal(size_str);
                if abs(size_dec) > size_thresh: active_raw = pos; active_raw['size_decimal'] = size_dec; lg.debug(f"Found active pos ({symbol}): Size={size_dec.normalize()}"); break
                else: lg.debug(f"Skipping pos near-zero size ({symbol}, {size_dec.normalize()}).")
                except (ValueError, InvalidOperation, TypeError) as e: lg.warning(f"Could not parse pos size '{size_str}' ({symbol}): {e}. Skipping."); continue
            if active_raw:
                std_pos = active_raw.copy(); info = std_pos.get('info', {}); side = std_pos.get('side'); parsed_size = std_pos['size_decimal']
                if side not in ['long', 'short']:
                    side_v5 = str(info.get('side', '')).strip().lower(); side = 'long' if side_v5 == 'buy' else 'short' if side_v5 == 'sell' else 'long' if parsed_size > size_thresh else 'short' if parsed_size < -size_thresh else None
                if not side: lg.error(f"Could not determine side ({symbol}). Size: {parsed_size}. Data: {info}"); return None
                std_pos['side'] = side; std_pos['entryPrice'] = _safe_market_decimal(std_pos.get('entryPrice') or info.get('avgPrice'), 'pos.entry', False); std_pos['leverage'] = _safe_market_decimal(std_pos.get('leverage') or info.get('leverage'), 'pos.lev', False); std_pos['liquidationPrice'] = _safe_market_decimal(std_pos.get('liquidationPrice') or info.get('liqPrice'), 'pos.liq', False); std_pos['unrealizedPnl'] = _safe_market_decimal(std_pos.get('unrealizedPnl') or info.get('unrealisedPnl'), 'pos.pnl', True, True)
                def get_prot(name: str) -> Optional[str]: v = info.get(name); s = str(v).strip() if v is not None else None; try: return s if s and abs(Decimal(s)) > Decimal('1e-12') else None; except: return None
                std_pos['stopLossPrice'] = get_prot('stopLoss'); std_pos['takeProfitPrice'] = get_prot('takeProfit'); std_pos['trailingStopLoss'] = get_prot('trailingStop'); std_pos['tslActivationPrice'] = get_prot('activePrice'); std_pos['be_activated'] = False; std_pos['tsl_activated'] = bool(std_pos['trailingStopLoss'])
                def fmt_log(val: Optional[Any]) -> str: dec = _safe_market_decimal(val, 'log', True, True); return dec.normalize() if dec is not None else 'N/A'
                ep = fmt_log(std_pos.get('entryPrice')); sz = std_pos['size_decimal'].normalize(); sl = fmt_log(std_pos.get('stopLossPrice')); tp = fmt_log(std_pos.get('takeProfitPrice')); tsl_d = fmt_log(std_pos.get('trailingStopLoss')); tsl_a = fmt_log(std_pos.get('tslActivationPrice')); tsl = f"Dist={tsl_d}/Act={tsl_a}" if tsl_d != 'N/A' or tsl_a != 'N/A' else "N/A"; pnl = fmt_log(std_pos.get('unrealizedPnl')); liq = fmt_log(std_pos.get('liquidationPrice'))
                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Pos Found ({symbol}):{RESET} Sz={sz}, Entry={ep}, Liq={liq}, PnL={pnl}, SL={sl}, TP={tp}, TSL={tsl}");
                try: final_pos: PositionInfo = std_pos; return final_pos # type: ignore
                except Exception as cast_err: lg.error(f"Error casting pos dict ({symbol}): {cast_err}"); return std_pos # type: ignore
            else: lg.info(f"No active position found ({symbol})."); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Net error positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit positions ({symbol}): {e}. Wait {wait}s...{RESET}"); time.sleep(wait); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error positions: {e}. Stop.{RESET}"); return None
        except ccxt.ExchangeError as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Exch error positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error positions ({symbol}): {e}{RESET}", exc_info=True); return None
        attempts += 1;
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed position info ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last: {last_exception}{RESET}"); return None
def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    lg = logger;
    if not market_info.get('is_contract', False): lg.info(f"Lev setting skipped ({symbol}): Not contract."); return True
    if not isinstance(leverage, int) or leverage <= 0: lg.warning(f"Lev setting skipped ({symbol}): Invalid leverage {leverage}."); return False
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'): lg.error(f"{exchange.id} lacks setLeverage."); return False
    market_id = market_info['id']; attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Setting leverage ({market_id} to {leverage}x, Attempt {attempts + 1})..."); params = {}; category = market_info.get('contract_type_str', 'Linear').lower()
            if 'bybit' in exchange.id.lower():
                 if category not in ['linear', 'inverse']: lg.error(f"Lev setting failed ({symbol}): Invalid category '{category}'."); return False
                 params = {'category': category, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}; lg.debug(f"Using Bybit V5 leverage params: {params}")
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params); lg.debug(f"set_leverage raw response ({symbol}): {response}")
            ret_code_str: Optional[str] = None; ret_msg: str = "N/A"
            if isinstance(response, dict): info_dict = response.get('info', {}); raw_code = info_dict.get('retCode') if info_dict.get('retCode') is not None else response.get('retCode'); ret_code_str = str(raw_code) if raw_code is not None else None; ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown msg'))
            if ret_code_str == '0': lg.info(f"{NEON_GREEN}Leverage set ({market_id} to {leverage}x, Code: 0).{RESET}"); return True
            elif ret_code_str == '110045': lg.info(f"{NEON_YELLOW}Leverage already {leverage}x ({market_id}, Code: 110045).{RESET}"); return True
            elif ret_code_str is not None and ret_code_str not in ['None', '0']: raise ccxt.ExchangeError(f"Bybit API error setting leverage ({symbol}): {ret_msg} (Code: {ret_code_str})")
            else: lg.info(f"{NEON_GREEN}Leverage set/confirmed ({market_id} to {leverage}x, No specific code).{RESET}"); return True
        except ccxt.ExchangeError as e:
            last_exception = e; err_code_str = ""; match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE); err_code_str = match.group(2) if match else str(getattr(e, 'code', '') or getattr(e, 'retCode', '')); err_str_lower = str(e).lower(); lg.error(f"{NEON_RED}Exch error setting leverage ({market_id}): {e} (Code: {err_code_str}){RESET}")
            if err_code_str == '110045' or "leverage not modified" in err_str_lower: lg.info(f"{NEON_YELLOW}Leverage already set (via error). Success.{RESET}"); return True
            fatal_codes = ['10001','10004','110009','110013','110028','110043','110044','110055','3400045']; fatal_messages = ["margin mode", "position exists", "risk limit", "parameter error", "insufficient available balance", "invalid leverage value"]
            if err_code_str in fatal_codes or any(msg in err_str_lower for msg in fatal_messages): lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE leverage error ({symbol}). Aborting.{RESET}"); return False
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries ExchError leverage ({symbol}).{RESET}"); return False
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Net error leverage ({market_id}): {e}. Retry {attempts + 1}...{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries NetError leverage ({symbol}).{RESET}"); return False
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error leverage ({symbol}): {e}. Stop.{RESET}"); return False
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error leverage ({market_id}): {e}{RESET}", exc_info=True); return False
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed leverage set ({market_id} to {leverage}x) after {MAX_API_RETRIES + 1} attempts. Last: {last_exception}{RESET}"); return False
def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal, market_info: MarketInfo, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    lg = logger; symbol = market_info['symbol']; quote_currency = market_info.get('quote', 'QUOTE'); base_currency = market_info.get('base', 'BASE'); is_inverse = market_info.get('is_inverse', False); size_unit = base_currency if market_info.get('spot', False) else "Contracts"
    lg.info(f"{BRIGHT}--- Position Sizing ({symbol}) ---{RESET}")
    if balance <= Decimal('0'): lg.error(f"Sizing fail ({symbol}): Invalid balance {balance.normalize()}."); return None
    try: risk_decimal = Decimal(str(risk_per_trade)); assert Decimal('0') < risk_decimal <= Decimal('1')
    except Exception as e: lg.error(f"Sizing fail ({symbol}): Invalid risk '{risk_per_trade}': {e}"); return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'): lg.error(f"Sizing fail ({symbol}): Entry/SL non-positive."); return None
    if initial_stop_loss_price == entry_price: lg.error(f"Sizing fail ({symbol}): SL==Entry."); return None
    try:
        amount_step = market_info['amount_precision_step_decimal']; price_step = market_info['price_precision_step_decimal']; min_amount = market_info['min_amount_decimal'] or Decimal('0'); max_amount = market_info['max_amount_decimal'] or Decimal('inf'); min_cost = market_info['min_cost_decimal'] or Decimal('0'); max_cost = market_info['max_cost_decimal'] or Decimal('inf'); contract_size = market_info['contract_size_decimal']
        assert amount_step and amount_step > 0; assert price_step and price_step > 0; assert contract_size > 0
        lg.debug(f"  Constraints ({symbol}): AmtStep={amount_step.normalize()}, Min/Max Amt={min_amount.normalize()}/{max_amount.normalize()}, Min/Max Cost={min_cost.normalize()}/{max_cost.normalize()}, ContrSize={contract_size.normalize()}")
    except Exception as e: lg.error(f"Sizing fail ({symbol}): Error validating market details: {e}"); lg.debug(f" MarketInfo: {market_info}"); return None
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN); stop_loss_distance = abs(entry_price - initial_stop_loss_price)
    if stop_loss_distance <= Decimal('0'): lg.error(f"Sizing fail ({symbol}): SL dist zero."); return None
    lg.info(f"  Balance: {balance.normalize()} {quote_currency}, Risk: {risk_decimal:.2%} ({risk_amount_quote.normalize()} {quote_currency}), Entry: {entry_price.normalize()}, SL: {initial_stop_loss_price.normalize()}, SL Dist: {stop_loss_distance.normalize()}, Type: {market_info['contract_type_str']}")
    calculated_size = Decimal('0')
    try:
        if not is_inverse: val_change = stop_loss_distance * contract_size; if val_change <= Decimal('1e-18'): lg.error(f"Sizing fail ({symbol}, Lin/Spot): Value change near zero."); return None; calculated_size = risk_amount_quote / val_change; lg.debug(f"  Lin/Spot Calc: {risk_amount_quote}/{val_change}={calculated_size}")
        else: if entry_price <= 0 or initial_stop_loss_price <= 0: lg.error(f"Sizing fail ({symbol}, Inv): Entry/SL non-positive."); return None; inv_factor = abs((Decimal('1')/entry_price)-(Decimal('1')/initial_stop_loss_price)); if inv_factor <= Decimal('1e-18'): lg.error(f"Sizing fail ({symbol}, Inv): Inv factor near zero."); return None; risk_per_contract = contract_size*inv_factor; if risk_per_contract <= Decimal('1e-18'): lg.error(f"Sizing fail ({symbol}, Inv): Risk/contract near zero."); return None; calculated_size = risk_amount_quote / risk_per_contract; lg.debug(f"  Inv Calc: {risk_amount_quote}/{risk_per_contract}={calculated_size}")
    except (InvalidOperation, OverflowError, ZeroDivisionError) as e: lg.error(f"Sizing fail ({symbol}): Calc error: {e}."); return None
    if calculated_size <= Decimal('0'): lg.error(f"Sizing fail ({symbol}): Initial size zero/negative ({calculated_size.normalize()})."); return None
    lg.info(f"  Initial Calc Size ({symbol}) = {calculated_size.normalize()} {size_unit}")
    adjusted_size = calculated_size
    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        if not isinstance(size, Decimal) or not isinstance(price, Decimal) or price <= 0 or size <= 0: return None
        try: cost = (size*price*contract_size) if not is_inverse else (size*contract_size)/price; return cost.quantize(Decimal('1e-8'), ROUND_UP)
        except: return None
    if min_amount > 0 and adjusted_size < min_amount: lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Calc size {adjusted_size.normalize()} < min {min_amount.normalize()}. Adjust UP.{RESET}"); adjusted_size = min_amount
    if max_amount < Decimal('inf') and adjusted_size > max_amount: lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Calc size {adjusted_size.normalize()} > max {max_amount.normalize()}. Adjust DOWN.{RESET}"); adjusted_size = max_amount
    lg.debug(f"  Size after Amt Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")
    cost_adj = False; est_cost = estimate_cost(adjusted_size, entry_price)
    if est_cost is not None:
        lg.debug(f"  Est Cost (after amt lim, {symbol}): {est_cost.normalize()} {quote_currency}")
        if min_cost > 0 and est_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Est cost {est_cost.normalize()} < min cost {min_cost.normalize()}. Increasing size.{RESET}")
            try: req_size = (min_cost/(entry_price*contract_size)) if not is_inverse else (min_cost*entry_price/contract_size); assert req_size > 0; lg.info(f"  Size req for min cost ({symbol}): {req_size.normalize()} {size_unit}"); assert req_size <= max_amount; adjusted_size = max(min_amount, req_size); cost_adj = True
            except Exception as e: lg.error(f"{NEON_RED}Sizing fail ({symbol}): Min cost size calc error: {e}.{RESET}"); return None
        elif max_cost < Decimal('inf') and est_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Est cost {est_cost.normalize()} > max cost {max_cost.normalize()}. Reducing size.{RESET}")
            try: max_sz = (max_cost/(entry_price*contract_size)) if not is_inverse else (max_cost*entry_price/contract_size); assert max_sz > 0; lg.info(f"  Max size for max cost ({symbol}): {max_sz.normalize()} {size_unit}"); adjusted_size = max(min_amount, min(adjusted_size, max_sz)); cost_adj = True
            except Exception as e: lg.error(f"{NEON_RED}Sizing fail ({symbol}): Max cost size calc error: {e}.{RESET}"); return None
    elif min_cost > 0 or max_cost < Decimal('inf'): lg.warning(f"Could not estimate cost ({symbol}) for limit check.")
    if cost_adj: lg.info(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")
    final_size = adjusted_size
    try: assert amount_step > 0; final_size = (adjusted_size / amount_step).quantize(Decimal('1'), ROUND_DOWN) * amount_step;
    if final_size != adjusted_size: lg.info(f"Applied amt precision ({symbol}, Step:{amount_step.normalize()}, DOWN): {adjusted_size.normalize()}->{final_size.normalize()} {size_unit}")
    except Exception as e: lg.error(f"{NEON_RED}Error applying amt precision ({symbol}): {e}. Using unrounded.{RESET}")
    if final_size <= Decimal('0'): lg.error(f"{NEON_RED}Sizing fail ({symbol}): Final size zero/neg ({final_size.normalize()}).{RESET}"); return None
    if min_amount > 0 and final_size < min_amount: lg.error(f"{NEON_RED}Sizing fail ({symbol}): Final size {final_size.normalize()} < min {min_amount.normalize()} after precision.{RESET}"); return None
    if max_amount < Decimal('inf') and final_size > max_amount: lg.error(f"{NEON_RED}Sizing fail ({symbol}): Final size {final_size.normalize()} > max {max_amount.normalize()} after precision.{RESET}"); return None
    final_cost = estimate_cost(final_size, entry_price)
    if final_cost is not None:
        lg.debug(f"  Final Est Cost ({symbol}): {final_cost.normalize()} {quote_currency}")
        if min_cost > 0 and final_cost < min_cost:
             lg.warning(f"{NEON_YELLOW}Sizing ({symbol}): Final cost {final_cost.normalize()} < min {min_cost.normalize()}. Trying bump.{RESET}")
             try:
                 next_size = final_size + amount_step; next_cost = estimate_cost(next_size, entry_price)
                 if next_cost is not None: can_bump = (next_cost >= min_cost) and (next_size <= max_amount) and (next_cost <= max_cost);
                 if can_bump: lg.info(f"{NEON_YELLOW}Bumping size ({symbol}) to {next_size.normalize()} for min cost.{RESET}"); final_size = next_size; final_cost = estimate_cost(final_size, entry_price); lg.debug(f"  Final Cost after bump: {final_cost.normalize() if final_cost else 'N/A'}")
                 else: lg.error(f"{NEON_RED}Sizing fail ({symbol}): Cannot meet min cost. Bump violates limits.{RESET}"); return None
                 else: lg.error(f"{NEON_RED}Sizing fail ({symbol}): Could not estimate bumped cost.{RESET}"); return None
             except Exception as e: lg.error(f"{NEON_RED}Sizing fail ({symbol}): Error bumping size: {e}.{RESET}"); return None
        elif max_cost < Decimal('inf') and final_cost > max_cost: lg.error(f"{NEON_RED}Sizing fail ({symbol}): Final cost {final_cost.normalize()} > max {max_cost.normalize()}.{RESET}"); return None
    elif min_cost > 0: lg.warning(f"Could not perform final cost check ({symbol}).")
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Size ({symbol}): {final_size.normalize()} {size_unit} <<< {RESET}")
    lg.info(f"{BRIGHT}--- End Sizing ({symbol}) ---{RESET}"); return final_size
def cancel_order(exchange: ccxt.Exchange, order_id: str, symbol: str, logger: logging.Logger) -> bool:
    lg = logger; attempts = 0; last_exception = None; lg.info(f"Cancelling order ID {order_id} ({symbol})...")
    market_id = symbol; params = {}
    if 'bybit' in exchange.id.lower():
        try: market = exchange.market(symbol); params['category'] = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'; params['symbol'] = market['id']
        except Exception as e: lg.warning(f"Could not get category/market_id for cancel ({symbol}): {e}")
    while attempts <= MAX_API_RETRIES:
        try: lg.debug(f"Cancel attempt {attempts + 1} ID {order_id} ({symbol})..."); exchange.cancel_order(order_id, symbol, params=params); lg.info(f"{NEON_GREEN}Cancelled order {order_id} ({symbol}).{RESET}"); return True
        except ccxt.OrderNotFound: lg.warning(f"{NEON_YELLOW}Order {order_id} ({symbol}) not found. Assume success.{RESET}"); return True
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Net error cancel ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait = RETRY_DELAY_SECONDS * 2; lg.warning(f"{NEON_YELLOW}Rate limit cancel ({symbol}): {e}. Wait {wait}s...{RESET}"); time.sleep(wait); continue
        except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exch error cancel ({symbol}): {e}{RESET}")
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error cancel ({symbol}): {e}. Stop.{RESET}"); return False
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error cancel ({symbol}): {e}{RESET}", exc_info=True); return False
        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed cancel order {order_id} ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last: {last_exception}{RESET}"); return False
def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: MarketInfo, logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    lg = logger; side_map = {"BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell"}; side = side_map.get(trade_signal.upper())
    if side is None: lg.error(f"Invalid signal '{trade_signal}' ({symbol})."); return None
    if not isinstance(position_size, Decimal) or position_size <= Decimal('0'): lg.error(f"Invalid size '{position_size}' ({symbol})."); return None
    order_type = 'market'; is_contract = market_info.get('is_contract', False); base = market_info.get('base', 'BASE'); size_unit = "Contracts" if is_contract else base; action = "Close/Reduce" if reduce_only else "Open/Increase"; market_id = market_info['id']
    try: amount_float = float(position_size); assert amount_float > 1e-15
    except Exception as e: lg.error(f"Failed size conversion ({symbol}): {e}"); return None
    order_args: Dict[str, Any] = {'symbol': market_id, 'type': order_type, 'side': side, 'amount': amount_float}; order_params: Dict[str, Any] = {}
    if 'bybit' in exchange.id.lower() and is_contract:
        try: category = market_info.get('contract_type_str', 'Linear').lower(); assert category in ['linear', 'inverse']; order_params = {'category': category, 'positionIdx': 0};
        if reduce_only: order_params['reduceOnly'] = True; order_params['timeInForce'] = 'IOC'; lg.debug(f"Using Bybit V5 order params ({symbol}): {order_params}")
        except Exception as e: lg.error(f"Failed Bybit V5 param setup ({symbol}): {e}.")
    if params: order_params.update(params); lg.debug(f"Added custom params ({symbol}): {params}")
    if order_params: order_args['params'] = order_params
    lg.warning(f"{BRIGHT}===> Placing {action} | {side.upper()} {order_type.upper()} | {symbol} | Size: {position_size.normalize()} {size_unit} <==={RESET}")
    if order_params: lg.debug(f"  Params ({symbol}): {order_params}")
    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing create_order ({symbol}, Attempt {attempts + 1})..."); order_result = exchange.create_order(**order_args)
            order_id = order_result.get('id', 'N/A'); status = order_result.get('status', 'N/A'); avg_price = _safe_market_decimal(order_result.get('average'), 'order.avg', True); filled = _safe_market_decimal(order_result.get('filled'), 'order.filled', True)
            log_msg = f"{NEON_GREEN}{action} Order Placed!{RESET} ID: {order_id}, Status: {status}" + (f", AvgFill: ~{avg_price.normalize()}" if avg_price else "") + (f", Filled: {filled.normalize()}" if filled else "")
            lg.info(log_msg); lg.debug(f"Full order result ({symbol}): {order_result}"); return order_result
        except ccxt.InsufficientFunds as e: last_exception = e; lg.error(f"{NEON_RED}Order Fail ({symbol} {action}): Insufficient Funds. {e}{RESET}"); return None
        except ccxt.InvalidOrder as e:
            last_exception = e; lg.error(f"{NEON_RED}Order Fail ({symbol} {action}): Invalid Params. {e}{RESET}"); lg.error(f"  Args: {order_args}"); err_lower = str(e).lower(); min_a = market_info.get('min_amount_decimal', 'N/A'); min_c = market_info.get('min_cost_decimal', 'N/A'); amt_s = market_info.get('amount_precision_step_decimal', 'N/A'); max_a = market_info.get('max_amount_decimal', 'N/A'); max_c = market_info.get('max_cost_decimal', 'N/A')
            if any(s in err_lower for s in ["minimum", "too small"]): lg.error(f"  >> Hint: Check size ({position_size.normalize()}) vs Mins (Amt:{min_a}, Cost:{min_c}).")
            elif any(s in err_lower for s in ["precision", "lot size", "step size"]): lg.error(f"  >> Hint: Check size ({position_size.normalize()}) vs Step ({amt_s}).")
            elif any(s in err_lower for s in ["exceed", "too large"]): lg.error(f"  >> Hint: Check size ({position_size.normalize()}) vs Maxs (Amt:{max_a}, Cost:{max_c}).")
            elif "reduce only" in err_lower: lg.error(f"  >> Hint: Reduce-only failed.")
            return None
        except ccxt.ExchangeError as e:
            last_exception = e; err_code = ""; match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE); err_code = match.group(2) if match else str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            lg.error(f"{NEON_RED}Order Fail ({symbol} {action}): Exch Error. {e} (Code: {err_code}){RESET}"); fatal_codes = ['10001','10004','110007','110013','110014','110017','110025','110040','30086','3303001','3303005','3400060','3400088']
            fatal_msgs = ["invalid parameter", "precision", "exceed limit", "risk limit", "invalid symbol", "reduce only", "lot size", "insufficient balance", "leverage exceed", "trigger liquidation"]
            if err_code in fatal_codes or any(msg in str(e).lower() for msg in fatal_msgs): lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE order error ({symbol}).{RESET}"); return None
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries ExchError order ({symbol}).{RESET}"); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Net error order ({symbol}): {e}. Retry {attempts + 1}...{RESET}"); if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries NetError order ({symbol}).{RESET}"); return None
        except ccxt.RateLimitExceeded as e: last_exception = e; wait = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit order ({symbol}): {e}. Wait {wait}s...{RESET}"); time.sleep(wait); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error order ({symbol}): {e}. Stop.{RESET}"); return None
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error order ({symbol}): {e}{RESET}", exc_info=True); return None
        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)
    lg.error(f"{NEON_RED}Failed place {action} order ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last: {last_exception}{RESET}"); return None
# _set_position_protection and set_trailing_stop_loss placeholders (use full code from previous step)
def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, logger: logging.Logger, stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None, trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool: print(f"_set_position_protection placeholder for {symbol}"); return True # Assume success for minified placeholder
def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, config: Dict[str, Any], logger: logging.Logger, take_profit_price: Optional[Decimal] = None) -> bool: print(f"set_trailing_stop_loss placeholder for {symbol}"); return True # Assume success for minified placeholder
# VolumaticOBStrategy and SignalGenerator placeholders (use full code from previous step)
class VolumaticOBStrategy:
    def __init__(self, config, market_info, logger): print(f"VolumaticOBStrategy init placeholder for {market_info['symbol']}"); self.lg=logger; self.market_info=market_info; self.min_data_len=100;
    def _ema_swma(self, series, length): return series.rolling(4).mean() # Placeholder calc
    def _find_pivots(self, series, left, right, is_high): return pd.Series(False, index=series.index) # Placeholder calc
    def update(self, df): print(f"Strategy update placeholder for {self.market_info['symbol']}"); atr_val = df['high'].iloc[-1]-df['low'].iloc[-1] if len(df)>0 else Decimal('1'); return StrategyAnalysisResults(dataframe=df, last_close=df['close'].iloc[-1] if len(df)>0 else Decimal('0'), current_trend_up=True if len(df)>0 and df['close'].iloc[-1]>df['open'].iloc[-1] else False, trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=50, atr=atr_val, upper_band=Decimal('1'), lower_band=Decimal('0'))
class SignalGenerator:
    def __init__(self, config, market_info, logger): print(f"SignalGenerator init placeholder for {market_info['symbol']}"); self.lg=logger; self.price_tick=market_info['price_precision_step_decimal'] or Decimal('0.01')
    def _calculate_initial_sl_tp(self, entry, side, atr): sl = entry - atr if side == 'long' else entry + atr; tp = entry + atr if side == 'long' else entry - atr; return (sl, tp) # Placeholder calc
    def generate_signal(self, analysis, pos, sym): print(f"Signal generate placeholder for {sym}"); return SignalResult(signal="HOLD", reason="Placeholder", initial_sl=None, initial_tp=None)
# analyze_and_trade_symbol placeholder
def analyze_and_trade_symbol(exchange, symbol, config, logger, strategy_engine, signal_generator, market_info, position_states): print(f"analyze_and_trade_symbol placeholder for {symbol}"); logger.debug(f"Analyzing {symbol}...") # Simplified placeholder
# manage_existing_position placeholder
def manage_existing_position(exchange, symbol, market_info, position_info, analysis_results, position_state, logger): print(f"manage_existing_position placeholder for {symbol}"); logger.debug(f"Managing {symbol}...") # Simplified placeholder
# execute_trade_action placeholder
def execute_trade_action(exchange, symbol, market_info, current_position, signal_info, analysis_results, position_state, logger): print(f"execute_trade_action placeholder for {symbol} with signal {signal_info['signal']}"); logger.debug(f"Executing {signal_info['signal']} for {symbol}...") # Simplified placeholder
def _handle_shutdown_signal(signum, frame): global _shutdown_requested; signal_name = signal.Signals(signum).name; init_logger.warning(f"\n{NEON_RED}{BRIGHT}Shutdown signal ({signal_name}) received! Exiting...{RESET}"); _shutdown_requested = True
def main():
    global CONFIG, _shutdown_requested; main_logger = setup_logger("main"); main_logger.info(f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Bot v{BOT_VERSION} Starting ---{Style.RESET_ALL}")
    signal.signal(signal.SIGINT, _handle_shutdown_signal); signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    exchange = initialize_exchange(main_logger);
    if not exchange: main_logger.critical("Exchange init failed. Shutting down."); sys.exit(1)
    trading_pairs = CONFIG.get("trading_pairs", []); valid_pairs: List[str] = []; all_valid = True
    main_logger.info(f"Validating trading pairs: {trading_pairs}")
    market_infos: Dict[str, MarketInfo] = {}; strategy_engines: Dict[str, 'VolumaticOBStrategy'] = {}; signal_generators: Dict[str, 'SignalGenerator'] = {}
    for pair in trading_pairs:
         market_info = get_market_info(exchange, pair, main_logger)
         if market_info and market_info.get('active'):
             valid_pairs.append(pair); market_infos[pair] = market_info; main_logger.info(f" -> {NEON_GREEN}{pair} valid.{RESET}")
             try:
                 strategy_engines[pair] = VolumaticOBStrategy(CONFIG, market_info, setup_logger(pair))
                 signal_generators[pair] = SignalGenerator(CONFIG, market_info, setup_logger(pair))
             except ValueError as init_err: main_logger.error(f" -> {NEON_RED}Failed init for {pair}: {init_err}. Skipping.{RESET}"); all_valid = False; valid_pairs.remove(pair); market_infos.pop(pair, None)
         else: main_logger.error(f" -> {NEON_RED}{pair} invalid/inactive. Skipping.{RESET}"); all_valid = False
    if not valid_pairs: main_logger.critical("No valid pairs. Shutting down."); sys.exit(1)
    if not all_valid: main_logger.warning(f"Proceeding with valid pairs: {valid_pairs}")
    if not CONFIG.get('enable_trading', False): main_logger.warning(f"{NEON_YELLOW}--- TRADING DISABLED ---{RESET}")
    main_logger.info(f"{Fore.CYAN}### Starting Main Loop ###{Style.RESET_ALL}"); loop_count = 0
    position_states: Dict[str, Dict[str, bool]] = {sym: {'be_activated': False, 'tsl_activated': False} for sym in valid_pairs}
    while not _shutdown_requested:
        loop_count += 1; main_logger.debug(f"--- Loop Cycle #{loop_count} ---"); start_time = time.monotonic()
        for symbol in valid_pairs:
            if _shutdown_requested: break
            symbol_logger = get_logger_for_symbol(symbol); symbol_logger.info(f"--- Processing: {symbol} (Cycle #{loop_count}) ---")
            try:
                 market_info = market_infos[symbol]
                 analyze_and_trade_symbol(exchange, symbol, CONFIG, symbol_logger, strategy_engines[symbol], signal_generators[symbol], market_info, position_states)
            except ccxt.AuthenticationError as e: symbol_logger.critical(f"{NEON_RED}Auth Error ({symbol}): {e}. Stopping bot.{RESET}"); _shutdown_requested = True; break
            except Exception as symbol_err: symbol_logger.error(f"{NEON_RED}!! Unhandled error ({symbol}): {symbol_err} !!{RESET}", exc_info=True)
            finally: symbol_logger.info(f"--- Finished: {symbol} ---")
            if _shutdown_requested: break
            time.sleep(0.2)
        if _shutdown_requested: break
        end_time = time.monotonic(); cycle_dur = end_time - start_time; loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS); wait_time = max(0, loop_delay - cycle_dur)
        main_logger.info(f"Cycle {loop_count} duration: {cycle_dur:.2f}s. Waiting {wait_time:.2f}s...")
        for _ in range(int(wait_time)):
             if _shutdown_requested: break
             time.sleep(1)
        if not _shutdown_requested and wait_time % 1 > 0: time.sleep(wait_time % 1)
    main_logger.info(f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Bot Shutting Down ---{Style.RESET_ALL}")
    logging.shutdown(); main_logger.info("Shutdown complete."); sys.exit(0)
if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: init_logger.info("KeyboardInterrupt caught in __main__. Exiting."); sys.exit(0)
    except Exception as global_err: init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL UNHANDLED EXCEPTION:{RESET} {global_err}", exc_info=True); sys.exit(1)