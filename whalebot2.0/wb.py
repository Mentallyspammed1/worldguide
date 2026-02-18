"""
🧙‍♂️ Pyrmethus Whaler Zenith Apex V12: High-Fidelity Bybit Trading Intelligence

This module implements an advanced algorithmic trading system designed for Bybit V5.
It utilizes a multi-layered intelligence approach:
1. Intelligence Layer (The Oracle): L2 flow, depth profiling, and a consensus of 40+ indicators.
2. Execution Layer (The Strike): High-precision Decimal math, exchange filter compliance, and adaptive trailing stops.
3. Stealth Layer (The Cloak): Integrated Tor proxy support and paper/live mode parity.

Architecture:
- APIClient: Hardened communication with Bybit V5 REST API.
- IndicatorCalculator: Vectorized technical analysis suite using NumPy and SciPy.
- TradingAnalyzer: Synthesis of technical indicators and L2 order book metrics.
- RiskManager: High-precision position sizing and automated circuit breakers.
- SignalHistoryTracker: SQLite-backed performance auditing and session management.

Operational Modes:
- Backtest: Historical simulation using vectorized indicators.
- Optimize: Parameter tuning for maximum net profit.
- Paper Trade: Live data analysis with simulated execution.
- Live Trade: Real-world execution on Bybit Linear Perpetual markets.

Codified by Pyrmethus for the digital sanctum.
"""

import argparse
import sys
import decimal
import hashlib
import hmac
import json
import logging
import os
import smtplib
import sqlite3
import statistics
import threading
import time
import warnings

import aiohttp
import websocket
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from email.mime.text import MIMEText
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from scipy.signal import lfilter

from logger_config import setup_custom_logger

warnings.filterwarnings("ignore")

# Set Decimal precision for financial calculations to avoid floating point errors
getcontext().prec = 40

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()


# --- Enums and Data Classes ---
class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class MarketCondition(Enum):
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING = "trending"
    RANGING = "ranging"


class MarketRegime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class TradingSignal:
    signal_type: SignalType | None
    confidence: float
    conditions_met: list[str]
    stop_loss: Decimal | None
    take_profit: Decimal | None
    timestamp: float
    symbol: str
    timeframe: str
    position_size: Decimal | None = None
    risk_reward_ratio: float | None = None


@dataclass
class IndicatorResult:
    name: str
    value: Any
    interpretation: str


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_profit: Decimal = Decimal("0")
    total_loss: Decimal = Decimal("0")
    net_profit: Decimal = Decimal("0")
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")


@dataclass
class SignalHistory:
    timestamp: float
    symbol: str
    timeframe: str
    signal_type: SignalType
    confidence: float
    entry_price: Decimal
    quantity: Decimal = Decimal("0")
    exit_price: Decimal | None = None
    stop_loss: Decimal | None = None
    trailing_sl: Decimal | None = None
    highest_price: Decimal | None = None
    lowest_price: Decimal | None = None
    take_profit: Decimal | None = None
    profit_loss: Decimal | None = None
    fees: Decimal = Decimal("0")
    net_pnl: Decimal | None = None
    exit_reason: str | None = None
    market_regime: MarketRegime | None = None


# --- Color Codex ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN  # Added NEON_CYAN definition
NEON_WHITE = Fore.WHITE
RESET = Style.RESET_ALL

# --- Configuration & Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DATA_DIRECTORY = "bot_data"
DATABASE_FILE = os.path.join(DATA_DIRECTORY, "trading_bot.db")
TIMEZONE = ZoneInfo("America/Chicago")
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5
MAX_SIGNAL_HISTORY = 1000

# Ensure directories exist
os.makedirs(LOG_DIRECTORY, exist_ok=True)
os.makedirs(DATA_DIRECTORY, exist_ok=True)

# Setup the main application logger with rotation
logger = setup_custom_logger("whalebot_main")


# --- Database Setup ---
def setup_database():
    # This function initializes the SQLite database and creates the necessary tables.
    """Set up the SQLite database for storing signal history and performance metrics."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Create signal_history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS signal_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        entry_price TEXT NOT NULL,
        exit_price TEXT,
        stop_loss TEXT,
        take_profit TEXT,
        profit_loss TEXT,
        exit_reason TEXT,
        market_regime TEXT
    )
    """)

    # Create performance_metrics table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        total_trades INTEGER NOT NULL,
        winning_trades INTEGER NOT NULL,
        losing_trades INTEGER NOT NULL,
        win_rate REAL NOT NULL,
        profit_factor REAL NOT NULL,
        max_drawdown REAL NOT NULL,
        sharpe_ratio REAL NOT NULL,
        total_profit TEXT NOT NULL,
        total_loss TEXT NOT NULL,
        net_profit TEXT NOT NULL,
        average_win TEXT NOT NULL,
        average_loss TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# --- Notification System ---
class NotificationSystem:
    """Handles sending notifications via email or webhooks."""

    def __init__(self, config: dict):
        self.config = config.get("notifications", {})
        self.enabled = self.config.get("enabled", False)
        self.email_config = self.config.get("email", {})
        self.webhook_config = self.config.get("webhook", {})
        self.sms_config = self.config.get("sms", {})

    def send_sms(self, message: str) -> bool:
        """Send an SMS notification via Termux API."""
        if not self.enabled or not self.sms_config.get("enabled", False):
            return False

        phone_number = self.sms_config.get("phone_number")
        if not phone_number:
            logger.error(
                f"{NEON_RED}SMS enabled but no phone_number provided in config.{RESET}"
            )
            return False

        try:
            import subprocess

            subprocess.run(
                ["termux-sms-send", "-n", str(phone_number), message],
                check=True,
                capture_output=True,
            )
            logger.info(
                f"{NEON_GREEN}Termux SMS notification sent successfully.{RESET}"
            )
            return True
        except Exception as e:
            logger.error(f"{NEON_RED}Failed to send Termux SMS: {e}{RESET}")
            return False

    def send_email(self, subject: str, message: str) -> bool:
        """Send an email notification."""
        if not self.enabled or not self.email_config.get("enabled", False):
            return False

        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = self.email_config.get("from")
            msg["To"] = self.email_config.get("to")

            with smtplib.SMTP(
                self.email_config.get("smtp_server"), self.email_config.get("smtp_port")
            ) as server:
                if self.email_config.get("use_tls", True):
                    server.starttls()
                server.login(
                    self.email_config.get("username"), self.email_config.get("password")
                )
                server.send_message(msg)

            logger.info(f"{NEON_GREEN}Email notification sent: {subject}{RESET}")
            return True
        except Exception as e:
            logger.error(f"{NEON_RED}Failed to send email notification: {e}{RESET}")
            return False

    def send_webhook(self, payload: dict) -> bool:
        """Send a webhook notification."""
        if not self.enabled or not self.webhook_config.get("enabled", False):
            return False

        try:
            response = requests.post(
                self.webhook_config.get("url"), json=payload, timeout=10
            )
            response.raise_for_status()
            logger.info(f"{NEON_GREEN}Webhook notification sent successfully{RESET}")
            return True
        except Exception as e:
            logger.error(f"{NEON_RED}Failed to send webhook notification: {e}{RESET}")
            return False

    def send_combined_notification(self, subject: str, message: str, payload: dict, sms_message: str) -> None:
        """Send combined notification (email + webhook + sms) in one call."""
        self.send_email(subject, message)
        self.send_webhook(payload)
        self.send_sms(sms_message)

    def send_signal_notification(
        self, signal: TradingSignal, l2_metrics: dict = None, depth_profile: dict = None
    ) -> None:
        """Send a notification for a trading signal with enhanced metrics."""
        subject = (
            f"Trading Signal: {signal.signal_type.value.upper()} for {signal.symbol}"
        )
        l2_info = ""
        if l2_metrics:
            l2_info = f"\nL2 Imbalance: {l2_metrics.get('imbalance_10', 0):.2f}"

        depth_info = ""
        if depth_profile:
            depth_info = f"\nDepth (0.5%): {depth_profile.get('imbalance_0.5%', 0):.2f}"

        message = f"""
Signal: {signal.signal_type.value.upper()}
Symbol: {signal.symbol}
Timeframe: {signal.timeframe}
Confidence: {signal.confidence:.2f}
Conditions: {", ".join(signal.conditions_met)}
Stop Loss: {signal.stop_loss}
Take Profit: {signal.take_profit}{l2_info}{depth_info}
Timestamp: {datetime.fromtimestamp(signal.timestamp).strftime("%Y-%m-%d %H:%M:%S")}
"""

        payload = {
            "signal_type": signal.signal_type.value,
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "confidence": signal.confidence,
            "conditions_met": signal.conditions_met,
            "stop_loss": str(signal.stop_loss) if signal.stop_loss else None,
            "take_profit": str(signal.take_profit) if signal.take_profit else None,
            "l2_metrics": l2_metrics,
            "depth_profile": depth_profile,
            "timestamp": signal.timestamp,
        }

        self.send_email(subject, message)
        self.send_webhook(payload)
        # Also send a condensed SMS
        sms_msg = f"{signal.signal_type.value.upper()} {signal.symbol} @ {signal.confidence:.2f} | SL: {signal.stop_loss}"
        self.send_sms(sms_msg)


# --- Configuration Management ---
def load_config(filepath: str) -> dict:
    """
    Loads configuration from a JSON file, merging with default values.
    If the file is not found or is invalid, it creates one with default settings.
    """
    default_config = {
        "interval": "15",
        "analysis_interval": 30,  # Time in seconds between main analysis cycles
        "retry_delay": 5,  # Delay in seconds for API retries
        "momentum_period": 10,
        "momentum_ma_short": 12,
        "momentum_ma_long": 26,
        "volume_ma_period": 20,
        "atr_period": 14,
        "trend_strength_threshold": 0.4,
        "sideways_atr_multiplier": 1.5,
        "signal_score_threshold": 1.0,  # Minimum combined weight for a signal to be valid
        "indicators": {
            "ema_alignment": True,
            "momentum": True,
            "volume_confirmation": True,
            "divergence": True,
            "stoch_rsi": True,
            "rsi": True,
            "macd": True,
            "vwap": False,
            "obv": True,
            "adi": True,
            "cci": True,
            "wr": True,
            "adx": True,
            "psar": True,
            "fve": True,
            "sma_10": False,
            "mfi": True,
            "stochastic_oscillator": True,
            "bollinger_bands": True,
            "keltner_channels": True,
            "ichimoku_cloud": True,
            "cmf": True,
            "emv": True,
            "force_index": True,
            "mass_index": True,
            "roc": True,
            "trix": True,
            "ultimate_oscillator": True,
            "vortex": True,
            "coppock_curve": True,
            "donchian_channels": True,
            "hma": True,
            "awesome_oscillator": True,
            "std_dev": True,
            "variance": True,
            "klinger_oscillator": True,
            "nvi": True,
            "pvi": True,
            "bop": True,
            "supersmoother": True,
            "ehlers_fisher": True,
            "laguerre_rsi": True,
            "supertrend": True,
            "cmo": True,
            "stc": True,
        },
        "weight_sets": {
            "low_volatility": {  # Weights for a low volatility market environment
                "ema_alignment": 0.3,
                "momentum": 0.2,
                "volume_confirmation": 0.2,
                "divergence": 0.1,
                "stoch_rsi": 0.5,
                "rsi": 0.3,
                "macd": 0.3,
                "vwap": 0.0,
                "obv": 0.1,
                "adi": 0.1,
                "cci": 0.1,
                "wr": 0.1,
                "adx": 0.1,
                "psar": 0.1,
                "fve": 0.2,
                "sma_10": 0.0,
                "mfi": 0.3,
                "stochastic_oscillator": 0.4,
                "bollinger_bands": 0.2,
                "keltner_channels": 0.2,
                "ichimoku_cloud": 0.1,
                "cmf": 0.1,
                "emv": 0.1,
                "force_index": 0.1,
                "mass_index": 0.1,
                "roc": 0.1,
                "trix": 0.1,
                "ultimate_oscillator": 0.2,
                "vortex": 0.2,
                "coppock_curve": 0.1,
                "donchian_channels": 0.1,
                "hma": 0.1,
                "awesome_oscillator": 0.1,
                "std_dev": 0.0,
                "variance": 0.0,
                "klinger_oscillator": 0.1,
                "nvi": 0.1,
                "pvi": 0.1,
                "bop": 0.1,
            },
            "high_volatility": {  # Weights for a high volatility market environment
                "ema_alignment": 0.1,
                "momentum": 0.4,
                "volume_confirmation": 0.1,
                "divergence": 0.2,
                "stoch_rsi": 0.4,
                "rsi": 0.4,
                "macd": 0.4,
                "vwap": 0.0,
                "obv": 0.1,
                "adi": 0.1,
                "cci": 0.1,
                "wr": 0.1,
                "adx": 0.1,
                "psar": 0.1,
                "fve": 0.3,
                "sma_10": 0.0,
                "mfi": 0.4,
                "stochastic_oscillator": 0.3,
                "bollinger_bands": 0.3,
                "keltner_channels": 0.3,
                "ichimoku_cloud": 0.2,
                "cmf": 0.2,
                "emv": 0.2,
                "force_index": 0.2,
                "mass_index": 0.2,
                "roc": 0.2,
                "trix": 0.2,
                "ultimate_oscillator": 0.3,
                "vortex": 0.3,
                "coppock_curve": 0.2,
                "donchian_channels": 0.2,
                "hma": 0.2,
                "awesome_oscillator": 0.2,
                "std_dev": 0.1,
                "variance": 0.1,
                "klinger_oscillator": 0.2,
                "nvi": 0.2,
                "pvi": 0.2,
                "bop": 0.2,
            },
        },
        "stoch_rsi_oversold_threshold": 20,
        "stoch_rsi_overbought_threshold": 80,
        "stoch_rsi_confidence_boost": 5,  # Additional boost for strong Stoch RSI signals
        "stoch_rsi_mandatory": False,  # If true, Stoch RSI must be a confirming factor
        "rsi_confidence_boost": 2,
        "mfi_confidence_boost": 2,
        "order_book_support_confidence_boost": 3,
        "order_book_resistance_confidence_boost": 3,
        "stop_loss_multiple": 1.5,  # Multiplier for ATR to determine stop loss distance
        "take_profit_multiple": 1.0,  # Multiplier for ATR to determine take profit distance
        "order_book_wall_threshold_multiplier": 2.0,  # Multiplier for average volume to identify a "wall"
        "order_book_depth_to_check": 10,  # Number of order book levels to check for walls
        "price_change_threshold": 0.005,  # % change in price to consider significant
        "atr_change_threshold": 0.005,  # % change in ATR to consider significant volatility change
        "signal_cooldown_s": 60,  # Seconds to wait before generating another signal
        "order_book_debounce_s": 10,  # Seconds to wait between order book API calls
        "ema_short_period": 12,
        "ema_long_period": 26,
        "volume_confirmation_multiplier": 1.5,  # Volume must be this many times average volume for confirmation
        "indicator_periods": {
            "rsi": 14,
            "mfi": 14,
            "cci": 20,
            "williams_r": 14,
            "adx": 14,
            "stoch_rsi_period": 14,  # Period for RSI calculation within Stoch RSI
            "stoch_rsi_k_period": 3,  # Smoothing period for %K line
            "stoch_rsi_d_period": 3,  # Smoothing period for %D line (signal line)
            "momentum": 10,
            "momentum_ma_short": 12,
            "momentum_ma_long": 26,
            "volume_ma": 20,
            "atr": 14,
            "sma_10": 10,
            "fve_price_ema": 10,  # EMA period for FVE price component
            "fve_obv_sma": 20,  # SMA period for OBV normalization
            "fve_atr_sma": 20,  # SMA period for ATR normalization
            "stoch_osc_k": 14,  # Stochastic Oscillator K period
            "stoch_osc_d": 3,  # Stochastic Oscillator D period
        },
        "order_book_analysis": {
            "enabled": True,
            "wall_threshold_multiplier": 2.0,
            "depth_to_check": 10,
            "support_boost": 3,
            "resistance_boost": 3,
        },
        "trailing_stop_loss": {
            "enabled": False,  # Disabled by default
            "initial_activation_percent": 0.5,  # Activate trailing stop after price moves X% in favor
            "trailing_stop_multiple_atr": 1.5,  # Trail stop based on ATR multiple
        },
        "take_profit_scaling": {
            "enabled": False,  # Disabled by default
            "targets": [
                {
                    "level": 1.5,
                    "percentage": 0.25,
                },  # Sell 25% when price hits 1.5x ATR TP
                {
                    "level": 2.0,
                    "percentage": 0.50,
                },  # Sell 50% of remaining when price hits 2.0x ATR TP
            ],
        },
        "risk_management": {
            "leverage": 10,
            "max_position_size": 0.1,  # Maximum position size as a percentage of portfolio
            "max_daily_loss": 0.05,  # Maximum daily loss as a percentage of portfolio
            "max_drawdown": 0.15,  # Maximum drawdown before stopping trading
            "risk_per_trade": 0.02,  # Risk percentage per trade
            "portfolio_value": 10000,  # Total portfolio value
            "circuit_breaker": {
                "enabled": True,
                "max_consecutive_losses": 5,
                "cooldown_period_minutes": 60,
            },
        },
        "data_validation": {
            "min_data_points": 50,  # Minimum data points required for analysis
            "max_data_age_minutes": 60,  # Maximum age of data before considering it stale
            "price_deviation_threshold": 0.1,  # Maximum allowed price deviation (%)
        },
        "notifications": {
            "enabled": False,
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "username": "",
                "password": "",
                "from": "",
                "to": "",
            },
            "webhook": {"enabled": False, "url": ""},
            "sms": {"enabled": True, "phone_number": "17145102759"},
        },
        "multi_timeframe": {
            "enabled": False,
            "timeframes": ["5", "15", "60"],
            "weighting": {"5": 0.2, "15": 0.5, "60": 0.3},
        },
        "backtesting": {
            "enabled": False,
            "start_date": "",
            "end_date": "",
            "initial_balance": 10000,
        },
        "logging": {
            "level": "INFO",
            "max_file_size": 10485760,  # 10MB
            "backup_count": 5,
        },
        "paper_mode": True,  # Default to safety
        "proxy": {
            "enabled": False,
            "url": "socks5h://127.0.0.1:9050",  # Default Tor proxy
        },
        "database": {
            "enabled": True,
            "path": DATABASE_FILE,
            "backup_enabled": True,
            "backup_interval_hours": 24,
        },
    }

    try:
        with open(filepath, encoding="utf-8") as f:
            config = json.load(f)
            # Merge loaded config with defaults. Prioritize loaded values, but ensure all default keys exist.
            merged_config = {**default_config, **config}
            # Basic validation for interval and analysis_interval
            if merged_config.get("interval") not in VALID_INTERVALS:
                logger.warning(
                    f"{NEON_YELLOW}Invalid 'interval' in config, using default: {default_config['interval']}{RESET}"
                )
                merged_config["interval"] = default_config["interval"]
            if (
                not isinstance(merged_config.get("analysis_interval"), int)
                or merged_config.get("analysis_interval") <= 0
            ):
                logger.warning(
                    f"{NEON_YELLOW}Invalid 'analysis_interval' in config, using default: {default_config['analysis_interval']}{RESET}"
                )
                merged_config["analysis_interval"] = default_config["analysis_interval"]
            return merged_config
    except FileNotFoundError:
        logger.warning(
            f"{NEON_YELLOW}Config file not found, loading defaults and creating {filepath}{RESET}"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config
    except json.JSONDecodeError:
        logger.error(f"{NEON_RED}Invalid JSON in config file, loading defaults.{RESET}")
        # Optionally, back up the corrupt file before overwriting
        try:
            os.rename(filepath, f"{filepath}.bak_{int(time.time())}")
            logger.info(
                f"{NEON_YELLOW}Backed up corrupt config file to {filepath}.bak_{int(time.time())}{RESET}"
            )
        except OSError as e:
            logger.error(f"{NEON_RED}Failed to backup corrupt config file: {e}{RESET}")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config


# Load the configuration
CONFIG = load_config(CONFIG_FILE)


# --- Database Operations ---
class DatabaseManager:
    """Manages database operations for signal history and performance metrics."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self._ensure_db_exists()

    def _get_conn(self) -> sqlite3.Connection:
        return self.conn

    def _ensure_db_exists(self):
        """Ensure the database and tables exist."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Create signal_history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            entry_price TEXT NOT NULL,
            exit_price TEXT,
            stop_loss TEXT,
            take_profit TEXT,
            profit_loss TEXT,
            exit_reason TEXT,
            market_regime TEXT
        )
        """)

        # Create performance_metrics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            total_trades INTEGER NOT NULL,
            winning_trades INTEGER NOT NULL,
            losing_trades INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            profit_factor REAL NOT NULL,
            max_drawdown REAL NOT NULL,
            sharpe_ratio REAL NOT NULL,
            total_profit TEXT NOT NULL,
            total_loss TEXT NOT NULL,
            net_profit TEXT NOT NULL,
            average_win TEXT NOT NULL,
            average_loss TEXT NOT NULL
        )
        """)

        conn.commit()
        conn.close()

    def save_signal(self, signal: SignalHistory) -> int:
        """Save a signal to the database and return its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO signal_history (
            timestamp, symbol, timeframe, signal_type, confidence,
            entry_price, exit_price, stop_loss, take_profit,
            profit_loss, exit_reason, market_regime
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                signal.timestamp,
                signal.symbol,
                signal.timeframe,
                signal.signal_type.value,
                signal.confidence,
                str(signal.entry_price),
                str(signal.exit_price) if signal.exit_price else None,
                str(signal.stop_loss) if signal.stop_loss else None,
                str(signal.take_profit) if signal.take_profit else None,
                str(signal.profit_loss) if signal.profit_loss else None,
                signal.exit_reason,
                signal.market_regime.value if signal.market_regime else None,
            ),
        )

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return signal_id

    def update_signal(
        self,
        signal_id: int,
        exit_price: Decimal,
        profit_loss: Decimal,
        exit_reason: str,
    ) -> bool:
        """Update a signal with exit information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        UPDATE signal_history
        SET exit_price = ?, profit_loss = ?, exit_reason = ?
        WHERE id = ?
        """,
            (str(exit_price), str(profit_loss), exit_reason, signal_id),
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def get_signal_history(
        self, symbol: str = None, timeframe: str = None, limit: int = 100
    ) -> list[SignalHistory]:
        """Retrieve signal history from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM signal_history"
        params = []

        if symbol or timeframe:
            conditions = []
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if timeframe:
                conditions.append("timeframe = ?")
                params.append(timeframe)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        signals = []
        for row in rows:
            signals.append(
                SignalHistory(
                    timestamp=row[1],
                    symbol=row[2],
                    timeframe=row[3],
                    signal_type=SignalType(row[4]),
                    confidence=row[5],
                    entry_price=Decimal(row[6]),
                    exit_price=Decimal(row[7]) if row[7] else None,
                    stop_loss=Decimal(row[8]) if row[8] else None,
                    take_profit=Decimal(row[9]) if row[9] else None,
                    profit_loss=Decimal(row[10]) if row[10] else None,
                    exit_reason=row[11],
                    market_regime=MarketRegime(row[12]) if row[12] else None,
                )
            )

        return signals

    def save_performance_metrics(
        self, metrics: PerformanceMetrics, symbol: str, timeframe: str
    ) -> int:
        """Save performance metrics to the database and return its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO performance_metrics (
            timestamp, symbol, timeframe, total_trades, winning_trades,
            losing_trades, win_rate, profit_factor, max_drawdown,
            sharpe_ratio, total_profit, total_loss, net_profit,
            average_win, average_loss
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                symbol,
                timeframe,
                metrics.total_trades,
                metrics.winning_trades,
                metrics.losing_trades,
                metrics.win_rate,
                metrics.profit_factor,
                metrics.max_drawdown,
                metrics.sharpe_ratio,
                str(metrics.total_profit),
                str(metrics.total_loss),
                str(metrics.net_profit),
                str(metrics.average_win),
                str(metrics.average_loss),
            ),
        )

        metrics_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return metrics_id

    def get_latest_performance_metrics(
        self, symbol: str, timeframe: str
    ) -> PerformanceMetrics | None:
        """Retrieve the latest performance metrics for a symbol and timeframe."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT * FROM performance_metrics
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC LIMIT 1
        """,
            (symbol, timeframe),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return PerformanceMetrics(
            total_trades=row[3],
            winning_trades=row[4],
            losing_trades=row[5],
            win_rate=row[6],
            profit_factor=row[7],
            max_drawdown=row[8],
            sharpe_ratio=row[9],
            total_profit=Decimal(row[10]),
            total_loss=Decimal(row[11]),
            net_profit=Decimal(row[12]),
            average_win=Decimal(row[13]),
            average_loss=Decimal(row[14]),
        )

    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            import shutil

            shutil.copy2(self.db_path, backup_path)
            logger.info(f"{NEON_GREEN}Database backed up to {backup_path}{RESET}")
            return True
        except Exception as e:
            logger.error(f"{NEON_RED}Failed to backup database: {e}{RESET}")
            return False

    def vacuum_database(self):
        """Perform database vacuum periodically to optimize database file size."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('VACUUM')
        conn.close()


# --- Performance Calculator ---
class PerformanceCalculator:
    """Calculates performance metrics from signal history."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def calculate_metrics(self, symbol: str, timeframe: str) -> PerformanceMetrics:
        """Calculate performance metrics for a symbol and timeframe."""
        signals = self.db_manager.get_signal_history(
            symbol, timeframe, limit=MAX_SIGNAL_HISTORY
        )

        # Filter completed signals (with exit price)
        completed_signals = [s for s in signals if s.exit_price is not None]

        if not completed_signals:
            return PerformanceMetrics()

        # Calculate basic metrics
        total_trades = len(completed_signals)
        winning_trades = sum(
            1 for s in completed_signals if s.profit_loss and s.profit_loss > 0
        )
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Calculate profit metrics
        total_profit = sum(
            s.profit_loss
            for s in completed_signals
            if s.profit_loss and s.profit_loss > 0
        )
        total_loss = abs(
            sum(
                s.profit_loss
                for s in completed_signals
                if s.profit_loss and s.profit_loss < 0
            )
        )

        net_profit = total_profit - total_loss

        average_win = (
            total_profit / winning_trades if winning_trades > 0 else Decimal("0")
        )
        average_loss = total_loss / losing_trades if losing_trades > 0 else Decimal("0")

        profit_factor = float(total_profit / total_loss) if total_loss > 0 else 0.0

        # Calculate max drawdown
        cumulative_pl = [Decimal("0")]
        for s in completed_signals:
            if s.profit_loss is not None:
                cumulative_pl.append(cumulative_pl[-1] + s.profit_loss)

        peak = cumulative_pl[0]
        max_drawdown = Decimal("0")
        for value in cumulative_pl[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak if peak > 0 else Decimal("0")
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        # Calculate Sharpe ratio (simplified, using daily returns)
        if len(completed_signals) > 1:
            returns = [
                float(s.profit_loss)
                for s in completed_signals
                if s.profit_loss is not None
            ]
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0.001
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=float(max_drawdown),
            sharpe_ratio=sharpe_ratio,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            average_win=average_win,
            average_loss=average_loss,
        )


# --- Data Validator ---
class DataValidator:
    """Validates market data before analysis."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.validation_config = config.get("data_validation", {})

    def validate_dataframe(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Validate a DataFrame of market data."""
        if df.empty:
            self.logger.error(
                f"{NEON_RED}Empty DataFrame for {symbol} {interval}{RESET}"
            )
            return False

        # Check minimum data points
        min_data_points = self.validation_config.get("min_data_points", 50)
        if len(df) < min_data_points:
            self.logger.error(
                f"{NEON_RED}Insufficient data points for {symbol} {interval}: {len(df)} < {min_data_points}{RESET}"
            )
            return False

        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(
                f"{NEON_RED}Missing required columns for {symbol} {interval}: {missing_columns}{RESET}"
            )
            return False

        # Check for NaN values
        if df[required_columns].isnull().any().any():
            self.logger.warning(
                f"{NEON_YELLOW}NaN values found in {symbol} {interval} data{RESET}"
            )
            # Fill NaN values with previous values
            df.ffill(inplace=True)
            # If there are still NaN values (at the beginning), drop those rows
            df.dropna(how='any', inplace=True)

        # Check data age - Fixed timezone issue
        max_data_age_minutes = self.validation_config.get("max_data_age_minutes", 60)
        if "start_time" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
                df['start_time'] = pd.to_datetime(df['start_time'])
            latest_timestamp = df["start_time"].max()

            # Make both timestamps timezone-aware or both timezone-naive
            # Bybit klines are in UTC
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo("UTC"))

            # If latest_timestamp is naive, localize it to UTC
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=ZoneInfo("UTC"))

            data_age = (now - latest_timestamp).total_seconds() / 60
            if data_age > max_data_age_minutes:
                self.logger.warning(
                    f"{NEON_YELLOW}Data for {symbol} {interval} is stale: {data_age:.1f} minutes old{RESET}"
                )

        # Check for price anomalies
        price_deviation_threshold = self.validation_config.get(
            "price_deviation_threshold", 0.1
        )
        if "close" in df.columns:
            prices = df["close"].values
            price_changes = np.abs(np.diff(prices) / prices[:-1])
            max_deviation = np.max(price_changes)
            if max_deviation > price_deviation_threshold:
                self.logger.warning(
                    f"{NEON_YELLOW}Large price deviation detected in {symbol} {interval}: {max_deviation:.2%}{RESET}"
                )

        return True


# --- Market Regime Detector ---
class MarketRegimeDetector:
    """Detects market regimes (bullish, bearish, sideways, volatile)."""

    def __init__(self, df: pd.DataFrame, config: dict, logger: logging.Logger):
        self.df = df.copy()
        self.config = config
        self.logger = logger
        self.atr_period = config.get("atr_period", 14)
        self.regime_window = config.get("regime_window", 20)

    def detect_regime(self) -> MarketRegime:
        """Detect the current market regime."""
        if len(self.df) < self.regime_window:
            return MarketRegime.UNKNOWN

        # Calculate indicators for regime detection
        close_prices = self.df["close"].values
        atr = self._calculate_atr()

        # Calculate price change over the window
        price_change = (
            close_prices[-1] - close_prices[-self.regime_window]
        ) / close_prices[-self.regime_window]

        # Calculate volatility (normalized ATR)
        volatility = atr / close_prices[-1] if close_prices[-1] > 0 else 0

        # Determine regime based on price change and volatility
        volatility_threshold = self.config.get("volatility_threshold", 0.02)
        trend_threshold = self.config.get("trend_threshold", 0.05)

        if volatility > volatility_threshold:
            return MarketRegime.VOLATILE
        elif price_change > trend_threshold:
            return MarketRegime.BULLISH
        elif price_change < -trend_threshold:
            return MarketRegime.BEARISH
        else:
            return MarketRegime.SIDEWAYS

    def _calculate_atr(self) -> float:
        """Calculate Average True Range."""
        high = self.df["high"].values
        low = self.df["low"].values
        close = self.df["close"].values

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])

        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = (np.mean(tr[-self.atr_period:]) if len(tr) >= self.atr_period and np.any(tr[-self.atr_period:] > 0) else np.mean(tr))

        return atr


# --- Risk Manager ---
class RiskManager:
    """Manages trading risk and position sizing with high-precision Decimal math."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.risk_config = config.get("risk_management", {})
        self.circuit_breaker_config = self.risk_config.get("circuit_breaker", {})
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_end_time = None

    def calculate_position_size(
        self,
        price: Decimal,
        stop_loss: Decimal,
        account_balance: Decimal,
        instrument_info: dict = None,
    ) -> Decimal:
        """Calculate position size based on risk rules, exchange filters, and available margin."""
        risk_per_trade = Decimal(str(self.risk_config.get("risk_per_trade", 0.02)))
        max_pos_pct = Decimal(str(self.risk_config.get("max_position_size", 0.1)))
        leverage = Decimal(str(self.risk_config.get("leverage", 10)))
        min_order_value = Decimal(str(self.risk_config.get("min_order_value", 5.0)))

        # 1. Equity Risk: How much we are willing to lose on this trade
        risk_amount = account_balance * risk_per_trade
        price_risk = abs(price - stop_loss)

        if price_risk == 0:
            self.logger.warning(
                f"{NEON_YELLOW}Stop loss matches entry price. Risk cannot be calculated.{RESET}"
            )
            return Decimal("0")

        # Qty based on risk amount
        pos_size_risk = risk_amount / price_risk

        # 2. Capital Cap: Maximum % of portfolio allowed in one trade
        max_pos_value = account_balance * max_pos_pct
        pos_size_cap = max_pos_value / price

        # 3. Leverage Constraint: Max qty allowed by buying power
        # We use 95% of max power to account for fees and slippage
        max_leverage_qty = (account_balance * leverage * Decimal("0.95")) / price

        # Final raw size is the minimum of all constraints
        final_size = min(pos_size_risk, pos_size_cap, max_leverage_qty)

        # 4. Exchange Filter Alignment (qtyStep, minOrderQty)
        if instrument_info:
            lot_filter = instrument_info.get("lotSizeFilter", {})
            qty_step = Decimal(lot_filter.get("qtyStep", "0.001"))
            min_qty = Decimal(lot_filter.get("minOrderQty", "0"))

            # Align with qtyStep
            final_size = (final_size / qty_step).quantize(
                Decimal("1"), rounding=decimal.ROUND_DOWN
            ) * qty_step

            if final_size < min_qty:
                if min_qty <= max_leverage_qty:
                    final_size = min_qty
                else:
                    self.logger.warning(
                        f"{NEON_RED}Calculated size {final_size} below min {min_qty} and exceeds leverage cap.{RESET}"
                    )
                    return Decimal("0")

        # 5. Min Order Value Enforcement (e.g. 5 USDT)
        if final_size * price < min_order_value:
            required_qty = min_order_value / price
            if instrument_info:
                qty_step = Decimal(
                    instrument_info.get("lotSizeFilter", {}).get("qtyStep", "0.001")
                )
                required_qty = (required_qty / qty_step).quantize(
                    Decimal("1"), rounding=decimal.ROUND_UP
                ) * qty_step

            if required_qty <= max_leverage_qty:
                final_size = required_qty
            else:
                self.logger.warning(
                    f"{NEON_RED}Portfolio too small to meet minimum order value ${min_order_value}.{RESET}"
                )
                return Decimal("0")

        self.logger.info(
            f"{NEON_BLUE}Position Sizing Engine:{RESET}\n"
            f"  Balance: ${float(account_balance):.2f} | Risk Amt: ${float(risk_amount):.2f}\n"
            f"  Constraints -> Risk: {float(pos_size_risk):.4f} | Cap: {float(pos_size_cap):.4f} | Final: {float(final_size):.4f}"
        )

        return final_size

    def check_circuit_breaker(self) -> bool:
        """Monitor for consecutive losses and pause trading if threshold reached."""
        if not self.circuit_breaker_config.get("enabled", False):
            return False

        if self.circuit_breaker_active:
            if (
                self.circuit_breaker_end_time
                and time.time() > self.circuit_breaker_end_time
            ):
                self.circuit_breaker_active = False
                self.consecutive_losses = 0
                self.logger.info(
                    f"{NEON_GREEN}Circuit breaker cooldown complete. Resuming operations.{RESET}"
                )
                return False
            return True

        if self.consecutive_losses >= self.circuit_breaker_config.get(
            "max_consecutive_losses", 5
        ):
            self.circuit_breaker_active = True
            cooldown = self.circuit_breaker_config.get("cooldown_period_minutes", 60)
            self.circuit_breaker_end_time = time.time() + (cooldown * 60)
            self.logger.error(
                f"{NEON_RED}CIRCUIT BREAKER: {self.consecutive_losses} losses. Cooldown: {cooldown}m{RESET}"
            )
            return True

        return False

    def update_trade_result(self, profit_loss: Decimal) -> None:
        """Update tracker with trade result."""
        if profit_loss < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def check_daily_loss_limit(
        self, current_loss: Decimal, account_balance: Decimal
    ) -> bool:
        """Enforce maximum daily drawdown."""
        limit_pct = Decimal(str(self.risk_config.get("max_daily_loss", 0.05)))
        limit_amt = account_balance * limit_pct
        if abs(current_loss) >= limit_amt:
            self.logger.error(
                f"{NEON_RED}DAILY LOSS LIMIT HIT: ${float(current_loss):.2f} >= ${float(limit_amt):.2f}{RESET}"
            )
            return True
        return False

    def check_drawdown_limit(self, current_drawdown: float) -> bool:
        """Enforce absolute strategy drawdown limit."""
        limit = self.risk_config.get("max_drawdown", 0.15)
        if current_drawdown >= limit:
            self.logger.error(
                f"{NEON_RED}MAX STRATEGY DRAWDOWN HIT: {current_drawdown:.2%} >= {limit:.2%}{RESET}"
            )
            return True
        return False


# --- API Client ---
class APIClient:
    """Hardened Bybit V5 API Client with Tor support and precise formatting."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        logger: logging.Logger,
        proxy_config: dict = None,
        paper_mode: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.logger = logger
        self.paper_mode = paper_mode
        self.recv_window = "5000"
        self.session = requests.Session()

        if proxy_config and proxy_config.get("enabled"):
            self.session.proxies = {
                "http": proxy_config["url"],
                "https": proxy_config["url"],
            }
            self.logger.info(
                f"{NEON_CYAN}Proxy routing active: {proxy_config['url']}{RESET}"
            )

        self.rate_limiter = RateLimiter(logger)
        self.instrument_info_cache = {}
        self._fee_cache = {}

    async def make_request_async(self, method: str, endpoint: str, params: dict = None):
        """Execute an asynchronous signed request."""
        async with aiohttp.ClientSession():
            # Implement similar logic as synchronous with retries
            # For now, this is a placeholder as per updates.md
            pass

    def fetch_instrument_info(self, symbol: str) -> dict | None:
        """Cache and return exchange filters for precision management."""
        if symbol in self.instrument_info_cache:
            return self.instrument_info_cache[symbol]

        res = self.make_request(
            "GET",
            "/v5/market/instruments-info",
            {"category": "linear", "symbol": symbol},
        )
        if res and res.get("retCode") == 0:
            for item in res["result"].get("list", []):
                if item["symbol"] == symbol:
                    self.instrument_info_cache[symbol] = item
                    return item
        return None

    def format_quantity(self, symbol: str, qty: Decimal) -> str:
        """Apply qtyStep filter."""
        info = self.fetch_instrument_info(symbol)
        step = (
            Decimal(info.get("lotSizeFilter", {}).get("qtyStep", "0.001"))
            if info
            else Decimal("0.001")
        )
        val = (qty / step).quantize(Decimal("1"), rounding=decimal.ROUND_DOWN) * step
        prec = len(str(step).split(".")[1]) if "." in str(step) else 0
        return f"{val:.{prec}f}"

    def format_price(self, symbol: str, price: Decimal) -> str:
        """Apply tickSize filter."""
        info = self.fetch_instrument_info(symbol)
        tick = (
            Decimal(info.get("priceFilter", {}).get("tickSize", "0.00001"))
            if info
            else Decimal("0.00001")
        )
        val = (price / tick).quantize(Decimal("1"), rounding=decimal.ROUND_HALF_UP) * tick
        prec = len(str(tick).split(".")[1]) if "." in str(tick) else 0
        return f"{val:.{prec}f}"

    def generate_signature(self, timestamp: str, payload: str) -> str:
        """Generate V5 HMAC signature."""
        param_str = timestamp + self.api_key + self.recv_window + payload
        return hmac.new(
            self.api_secret.encode("utf-8"), param_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def make_request(
        self, method: str, endpoint: str, params: dict = None
    ) -> dict | None:
        """Execute a signed request with automatic retries and rate limiting."""
        self.rate_limiter.wait_if_needed()
        params = params or {}
        timestamp = str(int(time.time() * 1000))

        if method == "GET":
            payload = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            url = (
                f"{self.base_url}{endpoint}?{payload}"
                if payload
                else f"{self.base_url}{endpoint}"
            )
            signature = self.generate_signature(timestamp, payload)
            body = None
        else:
            url = f"{self.base_url}{endpoint}"
            body = json.dumps(params, separators=(",", ":"))
            signature = self.generate_signature(timestamp, body)

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "Content-Type": "application/json",
        }

        for retry in range(MAX_API_RETRIES):
            try:
                response = self.session.request(
                    method, url, headers=headers, data=body, timeout=10
                )
                if 'X-RateLimit-Remaining' in response.headers:
                    self.logger.debug(f"API Rate Limit Remaining: {response.headers['X-RateLimit-Remaining']}")
                if response.status_code == 429:
                    self.logger.warning(
                        f"{NEON_YELLOW}Rate limit hit, backing off...{RESET}"
                    )
                    time.sleep(5)
                    continue
                if response.status_code == 403:
                    self.logger.error(f"{NEON_RED}403 Forbidden. IP might be blocked.{RESET}")
                    break
                response.raise_for_status()
                return response.json()
            except Exception as e:
                self.logger.error(f"{NEON_RED}Request Error ({retry + 1}): {e}{RESET}")
                time.sleep(RETRY_DELAY_SECONDS * (retry + 1))
        return None

    def fetch_fee_rates(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Fetch maker and taker fee rates with caching."""
        if hasattr(self, '_fee_cache') and symbol in self._fee_cache:
            return self._fee_cache[symbol]
        res = self.make_request('GET', '/v5/contract/fee-rate', {'symbol': symbol})
        if res and res.get('retCode') == 0:
            maker = Decimal(str(res['result'].get('makerFeeRate', '0')))
            taker = Decimal(str(res['result'].get('takerFeeRate', '0')))
            self._fee_cache[symbol] = (maker, taker)
            return maker, taker
        return Decimal('0'), Decimal('0')

    def fetch_balance(self, coin: str = "USDT") -> Decimal:
        """Fetch available balance with UNIFIED/CONTRACT auto-detection."""
        for acc in ["UNIFIED", "CONTRACT"]:
            res = self.make_request(
                "GET", "/v5/account/wallet-balance", {"accountType": acc, "coin": coin}
            )
            if res and res.get("retCode") == 0:
                coins = res["result"].get("list", [{}])[0].get("coin", [])
                for c in coins:
                    if c.get("coin") == coin:
                        return Decimal(
                            str(
                                c.get("availableToWithdraw")
                                or c.get("walletBalance")
                                or "0"
                            )
                        )
        return Decimal("0")

    def fetch_current_price(self, symbol: str) -> Decimal | None:
        """Fetch last traded price."""
        res = self.make_request(
            "GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol}
        )
        if res and res.get("retCode") == 0:
            for t in res["result"].get("list", []):
                if t["symbol"] == symbol:
                    return Decimal(t["lastPrice"])
        return None

    def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Fetch historical candlestick data. Tries live API first, then falls back to local CSV."""
        res = self.make_request(
            "GET",
            "/v5/market/kline",
            {
                "symbol": symbol,
                "interval": interval,
                "limit": str(limit),
                "category": "linear",
            },
        )
        if res and res.get("retCode") == 0:
            cols = ["start_time", "open", "high", "low", "close", "volume", "turnover"]
            df = pd.DataFrame(res["result"]["list"], columns=cols)
            df["start_time"] = pd.to_datetime(
                pd.to_numeric(df["start_time"]), unit="ms"
            ).dt.tz_localize("UTC")
            for col in cols[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.sort_values("start_time").reset_index(drop=True)

        # Fallback to local CSV if API fails or is blocked
        possible_paths = [
            f"../Gbotx/data/{symbol}-{interval}m.csv",
            f"../Gbotx/data/{symbol}-{interval}.csv",
            f"Gbotx/data/{symbol}-{interval}.csv",
            f"{symbol}-{interval}.csv"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"{NEON_CYAN}Loading offline klines from {path}{RESET}")
                df = pd.read_csv(path)
                if 'timestamp' in df.columns:
                    df.rename(columns={'timestamp': 'start_time'}, inplace=True)
                if 'start_time' in df.columns:
                    df['start_time'] = pd.to_datetime(df['start_time'])
                    if df['start_time'].dt.tz is None:
                        df['start_time'] = df['start_time'].dt.tz_localize('UTC')
                return df
        return pd.DataFrame()

    def fetch_order_book(self, symbol: str, limit: int = 50) -> dict | None:
        """Fetch L2 depth data."""
        res = self.make_request(
            "GET",
            "/v5/market/orderbook",
            {"symbol": symbol, "limit": str(limit), "category": "linear"},
        )
        return res["result"] if res and res.get("retCode") == 0 else None

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: Decimal,
        price: Decimal = None,
        stop_loss: Decimal = None,
        take_profit: Decimal = None,
        is_close: bool = False,
    ) -> dict | None:
        """Place a risk-managed order with TP/SL."""
        if self.paper_mode:
            self.logger.info(
                f"{NEON_YELLOW}PAPER ORDER: {side} {qty} {symbol} @ {price or 'Market'}{RESET}"
            )
            return {"retCode": 0, "result": {"orderId": f"paper_{int(time.time())}"}}

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": order_type,
            "qty": self.format_quantity(symbol, qty),
            "timeInForce": "GTC",
            "reduceOnly": is_close,
        }
        if price:
            params["price"] = self.format_price(symbol, price)
        if stop_loss:
            params["stopLoss"] = self.format_price(symbol, stop_loss)
        if take_profit:
            params["takeProfit"] = self.format_price(symbol, take_profit)

        return self.make_request("POST", "/v5/order/create", params)


# --- Rate Limiter ---
class RateLimiter:
    """Implements rate limiting for API requests."""

    def __init__(self, logger: logging.Logger, max_requests_per_minute: int = 100):
        self.logger = logger
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Wait if the rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [
                req_time for req_time in self.requests if now - req_time < 60
            ]

            if len(self.requests) >= self.max_requests_per_minute:
                # Calculate how long to wait
                oldest_request = min(self.requests)
                wait_time = 60 - (now - oldest_request)
                if wait_time > 0:
                    self.logger.warning(
                        f"{NEON_YELLOW}Rate limit reached, waiting {wait_time:.1f} seconds{RESET}"
                    )
                    time.sleep(wait_time)
                    # Remove requests older than 1 minute
                    now = time.time()
                    self.requests = [
                        req_time for req_time in self.requests if now - req_time < 60
                    ]

            # Record this request
            self.requests.append(now)


# --- Indicator Calculator ---
class IndicatorCalculator:
    """
    High-performance technical analysis suite for trading signal generation.
    Utilizes vectorized operations via NumPy and SciPy for maximum efficiency.
    """

    def __init__(self, df: pd.DataFrame, config: dict, logger: logging.Logger):
        self.df = df.copy()  # Work on a copy to avoid modifying original DataFrame
        self.config = config
        self.logger = logger
        self.indicator_values: dict[str, Any] = {}
        self.atr_value: float = 0.0
        self._validate_data()

    def _validate_data(self) -> None:
        """Ensures the input DataFrame meets the minimum requirements for analysis."""
        if self.df.empty:
            raise ValueError("DataFrame is empty")

        min_data_points = self.config.get("data_validation", {}).get(
            "min_data_points", 50
        )
        if len(self.df) < min_data_points:
            self.logger.warning(
                f"{NEON_YELLOW}Insufficient data points: {len(self.df)} < {min_data_points}{RESET}"
            )

        # Check for required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for NaN values
        if self.df[required_columns].isnull().any().any():
            self.logger.warning(
                f"{NEON_YELLOW}DataFrame contains NaN values. Applying forward-fill.{RESET}"
            )
            self.df[required_columns] = self.df[required_columns].ffill().bfill()

    def _safe_series_operation(
        self, column: str, operation: str, window: int = None, series: pd.Series = None
    ) -> pd.Series:
        """Vectorized wrapper for common rolling and smoothing operations."""
        data_series = series if series is not None else self.df.get(column)
        if data_series is None or data_series.empty:
            return pd.Series(dtype=float)

        try:
            ops = {
                "sma": lambda x: x.rolling(window=window).mean(),
                "ema": lambda x: x.ewm(span=window, adjust=False).mean(),
                "max": lambda x: x.rolling(window=window).max(),
                "min": lambda x: x.rolling(window=window).min(),
                "diff": lambda x: x.diff(window),
                "cumsum": lambda x: x.cumsum(),
                "std": lambda x: x.rolling(window=window).std(),
                "var": lambda x: x.rolling(window=window).var(),
                "abs_diff_mean": lambda x: x.rolling(window=window).apply(
                    lambda s: np.abs(s - s.mean()).mean(), raw=True
                ),
            }
            if operation in ops:
                return ops[operation](data_series)
            self.logger.error(f"{NEON_RED}Unsupported operation: {operation}{RESET}")
            return pd.Series(dtype=float)
        except Exception as e:
            self.logger.error(
                f"{NEON_RED}Indicator Error ({operation}) on {column}: {e}{RESET}"
            )
            return pd.Series(dtype=float)

    def calculate_sma(self, window: int, series: pd.Series = None) -> pd.Series:
        """Calculates Simple Moving Average (SMA)."""
        return self._safe_series_operation("close", "sma", window, series)

    def calculate_ema(self, window: int, series: pd.Series = None) -> pd.Series:
        """Calculates Exponential Moving Average (EMA)."""
        return self._safe_series_operation("close", "ema", window, series)

    def calculate_wma(self, window: int, series: pd.Series = None) -> pd.Series:
        """Calculates Weighted Moving Average (WMA) using NumPy dot product."""
        data = series if series is not None else self.df["close"]
        weights = np.arange(1, window + 1)
        return data.rolling(window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculates the Average True Range (ATR)."""
        high_low = self.df["high"] - self.df["low"]
        high_close = abs(self.df["high"] - self.df["close"].shift())
        low_close = abs(self.df["low"] - self.df["close"].shift())

        # True Range is the maximum of the three
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return self._safe_series_operation(
            None, "ema", window, tr
        )  # Use EMA for ATR for smoothing

    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = self._safe_series_operation(None, "ema", window, gain)
        avg_loss = self._safe_series_operation(None, "ema", window, loss)

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)

    def calculate_stoch_rsi(
        self,
        rsi_window: int = 14,
        stoch_window: int = 14,
        k_window: int = 3,
        d_window: int = 3,
    ) -> pd.DataFrame:
        """Calculates Stochastic RSI (%K and %D lines)."""
        rsi = self.calculate_rsi(window=rsi_window)
        if rsi.empty:
            return pd.DataFrame()

        # Calculate StochRSI
        stoch_rsi = (
            rsi - self._safe_series_operation(None, "min", stoch_window, rsi)
        ) / (
            self._safe_series_operation(None, "max", stoch_window, rsi)
            - self._safe_series_operation(None, "min", stoch_window, rsi)
        )

        # Handle division by zero for StochRSI (if max == min)
        stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan).fillna(0)
        k_line = (
            self._safe_series_operation(None, "sma", k_window, stoch_rsi) * 100
        )  # Scale to 0-100
        d_line = self._safe_series_operation(
            None, "sma", d_window, k_line
        )  # Signal line for %K

        return pd.DataFrame(
            {"stoch_rsi": stoch_rsi * 100, "k": k_line, "d": d_line}
        )  # Return StochRSI also scaled

    def calculate_stochastic_oscillator(self) -> pd.DataFrame:
        """Calculates the Stochastic Oscillator (%K and %D lines)."""
        k_period = self.config["indicator_periods"]["stoch_osc_k"]
        d_period = self.config["indicator_periods"]["stoch_osc_d"]

        highest_high = self._safe_series_operation("high", "max", k_period)
        lowest_low = self._safe_series_operation("low", "min", k_period)

        # Calculate %K
        k_line = (self.df["close"] - lowest_low) / (highest_high - lowest_low) * 100
        k_line = k_line.replace([np.inf, -np.inf], np.nan).fillna(
            0
        )  # Handle division by zero

        # Calculate %D (SMA of %K)
        d_line = self._safe_series_operation(None, "sma", d_period, k_line)

        return pd.DataFrame({"k": k_line, "d": d_line})

    def calculate_macd(self) -> pd.DataFrame:
        """Calculates Moving Average Convergence Divergence (MACD)."""
        ma_short = self._safe_series_operation("close", "ema", 12)
        ma_long = self._safe_series_operation("close", "ema", 26)
        macd = ma_short - ma_long
        signal = self._safe_series_operation(None, "ema", 9, macd)
        histogram = macd - signal

        return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram})

    def calculate_cci(self, window: int = 20, constant: float = 0.015) -> pd.Series:
        """Calculates the Commodity Channel Index (CCI)."""
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        sma_typical_price = self._safe_series_operation(
            None, "sma", window, typical_price
        )
        mean_deviation = self._safe_series_operation(
            None, "abs_diff_mean", window, typical_price
        )

        # Avoid division by zero
        cci = (typical_price - sma_typical_price) / (constant * mean_deviation)
        return cci.replace([np.inf, -np.inf], np.nan)

    def calculate_williams_r(self, window: int = 14) -> pd.Series:
        """Calculates the Williams %R indicator."""
        highest_high = self._safe_series_operation("high", "max", window)
        lowest_low = self._safe_series_operation("low", "min", window)

        # Avoid division by zero
        denominator = highest_high - lowest_low
        wr = ((highest_high - self.df["close"]) / denominator) * -100
        return wr.replace([np.inf, -np.inf], np.nan)

    def calculate_mfi(self, window: int = 14) -> pd.Series:
        """Calculates the Money Flow Index (MFI)."""
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        raw_money_flow = typical_price * self.df["volume"]

        # Calculate positive and negative money flow
        money_flow_direction = typical_price.diff()
        positive_flow = raw_money_flow.where(money_flow_direction > 0, 0)
        negative_flow = raw_money_flow.where(money_flow_direction < 0, 0)

        # Calculate sums over the window
        positive_mf = (
            self._safe_series_operation(None, "sma", window, positive_flow) * window
        )  # sum not mean
        negative_mf = (
            self._safe_series_operation(None, "sma", window, negative_flow) * window
        )  # sum not mean

        # Avoid division by zero
        money_ratio = positive_mf / negative_mf.replace(
            0, np.nan
        )  # Replace 0 with NaN to handle division by zero
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi.replace([np.inf, -np.inf], np.nan).fillna(
            0
        )  # Fill NaN from division by zero with 0

    def calculate_adx_series(self, window: int = 14) -> pd.DataFrame:
        """Calculates the Average Directional Index (ADX) series."""
        # True Range
        tr = pd.concat(
            [
                self.df["high"] - self.df["low"],
                abs(self.df["high"] - self.df["close"].shift()),
                abs(self.df["low"] - self.df["close"].shift()),
            ],
            axis=1,
        ).max(axis=1)

        # Directional Movement
        df_adx = pd.DataFrame(index=self.df.index)
        df_adx["+DM"] = (
            (self.df["high"] - self.df["high"].shift())
            > (self.df["low"].shift() - self.df["low"])
        ) & ((self.df["high"] - self.df["high"].shift()) > 0) * (
            self.df["high"] - self.df["high"].shift()
        )
        df_adx["-DM"] = (
            (self.df["low"].shift() - self.df["low"])
            > (self.df["high"] - self.df["high"].shift())
        ) & ((self.df["low"].shift() - self.df["low"]) > 0) * (
            self.df["low"].shift() - self.df["low"]
        )

        # Smoothed True Range and Directional Movement (using EMA)
        df_adx["TR_ema"] = self._safe_series_operation(None, "ema", window, tr)
        df_adx["+DM_ema"] = self._safe_series_operation(
            None, "ema", window, df_adx["+DM"]
        )
        df_adx["-DM_ema"] = self._safe_series_operation(
            None, "ema", window, df_adx["-DM"]
        )

        # Directional Indicators
        df_adx["+DI"] = 100 * (df_adx["+DM_ema"] / df_adx["TR_ema"].replace(0, np.nan))
        df_adx["-DI"] = 100 * (df_adx["-DM_ema"] / df_adx["TR_ema"].replace(0, np.nan))

        # Directional Movement Index (DX)
        df_adx["DX"] = (
            100
            * abs(df_adx["+DI"] - df_adx["-DI"])
            / (df_adx["+DI"] + df_adx["-DI"]).replace(0, np.nan)
        )

        # Average Directional Index (ADX)
        df_adx["ADX"] = self._safe_series_operation(None, "ema", window, df_adx["DX"])
        return df_adx

    def calculate_adx(self, window: int = 14) -> dict[str, float]:
        """Calculates the Average Directional Index (ADX)."""
        df_adx = self.calculate_adx_series(window)
        adx_value = df_adx["ADX"].iloc[-1] if not df_adx["ADX"].empty else 0.0

        return {
            "adx": float(adx_value) if not pd.isna(adx_value) else 0.0,
            "plus_di": float(df_adx["+DI"].iloc[-1]) if not df_adx["+DI"].empty else 0.0,
            "minus_di": float(df_adx["-DI"].iloc[-1]) if not df_adx["-DI"].empty else 0.0
        }

    def calculate_obv(self) -> pd.Series:
        """Calculates On-Balance Volume (OBV)."""
        return (np.sign(self.df["close"].diff().fillna(0)) * self.df["volume"]).cumsum()

    def calculate_adi(self) -> pd.Series:
        """Calculates Accumulation/Distribution Index (ADI)."""
        clv = (
            (self.df["close"] - self.df["low"]) - (self.df["high"] - self.df["close"])
        ) / (self.df["high"] - self.df["low"]).replace(0, np.nan)
        return (clv.fillna(0) * self.df["volume"]).cumsum()

    # --- Enhanced Calculations (EC5) ---
    def calculate_supersmoother(self, window: int = 10) -> pd.Series:
        """Calculates Ehlers SuperSmoother filter using Scipy lfilter for high performance."""
        close = self.df["close"].values
        a1 = np.exp(-1.414 * np.pi / window)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / window)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        b = np.array([c1 / 2, c1 / 2])
        a = np.array([1, -c2, -c3])
        return pd.Series(lfilter(b, a, close), index=self.df.index)

    def calculate_ehlers_fisher(self, window: int = 10) -> pd.Series:
        """Calculates Ehlers Fisher Transform for trend turning points with optimized loop."""
        high, low = self.df["high"], self.df["low"]
        hl2 = (high + low) / 2
        max_h = hl2.rolling(window=window).max()
        min_l = hl2.rolling(window=window).min()
        value = np.zeros_like(hl2)
        fisher = np.zeros_like(hl2)

        for i in range(len(hl2)):
            if i < window:
                continue
            denom = max_h.iloc[i] - min_l.iloc[i]
            val = (
                0.66 * ((hl2.iloc[i] - min_l.iloc[i]) / (denom if denom != 0 else 1e-10) - 0.5)
                + 0.67 * value[i - 1]
            )
            val = max(min(val, 0.999), -0.999)
            value[i] = val
            fisher[i] = 0.5 * np.log((1 + val) / (1 - val)) + 0.5 * fisher[i - 1]
        return pd.Series(fisher, index=self.df.index)

    def calculate_awesome_oscillator(self) -> pd.Series:
        """Calculates the Awesome Oscillator."""
        median_price = (self.df["high"] + self.df["low"]) / 2
        ao = self._safe_series_operation(None, "sma", 5, median_price) - self._safe_series_operation(None, "sma", 34, median_price)
        return ao

    def calculate_vortex(self, window: int = 14) -> pd.DataFrame:
        """Calculates the Vortex Indicator (VI+ and VI-)."""
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        vmp = abs(high - low.shift())
        vmm = abs(low - high.shift())

        str_sum = tr.rolling(window).sum()
        svmp = vmp.rolling(window).sum()
        svmm = vmm.rolling(window).sum()

        vi_plus = svmp / str_sum
        vi_minus = svmm / str_sum
        return pd.DataFrame({"vi_plus": vi_plus, "vi_minus": vi_minus})

    def calculate_fve(self, period: int = 22) -> pd.Series:
        """Calculates Finite Volume Element (FVE)."""
        tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        mf = self.df["volume"] * ((self.df["close"] - (self.df["high"] + self.df["low"]) / 2) + tp.diff())
        fve = (mf.rolling(period).sum() / self.df["volume"].rolling(period).sum().replace(0, np.nan)) * 100
        return fve.fillna(0)

    def calculate_laguerre_rsi(self, gamma: float = 0.5) -> pd.Series:
        """Calculates Laguerre RSI."""
        l0, l1, l2, l3 = 0.0, 0.0, 0.0, 0.0
        lrsi = []
        for i in range(len(self.df)):
            c = self.df["close"].iloc[i]
            l0 = (1 - gamma) * c + gamma * l0
            l1 = -gamma * l0 + l0 + gamma * l1
            l2 = -gamma * l1 + l1 + gamma * l2
            l3 = -gamma * l2 + l2 + gamma * l3
            cu = (l0 - l1 if l0 >= l1 else 0) + (l1 - l2 if l1 >= l2 else 0) + (l2 - l3 if l2 >= l3 else 0)
            cd = (l1 - l0 if l1 > l0 else 0) + (l2 - l1 if l2 > l1 else 0) + (l3 - l2 if l3 > l2 else 0)
            lrsi.append(cu / (cu + cd) if (cu + cd) != 0 else 0)
        return pd.Series(lrsi, index=self.df.index)

    def calculate_supertrend(self, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculate Supertrend indicator.

        Args:
            period (int): ATR period, default 10
            multiplier (float): ATR multiplier, default 3.0

        Returns:
            pd.Series: Supertrend values
            pd.Series: Supertrend direction (1=bullish, -1=bearish)
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # Initialize supertrend
        supertrend = pd.Series(index=self.df.index, dtype=float)
        direction = pd.Series(0, index=self.df.index)

        prev_upper = upper_band.iloc[period-1] if period > 0 else upper_band.iloc[0]
        prev_lower = lower_band.iloc[period-1] if period > 0 else lower_band.iloc[0]
        prev_direction = 1

        # Fill initial NaN values from ATR calculation
        for i in range(period):
            supertrend.iloc[i] = hl2.iloc[i]
            direction.iloc[i] = 1

        for i in range(period, len(self.df)):
            curr_close = close.iloc[i]
            prev_close = close.iloc[i-1]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]

            # Adjust Final Upper Band
            if curr_upper < prev_upper or prev_close > prev_upper:
                curr_upper = curr_upper
            else:
                curr_upper = prev_upper

            # Adjust Final Lower Band
            if curr_lower > prev_lower or prev_close < prev_lower:
                curr_lower = curr_lower
            else:
                curr_lower = prev_lower

            # Determine direction
            if prev_direction == 1:
                curr_direction = 1 if curr_close >= curr_lower else -1
            else:
                curr_direction = 1 if curr_close > curr_upper else -1

            # Calculate supertrend
            if curr_direction == 1:
                curr_supertrend = curr_lower
            else:
                curr_supertrend = curr_upper

            supertrend.iloc[i] = curr_supertrend
            direction.iloc[i] = curr_direction

            prev_upper = curr_upper
            prev_lower = curr_lower
            prev_direction = curr_direction

        return pd.DataFrame({"supertrend": supertrend, "direction": direction})

    def calculate_cmo(self, period: int = 14) -> pd.Series:
        """Calculates Chande Momentum Oscillator (CMO)."""
        close_diff = self.df["close"].diff()
        ups = close_diff.where(close_diff > 0, 0).rolling(period).sum()
        downs = abs(close_diff.where(close_diff < 0, 0)).rolling(period).sum()
        cmo = 100 * (ups - downs) / (ups + downs).replace(0, np.nan)
        return cmo.fillna(0)

    def calculate_stc(self, period: int = 10, fast: int = 23, slow: int = 50) -> pd.Series:
        """Calculates Schaff Trend Cycle (STC)."""
        macd = self.calculate_ema(fast) - self.calculate_ema(slow)

        def calculate_stoch(series, window):
            low = series.rolling(window).min()
            high = series.rolling(window).max()
            return 100 * (series - low) / (high - low).replace(0, np.nan)

        stoch_k = calculate_stoch(macd, period).fillna(0)
        stoch_d = stoch_k.rolling(3).mean().fillna(0)
        stc = stoch_d.rolling(3).mean().fillna(0)
        return stc

    def calculate_bollinger_bands(self, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculates Bollinger Bands."""
        middle = self.calculate_sma(window)
        std = self._safe_series_operation("close", "std", window)
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})

    def calculate_psar(self, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
        """Calculates Parabolic SAR."""
        high = self.df["high"]
        low = self.df["low"]
        psar = pd.Series(0.0, index=self.df.index)
        bull = True
        af = step
        hp = high.iloc[0]
        lp = low.iloc[0]
        psar.iloc[0] = low.iloc[0]

        for i in range(1, len(self.df)):
            prev_psar = psar.iloc[i-1]
            if bull:
                psar.iloc[i] = prev_psar + af * (hp - prev_psar)
            else:
                psar.iloc[i] = prev_psar + af * (lp - prev_psar)

            reverse = False
            if bull:
                if low.iloc[i] < psar.iloc[i]:
                    bull = False
                    reverse = True
                    psar.iloc[i] = hp
                    lp = low.iloc[i]
                    af = step
            else:
                if high.iloc[i] > psar.iloc[i]:
                    bull = True
                    reverse = True
                    psar.iloc[i] = lp
                    hp = high.iloc[i]
                    af = step

            if not reverse:
                if bull:
                    if high.iloc[i] > hp:
                        hp = high.iloc[i]
                        af = min(af + step, max_step)
                    if high.iloc[i-1] > psar.iloc[i]:
                        psar.iloc[i] = high.iloc[i-1]
                    if high.iloc[max(0, i-2)] > psar.iloc[i]:
                        psar.iloc[i] = high.iloc[max(0, i-2)]
                else:
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
                        af = min(af + step, max_step)
                    if low.iloc[i-1] < psar.iloc[i]:
                        psar.iloc[i] = low.iloc[i-1]
                    if low.iloc[max(0, i-2)] < psar.iloc[i]:
                        psar.iloc[i] = low.iloc[max(0, i-2)]
        return psar

    def calculate_ichimoku_cloud(self) -> pd.DataFrame:
        """Calculates Ichimoku Cloud components."""
        high = self.df["high"]
        low = self.df["low"]
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        chikou_span = self.df["close"].shift(-26)
        return pd.DataFrame({"tenkan_sen": tenkan_sen, "kijun_sen": kijun_sen, "senkou_span_a": senkou_span_a, "senkou_span_b": senkou_span_b, "chikou_span": chikou_span})

    def calculate_keltner_channels(self, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """Calculates Keltner Channels."""
        middle = self.calculate_ema(period)
        atr = self.calculate_atr(period)
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})

    def calculate_vwap(self) -> pd.Series:
        """Calculates Volume Weighted Average Price."""
        tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        return (tp * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

    def calculate_cmf(self, period: int = 20) -> pd.Series:
        """Calculates Chaikin Money Flow."""
        mfv = ((self.df["close"] - self.df["low"]) - (self.df["high"] - self.df["close"])) / (self.df["high"] - self.df["low"]).replace(0, np.nan)
        mfv = mfv.fillna(0) * self.df["volume"]
        return mfv.rolling(period).sum() / self.df["volume"].rolling(period).sum().replace(0, np.nan)

    def calculate_emv(self, period: int = 14) -> pd.Series:
        """Calculates Ease of Movement."""
        distance = ((self.df["high"] + self.df["low"]) / 2) - ((self.df["high"].shift() + self.df["low"].shift()) / 2)
        box_ratio = (self.df["volume"] / 1000000) / (self.df["high"] - self.df["low"]).replace(0, np.nan)
        emv = distance / box_ratio.replace(0, np.nan)
        return emv.rolling(period).mean()

    def calculate_force_index(self, period: int = 13) -> pd.Series:
        """Calculates Force Index."""
        return (self.df["close"].diff() * self.df["volume"]).ewm(span=period, adjust=False).mean()

    def calculate_mass_index(self, period: int = 25) -> pd.Series:
        """Calculates Mass Index."""
        diff = self.df["high"] - self.df["low"]
        ema1 = diff.ewm(span=9, adjust=False).mean()
        ema2 = ema1.ewm(span=9, adjust=False).mean()
        ratio = ema1 / ema2.replace(0, np.nan)
        return ratio.rolling(period).sum()

    def calculate_roc(self, period: int = 12) -> pd.Series:
        """Calculates Rate of Change."""
        return self.df["close"].diff(period) / self.df["close"].shift(period) * 100

    def calculate_trix(self, period: int = 15) -> pd.Series:
        """Calculates TRIX."""
        ema1 = self.df["close"].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ema3.diff() / ema3.shift().replace(0, np.nan) * 100

    def calculate_ultimate_oscillator(self) -> pd.Series:
        """Calculates Ultimate Oscillator."""
        bp = self.df["close"] - pd.concat([self.df["low"], self.df["close"].shift()], axis=1).min(axis=1)
        tr = pd.concat([self.df["high"], self.df["close"].shift()], axis=1).max(axis=1) - pd.concat([self.df["low"], self.df["close"].shift()], axis=1).min(axis=1)
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum().replace(0, np.nan)
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum().replace(0, np.nan)
        return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

    def calculate_coppock_curve(self, period: int = 10) -> pd.Series:
        """Calculates Coppock Curve."""
        roc1 = self.calculate_roc(14)
        roc2 = self.calculate_roc(11)
        return (roc1 + roc2).ewm(span=period, adjust=False).mean()

    def calculate_donchian_channels(self, period: int = 20) -> pd.DataFrame:
        """Calculates Donchian Channels."""
        upper = self.df["high"].rolling(period).max()
        lower = self.df["low"].rolling(period).min()
        middle = (upper + lower) / 2
        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})

    def calculate_hma(self, period: int = 20) -> pd.Series:
        """Calculates Hull Moving Average."""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        wma_half = self.calculate_wma(half_period)
        wma_full = self.calculate_wma(period)
        diff = 2 * wma_half - wma_full
        return self.calculate_wma(sqrt_period, series=diff)

    def calculate_std(self, window: int = 20) -> pd.Series:
        """Calculates Standard Deviation."""
        return self._safe_series_operation("close", "std", window)

    def calculate_variance(self, window: int = 20) -> pd.Series:
        """Calculates Variance."""
        return self._safe_series_operation("close", "var", window)

    def calculate_klinger_oscillator(self, fast: int = 34, slow: int = 55) -> pd.DataFrame:
        """Calculates Klinger Oscillator."""
        tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        sv = pd.Series(0.0, index=self.df.index)
        for i in range(1, len(self.df)):
            if tp.iloc[i] > tp.iloc[i-1]:
                sv.iloc[i] = self.df["volume"].iloc[i]
            else:
                sv.iloc[i] = -self.df["volume"].iloc[i]
        ko = sv.ewm(span=fast, adjust=False).mean() - sv.ewm(span=slow, adjust=False).mean()
        signal = ko.ewm(span=13, adjust=False).mean()
        return pd.DataFrame({"ko": ko, "signal": signal})

    def calculate_nvi(self) -> pd.Series:
        """Calculates Negative Volume Index."""
        roc = self.df["close"].pct_change()
        vol_roc = self.df["volume"].diff()
        nvi = pd.Series(1000.0, index=self.df.index)
        for i in range(1, len(self.df)):
            if vol_roc.iloc[i] < 0:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + roc.iloc[i])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        return nvi

    def calculate_pvi(self) -> pd.Series:
        """Calculates Positive Volume Index."""
        roc = self.df["close"].pct_change()
        vol_roc = self.df["volume"].diff()
        pvi = pd.Series(1000.0, index=self.df.index)
        for i in range(1, len(self.df)):
            if vol_roc.iloc[i] > 0:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + roc.iloc[i])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        return pvi

    def calculate_bop(self) -> pd.Series:
        """Calculates Balance of Power."""
        return (self.df["close"] - self.df["open"]) / (self.df["high"] - self.df["low"]).replace(0, np.nan)

    def calculate_chandelier_exit(self, period: int = 22, multiplier: float = 3.0) -> dict:
        """Calculates Chandelier Exit."""
        atr = self.calculate_atr(period)
        highest_high = self.df["high"].rolling(period).max()
        lowest_low = self.df["low"].rolling(period).min()

        long_exit = highest_high - (atr * multiplier)
        short_exit = lowest_low + (atr * multiplier)

        return {"long": float(long_exit.iloc[-1]), "short": float(short_exit.iloc[-1])}

    def calculate_all_indicators(self) -> dict[str, Any]:
        """Orchestrates technical analysis, returning the latest state of all enabled indicators."""
        res = {}
        atr = self.calculate_atr(self.config["atr_period"])
        self.atr_value = atr.iloc[-1] if not atr.empty else 0.0
        res["atr"] = self.atr_value

        cfg = self.config["indicators"]
        if cfg.get("rsi"):
            res["rsi"] = self.calculate_rsi().iloc[-1]
        if cfg.get("macd"):
            res["macd"] = self.calculate_macd().iloc[-1].to_dict()
        if cfg.get("supersmoother"):
            res["supersmoother"] = self.calculate_supersmoother().iloc[-1]
        if cfg.get("ehlers_fisher"):
            res["ehlers_fisher"] = self.calculate_ehlers_fisher().iloc[-1]
        if cfg.get("laguerre_rsi"):
            res["laguerre_rsi"] = self.calculate_laguerre_rsi().iloc[-1]
        if cfg.get("fve"):
            res["fve"] = self.calculate_fve().iloc[-1]
        if cfg.get("stoch_rsi"):
            stoch_rsi_df = self.calculate_stoch_rsi()
            if not stoch_rsi_df.empty:
                res["stoch_rsi"] = stoch_rsi_df.iloc[-1].to_dict()
                res["stoch_rsi_vals"] = stoch_rsi_df

        res["mom"] = self.determine_trend_momentum()
        return res

    def calculate_all_indicators_vectorized(self) -> dict[str, Any]:
        """Calculates all enabled indicators and returns the FULL historical series."""
        results = {}

        # 1. Base Momentum & Volatility
        atr_series = self.calculate_atr(window=self.config["atr_period"])
        results["atr"] = atr_series
        self.atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        self.df["momentum"] = self._safe_series_operation(
            "close", "diff", self.config["momentum_period"]
        )
        self.df["momentum_ma_short"] = self._safe_series_operation(
            None, "sma", self.config["momentum_ma_short"], self.df["momentum"]
        )
        self.df["momentum_ma_long"] = self._safe_series_operation(
            None, "sma", self.config["momentum_ma_long"], self.df["momentum"]
        )
        self.df["volume_ma"] = self._safe_series_operation(
            "volume", "sma", self.config["volume_ma_period"]
        )

        # 2. Vectorized Indicators
        results["rsi"] = self.calculate_rsi(
            window=self.config["indicator_periods"]["rsi"]
        )
        results["mfi"] = self.calculate_mfi(
            window=self.config["indicator_periods"]["mfi"]
        )
        results["cci"] = self.calculate_cci(
            window=self.config["indicator_periods"]["cci"]
        )
        results["wr"] = self.calculate_williams_r(
            window=self.config["indicator_periods"]["williams_r"]
        )
        adx_df = self.calculate_adx_series(
            window=self.config["indicator_periods"]["adx"]
        )
        results["adx"] = adx_df["ADX"]
        results["adx_plus_di"] = adx_df["+DI"]
        results["adx_minus_di"] = adx_df["-DI"]
        results["obv"] = self.calculate_obv()
        results["adi"] = self.calculate_adi()
        results["fve"] = self.calculate_fve()
        results["macd"] = self.calculate_macd()
        results["stoch_rsi_vals"] = self.calculate_stoch_rsi()
        results["stoch_osc_vals"] = self.calculate_stochastic_oscillator()
        results["bollinger_bands"] = self.calculate_bollinger_bands()
        results["awesome_oscillator"] = self.calculate_awesome_oscillator()
        results["vortex"] = self.calculate_vortex()
        results["supertrend"] = self.calculate_supertrend()
        results["ehlers_fisher"] = self.calculate_ehlers_fisher()
        results["laguerre_rsi"] = self.calculate_laguerre_rsi()
        results["stc"] = self.calculate_stc()
        results["cmo"] = self.calculate_cmo()
        results["ema_alignment"] = self._calculate_ema_alignment_series()

        # Placeholder for OB walls (cannot be vectorized from kline history alone)
        results["order_book_walls"] = {"bullish": False, "bearish": False}
        results["l2_metrics"] = {}

        return results

    def _calculate_ema_alignment_series(self) -> pd.Series:
        """Vectorized EMA alignment scoring for the entire history."""
        ema_short = self.calculate_ema(self.config["ema_short_period"])
        ema_long = self.calculate_ema(self.config["ema_long_period"])
        if ema_short.empty or ema_long.empty:
            return pd.Series(dtype=float)
        alignment = pd.Series(0.0, index=self.df.index)

        # Simple crossover logic for backtest history
        alignment[(ema_short > ema_long) & (self.df["close"] > ema_short)] = 1.0
        alignment[(ema_short < ema_long) & (self.df["close"] < ema_short)] = -1.0
        alignment = alignment.fillna(0.0)
        return alignment


class SignalHistoryTracker:
    """Tracks and analyzes signal history for performance evaluation with fee-aware logic."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        config: dict,
        logger: logging.Logger,
        risk_manager: Any = None,
    ):
        self.db_manager = db_manager
        self.config = config
        self.logger = logger
        self.risk_manager = risk_manager
        self.performance_calculator = PerformanceCalculator(db_manager)
        self.active_signals = {}  # signal_id -> SignalHistory

    def sync_with_exchange(self, api_client: APIClient, symbol: str) -> None:
        """Synchronize internal state with exchange positions."""
        try:
            res = api_client.make_request(
                "GET", "/v5/position/list", {"category": "linear", "symbol": symbol}
            )
            if res and res.get("retCode") == 0:
                for pos in res["result"].get("list", []):
                    size = Decimal(pos.get("size", "0"))
                    if size > 0:
                        side = pos.get("side")
                        entry = Decimal(pos.get("avgPrice", "0"))
                        signal = TradingSignal(
                            signal_type=SignalType.BUY
                            if side == "Buy"
                            else SignalType.SELL,
                            confidence=1.0,
                            conditions_met=["Sync"],
                            stop_loss=Decimal(pos.get("stopLoss", "0")) or None,
                            take_profit=Decimal(pos.get("takeProfit", "0")) or None,
                            timestamp=time.time(),
                            symbol=symbol,
                            timeframe="sync",
                            position_size=size,
                        )
                        self.add_signal(signal, entry)
        except Exception as e:
            self.logger.error(f"Error syncing with exchange: {e}")

    def add_signal(self, signal: TradingSignal, entry_price: Decimal) -> int | None:
        """Add signal to history and tracking."""
        hist = SignalHistory(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            signal_type=signal.signal_type,
            confidence=signal.confidence,
            entry_price=entry_price,
            quantity=signal.position_size or Decimal("0"),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        sid = self.db_manager.save_signal(hist)
        if sid:
            self.active_signals[sid] = hist
        return sid

    def update_signal(
        self, sid: int, exit_p: Decimal, reason: str, fee_rates: tuple[Decimal, Decimal]
    ) -> bool:
        """Close signal and calculate net results."""
        if sid not in self.active_signals:
            return False
        s = self.active_signals[sid]
        _, taker_rate = fee_rates

        if s.signal_type == SignalType.BUY:
            gross = (exit_p - s.entry_price) * s.quantity
        else:
            gross = (s.entry_price - exit_p) * s.quantity

        fees = (s.entry_price * s.quantity * taker_rate) + (
            exit_p * s.quantity * taker_rate
        )
        net = gross - fees

        if self.db_manager.update_signal(sid, exit_p, net, reason):
            s.exit_price, s.profit_loss, s.net_pnl, s.exit_reason = (
                exit_p,
                gross,
                net,
                reason,
            )
            del self.active_signals[sid]
            self.logger.info(
                f"{NEON_GREEN}Closed {sid} | Net: ${float(net):.2f} | Reason: {reason}{RESET}"
            )

            # Update risk manager with result
            if self.risk_manager:
                self.risk_manager.update_trade_result(net)

            return True
        return False

    def calculate_break_even(self, s: SignalHistory, rate: Decimal) -> Decimal:
        """Calculate exit price needed for zero net PnL."""
        if s.signal_type == SignalType.BUY:
            return s.entry_price * (1 + rate) / (1 - rate)
        return s.entry_price * (1 - rate) / (1 + rate)

    def display_active_status(
        self, current_price: Decimal, daily_loss: Decimal
    ) -> None:
        """Displays a summary of active positions and session performance."""
        # Get session metrics
        try:
            metrics = self.performance_calculator.calculate_metrics(
                self.active_signals[list(self.active_signals.keys())[0]].symbol
                if self.active_signals
                else "Session",
                "",
            )
        except Exception:
            metrics = None

        status_output = f"\n{NEON_CYAN}--- SESSION STATUS ---{RESET}\n"
        if metrics and metrics.total_trades > 0:
            status_output += (
                f"{NEON_BLUE}Trades:{RESET} {metrics.total_trades} | "
                f"{NEON_GREEN}Win Rate:{RESET} {metrics.win_rate:.1%} | "
                f"{NEON_PURPLE}PF:{RESET} {metrics.profit_factor:.2f}\n"
            )

        pnl_color = (
            NEON_GREEN if daily_loss > 0 else NEON_RED if daily_loss < 0 else NEON_WHITE
        )
        status_output += (
            f"{NEON_BLUE}Realized PnL: {pnl_color}${float(daily_loss):.2f}{RESET}\n"
        )

        if self.active_signals:
            status_output += f"\n{NEON_CYAN}--- ACTIVE POSITIONS ---{RESET}\n"
            for signal_id, signal in self.active_signals.items():
                # Calculate unrealized P&L
                if signal.signal_type == SignalType.BUY:
                    unrealized_pnl = (
                        current_price - signal.entry_price
                    ) * signal.quantity
                    pnl_pct = (
                        (current_price - signal.entry_price) / signal.entry_price * 100
                    )
                else:
                    unrealized_pnl = (
                        signal.entry_price - current_price
                    ) * signal.quantity
                    pnl_pct = (
                        (signal.entry_price - current_price) / signal.entry_price * 100
                    )

                pnl_color = NEON_GREEN if unrealized_pnl >= 0 else NEON_RED
                sl_val = signal.trailing_sl or signal.stop_loss

                sl_str = f"${float(sl_val):.4f}" if sl_val is not None else "N/A"
                tp_str = (
                    f"${float(signal.take_profit):.4f}"
                    if signal.take_profit is not None
                    else "N/A"
                )

                status_output += (
                    f"{NEON_PURPLE}ID:{signal_id}{RESET} | {signal.signal_type.value.upper()} {signal.symbol} | "
                    f"Entry: ${signal.entry_price:.4f} | Qty: {signal.quantity:.4f}\n"
                    f"  {pnl_color}Unrealized PnL: ${float(unrealized_pnl):.2f} ({pnl_pct:.2f}%){RESET} | "
                    f"SL: {sl_str} | TP: {tp_str}\n"
                )
        else:
            status_output += f"{NEON_YELLOW}No open positions.{RESET}\n"

        self.logger.info(status_output)

    def update_trailing_stops(
        self,
        current_price: Decimal,
        chandelier_exit: dict,
        notification_system: NotificationSystem,
    ) -> None:
        """Updates trailing stops for all active signals and sends notifications on significant moves."""
        if not self.config.get("trailing_stop_loss", {}).get("enabled", False):
            return  # Skip trailing stop updates if disabled

        for _signal_id, signal in self.active_signals.items():
            old_sl = signal.trailing_sl or signal.stop_loss

            # Update extreme prices reached
            if signal.highest_price is None or current_price > signal.highest_price:
                signal.highest_price = current_price
            if signal.lowest_price is None or current_price < signal.lowest_price:
                signal.lowest_price = current_price

            # Chandelier Trailing logic
            if signal.signal_type == SignalType.BUY:
                ce_long = chandelier_exit.get("long")
                if ce_long:
                    # Trailing SL can only move UP
                    if signal.trailing_sl is None or ce_long > signal.trailing_sl:
                        signal.trailing_sl = ce_long
            else:  # SELL
                ce_short = chandelier_exit.get("short")
                if ce_short:
                    # Trailing SL can only move DOWN
                    if signal.trailing_sl is None or ce_short < signal.trailing_sl:
                        signal.trailing_sl = ce_short

            # Notification on significant SL move (or initial set)
            if signal.trailing_sl and signal.trailing_sl != old_sl:
                # Calculate current P&L %
                if signal.signal_type == SignalType.BUY:
                    pnl_pct = (
                        (current_price - signal.entry_price) / signal.entry_price * 100
                    )
                else:
                    pnl_pct = (
                        (signal.entry_price - current_price) / signal.entry_price * 100
                    )

                msg = f"TSL Update {signal.symbol}: New SL ${signal.trailing_sl:.2f} | PnL: {pnl_pct:.2f}%"
                notification_system.send_sms(msg)

    def check_exit_conditions(
        self,
        current_price: Decimal,
        symbol: str,
        timeframe: str,
        fee_rates: tuple[Decimal, Decimal] = (Decimal("0.0002"), Decimal("0.00055")),
    ) -> list[tuple[int, str]]:
        """Check if any active signals should be exited, accounting for fees."""
        signals_to_exit = []
        maker_rate, taker_rate = fee_rates

        for signal_id, signal in self.active_signals.items():
            if signal.symbol != symbol or signal.timeframe != timeframe:
                continue

            exit_reason = None
            break_even = self.calculate_break_even(signal, taker_rate)
            current_sl = signal.trailing_sl if signal.trailing_sl else signal.stop_loss

            # Check stop loss (Hard Exit)
            if current_sl and (
                (signal.signal_type == SignalType.BUY and current_price <= current_sl)
                or (
                    signal.signal_type == SignalType.SELL
                    and current_price >= current_sl
                )
            ):
                exit_reason = (
                    "Trailing Stop Loss" if signal.trailing_sl else "Stop Loss"
                )

            # Check take profit (Requires Net Profit)
            if signal.take_profit and (
                (
                    signal.signal_type == SignalType.BUY
                    and current_price >= signal.take_profit
                )
                or (
                    signal.signal_type == SignalType.SELL
                    and current_price <= signal.take_profit
                )
            ):
                # Verify we are actually in net profit after fees
                if (
                    signal.signal_type == SignalType.BUY and current_price > break_even
                ) or (
                    signal.signal_type == SignalType.SELL and current_price < break_even
                ):
                    exit_reason = "Take Profit"
                else:
                    # Optional: Could log that we hit TP but fees make it a loss
                    pass

            if exit_reason:
                signals_to_exit.append((signal_id, exit_reason))

        return signals_to_exit


    def calculate_all_indicators(self) -> dict[str, Any]:
        """
        Calculates all enabled indicators and returns their latest values.
        For multi-component indicators (like BB, Ichimoku), the latest row is returned as a dict.
        """
        results = {}

        # Base Momentum & Volatility
        atr_series = self.calculate_atr(window=self.config["atr_period"])
        if not atr_series.empty and not pd.isna(atr_series.iloc[-1]):
            self.atr_value = atr_series.iloc[-1]
            results["atr"] = self.atr_value
        else:
            self.atr_value = 0.0
            results["atr"] = 0.0

        if self.config["indicators"].get("momentum"):
            self.df["momentum"] = self._safe_series_operation(
                "close", "diff", self.config["momentum_period"]
            )
            self.df["momentum_ma_short"] = self._safe_series_operation(
                None, "sma", self.config["momentum_ma_short"], self.df["momentum"]
            )
            self.df["momentum_ma_long"] = self._safe_series_operation(
                None, "sma", self.config["momentum_ma_long"], self.df["momentum"]
            )
            results["momentum_ma_short"] = (
                self.df["momentum_ma_short"].iloc[-1]
                if not self.df["momentum_ma_short"].empty
                else np.nan
            )
            results["momentum_ma_long"] = (
                self.df["momentum_ma_long"].iloc[-1]
                if not self.df["momentum_ma_long"].empty
                else np.nan
            )

        # Pre-calculate volume_ma for volume confirmation and other indicators
        self.df["volume_ma"] = self._safe_series_operation(
            "volume", "sma", self.config["volume_ma_period"]
        )
        results["volume_ma"] = (
            self.df["volume_ma"].iloc[-1] if not self.df["volume_ma"].empty else np.nan
        )

        # Individual Indicators
        if self.config["indicators"].get("rsi"):
            rsi_series = self.calculate_rsi(
                window=self.config["indicator_periods"]["rsi"]
            )
            results["rsi"] = rsi_series.iloc[-1] if not rsi_series.empty else np.nan

        if self.config["indicators"].get("mfi"):
            mfi_series = self.calculate_mfi(
                window=self.config["indicator_periods"]["mfi"]
            )
            results["mfi"] = mfi_series.iloc[-1] if not mfi_series.empty else np.nan

        if self.config["indicators"].get("cci"):
            cci_series = self.calculate_cci(
                window=self.config["indicator_periods"]["cci"]
            )
            results["cci"] = cci_series.iloc[-1] if not cci_series.empty else np.nan

        if self.config["indicators"].get("wr"):
            wr_series = self.calculate_williams_r(
                window=self.config["indicator_periods"]["williams_r"]
            )
            results["wr"] = wr_series.iloc[-1] if not wr_series.empty else np.nan

        if self.config["indicators"].get("adx"):
            adx_data = self.calculate_adx(
                window=self.config["indicator_periods"]["adx"]
            )
            results["adx"] = adx_data["adx"]
            results["adx_data"] = adx_data

        if self.config["indicators"].get("obv"):
            obv_series = self.calculate_obv()
            results["obv"] = obv_series.iloc[-1] if not obv_series.empty else np.nan

        if self.config["indicators"].get("adi"):
            adi_series = self.calculate_adi()
            results["adi"] = adi_series.iloc[-1] if not adi_series.empty else np.nan

        if self.config["indicators"].get("sma_10"):
            sma_series = self.calculate_sma(10)
            results["sma_10"] = sma_series.iloc[-1] if not sma_series.empty else np.nan

        if self.config["indicators"].get("psar"):
            psar_series = self.calculate_psar()
            results["psar"] = psar_series.iloc[-1] if not psar_series.empty else np.nan

        if self.config["indicators"].get("fve"):
            fve_series = self.calculate_fve()
            results["fve"] = (
                fve_series.iloc[-1]
                if not fve_series.empty and not fve_series.isnull().all()
                else np.nan
            )

        if self.config["indicators"].get("macd"):
            macd_df = self.calculate_macd()
            results["macd"] = macd_df.iloc[-1].to_dict() if not macd_df.empty else {}

        if self.config["indicators"].get("stoch_rsi"):
            stoch_rsi_df = self.calculate_stoch_rsi(
                rsi_window=self.config["indicator_periods"]["stoch_rsi_period"],
                stoch_window=self.config["indicator_periods"]["stoch_rsi_period"],
                k_window=self.config["indicator_periods"]["stoch_rsi_k_period"],
                d_window=self.config["indicator_periods"]["stoch_rsi_d_period"],
            )
            if not stoch_rsi_df.empty:
                results["stoch_rsi"] = stoch_rsi_df.iloc[-1].to_dict()
                results["stoch_rsi_vals"] = stoch_rsi_df
            else:
                results["stoch_rsi"] = {}

        if self.config["indicators"].get("stochastic_oscillator"):
            stoch_osc_df = self.calculate_stochastic_oscillator()
            if not stoch_osc_df.empty:
                results["stoch_oscillator"] = stoch_osc_df.iloc[-1].to_dict()
                results["stoch_osc_vals"] = stoch_osc_df
            else:
                results["stoch_oscillator"] = {}

        if self.config["indicators"].get("bollinger_bands"):
            bb_df = self.calculate_bollinger_bands()
            results["bollinger_bands"] = (
                bb_df.iloc[-1].to_dict() if not bb_df.empty else {}
            )

        if self.config["indicators"].get("keltner_channels"):
            kc_df = self.calculate_keltner_channels()
            results["keltner_channels"] = (
                kc_df.iloc[-1].to_dict() if not kc_df.empty else {}
            )

        if self.config["indicators"].get("ichimoku_cloud"):
            ichimoku_df = self.calculate_ichimoku_cloud()
            results["ichimoku_cloud"] = (
                ichimoku_df.iloc[-1].to_dict() if not ichimoku_df.empty else {}
            )

        if self.config["indicators"].get("vwap"):
            vwap_series = self.calculate_vwap()
            results["vwap"] = vwap_series.iloc[-1] if not vwap_series.empty else np.nan

        if self.config["indicators"].get("cmf"):
            cmf_series = self.calculate_cmf()
            results["cmf"] = cmf_series.iloc[-1] if not cmf_series.empty else np.nan

        if self.config["indicators"].get("emv"):
            emv_series = self.calculate_emv()
            results["emv"] = emv_series.iloc[-1] if not emv_series.empty else np.nan

        if self.config["indicators"].get("force_index"):
            fi_series = self.calculate_force_index()
            results["force_index"] = (
                fi_series.iloc[-1] if not fi_series.empty else np.nan
            )

        if self.config["indicators"].get("mass_index"):
            mi_series = self.calculate_mass_index()
            results["mass_index"] = (
                mi_series.iloc[-1] if not mi_series.empty else np.nan
            )

        if self.config["indicators"].get("roc"):
            roc_series = self.calculate_roc()
            results["roc"] = roc_series.iloc[-1] if not roc_series.empty else np.nan

        if self.config["indicators"].get("trix"):
            trix_series = self.calculate_trix()
            results["trix"] = trix_series.iloc[-1] if not trix_series.empty else np.nan

        if self.config["indicators"].get("ultimate_oscillator"):
            uo_series = self.calculate_ultimate_oscillator()
            results["ultimate_oscillator"] = (
                uo_series.iloc[-1] if not uo_series.empty else np.nan
            )

        if self.config["indicators"].get("vortex"):
            vortex_df = self.calculate_vortex()
            results["vortex"] = (
                vortex_df.iloc[-1].to_dict() if not vortex_df.empty else {}
            )

        if self.config["indicators"].get("coppock_curve"):
            coppock_series = self.calculate_coppock_curve()
            results["coppock_curve"] = (
                coppock_series.iloc[-1] if not coppock_series.empty else np.nan
            )

        if self.config["indicators"].get("donchian_channels"):
            donchian_df = self.calculate_donchian_channels()
            results["donchian_channels"] = (
                donchian_df.iloc[-1].to_dict() if not donchian_df.empty else {}
            )

        if self.config["indicators"].get("hma"):
            hma_series = self.calculate_hma()
            results["hma"] = hma_series.iloc[-1] if not hma_series.empty else np.nan

        if self.config["indicators"].get("awesome_oscillator"):
            ao_series = self.calculate_awesome_oscillator()
            results["awesome_oscillator"] = (
                ao_series.iloc[-1] if not ao_series.empty else np.nan
            )

        if self.config["indicators"].get("std_dev"):
            std_series = self.calculate_std()
            results["std_dev"] = std_series.iloc[-1] if not std_series.empty else np.nan

        if self.config["indicators"].get("variance"):
            var_series = self.calculate_variance()
            results["variance"] = (
                var_series.iloc[-1] if not var_series.empty else np.nan
            )

        if self.config["indicators"].get("klinger_oscillator"):
            ko_df = self.calculate_klinger_oscillator()
            results["klinger_oscillator"] = (
                ko_df.iloc[-1].to_dict() if not ko_df.empty else {}
            )

        if self.config["indicators"].get("nvi"):
            nvi_series = self.calculate_nvi()
            results["nvi"] = nvi_series.iloc[-1] if not nvi_series.empty else np.nan

        if self.config["indicators"].get("pvi"):
            pvi_series = self.calculate_pvi()
            results["pvi"] = pvi_series.iloc[-1] if not pvi_series.empty else np.nan

        if self.config["indicators"].get("bop"):
            bop_series = self.calculate_bop()
            results["bop"] = bop_series.iloc[-1] if not bop_series.empty else np.nan

        # EC5 Indicators
        if self.config["indicators"].get("supersmoother"):
            ss_series = self.calculate_supersmoother()
            results["supersmoother"] = (
                ss_series.iloc[-1] if not ss_series.empty else np.nan
            )

        if self.config["indicators"].get("ehlers_fisher"):
            fisher_series = self.calculate_ehlers_fisher()
            results["ehlers_fisher"] = (
                fisher_series.iloc[-1] if not fisher_series.empty else np.nan
            )

        if self.config["indicators"].get("laguerre_rsi"):
            lrsi_series = self.calculate_laguerre_rsi()
            results["laguerre_rsi"] = (
                lrsi_series.iloc[-1] if not lrsi_series.empty else np.nan
            )

        if self.config["indicators"].get("supertrend"):
            st_df = self.calculate_supertrend()
            results["supertrend"] = st_df.iloc[-1].to_dict() if not st_df.empty else {}

        if self.config["indicators"].get("cmo"):
            cmo_series = self.calculate_cmo()
            results["cmo"] = cmo_series.iloc[-1] if not cmo_series.empty else np.nan

        if self.config["indicators"].get("stc"):
            stc_series = self.calculate_stc()
            results["stc"] = stc_series.iloc[-1] if not stc_series.empty else np.nan

        # Chandelier Exit (often used for dynamic SL/TP, so get latest)
        results["chandelier_exit"] = self.calculate_chandelier_exit()

        # Store momentum trend data
        if self.config["indicators"].get("momentum"):
            trend_data = self.determine_trend_momentum()
            results["mom"] = trend_data

        # Calculate EMA alignment
        if self.config["indicators"].get("ema_alignment"):
            ema_alignment_score = self.calculate_ema_alignment()
            results["ema_alignment"] = ema_alignment_score

        return results

    def determine_trend_momentum(self) -> dict[str, str | float]:
        """Determines the current trend and its strength based on momentum MAs and ATR."""
        if self.df.empty or len(self.df) < max(
            self.config["momentum_ma_long"], self.config["atr_period"]
        ):
            return {"trend": "Insufficient Data", "strength": 0.0}

        # Ensure momentum_ma_short, momentum_ma_long, and atr_value are calculated
        if (
            self.df["momentum_ma_short"].empty
            or self.df["momentum_ma_long"].empty
            or self.atr_value == 0
        ):
            self.logger.warning(
                f"{NEON_YELLOW}Momentum MAs or ATR not available for trend calculation.{RESET}"
            )
            return {"trend": "Neutral", "strength": 0.0}

        latest_short_ma = self.df["momentum_ma_short"].iloc[-1]
        latest_long_ma = self.df["momentum_ma_long"].iloc[-1]

        trend = "Neutral"
        if latest_short_ma > latest_long_ma:
            trend = "Uptrend"
        elif latest_short_ma < latest_long_ma:
            trend = "Downtrend"

        # Strength is normalized by ATR to make it comparable across symbols/timeframes
        strength = abs(latest_short_ma - latest_long_ma) / self.atr_value
        return {"trend": trend, "strength": strength}

    def calculate_ema_alignment(self) -> float:
        """
        Calculates an EMA alignment score.
        Score is 1.0 for strong bullish alignment, -1.0 for strong bearish, 0.0 for neutral.
        """
        ema_short = self.calculate_ema(self.config["ema_short_period"])
        ema_long = self.calculate_ema(self.config["ema_long_period"])

        if (
            ema_short.empty
            or ema_long.empty
            or len(self.df)
            < max(self.config["ema_short_period"], self.config["ema_long_period"])
        ):
            return 0.0

        latest_short_ema = Decimal(str(ema_short.iloc[-1]))
        latest_long_ema = Decimal(str(ema_long.iloc[-1]))

        # Check for consistent alignment over the last few bars (e.g., 3 bars)
        alignment_period = 3
        if len(ema_short) < alignment_period or len(ema_long) < alignment_period:
            return 0.0

        bullish_aligned_count = 0
        bearish_aligned_count = 0

        for i in range(1, alignment_period + 1):
            if (
                ema_short.iloc[-i] > ema_long.iloc[-i]
                and self.df["close"].iloc[-i] > ema_short.iloc[-i]
            ):
                bullish_aligned_count += 1
            elif (
                ema_short.iloc[-i] < ema_long.iloc[-i]
                and self.df["close"].iloc[-i] < ema_short.iloc[-i]
            ):
                bearish_aligned_count += 1

        if (
            bullish_aligned_count >= alignment_period - 1
        ):  # At least (period-1) bars are aligned
            return 1.0  # Strong bullish alignment
        elif bearish_aligned_count >= alignment_period - 1:
            return -1.0  # Strong bearish alignment
        else:
            # Check for recent crossover as a weaker signal
            if (
                latest_short_ema > latest_long_ema
                and ema_short.iloc[-2] <= latest_long_ema
            ):
                return 0.5  # Recent bullish crossover
            elif (
                latest_short_ema < latest_long_ema
                and ema_short.iloc[-2] >= latest_long_ema
            ):
                return -0.5  # Recent bearish crossover
            return 0.0  # Neutral

    def detect_macd_divergence(self) -> str | None:
        """Detects bullish or bearish MACD divergence."""
        macd_df = self.calculate_macd()
        if (
            macd_df.empty or len(self.df) < 30
        ):  # Need sufficient data for reliable divergence
            return None

        prices = self.df["close"]
        macd_histogram = macd_df["histogram"]

        # Simple divergence check on last two bars (can be expanded for more robust detection)
        if (
            prices.iloc[-2] > prices.iloc[-1]
            and macd_histogram.iloc[-2] < macd_histogram.iloc[-1]
        ):
            self.logger.debug(f"{NEON_GREEN}Detected Bullish MACD Divergence.{RESET}")
            return "bullish"
        elif (
            prices.iloc[-2] < prices.iloc[-1]
            and macd_histogram.iloc[-2] > macd_histogram.iloc[-1]
        ):
            self.logger.debug(f"{NEON_RED}Detected Bearish MACD Divergence.{RESET}")
            return "bearish"
        return None

    def calculate_volume_confirmation(self) -> bool:
        """
        Checks if the current volume confirms a trend (e.g., significant spike).
        Returns True if current volume is significantly higher than average.
        """
        if "volume" not in self.df.columns or "volume_ma" not in self.df.columns:
            self.logger.error(
                f"{NEON_RED}Missing 'volume' or 'volume_ma' column for Volume Confirmation.{RESET}"
            )
            return False

        if self.df["volume"].empty or self.df["volume_ma"].empty:
            return False

        current_volume = self.df["volume"].iloc[-1]
        average_volume = self.df["volume_ma"].iloc[-1]

        if average_volume <= 0:  # Avoid division by zero or nonsensical average
            return False

        return (
            current_volume
            > average_volume * self.config["volume_confirmation_multiplier"]
        )


# --- Support/Resistance Analyzer ---
class SupportResistanceAnalyzer:
    """Analyzes support and resistance levels using Fibonacci retracements and pivot points."""

    def __init__(self, df: pd.DataFrame, config: dict, logger: logging.Logger):
        self.df = df
        self.config = config
        self.logger = logger
        self.levels: dict[str, Any] = {}
        self.fib_levels: dict[str, float] = {}

    def calculate_fibonacci_retracement(
        self, high: Decimal, low: Decimal, current_price: Decimal
    ) -> dict[str, Decimal]:
        """Calculates Fibonacci retracement levels based on a given high and low."""
        diff = high - low
        if diff <= 0:
            self.logger.warning(f"{NEON_YELLOW}High less or equal to Low, skipping Fibonacci retracement.{RESET}")
            return {}

        # Standard Fibonacci ratios
        fib_ratios = {
            "23.6%": Decimal("0.236"),
            "38.2%": Decimal("0.382"),
            "50.0%": Decimal("0.500"),
            "61.8%": Decimal("0.618"),
            "78.6%": Decimal("0.786"),
            "88.6%": Decimal("0.886"),
            "94.1%": Decimal("0.941"),
        }

        fib_levels_calculated: dict[str, Decimal] = {}

        # Assuming an uptrend (retracement from high to low)
        # Levels are calculated from the high, moving down
        for label, ratio in fib_ratios.items():
            level = high - (diff * ratio)
            fib_levels_calculated[f"Fib {label}"] = level.quantize(
                Decimal("0.00001")
            )  # Quantize for consistent precision

        self.fib_levels = fib_levels_calculated
        self.levels = {"Support": {}, "Resistance": {}}

        # Categorize levels as support or resistance relative to current price
        for label, value in self.fib_levels.items():
            if value < current_price:
                self.levels["Support"][label] = value
            elif value > current_price:
                self.levels["Resistance"][label] = value

        return self.fib_levels

    def calculate_pivot_points(self, high: Decimal, low: Decimal, close: Decimal):
        """Calculates standard Pivot Points."""
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        # Quantize all pivot points for consistent precision
        precision = Decimal("0.00001")
        self.levels.update(
            {
                "Pivot": pivot.quantize(precision),
                "R1": r1.quantize(precision),
                "S1": s1.quantize(precision),
                "R2": r2.quantize(precision),
                "S2": s2.quantize(precision),
                "R3": r3.quantize(precision),
                "S3": s3.quantize(precision),
            }
        )

    def find_nearest_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> tuple[list[tuple[str, Decimal]], list[tuple[str, Decimal]]]:
        """
        Finds the nearest support and resistance levels from calculated Fibonacci and Pivot Points.
        """
        all_support_levels: list[tuple[str, Decimal]] = []
        all_resistance_levels: list[tuple[str, Decimal]] = []

        def process_level(label: str, value: Decimal):
            if value < current_price:
                all_support_levels.append((label, value))
            elif value > current_price:
                all_resistance_levels.append((label, value))

        # Process all levels stored in self.levels (from Fibonacci and Pivot)
        for label, value in self.levels.items():
            if isinstance(
                value, dict
            ):  # For nested levels like "Support": {"Fib 23.6%": ...}
                for sub_label, sub_value in value.items():
                    if isinstance(sub_value, Decimal):
                        process_level(f"{label} ({sub_label})", sub_value)
            elif isinstance(value, Decimal):  # For direct levels like "Pivot"
                process_level(label, value)

        # Sort by distance to current price and select the 'num_levels' closest
        nearest_supports = sorted(
            all_support_levels, key=lambda x: current_price - x[1]
        )[:num_levels]
        nearest_resistances = sorted(
            all_resistance_levels, key=lambda x: x[1] - current_price
        )[:num_levels]

        return nearest_supports, nearest_resistances


# --- Order Book Analyzer ---
class OrderBookAnalyzer:
    """Analyzes order book data for support/resistance walls and liquidity."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def analyze_order_book_l2_metrics(
        self, order_book: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculates L2 order book metrics: imbalance, spread, depth, etc."""
        if not order_book or not order_book.get("bids") or not order_book.get("asks"):
            return {}

        bids = [(Decimal(price), Decimal(qty)) for price, qty in order_book["bids"]]
        asks = [(Decimal(price), Decimal(qty)) for price, qty in order_book["asks"]]

        if not bids or not asks:
            return {}

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100

        # Calculate imbalance at different depths
        def calculate_imbalance(depth: int):
            bid_vol = sum(q for _, q in bids[:depth])
            ask_vol = sum(q for _, q in asks[:depth])
            return (
                float((bid_vol - ask_vol) / (bid_vol + ask_vol))
                if (bid_vol + ask_vol) > 0
                else 0.0
            )

        imbalance_5 = calculate_imbalance(5)
        imbalance_10 = calculate_imbalance(10)
        imbalance_20 = calculate_imbalance(20)

        # Calculate total depth within X% of mid price
        def calculate_depth_within_pct(pct: float):
            lower_bound = mid_price * (1 - Decimal(str(pct / 100)))
            upper_bound = mid_price * (1 + Decimal(str(pct / 100)))
            bid_depth = sum(q for p, q in bids if p >= lower_bound)
            ask_depth = sum(q for p, q in asks if p <= upper_bound)
            return float(bid_depth), float(ask_depth)

        depth_1_pct_bid, depth_1_pct_ask = calculate_depth_within_pct(1.0)

        return {
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "mid_price": float(mid_price),
            "spread": float(spread),
            "spread_pct": float(spread_pct),
            "imbalance_5": imbalance_5,
            "imbalance_10": imbalance_10,
            "imbalance_20": imbalance_20,
            "depth_1_pct_bid": depth_1_pct_bid,
            "depth_1_pct_ask": depth_1_pct_ask,
            "bid_ask_ratio": float(depth_1_pct_bid / depth_1_pct_ask)
            if depth_1_pct_ask > 0
            else 1.0,
        }

    def find_liquidity_clusters(
        self, order_book: dict[str, Any], top_n: int = 3
    ) -> dict[str, list[tuple[Decimal, Decimal]]]:
        """Identifies price levels with the highest cumulative liquidity (clusters)."""
        if not order_book or not order_book.get("bids") or not order_book.get("asks"):
            return {"bids": [], "asks": []}

        # Convert strings to Decimals
        bids = [(Decimal(p), Decimal(q)) for p, q in order_book["bids"]]
        asks = [(Decimal(p), Decimal(q)) for p, q in order_book["asks"]]

        # Sort by quantity to find biggest levels
        top_bids = sorted(bids, key=lambda x: x[1], reverse=True)[:top_n]
        top_asks = sorted(asks, key=lambda x: x[1], reverse=True)[:top_n]

        return {"bids": top_bids, "asks": top_asks}

    def get_depth_profile(
        self, order_book: dict[str, Any], current_price: Decimal
    ) -> dict[str, float]:
        """Calculates cumulative volume at various percentage distances from mid price."""
        if not order_book or not order_book.get("bids") or not order_book.get("asks"):
            return {}

        bids = [(Decimal(p), Decimal(q)) for p, q in order_book["bids"]]
        asks = [(Decimal(p), Decimal(q)) for p, q in order_book["asks"]]

        percentages = [0.1, 0.5, 1.0, 2.0, 5.0]
        profile = {}

        for pct in percentages:
            lower = current_price * (1 - Decimal(str(pct / 100)))
            upper = current_price * (1 + Decimal(str(pct / 100)))

            bid_vol = sum(q for p, q in bids if p >= lower)
            ask_vol = sum(q for p, q in asks if p <= upper)

            profile[f"bid_depth_{pct}%"] = float(bid_vol)
            profile[f"ask_depth_{pct}%"] = float(ask_vol)
            profile[f"imbalance_{pct}%"] = (
                float((bid_vol - ask_vol) / (bid_vol + ask_vol))
                if (bid_vol + ask_vol) > 0
                else 0.0
            )

        return profile

    def analyze_order_book_walls(
        self, order_book: dict[str, Any], current_price: Decimal
    ) -> tuple[bool, bool, dict[str, Decimal], dict[str, Decimal]]:
        """
        Analyzes order book for significant bid (support) and ask (resistance) walls.
        Returns whether bullish/bearish walls are found and the wall details.
        """
        has_bullish_wall = False
        has_bearish_wall = False
        bullish_wall_details: dict[str, Decimal] = {}
        bearish_wall_details: dict[str, Decimal] = {}

        if not self.config["order_book_analysis"]["enabled"]:
            return False, False, {}, {}

        if order_book is None:
            return False, False, {}, {}

        if not order_book.get("bids") or not order_book.get("asks"):
            self.logger.warning(
                f"{NEON_YELLOW}Order book data incomplete for wall analysis.{RESET}"
            )
            return False, False, {}, {}

        bids = [
            (Decimal(price), Decimal(qty))
            for price, qty in order_book["bids"][
                : self.config["order_book_analysis"]["depth_to_check"]
            ]
        ]
        asks = [
            (Decimal(price), Decimal(qty))
            for price, qty in order_book["asks"][
                : self.config["order_book_analysis"]["depth_to_check"]
            ]
        ]

        # Calculate average quantity across relevant depth
        all_quantities = [qty for _, qty in bids + asks]
        if not all_quantities:
            return False, False, {}, {}

        avg_qty = Decimal(
            str(np.mean([float(q) for q in all_quantities]))
        )  # Convert to float for numpy, then back to Decimal
        wall_threshold = avg_qty * Decimal(
            str(self.config["order_book_analysis"]["wall_threshold_multiplier"])
        )

        # Check for bullish walls (large bids below current price)
        for bid_price, bid_qty in bids:
            if bid_qty >= wall_threshold and bid_price < current_price:
                has_bullish_wall = True
                bullish_wall_details[f"Bid@{bid_price}"] = bid_qty
                self.logger.info(
                    f"{NEON_GREEN}Detected Bullish Order Book Wall: Bid {bid_qty:.2f} at {bid_price:.2f}{RESET}"
                )
                break  # Only need to find one significant wall

        # Check for bearish walls (large asks above current price)
        for ask_price, ask_qty in asks:
            if ask_qty >= wall_threshold and ask_price > current_price:
                has_bearish_wall = True
                bearish_wall_details[f"Ask@{ask_price}"] = ask_qty
                self.logger.info(
                    f"{NEON_RED}Detected Bearish Order Book Wall: Ask {ask_qty:.2f} at {ask_price:.2f}{RESET}"
                )
                break  # Only need to find one significant wall

        return (
            has_bullish_wall,
            has_bearish_wall,
            bullish_wall_details,
            bearish_wall_details,
        )


# --- Multi-Timeframe Analyzer ---
class MultiTimeframeAnalyzer:
    """Analyzes multiple timeframes to generate consensus signals."""

    def __init__(self, api_client: APIClient, config: dict, logger: logging.Logger):
        self.api_client = api_client
        self.config = config
        self.logger = logger
        self.timeframes = config.get("multi_timeframe", {}).get(
            "timeframes", ["5", "15", "60"]
        )
        self.weighting = config.get("multi_timeframe", {}).get(
            "weighting", {"5": 0.2, "15": 0.5, "60": 0.3}
        )
        self.last_primary_analyzer = None

    def analyze_timeframes(self, symbol: str) -> dict[str, TradingSignal]:
        """Analyze multiple timeframes and return signals for each."""
        signals = {}
        self.last_primary_analyzer = None

        for i, timeframe in enumerate(self.timeframes):
            try:
                # Fetch data for this timeframe
                df = self.api_client.fetch_klines(symbol, timeframe, limit=200)
                if df.empty:
                    self.logger.warning(
                        f"{NEON_YELLOW}No data for {symbol} {timeframe}{RESET}"
                    )
                    continue

                # Create analyzer for this timeframe
                analyzer = TradingAnalyzer(
                    df, self.config, self.logger, symbol, timeframe
                )

                # Get current price
                current_price = self.api_client.fetch_current_price(symbol)
                if current_price is None:
                    self.logger.warning(
                        f"{NEON_YELLOW}No price for {symbol} {timeframe}{RESET}"
                    )
                    continue

                # NEW: Perform full analysis to populate indicator_values and log them
                timestamp = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
                # Order book might be None here, that's fine
                analyzer.analyze(current_price, timestamp, None)

                # Generate signal
                signal = analyzer.generate_trading_signal(current_price)
                signals[timeframe] = signal

                # Store the first (primary) analyzer
                if i == 0:
                    self.last_primary_analyzer = analyzer

            except Exception as e:
                self.logger.error(
                    f"{NEON_RED}Error analyzing {symbol} {timeframe}: {e}{RESET}"
                )

        return signals

    def generate_consensus_signal(self, symbol: str) -> TradingSignal:
        """Generate a consensus signal from multiple timeframes."""
        timeframe_signals = self.analyze_timeframes(symbol)

        if not timeframe_signals:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                conditions_met=["No signals from any timeframe"],
                stop_loss=None,
                take_profit=None,
                timestamp=time.time(),
                symbol=symbol,
                timeframe="multi",
            )

        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0

        buy_conditions = []
        sell_conditions = []

        for timeframe, signal in timeframe_signals.items():
            weight = self.weighting.get(timeframe, 1.0 / len(timeframe_signals))

            if signal.signal_type == SignalType.BUY:
                buy_score += signal.confidence * weight
                buy_conditions.extend(
                    [f"{timeframe}: {cond}" for cond in signal.conditions_met]
                )
            elif signal.signal_type == SignalType.SELL:
                sell_score += signal.confidence * weight
                sell_conditions.extend(
                    [f"{timeframe}: {cond}" for cond in signal.conditions_met]
                )

        # Determine consensus signal
        if buy_score > sell_score and buy_score >= self.config.get(
            "signal_score_threshold", 1.0
        ):
            return TradingSignal(
                signal_type=SignalType.BUY,
                confidence=buy_score,
                conditions_met=buy_conditions,
                stop_loss=None,  # Would need to determine from multiple timeframes
                take_profit=None,  # Would need to determine from multiple timeframes
                timestamp=time.time(),
                symbol=symbol,
                timeframe="multi",
            )
        elif sell_score > buy_score and sell_score >= self.config.get(
            "signal_score_threshold", 1.0
        ):
            return TradingSignal(
                signal_type=SignalType.SELL,
                confidence=sell_score,
                conditions_met=sell_conditions,
                stop_loss=None,  # Would need to determine from multiple timeframes
                take_profit=None,  # Would need to determine from multiple timeframes
                timestamp=time.time(),
                symbol=symbol,
                timeframe="multi",
            )
        else:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                conditions_met=["No clear consensus"],
                stop_loss=None,
                take_profit=None,
                timestamp=time.time(),
                symbol=symbol,
                timeframe="multi",
            )


# --- Trading Analyzer ---
class TradingAnalyzer:
    """
    Performs technical analysis on candlestick data and generates trading signals.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        logger: logging.Logger,
        symbol: str,
        interval: str,
    ):
        self.df = df.copy()  # Work on a copy to avoid modifying original DataFrame
        self.config = config
        self.logger = logger
        self.symbol = symbol
        self.interval = interval
        self.weight_sets = config["weight_sets"]
        self.indicator_values: dict[str, Any] = {
            "order_book_walls": {"bullish": False, "bearish": False},
            "l2_metrics": {},
            "chandelier_exit": {"long": None, "short": None},
        }  # Stores calculated indicator values
        self.atr_value: float = 0.0  # Stores the latest ATR value

        # Initialize component analyzers
        self.indicator_calc = IndicatorCalculator(df, config, logger)
        self.sr_analyzer = SupportResistanceAnalyzer(df, config, logger)
        self.order_book_analyzer = OrderBookAnalyzer(config, logger)
        self.market_regime_detector = MarketRegimeDetector(df, config, logger)
        self.data_validator = DataValidator(config, logger)

        # Pre-calculate common indicators needed for others or for weight selection
        self._pre_calculate_indicators()

        # Now that ATR is potentially calculated, select the weight set
        self.user_defined_weights = (
            self._select_weight_set()
        )  # Dynamically selected weights

        # Detect market regime
        self.market_regime = self.market_regime_detector.detect_regime()

    def _pre_calculate_indicators(self):
        """Pre-calculates indicators necessary for weight selection or other calculations."""
        if not self.df.empty:
            # Calculate ATR once for volatility assessment
            atr_series = self.indicator_calc.calculate_atr(
                window=self.config["atr_period"]
            )
            if not atr_series.empty and not pd.isna(atr_series.iloc[-1]):
                self.atr_value = atr_series.iloc[-1]
            else:
                self.atr_value = 0.0  # Default ATR to 0 if calculation fails or is NaN

            self.indicator_values["atr"] = (
                self.atr_value
            )  # Store ATR for logging/analysis

            # Calculate momentum MAs for trend determination
            self.indicator_calc.df["momentum"] = (
                self.indicator_calc._safe_series_operation(
                    "close", "diff", self.config["momentum_period"]
                )
            )
            self.indicator_calc.df["momentum_ma_short"] = (
                self.indicator_calc._safe_series_operation(
                    None,
                    "sma",
                    self.config["momentum_ma_short"],
                    self.indicator_calc.df["momentum"],
                )
            )
            self.indicator_calc.df["momentum_ma_long"] = (
                self.indicator_calc._safe_series_operation(
                    None,
                    "sma",
                    self.config["momentum_ma_long"],
                    self.indicator_calc.df["momentum"],
                )
            )
            # Pre-calculate volume_ma for volume confirmation
            self.indicator_calc.df["volume_ma"] = (
                self.indicator_calc._safe_series_operation(
                    "volume", "sma", self.config["volume_ma_period"]
                )
            )

    def _select_weight_set(self) -> dict[str, float]:
        """
        Selects a weight set (e.g., low_volatility, high_volatility) based on current ATR.
        """
        # Use the atr_value that was pre-calculated in _pre_calculate_indicators
        if self.atr_value > self.config["atr_change_threshold"]:
            self.logger.debug(
                f"{NEON_YELLOW}Market detected as HIGH VOLATILITY (ATR: {self.atr_value:.4f}). Using 'high_volatility' weights.{RESET}"
            )
            return self.weight_sets.get(
                "high_volatility", self.weight_sets["low_volatility"]
            )

        self.logger.debug(
            f"{NEON_BLUE}Market detected as LOW VOLATILITY (ATR: {self.atr_value:.4f}). Using 'low_volatility' weights.{RESET}"
        )
        return self.weight_sets["low_volatility"]

    def get_color_for_value(self, name: str, value: float) -> str:
        """Returns the appropriate ANSI color for an indicator value."""
        try:
            name = name.lower()
            if name == "rsi":
                return (
                    NEON_GREEN
                    if value < 30
                    else NEON_RED
                    if value > 70
                    else NEON_YELLOW
                )
            if name == "mfi":
                return (
                    NEON_GREEN
                    if value < 20
                    else NEON_RED
                    if value > 80
                    else NEON_YELLOW
                )
            if name == "cci":
                return (
                    NEON_GREEN
                    if value < -100
                    else NEON_RED
                    if value > 100
                    else NEON_YELLOW
                )
            if name == "wr":
                return (
                    NEON_GREEN
                    if value < -80
                    else NEON_RED
                    if value > -20
                    else NEON_YELLOW
                )
            if name == "fve":
                return (
                    NEON_GREEN if value > 0 else NEON_RED if value < 0 else NEON_YELLOW
                )
            if name == "macd":
                return (
                    NEON_GREEN if value > 0 else NEON_RED if value < 0 else NEON_YELLOW
                )
            if name == "stc":
                return (
                    NEON_PURPLE
                    if value > 75
                    else NEON_RED
                    if value < 25
                    else NEON_YELLOW
                )
            if name == "cmo":
                return (
                    NEON_GREEN
                    if value > 50
                    else NEON_RED
                    if value < -50
                    else NEON_YELLOW
                )
            if "pnl" in name or "profit" in name:
                return (
                    NEON_GREEN if value > 0 else NEON_RED if value < 0 else NEON_WHITE
                )
        except Exception:
            pass
        return NEON_WHITE

    def analyze(
        self, current_price: Decimal, timestamp: str, order_book: dict[str, Any]
    ):
        """Perform full technical and L2 depth analysis, logging the current state."""
        high_dec = Decimal(str(self.df["high"].max()))
        low_dec = Decimal(str(self.df["low"].min()))
        close_dec = Decimal(str(self.df["close"].iloc[-1]))

        # 1. Support/Resistance & Pivot Points
        nearest_supports, nearest_resistances = [], []
        self.sr_analyzer.calculate_fibonacci_retracement(
            high_dec, low_dec, current_price
        )
        self.sr_analyzer.calculate_pivot_points(high_dec, low_dec, close_dec)
        nearest_supports, nearest_resistances = self.sr_analyzer.find_nearest_levels(
            current_price
        )

        # 2. Technical Indicators (Consolidated)
        self.indicator_values = self.indicator_calc.calculate_all_indicators()

        # 3. L2 Order Book Intelligence
        l2 = self.order_book_analyzer.analyze_order_book_l2_metrics(order_book)
        clusters = self.order_book_analyzer.find_liquidity_clusters(order_book)
        depth = self.order_book_analyzer.get_depth_profile(order_book, current_price)
        walls = self.order_book_analyzer.analyze_order_book_walls(
            order_book, current_price
        )

        self.indicator_values.update(
            {
                "l2_metrics": l2,
                "liquidity_clusters": clusters,
                "depth_profile": depth,
                "order_book_walls": {
                    "bullish": walls[0],
                    "bearish": walls[1],
                    "bullish_details": walls[2],
                    "bearish_details": walls[3],
                },
            }
        )

        # 4. Display Dashboard
        output = f"\n{NEON_CYAN}--- ZENITH APEX SCAN: {self.symbol} [{self.interval}] ---{RESET}\n"
        output += f"{NEON_BLUE}Price:{RESET} ${current_price:.5f} | {NEON_BLUE}Regime:{RESET} {self.market_regime.value.upper()} | {NEON_BLUE}ATR:{RESET} {self.atr_value:.5f}\n"

        if l2:
            output += f"{NEON_PURPLE}L2 Flow:{RESET} Imb(10): {l2['imbalance_10']:.2f} | Depth Ratio: {l2['bid_ask_ratio']:.2f}\n"

        # Indicator Grid
        dashboard = []
        for name in [
            "rsi",
            "mfi",
            "cci",
            "fve",
            "ehlers_fisher",
            "laguerre_rsi",
            "stc",
            "cmo",
        ]:
            val = self.indicator_values.get(name)
            if val is not None and not pd.isna(val):
                color = self.get_color_for_value(name, float(val))
                dashboard.append(f"{name.upper()}: {color}{float(val):.2f}{RESET}")

        for i in range(0, len(dashboard), 4):
            output += "  " + " | ".join(dashboard[i : i + 4]) + "\n"

        if walls[0] or walls[1]:
            output += f"{NEON_YELLOW}OB Walls:{RESET} {'Bullish' if walls[0] else ''} {'Bearish' if walls[1] else ''}\n"

        self.logger.info(output)

    def generate_trading_signal(self, current_price: Decimal) -> TradingSignal:
        """
        Weighted consensus engine for high-probability entries.

        Returns a TradingSignal object with all relevant information.
        """
        signal_score = Decimal("0.0")
        signal_type = SignalType.HOLD
        conditions_met: list[str] = []
        stop_loss = None
        take_profit = None

        # --- Bullish Signal Logic ---
        # Sum weights of bullish conditions met
        if (
            self.config["indicators"].get("stoch_rsi")
            and isinstance(self.indicator_values.get("stoch_rsi_vals"), pd.DataFrame)
            and not self.indicator_values["stoch_rsi_vals"].empty
        ):
            stoch_rsi_k = Decimal(
                str(self.indicator_values["stoch_rsi_vals"]["k"].iloc[-1])
            )
            stoch_rsi_d = Decimal(
                str(self.indicator_values["stoch_rsi_vals"]["d"].iloc[-1])
            )
            if (
                stoch_rsi_k < self.config["stoch_rsi_oversold_threshold"]
                and stoch_rsi_k > stoch_rsi_d
            ):
                signal_score += Decimal(str(self.user_defined_weights["stoch_rsi"]))
                conditions_met.append("Stoch RSI Oversold Crossover")

        if (
            self.config["indicators"].get("rsi")
            and self.indicator_values.get("rsi") is not None
        ):
            rsi_val = (
                self.indicator_values["rsi"].iloc[-1]
                if isinstance(self.indicator_values["rsi"], pd.Series)
                else self.indicator_values["rsi"]
            )
            if rsi_val < 30:
                signal_score += Decimal(str(self.user_defined_weights["rsi"]))
                conditions_met.append("RSI Oversold")

        if (
            self.config["indicators"].get("mfi")
            and self.indicator_values.get("mfi") is not None
        ):
            mfi_val = (
                self.indicator_values["mfi"].iloc[-1]
                if isinstance(self.indicator_values["mfi"], pd.Series)
                else self.indicator_values["mfi"]
            )
            if mfi_val < 20:
                signal_score += Decimal(str(self.user_defined_weights["mfi"]))
                conditions_met.append("MFI Oversold")

        if (
            self.config["indicators"].get("ema_alignment")
            and self.indicator_values.get("ema_alignment", 0.0) > 0
        ):
            signal_score += Decimal(
                str(self.user_defined_weights["ema_alignment"])
            ) * Decimal(
                str(abs(self.indicator_values["ema_alignment"]))
            )  # Scale by score
            conditions_met.append("Bullish EMA Alignment")

        if (
            self.config["indicators"].get("volume_confirmation")
            and self.indicator_calc.calculate_volume_confirmation()
        ):
            signal_score += Decimal(
                str(self.user_defined_weights["volume_confirmation"])
            )
            conditions_met.append("Volume Confirmation")

        if (
            self.config["indicators"].get("divergence")
            and self.indicator_calc.detect_macd_divergence() == "bullish"
        ):
            signal_score += Decimal(str(self.user_defined_weights["divergence"]))
            conditions_met.append("Bullish MACD Divergence")

        if self.config["indicators"].get("momentum") and "mom" in self.indicator_values:
            mom_data = self.indicator_values["mom"]
            if mom_data["trend"] == "Uptrend":
                signal_score += Decimal(str(self.user_defined_weights["momentum"])) * Decimal(str(mom_data["strength"]))
                conditions_met.append(f"Momentum Uptrend (Strength: {mom_data['strength']:.2f})")

        if (self.config["indicators"].get("macd") and self.indicator_values.get("macd")):
            macd_vals = self.indicator_values["macd"]
            macd_line = Decimal(str(macd_vals.get("macd", 0)))
            signal_line = Decimal(str(macd_vals.get("signal", 0)))
            if (macd_line > signal_line and macd_line > 0):
                signal_score += Decimal(str(self.user_defined_weights["macd"]))
                conditions_met.append("MACD Bullish Crossover")

        if self.indicator_values["order_book_walls"].get("bullish"):
            signal_score += Decimal(
                str(self.config["order_book_support_confidence_boost"])
            )
            conditions_met.append("Bullish Order Book Wall")

        # Fibonacci Retracement Support
        for label, level in self.sr_analyzer.fib_levels.items():
            if abs(current_price - level) / current_price < Decimal("0.005"):
                if level < current_price:
                    signal_score += Decimal(str(self.user_defined_weights.get('fib_retracement', 0)))
                    conditions_met.append(f"Near Fibonacci Support: {label}")

        # ADX Trend Confirmation
        if self.indicator_values.get("adx", 0) > 25:
            signal_score += Decimal("0.1")
            conditions_met.append("ADX Strong Trend Confirmation")

        # ADX +DI > -DI Bullish Signal
        if "adx_data" in self.indicator_values:
            if self.indicator_values["adx_data"]["plus_di"] > self.indicator_values["adx_data"]["minus_di"]:
                signal_score += Decimal(str(self.user_defined_weights.get('adx', 0)))
                conditions_met.append("ADX Bullish (+DI > -DI)")

        # New Indicator Bullish Logic
        if self.config["indicators"].get(
            "bollinger_bands"
        ) and self.indicator_values.get("bollinger_bands"):
            if current_price < Decimal(
                str(self.indicator_values["bollinger_bands"]["lower"])
            ):
                signal_score += Decimal(
                    str(self.user_defined_weights["bollinger_bands"])
                )
                conditions_met.append("Price Below Bollinger Lower Band")

        if self.config["indicators"].get(
            "awesome_oscillator"
        ) and self.indicator_values.get("awesome_oscillator") is not None:
            ao_series = self.indicator_values["awesome_oscillator"]
            if isinstance(ao_series, pd.Series) and len(ao_series) >= 2:
                if ao_series.iloc[-1] > 0 and ao_series.iloc[-2] <= 0:
                    signal_score += Decimal(
                        str(self.user_defined_weights["awesome_oscillator"])
                    )
                    conditions_met.append("Awesome Oscillator Bullish Zero-Cross")
            elif not isinstance(ao_series, pd.Series):
                # Fallback for single value or other types if necessary
                pass

        if self.config["indicators"].get("vortex") and self.indicator_values.get(
            "vortex"
        ):
            if (
                self.indicator_values["vortex"]["vi_plus"]
                > self.indicator_values["vortex"]["vi_minus"]
            ):
                signal_score += Decimal(str(self.user_defined_weights["vortex"]))
                conditions_met.append("Vortex Bullish Crossover")

        if self.indicator_values.get("l2_metrics"):
            if self.indicator_values["l2_metrics"].get("imbalance_10", 0) > 0.3:
                signal_score += Decimal("0.2")
                conditions_met.append("Strong L2 Imbalance (Top 10)")

        if self.indicator_values.get("depth_profile"):
            if self.indicator_values["depth_profile"].get("imbalance_0.5%", 0) > 0.4:
                signal_score += Decimal("0.3")
                conditions_met.append("Heavy Buy Liquidity (0.5% Range)")

        if self.indicator_values.get("liquidity_clusters"):
            # Check if current price is just above a major bid cluster
            for p, _q in self.indicator_values["liquidity_clusters"]["bids"]:
                if current_price > p and (current_price - p) / current_price < Decimal(
                    "0.002"
                ):
                    signal_score += Decimal("0.4")
                    conditions_met.append(
                        f"Price Near Heavy Support Cluster (${p:.2f})"
                    )
                    break

        # Stochastic Oscillator Bullish Signal
        if (
            self.config["indicators"].get("stochastic_oscillator")
            and isinstance(self.indicator_values.get("stoch_osc_vals"), pd.DataFrame)
            and not self.indicator_values["stoch_osc_vals"].empty
        ):
            stoch_k = Decimal(
                str(self.indicator_values["stoch_osc_vals"]["k"].iloc[-1])
            )
            stoch_d = Decimal(
                str(self.indicator_values["stoch_osc_vals"]["d"].iloc[-1])
            )
            if stoch_k < 20 and stoch_k > stoch_d:  # Oversold and K crossing above D
                signal_score += Decimal(
                    str(self.user_defined_weights.get("stochastic_oscillator", 0.4))
                )
                conditions_met.append("Stoch Oscillator Oversold Crossover")

        # EC5 Bullish Logic (Ehlers & Advanced)
        if (
            self.indicator_values.get("ehlers_fisher") is not None
            and isinstance(self.indicator_values["ehlers_fisher"], (pd.Series, np.ndarray, list))
            and len(self.indicator_values["ehlers_fisher"]) >= 2
        ):
            ef_series = self.indicator_values["ehlers_fisher"]
            ef_curr = ef_series.iloc[-1] if isinstance(ef_series, pd.Series) else ef_series[-1]
            ef_prev = ef_series.iloc[-2] if isinstance(ef_series, pd.Series) else ef_series[-2]
            if ef_curr > 0 and ef_prev <= 0:
                signal_score += Decimal(
                    str(self.user_defined_weights.get("ehlers_fisher", 0.5))
                )
                conditions_met.append("Ehlers Fisher Bullish Crossover")

        if (
            self.indicator_values.get("laguerre_rsi") is not None
        ):
            lrsi_series = self.indicator_values["laguerre_rsi"]
            lrsi_val = lrsi_series.iloc[-1] if isinstance(lrsi_series, pd.Series) else (lrsi_series[-1] if isinstance(lrsi_series, (np.ndarray, list)) else lrsi_series)
            if lrsi_val < 0.2:
                signal_score += Decimal(
                    str(self.user_defined_weights.get("laguerre_rsi", 0.4))
                )
                conditions_met.append("Laguerre RSI Oversold")

        if (
            self.indicator_values.get("supertrend")
            and self.indicator_values["supertrend"].get("direction") == 1
        ):
            signal_score += Decimal(
                str(self.user_defined_weights.get("supertrend", 0.3))
            )
            conditions_met.append("Supertrend Bullish Alignment")

        if (
            self.indicator_values.get("cmo") is not None
        ):
            cmo_series = self.indicator_values["cmo"]
            cmo_val = cmo_series.iloc[-1] if isinstance(cmo_series, pd.Series) else (cmo_series[-1] if isinstance(cmo_series, (np.ndarray, list)) else cmo_series)
            if cmo_val > 50:
                signal_score += Decimal(str(self.user_defined_weights.get("cmo", 0.3)))
                conditions_met.append("CMO Bullish Extreme")

        if (
            self.indicator_values.get("stc") is not None
        ):
            stc_series = self.indicator_values["stc"]
            stc_val = stc_series.iloc[-1] if isinstance(stc_series, pd.Series) else (stc_series[-1] if isinstance(stc_series, (np.ndarray, list)) else stc_series)
            if stc_val > 75:
                signal_score += Decimal(str(self.user_defined_weights.get("stc", 0.4)))
                conditions_met.append("STC Bullish Overbought (Strong Trend)")

        if (
            self.indicator_values.get("fve") is not None
            and isinstance(self.indicator_values["fve"], (pd.Series, np.ndarray, list))
            and len(self.indicator_values["fve"]) >= 2
        ):
            fve_series = self.indicator_values["fve"]
            fve_curr = fve_series.iloc[-1] if isinstance(fve_series, pd.Series) else fve_series[-1]
            fve_prev = fve_series.iloc[-2] if isinstance(fve_series, pd.Series) else fve_series[-2]
            if (
                fve_curr > 0
                and fve_curr > fve_prev
            ):
                signal_score += Decimal("0.3")
                conditions_met.append("FVE Bullish Money Flow")

        # Final check for Bullish signal
        if signal_score >= Decimal(str(self.config["signal_score_threshold"])):
            signal_type = SignalType.BUY
            # Calculate Stop Loss and Take Profit
            if self.atr_value > 0:
                stop_loss = (current_price - (
                    Decimal(str(self.atr_value))
                    * Decimal(str(self.config["stop_loss_multiple"]))
                )).quantize(Decimal('0.00001'))
                take_profit = (current_price + (
                    Decimal(str(self.atr_value))
                    * Decimal(str(self.config["take_profit_multiple"]))
                )).quantize(Decimal('0.00001'))

        # --- Bearish Signal Logic (similar structure) ---
        bearish_score = Decimal("0.0")
        bearish_conditions: list[str] = []

        if (
            self.config["indicators"].get("stoch_rsi")
            and isinstance(self.indicator_values.get("stoch_rsi_vals"), pd.DataFrame)
            and not self.indicator_values["stoch_rsi_vals"].empty
        ):
            stoch_rsi_k = Decimal(
                str(self.indicator_values["stoch_rsi_vals"]["k"].iloc[-1])
            )
            stoch_rsi_d = Decimal(
                str(self.indicator_values["stoch_rsi_vals"]["d"].iloc[-1])
            )
            if (
                stoch_rsi_k > self.config["stoch_rsi_overbought_threshold"]
                and stoch_rsi_k < stoch_rsi_d
            ):
                bearish_score += Decimal(str(self.user_defined_weights["stoch_rsi"]))
                bearish_conditions.append("Stoch RSI Overbought Crossover")

        if (
            self.config["indicators"].get("rsi")
            and self.indicator_values.get("rsi")
            and self.indicator_values["rsi"][-1] > 70
        ):
            bearish_score += Decimal(str(self.user_defined_weights["rsi"]))
            bearish_conditions.append("RSI Overbought")

        if (
            self.config["indicators"].get("mfi")
            and self.indicator_values.get("mfi")
            and self.indicator_values["mfi"][-1] > 80
        ):
            bearish_score += Decimal(str(self.user_defined_weights["mfi"]))
            bearish_conditions.append("MFI Overbought")

        if (
            self.config["indicators"].get("ema_alignment")
            and self.indicator_values.get("ema_alignment", 0.0) < 0
        ):
            bearish_score += Decimal(
                str(self.user_defined_weights["ema_alignment"])
            ) * Decimal(str(abs(self.indicator_values["ema_alignment"])))
            bearish_conditions.append("Bearish EMA Alignment")

        if (
            self.config["indicators"].get("divergence")
            and self.indicator_calc.detect_macd_divergence() == "bearish"
        ):
            bearish_score += Decimal(str(self.user_defined_weights["divergence"]))
            bearish_conditions.append("Bearish MACD Divergence")

        if self.config["indicators"].get("momentum") and "mom" in self.indicator_values:
            mom_data = self.indicator_values["mom"]
            if mom_data["trend"] == "Downtrend":
                bearish_score += Decimal(str(self.user_defined_weights["momentum"])) * Decimal(str(mom_data["strength"]))
                bearish_conditions.append(f"Momentum Downtrend (Strength: {mom_data['strength']:.2f})")

        if (self.config["indicators"].get("macd") and self.indicator_values.get("macd")):
            macd_vals = self.indicator_values["macd"]
            macd_line = Decimal(str(macd_vals.get("macd", 0)))
            signal_line = Decimal(str(macd_vals.get("signal", 0)))
            if (macd_line < signal_line and macd_line < 0):
                bearish_score += Decimal(str(self.user_defined_weights["macd"]))
                bearish_conditions.append("MACD Bearish Crossover")

        if self.indicator_values["order_book_walls"].get("bearish"):
            bearish_score += Decimal(
                str(self.config["order_book_resistance_confidence_boost"])
            )
            bearish_conditions.append("Bearish Order Book Wall")

        # Fibonacci Retracement Resistance
        for label, level in self.sr_analyzer.fib_levels.items():
            if abs(current_price - level) / current_price < Decimal("0.005"):
                if level > current_price:
                    bearish_score += Decimal(str(self.user_defined_weights.get('fib_retracement', 0)))
                    bearish_conditions.append(f"Near Fibonacci Resistance: {label}")

        # ADX -DI > +DI Bearish Signal
        if "adx_data" in self.indicator_values:
            if self.indicator_values["adx_data"]["minus_di"] > self.indicator_values["adx_data"]["plus_di"]:
                bearish_score += Decimal(str(self.user_defined_weights.get('adx', 0)))
                bearish_conditions.append("ADX Bearish (-DI > +DI)")

        # New Indicator Bearish Logic
        if self.config["indicators"].get(
            "bollinger_bands"
        ) and self.indicator_values.get("bollinger_bands"):
            if current_price > Decimal(
                str(self.indicator_values["bollinger_bands"]["upper"])
            ):
                bearish_score += Decimal(
                    str(self.user_defined_weights["bollinger_bands"])
                )
                bearish_conditions.append("Price Above Bollinger Upper Band")

        if self.config["indicators"].get(
            "awesome_oscillator"
        ) and self.indicator_values.get("awesome_oscillator"):
            if (
                self.indicator_values["awesome_oscillator"][-1] < 0
                and self.indicator_values["awesome_oscillator"][-2] >= 0
            ):
                bearish_score += Decimal(
                    str(self.user_defined_weights["awesome_oscillator"])
                )
                bearish_conditions.append("Awesome Oscillator Bearish Zero-Cross")

        if self.config["indicators"].get("vortex") and self.indicator_values.get(
            "vortex"
        ):
            if (
                self.indicator_values["vortex"]["vi_minus"]
                > self.indicator_values["vortex"]["vi_plus"]
            ):
                bearish_score += Decimal(str(self.user_defined_weights["vortex"]))
                bearish_conditions.append("Vortex Bearish Crossover")

        if self.indicator_values.get("l2_metrics"):
            if self.indicator_values["l2_metrics"].get("imbalance_10", 0) < -0.3:
                bearish_score += Decimal("0.2")
                bearish_conditions.append("Strong L2 Sell Imbalance (Top 10)")


        if self.indicator_values.get("depth_profile"):
            if self.indicator_values["depth_profile"].get("imbalance_0.5%", 0) < -0.4:
                bearish_score += Decimal("0.3")
                bearish_conditions.append("Heavy Sell Liquidity (0.5% Range)")

        if self.indicator_values.get("liquidity_clusters"):
            # Check if current price is just below a major ask cluster
            for p, _q in self.indicator_values["liquidity_clusters"]["asks"]:
                if current_price < p and (p - current_price) / current_price < Decimal(
                    "0.002"
                ):
                    bearish_score += Decimal("0.4")
                    bearish_conditions.append(
                        f"Price Near Heavy Resistance Cluster (${p:.2f})"
                    )
                    break

        # Stochastic Oscillator Bearish Signal
        if (
            self.config["indicators"].get("stochastic_oscillator")
            and isinstance(self.indicator_values.get("stoch_osc_vals"), pd.DataFrame)
            and not self.indicator_values["stoch_osc_vals"].empty
        ):
            stoch_k = Decimal(
                str(self.indicator_values["stoch_osc_vals"]["k"].iloc[-1])
            )
            stoch_d = Decimal(
                str(self.indicator_values["stoch_osc_vals"]["d"].iloc[-1])
            )
            if stoch_k > 80 and stoch_k < stoch_d:  # Overbought and K crossing below D
                bearish_score += Decimal(
                    str(self.user_defined_weights["stochastic_oscillator"])
                )
                bearish_conditions.append("Stoch Oscillator Overbought Crossover")

        # EC5 Bearish Logic
        if (
            self.indicator_values.get("ehlers_fisher") is not None
            and isinstance(self.indicator_values["ehlers_fisher"], (pd.Series, np.ndarray, list))
            and len(self.indicator_values["ehlers_fisher"]) >= 2
        ):
            ef_series = self.indicator_values["ehlers_fisher"]
            ef_curr = ef_series.iloc[-1] if isinstance(ef_series, pd.Series) else ef_series[-1]
            ef_prev = ef_series.iloc[-2] if isinstance(ef_series, pd.Series) else ef_series[-2]
            if ef_curr < 0 and ef_prev >= 0:
                bearish_score += Decimal("0.5")
                bearish_conditions.append("Ehlers Fisher Bearish Crossover")

        if (
            self.indicator_values.get("laguerre_rsi") is not None
        ):
            lrsi_series = self.indicator_values["laguerre_rsi"]
            lrsi_val = lrsi_series.iloc[-1] if isinstance(lrsi_series, pd.Series) else (lrsi_series[-1] if isinstance(lrsi_series, (np.ndarray, list)) else lrsi_series)
            if lrsi_val > 0.8:
                bearish_score += Decimal("0.4")
                bearish_conditions.append("Laguerre RSI Overbought")


        if (
            self.indicator_values.get("supertrend")
            and self.indicator_values["supertrend"].get("direction") == -1
        ):
            bearish_score += Decimal("0.3")
            bearish_conditions.append("Supertrend Bearish Alignment")

        if (
            self.indicator_values.get("cmo") is not None
        ):
            cmo_series = self.indicator_values["cmo"]
            cmo_val = cmo_series.iloc[-1] if isinstance(cmo_series, pd.Series) else (cmo_series[-1] if isinstance(cmo_series, (np.ndarray, list)) else cmo_series)
            if cmo_val < -50:
                bearish_score += Decimal("0.3")
                bearish_conditions.append("CMO Bearish Extreme")

        if (
            self.indicator_values.get("stc") is not None
        ):
            stc_series = self.indicator_values["stc"]
            stc_val = stc_series.iloc[-1] if isinstance(stc_series, pd.Series) else (stc_series[-1] if isinstance(stc_series, (np.ndarray, list)) else stc_series)
            if stc_val < 25:
                bearish_score += Decimal("0.4")
                bearish_conditions.append("STC Bearish Oversold (Strong Trend)")

        if (
            self.indicator_values.get("fve") is not None
            and isinstance(self.indicator_values["fve"], (pd.Series, np.ndarray, list))
            and len(self.indicator_values["fve"]) >= 2
        ):
            fve_series = self.indicator_values["fve"]
            fve_curr = fve_series.iloc[-1] if isinstance(fve_series, pd.Series) else fve_series[-1]
            fve_prev = fve_series.iloc[-2] if isinstance(fve_series, pd.Series) else fve_series[-2]
            if (
                fve_curr < 0
                and fve_curr < fve_prev
            ):
                bearish_score += Decimal("0.3")
                bearish_conditions.append("FVE Bearish Money Flow")

        # Final check for Bearish signal (only if no bullish signal already)
        if signal_type == SignalType.HOLD and bearish_score >= Decimal(
            str(self.config["signal_score_threshold"])
        ):
            signal_type = SignalType.SELL
            signal_score = bearish_score  # Use bearish score if it's the chosen signal
            conditions_met = bearish_conditions  # Use bearish conditions

            # Calculate Stop Loss and Take Profit for sell signal
            if self.atr_value > 0:
                stop_loss = (current_price + (
                    Decimal(str(self.atr_value))
                    * Decimal(str(self.config["stop_loss_multiple"]))
                )).quantize(Decimal('0.00001'))
                take_profit = (current_price - (
                    Decimal(str(self.atr_value))
                    * Decimal(str(self.config["take_profit_multiple"]))
                )).quantize(Decimal('0.00001'))

        # Calculate risk/reward ratio
        risk_reward_ratio = None
        if stop_loss and take_profit and signal_type != SignalType.HOLD:
            if signal_type == SignalType.BUY:
                risk = float(current_price - stop_loss)
                reward = float(take_profit - current_price)
            else:  # SELL
                risk = float(stop_loss - current_price)
                reward = float(current_price - take_profit)

            risk_reward_ratio = reward / risk if risk > 0 else None

        return TradingSignal(
            signal_type=signal_type,
            confidence=float(signal_score),
            conditions_met=conditions_met,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=time.time(),
            symbol=self.symbol,
            timeframe=self.interval,
            risk_reward_ratio=risk_reward_ratio,
        )


# --- Signal History Tracker ---
class SignalGenerator:
    """Generates a trading signal (buy/sell/hold) based on weighted indicator values and market regime."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.weight_sets = config["weight_sets"]
        self.signal_score_threshold = Decimal(
            str(config.get("signal_score_threshold", 1.0))
        )

    def _get_indicator_weight(
        self, indicator_name: str, market_regime: MarketRegime
    ) -> float:
        """Retrieves the weight for a given indicator based on the current market regime."""
        weights = self.weight_sets.get(
            market_regime.value.lower(), self.weight_sets["low_volatility"]
        )
        return weights.get(indicator_name, 0.0)

    def generate_signal(
        self,
        indicator_values: dict[str, Any],
        previous_values: dict[str, Any],
        market_regime: MarketRegime,
        current_price: Decimal,
        atr_value: Decimal,
    ) -> TradingSignal:
        """
        Generates a trading signal (buy/sell/hold) based on indicator values,
        market regime, and weighted scoring.
        """
        bullish_score = Decimal("0.0")
        bearish_score = Decimal("0.0")
        conditions_met: list[str] = []

        # --- Evaluate Indicators and Apply Weights ---

        # EMA Alignment
        if (
            self.config["indicators"].get("ema_alignment")
            and "ema_alignment" in indicator_values
        ):
            ema_align_value = indicator_values["ema_alignment"]
            weight = Decimal(
                str(self._get_indicator_weight("ema_alignment", market_regime))
            )
            if ema_align_value > 0:  # Bullish alignment or crossover
                bullish_score += weight * Decimal(str(abs(ema_align_value)))
                conditions_met.append("Bullish EMA Alignment")
            elif ema_align_value < 0:  # Bearish alignment or crossover
                bearish_score += weight * Decimal(str(abs(ema_align_value)))
                conditions_met.append("Bearish EMA Alignment")

        # Momentum (trend based on MA crossover)
        if self.config["indicators"].get("momentum") and "mom" in indicator_values:
            mom_data = indicator_values["mom"]
            if mom_data["trend"] == "Uptrend":
                bullish_score += Decimal(
                    str(self._get_indicator_weight("momentum", market_regime))
                ) * Decimal(str(mom_data["strength"]))
                conditions_met.append(
                    f"Momentum Uptrend (Strength: {mom_data['strength']:.2f})"
                )
            elif mom_data["trend"] == "Downtrend":
                bearish_score += Decimal(
                    str(self._get_indicator_weight("momentum", market_regime))
                ) * Decimal(str(mom_data["strength"]))
                conditions_met.append(
                    f"Momentum Downtrend (Strength: {mom_data['strength']:.2f})"
                )

        # Divergence (MACD) - this will be passed from TradingAnalyzer
        if (
            self.config["indicators"].get("divergence")
            and "macd_divergence" in indicator_values
        ):
            divergence = indicator_values["macd_divergence"]
            weight = Decimal(
                str(self._get_indicator_weight("divergence", market_regime))
            )
            if divergence == "bullish":
                bullish_score += weight
                conditions_met.append("Bullish MACD Divergence")
            elif divergence == "bearish":
                bearish_score += weight
                conditions_met.append("Bearish MACD Divergence")

        # Stoch RSI
        if (
            self.config["indicators"].get("stoch_rsi")
            and "stoch_rsi" in indicator_values
            and indicator_values["stoch_rsi"]
        ):
            stoch_rsi = indicator_values["stoch_rsi"]
            if stoch_rsi.get("k") is not None and stoch_rsi.get("d") is not None:
                stoch_k = Decimal(str(stoch_rsi["k"]))
                stoch_d = Decimal(str(stoch_rsi["d"]))
                threshold_oversold = Decimal(
                    str(self.config["stoch_rsi_oversold_threshold"])
                )
                threshold_overbought = Decimal(
                    str(self.config["stoch_rsi_overbought_threshold"])
                )
                weight = Decimal(
                    str(self._get_indicator_weight("stoch_rsi", market_regime))
                )

                if stoch_k < threshold_oversold and stoch_k > stoch_d:
                    bullish_score += weight
                    conditions_met.append("Stoch RSI Oversold Crossover")
                elif stoch_k > threshold_overbought and stoch_k < stoch_d:
                    bearish_score += weight
                    conditions_met.append("Stoch RSI Overbought Crossover")

        # RSI
        if (
            self.config["indicators"].get("rsi")
            and "rsi" in indicator_values
            and not pd.isna(indicator_values["rsi"])
        ):
            rsi_val = Decimal(str(indicator_values["rsi"]))
            weight = Decimal(str(self._get_indicator_weight("rsi", market_regime)))
            if rsi_val < 30:
                bullish_score += weight
                conditions_met.append("RSI Oversold")
            elif rsi_val > 70:
                bearish_score += weight
                conditions_met.append("RSI Overbought")

        # MACD
        if (
            self.config["indicators"].get("macd")
            and "macd" in indicator_values
            and indicator_values["macd"]
        ):
            macd_vals = indicator_values["macd"]
            macd_line = Decimal(str(macd_vals.get("macd", 0)))
            signal_line = Decimal(str(macd_vals.get("signal", 0)))
            weight = Decimal(str(self._get_indicator_weight("macd", market_regime)))
            if (
                macd_line > signal_line and macd_line > 0
            ):  # Bullish crossover and positive territory
                bullish_score += weight
                conditions_met.append("MACD Bullish Crossover")
            elif (
                macd_line < signal_line and macd_line < 0
            ):  # Bearish crossover and negative territory
                bearish_score += weight
                conditions_met.append("MACD Bearish Crossover")

        # VWAP
        if (
            self.config["indicators"].get("vwap")
            and "vwap" in indicator_values
            and not pd.isna(indicator_values["vwap"])
        ):
            vwap_val = Decimal(str(indicator_values["vwap"]))
            weight = Decimal(str(self._get_indicator_weight("vwap", market_regime)))
            if current_price > vwap_val:
                bullish_score += weight
                conditions_met.append("Price Above VWAP")
            elif current_price < vwap_val:
                bearish_score += weight
                conditions_met.append("Price Below VWAP")

        # OBV (On-Balance Volume) - needs previous value to determine trend
        if (
            self.config["indicators"].get("obv")
            and "obv" in indicator_values
            and "obv" in previous_values
        ):
            current_obv = Decimal(str(indicator_values["obv"]))
            prev_obv = Decimal(str(previous_values["obv"]))
            weight = Decimal(str(self._get_indicator_weight("obv", market_regime)))
            if current_obv > prev_obv:
                bullish_score += weight
                conditions_met.append("OBV Increasing")
            elif current_obv < prev_obv:
                bearish_score += weight
                conditions_met.append("OBV Decreasing")

        # ADI (Accumulation/Distribution Index)
        if (
            self.config["indicators"].get("adi")
            and "adi" in indicator_values
            and "adi" in previous_values
        ):
            current_adi = Decimal(str(indicator_values["adi"]))
            prev_adi = Decimal(str(previous_values["adi"]))
            weight = Decimal(str(self._get_indicator_weight("adi", market_regime)))
            if current_adi > prev_adi:
                bullish_score += weight
                conditions_met.append("ADI Increasing (Accumulation)")
            elif current_adi < prev_adi:
                bearish_score += weight
                conditions_met.append("ADI Decreasing (Distribution)")

        # CCI (Commodity Channel Index)
        if (
            self.config["indicators"].get("cci")
            and "cci" in indicator_values
            and not pd.isna(indicator_values["cci"])
        ):
            cci_val = Decimal(str(indicator_values["cci"]))
            weight = Decimal(str(self._get_indicator_weight("cci", market_regime)))
            if cci_val > 100:
                bullish_score += weight
                conditions_met.append("CCI > 100 (Strong Buy)")
            elif cci_val < -100:
                bearish_score += weight
                conditions_met.append("CCI < -100 (Strong Sell)")

        # WR (Williams %R)
        if (
            self.config["indicators"].get("wr")
            and "wr" in indicator_values
            and not pd.isna(indicator_values["wr"])
        ):
            wr_val = Decimal(str(indicator_values["wr"]))
            weight = Decimal(str(self._get_indicator_weight("wr", market_regime)))
            if wr_val < -80:
                bullish_score += weight
                conditions_met.append("Williams %R Oversold")
            elif wr_val > -20:
                bearish_score += weight
                conditions_met.append("Williams %R Overbought")

        # ADX (Average Directional Index)
        if (
            self.config["indicators"].get("adx")
            and "adx" in indicator_values
            and not pd.isna(indicator_values["adx"])
        ):
            adx_val = Decimal(str(indicator_values["adx"]))
            if adx_val > 25:
                conditions_met.append(f"ADX Confirms Strong Trend ({adx_val:.2f})")

        # ADX +DI / -DI Directional Signal
        if (
            self.config["indicators"].get("adx")
            and "adx_data" in indicator_values
        ):
            adx_data = indicator_values["adx_data"]
            plus_di = Decimal(str(adx_data.get("plus_di", 0)))
            minus_di = Decimal(str(adx_data.get("minus_di", 0)))
            weight = Decimal(str(self._get_indicator_weight("adx", market_regime)))

            if plus_di > minus_di:
                bullish_score += weight
                conditions_met.append("ADX Bullish (+DI > -DI)")
            elif minus_di > plus_di:
                bearish_score += weight
                conditions_met.append("ADX Bearish (-DI > +DI)")

        # PSAR (Parabolic SAR)
        if (
            self.config["indicators"].get("psar")
            and "psar" in indicator_values
            and not pd.isna(indicator_values["psar"])
        ):
            psar_val = Decimal(str(indicator_values["psar"]))
            weight = Decimal(str(self._get_indicator_weight("psar", market_regime)))
            if current_price > psar_val:
                bullish_score += weight
                conditions_met.append("PSAR Bullish (Price Above SAR)")
            elif current_price < psar_val:
                bearish_score += weight
                conditions_met.append("PSAR Bearish (Price Below SAR)")

        # FVE (Finite Volume Element)
        if (
            self.config["indicators"].get("fve")
            and "fve" in indicator_values
            and not pd.isna(indicator_values["fve"])
        ):
            fve_val = Decimal(str(indicator_values["fve"]))
            weight = Decimal(str(self._get_indicator_weight("fve", market_regime)))
            if fve_val > 0:
                bullish_score += weight
                conditions_met.append("FVE Positive (Money Flow In)")
            elif fve_val < 0:
                bearish_score += weight
                conditions_met.append("FVE Negative (Money Flow Out)")

        # MFI (Money Flow Index)
        if (
            self.config["indicators"].get("mfi")
            and "mfi" in indicator_values
            and not pd.isna(indicator_values["mfi"])
        ):
            mfi_val = Decimal(str(indicator_values["mfi"]))
            weight = Decimal(str(self._get_indicator_weight("mfi", market_regime)))
            if mfi_val < 20:
                bullish_score += weight
                conditions_met.append("MFI Oversold")
            elif mfi_val > 80:
                bearish_score += weight
                conditions_met.append("MFI Overbought")

        # Stochastic Oscillator
        if (
            self.config["indicators"].get("stochastic_oscillator")
            and "stoch_oscillator" in indicator_values
            and indicator_values["stoch_oscillator"]
        ):
            stoch_osc = indicator_values["stoch_oscillator"]
            if stoch_osc.get("k") is not None and stoch_osc.get("d") is not None:
                stoch_k = Decimal(str(stoch_osc["k"]))
                stoch_d = Decimal(str(stoch_osc["d"]))
                weight = Decimal(
                    str(
                        self._get_indicator_weight(
                            "stochastic_oscillator", market_regime
                        )
                    )
                )
                if stoch_k < 20 and stoch_k > stoch_d:
                    bullish_score += weight
                    conditions_met.append("Stoch Oscillator Oversold Crossover")
                elif stoch_k > 80 and stoch_k < stoch_d:
                    bearish_score += weight
                    conditions_met.append("Stoch Oscillator Overbought Crossover")

        # Bollinger Bands
        if (
            self.config["indicators"].get("bollinger_bands")
            and "bollinger_bands" in indicator_values
            and indicator_values["bollinger_bands"]
        ):
            bb = indicator_values["bollinger_bands"]
            upper = Decimal(str(bb.get("upper", 0)))
            lower = Decimal(str(bb.get("lower", 0)))
            weight = Decimal(
                str(self._get_indicator_weight("bollinger_bands", market_regime))
            )
            if current_price < lower:
                bullish_score += weight
                conditions_met.append("Price Below Bollinger Lower Band")
            elif current_price > upper:
                bearish_score += weight
                conditions_met.append("Price Above Bollinger Upper Band")

        # Keltner Channels
        if (
            self.config["indicators"].get("keltner_channels")
            and "keltner_channels" in indicator_values
            and indicator_values["keltner_channels"]
        ):
            kc = indicator_values["keltner_channels"]
            upper = Decimal(str(kc.get("upper", 0)))
            lower = Decimal(str(kc.get("lower", 0)))
            weight = Decimal(
                str(self._get_indicator_weight("keltner_channels", market_regime))
            )
            if current_price < lower:
                bullish_score += weight
                conditions_met.append("Price Below Keltner Lower Channel")
            elif current_price > upper:
                bearish_score += weight
                conditions_met.append("Price Above Keltner Upper Channel")

        # Ichimoku Cloud (simplified)
        if (
            self.config["indicators"].get("ichimoku_cloud")
            and "ichimoku_cloud" in indicator_values
            and indicator_values["ichimoku_cloud"]
        ):
            ichimoku = indicator_values["ichimoku_cloud"]
            span_a = Decimal(str(ichimoku.get("span_a", 0)))  # Senkou Span A
            span_b = Decimal(str(ichimoku.get("span_b", 0)))  # Senkou Span B
            kijun = Decimal(str(ichimoku.get("kijun", 0)))
            tenkan = Decimal(str(ichimoku.get("tenkan", 0)))

            weight = Decimal(
                str(self._get_indicator_weight("ichimoku_cloud", market_regime))
            )

            if span_a > span_b and current_price > max(span_a, span_b):
                bullish_score += weight
                conditions_met.append("Ichimoku Price Above Green Cloud")
            elif span_a < span_b and current_price < min(span_a, span_b):
                bearish_score += weight
                conditions_met.append("Ichimoku Price Below Red Cloud")

            if tenkan > kijun:
                bullish_score += weight * Decimal("0.5")
                conditions_met.append("Ichimoku Tenkan/Kijun Bullish")
            elif tenkan < kijun:
                bearish_score += weight * Decimal("0.5")
                conditions_met.append("Ichimoku Tenkan/Kijun Bearish")

        # CMF (Chaikin Money Flow)
        if (
            self.config["indicators"].get("cmf")
            and "cmf" in indicator_values
            and not pd.isna(indicator_values["cmf"])
        ):
            cmf_val = Decimal(str(indicator_values["cmf"]))
            weight = Decimal(str(self._get_indicator_weight("cmf", market_regime)))
            if cmf_val > 0.05:
                bullish_score += weight
                conditions_met.append("CMF Positive Money Flow")
            elif cmf_val < -0.05:
                bearish_score += weight
                conditions_met.append("CMF Negative Money Flow")

        # EMV (Ease of Movement)
        if (
            self.config["indicators"].get("emv")
            and "emv" in indicator_values
            and not pd.isna(indicator_values["emv"])
        ):
            emv_val = Decimal(str(indicator_values["emv"]))
            weight = Decimal(str(self._get_indicator_weight("emv", market_regime)))
            if emv_val > 0:
                bullish_score += weight
                conditions_met.append("EMV Positive (Easy Upward Movement)")
            elif emv_val < 0:
                bearish_score += weight
                conditions_met.append("EMV Negative (Easy Downward Movement)")

        # Force Index
        if (
            self.config["indicators"].get("force_index")
            and "force_index" in indicator_values
            and not pd.isna(indicator_values["force_index"])
        ):
            fi_val = Decimal(str(indicator_values["force_index"]))
            weight = Decimal(
                str(self._get_indicator_weight("force_index", market_regime))
            )
            if fi_val > 0:
                bullish_score += weight
                conditions_met.append("Force Index Positive")
            elif fi_val < 0:
                bearish_score += weight
                conditions_met.append("Force Index Negative")

        # Mass Index
        if (
            self.config["indicators"].get("mass_index")
            and "mass_index" in indicator_values
            and not pd.isna(indicator_values["mass_index"])
        ):
            mi_val = Decimal(str(indicator_values["mass_index"]))
            weight = Decimal(
                str(self._get_indicator_weight("mass_index", market_regime))
            )
            if mi_val > 27:
                conditions_met.append("Mass Index Reversal Warning")

        # ROC (Rate of Change)
        if (
            self.config["indicators"].get("roc")
            and "roc" in indicator_values
            and not pd.isna(indicator_values["roc"])
        ):
            roc_val = Decimal(str(indicator_values["roc"]))
            weight = Decimal(str(self._get_indicator_weight("roc", market_regime)))
            if roc_val > 0:
                bullish_score += weight
                conditions_met.append("ROC Positive")
            elif roc_val < 0:
                bearish_score += weight
                conditions_met.append("ROC Negative")

        # TRIX
        if (
            self.config["indicators"].get("trix")
            and "trix" in indicator_values
            and not pd.isna(indicator_values["trix"])
        ):
            trix_val = Decimal(str(indicator_values["trix"]))
            weight = Decimal(str(self._get_indicator_weight("trix", market_regime)))
            if trix_val > 0:
                bullish_score += weight
                conditions_met.append("TRIX Positive")
            elif trix_val < 0:
                bearish_score += weight
                conditions_met.append("TRIX Negative")

        # Ultimate Oscillator
        if (
            self.config["indicators"].get("ultimate_oscillator")
            and "ultimate_oscillator" in indicator_values
            and not pd.isna(indicator_values["ultimate_oscillator"])
        ):
            uo_val = Decimal(str(indicator_values["ultimate_oscillator"]))
            weight = Decimal(
                str(self._get_indicator_weight("ultimate_oscillator", market_regime))
            )
            if uo_val > 70:
                bearish_score += weight
                conditions_met.append("Ultimate Oscillator Overbought")
            elif uo_val < 30:
                bullish_score += weight
                conditions_met.append("Ultimate Oscillator Oversold")

        # Vortex
        if (
            self.config["indicators"].get("vortex")
            and "vortex" in indicator_values
            and indicator_values["vortex"]
        ):
            vortex_vals = indicator_values["vortex"]
            vi_plus = Decimal(str(vortex_vals.get("vi_plus", 0)))
            vi_minus = Decimal(str(vortex_vals.get("vi_minus", 0)))
            weight = Decimal(str(self._get_indicator_weight("vortex", market_regime)))
            if vi_plus > vi_minus:
                bullish_score += weight
                conditions_met.append("Vortex Bullish Cross")
            elif vi_minus > vi_plus:
                bearish_score += weight
                conditions_met.append("Vortex Bearish Cross")

        # Coppock Curve
        if (
            self.config["indicators"].get("coppock_curve")
            and "coppock_curve" in indicator_values
            and not pd.isna(indicator_values["coppock_curve"])
        ):
            cc_val = Decimal(str(indicator_values["coppock_curve"]))
            weight = Decimal(
                str(self._get_indicator_weight("coppock_curve", market_regime))
            )
            if cc_val > 0:
                bullish_score += weight
                conditions_met.append("Coppock Curve Positive")
            elif cc_val < 0:
                bearish_score += weight
                conditions_met.append("Coppock Curve Negative")

        # Donchian Channels
        if (
            self.config["indicators"].get("donchian_channels")
            and "donchian_channels" in indicator_values
            and indicator_values["donchian_channels"]
        ):
            dc = indicator_values["donchian_channels"]
            upper = Decimal(str(dc.get("upper", 0)))
            lower = Decimal(str(dc.get("lower", 0)))
            weight = Decimal(
                str(self._get_indicator_weight("donchian_channels", market_regime))
            )
            if current_price > upper:
                bullish_score += weight
                conditions_met.append("Price Above Donchian Upper Channel")
            elif current_price < lower:
                bearish_score += weight
                conditions_met.append("Price Below Donchian Lower Channel")

        # HMA (Hull Moving Average)
        if (
            self.config["indicators"].get("hma")
            and "hma" in indicator_values
            and not pd.isna(indicator_values["hma"])
        ):
            hma_val = Decimal(str(indicator_values["hma"]))
            weight = Decimal(str(self._get_indicator_weight("hma", market_regime)))
            if hma_val > current_price:
                bearish_score += weight
                conditions_met.append("HMA Bearish (Price Below)")
            elif hma_val < current_price:
                bullish_score += weight
                conditions_met.append("HMA Bullish (Price Above)")

        # Awesome Oscillator
        if (
            self.config["indicators"].get("awesome_oscillator")
            and "awesome_oscillator" in indicator_values
            and not pd.isna(indicator_values["awesome_oscillator"])
        ):
            ao_val = Decimal(str(indicator_values["awesome_oscillator"]))
            weight = Decimal(
                str(self._get_indicator_weight("awesome_oscillator", market_regime))
            )
            if ao_val > 0:
                bullish_score += weight
                conditions_met.append("Awesome Oscillator Positive")
            elif ao_val < 0:
                bearish_score += weight
                conditions_met.append("Awesome Oscillator Negative")

        # Klinger Oscillator
        if (
            self.config["indicators"].get("klinger_oscillator")
            and "klinger_oscillator" in indicator_values
            and indicator_values["klinger_oscillator"]
        ):
            ko_vals = indicator_values["klinger_oscillator"]
            ko = Decimal(str(ko_vals.get("ko", 0)))
            signal_line = Decimal(str(ko_vals.get("signal", 0)))
            weight = Decimal(
                str(self._get_indicator_weight("klinger_oscillator", market_regime))
            )
            if ko > signal_line:
                bullish_score += weight
                conditions_met.append("Klinger Oscillator Bullish Cross")
            elif ko < signal_line:
                bearish_score += weight
                conditions_met.append("Klinger Oscillator Bearish Cross")

        # NVI (Negative Volume Index)
        if (
            self.config["indicators"].get("nvi")
            and "nvi" in indicator_values
            and not pd.isna(indicator_values["nvi"])
        ):
            # NVI requires historical context for trend, simplify to current > previous
            if "nvi" in previous_values:
                current_nvi = Decimal(str(indicator_values["nvi"]))
                prev_nvi = Decimal(str(previous_values["nvi"]))
                weight = Decimal(str(self._get_indicator_weight("nvi", market_regime)))
                if current_nvi > prev_nvi:
                    bullish_score += weight
                    conditions_met.append("NVI Increasing (Bullish Low Volume)")
                elif current_nvi < prev_nvi:
                    bearish_score += weight
                    conditions_met.append("NVI Decreasing (Bearish Low Volume)")

        # PVI (Positive Volume Index)
        if (
            self.config["indicators"].get("pvi")
            and "pvi" in indicator_values
            and not pd.isna(indicator_values["pvi"])
        ):
            # PVI requires historical context for trend, simplify to current > previous
            if "pvi" in previous_values:
                current_pvi = Decimal(str(indicator_values["pvi"]))
                prev_pvi = Decimal(str(previous_values["pvi"]))
                weight = Decimal(str(self._get_indicator_weight("pvi", market_regime)))
                if current_pvi > prev_pvi:
                    bullish_score += weight
                    conditions_met.append("PVI Increasing (Bullish High Volume)")
                elif current_pvi < prev_pvi:
                    bearish_score += weight
                    conditions_met.append("PVI Decreasing (Bearish High Volume)")

        # BOP (Balance of Power)
        if (
            self.config["indicators"].get("bop")
            and "bop" in indicator_values
            and not pd.isna(indicator_values["bop"])
        ):
            bop_val = Decimal(str(indicator_values["bop"]))
            weight = Decimal(str(self._get_indicator_weight("bop", market_regime)))
            if bop_val > 0:
                bullish_score += weight
                conditions_met.append("BOP Positive (Buying Pressure)")
            elif bop_val < 0:
                bearish_score += weight
                conditions_met.append("BOP Negative (Selling Pressure)")

        # Supersmoother (Ehlers Filter)
        if (
            self.config["indicators"].get("supersmoother")
            and "supersmoother" in indicator_values
            and not pd.isna(indicator_values["supersmoother"])
        ):
            ss_val = Decimal(str(indicator_values["supersmoother"]))
            weight = Decimal(
                str(self._get_indicator_weight("supersmoother", market_regime))
            )
            if current_price > ss_val:
                bullish_score += weight
                conditions_met.append("Price Above SuperSmoother")
            elif current_price < ss_val:
                bearish_score += weight
                conditions_met.append("Price Below SuperSmoother")

        # Ehlers Fisher Transform
        if (
            self.config["indicators"].get("ehlers_fisher")
            and "ehlers_fisher" in indicator_values
            and not pd.isna(indicator_values["ehlers_fisher"])
        ):
            fisher_val = Decimal(str(indicator_values["ehlers_fisher"]))
            # Need previous Fisher value for crossover. Assume if > 0 bullish for simplicity
            weight = Decimal(
                str(self._get_indicator_weight("ehlers_fisher", market_regime))
            )
            if fisher_val > 0.0:
                bullish_score += weight
                conditions_met.append("Ehlers Fisher Positive")
            elif fisher_val < 0.0:
                bearish_score += weight
                conditions_met.append("Ehlers Fisher Negative")

        # Laguerre RSI
        if (
            self.config["indicators"].get("laguerre_rsi")
            and "laguerre_rsi" in indicator_values
            and not pd.isna(indicator_values["laguerre_rsi"])
        ):
            lrsi_val = Decimal(str(indicator_values["laguerre_rsi"]))
            weight = Decimal(
                str(self._get_indicator_weight("laguerre_rsi", market_regime))
            )
            if lrsi_val < 0.2:
                bullish_score += weight
                conditions_met.append("Laguerre RSI Oversold")
            elif lrsi_val > 0.8:
                bearish_score += weight
                conditions_met.append("Laguerre RSI Overbought")

        # Supertrend
        if (
            self.config["indicators"].get("supertrend")
            and "supertrend" in indicator_values
            and indicator_values["supertrend"]
        ):
            st_dir = indicator_values["supertrend"].get("direction")
            weight = Decimal(
                str(self._get_indicator_weight("supertrend", market_regime))
            )
            if st_dir == 1:
                bullish_score += weight
                conditions_met.append("Supertrend Bullish")
            elif st_dir == -1:
                bearish_score += weight
                conditions_met.append("Supertrend Bearish")

        # CMO (Chande Momentum Oscillator)
        if (
            self.config["indicators"].get("cmo")
            and "cmo" in indicator_values
            and not pd.isna(indicator_values["cmo"])
        ):
            cmo_val = Decimal(str(indicator_values["cmo"]))
            weight = Decimal(str(self._get_indicator_weight("cmo", market_regime)))
            if cmo_val > 50:
                bullish_score += weight
                conditions_met.append("CMO Bullish Extreme")
            elif cmo_val < -50:
                bearish_score += weight
                conditions_met.append("CMO Bearish Extreme")

        # STC (Schaff Trend Cycle)
        if (
            self.config["indicators"].get("stc")
            and "stc" in indicator_values
            and not pd.isna(indicator_values["stc"])
        ):
            stc_val = Decimal(str(indicator_values["stc"]))
            weight = Decimal(str(self._get_indicator_weight("stc", market_regime)))
            if stc_val > 75:  # Often indicates overbought conditions or strong uptrend
                bullish_score += weight * Decimal(
                    "0.5"
                )  # Partial weight as it can be overbought continuation
                conditions_met.append("STC High (Strong Bullish Momentum)")
            elif (
                stc_val < 25
            ):  # Often indicates oversold conditions or strong downtrend
                bearish_score += weight * Decimal("0.5")
                conditions_met.append("STC Low (Strong Bearish Momentum)")

        # Order Book Walls
        if indicator_values.get("order_book_walls", {}).get("bullish"):
            bullish_score += Decimal(
                str(self.config["order_book_support_confidence_boost"])
            )
            conditions_met.append("Bullish Order Book Wall")
        if indicator_values.get("order_book_walls", {}).get("bearish"):
            bearish_score += Decimal(
                str(self.config["order_book_resistance_confidence_boost"])
            )
            conditions_met.append("Bearish Order Book Wall")

        # L2 Metrics (Imbalance, Depth Profile, Liquidity Clusters)
        l2_metrics = indicator_values.get("l2_metrics", {})
        if l2_metrics.get("imbalance_10", 0) > 0.3:
            bullish_score += Decimal("0.2")  # Fixed boost for strong L2 imbalance
            conditions_met.append("Strong L2 Imbalance (Top 10)")
        elif l2_metrics.get("imbalance_10", 0) < -0.3:
            bearish_score += Decimal("0.2")
            conditions_met.append("Strong L2 Sell Imbalance (Top 10)")

        depth_profile = indicator_values.get("depth_profile", {})
        if depth_profile.get("imbalance_0.5%", 0) > 0.4:
            bullish_score += Decimal("0.3")
            conditions_met.append("Heavy Buy Liquidity (0.5% Range)")
        elif depth_profile.get("imbalance_0.5%", 0) < -0.4:
            bearish_score += Decimal("0.3")
            conditions_met.append("Heavy Sell Liquidity (0.5% Range)")

        liquidity_clusters = indicator_values.get("liquidity_clusters", {})
        # Check if price is near a major bid/ask cluster
        for p, _q in liquidity_clusters.get("bids", []):
            if current_price > p and (current_price - p) / current_price < Decimal(
                "0.002"
            ):
                bullish_score += Decimal("0.4")
                conditions_met.append(f"Price Near Heavy Support Cluster (${p:.2f})")
                break
        for p, _q in liquidity_clusters.get("asks", []):
            if current_price < p and (p - current_price) / current_price < Decimal(
                "0.002"
            ):
                bearish_score += Decimal("0.4")
                conditions_met.append(f"Price Near Heavy Resistance Cluster (${p:.2f})")
                break

        # --- Determine Final Signal ---
        final_signal_type = SignalType.HOLD
        final_confidence = Decimal("0.0")
        final_stop_loss = None
        final_take_profit = None

        if (
            bullish_score > bearish_score
            and bullish_score >= self.signal_score_threshold
        ):
            final_signal_type = SignalType.BUY
            final_confidence = bullish_score
            if atr_value > 0:
                final_stop_loss = (current_price - (
                    atr_value * Decimal(str(self.config["stop_loss_multiple"]))
                )).quantize(Decimal('0.00001'))
                final_take_profit = (current_price + (
                    atr_value * Decimal(str(self.config["take_profit_multiple"]))
                )).quantize(Decimal('0.00001'))
        elif (
            bearish_score > bullish_score
            and bearish_score >= self.signal_score_threshold
        ):
            final_signal_type = SignalType.SELL
            final_confidence = bearish_score
            if atr_value > 0:
                final_stop_loss = (current_price + (
                    atr_value * Decimal(str(self.config["stop_loss_multiple"]))
                )).quantize(Decimal('0.00001'))
                final_take_profit = (current_price - (
                    atr_value * Decimal(str(self.config["take_profit_multiple"]))
                )).quantize(Decimal('0.00001'))

        # Calculate risk/reward ratio
        risk_reward_ratio = None
        if (
            final_stop_loss
            and final_take_profit
            and final_signal_type != SignalType.HOLD
        ):
            if final_signal_type == SignalType.BUY:
                risk = float(current_price - final_stop_loss)
                reward = float(final_take_profit - current_price)
            else:  # SELL
                risk = float(final_stop_loss - current_price)
                reward = float(current_price - final_take_profit)

            risk_reward_ratio = reward / risk if risk > 0 else None

        return TradingSignal(
            signal_type=final_signal_type,
            confidence=float(final_confidence),
            conditions_met=conditions_met,
            stop_loss=final_stop_loss,
            take_profit=final_take_profit,
            timestamp=time.time(),
            symbol=indicator_values.get("symbol", ""),
            timeframe=indicator_values.get("timeframe", ""),
            risk_reward_ratio=risk_reward_ratio,
        )


# --- Interpret Indicator Function ---
def interpret_indicator(
    logger: logging.Logger,
    indicator_name: str,
    values: list[float] | float | dict[str, Any],
) -> str | None:
    """
    Provides a human-readable interpretation of indicator values.
    """
    if (
        values is None
        or (isinstance(values, list) and not values)
        or (isinstance(values, pd.DataFrame) and values.empty)
    ):
        return f"{NEON_YELLOW}{indicator_name.upper()}:{RESET} No data available."

    try:
        # Convert single float values to list for consistent indexing if needed
        if isinstance(values, (float, int)):
            values = [values]
        elif isinstance(values, dict):  # For 'mom' which is a dict
            if indicator_name == "mom":
                trend = values.get("trend", "N/A")
                strength = values.get("strength", 0.0)
                return f"{NEON_PURPLE}Momentum Trend:{RESET} {trend} (Strength: {strength:.2f})"
            else:
                return f"{NEON_YELLOW}{indicator_name.upper()}:{RESET} Dictionary format not specifically interpreted."
        elif isinstance(
            values, pd.DataFrame
        ):  # For stoch_rsi_vals which is a DataFrame
            if indicator_name == "stoch_rsi_vals":
                # Stoch RSI interpretation is handled directly in analyze function
                return None
            else:
                return f"{NEON_YELLOW}{indicator_name.upper()}:{RESET} DataFrame format not specifically interpreted."

        # Interpret based on indicator name
        last_value = (
            values[-1]
            if isinstance(values, list) and values
            else values[0]
            if isinstance(values, list)
            else values
        )  # Handles single value lists too

        if indicator_name == "rsi":
            if last_value > 70:
                return f"{NEON_RED}RSI:{RESET} Overbought ({last_value:.2f})"
            elif last_value < 30:
                return f"{NEON_GREEN}RSI:{RESET} Oversold ({last_value:.2f})"
            else:
                return f"{NEON_YELLOW}RSI:{RESET} Neutral ({last_value:.2f})"

        elif indicator_name == "mfi":
            if last_value > 80:
                return f"{NEON_RED}MFI:{RESET} Overbought ({last_value:.2f})"
            elif last_value < 20:
                return f"{NEON_GREEN}MFI:{RESET} Oversold ({last_value:.2f})"
            else:
                return f"{NEON_YELLOW}MFI:{RESET} Neutral ({last_value:.2f})"

        elif indicator_name == "cci":
            if last_value > 100:
                return f"{NEON_RED}CCI:{RESET} Overbought ({last_value:.2f})"
            elif last_value < -100:
                return f"{NEON_GREEN}CCI:{RESET} Oversold ({last_value:.2f})"
            else:
                return f"{NEON_YELLOW}CCI:{RESET} Neutral ({last_value:.2f})"

        elif indicator_name == "wr":
            if last_value < -80:
                return f"{NEON_GREEN}Williams %R:{RESET} Oversold ({last_value:.2f})"
            elif last_value > -20:
                return f"{NEON_RED}Williams %R:{RESET} Overbought ({last_value:.2f})"
            else:
                return f"{NEON_YELLOW}Williams %R:{RESET} Neutral ({last_value:.2f})"

        elif indicator_name == "adx":
            if last_value > 25:
                return f"{NEON_GREEN}ADX:{RESET} Trending ({last_value:.2f})"
            else:
                return f"{NEON_YELLOW}ADX:{RESET} Ranging ({last_value:.2f})"

        elif indicator_name == "obv":
            if len(values) >= 2:
                return f"{NEON_BLUE}OBV:{RESET} {'Bullish' if values[-1] > values[-2] else 'Bearish' if values[-1] < values[-2] else 'Neutral'}"
            else:
                return f"{NEON_BLUE}OBV:{RESET} {last_value:.2f} (Insufficient history for trend)"

        elif indicator_name == "adi":
            if len(values) >= 2:
                return f"{NEON_BLUE}ADI:{RESET} {'Accumulation' if values[-1] > values[-2] else 'Distribution' if values[-1] < values[-2] else 'Neutral'}"
            else:
                return f"{NEON_BLUE}ADI:{RESET} {last_value:.2f} (Insufficient history for trend)"

        elif indicator_name == "sma_10":
            return f"{NEON_YELLOW}SMA (10):{RESET} {last_value:.2f}"

        elif indicator_name == "psar":
            return f"{NEON_BLUE}PSAR:{RESET} {last_value:.4f} (Last Value)"

        elif indicator_name == "fve":
            return f"{NEON_BLUE}FVE:{RESET} {last_value:.2f} (Last Value)"

        elif indicator_name == "macd":
            # values for MACD are [macd_line, signal_line, histogram]
            if len(values[-1]) == 3:
                macd_line, signal_line, histogram = (
                    values[-1][0],
                    values[-1][1],
                    values[-1][2],
                )
                return f"{NEON_GREEN}MACD:{RESET} MACD={macd_line:.2f}, Signal={signal_line:.2f}, Histogram={histogram:.2f}"
            else:
                return f"{NEON_RED}MACD:{RESET} Calculation issue."

        elif indicator_name == "bollinger_bands":
            if isinstance(values, dict):
                return f"{NEON_BLUE}Bollinger Bands:{RESET} %B={values.get('percent_b', 0):.2f}, BW={values.get('bandwidth', 0):.4f}"
            return None

        elif indicator_name == "ichimoku_cloud":
            if isinstance(values, dict):
                return f"{NEON_BLUE}Ichimoku:{RESET} Tenkan={values.get('tenkan', 0):.2f}, Kijun={values.get('kijun', 0):.2f}"
            return None

        elif indicator_name == "cmf":
            return f"{NEON_BLUE}CMF:{RESET} {last_value:.4f}"

        elif indicator_name == "roc":
            return f"{NEON_BLUE}ROC:{RESET} {last_value:.2f}%"

        elif indicator_name == "vortex":
            if isinstance(values, dict):
                return f"{NEON_BLUE}Vortex:{RESET} VI+={values.get('vi_plus', 0):.2f}, VI-={values.get('vi_minus', 0):.2f}"
            return None

        elif indicator_name == "hma":
            return f"{NEON_BLUE}HMA:{RESET} {last_value:.4f}"

        elif indicator_name == "awesome_oscillator":
            return f"{NEON_BLUE}Awesome Osc:{RESET} {last_value:.4f}"

        elif indicator_name == "bop":
            return f"{NEON_BLUE}BOP:{RESET} {last_value:.4f}"

        elif indicator_name == "klinger_oscillator":
            if isinstance(values, dict):
                return f"{NEON_BLUE}Klinger:{RESET} KO={values.get('ko', 0):,.0f}"
            return None

        elif indicator_name == "nvi":
            return f"{NEON_BLUE}NVI:{RESET} {last_value:.2f}"

        elif indicator_name == "pvi":
            return f"{NEON_BLUE}PVI:{RESET} {last_value:.2f}"

        elif indicator_name == "supersmoother":
            return f"{NEON_PURPLE}SuperSmoother:{RESET} {last_value:.4f}"

        elif indicator_name == "ehlers_fisher":
            return f"{NEON_GREEN}Ehlers Fisher:{RESET} {last_value:.4f} ({'Bullish' if last_value > 0 else 'Bearish' if last_value < 0 else 'Neutral'})"

        elif indicator_name == "laguerre_rsi":
            return f"{NEON_CYAN}Laguerre RSI:{RESET} {last_value:.4f} ({'Overbought' if last_value > 0.8 else 'Oversold' if last_value < 0.2 else 'Neutral'})"

        elif indicator_name == "supertrend":
            if isinstance(values, dict):
                trend_dir = "UP" if values.get("direction", 0) == 1 else "DOWN"
                color = NEON_GREEN if trend_dir == "UP" else NEON_RED
                return f"{color}Supertrend:{RESET} {trend_dir} (${values.get('supertrend', 0):.2f})"
            return None

        elif indicator_name == "cmo":
            return f"{NEON_BLUE}CMO:{RESET} {last_value:.2f} ({'Bullish' if last_value > 50 else 'Bearish' if last_value < -50 else 'Neutral'})"

        elif indicator_name == "stc":
            return f"{NEON_PURPLE}STC:{RESET} {last_value:.2f} ({'Bullish' if last_value > 75 else 'Bearish' if last_value < 25 else 'Neutral'})"

        else:
            return f"{NEON_YELLOW}{indicator_name.upper()}:{RESET} No specific interpretation available."

    except (TypeError, IndexError, KeyError, ValueError) as e:
        logger.error(
            f"{NEON_RED}Error interpreting {indicator_name}: {e}. Values: {values}{RESET}"
        )
        return f"{NEON_RED}{indicator_name.upper()}:{RESET} Interpretation error."


# --- Main Function ---
class StrategyOptimizer:
    """Optimizes strategy weights using historical data."""

    def __init__(self, api_client: APIClient, config: dict, logger: logging.Logger):
        self.api_client = api_client
        self.config = config
        self.logger = logger

    def optimize(self, symbol: str, interval: str) -> dict | None:
        """Find best weights for the current market."""
        self.logger.info(
            f"{NEON_PURPLE}Starting Optimization for {symbol} ({interval})...{RESET}"
        )

        to_optimize = ['stoch_rsi', 'rsi', 'macd', 'ema_alignment']
        best_pnl = Decimal("-1000000")
        best_weights = None

        import itertools
        options = [0.2, 0.8]
        combinations = list(itertools.product(options, repeat=len(to_optimize)))

        self.logger.info(f"Testing {len(combinations)} weight combinations...")

        engine = BacktestingEngine(self.api_client, self.config, self.logger)

        for combo in combinations:
            test_weights = dict(zip(to_optimize, combo))
            temp_config = self.config.copy()
            temp_config['weight_sets']['low_volatility'].update(test_weights)
            temp_config['weight_sets']['high_volatility'].update(test_weights)

            engine.config = temp_config
            final_balance = engine.run_backtest(symbol, interval, "", "", quiet=True)

            if final_balance > best_pnl:
                best_pnl = final_balance
                best_weights = test_weights

        self.logger.info(f"{NEON_GREEN}Optimization complete! Best PnL: ${float(best_pnl):.2f}{RESET}")
        return best_weights


class BacktestingEngine:
    """Runs simulations on historical kline data."""

    def __init__(self, api_client: APIClient, config: dict, logger: logging.Logger):
        self.api_client = api_client
        self.config = config
        self.logger = logger

    def run_backtest(self, symbol: str, interval: str, start_date: str, end_date: str, quiet: bool = False):
        """Execute historical simulation."""
        if not quiet:
            self.logger.info(
                f"{NEON_BLUE}Running Backtest for {symbol} ({interval}) ...{RESET}"
            )

        # Fetch data (Bybit limit 1000)
        df = self.api_client.fetch_klines(symbol, interval, limit=1000)
        if df.empty:
            self.logger.error(f"{NEON_RED}No data for backtesting{RESET}")
            return

        self.logger.info(f"{NEON_GREEN}Fetched {len(df)} candles for simulation.{RESET}")

        # Pre-calculate indicators vectorized
        indicator_calc = IndicatorCalculator(df, self.config, self.logger)
        all_indicators = indicator_calc.calculate_all_indicators_vectorized()

        # Simulation settings
        balance = Decimal(str(self.config['risk_management']['portfolio_value']))
        initial_balance = balance
        position = None
        trades = []
        fee_rate = Decimal("0.00055")  # Taker fee

        # Iterate through bars
        for i in range(50, len(df)):
            bar = df.iloc[i]
            price = Decimal(str(bar['close']))

            # 1. Exit Logic
            if position:
                exit_price = None
                reason = ""

                if position['side'] == SignalType.BUY:
                    if Decimal(str(bar['low'])) <= position['sl']:
                        exit_price = position['sl']
                        reason = "Stop Loss"
                    elif Decimal(str(bar['high'])) >= position['tp']:
                        exit_price = position['tp']
                        reason = "Take Profit"
                else:  # SELL
                    if Decimal(str(bar['high'])) >= position['sl']:
                        exit_price = position['sl']
                        reason = "Stop Loss"
                    elif Decimal(str(bar['low'])) <= position['tp']:
                        exit_price = position['tp']
                        reason = "Take Profit"

                if exit_price:
                    pnl = (exit_price - position['entry']) * position['qty'] if position['side'] == SignalType.BUY else (position['entry'] - exit_price) * position['qty']
                    fees = (position['entry'] * position['qty'] * fee_rate) + (exit_price * position['qty'] * fee_rate)
                    net_pnl = pnl - fees
                    balance += net_pnl
                    trades.append({'net_pnl': net_pnl, 'reason': reason})
                    position = None

            # 2. Entry Logic
            if not position:
                current_indicators = {}
                prev_indicators = {}
                for k, v in all_indicators.items():
                    if isinstance(v, pd.Series):
                        current_indicators[k] = v.iloc[i]
                        prev_indicators[k] = v.iloc[i-1] if i > 0 else v.iloc[i]
                    elif isinstance(v, pd.DataFrame):
                        current_indicators[k] = v.iloc[i].to_dict()
                        prev_indicators[k] = v.iloc[i-1].to_dict() if i > 0 else v.iloc[i].to_dict()

                current_indicators['symbol'] = symbol
                current_indicators['timeframe'] = interval

                # Simple regime detection for backtest (simplified)
                regime = MarketRegime.SIDEWAYS
                if 'atr' in current_indicators and 'close' in bar:
                    if current_indicators['atr'] / bar['close'] > 0.02:
                        regime = MarketRegime.VOLATILE

                sig_gen = SignalGenerator(self.config, self.logger)
                signal = sig_gen.generate_signal(
                    current_indicators,
                    prev_indicators,
                    regime,
                    price,
                    Decimal(str(current_indicators.get('atr', 0)))
                )

                if signal.signal_type != SignalType.HOLD and signal.stop_loss:
                    risk_mgr = RiskManager(self.config, self.logger)
                    qty = risk_mgr.calculate_position_size(price, signal.stop_loss, balance)
                    if qty > 0:
                        position = {
                            'side': signal.signal_type,
                            'entry': price,
                            'qty': qty,
                            'sl': signal.stop_loss,
                            'tp': signal.take_profit
                        }

        if not quiet:
            self.report(initial_balance, balance, trades)
        return balance

    def report(self, initial, final, trades):
        if not trades:
            self.logger.info(f"{NEON_YELLOW}Backtest complete: No trades executed.{RESET}")
            return

        wins = [t for t in trades if t['net_pnl'] > 0]
        net = final - initial
        win_rate = len(wins) / len(trades)
        self.logger.info(f"\n{NEON_CYAN}=== BACKTEST SUMMARY: {len(trades)} Trades ==={RESET}")
        self.logger.info(f"Win Rate: {win_rate:.1%}")
        self.logger.info(f"Net Profit: ${float(net):.2f} ({float(net/initial):.2%})")
        self.logger.info(f"Final Balance: ${float(final):.2f}")


# --- UI and Output Helpers ---

def show_loading_spinner():
    import itertools
    spinner_state = {'done': False}
    def animate():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if spinner_state['done']:
                break
            sys.stdout.write(f'\rLoading {c}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rDone!     \n')
    t = threading.Thread(target=animate)
    t.start()
    return lambda: spinner_state.update({'done': True})

def format_pnl_output(pnl: Decimal) -> str:
    if pnl > 0:
        color = NEON_GREEN
    elif pnl < 0:
        color = NEON_RED
    else:
        color = NEON_WHITE
    return f'{color}${pnl:.2f}{RESET}'

def confidence_bar(confidence: float, length: int = 20) -> str:
    filled_length = int(length * confidence)
    bar = NEON_GREEN + '█' * filled_length + NEON_WHITE + '-' * (length - filled_length) + RESET
    return bar

def display_signal_with_confidence(signal: TradingSignal):
    bar = confidence_bar(signal.confidence)
    return f'Signal: {signal.signal_type.value.upper()} {bar} Confidence: {signal.confidence:.2f}'

def format_timestamp(timestamp: float, tz: ZoneInfo = TIMEZONE) -> str:
    dt = datetime.fromtimestamp(timestamp, tz)
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')

def signal_to_json(signal: TradingSignal) -> str:
    output = {
        'type': signal.signal_type.value if signal.signal_type else None,
        'confidence': signal.confidence,
        'conditions_met': signal.conditions_met,
        'stop_loss': str(signal.stop_loss) if signal.stop_loss else None,
        'take_profit': str(signal.take_profit) if signal.take_profit else None,
        'timestamp': format_timestamp(signal.timestamp),
        'symbol': signal.symbol,
        'timeframe': signal.timeframe,
        'position_size': str(signal.position_size) if signal.position_size else None,
        'risk_reward_ratio': signal.risk_reward_ratio
    }
    return json.dumps(output, indent=2)

def display_backtest_progress(current: int, total: int) -> None:
    percent = (current / total) * 100
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = NEON_GREEN + '█' * filled_length + NEON_WHITE + '-' * (bar_length - filled_length) + RESET
    sys.stdout.write(f'\rBacktesting: |{bar}| {percent:.2f}% Complete')
    sys.stdout.flush()
    if current == total:
        print()

def summarized_indicator_output(indicators: dict[str, Any]) -> str:
    lines = []
    for name, val in indicators.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        color = NEON_WHITE
        try:
            val_float = float(val) if not isinstance(val, dict) else None
            if val_float is not None:
                if val_float > 0:
                    color = NEON_GREEN
                elif val_float < 0:
                    color = NEON_RED
                else:
                    color = NEON_YELLOW
            lines.append(f'{name.upper()}: {color}{val}{RESET}')
        except Exception:
            lines.append(f'{name.upper()}: {NEON_WHITE}{val}{RESET}')
    return ' | '.join(lines)

def display_support_resistance(supports: list[tuple[str, Decimal]], resistances: list[tuple[str, Decimal]]) -> str:
    sup_str = ', '.join([f'{label}@${value:.4f}' for label, value in supports]) or 'None'
    res_str = ', '.join([f'{label}@${value:.4f}' for label, value in resistances]) or 'None'
    return f'Supports: {NEON_GREEN}{sup_str}{RESET} | Resistances: {NEON_RED}{res_str}{RESET}'

def conditions_to_text(conditions: list[str]) -> str:
    if not conditions:
        return 'None'
    bullet = '\u2022'
    return '\n'.join([f'{bullet} {cond}' for cond in conditions])

def indicators_to_compact_json(indicators: dict[str, Any]) -> dict:
    output = {}
    for key, val in indicators.items():
        if isinstance(val, (float, int, str)):
            output[key] = val
        elif isinstance(val, dict):
            output[key] = {k: v for k, v in val.items() if isinstance(v, (float, int, str))}
        elif isinstance(val, pd.Series) and not val.empty:
            output[key] = float(val.iloc[-1])
    return output

def display_trailing_stop_update(symbol: str, old_sl: Decimal, new_sl: Decimal) -> str:
    if new_sl > old_sl:
        color = NEON_GREEN
        direction = 'Increased'
    elif new_sl < old_sl:
        color = NEON_RED
        direction = 'Decreased'
    else:
        color = NEON_WHITE
        direction = 'Unchanged'
    return f'{color}Trailing Stop {direction} for {symbol}: {old_sl:.4f} -> {new_sl:.4f}{RESET}'

def format_position_size(size: Decimal) -> str:
    return f'{size.quantize(Decimal("0.0001"))}'

def display_open_positions(signals: dict[int, SignalHistory], current_price: Decimal) -> str:
    lines = []
    for sid, signal in signals.items():
        if signal.signal_type == SignalType.BUY:
            unrealized_pnl = (current_price - signal.entry_price) * signal.quantity
        else:
            unrealized_pnl = (signal.entry_price - current_price) * signal.quantity
        r_str = f'R:R={float(signal.risk_reward_ratio):.2f}' if signal.risk_reward_ratio else 'R:R=N/A'
        lines.append(f'ID:{sid} {signal.signal_type.value.upper()} {signal.symbol} Qty:{signal.quantity:.4f} Entry:${signal.entry_price:.4f} PnL:${unrealized_pnl:.2f} {r_str}')
    return '\n'.join(lines)

def active_positions_to_json(signals: dict[int, SignalHistory]) -> str:
    results = []
    for sid, s in signals.items():
        results.append({
            'id': sid,
            'symbol': s.symbol,
            'signal_type': s.signal_type.value,
            'entry_price': str(s.entry_price),
            'quantity': str(s.quantity),
            'stop_loss': str(s.stop_loss) if s.stop_loss else None,
            'take_profit': str(s.take_profit) if s.take_profit else None,
            'trailing_sl': str(s.trailing_sl) if s.trailing_sl else None,
            'highest_price': str(s.highest_price) if s.highest_price else None,
            'lowest_price': str(s.lowest_price) if s.lowest_price else None,
            'profit_loss': str(s.profit_loss) if s.profit_loss else None,
            'net_pnl': str(s.net_pnl) if s.net_pnl else None,
            'exit_reason': s.exit_reason,
            'market_regime': s.market_regime.value if s.market_regime else None
        })
    return json.dumps(results, indent=2)

def nearest_levels_ui(current_price: Decimal, supports: list[tuple[str, Decimal]], resistances: list[tuple[str, Decimal]]) -> str:
    def format_level(label, val):
        diff = abs((val - current_price) / current_price) * 100
        color = NEON_GREEN if val < current_price else NEON_RED
        return f'{color}{label}@${val:.4f} ({diff:.2f}%) {RESET}'
    sup_lines = [format_level(label, v) for label, v in sorted(supports, key=lambda x: abs((current_price - x[1])/current_price))]
    res_lines = [format_level(label, v) for label, v in sorted(resistances, key=lambda x: abs((x[1] - current_price)/current_price))]
    return 'Supports: ' + ', '.join(sup_lines) + '\nResistances: ' + ', '.join(res_lines)

def terminal_indicator_dashboard(indicators: dict[str, float]) -> str:
    parts = []
    for name in ['rsi', 'mfi', 'cci', 'fve', 'stc', 'cmo']:
        val = indicators.get(name, None)
        if val is not None and not pd.isna(val):
            parts.append(f'{name.upper()}: {val:.2f}')
    return ' | '.join(parts)

def order_book_imbalance_ui(imbalance: float) -> str:
    if imbalance > 0.3:
        color = NEON_GREEN
        state = "Strong Buy"
    elif imbalance < -0.3:
        color = NEON_RED
        state = "Strong Sell"
    else:
        color = NEON_YELLOW
        state = "Neutral"
    return f'Order Book Imbalance: {color}{imbalance:.2f} ({state}){RESET}'

def notification_summary(signal: TradingSignal, indicators: dict[str, Any]) -> str:
    parts = [f'Signal: {signal.signal_type.value.upper()} {signal.symbol}', f'Confidence: {signal.confidence:.2f}']
    key_indicators = ['rsi', 'mfi', 'atr', 'momentum_ma_short', 'momentum_ma_long']
    for k in key_indicators:
        v = indicators.get(k, None)
        if v is not None and not pd.isna(v):
            parts.append(f'{k.upper()}: {v:.2f}')
    return ' | '.join(parts)

def performance_metrics_to_json(metrics_list: list[PerformanceMetrics]) -> str:
    results = []
    for m in metrics_list:
        results.append({
            'total_trades': m.total_trades,
            'winning_trades': m.winning_trades,
            'losing_trades': m.losing_trades,
            'win_rate': m.win_rate,
            'profit_factor': m.profit_factor,
            'max_drawdown': m.max_drawdown,
            'sharpe_ratio': m.sharpe_ratio,
            'total_profit': str(m.total_profit),
            'total_loss': str(m.total_loss),
            'net_profit': str(m.net_profit),
            'average_win': str(m.average_win),
            'average_loss': str(m.average_loss)
        })
    return json.dumps(results, indent=2)

def print_separator():
    print(f'{NEON_CYAN}{"-" * 60}{RESET}')


# --- Utility Snippets ---

def connect_websocket(url, on_message, headers=None):
    def on_open(ws):
        print('WebSocket connection opened')

    def on_close(ws, close_status_code, close_msg):
        print('WebSocket connection closed')

    ws = websocket.WebSocketApp(
        url,
        header=headers or [],
        on_open=on_open,
        on_message=on_message,
        on_close=on_close
    )

    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    return ws

def safe_rest_get(api_client, endpoint, params=None):
    import time
    attempts = 0
    while attempts < MAX_API_RETRIES:
        response = api_client.make_request('GET', endpoint, params)
        if response is not None:
            return response
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts)
    return None

def send_websocket_ping(ws):
    try:
        ws.send('ping')
    except Exception as e:
        logger.error(f'Failed to send ping: {e}')

def rest_post_with_retry(api_client, endpoint, payload, retries=3, delay=5):
    for attempt in range(1, retries + 1):
        response = api_client.make_request('POST', endpoint, payload)
        if response and response.get('retCode') == 0:
            return response
        logger.warning(f'POST request failed attempt {attempt}'.format(attempt))
        time.sleep(delay * attempt)
    return None

def reconnect_websocket(ws, url, on_message):
    ws.close()
    time.sleep(1)
    backoff = 1
    while True:
        try:
            ws = connect_websocket(url, on_message)
            return ws
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

def parse_websocket_message(message):
    try:
        data = json.loads(message)
        return data
    except json.JSONDecodeError:
        logger.error('WebSocket message JSON decode error')
        return None

def get_current_price_rest(api_client, symbol):
    res = safe_rest_get(api_client, '/v5/market/tickers', {'category': 'linear', 'symbol': symbol})
    if res and res.get('retCode') == 0:
        for ticker in res['result'].get('list', []):
            if ticker['symbol'] == symbol:
                return Decimal(ticker['lastPrice'])
    return None

def subscribe_to_orderbook_ws(ws, symbol):
    sub_request = {
        "op": "subscribe",
        "args": [
            {
                "channel": "orderbook",
                "symbol": symbol
            }
        ]
    }
    ws.send(json.dumps(sub_request))

def fetch_orderbook_rest(api_client, symbol, limit=50):
    response = safe_rest_get(api_client, '/v5/market/orderbook', {'symbol': symbol, 'limit': str(limit), 'category': 'linear'})
    if response and response.get('retCode') == 0:
        return response.get('result')
    return None

def websocket_message_handler(ws, message):
    data = parse_websocket_message(message)
    if not data:
        return
    if 'topic' in data:
        if data['topic'].startswith('orderbook'):
            orderbook_data = data.get('data')
            # Process orderbook_data here
            logger.info(f'Received orderbook update: {orderbook_data}')

def rest_get_with_headers(api_client, endpoint, params=None, headers=None):
    api_client.session.headers.update(headers or {})
    return safe_rest_get(api_client, endpoint, params)

def format_order_quantity(api_client, symbol, qty):
    # Match the APIClient's formatting
    return api_client.format_quantity(symbol, qty)

def format_order_price(api_client, symbol, price):
    return api_client.format_price(symbol, price)

def send_order_ws(ws, symbol, side, order_type, qty, price=None, stop_loss=None, take_profit=None):
    order = {
        "category": "linear",
        "symbol": symbol,
        "side": side.capitalize(),
        "orderType": order_type,
        "qty": str(qty),
        "timeInForce": "GTC"
    }
    if price:
        order["price"] = str(price)
    if stop_loss:
        order["stopLoss"] = str(stop_loss)
    if take_profit:
        order["takeProfit"] = str(take_profit)
    ws.send(json.dumps({"op": "order", "args": [order]}))

def websocket_heartbeat(ws, interval=30):
    import threading
    def run():
        while True:
            send_websocket_ping(ws)
            time.sleep(interval)
    threading.Thread(target=run, daemon=True).start()

def fetch_multiple_klines(api_client, symbol, intervals):
    data = {}
    for interval in intervals:
        df = api_client.fetch_klines(symbol, interval, limit=200)
        data[interval] = df
    return data

def fetch_fee_rates_rest(api_client, symbol):
    return api_client.fetch_fee_rates(symbol)

def websocket_close(ws):
    ws.close()

def send_rest_delete(api_client, endpoint, params=None):
    # Generalized DELETE call
    return api_client.make_request('DELETE', endpoint, params)

def websocket_is_alive(ws):
    try:
        return ws.keep_running and ws.sock and ws.sock.connected
    except Exception:
        return False

def main():
    """
    Main function to run the trading analysis bot.
    Handles CLI arguments, user input, and the main analysis loop.
    """
    parser = argparse.ArgumentParser(description="🧙‍♂️ Pyrmethus Whaler Zenith Apex V12")
    parser.add_argument("--symbol", type=str, help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument(
        "--timeframe", type=str, help=f"Timeframe ({', '.join(VALID_INTERVALS)})"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable Paper Trading mode (simulated orders)",
    )
    parser.add_argument(
        "--live", action="store_true", help="Enable Live Trading mode (actual orders)"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run in backtesting mode"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run parameter optimizer for best net profit",
    )
    parser.add_argument("--proxy", type=str, help="Override proxy URL")
    args = parser.parse_args()

    if not API_KEY or not API_SECRET:
        logger.error(
            f"{NEON_RED}BYBIT_API_KEY and BYBIT_API_SECRET must be set in your .env file.{RESET}"
        )
        return

    # Determine Mode Interactively if no flags provided
    if not any([args.paper, args.live, args.backtest, args.optimize]):
        print(f"\n{NEON_CYAN}🧙‍♂️ SELECT OPERATIONAL MODE:{RESET}")
        print(f"  1. {NEON_BLUE}BACKTEST{RESET} - Run simulation on historical data")
        print(f"  2. {NEON_PURPLE}OPTIMIZE{RESET} - Find best weights for net profit")
        print(
            f"  3. {NEON_YELLOW}PAPER TRADE{RESET} - Live data, simulated orders (SAFE)"
        )
        print(f"  4. {NEON_RED}LIVE TRADE{RESET} - Real money execution (WARNING)")

        choice = input(f"\n{NEON_WHITE}Enter choice [1-4, default 3]: {RESET}").strip()
        if choice == "1":
            args.backtest = True
            is_paper = True
        elif choice == "2":
            args.optimize = True
            is_paper = True
        elif choice == "4":
            args.live = True
            is_paper = False
        else:
            args.paper = True
            is_paper = True
    else:
        # Determine Paper vs Live mode from flags
        is_paper = True  # Default
        if args.live:
            is_paper = False
        elif args.paper:
            is_paper = True
        else:
            is_paper = CONFIG.get("paper_mode", True)

    # Setup database
    setup_database()

    # Initialize components
    db_manager = DatabaseManager(CONFIG.get("database", {}).get("path", DATABASE_FILE))
    notification_system = NotificationSystem(CONFIG)
    risk_manager = RiskManager(CONFIG, logger)
    signal_tracker = SignalHistoryTracker(db_manager, CONFIG, logger, risk_manager)

    # Resolve Symbol
    symbol = args.symbol.upper() if args.symbol else None
    if not symbol:
        symbol_input = (
            input(f"{NEON_BLUE}Enter trading symbol (e.g., BTCUSDT): {RESET}")
            .upper()
            .strip()
        )
        symbol = symbol_input if symbol_input else "BTCUSDT"

    # Validate Symbol
    if not any(x in symbol for x in ["USDT", "USDC"]):
        logger.warning(
            f"{NEON_YELLOW}Invalid symbol '{symbol}' detected. Defaulting to BTCUSDT.{RESET}"
        )
        symbol = "BTCUSDT"

    # Resolve Timeframe
    interval = (
        args.timeframe if args.timeframe and args.timeframe in VALID_INTERVALS else None
    )
    if not interval:
        interval_input = input(
            f"{NEON_BLUE}Enter timeframe (e.g., {', '.join(VALID_INTERVALS)} or press Enter for default {CONFIG['interval']}): {RESET}"
        ).strip()
        interval = (
            interval_input
            if interval_input and interval_input in VALID_INTERVALS
            else CONFIG["interval"]
        )

    # Proxy Override
    proxy_config = CONFIG.get("proxy", {"enabled": False, "url": ""})
    if args.proxy:
        proxy_config = {"enabled": True, "url": args.proxy}

    # Run Optimization if requested
    if args.optimize:
        api_client = APIClient(
            API_KEY,
            API_SECRET,
            BASE_URL,
            logger,
            proxy_config=proxy_config,
            paper_mode=is_paper,
        )
        optimizer = StrategyOptimizer(api_client, CONFIG, logger)
        best_weights = optimizer.optimize(symbol, interval)
        if best_weights:
            CONFIG["weight_sets"]["low_volatility"].update(best_weights)
            CONFIG["weight_sets"]["high_volatility"].update(best_weights)
            logger.info(
                f"{NEON_GREEN}Updated active weights with optimized values.{RESET}"
            )

        # After optimization, prompt to continue to trading if in interactive mode
        if (
            not any([args.paper, args.live])
            and input(
                f"\n{NEON_BLUE}Start Paper Trading with these weights? [Y/n]: {RESET}"
            ).lower()
            == "n"
        ):
            return

    # Check if backtesting mode requested
    if args.backtest:
        start_date = CONFIG.get("backtesting", {}).get("start_date", "")
        end_date = CONFIG.get("backtesting", {}).get("end_date", "")

        if not start_date or not end_date:
            start_date = input(
                f"{NEON_BLUE}Enter start date (YYYY-MM-DD): {RESET}"
            ).strip()
            if not start_date:
                start_date = "2026-01-01"
            end_date = input(f"{NEON_BLUE}Enter end date (YYYY-MM-DD): {RESET}").strip()
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")

        # Run backtest
        api_client = APIClient(
            API_KEY,
            API_SECRET,
            BASE_URL,
            logger,
            proxy_config=proxy_config,
            paper_mode=is_paper,
        )
        backtesting_engine = BacktestingEngine(api_client, CONFIG, logger)
        backtesting_engine.run_backtest(symbol, interval, start_date, end_date)
        return

    # Setup a dedicated logger for this symbol's activities
    symbol_logger = setup_custom_logger(symbol)

    mode_str = (
        f"{NEON_YELLOW}PAPER TRADING{RESET}"
        if is_paper
        else f"{NEON_RED}LIVE TRADING{RESET}"
    )
    symbol_logger.info(f"{NEON_CYAN}--- STARTING APEX V12 [{mode_str}] ---{RESET}")
    symbol_logger.info(f"{NEON_BLUE}Symbol: {symbol} | Interval: {interval}{RESET}")

    # Initialize API client
    api_client = APIClient(
        API_KEY,
        API_SECRET,
        BASE_URL,
        symbol_logger,
        proxy_config=proxy_config,
        paper_mode=is_paper,
    )

    # Check Tor if proxy enabled
    if CONFIG.get("proxy", {}).get("enabled") and "127.0.0.1:9050" in CONFIG[
        "proxy"
    ].get("url", ""):
        try:
            import socket

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            if s.connect_ex(("127.0.0.1", 9050)) == 0:
                symbol_logger.info(
                    f"{NEON_GREEN}Tor status check: OK (Port 9050 reachable){RESET}"
                )
            else:
                symbol_logger.warning(
                    f"{NEON_RED}Tor status check: FAILED (Port 9050 unreachable). Please run 'tor' in another terminal.{RESET}"
                )
            s.close()
        except Exception:
            pass

    # Initialize multi-timeframe analyzer if enabled
    multi_tf_analyzer = None
    if CONFIG.get("multi_timeframe", {}).get("enabled", False):
        multi_tf_analyzer = MultiTimeframeAnalyzer(api_client, CONFIG, symbol_logger)

    # Synchronize with exchange
    signal_tracker.sync_with_exchange(api_client, symbol)

    # Initialize account balance from exchange
    account_balance = api_client.fetch_balance("USDT")
    if account_balance <= 0:
        account_balance = Decimal(
            str(CONFIG.get("risk_management", {}).get("portfolio_value", 10000))
        )
        symbol_logger.warning(
            f"{NEON_YELLOW}Could not fetch live balance, using config value: ${account_balance}{RESET}"
        )
    else:
        symbol_logger.info(f"{NEON_GREEN}Live balance fetched: ${account_balance}{RESET}")

    daily_loss = Decimal("0")
    peak_balance = account_balance

    last_signal_time = 0.0  # Tracks the last time a signal was triggered for cooldown
    last_order_book_fetch_time = 0.0  # Tracks last order book fetch time for debouncing
    last_db_backup_time = time.time()

    # Main loop
    while True:
        try:
            # Check circuit breaker
            if risk_manager.check_circuit_breaker():
                symbol_logger.warning(
                    f"{NEON_RED}Circuit breaker is active. Pausing trading.{RESET}"
                )
                time.sleep(CONFIG["analysis_interval"])
                continue

            # Fetch current price
            current_price = api_client.fetch_current_price(symbol)
            if current_price is None:
                symbol_logger.error(
                    f"{NEON_RED}Failed to fetch current price for {symbol}. Skipping cycle.{RESET}"
                )
                time.sleep(CONFIG["retry_delay"])
                continue

            # Fetch kline data
            df = api_client.fetch_klines(symbol, interval, limit=200)
            if df.empty:
                symbol_logger.error(
                    f"{NEON_RED}Failed to fetch Kline data for {symbol}. Skipping cycle.{RESET}"
                )
                time.sleep(CONFIG["retry_delay"])
                continue

            # Validate data
            data_validator = DataValidator(CONFIG, symbol_logger)
            if not data_validator.validate_dataframe(df, symbol, interval):
                symbol_logger.error(
                    f"{NEON_RED}Data validation failed for {symbol} {interval}. Skipping cycle.{RESET}"
                )
                time.sleep(CONFIG["retry_delay"])
                continue

            # Debounce order book fetching to reduce API calls
            order_book_data = None
            if (
                time.time() - last_order_book_fetch_time
                >= CONFIG["order_book_debounce_s"]
            ):
                order_book_data = api_client.fetch_order_book(
                    symbol, limit=CONFIG["order_book_depth_to_check"]
                )
                last_order_book_fetch_time = time.time()
            else:
                symbol_logger.debug(
                    f"{NEON_YELLOW}Order book fetch debounced. Next fetch in {CONFIG['order_book_debounce_s'] - (time.time() - last_order_book_fetch_time):.1f}s{RESET}"
                )

            # Generate trading signal
            if multi_tf_analyzer:
                # Use multi-timeframe analysis
                trading_signal = multi_tf_analyzer.generate_consensus_signal(symbol)

                # Update Trailing Stops using the primary analyzer's data
                if multi_tf_analyzer.last_primary_analyzer:
                    signal_tracker.update_trailing_stops(
                        current_price,
                        multi_tf_analyzer.last_primary_analyzer.indicator_values.get(
                            "chandelier_exit", {}
                        ),
                        notification_system,
                    )
            else:
                # Use single timeframe analysis
                analyzer = TradingAnalyzer(df, CONFIG, symbol_logger, symbol, interval)
                timestamp = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")

                # Perform analysis and log the current state of indicators
                analyzer.analyze(current_price, timestamp, order_book_data)

                # Update Trailing Stops
                signal_tracker.update_trailing_stops(
                    current_price,
                    analyzer.indicator_values.get("chandelier_exit", {}),
                    notification_system,
                )

                # Generate trading signal based on the analysis
                trading_signal = analyzer.generate_trading_signal(current_price)

            # Display active position status and session PnL
            signal_tracker.display_active_status(current_price, daily_loss)

            # Check for exit conditions on active signals
            fee_rates = api_client.fetch_fee_rates(symbol)
            signals_to_exit = signal_tracker.check_exit_conditions(
                current_price, symbol, interval, fee_rates=fee_rates
            )
            for signal_id, exit_reason in signals_to_exit:
                signal = signal_tracker.active_signals.get(signal_id)
                if signal:
                    # Place closing order
                    close_side = (
                        "Sell" if signal.signal_type == SignalType.BUY else "Buy"
                    )
                    api_client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="Market",
                        qty=signal.quantity,
                        is_close=True,
                    )
                signal_tracker.update_signal(
                    signal_id, current_price, exit_reason, fee_rates=fee_rates
                )

                # Update risk manager with trade result
                signal = signal_tracker.active_signals.get(signal_id)
                if signal and signal.profit_loss:
                    risk_manager.update_trade_result(signal.profit_loss)
                    daily_loss += signal.profit_loss

                    # Update account balance
                    account_balance += signal.profit_loss

                    # Check daily loss limit
                    if risk_manager.check_daily_loss_limit(daily_loss, account_balance):
                        symbol_logger.error(
                            f"{NEON_RED}Daily loss limit reached. Stopping trading for today.{RESET}"
                        )
                        return

                    # Check drawdown limit
                    current_drawdown = float(
                        (peak_balance - account_balance) / peak_balance
                    )
                    if risk_manager.check_drawdown_limit(current_drawdown):
                        symbol_logger.error(
                            f"{NEON_RED}Maximum drawdown reached. Stopping trading.{RESET}"
                        )
                        return

                    # Update peak balance
                    if account_balance > peak_balance:
                        peak_balance = account_balance

            # Process new signal
            current_time_seconds = time.time()
            if trading_signal.signal_type != SignalType.HOLD and (
                current_time_seconds - last_signal_time >= CONFIG["signal_cooldown_s"]
            ):
                symbol_logger.info(
                    f"\n{NEON_PURPLE}--- TRADING SIGNAL TRIGGERED ---{RESET}"
                )
                symbol_logger.info(
                    f"{NEON_BLUE}Signal:{RESET} {trading_signal.signal_type.value.upper()} (Confidence: {trading_signal.confidence:.2f})"
                )
                symbol_logger.info(
                    f"{NEON_BLUE}Conditions Met:{RESET} {', '.join(trading_signal.conditions_met) if trading_signal.conditions_met else 'None'}"
                )

                if trading_signal.stop_loss and trading_signal.take_profit:
                    symbol_logger.info(
                        f"{NEON_GREEN}Suggested Stop Loss:{RESET} {trading_signal.stop_loss:.5f}"
                    )
                    symbol_logger.info(
                        f"{NEON_GREEN}Suggested Take Profit:{RESET} {trading_signal.take_profit:.5f}"
                    )

                    if trading_signal.risk_reward_ratio:
                        symbol_logger.info(
                            f"{NEON_BLUE}Risk/Reward Ratio:{RESET} {trading_signal.risk_reward_ratio:.2f}"
                        )

                # Calculate position size
                if trading_signal.stop_loss:
                    # Refresh balance before trade
                    fresh_balance = api_client.fetch_balance("USDT")
                    if fresh_balance > 0:
                        account_balance = fresh_balance

                    instrument_info = api_client.fetch_instrument_info(symbol)
                    position_size = risk_manager.calculate_position_size(
                        current_price,
                        trading_signal.stop_loss,
                        account_balance,
                        instrument_info=instrument_info,
                    )

                    if position_size <= 0:
                        symbol_logger.warning(
                            f"{NEON_YELLOW}Position size calculated as zero. Skipping trade execution.{RESET}"
                        )
                        last_signal_time = current_time_seconds
                        continue

                    trading_signal.position_size = position_size
                    symbol_logger.info(
                        f"{NEON_BLUE}Position Size:{RESET} {float(position_size):.4f}"
                    )

                # Add signal to history
                signal_id = signal_tracker.add_signal(trading_signal, current_price)

                # Send notification
                if CONFIG.get("notifications", {}).get("enabled", False):
                    # Get relevant metrics for notification
                    l2 = None
                    dp = None

                    if multi_tf_analyzer and multi_tf_analyzer.last_primary_analyzer:
                        l2 = multi_tf_analyzer.last_primary_analyzer.indicator_values.get(
                            "l2_metrics"
                        )
                        dp = multi_tf_analyzer.last_primary_analyzer.indicator_values.get(
                            "depth_profile"
                        )
                    elif locals().get("analyzer"):
                        l2 = analyzer.indicator_values.get("l2_metrics")
                        dp = analyzer.indicator_values.get("depth_profile")

                    notification_system.send_signal_notification(
                        trading_signal, l2_metrics=l2, depth_profile=dp
                    )

                # Place Actual Order
                order_response = api_client.place_order(
                    symbol=symbol,
                    side=trading_signal.signal_type.value,
                    order_type="Market",
                    qty=trading_signal.position_size,
                    stop_loss=trading_signal.stop_loss,
                    take_profit=trading_signal.take_profit,
                )

                if order_response and order_response.get("retCode") == 0:
                    symbol_logger.info(
                        f"{NEON_GREEN}Order placed successfully! Order ID: {order_response['result'].get('orderId')}{RESET}"
                    )
                else:
                    symbol_logger.error(
                        f"{NEON_RED}Failed to place order: {order_response}{RESET}"
                    )

                last_signal_time = current_time_seconds  # Update last signal time

            # Backup database periodically
            if CONFIG.get("database", {}).get("backup_enabled", True):
                backup_interval = (
                    CONFIG.get("database", {}).get("backup_interval_hours", 24) * 3600
                )
                if time.time() - last_db_backup_time >= backup_interval:
                    backup_path = f"{DATABASE_FILE}.bak_{int(time.time())}"
                    if db_manager.backup_database(backup_path):
                        last_db_backup_time = time.time()

            time.sleep(CONFIG["analysis_interval"])

        except requests.exceptions.RequestException as e:
            symbol_logger.error(
                f"{NEON_RED}Network or API communication error: {e}. Retrying in {CONFIG['retry_delay']} seconds...{RESET}"
            )
            time.sleep(CONFIG["retry_delay"])

        except KeyboardInterrupt:
            symbol_logger.info(f"{NEON_YELLOW}Analysis stopped by user.{RESET}")
            break

        except Exception as e:
            symbol_logger.exception(
                f"{NEON_RED}An unexpected error occurred: {e}. Retrying in {CONFIG['retry_delay']} seconds...{RESET}"
            )
            time.sleep(CONFIG["retry_delay"])


if __name__ == "__main__":
    main()
