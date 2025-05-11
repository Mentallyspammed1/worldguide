# config.py
"""Master Configuration File for the Trading Bot

Stores all settings for API connection, logging, strategy parameters,
risk management, and optional SMS alerts.

Environment variables are used for sensitive data (API keys, secrets, phone numbers).
Create a `.env` file in the same directory or set system environment variables.
Example .env file:
BYBIT_API_KEY=YOUR_API_KEY
BYBIT_API_SECRET=YOUR_API_SECRET
BYBIT_TESTNET_MODE=true # or false
SYMBOL=BTC/USDT:USDT
LEVERAGE=10
TIMEFRAME=5m
RISK_PER_TRADE=0.01 # 1%
# Optional SMS/Twilio
ENABLE_SMS_ALERTS=false
SMS_RECIPIENT_NUMBER=+12345678900 # Your phone number
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+11234567890 # Your Twilio number
"""

import os
from decimal import Decimal

from dotenv import load_dotenv

# --- Attempt to import PositionIdx for typing ---
try:
    from bybit_helpers import PositionIdx
except ImportError:
    print(
        "Warning: Could not import PositionIdx from bybit_helpers. Using integer values (0, 1, 2) for position_idx."
    )

    # Define simple integer constants as fallback
    class PositionIdxFallback:
        ONE_WAY = 0
        BUY_SIDE = 1
        SELL_SIDE = 2

    PositionIdx = PositionIdxFallback()  # type: ignore

# --- Load .env file if it exists ---
# Create a .env file in the same directory for API keys and other secrets
load_dotenv_result = load_dotenv()
print(f".env file loaded: {load_dotenv_result}")  # Confirm if .env was loaded

# ==============================================================================
# API Configuration
# ==============================================================================
API_CONFIG = {
    # --- Exchange Details ---
    "EXCHANGE_ID": "bybit",  # Keep as 'bybit'
    "API_KEY": os.getenv("BYBIT_API_KEY", "YOUR_API_KEY_PLACEHOLDER"),
    "API_SECRET": os.getenv("BYBIT_API_SECRET", "YOUR_API_SECRET_PLACEHOLDER"),
    "TESTNET_MODE": os.getenv("BYBIT_TESTNET_MODE", "true").lower()
    == "true",  # Default to Testnet
    "DEFAULT_RECV_WINDOW": int(
        os.getenv("DEFAULT_RECV_WINDOW", 10000)
    ),  # API request validity window
    # --- Market & Symbol ---
    "SYMBOL": os.getenv(
        "SYMBOL", "BTC/USDT:USDT"
    ),  # Primary trading symbol (V5 format)
    "USDT_SYMBOL": "USDT",  # Quote currency for balance reporting
    "EXPECTED_MARKET_TYPE": "swap",  # Expected type for validation ('swap', 'spot', 'future')
    "EXPECTED_MARKET_LOGIC": "linear",  # Expected logic for validation ('linear', 'inverse')
    # --- Account Settings ---
    "DEFAULT_MARGIN_MODE": os.getenv(
        "MARGIN_MODE", "cross"
    ).lower(),  # 'cross' or 'isolated' - Affects some API calls
    # Note: Position Mode (One-Way/Hedge) might need setting via API if not account default
    # See STRATEGY_CONFIG['position_idx'] for strategy-specific handling
    # --- Retry & Rate Limit (Used by retry decorators/helpers) ---
    "RETRY_COUNT": int(os.getenv("RETRY_COUNT", 3)),  # Number of retries for API calls
    "RETRY_DELAY_SECONDS": float(
        os.getenv("RETRY_DELAY_SECONDS", 2.0)
    ),  # Base delay for retries
    # --- Fees (Update with your actual fee tier) ---
    "MAKER_FEE_RATE": Decimal(
        os.getenv("BYBIT_MAKER_FEE", "0.0002")
    ),  # Example VIP 0 Maker fee (0.02%)
    "TAKER_FEE_RATE": Decimal(
        os.getenv("BYBIT_TAKER_FEE", "0.00055")
    ),  # Example VIP 0 Taker fee (0.055%)
    # --- Order Defaults & Helpers ---
    "DEFAULT_SLIPPAGE_PCT": Decimal(
        os.getenv("DEFAULT_SLIPPAGE_PCT", "0.005")
    ),  # 0.5% slippage check for market orders
    "POSITION_QTY_EPSILON": Decimal("1e-9"),  # Small value for zero quantity checks
    "SHALLOW_OB_FETCH_DEPTH": int(
        os.getenv("SHALLOW_OB_FETCH_DEPTH", 5)
    ),  # Depth for slippage check OB fetch
    "ORDER_BOOK_FETCH_LIMIT": int(
        os.getenv("ORDER_BOOK_FETCH_LIMIT", 25)
    ),  # Default depth for fetching L2 OB
    # --- Position/Side Constants (Used internally by strategy/helpers) ---
    "POS_NONE": "NONE",
    "POS_LONG": "LONG",
    "POS_SHORT": "SHORT",
    "SIDE_BUY": "buy",
    "SIDE_SELL": "sell",
}

# ==============================================================================
# Logger Configuration
# ==============================================================================
LOGGING_CONFIG = {
    "LOGGER_NAME": os.getenv("LOGGER_NAME", "TradingBot"),
    "LOG_FILE": os.getenv(
        "LOG_FILE_PATH", "trading_bot.log"
    ),  # File name/path for logs
    "CONSOLE_LEVEL_STR": os.getenv(
        "LOG_CONSOLE_LEVEL", "INFO"
    ).upper(),  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "FILE_LEVEL_STR": os.getenv("LOG_FILE_LEVEL", "DEBUG").upper(),
    "LOG_ROTATION_BYTES": int(os.getenv("LOG_ROTATION_MB", "5"))
    * 1024
    * 1024,  # Max log size in MB before rotating
    "LOG_BACKUP_COUNT": int(
        os.getenv("LOG_BACKUP_COUNT", "5")
    ),  # Number of backup log files to keep
    "THIRD_PARTY_LOG_LEVEL_STR": os.getenv(
        "THIRD_PARTY_LOG_LEVEL", "WARNING"
    ).upper(),  # Quieten libs like ccxt, websockets
}

# ==============================================================================
# Strategy Configuration (Example: Ehlers Volumetric)
# ==============================================================================
# *** IMPORTANT: Ensure this section matches the strategy you are running in main.py ***
STRATEGY_CONFIG = {
    # --- General Strategy Settings ---
    "name": "EhlersVolumetricStrategy",  # *** CHANGE THIS if using a different strategy ***
    "timeframe": os.getenv("TIMEFRAME", "5m"),  # Timeframe for OHLCV data
    "polling_interval_seconds": int(
        os.getenv("LOOP_DELAY", "60")
    ),  # How often bot checks signals
    # --- Account/Position Settings for Strategy ---
    "leverage": int(os.getenv("LEVERAGE", "10")),
    # Ensure this matches your actual Bybit account setting (0: One-Way, 1: Hedge Buy, 2: Hedge Sell)
    # Use the Enum for type safety if possible
    "position_idx": PositionIdx(
        int(os.getenv("POSITION_IDX", "0"))
    ),  # Default to One-Way
    # --- Risk Management ---
    "risk_per_trade": Decimal(
        os.getenv("RISK_PER_TRADE", "0.01")
    ),  # e.g., 0.01 = 1% of available balance
    "stop_loss_atr_multiplier": Decimal(
        os.getenv("STOP_LOSS_ATR_MULTIPLIER", "2.5")
    ),  # Multiplier for ATR stop loss distance
    # --- Order Sizing ---
    # Example: Fixed USD amount per trade (strategy calculates quantity based on this and risk)
    "order_amount_usd": Decimal(
        os.getenv("ORDER_AMOUNT_USD", "50.0")
    ),  # Approx. USD value target
    # --- Ehlers Volumetric Specific ---
    # This key is checked directly in the Ehlers Strategy __init__
    "EVT_ENABLED": os.getenv("EVT_ENABLED", "true").lower()
    == "true",  # Must be true for this strategy
    # --- Indicator Settings (Must contain keys needed by indicators.py and strategy) ---
    "indicator_settings": {
        "min_data_periods": int(
            os.getenv("MIN_DATA_PERIODS", "100")
        ),  # Min candles needed
        # EVT Parameters (ensure these match top-level EVT params below)
        "evt_length": int(os.getenv("EVT_LENGTH", "7")),
        "evt_multiplier": float(os.getenv("EVT_MULTIPLIER", "2.5")),
        # ATR Parameters (ensure this matches top-level ATR param below)
        "atr_period": int(os.getenv("STOP_LOSS_ATR_PERIOD", "14")),
        # Add other indicators if needed (e.g., for MACD+RSI strategy)
        # "rsi_period": int(os.getenv("RSI_PERIOD", "14")),
        # "macd_fast": int(os.getenv("MACD_FAST", "12")),
        # "macd_slow": int(os.getenv("MACD_SLOW", "26")),
        # "macd_signal": int(os.getenv("MACD_SIGNAL", "9")),
    },
    # --- Analysis Flags (Control which indicators are calculated by indicators.py) ---
    "analysis_flags": {
        "use_evt": True,  # MUST be True for Ehlers strategy
        "use_atr": True,  # MUST be True for ATR stop-loss
        # Set other flags based on indicators needed
        # "use_rsi": False,
        # "use_macd": False,
        # ... etc
    },
    # --- Top-level Strategy Parameters (Checked directly by strategy class __init__) ---
    # These often duplicate values in indicator_settings for direct access/validation, ensure consistency
    "EVT_LENGTH": int(os.getenv("EVT_LENGTH", "7")),
    "STOP_LOSS_ATR_PERIOD": int(os.getenv("STOP_LOSS_ATR_PERIOD", "14")),
    # --- Strategy Identification for indicators.py (if it uses this structure) ---
    # These might be optional depending on how indicators.py is implemented
    "strategy_params": {
        "ehlers_volumetric": {
            "evt_length": int(os.getenv("EVT_LENGTH", "7")),
            "evt_multiplier": float(os.getenv("EVT_MULTIPLIER", "2.5")),
        }
    },
    "strategy": {
        "name": "ehlers_volumetric"  # Lowercase version often used internally
    },
}  # <<< THIS IS THE CORRECT CLOSING BRACE FOR STRATEGY_CONFIG

# ==============================================================================
# SMS Alert Configuration (Optional - Requires Twilio or other provider logic)
# ==============================================================================
SMS_CONFIG = {
    "ENABLE_SMS_ALERTS": os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true",
    # --- Provider Specific Credentials (Example: Twilio) ---
    "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
    "TWILIO_AUTH_TOKEN": os.getenv("TWILIO_AUTH_TOKEN"),
    "TWILIO_PHONE_NUMBER": os.getenv("TWILIO_PHONE_NUMBER"),  # Your Twilio phone number
    # --- Recipient ---
    "SMS_RECIPIENT_NUMBER": os.getenv("SMS_RECIPIENT_NUMBER"),  # Your phone number
    # --- Other ---
    "SMS_TIMEOUT_SECONDS": int(os.getenv("SMS_TIMEOUT_SECONDS", "30")),
}

# --- Basic Validation for SMS Config ---
if SMS_CONFIG["ENABLE_SMS_ALERTS"]:
    missing_sms_keys = [
        key
        for key in [
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_PHONE_NUMBER",
            "SMS_RECIPIENT_NUMBER",
        ]
        if not SMS_CONFIG.get(key)
    ]
    if missing_sms_keys:
        print(
            f"Warning: SMS alerts enabled, but required configuration keys are missing: {', '.join(missing_sms_keys)}"
        )
        print("         Disabling SMS alerts.")
        SMS_CONFIG["ENABLE_SMS_ALERTS"] = False

# --- Basic Validation for API Config ---
if (
    not API_CONFIG.get("API_KEY")
    or not API_CONFIG.get("API_SECRET")
    or "PLACEHOLDER" in API_CONFIG.get("API_KEY", "")
    or "PLACEHOLDER" in API_CONFIG.get("API_SECRET", "")
):
    print("-" * 60)
    print(
        "WARNING: API_KEY or API_SECRET environment variable not set or using placeholder."
    )
    print("         Authenticated functions WILL FAIL.")
    print("         Set environment variables (e.g., in .env file) for live trading.")
    print("-" * 60)
    # Optional: Exit if keys are missing and required for operation?
    # sys.exit("API keys missing, cannot proceed.")

print(
    f"Configuration loaded successfully. Testnet Mode: {API_CONFIG.get('TESTNET_MODE')}"
)
