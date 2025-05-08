# config_models.py
"""Pydantic Models for Application Configuration using pydantic-settings.

Loads configuration from environment variables and/or a .env file.
Provides type validation and default values for the trading bot.
"""

import os
from decimal import Decimal
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define basic types or Literals as placeholders - MUST match the expected values
# These are used if direct import from helpers fails during early setup,
# but the helpers should ideally define and export these properly.
PositionIdx = Literal[0, 1, 2]  # 0: One-Way, 1: Hedge Buy, 2: Hedge Sell
Category = Literal["linear", "inverse", "spot", "option"]
OrderFilter = Literal["Order", "StopOrder", "tpslOrder", "TakeProfit", "StopLoss"]  # Add others as needed
Side = Literal["buy", "sell"]  # Use lowercase consistent with strategy/ccxt args
TimeInForce = Literal["GTC", "IOC", "FOK", "PostOnly"]
TriggerBy = Literal["LastPrice", "MarkPrice", "IndexPrice"]
TriggerDirection = Literal[1, 2]  # 1: Rise, 2: Fall


class APIConfig(BaseModel):
    """Configuration for Bybit API Connection and Market Defaults."""

    exchange_id: Literal["bybit"] = Field("bybit", description="CCXT Exchange ID")
    api_key: str | None = Field(None, description="Bybit API Key")
    api_secret: str | None = Field(None, description="Bybit API Secret")
    testnet_mode: bool = Field(True, description="Use Bybit Testnet environment")
    default_recv_window: PositiveInt = Field(10000, ge=100, le=60000, description="API request validity window (ms)")

    symbol: str = Field(..., description="Primary trading symbol (e.g., BTC/USDT:USDT)")
    usdt_symbol: str = Field("USDT", description="Quote currency for balance reporting")
    expected_market_type: Literal["swap", "spot", "option", "future"] = Field("swap")
    expected_market_logic: Literal["linear", "inverse"] = Field("linear")

    retry_count: NonNegativeInt = Field(3, description="Default API call retry count")
    retry_delay_seconds: PositiveFloat = Field(2.0, gt=0, description="Default base retry delay (s)")

    maker_fee_rate: Decimal = Field(Decimal("0.0002"), ge=0, description="Maker fee rate")
    taker_fee_rate: Decimal = Field(Decimal("0.00055"), ge=0, description="Taker fee rate")

    default_slippage_pct: Decimal = Field(
        Decimal("0.005"), gt=0, le=Decimal("0.1"), description="Max slippage % for market orders (0.005 = 0.5%)"
    )
    position_qty_epsilon: Decimal = Field(Decimal("1e-9"), gt=0, description="Small value for quantity comparisons")
    shallow_ob_fetch_depth: PositiveInt = Field(5, ge=1, le=50, description="Order book depth for slippage check")
    order_book_fetch_limit: PositiveInt = Field(
        25, ge=1, le=1000, description="Default depth for fetching L2 order book"
    )

    pos_none: Literal["NONE"] = "NONE"
    pos_long: Literal["LONG"] = "LONG"
    pos_short: Literal["SHORT"] = "SHORT"
    side_buy: Side = "buy"
    side_sell: Side = "sell"

    @field_validator("api_key", "api_secret", mode="before")
    @classmethod
    def check_not_placeholder(cls, v: str | None, info) -> str | None:
        if v and isinstance(v, str) and "PLACEHOLDER" in v.upper():
            print(
                f"\033[93mWarning [APIConfig]: Field '{info.field_name}' looks like a placeholder: '{v[:15]}...'\033[0m"
            )
        return v

    @field_validator("symbol", mode="before")
    @classmethod
    def check_and_format_symbol(cls, v: Any) -> str:
        if not isinstance(v, str) or not v:
            raise ValueError("Symbol must be a non-empty string")
        if ":" not in v and "/" not in v:
            raise ValueError(f"Invalid symbol format: '{v}'. Expected 'BASE/QUOTE:SETTLE' or 'BASE/QUOTE'.")
        return v.strip().upper()


class IndicatorSettings(BaseModel):
    """Parameters for Technical Indicator Calculations."""

    min_data_periods: PositiveInt = Field(100, ge=20, description="Min candles for indicators")
    evt_length: PositiveInt = Field(7, gt=1, description="EVT indicator length")
    evt_multiplier: PositiveFloat = Field(2.5, gt=0, description="EVT bands multiplier")
    atr_period: PositiveInt = Field(14, gt=0, description="ATR indicator length")


class AnalysisFlags(BaseModel):
    """Flags to Enable/Disable Specific Indicator Calculations or Features."""

    use_evt: bool = Field(True, description="Enable EVT calculation")
    use_atr: bool = Field(True, description="Enable ATR calculation (for SL/TP)")


class StrategyConfig(BaseModel):
    """Core Strategy Behavior and Parameters."""

    name: str = Field("EhlersVolumetricV1", description="Name of the strategy instance")
    timeframe: str = Field("5m", pattern=r"^\d+[mhdMy]$", description="Candlestick timeframe (e.g., '5m')")
    ohlcv_limit: PositiveInt = Field(200, ge=50, description="Number of candles to fetch for indicators")

    leverage: PositiveInt = Field(10, ge=1, description="Desired leverage")
    default_margin_mode: Literal["cross", "isolated"] = Field("cross")  # Requires UTA Pro for isolated usually
    default_position_mode: Literal["one-way", "hedge"] = Field("one-way")
    risk_per_trade: Decimal = Field(
        Decimal("0.01"), gt=0, le=Decimal("0.1"), description="Fraction of balance to risk (0.01 = 1%)"
    )

    stop_loss_atr_multiplier: Decimal = Field(Decimal("2.0"), gt=0, description="ATR multiplier for stop loss")
    take_profit_atr_multiplier: Decimal = Field(Decimal("3.0"), gt=0, description="ATR multiplier for take profit")
    place_tpsl_as_limit: bool = Field(
        True, description="Place TP/SL as reduce-only Limit orders (True) or use native stops (False)"
    )

    loop_delay_seconds: PositiveInt = Field(60, ge=5, description="Frequency (seconds) to fetch data and check signals")

    # Link indicator settings and flags
    indicator_settings: IndicatorSettings = Field(default_factory=IndicatorSettings)
    analysis_flags: AnalysisFlags = Field(default_factory=AnalysisFlags)
    strategy_params: dict[str, Any] = Field(
        {}, description="Strategy-specific parameters dictionary"
    )  # Populated later
    strategy_info: dict[str, Any] = Field({}, description="Strategy identification dictionary")  # Populated later

    @field_validator("timeframe")
    @classmethod
    def check_timeframe_format(cls, v: str) -> str:
        import re

        if not re.match(r"^\d+[mhdMy]$", v):
            raise ValueError(f"Invalid timeframe format: '{v}'.")
        return v

    @model_validator(mode="after")
    def check_consistency(self) -> "StrategyConfig":
        if self.stop_loss_atr_multiplier > 0 and not self.analysis_flags.use_atr:
            raise ValueError("ATR Multiplier > 0 requires use_atr flag to be True.")
        if self.take_profit_atr_multiplier > 0 and not self.analysis_flags.use_atr:
            raise ValueError("TP ATR Multiplier > 0 requires use_atr flag to be True.")
        # Populate helper dicts after validation
        self.strategy_params = {
            "ehlers_volumetric": {
                "evt_length": self.indicator_settings.evt_length,
                "evt_multiplier": self.indicator_settings.evt_multiplier,
            }
        }
        self.strategy_info = {"name": "ehlers_volumetric"}  # Match key used in strategy file
        return self


class LoggingConfig(BaseModel):
    """Configuration for the Logger Setup."""

    logger_name: str = Field("TradingBot", description="Name for the logger instance")
    log_file: str | None = Field("trading_bot.log", description="Log file path (None to disable)")
    console_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field("INFO")
    file_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field("DEBUG")
    log_rotation_bytes: NonNegativeInt = Field(5 * 1024 * 1024, description="Max log size (bytes), 0 disables rotation")
    log_backup_count: NonNegativeInt = Field(5, description="Number of backup logs")
    third_party_log_level_str: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("WARNING")

    @field_validator("log_file", mode="before")
    @classmethod
    def validate_log_file(cls, v: str | None) -> str | None:
        if v is None or v.strip() == "":
            return None
        if any(char in v for char in ["<", ">", ":", '"', "|", "?", "*"]):
            raise ValueError(f"Log file path '{v}' contains invalid characters.")
        return v.strip()


class SMSConfig(BaseModel):
    """Configuration for SMS Alerting."""

    enable_sms_alerts: bool = Field(False, description="Globally enable/disable SMS alerts")
    use_termux_api: bool = Field(False, description="Use Termux:API for SMS")
    sms_recipient_number: str | None = Field(
        None, pattern=r"^\+?[1-9]\d{1,14}$", description="Recipient phone number (E.164 format)"
    )
    sms_timeout_seconds: PositiveInt = Field(30, ge=5, le=120, description="Timeout for Termux API call (s)")

    @model_validator(mode="after")
    def check_sms_details(self) -> "SMSConfig":
        if self.enable_sms_alerts and self.use_termux_api and not self.sms_recipient_number:
            raise ValueError("Termux SMS enabled, but 'sms_recipient_number' is missing.")
        # Add checks for other providers (e.g., Twilio) if implemented
        return self


class AppConfig(BaseSettings):
    """Master Configuration Model."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        env_prefix="BOT_",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    api_config: APIConfig = Field(default_factory=APIConfig)
    strategy_config: StrategyConfig = Field(default_factory=StrategyConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    sms_config: SMSConfig = Field(default_factory=SMSConfig)

    # Helper method to convert to the old single-class Config format if needed temporarily
    def to_legacy_config_dict(self) -> dict:
        legacy = {}
        legacy.update(self.api_config.model_dump())
        legacy.update(
            self.strategy_config.model_dump(
                exclude={"indicator_settings", "analysis_flags", "strategy_params", "strategy_info"}
            )
        )  # Exclude nested models
        legacy.update(self.logging_config.model_dump())
        legacy.update(self.sms_config.model_dump())
        # Add back nested items needed by old format
        legacy["indicator_settings"] = self.strategy_config.indicator_settings.model_dump()
        legacy["analysis_flags"] = self.strategy_config.analysis_flags.model_dump()
        legacy["strategy_params"] = self.strategy_config.strategy_params
        legacy["strategy"] = self.strategy_config.strategy_info  # Renamed from strategy_info
        # Map level strings back if needed by old logger setup
        legacy["LOG_CONSOLE_LEVEL"] = self.logging_config.console_level_str
        legacy["LOG_FILE_LEVEL"] = self.logging_config.file_level_str
        legacy["LOG_FILE_PATH"] = self.logging_config.log_file
        return legacy


def load_config() -> AppConfig:
    """Loads and validates the application configuration."""
    try:
        print("\033[36mLoading configuration...\033[0m")
        # Determine .env path relative to CWD where script is run
        env_file_path = os.path.join(os.getcwd(), ".env")

        if os.path.exists(env_file_path):
            print(f"Attempting to load from: {env_file_path}")
            config = AppConfig(_env_file=env_file_path)
        else:
            print(f"'.env' file not found at {env_file_path}. Loading from environment variables only.")
            config = AppConfig()

        if config.api_config.api_key and "PLACEHOLDER" in config.api_config.api_key.upper():
            print("\033[91m\033[1mCRITICAL WARNING: API Key is a placeholder. Bot WILL fail authentication.\033[0m")
        if not config.api_config.testnet_mode:
            print("\033[93m\033[1mWARNING: Testnet mode is DISABLED. Bot will attempt LIVE trading.\033[0m")

        print("\033[32mConfiguration loaded successfully.\033[0m")
        return config

    except ValidationError as e:
        print(f"\n{'-' * 20}\033[91m CONFIGURATION VALIDATION FAILED \033[0m{'-' * 20}")
        # error.loc gives a tuple path, e.g., ('api_config', 'symbol')
        for error in e.errors():
            loc_path = " -> ".join(map(str, error["loc"])) if error["loc"] else "AppConfig"
            env_var_suggestion = "BOT_" + "__".join(map(str, error["loc"])).upper()
            print(f"  \033[91mField:\033[0m {loc_path}")
            print(f"  \033[91mError:\033[0m {error['msg']}")
            val_display = repr(error.get("input", "N/A"))
            is_secret = any(s in loc_path.lower() for s in ["key", "secret", "token"])
            if is_secret and isinstance(error.get("input"), str):
                val_display = "'*****'"
            print(f"  \033[91mValue:\033[0m {val_display}")
            print(f"  \033[93mSuggestion:\033[0m Check env var '{env_var_suggestion}' or the field in '.env'.")
            print("-" * 25)
        print(f"{'-' * 60}\n")
        raise SystemExit("\033[91mConfiguration validation failed.\033[0m")

    except Exception as e:
        print(f"\033[91m\033[1mFATAL: Unexpected error loading configuration: {e}\033[0m")
        import traceback

        traceback.print_exc()
        raise SystemExit("\033[91mFailed to load configuration.\033[0m")


# Example usage
if __name__ == "__main__":
    print("Running config_models.py directly for testing...")
    try:
        app_settings = load_config()
        print("\n\033[1mLoaded Config (Partial Example):\033[0m")
        print(f"  Symbol: {app_settings.api_config.symbol}")
        print(f"  Timeframe: {app_settings.strategy_config.timeframe}")
        print(f"  Testnet: {app_settings.api_config.testnet_mode}")
        print(f"  Log File: {app_settings.logging_config.log_file}")
        print(f"  Risk %: {app_settings.strategy_config.risk_per_trade * 100:.2f}%")
        # print("\nFull Config (JSON):\n", app_settings.model_dump_json(indent=2)) # Careful with secrets
        print("\n\033[32mConfiguration test successful.\033[0m")
    except SystemExit as e:
        print(f"\n\033[91mExiting due to configuration error: {e}\033[0m")
