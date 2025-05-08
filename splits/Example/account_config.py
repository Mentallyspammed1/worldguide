# File: account_config.py
# -*- coding: utf-8 -*-

"""
Functions for Configuring Bybit Account Settings (Leverage, Margin, Position Mode).

This module provides functions to interact with the Bybit V5 API (via ccxt)
to configure critical account and position settings like leverage,
margin mode (isolated/cross), and position mode (one-way/hedge).
It includes error handling, specific Bybit API code interpretation,
and leverages a retry mechanism for API calls.
"""

import logging
import sys
from typing import Optional, Literal, Dict, Any

# Standard Library Imports
# (None needed beyond core types used in typing)

# Third-Party Imports
try:
    import ccxt
    from ccxt.base.errors import (
        ExchangeError,
        NetworkError,
        AuthenticationError,
        BadSymbol,
        InvalidOrder,
        OrderNotFound,  # Import more specific errors if needed
    )
except ImportError:
    print("ERROR: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)

try:
    from colorama import Fore, Style, Back

    COLORAMA_AVAILABLE = True
except ImportError:
    print("WARNING: colorama library not found. Logs will not be colored. Install: pip install colorama")
    COLORAMA_AVAILABLE = False

    # Define a dummy class to avoid errors if colorama is not installed
    class DummyColor:
        """Acts as a no-op placeholder if colorama is not installed."""

        def __getattr__(self, name: str) -> str:
            """Return empty string for any color attribute."""
            return ""

    Fore = Style = Back = DummyColor()  # type: ignore

# Local Application/Library Specific Imports
from config import Config  # Assuming Config is a Pydantic model or dataclass
from utils import retry_api_call, _get_v5_category, send_sms_alert

# --- Constants ---
logger = logging.getLogger(__name__)  # Best practice: get logger at module level

# Bybit V5 API Response Codes (Informational)
# See: https://bybit-exchange.github.io/docs/v5/error_code
BYBIT_RET_CODE_OK = 0
# Leverage Specific
BYBIT_ERR_LEVERAGE_NOT_MODIFIED = 110044
# Position Mode Specific
BYBIT_ERR_POS_MODE_NOT_MODIFIED = 110021
BYBIT_ERR_POS_MODE_HAS_POS_ORDER = 110020
# Margin Mode Specific
BYBIT_ERR_MARGIN_MODE_NOT_MODIFIED = 110026
BYBIT_ERR_MARGIN_MODE_HAS_POS_ORDER = 110020  # Same code as position mode switch

# Bybit V5 API Parameter Values
BYBIT_POS_MODE_ONE_WAY = "0"  # Merged Single Position Mode
BYBIT_POS_MODE_HEDGE = "3"  # Both Sides Position Mode (Hedge Mode)
BYBIT_TRADE_MODE_CROSS = 0  # Cross Margin (Default for category-level setting)
BYBIT_TRADE_MODE_ISOLATED = 1  # Isolated Margin (Used in switch_isolated endpoint)

# --- Helper Functions ---


def _log_prefix(func_name: str) -> str:
    """Generates a standardized log prefix."""
    return f"[{func_name}]"


# --- Core Configuration Functions ---


@retry_api_call(max_retries=3, initial_delay=1.0)
def set_leverage(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """
    Sets the leverage for a specific symbol on Bybit V5 (Linear/Inverse).

    Validates the requested leverage against the market's limits and handles the
    'leverage not modified' case gracefully. Applies to both buy and sell leverage
    simultaneously.

    Args:
        exchange: The initialized ccxt.bybit exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        leverage: The desired integer leverage level (must be positive).
        config: The application configuration object.

    Returns:
        True if leverage was set successfully or already set to the desired value.
        False if validation failed, or an unexpected non-retryable error occurred.

    Raises:
        ccxt.NetworkError: If a network issue occurs (for retry decorator).
        ccxt.AuthenticationError: If authentication fails (for retry decorator).
        ccxt.ExchangeError: For other API errors not handled internally (for retry decorator).
        ValueError: If leverage is not positive.
    """
    func_name = "set_leverage"
    log_prefix = _log_prefix(func_name)
    logger.info(f"{Fore.CYAN}{log_prefix} Attempting to set leverage to {leverage}x for {symbol}...{Style.RESET_ALL}")

    if not isinstance(leverage, int) or leverage <= 0:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid leverage value: {leverage}. Leverage must be a positive integer.{Style.RESET_ALL}"
        )
        # Raise ValueError for fundamentally incorrect input, distinguishing from API errors.
        raise ValueError("Leverage must be a positive integer.")

    try:
        # Fetch market data to validate leverage and get category
        market = exchange.load_markets([symbol])[symbol]
        category = _get_v5_category(market)

        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Invalid market type for leverage setting: {symbol} (Category: {category}). Only Linear/Inverse supported.{Style.RESET_ALL}"
            )
            return False

        # Validate leverage against market limits (safer access using .get)
        limits = market.get("limits", {})
        leverage_limits = limits.get("leverage", {})
        min_leverage = leverage_limits.get("min", 1.0)  # Default min 1x
        max_leverage = leverage_limits.get("max", 100.0)  # Default max 100x if not found

        # Allow floating point comparison initially, then cast if needed
        if not (min_leverage <= float(leverage) <= max_leverage):
            logger.error(
                f"{Fore.RED}{log_prefix} Invalid leverage requested: {leverage}x. Allowed range for {symbol}: {min_leverage}x - {max_leverage}x.{Style.RESET_ALL}"
            )
            return False

        # Prepare V5 specific parameters
        params = {
            "category": category,
            "buyLeverage": str(leverage),  # API expects string representation
            "sellLeverage": str(leverage),
        }

        logger.debug(
            f"{log_prefix} Calling exchange.set_leverage with symbol='{symbol}', leverage={leverage}, params={params}"
        )
        response = exchange.set_leverage(leverage, symbol, params=params)
        logger.debug(
            f"{log_prefix} Raw API response: {response}"
        )  # Raw response can be large, consider truncating if needed

        # Assuming success if no exception is raised by ccxt after the call
        logger.info(
            f"{Fore.GREEN}{log_prefix} Leverage set/confirmed to {leverage}x for {symbol} (Category: {category}).{Style.RESET_ALL}"
        )
        return True

    except ExchangeError as e:
        error_code_str = str(e)  # Check the string representation for codes/messages
        # Check if the error indicates leverage was already set
        if (
            f"ret_code={BYBIT_ERR_LEVERAGE_NOT_MODIFIED}" in error_code_str
            or "leverage not modified" in error_code_str.lower()
        ):
            logger.info(
                f"{Fore.CYAN}{log_prefix} Leverage for {symbol} is already set to {leverage}x.{Style.RESET_ALL}"
            )
            return True  # Treat as success
        else:
            logger.error(
                f"{Fore.RED}{log_prefix} ExchangeError setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}"
            )
            raise  # Re-raise for the retry decorator

    except (NetworkError, AuthenticationError) as e:
        # These are typically retryable or indicate configuration issues
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Network/Auth error setting leverage for {symbol}: {e}{Style.RESET_ALL}"
        )
        raise  # Re-raise for the retry decorator

    except BadSymbol as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid symbol error setting leverage: {symbol}. Details: {e}{Style.RESET_ALL}"
        )
        # BadSymbol is usually not retryable with the same input
        return False

    except Exception as e:
        # Catch unexpected errors
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        # Consider sending an alert for unexpected issues
        send_sms_alert(
            f"[{symbol.split('/')[0]}] ERROR: Failed set leverage {leverage}x (Unexpected: {type(e).__name__})", config
        )
        return False


@retry_api_call(max_retries=2, initial_delay=1.0)
def set_position_mode_bybit_v5(
    exchange: ccxt.bybit, symbol_or_category: str, mode: Literal["one-way", "hedge"], config: Config
) -> bool:
    """
    Sets the position mode (One-Way or Hedge) for a Bybit V5 category (Linear/Inverse).

    Uses the `private_post_v5_position_switch_mode` endpoint. This setting applies
    to the entire category (e.g., all linear contracts) derived from the input.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol_or_category: A market symbol (e.g., 'BTC/USDT:USDT') or category name ('linear', 'inverse').
        mode: The desired position mode: 'one-way' or 'hedge'.
        config: The application configuration object.

    Returns:
        True if the mode was set successfully or already set to the target mode.
        False if the category is invalid, input mode is wrong, switching failed due to
              existing positions/orders, or an unexpected non-retryable error occurred.

    Raises:
        ccxt.NetworkError: If a network issue occurs (for retry decorator).
        ccxt.AuthenticationError: If authentication fails (for retry decorator).
        ccxt.ExchangeError: For API errors not handled internally (for retry decorator).
    """
    func_name = "set_position_mode_bybit_v5"
    log_prefix = _log_prefix(func_name)
    mode_str = mode.lower()  # Normalize input mode

    logger.info(
        f"{Fore.CYAN}{log_prefix} Attempting to set position mode to '{mode_str}' for category derived from '{symbol_or_category}'...{Style.RESET_ALL}"
    )

    # Map user-friendly mode name to Bybit API code
    mode_map = {"one-way": BYBIT_POS_MODE_ONE_WAY, "hedge": BYBIT_POS_MODE_HEDGE}
    target_mode_code = mode_map.get(mode_str)

    if target_mode_code is None:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid mode specified: '{mode}'. Must be 'one-way' or 'hedge'.{Style.RESET_ALL}"
        )
        return False

    # Determine the target category
    target_category: Optional[Literal["linear", "inverse"]] = None
    input_lower = symbol_or_category.lower()
    if input_lower in ["linear", "inverse"]:
        target_category = input_lower  # type: ignore
    else:
        try:
            # Attempt to load market to infer category
            market = exchange.load_markets([symbol_or_category])[symbol_or_category]
            category = _get_v5_category(market)
            if category in ["linear", "inverse"]:
                target_category = category
        except BadSymbol:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Input '{symbol_or_category}' is not a valid symbol. Cannot determine category from it.{Style.RESET_ALL}"
            )
        except Exception as e:
            # Catch potential errors during market loading besides BadSymbol
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Could not load market for '{symbol_or_category}' to determine category: {e}{Style.RESET_ALL}"
            )

    if not target_category:
        logger.error(
            f"{Fore.RED}{log_prefix} Could not determine a valid contract category (linear/inverse) from input '{symbol_or_category}'. Cannot set position mode.{Style.RESET_ALL}"
        )
        return False

    logger.debug(
        f"{log_prefix} Target Category: {target_category}, Target Mode Code: {target_mode_code} (representing '{mode_str}')"
    )

    try:
        # Check if the required implicit API method exists in the ccxt instance
        method_name = "private_post_v5_position_switch_mode"
        if not hasattr(exchange, method_name):
            logger.error(
                f"{Fore.RED}{log_prefix} CCXT version or exchange instance does not support '{method_name}'. Cannot set position mode.{Style.RESET_ALL}"
            )
            # This is a setup issue, not retryable.
            return False

        # Prepare parameters for the V5 endpoint
        params = {
            "category": target_category,
            "mode": target_mode_code,
            # 'symbol' is NOT applicable for this endpoint
        }
        logger.debug(f"{log_prefix} Calling '{method_name}' with params: {params}")

        # Make the API call using the implicit method
        response: Dict[str, Any] = getattr(exchange, method_name)(params)
        logger.debug(f"{log_prefix} Raw response from position mode switch: {response}")

        # Process the response
        ret_code = response.get("retCode")
        ret_msg = response.get("retMsg", "").lower()  # Use lower for case-insensitive checks

        # Check for success
        if ret_code == BYBIT_RET_CODE_OK:
            logger.info(
                f"{Fore.GREEN}{log_prefix} Position mode successfully set to '{mode_str}' for category '{target_category}'.{Style.RESET_ALL}"
            )
            return True
        # Check if already set to the desired mode
        elif ret_code == BYBIT_ERR_POS_MODE_NOT_MODIFIED or "position mode is not modified" in ret_msg:
            logger.info(
                f"{Fore.CYAN}{log_prefix} Position mode for category '{target_category}' is already set to '{mode_str}'.{Style.RESET_ALL}"
            )
            return True  # Treat as success
        # Check if switching is blocked by positions or orders
        elif ret_code == BYBIT_ERR_POS_MODE_HAS_POS_ORDER or "have position" in ret_msg or "active order" in ret_msg:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot switch position mode for '{target_category}' to '{mode_str}': Active positions or orders exist. Clear them first. API Msg: '{response.get('retMsg')}'{Style.RESET_ALL}"
            )
            return False  # Not retryable without user action
        # Handle other specific API errors if known, otherwise raise a generic ExchangeError
        else:
            # Raise an error that the retry decorator might catch
            raise ExchangeError(
                f"{log_prefix} Bybit API error setting position mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'"
            )

    except (NetworkError, AuthenticationError) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Auth error setting position mode: {e}{Style.RESET_ALL}")
        raise  # Re-raise for the retry decorator

    except ExchangeError as e:
        # Catch errors raised from the checks above or other ccxt issues
        # Avoid logging duplicate errors if already logged specific codes
        if str(BYBIT_ERR_POS_MODE_HAS_POS_ORDER) not in str(
            e
        ):  # Check if it's the "has position/order" error we already logged
            logger.warning(f"{Fore.YELLOW}{log_prefix} ExchangeError setting position mode: {e}{Style.RESET_ALL}")
        # Decide whether to re-raise based on whether it's potentially retryable
        # For now, assume most ExchangeErrors caught here (except the handled ones) might be worth retrying once.
        raise e

    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error setting position mode: {e}{Style.RESET_ALL}", exc_info=True
        )
        return False


@retry_api_call(max_retries=2, initial_delay=1.0)
def fetch_account_info_bybit_v5(exchange: ccxt.bybit, config: Config) -> Optional[Dict[str, Any]]:
    """
    Fetches general account information from Bybit V5 API (`/v5/account/info`).

    Provides insights into Unified Trading Account (UTA) status, margin mode settings,
    disconnect protection (DCP) status, etc.

    Args:
        exchange: Initialized ccxt.bybit instance.
        config: The application configuration object (potentially used for logging/alerts).

    Returns:
        A dictionary containing parsed account information (including UTA status,
        margin mode, etc.) if successful.
        None if the API call fails after retries or encounters an unexpected error.

    Raises:
        ccxt.NetworkError: If a network issue occurs (for retry decorator).
        ccxt.AuthenticationError: If authentication fails (for retry decorator).
        ccxt.ExchangeError: For API errors not handled internally (for retry decorator).
    """
    func_name = "fetch_account_info_bybit_v5"
    log_prefix = _log_prefix(func_name)
    logger.debug(f"{log_prefix} Fetching Bybit V5 account info...")

    try:
        # Check if the required implicit API method exists
        method_name = "private_get_v5_account_info"
        if not hasattr(exchange, method_name):
            logger.error(
                f"{Fore.RED}{log_prefix} CCXT version or exchange instance does not support '{method_name}'. Cannot fetch account info.{Style.RESET_ALL}"
            )
            return None  # Setup issue

        logger.debug(f"{log_prefix} Calling '{method_name}' endpoint.")
        account_info_raw: Dict[str, Any] = getattr(exchange, method_name)()
        # Log only a portion of the raw response to avoid overly verbose logs
        logger.debug(f"{log_prefix} Raw Account Info response (truncated): {str(account_info_raw)[:500]}...")

        ret_code = account_info_raw.get("retCode")
        ret_msg = account_info_raw.get("retMsg", "N/A")

        if ret_code == BYBIT_RET_CODE_OK and "result" in account_info_raw:
            result = account_info_raw["result"]
            # Define a mapping for unifiedMarginStatus for better logging
            status_map = {
                1: "Regular Account",
                2: "Unified Margin Account (UMA)",
                3: "Unified Trade Account (UTA)",
                4: "Classic Account",  # Added based on potential API values
            }
            unified_status_code = result.get("unifiedMarginStatus")
            uta_status_str = status_map.get(unified_status_code, f"Unknown ({unified_status_code})")

            # Extract key fields into a structured dictionary
            parsed_info = {
                "unifiedMarginStatus": unified_status_code,
                "unifiedMarginStatusStr": uta_status_str,
                "marginMode": result.get("marginMode"),  # e.g., 'REGULAR_MARGIN', 'PORTFOLIO_MARGIN'
                "dcpStatus": result.get("dcpStatus"),  # e.g., 'OFF', 'ON'
                "timeWindow": result.get("timeWindow"),  # DCP time window in seconds
                "smtCode": result.get("smtCode"),  # SMP group ID (if applicable)
                "isMasterTrader": result.get("isMasterTrader"),  # Boolean
                "updateTime": result.get("updateTime"),  # Timestamp string (ms)
                # Include the raw 'result' dict for access to all fields if needed later
                "rawResult": result,
            }

            logger.info(
                f"{log_prefix} Account Info Fetched Successfully: Status={uta_status_str}, "
                f"MarginMode={parsed_info.get('marginMode', 'N/A')}, "
                f"DCP Status={parsed_info.get('dcpStatus', 'N/A')}"
            )
            return parsed_info
        else:
            # Raise an error if the API call was not successful
            raise ExchangeError(
                f"{log_prefix} Failed to fetch account info from Bybit API. Code={ret_code}, Msg='{ret_msg}'"
            )

    except (NetworkError, AuthenticationError) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Auth error fetching account info: {e}{Style.RESET_ALL}")
        raise  # Re-raise for retry decorator

    except ExchangeError as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} ExchangeError fetching account info: {e}{Style.RESET_ALL}")
        raise  # Re-raise for retry decorator

    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching account info: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


@retry_api_call(max_retries=2, initial_delay=1.0)
def set_isolated_margin_bybit_v5(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """
    Sets margin mode to ISOLATED for a specific symbol on Bybit V5 and sets its leverage.

    Uses the V5 endpoint `private_post_v5_position_switch_isolated`. This operation
    cannot be performed if there is an existing position or active orders for the symbol.
    If the symbol is already in isolated mode, it proceeds to set/confirm the leverage.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT') to set isolated margin for.
        leverage: The desired leverage (applied to both buy and sell sides) for the
                  isolated position. Must be a positive integer.
        config: The application configuration object.

    Returns:
        True if isolated mode was set/confirmed successfully AND leverage was applied/confirmed.
        False otherwise (e.g., validation failure, blocked by position/order, leverage setting failed,
              or unexpected non-retryable error).

    Raises:
        ccxt.NetworkError: If a network issue occurs (for retry decorator).
        ccxt.AuthenticationError: If authentication fails (for retry decorator).
        ccxt.ExchangeError: For API errors not handled internally (for retry decorator).
        ccxt.BadSymbol: If the provided symbol is invalid.
        ValueError: If the provided leverage is not a positive integer.
    """
    func_name = "set_isolated_margin_bybit_v5"
    log_prefix = _log_prefix(func_name)
    logger.info(
        f"{Fore.CYAN}{log_prefix} Attempting to set ISOLATED margin for {symbol} with {leverage}x leverage...{Style.RESET_ALL}"
    )

    # --- Input Validation ---
    if not isinstance(leverage, int) or leverage <= 0:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid leverage value: {leverage}. Must be a positive integer.{Style.RESET_ALL}"
        )
        raise ValueError("Leverage must be a positive integer.")

    # Need market info for category and market ID
    try:
        market = exchange.load_markets([symbol])[symbol]
        category = _get_v5_category(market)
        market_id = market["id"]  # Use the exchange-specific market ID

        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot set isolated margin for non-contract symbol: {symbol} (Category: {category}).{Style.RESET_ALL}"
            )
            return False

    except BadSymbol as e:
        logger.error(f"{Fore.RED}{log_prefix} Invalid symbol provided: {symbol}. Error: {e}{Style.RESET_ALL}")
        raise  # Re-raise BadSymbol as it's a fundamental input error

    except Exception as e:
        # Catch errors during market loading
        logger.error(
            f"{Fore.RED}{log_prefix} Failed to load market info for {symbol}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return False

    # --- API Call to Switch Mode ---
    mode_switched_or_confirmed = False
    try:
        # Check if the required implicit API method exists
        method_name = "private_post_v5_position_switch_isolated"
        if not hasattr(exchange, method_name):
            logger.error(
                f"{Fore.RED}{log_prefix} CCXT version or exchange instance does not support '{method_name}'. Cannot set isolated margin.{Style.RESET_ALL}"
            )
            return False  # Setup issue

        # Prepare parameters for the switch_isolated endpoint
        params_switch = {
            "category": category,
            "symbol": market_id,
            "tradeMode": BYBIT_TRADE_MODE_ISOLATED,  # 1 for Isolated Margin
            "buyLeverage": str(leverage),  # API expects string
            "sellLeverage": str(leverage),
        }
        logger.debug(f"{log_prefix} Calling '{method_name}' with params: {params_switch}")

        response: Dict[str, Any] = getattr(exchange, method_name)(params_switch)
        logger.debug(f"{log_prefix} Raw response from {method_name}: {response}")

        ret_code = response.get("retCode")
        ret_msg = response.get("retMsg", "").lower()

        # Process response for the switch attempt
        if ret_code == BYBIT_RET_CODE_OK:
            logger.info(
                f"{Fore.GREEN}{log_prefix} Successfully switched {symbol} to ISOLATED mode (leverage setting pending confirmation).{Style.RESET_ALL}"
            )
            mode_switched_or_confirmed = True
        elif ret_code == BYBIT_ERR_MARGIN_MODE_NOT_MODIFIED or "margin mode is not modified" in ret_msg:
            logger.info(
                f"{Fore.CYAN}{log_prefix} {symbol} is already in ISOLATED mode. Proceeding to verify/set leverage...{Style.RESET_ALL}"
            )
            mode_switched_or_confirmed = True
        elif ret_code == BYBIT_ERR_MARGIN_MODE_HAS_POS_ORDER or "have position" in ret_msg or "active order" in ret_msg:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot switch {symbol} to ISOLATED: Active positions or orders exist. API Msg: '{response.get('retMsg')}'{Style.RESET_ALL}"
            )
            return False  # Not retryable without user action
        else:
            # Raise an error for unexpected API responses during the switch
            raise ExchangeError(
                f"{log_prefix} Bybit API error switching {symbol} to isolated mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'"
            )

    except (NetworkError, AuthenticationError) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Network/Auth error switching to isolated margin for {symbol}: {e}{Style.RESET_ALL}"
        )
        raise  # Re-raise for retry decorator

    except ExchangeError as e:
        # Catch errors raised from checks above or other ccxt issues during switch
        # Avoid duplicate logging if already handled
        if str(BYBIT_ERR_MARGIN_MODE_HAS_POS_ORDER) not in str(e):
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} ExchangeError switching to isolated margin for {symbol}: {e}{Style.RESET_ALL}"
            )
        raise e  # Re-raise for retry

    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error during isolated margin switch for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False

    # --- Set/Confirm Leverage (if mode switch was successful or already set) ---
    if mode_switched_or_confirmed:
        logger.debug(
            f"{log_prefix} Mode is ISOLATED. Now explicitly calling set_leverage to ensure {leverage}x for {symbol}..."
        )
        try:
            # Call the standard set_leverage function (defined earlier)
            # This handles validation, API call, and "already set" messages for leverage itself.
            leverage_set_success = set_leverage(exchange, symbol, leverage, config)

            if leverage_set_success:
                # Final success log only if both steps (mode + leverage) are confirmed
                logger.info(
                    f"{Fore.GREEN}{log_prefix} ISOLATED margin mode and {leverage}x leverage successfully confirmed/set for {symbol}.{Style.RESET_ALL}"
                )
                return True
            else:
                # Mode switch might have worked, but leverage setting failed.
                logger.error(
                    f"{Fore.RED}{log_prefix} Mode for {symbol} is ISOLATED, but failed to set/confirm leverage to {leverage}x. Overall operation failed.{Style.RESET_ALL}"
                )
                return False
        except Exception as e:
            # Catch any exception from the set_leverage call itself
            logger.error(
                f"{Fore.RED}{log_prefix} Error occurred during the final leverage setting step for {symbol} (already in isolated mode): {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            return False
    else:
        # This path should theoretically not be reached if the logic above is correct,
        # but serves as a fallback.
        logger.error(
            f"{Fore.RED}{log_prefix} Failed to switch or confirm ISOLATED mode for {symbol}. Cannot proceed to set leverage.{Style.RESET_ALL}"
        )
        return False


# --- END OF FILE account_config.py ---

# ---------------------------------------------------------------------------
