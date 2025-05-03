# File: account_config.py
# -*- coding: utf-8 -*-

"""
Functions for Configuring Bybit Account Settings (Leverage, Margin, Position Mode)
"""

import logging
import sys
from decimal import Decimal
from typing import Optional, Literal

try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    from colorama import Fore, Style, Back
except ImportError:
    print("Warning: colorama library not found. Logs will not be colored. Install: pip install colorama")
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

from config import Config
from utils import retry_api_call, _get_v5_category, safe_decimal_conversion, send_sms_alert

logger = logging.getLogger(__name__)


# Snippet 2 / Function 2: Set Leverage
@retry_api_call(max_retries=3, initial_delay=1.0)
def set_leverage(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """
    Sets the leverage for a specific symbol on Bybit V5 (Linear/Inverse).

    Validates the requested leverage against the market's limits. Handles the
    'leverage not modified' case gracefully. Applies to both buy and sell leverage.

    Args:
        exchange: The initialized ccxt.bybit exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        leverage: The desired integer leverage level.
        config: The configuration object.

    Returns:
        True if leverage was set successfully or already set to the desired value, False otherwise.

    Raises:
        Reraises CCXT exceptions for the retry decorator. ValueError for invalid leverage input.
    """
    func_name = "set_leverage"
    logger.info(f"{Fore.CYAN}[{func_name}] Setting leverage to {leverage}x for {symbol}...{Style.RESET_ALL}")

    if leverage <= 0:
        logger.error(f"{Fore.RED}[{func_name}] Leverage must be positive. Received: {leverage}{Style.RESET_ALL}")
        return False # Invalid input, don't call API

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']:
            logger.error(f"{Fore.RED}[{func_name}] Invalid market type for leverage setting: {symbol} (Category: {category}).{Style.RESET_ALL}")
            return False

        # Validate leverage against market limits from `market['info']`
        leverage_filter = market.get('info', {}).get('leverageFilter', {})
        max_leverage_str = leverage_filter.get('maxLeverage')
        min_leverage_str = leverage_filter.get('minLeverage', '1') # Default min is 1

        # Use safe decimal conversion for robustness
        max_leverage = int(safe_decimal_conversion(max_leverage_str, default=Decimal('100'))) # Assume 100x if parsing fails
        min_leverage = int(safe_decimal_conversion(min_leverage_str, default=Decimal('1')))   # Assume 1x if parsing fails

        if not (min_leverage <= leverage <= max_leverage):
            logger.error(f"{Fore.RED}[{func_name}] Invalid leverage requested: {leverage}x. Allowed range for {symbol}: {min_leverage}x - {max_leverage}x.{Style.RESET_ALL}")
            return False

        # V5 requires category and string values for buy/sellLeverage
        params = {
            'category': category,
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage)
        }

        logger.debug(f"[{func_name}] Calling exchange.set_leverage with symbol='{symbol}', leverage={leverage}, params={params}")
        response = exchange.set_leverage(leverage, symbol, params=params)

        logger.debug(f"[{func_name}] Leverage API call response (raw): {response}")
        # Use logger.info for success
        logger.info(f"{Fore.GREEN}[{func_name}] Leverage set/confirmed to {leverage}x for {symbol} (Category: {category}).{Style.RESET_ALL}")
        return True

    except ccxt.ExchangeError as e:
        error_str = str(e).lower()
        # Bybit V5 specific error codes/messages for "already set" or "not modified"
        # 110044: Leverage not modified
        # Check for common phrases and the code
        if "leverage not modified" in error_str or "same as input" in error_str or "110044" in str(e):
            logger.info(f"{Fore.CYAN}[{func_name}] Leverage for {symbol} is already set to {leverage}x.{Style.RESET_ALL}")
            return True
        else:
            logger.error(f"{Fore.RED}[{func_name}] ExchangeError setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}")
            raise # Re-raise for retry decorator

    except (ccxt.NetworkError, ccxt.AuthenticationError, ccxt.BadSymbol) as e:
        logger.error(f"{Fore.RED}[{func_name}] API/Symbol error setting leverage for {symbol}: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator

    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{symbol.split('/')[0]}] ERROR: Failed set leverage {leverage}x (Unexpected)", config)
        return False


# Snippet 12 / Function 12: Set Position Mode (One-Way / Hedge)
@retry_api_call(max_retries=2, initial_delay=1.0)
def set_position_mode_bybit_v5(exchange: ccxt.bybit, symbol_or_category: str, mode: Literal['one-way', 'hedge'], config: Config) -> bool:
    """
    Sets the position mode (One-Way or Hedge) for a specific category (Linear/Inverse) on Bybit V5.
    Uses the `private_post_v5_position_switch_mode` endpoint. Handles specific V5 errors.
    Note: This affects the entire category (e.g., all linear contracts), not just one symbol.
    """
    func_name = "set_position_mode_bybit_v5"
    logger.info(f"{Fore.CYAN}[{func_name}] Setting position mode to '{mode}' for category derived from '{symbol_or_category}'...{Style.RESET_ALL}")

    # Map user-friendly mode to Bybit API code (0: Merged Single/One-Way, 3: Both Sides/Hedge)
    mode_map = {'one-way': '0', 'hedge': '3'}
    target_mode_code = mode_map.get(mode.lower())

    if target_mode_code is None:
        logger.error(f"{Fore.RED}[{func_name}] Invalid mode specified: '{mode}'. Use 'one-way' or 'hedge'.{Style.RESET_ALL}")
        return False

    # Determine the target category (linear or inverse)
    target_category: Optional[Literal['linear', 'inverse']] = None
    if symbol_or_category.lower() in ['linear', 'inverse']:
        target_category = symbol_or_category.lower() # type: ignore
    else:
        try:
            market = exchange.market(symbol_or_category)
            target_category = _get_v5_category(market)
            if target_category not in ['linear', 'inverse']:
                target_category = None # Only applicable to linear/inverse contracts
        except (ccxt.BadSymbol, Exception):
            logger.warning(f"[{func_name}] Could not load market for '{symbol_or_category}' to determine category.")
            target_category = None # Fallback

    if not target_category:
        logger.error(f"{Fore.RED}[{func_name}] Could not determine a valid contract category (linear/inverse) from input '{symbol_or_category}'. Cannot set position mode.{Style.RESET_ALL}")
        return False

    logger.debug(f"[{func_name}] Target Category: {target_category}, Target Mode Code: {target_mode_code} (representing '{mode}')")

    try:
        # Check if the specific V5 method exists in the CCXT version
        if not hasattr(exchange, 'private_post_v5_position_switch_mode'):
            logger.error(f"{Fore.RED}[{func_name}] CCXT version does not support 'private_post_v5_position_switch_mode'. Cannot set position mode via this function.{Style.RESET_ALL}")
            return False

        # Prepare parameters for the V5 endpoint
        params = {
            'category': target_category,
            'mode': target_mode_code
            # 'symbol' is NOT used for this endpoint, it applies to the category
        }
        logger.debug(f"[{func_name}] Calling 'private_post_v5_position_switch_mode' with params: {params}")

        # Make the API call
        response = exchange.private_post_v5_position_switch_mode(params)
        logger.debug(f"[{func_name}] Raw response from position mode switch: {response}")

        # Process the response
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', '').lower()

        # Success
        if ret_code == 0:
            logger.info(f"{Fore.GREEN}[{func_name}] Position mode successfully set to '{mode}' for category '{target_category}'.{Style.RESET_ALL}")
            return True
        # Already set to the desired mode
        elif ret_code == 110021 or "position mode is not modified" in ret_msg:
            logger.info(f"{Fore.CYAN}[{func_name}] Position mode for category '{target_category}' is already set to '{mode}'.{Style.RESET_ALL}")
            return True
        # Cannot switch due to existing position or orders
        elif ret_code == 110020 or "have position" in ret_msg or "active order" in ret_msg:
            logger.error(f"{Fore.RED}[{func_name}] Cannot switch position mode for '{target_category}': Active positions or orders exist. Clear them first. API Msg: '{response.get('retMsg')}'{Style.RESET_ALL}")
            return False
        # Other API errors
        else:
            raise ccxt.ExchangeError(f"Bybit API error setting position mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'")

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError) as e:
        # Avoid logging duplicate errors if already handled specific codes above
        if not (isinstance(e, ccxt.ExchangeError) and ret_code in [110020]):
            logger.warning(f"{Fore.YELLOW}[{func_name}] API Error setting position mode: {e}{Style.RESET_ALL}")
        # Re-raise network/auth errors for retry decorator
        if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)):
            raise e
        return False # Return False for handled ExchangeErrors like position exists

    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting position mode: {e}{Style.RESET_ALL}", exc_info=True)
        return False


# Snippet 19 / Function 19: Fetch Account Info (UTA Status, Margin Mode)
@retry_api_call(max_retries=2, initial_delay=1.0)
def fetch_account_info_bybit_v5(exchange: ccxt.bybit, config: Config) -> Optional[Dict[str, Any]]:
    """
    Fetches general account information from Bybit V5 API (`/v5/account/info`).
    Provides insights into UTA status, margin mode settings, etc.
    """
    func_name = "fetch_account_info_bybit_v5"
    logger.debug(f"[{func_name}] Fetching Bybit V5 account info...")

    try:
        # Check if the specific V5 method exists
        if not hasattr(exchange, 'private_get_v5_account_info'):
            logger.error(f"{Fore.RED}[{func_name}] CCXT version does not support 'private_get_v5_account_info'. Cannot fetch detailed account info.{Style.RESET_ALL}")
            return None

        logger.debug(f"[{func_name}] Calling private_get_v5_account_info endpoint.")
        account_info_raw = exchange.private_get_v5_account_info()
        logger.debug(f"[{func_name}] Raw Account Info response: {str(account_info_raw)[:400]}...") # Log truncated response

        ret_code = account_info_raw.get('retCode')
        ret_msg = account_info_raw.get('retMsg')

        if ret_code == 0 and 'result' in account_info_raw:
            result = account_info_raw['result']
            # Extract key fields
            parsed_info = {
                'unifiedMarginStatus': result.get('unifiedMarginStatus'), # 1: Regular Account; 2: Unified Margin Account; 3: Unified Trade Account
                'marginMode': result.get('marginMode'), # REGULAR_MARGIN, PORTFOLIO_MARGIN (Unified accounts only)
                'dcpStatus': result.get('dcpStatus'), # Disconnect protection status
                'timeWindow': result.get('timeWindow'), # Disconnect protection time window
                'smtCode': result.get('smtCode'), # SMP group ID
                'isMasterTrader': result.get('isMasterTrader'), # Whether the account is a master trader account
                'updateTime': result.get('updateTime'), # Last update time (string ms)
                'rawInfo': result # Include the raw result dict for full details
            }
            status_map = {1: "Regular", 2: "Unified Margin", 3: "Unified Trade"}
            uta_status_str = status_map.get(parsed_info['unifiedMarginStatus'], 'Unknown')

            logger.info(f"[{func_name}] Account Info Fetched: UTA Status={uta_status_str} ({parsed_info.get('unifiedMarginStatus', 'N/A')}), "
                        f"MarginMode={parsed_info.get('marginMode', 'N/A')}, DCP Status={parsed_info.get('dcpStatus', 'N/A')}")
            return parsed_info
        else:
            raise ccxt.ExchangeError(f"Failed to fetch/parse account info from Bybit API. Code={ret_code}, Msg='{ret_msg}'")

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching account info: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching account info: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# Snippet 24 / Function 24: Set Isolated Margin (Bybit V5 Specific)
@retry_api_call(max_retries=2, initial_delay=1.0)
def set_isolated_margin_bybit_v5(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """
    Sets margin mode to ISOLATED for a specific symbol on Bybit V5 and sets leverage for it.

    Uses V5 endpoint 'private_post_v5_position_switch_isolated'. Cannot be done if there's
    an existing position or active orders for the symbol.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT') to set isolated margin for.
        leverage: The desired leverage (buy and sell) to set for the isolated position.
        config: Configuration object.

    Returns:
        True if isolated mode was set successfully (or already set) and leverage was applied,
        False otherwise.

    Raises:
        Reraises CCXT exceptions for the retry decorator. Handles specific V5 errors internally.
        ValueError for invalid leverage.
    """
    func_name = "set_isolated_margin_bybit_v5"
    logger.info(f"{Fore.CYAN}[{func_name}] Attempting to set ISOLATED margin mode for {symbol} with {leverage}x leverage...{Style.RESET_ALL}")

    if leverage <= 0:
        logger.error(f"[{func_name}] Invalid leverage provided: {leverage}. Must be positive.")
        # Raise ValueError for invalid input to signal incorrect usage
        raise ValueError("Leverage must be positive.")

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']:
            logger.error(f"{Fore.RED}[{func_name}] Cannot set isolated margin for non-contract symbol: {symbol} (Category: {category}).{Style.RESET_ALL}")
            return False

        # Check if the required V5 method exists
        if not hasattr(exchange, 'private_post_v5_position_switch_isolated'):
            logger.error(f"{Fore.RED}[{func_name}] CCXT version does not support 'private_post_v5_position_switch_isolated'. Cannot set isolated margin via this function.{Style.RESET_ALL}")
            return False

        # Attempt to switch to Isolated Margin Mode using V5 endpoint
        # tradeMode=1 means Isolated Margin
        params_switch = {
            'category': category,
            'symbol': market['id'],
            'tradeMode': 1,
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage)
        }
        logger.debug(f"[{func_name}] Calling 'private_post_v5_position_switch_isolated' with params: {params_switch}")
        response = exchange.private_post_v5_position_switch_isolated(params_switch)
        logger.debug(f"[{func_name}] Raw response from switch_isolated endpoint: {response}")

        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', '').lower()
        already_isolated_or_ok = False

        # Success
        if ret_code == 0:
            logger.info(f"{Fore.GREEN}[{func_name}] Successfully switched {symbol} to ISOLATED with {leverage}x leverage.{Style.RESET_ALL}")
            already_isolated_or_ok = True
        # Already isolated (V5 specific code)
        elif ret_code == 110026 or "margin mode is not modified" in ret_msg:
            logger.info(f"{Fore.CYAN}[{func_name}] {symbol} is already in ISOLATED mode. Verifying leverage...{Style.RESET_ALL}")
            already_isolated_or_ok = True
        # Cannot switch due to existing position or orders
        elif ret_code == 110020 or "have position" in ret_msg or "active order" in ret_msg:
            logger.error(f"{Fore.RED}[{func_name}] Cannot switch {symbol} to ISOLATED: active positions or orders exist. API Msg: '{response.get('retMsg')}'{Style.RESET_ALL}")
            return False
        # Other API errors during switch attempt
        else:
            raise ccxt.ExchangeError(f"Bybit API error switching to isolated mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'")

        # If mode is now isolated (or was already), ensure leverage is set correctly using the standard set_leverage function
        if already_isolated_or_ok:
            logger.debug(f"[{func_name}] Explicitly calling set_leverage to confirm/set leverage {leverage}x for ISOLATED {symbol}...")
            # Call the set_leverage function (defined earlier in this module)
            leverage_set_success = set_leverage(exchange, symbol, leverage, config)
            if leverage_set_success:
                logger.info(f"{Fore.GREEN}[{func_name}] Leverage confirmed/set to {leverage}x for ISOLATED {symbol}. Overall success.{Style.RESET_ALL}")
                return True
            else:
                # This indicates the mode switch might have succeeded (or was already set), but setting leverage failed.
                logger.error(f"{Fore.RED}[{func_name}] Failed to set/confirm leverage {leverage}x after ISOLATED mode switch/check for {symbol}. Overall failure.{Style.RESET_ALL}")
                return False

        return False # Should not be reached if logic is correct

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.BadSymbol, ValueError) as e:
        # Avoid raising again if it's an error code handled explicitly above or a ValueError
        if not (isinstance(e, ccxt.ExchangeError) and ret_code in [110020]) and not isinstance(e, ValueError):
            logger.warning(f"{Fore.YELLOW}[{func_name}] API/Input Error setting isolated margin: {e}{Style.RESET_ALL}")
            raise e # Re-raise other errors for retry decorator
        return False # Return False for handled errors like position exists or invalid input

    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting isolated margin for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return False


# --- END OF FILE account_config.py ---

# ---------------------------------------------------------------------------

