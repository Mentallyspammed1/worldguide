#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: 2.2
# Changelog:
# - v2.2: Added initial system clock synchronization check against Bybit server time.
#         Improved configuration validation and error handling with SystemExit.
#         Refined logging messages and error handling within the position loop.
#         Added request timeouts and exponential backoff for retries on specific errors.
#         Added explicit warning about .env security.
# - v2.1: Refined trailing stop logic to set the distance only once when position becomes profitable and stop is not set.
#         Improved configuration validation and error handling. Added precision formatting for trailing stop distance.
# - v2.0: Fixed error 10004 by including recv_window in signature, added logging, retry logic, and improved configurability.
# - v1.0: Initial version with basic trailing stop functionality.

import os
import time
import hmac
import hashlib
import requests
import logging
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# --- CONFIGURATION ---

# Load environment variables from a .env file
load_dotenv()

# --- !!! SECURITY NOTE !!! ---
# Ensure your .env file is NOT committed to version control.
# Add .env to your .gitignore file.
# Example .env:
# BYBIT_API_KEY=your_api_key
# BYBIT_API_SECRET=your_api_secret
# BYBIT_BASE_URL=https://api.bybit.com # Use https://api-testnet.bybit.com for testnet
# BYBIT_RECV_WINDOW=5000 # Recommended default is 5000, can be up to 60000
# TRAILING_PERCENT=0.01  # 1% trailing distance relative to mark price when set
# CHECK_INTERVAL=5       # Check interval in seconds
# CATEGORY="linear"      # Trading category: "linear", "inverse", "spot"
# LOG_LEVEL="INFO"       # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
# TRAILING_DISTANCE_PRECISION=8 # Decimal places for trailing stop distance
# MAX_RETRIES=5          # Maximum retries for API calls
# RETRY_BACKOFF_FACTOR=2 # Seconds to wait before retrying (exponential backoff base)
# REQUEST_TIMEOUT=15     # Timeout for API requests in seconds
# ----------------------------

# Initialize logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Validate environment variables and set configuration
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

if not API_KEY or not API_SECRET:
    logger.error("BYBIT_API_KEY and BYBIT_API_SECRET must be set in your .env file.")
    # Exit immediately as API access is not possible
    raise SystemExit(1)

BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
RECV_WINDOW = os.getenv("BYBIT_RECV_WINDOW", "5000")  # Default 5000ms as per docs

try:
    TRAILING_PERCENT = float(os.getenv("TRAILING_PERCENT", "0.01"))
    if not 0 < TRAILING_PERCENT < 1:
        logger.warning(
            f"TRAILING_PERCENT ({TRAILING_PERCENT}) is outside typical range (0, 1). Ensure this is intended."
        )
    CHECK_INTERVAL = float(os.getenv("CHECK_INTERVAL", "5"))
    if CHECK_INTERVAL <= 0:
        logger.error("CHECK_INTERVAL must be a positive number.")
        raise ValueError("Invalid CHECK_INTERVAL")  # Raise to be caught below
    RECV_WINDOW_INT = int(RECV_WINDOW)  # Validate recv_window is convertible to int
    if RECV_WINDOW_INT <= 0 or RECV_WINDOW_INT > 60000:
        logger.warning(
            f"BYBIT_RECV_WINDOW ({RECV_WINDOW}) is outside typical range (1, 60000). Ensure this is intended."
        )
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    if MAX_RETRIES < 0:
        logger.error("MAX_RETRIES cannot be negative.")
        raise ValueError("Invalid MAX_RETRIES")
    RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "2"))
    if RETRY_BACKOFF_FACTOR <= 0:
        logger.error("RETRY_BACKOFF_FACTOR must be positive.")
        raise ValueError("Invalid RETRY_BACKOFF_FACTOR")
    REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
    if REQUEST_TIMEOUT <= 0:
        logger.error("REQUEST_TIMEOUT must be positive.")
        raise ValueError("Invalid REQUEST_TIMEOUT")
    TRAILING_DISTANCE_PRECISION = int(os.getenv("TRAILING_DISTANCE_PRECISION", "8"))
    if TRAILING_DISTANCE_PRECISION < 0:
        logger.error("TRAILING_DISTANCE_PRECISION cannot be negative.")
        raise ValueError("Invalid TRAILING_DISTANCE_PRECISION")

except ValueError as e:
    logger.error(f"Invalid numeric configuration value: {e}. Check your .env file.")
    # Exit on critical configuration errors
    raise SystemExit(1)

CATEGORY = os.getenv("CATEGORY", "linear").lower()
if CATEGORY not in ["linear", "inverse", "spot"]:
    logger.error(f"Unsupported CATEGORY: {CATEGORY}. Must be 'linear', 'inverse', or 'spot'.")
    # Exit on critical configuration errors
    raise SystemExit(1)
if CATEGORY == "spot":
    logger.warning(
        "Trailing stops are typically for derivatives (linear/inverse). Spot market trailing stops might behave differently or not be supported via this endpoint."
    )

# Define API Endpoints
ENDPOINTS = {
    "POSITION_LIST": "/v5/position/list",
    "TRADING_STOP": "/v5/position/trading-stop",
    "MARKET_TIME": "/v5/market/time",
}

logger.info(
    f"Configuration loaded: BASE_URL={BASE_URL}, CATEGORY={CATEGORY}, "
    f"TRAILING_PERCENT={TRAILING_PERCENT:.2%}, CHECK_INTERVAL={CHECK_INTERVAL}s, "
    f"RECV_WINDOW={RECV_WINDOW}ms, MAX_RETRIES={MAX_RETRIES}, "
    f"RETRY_BACKOFF_FACTOR={RETRY_BACKOFF_FACTOR}s, REQUEST_TIMEOUT={REQUEST_TIMEOUT}s, "
    f"TRAILING_DISTANCE_PRECISION={TRAILING_DISTANCE_PRECISION}"
)

# --- SIGNING HELPERS ---


def sign_v5(api_key: str, api_secret: str, timestamp: str, params: Dict[str, Any], recv_window: str) -> str:
    """
    Generates the signature for Bybit V5 API requests.

    Args:
        api_key: Bybit API key.
        api_secret: Bybit API secret.
        timestamp: Current timestamp in milliseconds (as a string).
        params: Dictionary of request parameters.
        recv_window: Receive window in milliseconds (as a string).

    Returns:
        HMAC-SHA256 signature as a hexadecimal string.
    """
    # Bybit V5 signature string format: timestamp + api_key + recv_window + sorted_params_string
    # Params must be sorted alphabetically by key
    # Ensure boolean values are represented as lowercase strings 'true' or 'false' for signing
    sorted_params = sorted(params.items())
    param_list = []
    for k, v in sorted_params:
        if isinstance(v, bool):
            param_list.append(f"{k}={str(v).lower()}")
        else:
            # urlencode handles string conversion for other types appropriately
            param_list.append(f"{k}={requests.utils.quote(str(v))}")  # Use requests' quote for consistency

    param_str = "&".join(param_list)

    to_sign = f"{timestamp}{api_key}{recv_window}{param_str}"
    logger.debug(f"String to sign: {to_sign}")

    signature = hmac.new(api_secret.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return signature


def make_request(
    method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, retries: int = MAX_RETRIES
) -> Optional[Dict]:
    """
    Makes a signed request to the Bybit V5 API with retry logic.

    Args:
        method: HTTP method ("GET" or "POST").
        endpoint: API endpoint path (e.g., "/v5/position/list").
        params: Dictionary of parameters (query params for GET, JSON body for POST).
        retries: Number of retries for failed requests.

    Returns:
        JSON response as a dictionary, or None if the request failed after retries.
    """
    if params is None:
        params = {}

    url = BASE_URL + endpoint

    for attempt in range(1, retries + 1):
        timestamp = str(int(time.time() * 1000))
        # Parameters for signing are the actual parameters sent in the request
        signature = sign_v5(API_KEY, API_SECRET, timestamp, params, RECV_WINDOW)

        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "Content-Type": "application/json",  # Required for POST, good practice for others
        }

        try:
            logger.debug(f"Attempt {attempt}/{retries}: {method} {url}")
            logger.debug(f"  Headers: {headers}")
            logger.debug(f"  Params: {params}")

            response = None
            if method == "GET":
                # GET parameters go in the query string for the actual request
                response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            elif method == "POST":
                # POST parameters go in the JSON body for the actual request
                response = requests.post(url, headers=headers, json=params, timeout=REQUEST_TIMEOUT)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None

            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            json_response = response.json()

            ret_code = json_response.get("retCode")
            ret_msg = json_response.get("retMsg")

            if ret_code == 0:
                logger.debug(f"API Success: {endpoint}")
                return json_response
            else:
                # Log API-specific errors but potentially retry depending on the code
                logger.warning(
                    f"API error (retCode={ret_code}): {ret_msg} for {endpoint}. Attempt {attempt}/{retries}."
                )
                # Common retryable errors: rate limits (10001), system errors (10006), recv_window (10004), network issues
                # Bybit V5 API retCode list: https://bybit-exchange.github.io/docs/v5/error-code
                # Add 10004 (recv_window error) to retryable list as time sync can be tricky
                retryable_codes = [10001, 10004, 10006]
                if ret_code in retryable_codes and attempt < retries:
                    sleep_time = RETRY_BACKOFF_FACTOR * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {sleep_time:.2f} seconds due to retCode {ret_code}...")
                    time.sleep(sleep_time)
                    continue  # Go to next attempt
                else:
                    # Non-retryable or max retries reached
                    logger.error(
                        f"API call failed permanently after {attempt} attempts: {endpoint} (retCode={ret_code}, retMsg={ret_msg})"
                    )
                    return None

        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out for {endpoint}. Attempt {attempt}/{retries}.")
            if attempt < retries:
                sleep_time = RETRY_BACKOFF_FACTOR * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time:.2f} seconds due to timeout...")
                time.sleep(sleep_time)
                continue  # Go to next attempt
            else:
                logger.error(f"Max retries reached for timeout on {endpoint}. Giving up.")
                return None

        except requests.exceptions.RequestException as e:
            # Catch other requests exceptions (connection errors, http errors etc.)
            logger.error(f"Request failed for {endpoint} (attempt {attempt}/{retries}): {e}")
            if response is not None:
                logger.error(f"Status Code: {response.status_code}, Response Body: {response.text}")
            if attempt < retries:
                sleep_time = RETRY_BACKOFF_FACTOR * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time:.2f} seconds due to request exception...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for {endpoint}. Giving up.")
                return None
        except Exception as e:
            # Catch unexpected errors during request processing (e.g., JSON parsing)
            logger.exception(f"Unexpected error during request processing for {endpoint}: {e}")
            return None  # Do not retry on unexpected errors unless specifically handled


# --- TRAILING STOP LOGIC ---


def fetch_positions() -> Optional[Dict]:
    """
    Fetches the list of current positions.

    Returns:
        Dictionary containing position data ('result' key with 'list' inside), or None on failure.
    """
    params = {"category": CATEGORY}
    response = make_request("GET", ENDPOINTS["POSITION_LIST"], params)

    if response and response.get("retCode") == 0:
        # Basic validation of response structure
        result = response.get("result")
        if result and isinstance(result.get("list"), list):
            return result
        else:
            logger.error(f"Unexpected position list response structure: {response}. Missing 'result' or 'list'.")
            return None
    # make_request already logged the error if response is None or retCode != 0
    return None


def set_trailing_stop(symbol: str, side: str, trailing_distance: float) -> bool:
    """
    Sets or updates the trailing stop for a given position.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        side: Position side ("Buy" for long, "Sell" for short) - primarily for logging context.
        trailing_distance: Absolute distance from market price for the trailing stop.
                         Bybit's trailing stop mechanism uses this distance.

    Returns:
        True if successful, False otherwise.
    """
    # Format distance to the required precision
    # Note: The API parameter is `trailingStop`, which is the *absolute distance*.
    # Bybit's system then manages the actual Stop Loss price based on price movement
    # and this distance.
    try:
        # Use f-string formatting with dynamic precision
        trailing_distance_str = f"{trailing_distance:.{TRAILING_DISTANCE_PRECISION}f}"
    except Exception as e:
        logger.error(
            f"Error formatting trailing distance {trailing_distance} with precision {TRAILING_DISTANCE_PRECISION}: {e}"
        )
        return False  # Cannot proceed if formatting fails

    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "trailingStop": trailing_distance_str,
        # "tpslMode": "Full" # No longer required for V5 position TP/SL/Trailing updates
    }
    # Use the dedicated trading-stop endpoint
    response = make_request("POST", ENDPOINTS["TRADING_STOP"], params)

    if response and response.get("retCode") == 0:
        logger.info(
            f"Successfully set trailing stop for {symbol} [{side}] at absolute distance {trailing_distance_str}"
        )
        return True
    else:
        # make_request logs the error details
        return False


def check_server_time_sync() -> bool:
    """
    Checks if the system clock is synchronized with Bybit's server time within the recv_window.

    Returns:
        True if synchronized within RECV_WINDOW, False otherwise.
    """
    # Use fewer retries for a quick check, as time sync issues are often persistent
    response = make_request("GET", ENDPOINTS["MARKET_TIME"], retries=2)
    if response and response.get("retCode") == 0:
        try:
            server_time_nano = int(response.get("result", {}).get("timeNano", 0))
            if server_time_nano == 0:
                logger.warning("Received invalid server time (0) for clock sync check.")
                return False
            server_time_ms = server_time_nano // 1_000_000  # Convert nanoseconds to milliseconds
            local_time_ms = int(time.time() * 1000)
            time_diff = abs(server_time_ms - local_time_ms)

            # Ensure RECV_WINDOW is treated as an integer for comparison
            recv_window_int = int(RECV_WINDOW)

            if time_diff > recv_window_int:
                logger.warning(
                    f"System clock out of sync with Bybit server by {time_diff}ms. Required sync <= {recv_window_int}ms."
                )
                logger.warning(
                    "Consider using NTP or similar to synchronize your system clock. API errors (like retCode 10004) may occur."
                )
                return False
            logger.debug(f"System clock synchronized (diff: {time_diff}ms < {recv_window_int}ms)")
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing server time response: {e}")
            return False
    # make_request logs failure to fetch time
    logger.error("Failed to fetch server time for clock synchronization check.")
    return False


def manage_trailing_stops():
    """
    Main loop to fetch positions and manage trailing stops.
    The strategy is to set a trailing stop distance once a position becomes profitable
    and a trailing stop is not already active for that position.
    The distance is calculated as TRAILING_PERCENT of the current mark price.
    Bybit's system then manages the actual trailing stop price based on this distance.
    """
    logger.info("Starting Bybit Trailing Stop Manager...")

    # Initial clock sync check
    if not check_server_time_sync():
        # Logged inside the function. Decide if this is a fatal error or just a warning.
        # For now, warn and continue, but API requests might fail due to 10004.
        pass

    while True:
        logger.info(f"--- Checking positions at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        position_data = fetch_positions()

        if not position_data:
            logger.error("Failed to fetch positions. Skipping this check cycle.")
            time.sleep(CHECK_INTERVAL)
            continue

        positions = position_data.get("list", [])
        if not positions:
            logger.info("No open positions found.")
            time.sleep(CHECK_INTERVAL)
            continue

        profitable_positions_found = 0
        profitable_positions_ts_not_set = 0
        profitable_positions_ts_set = 0

        for pos in positions:
            try:
                # Extract required position details with default values for safety
                # Use .get() with a default that allows type conversion to fail gracefully below
                symbol = pos.get("symbol")
                side = pos.get("side")
                size_str = pos.get("size", "0")
                unrealised_pnl_str = pos.get("unrealisedPnl", "0")
                mark_price_str = pos.get("markPrice", "0")
                # 'trailingStop' from the list endpoint is the parameter distance, not the actual price
                current_trailing_stop_param = pos.get("trailingStop", "")  # Can be '', '0', or a number string

                # Validate essential fields
                if not symbol or not side:
                    logger.warning(f"Skipping position with missing symbol or side: {pos}")
                    continue

                # Convert numeric fields, handling potential errors
                try:
                    size = float(size_str)
                    unrealised_pnl = float(unrealised_pnl_str)
                    mark_price = float(mark_price_str)
                except (ValueError, TypeError):
                    logger.error(f"Skipping position {symbol} [{side}] due to invalid numeric data. Raw data: {pos}")
                    continue  # Skip this position

                if size == 0:
                    # Skip positions with zero size
                    logger.debug(f"Skipping zero-size position: {symbol} [{side}]")
                    continue

                is_profitable = unrealised_pnl > 0
                # Consider trailing stop set if the parameter is a non-empty string representation of a number greater than 0
                try:
                    is_trailing_stop_set = float(current_trailing_stop_param or "0") > 0
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not parse trailingStop parameter '{current_trailing_stop_param}' for {symbol}. Assuming not set."
                    )
                    is_trailing_stop_set = False

                log_msg = (
                    f"Position: {symbol} [{side}], Size: {size}, PnL: {unrealised_pnl:.4f}, "
                    f"Price: {mark_price:.4f}, Trailing Stop Param: {current_trailing_stop_param or 'None Set'}"
                )

                if is_profitable:
                    profitable_positions_found += 1
                    logger.info(f"{log_msg} - Profitable.")

                    if not is_trailing_stop_set:
                        profitable_positions_ts_not_set += 1
                        # Position is profitable and trailing stop is NOT set.
                        # Calculate the distance based on the current mark price and TRAILING_PERCENT.
                        # This distance will be set on the position, and Bybit will manage the stop price.
                        trailing_distance_to_set = TRAILING_PERCENT * mark_price

                        # Ensure distance is positive (should be if mark_price > 0 and TRAILING_PERCENT > 0)
                        if trailing_distance_to_set <= 0:
                            logger.warning(
                                f"Calculated trailing distance for {symbol} is zero or negative ({trailing_distance_to_set:.8f}). Skipping set."
                            )
                            continue

                        logger.info(
                            f"  -> Profitable and TS not set. Calculating initial TS distance: "
                            f"{TRAILING_PERCENT:.2%} of {mark_price:.4f} = {trailing_distance_to_set:.8f}"
                        )

                        # Set the trailing stop distance via API
                        success = set_trailing_stop(symbol, side, trailing_distance_to_set)
                        if not success:
                            logger.error(
                                f"  -> Failed to set trailing stop for {symbol} [{side}]. Will retry on next cycle."
                            )
                    else:
                        profitable_positions_ts_set += 1
                        # Position is profitable and trailing stop is already set.
                        # Assuming Bybit is managing the trail based on the parameter already set.
                        # No action needed by the script in this simple strategy.
                        logger.debug(
                            f"  -> Trailing stop already set for profitable position {symbol} [{side}]. No update needed."
                        )

                else:
                    # Position is not profitable or PnL is zero.
                    logger.debug(f"{log_msg} - Not profitable. No trailing stop action needed.")

            except Exception as e:
                logger.exception(
                    f"Unexpected error processing position data for {pos.get('symbol', 'Unknown')}: {e}. Raw data: {pos}"
                )
                # Continue to the next position even if one fails

        if positions:  # Only log summary if there were positions fetched
            logger.info(
                f"Summary: Total positions checked: {len(positions)}, "
                f"Profitable: {profitable_positions_found}, "
                f"TS not set (actionable): {profitable_positions_ts_not_set}, "
                f"TS already set: {profitable_positions_ts_set}."
            )
        else:
            logger.info("No positions to process.")

        logger.info(f"--- Finished checking positions. Waiting {CHECK_INTERVAL} seconds. ---")
        time.sleep(CHECK_INTERVAL)


# --- MAIN START ---

if __name__ == "__main__":
    # Configuration validation already happens at the top level
    # and raises SystemExit if critical errors are found.

    try:
        manage_trailing_stops()
    except KeyboardInterrupt:
        logger.info("Script stopped by user (KeyboardInterrupt).")
        # Exit cleanly
        SystemExit(0)
    except SystemExit:
        # Catch SystemExit raised internally for controlled shutdown
        # The appropriate error message is already logged.
        pass
    except Exception as e:
        logger.exception(f"Unhandled fatal error occurred: {e}")
        # Exit with a non-zero code indicating failure
        raise SystemExit(1)
