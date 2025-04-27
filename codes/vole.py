Okay, let's break down the issues shown in the logs and apply fixes to the provided Python code.

**Analysis of the Log Issues:**

1.  **Initial Balance Error:**
    *   **Log Message:** `Could not fetch initial balance... bybit {"retCode":10001,"retMsg":"accountType only support UNIFIED."...}`
    *   **Problem:** The code is trying to fetch the balance using an `accountType` parameter that isn't compatible with the user's Bybit account setup. The API explicitly states it *only* supports `UNIFIED` for this specific key/account. The code likely defaults to `CONTRACT` or another type.
    *   **Location in Code:** The `fetch_balance` function hardcodes `params = {'accountType': 'CONTRACT'}`.
    *   **Fix:** Change the `accountType` parameter within the `fetch_balance` function to `'UNIFIED'`.

2.  **Insufficient Kline Data Warning:**
    *   **Log Message:** `Insufficient kline data after fetch: 1000 rows, require 1110. Skipping cycle.` (Repeatedly)
    *   **Problem:** The bot successfully fetches 1000 historical data points (klines), which is often the maximum limit for a single API request on Bybit. However, the strategy calculations (`strategy_engine.min_data_len`) require 1110 data points. This requirement likely comes from a very long lookback period in one of the indicators.
    *   **Location in Code:** The requirement `1110` stems from `VolumaticOBStrategy._calculate_min_data_length()`. Looking at the default indicator parameters, `DEFAULT_VT_VOL_EMA_LENGTH = 1060` is the largest. The calculation is `max(lookbacks) + buffer`, so `1060 + 50 = 1110`. This 1060-period lookback for volume normalization (percentile rank) is exceptionally long and causes the issue.
    *   **Fix Options:**
        *   **(Recommended & Simpler):** Reduce the lookback period causing the problem. Decrease `DEFAULT_VT_VOL_EMA_LENGTH` to a more reasonable value (e.g., 100 or 200). A 1060-period volume percentile rank is unlikely to be necessary.
        *   **(More Complex):** Implement pagination in `fetch_klines_ccxt`. Make multiple API calls to fetch data in chunks (e.g., 1000 per call) until the required `min_data_len` is met. This is more robust but involves significant changes to the data fetching logic.
    *   **Chosen Fix:** We will apply the simpler fix by reducing the default `DEFAULT_VT_VOL_EMA_LENGTH`.

**Code Fixes:**

1.  **Fix `fetch_balance` Account Type:**

    ```python
    @ccxt_retry_decorator
    def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
        """Fetches the available balance for a specific currency.

        Args:
            exchange: Initialized CCXT exchange object.
            currency: The currency code (e.g., 'USDT').
            logger: The logger instance.

        Returns:
            The available balance as a Decimal, or None if fetching fails.
        """
        logger.debug(f"Fetching available balance for {currency}...")
        balance_info = {} # Initialize
        try:
            # Adjust params for Bybit account types if necessary
            # Common params: {'type': 'CONTRACT', 'accountType': 'CONTRACT'} for derivatives
            # Or {'type': 'SPOT', 'accountType': 'SPOT'} for spot
            # Or {'accountType': 'UNIFIED'} for UTA

            # *** FIX APPLIED HERE ***
            # Changed 'CONTRACT' to 'UNIFIED' based on API error message 10001.
            # Ensure your API key corresponds to a Unified Trading Account on Bybit.
            # If using a different account type (Standard Contract/Spot), adjust accordingly
            # or check Bybit API documentation for the correct parameter.
            params = {'accountType': 'UNIFIED'}
            balance_info = exchange.fetch_balance(params=params)

            # ... (rest of the function remains the same) ...

            # Navigate the balance structure (can vary slightly between exchanges/accounts)
            # Prefer 'free' balance, fallback to specific currency entry's 'free'
            free_balance = balance_info.get('free', {}).get(currency)
            currency_data = balance_info.get(currency, {})
            available = currency_data.get('free') # Check currency-specific 'free'
            total_balance = balance_info.get('total', {}).get(currency) # Use total as last resort
            # Bybit UTA might structure balance differently, e.g., within account types
            # Check the raw balance_info if the below logic fails for UTA
            # Example potential path for UTA USDT: balance_info['info']['result']['list'][0]['coin'][idx]['availableToWithdraw']
            # The standard ccxt fetch_balance *should* normalize this, but double-check if issues persist.
            logger.debug(f"Raw balance info structure: {balance_info}") # Add debug log

            balance_value = None
            if free_balance is not None:
                balance_value = free_balance
            elif available is not None:
                 balance_value = available
            # Check UTA specific structure if needed (Example, adapt based on actual structure)
            elif 'info' in balance_info and 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
                 try:
                     # Find the relevant account section (e.g., UNIFIED or CONTRACT within UTA)
                     unified_acc = next((acc for acc in balance_info['info']['result']['list'] if acc.get('accountType') == 'UNIFIED'), None)
                     if unified_acc and 'coin' in unified_acc:
                         usdt_coin = next((coin for coin in unified_acc['coin'] if coin.get('coin') == currency), None)
                         if usdt_coin and 'availableToWithdraw' in usdt_coin: # Or 'walletBalance' or 'availableBalance' depending on need
                              balance_value = usdt_coin['availableToWithdraw']
                              logger.debug(f"Using balance from UTA structure: {balance_value}")
                 except (IndexError, TypeError, StopIteration):
                     logger.debug("Could not extract balance from specific UTA structure path.")

            elif total_balance is not None:
                 logger.warning(f"Using 'total' balance for {currency} as 'free'/'available' not found directly.")
                 balance_value = total_balance
            else:
                 logger.error(f"Could not find balance information for {currency} in response.")
                 logger.debug(f"Full Balance Info: {balance_info}")
                 return None

            balance = safe_decimal(balance_value)
            if balance is None:
                logger.error(f"Failed to convert balance value '{balance_value}' to Decimal for {currency}.")
                return None

            if balance < 0:
                 logger.warning(f"Reported available balance for {currency} is negative ({balance}). Treating as 0.")
                 return Decimal('0')

            logger.info(f"Available {currency} balance: {balance:.4f}")
            return balance

        except ccxt.AuthenticationError as e: # Catch Auth Error specifically here too
            logger.critical(f"{NEON_RED}Authentication failed fetching balance: {e}. Check API Key/Secret.{RESET}")
            raise # Reraise for retry logic or main loop handling
        except ccxt.ExchangeError as e:
             # Check if it's the same accountType error again
             if "10001" in str(e) and "accountType" in str(e):
                 logger.error(f"Still receiving accountType error (10001) for {currency} even after setting UNIFIED. Check Bybit account status/API key permissions. Error: {e}")
             else:
                 logger.error(f"CCXT ExchangeError fetching balance for {currency}: {e}")
             raise # Reraise for retry
        except ccxt.NetworkError as e:
             logger.error(f"CCXT NetworkError fetching balance for {currency}: {e}")
             raise # Reraise for retry
        except (InvalidOperation, TypeError, ValueError) as e:
             logger.error(f"Error converting balance data for {currency}: {e}. Balance Info: {balance_info}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error fetching balance for {currency}: {e}", exc_info=True)
            raise # Reraise for potential retry
    ```

2.  **Fix Insufficient Kline Data Requirement:**

    *   **Modify Constant:** Change the default value for the long lookback period.

        ```python
        # --- Constants ---
        # ... (other constants) ...

        # Default Strategy/Indicator Parameters (Overridden by config.json)
        # Volumatic Trend Params
        DEFAULT_VT_LENGTH = 40
        DEFAULT_VT_ATR_PERIOD = 200
        # *** FIX APPLIED HERE ***
        # Reduced from 1060. The original value required fetching >1000 candles,
        # which often exceeds single API call limits and caused "Insufficient kline data" warnings.
        # 200 periods should be sufficient for volume percentile ranking in most cases.
        # Adjust this value in config.json if needed.
        DEFAULT_VT_VOL_EMA_LENGTH = 200
        DEFAULT_VT_ATR_MULTIPLIER = 3.0
        DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0 # Used for visualization only
        # ... (rest of constants) ...
        ```

    *   **Modify Pydantic Default:** Ensure the Pydantic model `StrategyParams` also uses the new default.

        ```python
        class StrategyParams(BaseModel):
            """Strategy-specific parameters."""
            vt_length: int = Field(DEFAULT_VT_LENGTH, gt=0, description="Length for Volumatic Trend Moving Average")
            vt_atr_period: int = Field(DEFAULT_VT_ATR_PERIOD, gt=0, description="ATR period for Volumatic Trend bands")
            # *** FIX APPLIED HERE *** Ensure Pydantic default matches the constant
            vt_vol_ema_length: int = Field(DEFAULT_VT_VOL_EMA_LENGTH, gt=0, description="Lookback period for Volume Normalization (Percentile Rank Window)")
            vt_atr_multiplier: float = Field(DEFAULT_VT_ATR_MULTIPLIER, gt=0, description="ATR multiplier for Volumatic Trend upper/lower bands")
            # ... (rest of StrategyParams fields) ...
        ```

**Explanation of Fixes:**

1.  **Balance Fetch:** The `fetch_balance` function was modified to pass `{'accountType': 'UNIFIED'}` in the `params` dictionary when calling `exchange.fetch_balance()`. This directly addresses the `retCode: 10001` error from the Bybit API, which requires this specific account type for your API key. Added debug logging for the raw balance structure and more specific error handling for the 10001 code in case it persists.
2.  **Kline Data Requirement:** The core issue causing the `Insufficient kline data` warning was the `DEFAULT_VT_VOL_EMA_LENGTH` being set to `1060`. This forced the strategy to require 1110 candles (`1060 + 50` buffer), while the bot could only fetch 1000 in a single API call. By reducing this default to `200` (both in the constant and the Pydantic model default), the `min_data_len` required by the strategy will decrease significantly (likely to around `200 + 50 = 250` or slightly more depending on other lookbacks), which is well within the 1000 candles fetched per cycle. This allows the strategy analysis to proceed.

After applying these fixes, the bot should:

1.  Successfully fetch the initial balance (assuming the API key is indeed for a Bybit Unified Trading Account).
2.  Fetch 1000 klines, which will now be *sufficient* for the strategy calculations because the maximum lookback period has been reduced.
3.  Proceed with the strategy analysis in each cycle instead of skipping due to insufficient data.
