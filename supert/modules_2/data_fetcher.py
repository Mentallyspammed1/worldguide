# File: data_fetcher.py
import traceback
from typing import Any

# Third-party Libraries
try:
    import ccxt
    import pandas as pd
    from colorama import Fore, Style
except ImportError:
    class DummyCCXTExchange: pass
    class DummyCCXT:
        Exchange = DummyCCXTExchange
        NetworkError = Exception
        ExchangeError = Exception
    ccxt = DummyCCXT() # type: ignore[assignment]
    pd = None # type: ignore[assignment] # pandas is crucial here
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Style = DummyColor(), DummyColor()


# Custom module imports
from logger_setup import logger


def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Fetches and prepares OHLCV data, ensuring numeric types."""
    if pd is None: # Check if pandas failed to import
        logger.critical("Pandas library is not available. Cannot fetch market data.")
        return None
        
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}Data Fetch: Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None
    try:
        logger.debug(f"Data Fetch: Fetching {limit} OHLCV candles for {symbol} ({interval})...")
        ohlcv: list[list[int | float | str]] = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: No OHLCV data returned for {symbol} ({interval}).{Style.RESET_ALL}"
            )
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            logger.warning(
                f"{Fore.YELLOW}Data Fetch: OHLCV contains NaNs after conversion:\n{nan_counts[nan_counts > 0]}\nAttempting ffill...{Style.RESET_ALL}"
            )
            df.ffill(inplace=True)
            if df.isnull().values.any():
                logger.warning(f"{Fore.YELLOW}NaNs remain after ffill, attempting bfill...{Style.RESET_ALL}")
                df.bfill(inplace=True)
                if df.isnull().values.any():
                    logger.error(
                        f"{Fore.RED}Data Fetch: NaNs persist after ffill/bfill. Cannot proceed.{Style.RESET_ALL}"
                    )
                    return None
        logger.debug(f"Data Fetch: Processed {len(df)} OHLCV candles for {symbol}.")
        return df

    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.warning(
            f"{Fore.YELLOW}Data Fetch: Error fetching OHLCV for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return None

# End of data_fetcher.py
```

```python
