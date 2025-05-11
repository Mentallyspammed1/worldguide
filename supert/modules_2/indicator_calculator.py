# File: indicator_calculator.py
import traceback
from decimal import Decimal
from typing import Any

# Third-party Libraries
try:
    import pandas as pd
    import pandas_ta as ta # type: ignore[import]
    import ccxt
    from colorama import Fore, Style
except ImportError:
    pd = None # type: ignore[assignment]
    ta = None # type: ignore[assignment]
    class DummyCCXTExchange: pass
    class DummyCCXT:
        Exchange = DummyCCXTExchange
        NetworkError = Exception
        ExchangeError = Exception
    ccxt = DummyCCXT() # type: ignore[assignment]
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Style = DummyColor(), DummyColor()


# Custom module imports
from logger_setup import logger
from config import CONFIG
from utils import safe_decimal_conversion


def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: Decimal, prefix: str = "") -> pd.DataFrame:
    """Calculates the Supertrend indicator using pandas_ta, returns Decimal where applicable."""
    if pd is None or ta is None:
        logger.critical("Pandas or Pandas_TA library not available for Supertrend calculation.")
        return df # Return original df or handle error appropriately

    col_prefix = f"{prefix}" if prefix else ""
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]
    st_col = f"SUPERT_{length}_{float(multiplier)}"
    st_trend_col = f"SUPERTd_{length}_{float(multiplier)}"
    st_long_col = f"SUPERTl_{length}_{float(multiplier)}"
    st_short_col = f"SUPERTs_{length}_{float(multiplier)}"
    required_input_cols = ["high", "low", "close"]

    if df is None or df.empty or not all(c in df.columns for c in required_input_cols) or len(df) < length + 1:
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc ({col_prefix}ST): Invalid input (Len: {len(df) if df is not None else 0}, Need: {length + 1}).{Style.RESET_ALL}"
        )
        for col in target_cols:
            df[col] = pd.NA
        return df
    try:
        df.ta.supertrend(length=length, multiplier=float(multiplier), append=True)
        if st_col not in df.columns or st_trend_col not in df.columns:
            raise KeyError(f"pandas_ta failed to create expected raw columns: {st_col}, {st_trend_col}")

        df[f"{col_prefix}supertrend"] = df[st_col].apply(safe_decimal_conversion)
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1
        prev_trend = df[st_trend_col].shift(1)
        df[f"{col_prefix}st_long"] = (prev_trend == -1) & (df[st_trend_col] == 1)
        df[f"{col_prefix}st_short"] = (prev_trend == 1) & (df[st_trend_col] == -1)

        raw_st_cols = [st_col, st_trend_col, st_long_col, st_short_col]
        df.drop(columns=raw_st_cols, errors="ignore", inplace=True)

        last_st_val = df[f"{col_prefix}supertrend"].iloc[-1]
        if pd.notna(last_st_val):
            last_trend = "Up" if df[f"{col_prefix}trend"].iloc[-1] else "Down"
            signal = (
                "LONG" if df[f"{col_prefix}st_long"].iloc[-1]
                else ("SHORT" if df[f"{col_prefix}st_short"].iloc[-1] else "None")
            )
            logger.debug(
                f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Trend={last_trend}, Val={last_st_val:.4f}, Signal={signal}"
            )
        else:
            logger.debug(f"Indicator Calc ({col_prefix}ST({length},{multiplier})): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc ({col_prefix}ST): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA
    return df


def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, Decimal | None]:
    """Calculates ATR, Volume MA, checks spikes. Returns Decimals."""
    if pd is None or ta is None:
        logger.critical("Pandas or Pandas_TA library not available for Vol/ATR analysis.")
        return {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}

    results: dict[str, Decimal | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    required_cols = ["high", "low", "close", "volume"]
    min_len = max(atr_len, vol_ma_len)

    if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc (Vol/ATR): Invalid input (Len: {len(df) if df is not None else 0}, Need: {min_len}).{Style.RESET_ALL}"
        )
        return results
    try:
        atr_col = f"ATRr_{atr_len}"
        df.ta.atr(length=atr_len, append=True)
        if atr_col in df.columns:
            last_atr = df[atr_col].iloc[-1]
            if pd.notna(last_atr): results["atr"] = safe_decimal_conversion(last_atr)
            df.drop(columns=[atr_col], errors="ignore", inplace=True)

        volume_ma_col = "volume_ma"
        df[volume_ma_col] = df["volume"].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
        last_vol_ma = df[volume_ma_col].iloc[-1]
        last_vol = df["volume"].iloc[-1]

        if pd.notna(last_vol_ma): results["volume_ma"] = safe_decimal_conversion(last_vol_ma)
        if pd.notna(last_vol): results["last_volume"] = safe_decimal_conversion(last_vol)

        if (results["volume_ma"] is not None and results["volume_ma"] > CONFIG.position_qty_epsilon and results["last_volume"] is not None):
            try: results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            except Exception: results["volume_ratio"] = None
        if volume_ma_col in df.columns: df.drop(columns=[volume_ma_col], errors="ignore", inplace=True)

        atr_str = f"{results['atr']:.5f}" if results["atr"] else "N/A"
        vol_ma_str = f"{results['volume_ma']:.2f}" if results["volume_ma"] else "N/A"
        vol_ratio_str = f"{results['volume_ratio']:.2f}" if results["volume_ratio"] else "N/A"
        logger.debug(
            f"Indicator Calc: ATR({atr_len})={atr_str}, VolMA({vol_ma_len})={vol_ma_str}, VolRatio={vol_ratio_str}"
        )
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (Vol/ATR): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results, None) # type: ignore[assignment]
    return results


def calculate_stochrsi_momentum(
    df: pd.DataFrame, rsi_len: int, stoch_len: int, k: int, d: int, mom_len: int
) -> pd.DataFrame:
    """Calculates StochRSI and Momentum, returns Decimals."""
    if pd is None or ta is None:
        logger.critical("Pandas or Pandas_TA library not available for StochRSI/Momentum.")
        return df

    target_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
    min_len = max(rsi_len + stoch_len, mom_len) + 5
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc (StochRSI/Mom): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"
        )
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        stochrsi_df = df.ta.stochrsi(length=stoch_len, rsi_length=rsi_len, k=k, d=d, append=False)
        k_col, d_col = f"STOCHRSIk_{stoch_len}_{rsi_len}_{k}_{d}", f"STOCHRSId_{stoch_len}_{rsi_len}_{k}_{d}"
        if k_col in stochrsi_df.columns: df["stochrsi_k"] = stochrsi_df[k_col].apply(safe_decimal_conversion)
        else: logger.warning("StochRSI K column not found"); df["stochrsi_k"] = pd.NA
        if d_col in stochrsi_df.columns: df["stochrsi_d"] = stochrsi_df[d_col].apply(safe_decimal_conversion)
        else: logger.warning("StochRSI D column not found"); df["stochrsi_d"] = pd.NA

        mom_col = f"MOM_{mom_len}"
        df.ta.mom(length=mom_len, append=True)
        if mom_col in df.columns:
            df["momentum"] = df[mom_col].apply(safe_decimal_conversion)
            df.drop(columns=[mom_col], errors="ignore", inplace=True)
        else: logger.warning("Momentum column not found"); df["momentum"] = pd.NA

        k_val, d_val, mom_val = df["stochrsi_k"].iloc[-1], df["stochrsi_d"].iloc[-1], df["momentum"].iloc[-1]
        if pd.notna(k_val) and pd.notna(d_val) and pd.notna(mom_val):
            logger.debug(f"Indicator Calc (StochRSI/Mom): K={k_val:.2f}, D={d_val:.2f}, Mom={mom_val:.4f}")
        else: logger.debug("Indicator Calc (StochRSI/Mom): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (StochRSI/Mom): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def calculate_ehlers_fisher(df: pd.DataFrame, length: int, signal: int) -> pd.DataFrame:
    """Calculates Ehlers Fisher Transform, returns Decimals."""
    if pd is None or ta is None:
        logger.critical("Pandas or Pandas_TA library not available for Ehlers Fisher.")
        return df
        
    target_cols = ["ehlers_fisher", "ehlers_signal"]
    if df is None or df.empty or not all(c in df.columns for c in ["high", "low"]) or len(df) < length + 1:
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc (EhlersFisher): Invalid input (Len: {len(df) if df is not None else 0}, Need {length + 1}).{Style.RESET_ALL}"
        )
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        fisher_df = df.ta.fisher(length=length, signal=signal, append=False)
        fish_col, signal_col = f"FISHERT_{length}_{signal}", f"FISHERTs_{length}_{signal}"
        if fish_col in fisher_df.columns: df["ehlers_fisher"] = fisher_df[fish_col].apply(safe_decimal_conversion)
        else: logger.warning("Ehlers Fisher column not found"); df["ehlers_fisher"] = pd.NA
        if signal_col in fisher_df.columns: df["ehlers_signal"] = fisher_df[signal_col].apply(safe_decimal_conversion)
        else: logger.warning("Ehlers Signal column not found"); df["ehlers_signal"] = pd.NA

        fish_val, sig_val = df["ehlers_fisher"].iloc[-1], df["ehlers_signal"].iloc[-1]
        if pd.notna(fish_val) and pd.notna(sig_val):
            logger.debug(
                f"Indicator Calc (EhlersFisher({length},{signal})): Fisher={fish_val:.4f}, Signal={sig_val:.4f}"
            )
        else: logger.debug("Indicator Calc (EhlersFisher): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersFisher): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def calculate_ehlers_ma(df: pd.DataFrame, fast_len: int, slow_len: int) -> pd.DataFrame:
    """Calculates Ehlers Super Smoother Moving Averages (using EMA as placeholder), returns Decimals."""
    if pd is None or ta is None:
        logger.critical("Pandas or Pandas_TA library not available for Ehlers MA.")
        return df

    target_cols = ["fast_ema", "slow_ema"]
    min_len = max(fast_len, slow_len) + 5
    if df is None or df.empty or not all(c in df.columns for c in ["close"]) or len(df) < min_len:
        logger.warning(
            f"{Fore.YELLOW}Indicator Calc (EhlersMA): Invalid input (Len: {len(df) if df is not None else 0}, Need ~{min_len}).{Style.RESET_ALL}"
        )
        for col in target_cols: df[col] = pd.NA
        return df
    try:
        logger.warning(
            f"{Fore.YELLOW}Using EMA as placeholder for Ehlers Super Smoother. Replace with actual implementation if needed.{Style.RESET_ALL}"
        )
        df["fast_ema"] = df.ta.ema(length=fast_len).apply(safe_decimal_conversion)
        df["slow_ema"] = df.ta.ema(length=slow_len).apply(safe_decimal_conversion)

        fast_val, slow_val = df["fast_ema"].iloc[-1], df["slow_ema"].iloc[-1]
        if pd.notna(fast_val) and pd.notna(slow_val):
            logger.debug(f"Indicator Calc (EhlersMA({fast_len},{slow_len})): Fast={fast_val:.4f}, Slow={slow_val:.4f}")
        else: logger.debug("Indicator Calc (EhlersMA): Resulted in NA for last candle.")
    except Exception as e:
        logger.error(f"{Fore.RED}Indicator Calc (EhlersMA): Error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        for col in target_cols: df[col] = pd.NA
    return df


def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int, fetch_limit: int) -> dict[str, Decimal | None]:
    """Fetches and analyzes L2 order book pressure and spread. Returns Decimals."""
    results: dict[str, Decimal | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    logger.debug(f"Order Book: Fetching L2 {symbol} (Depth:{depth}, Limit:{fetch_limit})...")
    if not exchange.has.get("fetchL2OrderBook"):
        logger.warning(f"{Fore.YELLOW}fetchL2OrderBook not supported by {exchange.id}.{Style.RESET_ALL}")
        return results
    try:
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        bids: list[list[float | str]] = order_book.get("bids", [])
        asks: list[list[float | str]] = order_book.get("asks", [])

        best_bid = safe_decimal_conversion(bids[0][0]) if bids and len(bids[0]) > 0 else None
        best_ask = safe_decimal_conversion(asks[0][0]) if asks and len(asks[0]) > 0 else None
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask

        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            results["spread"] = best_ask - best_bid
            logger.debug(f"OB: Bid={best_bid:.4f}, Ask={best_ask:.4f}, Spread={results['spread']:.4f}")
        else: logger.debug(f"OB: Bid={best_bid or 'N/A'}, Ask={best_ask or 'N/A'} (Spread N/A)")

        bid_vol = sum(safe_decimal_conversion(bid[1]) for bid in bids[:depth] if len(bid) > 1)
        ask_vol = sum(safe_decimal_conversion(ask[1]) for ask in asks[:depth] if len(ask) > 1)
        logger.debug(f"OB (Depth {depth}): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}")

        if ask_vol > CONFIG.position_qty_epsilon:
            try:
                results["bid_ask_ratio"] = bid_vol / ask_vol
                logger.debug(f"OB Ratio: {results['bid_ask_ratio']:.3f}")
            except Exception: logger.warning("Error calculating OB ratio."); results["bid_ask_ratio"] = None
        else: logger.debug("OB Ratio: N/A (Ask volume zero or negligible)")
    except (ccxt.NetworkError, ccxt.ExchangeError, IndexError, Exception) as e:
        logger.warning(f"{Fore.YELLOW}Order Book Error for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        results = dict.fromkeys(results, None) # type: ignore[assignment]
    return results

# End of indicator_calculator.py
```

```python
