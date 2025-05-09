# File: strategy.py
import traceback
from typing import Any

# Third-party Libraries
try:
    import pandas as pd
    from colorama import Fore, Style
except ImportError:
    pd = None # type: ignore[assignment]
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Style = DummyColor(), DummyColor()

# Custom module imports
from logger_setup import logger
from config import CONFIG


def generate_signals(df: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    """Generates entry/exit signals based on the selected strategy."""
    if pd is None:
        logger.critical("Pandas library is not available. Cannot generate signals.")
        return {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Pandas Error"}

    signals = {"enter_long": False, "enter_short": False, "exit_long": False, "exit_short": False, "exit_reason": "Strategy Exit"}
    if len(df) < 2: return signals

    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        if strategy_name == "DUAL_SUPERTREND":
            if ("st_long" in last and pd.notna(last["st_long"]) and last["st_long"] and
                "confirm_trend" in last and pd.notna(last["confirm_trend"]) and last["confirm_trend"]):
                signals["enter_long"] = True
            if ("st_short" in last and pd.notna(last["st_short"]) and last["st_short"] and
                "confirm_trend" in last and pd.notna(last["confirm_trend"]) and not last["confirm_trend"]):
                signals["enter_short"] = True
            if "st_short" in last and pd.notna(last["st_short"]) and last["st_short"]:
                signals["exit_long"] = True; signals["exit_reason"] = "Primary ST Short Flip"
            if "st_long" in last and pd.notna(last["st_long"]) and last["st_long"]:
                signals["exit_short"] = True; signals["exit_reason"] = "Primary ST Long Flip"

        elif strategy_name == "STOCHRSI_MOMENTUM":
            req_cols = ["stochrsi_k", "stochrsi_d", "momentum"]
            if not all(col in last and pd.notna(last[col]) for col in req_cols) or \
               not all(col in prev and pd.notna(prev[col]) for col in ["stochrsi_k", "stochrsi_d"]):
                logger.debug("StochRSI/Mom signals skipped due to NA values."); return signals
            k_n, d_n, m_n = last["stochrsi_k"], last["stochrsi_d"], last["momentum"]
            k_p, d_p = prev["stochrsi_k"], prev["stochrsi_d"]
            if k_p <= d_p and k_n > d_n and k_n < CONFIG.stochrsi_oversold and m_n > CONFIG.position_qty_epsilon:
                signals["enter_long"] = True
            if k_p >= d_p and k_n < d_n and k_n > CONFIG.stochrsi_overbought and m_n < -CONFIG.position_qty_epsilon:
                signals["enter_short"] = True
            if k_p >= d_p and k_n < d_n: signals["exit_long"] = True; signals["exit_reason"] = "StochRSI K below D"
            if k_p <= d_p and k_n > d_n: signals["exit_short"] = True; signals["exit_reason"] = "StochRSI K above D"

        elif strategy_name == "EHLERS_FISHER":
            req_cols = ["ehlers_fisher", "ehlers_signal"]
            if not all(col in last and pd.notna(last[col]) for col in req_cols) or \
               not all(col in prev and pd.notna(prev[col]) for col in req_cols):
                logger.debug("Ehlers Fisher signals skipped due to NA values."); return signals
            f_n, s_n = last["ehlers_fisher"], last["ehlers_signal"]
            f_p, s_p = prev["ehlers_fisher"], prev["ehlers_signal"]
            if f_p <= s_p and f_n > s_n: signals["enter_long"] = True
            if f_p >= s_p and f_n < s_n: signals["enter_short"] = True
            if f_p >= s_p and f_n < s_n: signals["exit_long"] = True; signals["exit_reason"] = "Ehlers Fisher Short Cross"
            if f_p <= s_p and f_n > s_n: signals["exit_short"] = True; signals["exit_reason"] = "Ehlers Fisher Long Cross"

        elif strategy_name == "EHLERS_MA_CROSS":
            req_cols = ["fast_ema", "slow_ema"]
            if not all(col in last and pd.notna(last[col]) for col in req_cols) or \
               not all(col in prev and pd.notna(prev[col]) for col in req_cols):
                logger.debug("Ehlers MA signals skipped due to NA values."); return signals
            fm_n, sm_n = last["fast_ema"], last["slow_ema"]
            fm_p, sm_p = prev["fast_ema"], prev["slow_ema"]
            if fm_p <= sm_p and fm_n > sm_n: signals["enter_long"] = True
            if fm_p >= sm_p and fm_n < sm_n: signals["enter_short"] = True
            if fm_p >= sm_p and fm_n < sm_n: signals["exit_long"] = True; signals["exit_reason"] = "Ehlers MA Short Cross"
            if fm_p <= sm_p and fm_n > sm_n: signals["exit_short"] = True; signals["exit_reason"] = "Ehlers MA Long Cross"

    except KeyError as e: logger.error(f"{Fore.RED}Signal Generation Error: Missing column: {e}{Style.RESET_ALL}")
    except Exception as e: logger.error(f"{Fore.RED}Signal Generation Error: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
    return signals

# End of strategy.py
```

```python
