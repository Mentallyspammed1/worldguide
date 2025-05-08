# File: indicator_analyzer.py
import pandas as pd
import pandas_ta as ta
import logging
import math # For math.isfinite
from decimal import Decimal, getcontext, InvalidOperation, ROUND_HALF_UP
from typing import Dict, Any, Optional, Tuple

# Assuming app_config.py and trading_enums.py are in the same directory
from app_config import CALCULATION_PRECISION
from trading_enums import Signal

class TradingAnalyzer:
    """Analyzes market data using technical indicators to generate trading signals."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.indicator_config = config.get("indicator_settings", {})
        self.weights = self.indicator_config.get("signal_weights", {})
        self._initialize_weights()

    def _initialize_weights(self):
        if not self.weights:
            self.logger.warning("No 'signal_weights' in config. Using default weights.")
            self.weights = { "rsi": Decimal("0.3"), "macd": Decimal("0.4"), "ema_cross": Decimal("0.3") }
        else:
            valid_weights = {}
            for k, v in self.weights.items():
                if isinstance(v, Decimal) and v.is_finite() and v >= 0: valid_weights[k] = v
                else: self.logger.warning(f"Invalid signal weight for '{k}' ({v}). Removing.")
            self.weights = valid_weights
        if not self.weights: self.logger.error("No valid signal weights. Cannot generate weighted signals."); return
        
        total_weight = sum(self.weights.values())
        if total_weight <= Decimal('1e-18'): self.logger.error("Total signal weight is zero. Disabling weighted signals."); self.weights = {}; return
        elif abs(total_weight - Decimal("1.0")) > Decimal("1e-9"):
             self.logger.warning(f"Signal weights sum to {total_weight}, not 1. Normalizing.")
             try:
                  with getcontext() as ctx:
                       ctx.prec = CALCULATION_PRECISION
                       self.weights = {k: (v / total_weight).quantize(Decimal("1e-6"), rounding=ROUND_HALF_UP) for k, v in self.weights.items()}
                  self.logger.info(f"Normalized weights: {self.weights}")
             except InvalidOperation as e: self.logger.error(f"Error normalizing weights: {e}. Disabling weighted signals."); self.weights = {}

    def calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if ohlcv_df is None or ohlcv_df.empty: self.logger.error("OHLCV data missing/empty."); return None
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in ohlcv_df.columns for col in required_cols): self.logger.error(f"Missing OHLCV columns: {required_cols}"); return None
        try:
             if not all(isinstance(ohlcv_df[col].iloc[0], Decimal) for col in required_cols if not ohlcv_df[col].empty):
                  self.logger.error("OHLCV columns not all Decimal type."); return None
        except IndexError: self.logger.error("OHLCV DataFrame empty/malformed."); return None
        
        self.logger.debug(f"Calculating indicators for {len(ohlcv_df)} candles...")
        df = ohlcv_df.copy()
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        df_float = pd.DataFrame(index=df.index)
        conversion_failed = False
        for col in float_cols:
            try:
                df_float[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) and x.is_finite() else pd.NA)
                df_float[col] = pd.to_numeric(df_float[col], errors='coerce')
            except Exception as e: self.logger.error(f"Error converting column {col} to float: {e}", exc_info=True); conversion_failed = True
        if conversion_failed: return None
        
        initial_len = len(df_float)
        df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        if len(df_float) < initial_len: self.logger.warning(f"Dropped {initial_len - len(df_float)} rows with NaN in OHLC columns.")
        if df_float.empty: self.logger.error("DataFrame empty after NaN handling."); return None
        
        try:
            rsi_period = self.indicator_config.get("rsi_period")
            if isinstance(rsi_period, int) and rsi_period > 0: df_float.ta.rsi(length=rsi_period, append=True, col_names=(f"RSI_{rsi_period}"))
            
            macd_fast, macd_slow, macd_signal = self.indicator_config.get("macd_fast"), self.indicator_config.get("macd_slow"), self.indicator_config.get("macd_signal")
            if all(isinstance(p, int) and p > 0 for p in [macd_fast, macd_slow, macd_signal]): df_float.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
            
            ema_short_period = self.indicator_config.get("ema_short_period")
            if isinstance(ema_short_period, int) and ema_short_period > 0: df_float.ta.ema(length=ema_short_period, append=True, col_names=(f"EMA_{ema_short_period}"))
            ema_long_period = self.indicator_config.get("ema_long_period")
            if isinstance(ema_long_period, int) and ema_long_period > 0: df_float.ta.ema(length=ema_long_period, append=True, col_names=(f"EMA_{ema_long_period}"))
            
            atr_period = self.indicator_config.get("atr_period")
            if isinstance(atr_period, int) and atr_period > 0: df_float.ta.atr(length=atr_period, append=True, mamode='rma', col_names=(f"ATR_{atr_period}"))
            
            self.logger.debug(f"Indicators calculated (float). Columns added: {df_float.columns.difference(float_cols).tolist()}")
            indicator_cols = df_float.columns.difference(float_cols).tolist()
            with getcontext() as ctx:
                ctx.prec = CALCULATION_PRECISION
                for col in indicator_cols:
                    df[col] = df_float[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and isinstance(x, (float, int)) and math.isfinite(x) else Decimal('NaN'))
            df = df.reindex(ohlcv_df.index)
            self.logger.debug("Indicators converted back to Decimal and merged.")
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators with pandas_ta: {e}", exc_info=True)
            return None

    def generate_signal(self, indicators_df: pd.DataFrame) -> Tuple[Signal, Dict[str, Any]]:
        if indicators_df is None or indicators_df.empty: self.logger.warning("Cannot generate signal: Indicators data missing."); return Signal.HOLD, {}
        try: latest_data = indicators_df.iloc[-1]
        except IndexError: self.logger.error("Cannot get latest data: Indicators DataFrame empty."); return Signal.HOLD, {}
        
        scores, contributing_factors = {}, {}
        with getcontext() as ctx:
            ctx.prec = CALCULATION_PRECISION
            rsi_key, rsi_weight = "rsi", self.weights.get("rsi", Decimal(0))
            rsi_period, rsi_col = self.indicator_config.get("rsi_period"), (f"RSI_{self.indicator_config.get('rsi_period')}" if isinstance(self.indicator_config.get("rsi_period"), int) else None)
            if rsi_weight > 0 and rsi_col and rsi_col in latest_data:
                rsi_value = latest_data[rsi_col]
                if isinstance(rsi_value, Decimal) and rsi_value.is_finite():
                    overbought, oversold = self.indicator_config.get("rsi_overbought", Decimal("70")), self.indicator_config.get("rsi_oversold", Decimal("30"))
                    rsi_score = Decimal(0)
                    if rsi_value > overbought: rsi_score = Decimal("-1")
                    elif rsi_value < oversold: rsi_score = Decimal("1")
                    scores[rsi_key], contributing_factors[rsi_key] = rsi_score * rsi_weight, {"value": rsi_value, "score": rsi_score, "weight": rsi_weight}
                else: self.logger.debug(f"Invalid RSI value ({rsi_value}). Skipping.")

            macd_key, macd_weight = "macd", self.weights.get("macd", Decimal(0))
            mf, ms, msg = self.indicator_config.get("macd_fast", 12), self.indicator_config.get("macd_slow", 26), self.indicator_config.get("macd_signal", 9)
            macdh_col = f"MACDh_{mf}_{ms}_{msg}" if all(isinstance(p, int) for p in [mf, ms, msg]) else None
            if macd_weight > 0 and macdh_col and macdh_col in latest_data:
                macd_hist = latest_data[macdh_col]
                if isinstance(macd_hist, Decimal) and macd_hist.is_finite():
                    hist_thresh = self.indicator_config.get("macd_hist_threshold", Decimal("0"))
                    macd_score = Decimal(0)
                    if macd_hist > hist_thresh: macd_score = Decimal("1")
                    elif macd_hist < -hist_thresh: macd_score = Decimal("-1")
                    scores[macd_key], contributing_factors[macd_key] = macd_score * macd_weight, {"histogram": macd_hist, "score": macd_score, "weight": macd_weight}
                else: self.logger.debug(f"Invalid MACD Hist value ({macd_hist}). Skipping.")
            
            ema_key, ema_cross_weight = "ema_cross", self.weights.get("ema_cross", Decimal(0))
            ema_s_p, ema_l_p = self.indicator_config.get("ema_short_period"), self.indicator_config.get("ema_long_period")
            ema_s_col, ema_l_col = (f"EMA_{ema_s_p}" if isinstance(ema_s_p, int) else None), (f"EMA_{ema_l_p}" if isinstance(ema_l_p, int) else None)
            if ema_cross_weight > 0 and ema_s_col and ema_l_col and all(c in latest_data for c in [ema_s_col, ema_l_col]):
                ema_short, ema_long = latest_data[ema_s_col], latest_data[ema_l_col]
                if isinstance(ema_short, Decimal) and ema_short.is_finite() and isinstance(ema_long, Decimal) and ema_long.is_finite():
                    ema_cross_score = Decimal(0)
                    if ema_short > ema_long: ema_cross_score = Decimal("1")
                    elif ema_short < ema_long: ema_cross_score = Decimal("-1")
                    scores[ema_key], contributing_factors[ema_key] = ema_cross_score * ema_cross_weight, {"short_ema": ema_short, "long_ema": ema_long, "score": ema_cross_score, "weight": ema_cross_weight}
                else: self.logger.debug(f"Invalid EMA values (S={ema_short}, L={ema_long}). Skipping.")

        final_score = sum(scores.values()) if scores else Decimal(0)
        sb_t, b_t, s_t, ss_t = self.indicator_config.get("strong_buy_threshold", Decimal("0.7")), self.indicator_config.get("buy_threshold", Decimal("0.2")), self.indicator_config.get("sell_threshold", Decimal("-0.2")), self.indicator_config.get("strong_sell_threshold", Decimal("-0.7"))
        
        if not (ss_t <= s_t < b_t <= sb_t):
             self.logger.error("Signal thresholds improperly configured. Defaulting to HOLD.")
             final_signal = Signal.HOLD
        else:
            if final_score >= sb_t: final_signal = Signal.STRONG_BUY
            elif final_score >= b_t: final_signal = Signal.BUY
            elif final_score <= ss_t: final_signal = Signal.STRONG_SELL
            elif final_score <= s_t: final_signal = Signal.SELL
            else: final_signal = Signal.HOLD
        
        quantized_score = final_score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        self.logger.info(f"Signal generated: {final_signal.name} (Score: {quantized_score})")
        self.logger.debug(f"Contributing factors: {contributing_factors}")
        signal_details_out = {"final_score": quantized_score, "factors": contributing_factors}
        return final_signal, signal_details_out

```

```python
