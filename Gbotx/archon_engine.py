# archon_engine.py
# The Grand Ritual of Vector & Gemini: The Archon Edition
# A masterfully crafted spell for high-efficiency backtesting and analysis.

import os
import time
import argparse
from datetime import datetime
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from colorama import Fore, Style, init
import google.generativeai as genai

# Channeling the arcane energies of the terminal's glow
init(autoreset=True)

# --- The Scroll of Strategy (Centralized Configuration) ---
STRATEGY_CONFIG = {
    "rsi_period": 14, "adx_period": 14, "aroon_period": 14,
    "bb_period": 20, "bb_std": 2.0,
    "supertrend_period": 10, "supertrend_multiplier": 3.0,
    "z_score_period": 20,
    "thresholds": {
        "adx_strong_trend": 25, "aroon_strong_momentum": 50,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "z_score_overextended": 2.0,
        "signal_confidence": 0.4
    }
}

# --- The Chromatic Palette of the Spell ---
COLOR_HEADER = Fore.CYAN + Style.BRIGHT
COLOR_SIGNAL_LONG = Fore.GREEN + Style.BRIGHT
COLOR_SIGNAL_SHORT = Fore.RED + Style.BRIGHT
COLOR_SIGNAL_HOLD = Fore.YELLOW
COLOR_CONFIDENCE = Fore.MAGENTA + Style.BRIGHT
COLOR_REASON_POS = Fore.GREEN
COLOR_REASON_NEG = Fore.RED
COLOR_STATUS = Fore.BLUE + Style.BRIGHT
COLOR_ORACLE = Fore.LIGHTYELLOW_EX
COLOR_ERROR = Fore.RED + Style.BRIGHT

# --- DataManager: The Keeper of Scrolls and Ethers ---
class DataManager:
    def __init__(self):
        print(f"{COLOR_HEADER}DataManager: Forged to manage data from all sources.{Style.RESET_ALL}")

    def load_from_scroll(self, file_path: str) -> pd.DataFrame:
        try:
            print(f"{Fore.WHITE}Unfurling the scroll: {file_path}...{Style.RESET_ALL}")
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            print(f"{COLOR_ERROR}A flaw was found in the scroll's inscription: {e}{Style.RESET_ALL}")
            return pd.DataFrame()

    def weave_indicators(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        print(f"{Fore.WHITE}Weaving the indicator sigils onto the data scroll...{Style.RESET_ALL}")
        df.ta.rsi(length=config["rsi_period"], append=True)
        df.ta.adx(length=config["adx_period"], append=True)
        df.ta.aroon(length=config["aroon_period"], append=True)
        df.ta.bbands(length=config["bb_period"], std=config["bb_std"], append=True)
        df.ta.supertrend(length=config["supertrend_period"], multiplier=config["supertrend_multiplier"], append=True)
        # Custom Z-Score calculation
        df['Z_SCORE'] = (df['close'] - df['close'].rolling(window=config["z_score_period"]).mean()) / df['close'].rolling(window=config["z_score_period"]).std()
        return df.dropna()

# --- GeminiOracle: The Voice of Synthetic Wisdom ---
class GeminiOracle:
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            print(f"{COLOR_ORACLE}Gemini Oracle: Consciousness awakened.{Style.RESET_ALL}")
        except Exception as e:
            self.model = None
            print(f"{COLOR_ERROR}Failed to awaken the Gemini Oracle: {e}{Style.RESET_ALL}")

    def summarize_backtest(self, results_df: pd.DataFrame, symbol: str, config: dict) -> str:
        if not self.model: return "The Oracle is silent."
        
        trade_signals = results_df[results_df['signal'] != 'HOLD']
        num_longs = len(trade_signals[trade_signals['signal'] == 'LONG'])
        num_shorts = len(trade_signals[trade_signals['signal'] == 'SHORT'])
        avg_confidence = trade_signals['confidence'].mean()

        prompt = f"""
        You are a master strategist, reviewing a completed trading simulation.
        Analyze the following backtest summary and provide a concise, expert narrative (3-4 sentences) on the strategy's overall behavior.

        Simulation Summary:
        - Asset: {symbol}
        - Total Candles Analyzed: {len(results_df)}
        - Total Trade Signals Generated: {len(trade_signals)}
        - Long Signals: {num_longs}
        - Short Signals: {num_shorts}
        - Average Signal Confidence: {avg_confidence:.2f}

        Strategy Parameters:
        - RSI Period: {config['rsi_period']}, Overbought: {config['thresholds']['rsi_overbought']}, Oversold: {config['thresholds']['rsi_oversold']}
        - ADX Period: {config['adx_period']}, Strong Trend Threshold: {config['thresholds']['adx_strong_trend']}
        - Supertrend: {config['supertrend_period']}/{config['supertrend_multiplier']}

        Provide your strategic overview.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"The Oracle's thoughts are clouded: {e}"

# --- VectorEngine: The Heart of Quantitative Analysis ---
class VectorEngine:
    def __init__(self, config: dict):
        self.config = config
        self.t = config["thresholds"]

    def analyze_row(self, primary_row: pd.Series, htf_dir: int, symbol: str) -> dict:
        reasoning, score, signal = [], 0, "HOLD"
        
        # 1. Multi-Timeframe Context
        htf_trend = "UP" if htf_dir == 1 else "DOWN"
        reasoning.append(f"{COLOR_REASON_POS}[+] CONTEXT: HTF current is {htf_trend}.")
        if htf_dir == 1: score, signal = 0.2, "LONG"
        else: score, signal = 0.2, "SHORT"

        # 2. Indicator Confluence & Divergence
        if primary_row[f'ADX_{self.config["adx_period"]}'] > self.t['adx_strong_trend']:
            reasoning.append(f"{COLOR_REASON_POS}[+] TREND: ADX shows a strong current.")
            score += 0.15
        
        if primary_row[f'AROONOSC_{self.config["aroon_period"]}'] > self.t['aroon_strong_momentum']:
            reasoning.append(f"{COLOR_REASON_POS}[+] MOMENTUM: Aroon confirms bullish surge.")
            score += 0.1
        elif primary_row[f'AROONOSC_{self.config["aroon_period"]}'] < -self.t['aroon_strong_momentum']:
            reasoning.append(f"{COLOR_REASON_POS}[+] MOMENTUM: Aroon confirms bearish plunge.")
            score += 0.1

        # 3. Risk Analysis & Signal Invalidation
        if primary_row[f'RSI_{self.config["rsi_period"]}'] > self.t['rsi_overbought']:
            reasoning.append(f"{COLOR_REASON_NEG}[-] RISK: RSI is overbought.")
            if signal == "LONG": signal, score = "HOLD", score - 0.3
        elif primary_row[f'RSI_{self.config["rsi_period"]}'] < self.t['rsi_oversold']:
            reasoning.append(f"{COLOR_REASON_NEG}[-] RISK: RSI is oversold.")
            if signal == "SHORT": signal, score = "HOLD", score - 0.3

        if abs(primary_row['Z_SCORE']) > self.t['z_score_overextended']:
            reasoning.append(f"{COLOR_REASON_NEG}[-] RISK: Z-Score shows extreme deviation.")
            if (primary_row['Z_SCORE'] > 0 and signal == "LONG") or (primary_row['Z_SCORE'] < 0 and signal == "SHORT"):
                signal, score = "HOLD", score - 0.2

        st_dir_col = f'SUPERTd_{self.config["supertrend_period"]}_{self.config["supertrend_multiplier"]}'
        if (signal == "LONG" and primary_row[st_dir_col] == -1) or (signal == "SHORT" and primary_row[st_dir_col] == 1):
            reasoning.append(f"{COLOR_REASON_NEG}[-] CONFLICT: Primary Supertrend opposes HTF bias.")
            signal, score = "HOLD", score - 0.3

        # 4. Final Verdict
        final_signal = signal if score >= self.t['signal_confidence'] else "HOLD"
        return {"signal": final_signal, "confidence": round(max(0, score), 2), "reasoning": " | ".join(r.split(']', 1)[-1].strip() for r in reasoning)}

# --- Chronomancer: The Master of Time Simulation ---
class Chronomancer:
    def __init__(self, engine: VectorEngine, verbose: bool):
        self.engine = engine
        self.verbose = verbose
        self.results = []

    def scry(self, df_primary: pd.DataFrame, df_htf: pd.DataFrame, symbol: str):
        print(f"{COLOR_HEADER}Chronomancer begins scrying the threads of past time...{Style.RESET_ALL}")
        
        # Align HTF data to primary timestamps for efficient lookup
        df_primary['htf_dir'] = df_htf[f'SUPERTd_{STRATEGY_CONFIG["supertrend_period"]}_{STRATEGY_CONFIG["supertrend_multiplier"]}'].reindex(df_primary.index, method='ffill')
        df_primary.dropna(inplace=True)

        for timestamp, row in df_primary.iterrows():
            analysis = self.engine.analyze_row(row, row['htf_dir'], symbol)
            self.results.append({
                "timestamp": timestamp, "close": row['close'],
                "signal": analysis['signal'], "confidence": analysis['confidence'],
                "reasoning": analysis['reasoning']
            })
            if self.verbose or analysis['signal'] != 'HOLD':
                self.print_analysis(self.results[-1])
        
        print(f"{COLOR_HEADER}Scrying complete. {len(self.results)} moments in time were analyzed.{Style.RESET_ALL}")

    def save_ledger(self, filename: str):
        if not self.results: return
        print(f"{Fore.WHITE}Inscribing the Ledger of Fates to {filename}...{Style.RESET_ALL}")
        pd.DataFrame(self.results).to_csv(filename, index=False)
        print(f"{Fore.GREEN}The Ledger has been written.{Style.RESET_ALL}")

    def print_analysis(self, result: dict):
        signal_color = {"LONG": COLOR_SIGNAL_LONG, "SHORT": COLOR_SIGNAL_SHORT, "HOLD": COLOR_SIGNAL_HOLD}.get(result['signal'])
        print(f"{result['timestamp']} | {COLOR_HEADER}CLOSE: {result['close']:.4f} | {signal_color}SIG: {result['signal']:<4} {COLOR_CONFIDENCE}({result['confidence']:.2f}){Style.RESET_ALL}")

# --- The Main Summoning Ritual ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archon Engine: Analyze historical market data from scrolls.")
    parser.add_argument("primary_scroll", help="Path to the primary timeframe CSV scroll.")
    parser.add_argument("htf_scroll", help="Path to the higher timeframe CSV scroll.")
    parser.add_argument("--symbol", default="HISTORICAL_ASSET", help="Name of the asset being analyzed.")
    parser.add_argument("-o", "--output", help="Path to save the Ledger of Fates (results CSV).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print analysis for every candle, not just signals.")
    parser.add_argument("--gemini-summary", action="store_true", help="Invoke the Gemini Oracle for a post-run summary.")
    args = parser.parse_args()

    # Awaken Gemini Oracle
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print(f"{COLOR_ERROR}Gemini API key is missing. The Oracle cannot be awakened.")
        gemini_oracle = GeminiOracle(api_key="dummy")
    else:
        gemini_oracle = GeminiOracle(api_key=GEMINI_API_KEY)

    # Initialize the conduit and Vector
    dm = DataManager()
    engine = VectorEngine(STRATEGY_CONFIG)
    chronomancer = Chronomancer(engine, args.verbose)

    # --- Data Preparation ---
    df_primary = dm.load_from_scroll(args.primary_scroll)
    df_htf = dm.load_from_scroll(args.htf_scroll)
    if df_primary.empty or df_htf.empty:
        print(f"{COLOR_ERROR}Cannot perform chronomancy without valid scrolls. The spell is broken.{Style.RESET_ALL}")
        exit()

    df_primary = dm.weave_indicators(df_primary, STRATEGY_CONFIG)
    df_htf = dm.weave_indicators(df_htf, STRATEGY_CONFIG)

    # --- Execution ---
    chronomancer.scry(df_primary, df_htf, args.symbol)

    if args.output:
        chronomancer.save_ledger(args.output)

    if args.gemini_summary:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            print(f"{COLOR_ERROR}Cannot invoke the Oracle: Gemini API key is missing from .env scroll.{Style.RESET_ALL}")
        else:
            oracle = GeminiOracle(GEMINI_API_KEY)
            print(f"\n{COLOR_ORACLE}Invoking the Gemini Oracle for a strategic summary...{Style.RESET_ALL}")
            summary = oracle.summarize_backtest(pd.DataFrame(chronomancer.results), args.symbol, STRATEGY_CONFIG)
            print(f"{Fore.WHITE}{summary}{Style.RESET_ALL}")