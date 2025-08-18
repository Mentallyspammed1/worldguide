# vector_ai.py
# The Chronomancer's Edition: Binding Vector & Gemini to historical scrolls.

import os
import time
import argparse # The rune for interpreting seeker commands
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from colorama import Fore, Style, init
import google.generativeai as genai

# ... (All constants and color definitions remain the same) ...
# ... (All indicator calculation functions remain the same) ...
# ... (The GeminiOracle class remains the same) ...

# --- Reforging the Conduit: From Live Ether to Etched Scrolls ---
class DataConduit:
    """
    A conduit that reads knowledge from historical scrolls (CSV files)
    instead of the live ether.
    """
    def __init__(self):
        print(f"{COLOR_HEADER}Data Conduit: Forged to read from historical scrolls.{Style.RESET_ALL}")

    def read_scroll(self, file_path: str) -> pd.DataFrame:
        """Reads a CSV file and prepares it for analysis."""
        try:
            print(f"{Fore.WHITE}Unfurling the scroll: {file_path}...{Style.RESET_ALL}")
            df = pd.read_csv(file_path)
            # Ensure the essential columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Scroll is missing required runes. Must contain: {required_cols}")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True) # Ensure time flows forward
            print(f"{Fore.GREEN}Scroll successfully deciphered.{Style.RESET_ALL}")
            return df
        except FileNotFoundError:
            print(f"{COLOR_ERROR}The scroll '{file_path}' could not be found in the archives.{Style.RESET_ALL}")
            return pd.DataFrame()
        except Exception as e:
            print(f"{COLOR_ERROR}A flaw was found in the scroll's inscription: {e}{Style.RESET_ALL}")
            return pd.DataFrame()

# --- Vector's Consciousness: The Core Logic (Adapted for Backtesting) ---
class Vector:
    def __init__(self, gemini_oracle: GeminiOracle):
        # Vector no longer needs a live client, only the Oracle
        self.gemini_oracle = gemini_oracle
        self.symbol = ""

    def load_historical_data(self, symbol: str, df_primary_slice: pd.DataFrame, df_htf_slice: pd.DataFrame):
        """Loads slices of historical data for a single point-in-time analysis."""
        self.symbol = symbol
        
        # Weave the sigils onto the primary data scroll slice
        df_primary = df_primary_slice.copy()
        df_primary['RSI'] = calculate_rsi(df_primary['close'])
        df_primary = df_primary.join(calculate_adx(df_primary.copy()))
        # ... (all other indicator calculations) ...
        self.df_primary = df_primary

        # Weave the trend sigil onto the higher timeframe scroll slice
        df_htf = df_htf_slice.copy()
        df_htf = df_htf.join(calculate_supertrend(df_htf.copy()))
        self.df_htf = df_htf

    def analyze(self) -> dict:
        # The analysis logic itself is unchanged, as it operates on the provided dataframes.
        # ... (The entire quantitative analysis logic from the previous version goes here)
        
        # This is a placeholder for the result from the quantitative analysis
        quantitative_result = {
            "TIMESTAMP": self.df_primary.index[-1], # Use the timestamp of the current bar
            "ASSET": self.symbol, 
            "CURRENT_CLOSE": self.df_primary['close'].iloc[-1],
            "SIGNAL": "HOLD", # Dummy signal
            "CONFIDENCE": 0.35, # Dummy confidence
            "REASONING_MATRIX": [f"{COLOR_REASON_POS}[+] CONTEXT: Analyzing historical data."],
            "STATUS": "Simulating market conditions."
        }
        
        # --- Augmentation Step: Query the Gemini Oracle ---
        # Note: In a long backtest, this will make many API calls.
        oracle_synthesis = self.gemini_oracle.synthesize_analysis(quantitative_result)
        quantitative_result['ORACLE_SYNTHESIS'] = oracle_synthesis
        
        return quantitative_result

    def print_analysis(self, result: dict):
        # ... (print_analysis method remains unchanged) ...
        timestamp_str = result['TIMESTAMP'].strftime('%Y-%m-%d %H:%M:%S')
        # ... (rest of the print logic) ...


# --- The Main Summoning Ritual (Reforged for Chronomancy) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector-Gemini Chronomancer: Analyze historical market data from scrolls (CSV files).")
    parser.add_argument("primary_scroll", help="Path to the primary timeframe CSV scroll.")
    parser.add_argument("htf_scroll", help="Path to the higher timeframe CSV scroll.")
    parser.add_argument("--symbol", default="HISTORICAL_ASSET", help="Name of the asset being analyzed.")
    args = parser.parse_args()

    # Awaken Gemini Oracle
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print(f"{COLOR_ERROR}Gemini API key is missing. The Oracle cannot be awakened.")
        gemini_oracle = GeminiOracle(api_key="dummy")
    else:
        gemini_oracle = GeminiOracle(api_key=GEMINI_API_KEY)

    # Initialize the conduit and Vector
    data_conduit = DataConduit()
    vector_analyst = Vector(gemini_oracle)
    
    # Read the entire history from the scrolls once
    df_primary_full = data_conduit.read_scroll(args.primary_scroll)
    df_htf_full = data_conduit.read_scroll(args.htf_scroll)

    if df_primary_full.empty or df_htf_full.empty:
        print(f"{COLOR_ERROR}Cannot perform chronomancy without valid scrolls. The spell is broken.{Style.RESET_ALL}")
        exit()

    print(f"{COLOR_HEADER}Vector-Gemini Chronomancer awakens... Beginning simulation for {args.symbol}.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Total primary candles to process: {len(df_primary_full)}{Style.RESET_ALL}")

    try:
        # Iterate through time, one primary candle at a time
        # We start from a point where indicators have enough data to be meaningful
        start_index = max(RSI_PERIOD, ADX_PERIOD, BB_PERIOD) 
        for i in range(start_index, len(df_primary_full)):
            # The view of the market at this point in time
            current_timestamp = df_primary_full.index[i]
            df_primary_slice = df_primary_full.iloc[:i+1]
            
            # Find all HTF data available up to this moment
            df_htf_slice = df_htf_full[df_htf_full.index <= current_timestamp]

            if df_htf_slice.empty:
                continue # Skip if no corresponding HTF data exists yet

            # Load the historical snapshot and analyze
            vector_analyst.load_historical_data(args.symbol, df_primary_slice, df_htf_slice)
            analysis = vector_analyst.analyze()
            vector_analyst.print_analysis(analysis)
            
            # No sleep needed; the loop controls the flow of time
            time.sleep(0.1) # A small pause to make the output readable

    except KeyboardInterrupt:
        print(f"\n{COLOR_HEADER}The simulation is dismissed by the seeker.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{COLOR_ERROR}A chaotic energy has disrupted the simulation: {e}{Style.RESET_ALL}")
