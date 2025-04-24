```python
import pandas as pd
import numpy as np
import pandas_ta as ta
from colorama import init, Fore, Style

# Initialize Colorama for vibrant terminal output
init()

# —————————————————————————————————————————————————————————————————————————————
# ✨ Pyrmethus Configuration Spell - Weave your parameters here ✨
# —————————————————————————————————————————————————————————————————————————————
# Volumatic Trend Settings
LENGTH = 40
ATR_LENGTH = 200 # ATR period used in the original script for level calculation
VOLUME_PERCENTILE_LOOKBACK = 1000
VOLUME_NORMALIZATION_PERCENTILE = 100 # 100th percentile (max)

# Order Block Settings
OB_SOURCE = "Wicks" # "Wicks" or "Bodys"
PIVOT_LEFT_LEN_H = 25
PIVOT_RIGHT_LEN_H = 25
PIVOT_LEFT_LEN_L = 25
PIVOT_RIGHT_LEN_L = 25
MAX_BOXES = 50 # Max number of active Bull/Bear boxes to keep track of

# Colors (using Colorama constants)
COLOR_UP = Fore.CYAN + Style.BRIGHT
COLOR_DN = Fore.YELLOW + Style.BRIGHT
COLOR_BULL_BOX = Fore.GREEN
COLOR_BEAR_BOX = Fore.RED
COLOR_CLOSED_BOX = Fore.LIGHTBLACK_EX
COLOR_INFO = Fore.MAGENTA
COLOR_HEADER = Fore.BLUE + Style.BRIGHT
COLOR_RESET = Style.RESET_ALL

print(COLOR_HEADER + "~~~ Pyrmethus Volumatic Trend + OB Strategy Engine ~~~" + COLOR_RESET)

# —————————————————————————————————————————————————————————————————————————————
# 📜 Helper Incantations - Utility functions 📜
# —————————————————————————————————————————————————————————————————————————————

def ema_swma(series: pd.Series, length: int) -> pd.Series:
    """
    Calculates a smoothed EMA based on weighted average of last 4 values.
    Equivalent to: ta.ema(x[3]*1/6 + x[2]*2/6 + x[1]*2/6 + x[0]*1/6, len)
    """
    print(COLOR_INFO + f"# Calculating Smoothed Weighted EMA (Length: {length})..." + COLOR_RESET)
    # Ensure the series has enough data points for weighting
    if len(series) < 4:
        print(Fore.RED + "Warning: Series too short for SWMA calculation, returning standard EMA." + COLOR_RESET)
        return ta.ema(series, length=length)
        
    # Apply weights to the last 4 periods
    weighted_series = (series.shift(3) / 6 + 
                       series.shift(2) * 2 / 6 + 
                       series.shift(1) * 2 / 6 + 
                       series * 1 / 6)
    
    # Calculate EMA on the weighted series
    # Use adjust=False for behavior closer to TradingView's EMA calculation
    smoothed_ema = ta.ema(weighted_series.dropna(), length=length) 
    
    # Reindex to match the original series, filling initial NaNs
    return smoothed_ema.reindex(series.index)


def calculate_volatility_levels(df, length, atr_length):
    """Calculates trend, EMAs, ATR and dynamic levels."""
    print(COLOR_INFO + "# Calculating Volatility Trend Levels..." + COLOR_RESET)
    df['ema1'] = ema_swma(df['close'], length)
    df['ema2'] = ta.ema(df['close'], length=length)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length)

    # Determine trend (ema1 lagging ema2 indicates uptrend in the original logic)
    # Shift ema1 to compare with current ema2, matching ema1[1] < ema2
    df['trend_up'] = df['ema1'].shift(1) < df['ema2']
    df['trend_changed'] = df['trend_up'] != df['trend_up'].shift(1)

    # Initialize level columns
    for col in ['upper', 'lower', 'lower_vol', 'upper_vol', 'step_up', 'step_dn', 'last_trend_change_index']:
        df[col] = np.nan

    last_trend_change_idx = 0
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'trend_changed']:
            # Use values from the *previous* bar for calculation, as in Pine Script
            prev_idx = df.index[i-1]
            ema1_val = df.loc[prev_idx, 'ema1']
            atr_val = df.loc[prev_idx, 'atr']

            if pd.notna(ema1_val) and pd.notna(atr_val):
                upper = ema1_val + atr_val * 3
                lower = ema1_val - atr_val * 3
                lower_vol = lower + atr_val * 4
                upper_vol = upper - atr_val * 4
                step_up = (lower_vol - lower) / 100 if lower_vol > lower else 0 # Avoid division by zero or negative steps
                step_dn = (upper - upper_vol) / 100 if upper > upper_vol else 0

                df.loc[df.index[i]:, 'upper'] = upper
                df.loc[df.index[i]:, 'lower'] = lower
                df.loc[df.index[i]:, 'lower_vol'] = lower_vol
                df.loc[df.index[i]:, 'upper_vol'] = upper_vol
                df.loc[df.index[i]:, 'step_up'] = step_up
                df.loc[df.index[i]:, 'step_dn'] = step_dn
                
                last_trend_change_idx = i
                df.loc[df.index[i]:, 'last_trend_change_index'] = last_trend_change_idx
            else:
                 # If ema1 or atr is NaN, propagate the previous state or keep NaN
                 df.loc[df.index[i], ['upper', 'lower', 'lower_vol', 'upper_vol', 'step_up', 'step_dn', 'last_trend_change_index']] = df.loc[df.index[i-1], ['upper', 'lower', 'lower_vol', 'upper_vol', 'step_up', 'step_dn', 'last_trend_change_index']]

        elif i > 0 : # If trend didn't change, carry forward the levels
             df.loc[df.index[i], ['upper', 'lower', 'lower_vol', 'upper_vol', 'step_up', 'step_dn', 'last_trend_change_index']] = df.loc[df.index[i-1], ['upper', 'lower', 'lower_vol', 'upper_vol', 'step_up', 'step_dn', 'last_trend_change_index']]


    # Calculate normalized volume and volume steps
    # Using rolling max (100th percentile) as approximation for percentile_linear_interpolation(..., 100)
    df['percentile_vol'] = df['volume'].rolling(window=VOLUME_PERCENTILE_LOOKBACK, min_periods=1).max()
    df['vol_norm'] = np.where(df['percentile_vol'] != 0, (df['volume'] / df['percentile_vol'] * 100), 0).fillna(0)
    
    df['vol_up_step'] = df['step_up'] * df['vol_norm']
    df['vol_dn_step'] = df['step_dn'] * df['vol_norm']

    # Define Volumatic Trend levels based on volume
    df['vol_trend_up_level'] = df['lower'] + df['vol_up_step']
    df['vol_trend_dn_level'] = df['upper'] - df['vol_dn_step']
    
    # Calculate cumulative volume delta since last trend change
    df['volume_delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    df['volume_total'] = df['volume']
    
    # Group by trend change block and calculate cumulative sums
    trend_block = df['trend_changed'].cumsum()
    df['cum_vol_delta_since_change'] = df.groupby(trend_block)['volume_delta'].cumsum()
    df['cum_vol_total_since_change'] = df.groupby(trend_block)['volume_total'].cumsum()


    return df

def calculate_pivot_order_blocks(df, source, left_h, right_h, left_l, right_l):
    """Identifies pivot highs/lows for Order Blocks."""
    print(COLOR_INFO + f"# Calculating Pivot Order Blocks (Source: {source})..." + COLOR_RESET)
    
    high_col = df['high'] if source == "Wicks" else df['close']
    low_col = df['low'] if source == "Wicks" else df['close'] # Note: Pine used close for high pivot, open for low pivot if Bodys
    open_col = df['open']
    close_col = df['close']

    # Use pandas_ta for pivot points - Note: pandas_ta might differ slightly from TradingView's exact pivot logic
    # We need the index of the pivot, not just the price level.
    # Custom implementation for finding pivot indices:
    df['ph'] = np.nan
    df['pl'] = np.nan

    for i in range(left_h, len(df) - right_h):
        is_pivot_high = True
        pivot_high_val = high_col.iloc[i]
        # Check left side
        for j in range(1, left_h + 1):
            if high_col.iloc[i-j] >= pivot_high_val: # Use >= to match TV's behavior (equal highs don't break pivot)
                is_pivot_high = False
                break
        if not is_pivot_high: continue
        # Check right side
        for j in range(1, right_h + 1):
             if high_col.iloc[i+j] > pivot_high_val: # Use > for the right side
                is_pivot_high = False
                break
        if is_pivot_high:
            df.loc[df.index[i], 'ph'] = pivot_high_val # Mark the pivot high price at the pivot bar index

    for i in range(left_l, len(df) - right_l):
        is_pivot_low = True
        pivot_low_val = low_col.iloc[i]
        # Check left side
        for j in range(1, left_l + 1):
            if low_col.iloc[i-j] <= pivot_low_val: # Use <= for left side lows
                is_pivot_low = False
                break
        if not is_pivot_low: continue
        # Check right side
        for j in range(1, right_l + 1):
             if low_col.iloc[i+j] < pivot_low_val: # Use < for right side lows
                is_pivot_low = False
                break
        if is_pivot_low:
            df.loc[df.index[i], 'pl'] = pivot_low_val # Mark the pivot low price at the pivot bar index

    return df


def manage_order_blocks(df, right_h, right_l, source, max_boxes):
    """Creates, manages, and tracks Order Block states."""
    print(COLOR_INFO + "# Managing Order Block Boxes..." + COLOR_RESET)
    bull_boxes = [] # Stores active bullish OBs: [ {id, start_idx, end_idx, top, bottom, state} ]
    bear_boxes = [] # Stores active bearish OBs: [ {id, start_idx, end_idx, top, bottom, state} ]
    box_counter = 0

    # Add columns to store OB info for each bar (optional, mainly for analysis)
    df['active_bull_ob'] = None # Store ref to active bull OB if price is within one
    df['active_bear_ob'] = None # Store ref to active bear OB if price is within one

    for i in range(len(df)):
        current_idx = df.index[i]
        current_close = df.loc[current_idx, 'close']

        # --- Create new boxes ---
        # Bearish OB from Pivot High
        # Pivot is identified at index `i`, but the OB candle is `right_h` bars ago.
        if pd.notna(df.loc[current_idx, 'ph']):
            ob_candle_idx_num = i - right_h # Numerical index of the OB candle
            if ob_candle_idx_num >= 0:
                ob_candle_idx = df.index[ob_candle_idx_num]
                
                if source == "Bodys":
                    top_price = df.loc[ob_candle_idx, 'close']
                    bottom_price = df.loc[ob_candle_idx, 'open']
                else: # Wicks
                    top_price = df.loc[ob_candle_idx, 'high']
                    bottom_price = df.loc[ob_candle_idx, 'close'] # Pine used close for wick OB bottom

                # Ensure top > bottom
                if bottom_price > top_price:
                    top_price, bottom_price = bottom_price, top_price
                
                if pd.notna(top_price) and pd.notna(bottom_price):
                    box_counter += 1
                    new_box = {
                        'id': f'BearOB_{box_counter}',
                        'type': 'bear',
                        'start_idx': ob_candle_idx,
                        'end_idx': current_idx, # Initially ends at pivot detection bar
                        'top': top_price,
                        'bottom': bottom_price,
                        'state': 'active' # 'active' or 'closed'
                    }
                    bear_boxes.append(new_box)
                    # print(f"Created Bear Box {new_box['id']} at index {current_idx} from candle {ob_candle_idx}")


        # Bullish OB from Pivot Low
        # Pivot is identified at index `i`, OB candle is `right_l` bars ago.
        if pd.notna(df.loc[current_idx, 'pl']):
            ob_candle_idx_num = i - right_l
            if ob_candle_idx_num >= 0:
                ob_candle_idx = df.index[ob_candle_idx_num]

                if source == "Bodys":
                     # Pine used close[rightLenL] and open[rightLenL] - seems like a typo, should be from OB candle
                     # Assuming it meant body of the candle `right_l` bars ago
                     top_price = df.loc[ob_candle_idx, 'close'] 
                     bottom_price = df.loc[ob_candle_idx, 'open']
                else: # Wicks
                     top_price = df.loc[ob_candle_idx, 'open'] # Pine used open for wick OB top
                     bottom_price = df.loc[ob_candle_idx, 'low']

                # Ensure top > bottom
                if bottom_price > top_price:
                    top_price, bottom_price = bottom_price, top_price

                if pd.notna(top_price) and pd.notna(bottom_price):
                    box_counter += 1
                    new_box = {
                        'id': f'BullOB_{box_counter}',
                        'type': 'bull',
                        'start_idx': ob_candle_idx,
                        'end_idx': current_idx,
                        'top': top_price,
                        'bottom': bottom_price,
                        'state': 'active'
                    }
                    bull_boxes.append(new_box)
                    # print(f"Created Bull Box {new_box['id']} at index {current_idx} from candle {ob_candle_idx}")

        # --- Manage existing boxes ---
        active_bull_ref = None
        for box in bull_boxes:
            if box['state'] == 'active':
                # Check if closed: close goes below the bottom of the bull box
                if current_close < box['bottom']:
                    box['state'] = 'closed'
                    box['end_idx'] = current_idx # Close the box at the current bar
                    # print(f"Closed Bull Box {box['id']} at index {current_idx}")
                else:
                    # Extend the box visually (conceptually)
                    box['end_idx'] = current_idx
                    # Check if price is within this active box
                    if box['bottom'] <= current_close <= box['top']:
                         active_bull_ref = box # Mark that price is in an active bull OB

        active_bear_ref = None
        for box in bear_boxes:
            if box['state'] == 'active':
                # Check if closed: close goes above the top of the bear box
                if current_close > box['top']:
                    box['state'] = 'closed'
                    box['end_idx'] = current_idx
                    # print(f"Closed Bear Box {box['id']} at index {current_idx}")
                else:
                    # Extend the box
                    box['end_idx'] = current_idx
                    # Check if price is within this active box
                    if box['bottom'] <= current_close <= box['top']:
                         active_bear_ref = box # Mark that price is in an active bear OB
        
        # Store references to active OBs price is currently within
        # Using .loc might be slow in a loop; consider alternatives for large datasets
        df.loc[current_idx, 'active_bull_ob'] = active_bull_ref
        df.loc[current_idx, 'active_bear_ob'] = active_bear_ref


        # --- Clean up old boxes (based on creation order, keep only active ones) ---
        # This differs from Pine's array management but achieves similar goal in Python context
        active_bull_boxes = [b for b in bull_boxes if b['state'] == 'active']
        if len(active_bull_boxes) > max_boxes:
            # Remove the oldest active boxes
            num_to_remove = len(active_bull_boxes) - max_boxes
            oldest_active_indices = sorted(range(len(bull_boxes)), key=lambda k: bull_boxes[k]['start_idx'])
            removed_count = 0
            kept_boxes = []
            for idx in oldest_active_indices:
                 box_to_check = bull_boxes[idx]
                 if box_to_check['state'] == 'active' and removed_count < num_to_remove:
                     # Mark as inactive instead of deleting, or filter out later
                     # For simplicity here, we rebuild the list excluding the oldest *active* ones
                     removed_count += 1
                 else:
                     kept_boxes.append(box_to_check) # Keep closed boxes and newer active boxes
            bull_boxes = kept_boxes


        active_bear_boxes = [b for b in bear_boxes if b['state'] == 'active']
        if len(active_bear_boxes) > max_boxes:
            num_to_remove = len(active_bear_boxes) - max_boxes
            oldest_active_indices = sorted(range(len(bear_boxes)), key=lambda k: bear_boxes[k]['start_idx'])
            removed_count = 0
            kept_boxes = []
            for idx in oldest_active_indices:
                 box_to_check = bear_boxes[idx]
                 if box_to_check['state'] == 'active' and removed_count < num_to_remove:
                     removed_count += 1
                 else:
                     kept_boxes.append(box_to_check)
            bear_boxes = kept_boxes


    # Return the list of all boxes (active and closed) for potential analysis
    return df, bull_boxes, bear_boxes

# —————————————————————————————————————————————————————————————————————————————
# 🔮 Main Spell Execution - Processing the data 🔮
# —————————————————————————————————————————————————————————————————————————————

def run_strategy(data_path):
    """Loads data, runs calculations, and prints the final state."""
    try:
        # Summon data from the CSV realm
        print(COLOR_INFO + f"# Summoning data from {data_path}..." + COLOR_RESET)
        # Ensure standard column names: 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        df = pd.read_csv(data_path, parse_dates=['timestamp']) 
        df = df.set_index('timestamp')
        # Basic validation
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"Input CSV must contain columns: {required_cols}")
        
        print(Fore.GREEN + f"Data summoned successfully. Rows: {len(df)}" + COLOR_RESET)

    except FileNotFoundError:
        print(Fore.RED + Style.BRIGHT + f"Fatal Error: Cannot find the sacred scroll (file) at {data_path}" + COLOR_RESET)
        return
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"Fatal Error during data summoning: {e}" + COLOR_RESET)
        return

    # --- Apply Calculations ---
    print(COLOR_INFO + "# Weaving the Volumatic Trend enchantment..." + COLOR_RESET)
    df = calculate_volatility_levels(df, LENGTH, ATR_LENGTH)

    print(COLOR_INFO + "# Conjuring Pivot Order Blocks..." + COLOR_RESET)
    df = calculate_pivot_order_blocks(df, OB_SOURCE, PIVOT_LEFT_LEN_H, PIVOT_RIGHT_LEN_H, PIVOT_LEFT_LEN_L, PIVOT_RIGHT_LEN_L)
    df, final_bull_boxes, final_bear_boxes = manage_order_blocks(df, PIVOT_RIGHT_LEN_H, PIVOT_RIGHT_LEN_L, OB_SOURCE, MAX_BOXES)

    # --- Display Final State ---
    print(COLOR_HEADER + "\n~~~ Final State of the Aether ~~~" + COLOR_RESET)
    if not df.empty:
        last_row = df.iloc[-1]
        last_timestamp = df.index[-1]

        print(f"Timestamp: {Fore.WHITE}{last_timestamp}{COLOR_RESET}")
        
        # Volumatic Trend Info
        trend_state = "UP" if last_row['trend_up'] else "DOWN"
        trend_color = COLOR_UP if last_row['trend_up'] else COLOR_DN
        print(f"Volumatic Trend: {trend_color}{trend_state}{COLOR_RESET}")
        print(f"  EMA1 (Smoothed): {Fore.WHITE}{last_row['ema1']:.4f}{COLOR_RESET}")
        print(f"  EMA2 (Standard): {Fore.WHITE}{last_row['ema2']:.4f}{COLOR_RESET}")
        if last_row['trend_up']:
             print(f"  Support Level (Lower): {Fore.GREEN}{last_row['lower']:.4f}{COLOR_RESET}")
             print(f"  Vol-Adj Support (VolTrend Up): {Fore.GREEN}{last_row['vol_trend_up_level']:.4f}{COLOR_RESET}")
        else:
             print(f"  Resistance Level (Upper): {Fore.RED}{last_row['upper']:.4f}{COLOR_RESET}")
             print(f"  Vol-Adj Resistance (VolTrend Dn): {Fore.RED}{last_row['vol_trend_dn_level']:.4f}{COLOR_RESET}")
        
        if pd.notna(last_row['last_trend_change_index']):
            last_change_bar_index = int(last_row['last_trend_change_index'])
            last_change_timestamp = df.index[last_change_bar_index]
            print(f"  Last Trend Change: {Fore.WHITE}{last_change_timestamp}{COLOR_RESET} ({len(df) - last_change_bar_index} bars ago)")
            print(f"  Volume Delta since change: {Fore.WHITE}{last_row['cum_vol_delta_since_change']:.2f}{COLOR_RESET}")
            print(f"  Volume Total since change: {Fore.WHITE}{last_row['cum_vol_total_since_change']:.2f}{COLOR_RESET}")
        else:
             print(f"  Last Trend Change: {Fore.YELLOW}N/A (Insufficient history or no change){COLOR_RESET}")


        # Order Block Info
        print(f"\nActive Order Blocks (Max {MAX_BOXES}):")
        active_bull = [b for b in final_bull_boxes if b['state'] == 'active']
        active_bear = [b for b in final_bear_boxes if b['state'] == 'active']

        if not active_bull:
            print(f"  {COLOR_BULL_BOX}Bullish: None active{COLOR_RESET}")
        else:
            print(f"  {COLOR_BULL_BOX}Bullish ({len(active_bull)} active):{COLOR_RESET}")
            # Display latest few active bull boxes
            for box in sorted(active_bull, key=lambda x: x['start_idx'], reverse=True)[:5]: # Show newest 5
                print(f"    - ID: {box['id']}, Range: {Fore.WHITE}{box['bottom']:.4f} - {box['top']:.4f}{COLOR_RESET}, Started: {box['start_idx'].date()}")

        if not active_bear:
            print(f"  {COLOR_BEAR_BOX}Bearish: None active{COLOR_RESET}")
        else:
            print(f"  {COLOR_BEAR_BOX}Bearish ({len(active_bear)} active):{COLOR_RESET}")
            # Display latest few active bear boxes
            for box in sorted(active_bear, key=lambda x: x['start_idx'], reverse=True)[:5]: # Show newest 5
                 print(f"    - ID: {box['id']}, Range: {Fore.WHITE}{box['bottom']:.4f} - {box['top']:.4f}{COLOR_RESET}, Started: {box['start_idx'].date()}")
        
        # Check if current price is inside an active OB
        if last_row['active_bull_ob']:
            ob = last_row['active_bull_ob']
            print(f"\n{Fore.GREEN}Current price ({last_row['close']:.4f}) is within active Bull OB {ob['id']} ({ob['bottom']:.4f} - {ob['top']:.4f}){COLOR_RESET}")
        if last_row['active_bear_ob']:
            ob = last_row['active_bear_ob']
            print(f"\n{Fore.RED}Current price ({last_row['close']:.4f}) is within active Bear OB {ob['id']} ({ob['bottom']:.4f} - {ob['top']:.4f}){COLOR_RESET}")


        # --- Potential Strategy Signals (Example) ---
        # This part is illustrative; actual trading logic depends on specific rules.
        print(COLOR_HEADER + "\n~~~ Potential Signals (Illustrative) ~~~" + COLOR_RESET)
        if last_row['trend_changed'] and last_row['trend_up']:
            print(f"{COLOR_UP}Potential LONG Signal: Trend flipped UP at {last_timestamp}{COLOR_RESET}")
        elif last_row['trend_changed'] and not last_row['trend_up']:
             print(f"{COLOR_DN}Potential SHORT Signal: Trend flipped DOWN at {last_timestamp}{COLOR_RESET}")
        
        # Example: Enter long if trend is up and price enters an active Bull OB
        if last_row['trend_up'] and last_row['active_bull_ob'] and not df.iloc[-2]['active_bull_ob']: # Check previous bar to signal only on entry
             ob = last_row['active_bull_ob']
             print(f"{COLOR_BULL_BOX}Potential LONG Entry: Price entered Bull OB {ob['id']} during uptrend.{COLOR_RESET}")

        # Example: Enter short if trend is down and price enters an active Bear OB
        if not last_row['trend_up'] and last_row['active_bear_ob'] and not df.iloc[-2]['active_bear_ob']:
             ob = last_row['active_bear_ob']
             print(f"{COLOR_BEAR_BOX}Potential SHORT Entry: Price entered Bear OB {ob['id']} during downtrend.{COLOR_RESET}")

    else:
        print(Fore.YELLOW + "No data processed or DataFrame is empty." + COLOR_RESET)

    print(COLOR_MAGENTA + "\n# Incantation complete. The digital energies settle." + COLOR_RESET)
    
    # Optionally return the DataFrame with all calculations
    # return df


# —————————————————————————————————————————————————————————————————————————————
# 🚀 Summoning the Script's Power - Entry Point 🚀
# —————————————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    # === Configuration for Termux ===
    # 1. Ensure you have Python and pip installed in Termux:
    #    pkg update && pkg upgrade
    #    pkg install python
    # 2. Install required libraries:
    #    pip install pandas numpy pandas_ta colorama
    # 3. Prepare your data:
    #    Create a CSV file (e.g., 'crypto_data.csv') in your Termux home directory
    #    (/data/data/com.termux/files/home/) with columns:
    #    timestamp,open,high,low,close,volume
    #    Example timestamp format: '2023-10-26 08:00:00'
    # 4. Set the path to your data file below.

    # Path to your crypto data CSV file within Termux's home directory
    # Adjust the filename as needed
    data_file_path = "/data/data/com.termux/files/home/crypto_data.csv" 
    
    print(f"{Fore.CYAN}# Pyrmethus is preparing the ritual...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}# Attempting to read data from: {data_file_path}{Style.RESET_ALL}")
    
    # Execute the main strategy function
    run_strategy(data_file_path)

    # Example: How to save the processed data (optional)
    # processed_df = run_strategy(data_file_path)
    # if processed_df is not None:
    #     output_path = "/data/data/com.termux/files/home/processed_crypto_data.csv"
    #     try:
    #         processed_df.to_csv(output_path)
    #         print(f"\n{Fore.GREEN}Processed data saved to: {output_path}{Style.RESET_ALL}")
    #     except Exception as e:
    #         print(f"\n{Fore.RED}Failed to save processed data: {e}{Style.RESET_ALL}")

```

**Explanation and How to Use in Termux:**

1.  **Save the Code:** Save the Python code above as a file in your Termux home directory, for example, `volumatic_strategy.py`.
2.  **Install Dependencies:** Open Termux and run these commands:
    ```bash
    pkg update && pkg upgrade
    pkg install python libexpat libllvm # Dependencies for numpy/pandas if needed
    pip install pandas numpy pandas_ta colorama
    ```
3.  **Prepare Data:**
    *   You need a CSV file containing cryptocurrency OHLCV (Open, High, Low, Close, Volume) data.
    *   The CSV **must** have columns named exactly: `timestamp`, `open`, `high`, `low`, `close`, `volume`.
    *   The `timestamp` column should be in a format pandas can parse (e.g., `YYYY-MM-DD HH:MM:SS` or Unix timestamps).
    *   Place this CSV file in your Termux home directory (`/data/data/com.termux/files/home/`). Let's assume you name it `crypto_data.csv`.
4.  **Configure Data Path:** In the `if __name__ == "__main__":` block at the bottom of the Python script, make sure the `data_file_path` variable points to your CSV file:
    ```python
    data_file_path = "/data/data/com.termux/files/home/crypto_data.csv" 
    ```
5.  **Run the Script:** Execute the script from your Termux terminal:
    ```bash
    python volumatic_strategy.py
    ```

**What the Script Does:**

*   **Imports Libraries:** Brings in `pandas` for data manipulation, `numpy` for numerical operations, `pandas_ta` for technical analysis indicators (EMA, ATR), and `colorama` for the colorful terminal output.
*   **Configuration:** Sets up parameters like indicator lengths, OB settings, and colors using Colorama constants.
*   **Helper Functions:**
    *   `ema_swma`: Implements the custom smoothed EMA from the Pine Script.
    *   `calculate_volatility_levels`: Computes the core Volumatic Trend logic (EMAs, ATR, trend, levels, normalized volume).
    *   `calculate_pivot_order_blocks`: Identifies pivot points based on the specified source (`Wicks` or `Bodys`). *Note: Pivot point calculation might have slight differences from TradingView's specific implementation.*
    *   `manage_order_blocks`: Creates data structures (dictionaries within lists) to represent the Order Block boxes, tracks their state (active/closed), and manages the maximum number of active boxes.
*   **`run_strategy` Function:**
    *   Loads the data from the specified CSV file.
    *   Calls the calculation functions in sequence.
    *   Prints a summary of the *latest* data point's status, including:
        *   Current Volumatic Trend state (Up/Down) and relevant levels.
        *   Volume delta/total since the last trend change.
        *   Lists the most recent active Bullish and Bearish Order Blocks.
        *   Indicates if the current price falls within an active OB.
        *   Provides *illustrative* potential buy/sell signals based on trend changes or OB entries. **This is not financial advice and needs refinement for a real trading system.**
*   **Colorama Output:** Uses `Fore`, `Style`, and `COLOR_RESET` to make the terminal output vibrant and easy to read, highlighting important information like trend state, OB types, and signals.
*   **Termux Ready:** The script uses standard libraries installable via `pip` and references the Termux home directory structure.

This script provides the core logic translated from Pine Script into Python, focusing on calculating the indicator values and OB states, ready to be run within the Termux environment. Remember to adapt the data loading and any specific trading rules based on your needs.