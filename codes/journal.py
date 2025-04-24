import os
import json
import pandas as pd
import datetime as dt
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import schedule
import time
import logging
from colorama import init, Fore, Style, Back
import sys

# Initialize Colorama - Let the terminal bloom with color!
# autoreset=True ensures styles reset after each print automatically
init(autoreset=True)

# --- Arcane Configuration ---
CSV_FILENAME = 'bybit_trading_journal.csv'
LOG_FILENAME = 'trading_journal.log'
BACKUP_PREFIX = 'backup_'
# Define the realms (categories) to explore - Add "spot", "option" if needed
CATEGORIES = ["linear", "inverse"]
# Define the dawn of your historical quest (Format: YYYY-MM-DD)
HISTORICAL_START_DATE_STR = "2023-01-01"
# Define the temporal rift for daily fetches (in hours)
DAILY_FETCH_HOURS = 24
# Define the pause between API whispers (in seconds) to appease rate limits
API_DELAY = 0.6 # Increased slightly for safety
# Define the number of retries for transient API phantoms
MAX_RETRIES = 3
# Define the delay between retry attempts (in seconds)
RETRY_DELAY = 5

# --- Setup the Logging Sigils ---
# Configure logging to capture whispers to a file
logging.basicConfig(
    filename=LOG_FILENAME,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO # Log INFO level and above to the file
)
# Configure logging to echo vibrant messages to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # Show INFO level messages and above on console
# Use a formatter that weaves in Colorama magic
formatter = logging.Formatter(
    f'{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - {Fore.YELLOW}%(levelname)s{Style.RESET_ALL} - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
# Get the root logger
logger = logging.getLogger()
# Remove default handlers if any to avoid duplicate console logs (important!)
# This prevents messages from being printed twice to the console
if logger.hasHandlers():
    logger.handlers.clear()
# Add our configured file and console handlers
logger.addHandler(logging.FileHandler(LOG_FILENAME)) # File handler first
logger.addHandler(console_handler) # Then console handler
logger.setLevel(logging.INFO) # Ensure root logger level captures INFO messages

class TradingJournal:
    """
    A mystical class to chronicle Bybit trades, enhanced by Pyrmethus.
    Handles API interaction, data processing, CSV storage with deduplication,
    and backups, all illuminated by Colorama.
    """
    def __init__(self, csv_file=CSV_FILENAME):
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}Initializing the Trading Journal Chronicle...{Style.RESET_ALL}")
        load_dotenv()
        self.csv_file = csv_file

        # Summon API credentials from the ether (environment variables)
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')

        if not self.api_key or not self.api_secret:
            error_msg = f"{Back.RED}{Fore.WHITE}Fatal Error:{Style.RESET_ALL}{Fore.RED} BYBIT_API_KEY or BYBIT_API_SECRET not found in .env file or environment.{Style.RESET_ALL}"
            logger.critical(error_msg) # Use critical for fatal startup errors
            raise ValueError("API credentials not found. Ensure .env file is configured correctly.")

        # Weave the connection to Bybit's realm
        try:
            logger.info("Attempting to weave connection to Bybit's realm...")
            self.session = HTTP(
                api_key=self.api_key,
                api_secret=self.api_secret,
                # testnet=True # Uncomment to conjure on the testnet plane
            )
            # Verify the connection pact by fetching basic account info
            logger.info("Verifying API connection pact...")
            self._api_call_with_retry(self.session.get_account_info) # Use retry wrapper here too
            logger.info(f"{Fore.GREEN}API connection pact sealed successfully.{Style.RESET_ALL}")
        except ConnectionError as e: # Catch errors raised by _api_call_with_retry
            logger.critical(f"{Back.RED}{Fore.WHITE}API connection failed during verification:{Style.RESET_ALL}{Fore.RED} {str(e)}{Style.RESET_ALL}")
            raise # Re-raise the caught ConnectionError to halt initialization
        except Exception as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected error during API initialization:{Style.RESET_ALL}{Fore.RED} {str(e)}{Style.RESET_ALL}")
            raise ConnectionError(f"Unexpected API initialization error: {str(e)}") # Raise as ConnectionError

    def _api_call_with_retry(self, method, *args, **kwargs):
        """Invoke an API method with resilience against transient phantoms."""
        for attempt in range(MAX_RETRIES):
            try:
                # Channel the API call
                response = method(*args, **kwargs)

                # Check Bybit's specific success code
                ret_code = response.get('retCode', -1) # Default to -1 if not present
                ret_msg = response.get('retMsg', '')

                if ret_code == 0:
                    # Success
                    return response
                else:
                    # Bybit API Error
                    error_msg = f"API Error (Code: {ret_code}, Msg: {ret_msg})"
                    # Decide if retryable based on Bybit codes
                    # Common Bybit rate limit codes: 10002, 10016 (Request freq too high)
                    # Common Bybit timeout code: 10006 (Request timeout)
                    # Add other potentially transient codes if known
                    is_rate_limit = ret_code in [10002, 10016] or "rate limit" in ret_msg.lower()
                    is_timeout = ret_code == 10006
                    is_retryable_bybit_error = is_rate_limit or is_timeout

                    if is_retryable_bybit_error and attempt < MAX_RETRIES - 1:
                        logger.warning(f"{Fore.YELLOW}Attempt {attempt + 1}/{MAX_RETRIES}: API issue encountered ({error_msg}). Retrying in {RETRY_DELAY}s...{Style.RESET_ALL}")
                        time.sleep(RETRY_DELAY)
                        continue # Go to next attempt
                    else:
                        # Non-retryable Bybit API error or last attempt failed
                        logger.error(f"{Fore.RED}API call failed permanently after {attempt + 1} attempts: {error_msg}{Style.RESET_ALL}")
                        raise ConnectionError(f"API call failed: {error_msg}") # Raise specific error

            except ConnectionError as ce: # Catch specific connection errors first
                 error_str = str(ce).lower()
                 is_retryable_exception = "timeout" in error_str or "connection" in error_str or "429" in error_str

                 if is_retryable_exception and attempt < MAX_RETRIES - 1:
                    logger.warning(f"{Fore.YELLOW}Attempt {attempt + 1}/{MAX_RETRIES}: Transient network phantom encountered ({ce}). Retrying in {RETRY_DELAY}s...{Style.RESET_ALL}")
                    time.sleep(RETRY_DELAY)
                 else:
                    logger.error(f"{Fore.RED}Network/Request failed permanently after {attempt + 1} attempts: {str(ce)}{Style.RESET_ALL}")
                    raise # Re-raise the caught exception
            except Exception as e:
                # Handle other potential exceptions (e.g., JSON decoding errors, unexpected issues)
                logger.error(f"{Fore.RED}Unexpected error during API call attempt {attempt + 1}/{MAX_RETRIES}: {str(e)}{Style.RESET_ALL}")
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"{Fore.YELLOW}Retrying in {RETRY_DELAY}s due to unexpected error...{Style.RESET_ALL}")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"{Back.RED}{Fore.WHITE}Giving up after {MAX_RETRIES} attempts due to unexpected error.{Style.RESET_ALL}")
                    raise ConnectionError(f"Unexpected API call error after retries: {str(e)}") # Raise specific error

        # If loop finishes without returning/raising (should be unreachable with current logic, but as safeguard)
        logger.error(f"{Back.RED}{Fore.WHITE}API call failed after {MAX_RETRIES} attempts (Logic Error?).{Style.RESET_ALL}")
        raise ConnectionError(f"API call failed after {MAX_RETRIES} attempts.")


    def fetch_closed_pnl(self, category="linear", start_time=None, end_time=None, limit=100):
        """Fetch closed PnL data, navigating the pagination currents with resilience."""
        logger.info(f"{Fore.CYAN}Summoning closed PnL spirits for realm: {Style.BRIGHT}{category}{Style.RESET_ALL}...")
        all_data = []
        cursor = None
        total_fetched = 0
        page_count = 0
        try:
            while True:
                page_count += 1
                logger.debug(f"Fetching page {page_count} for {category} with cursor: {cursor}")
                result = self._api_call_with_retry(
                    self.session.get_closed_pnl,
                    category=category,
                    startTime=start_time,
                    endTime=end_time,
                    limit=limit,
                    cursor=cursor
                )

                # _api_call_with_retry now raises an exception on permanent failure or non-zero retCode
                data_list = result.get('result', {}).get('list', []) # Safely access nested data
                page_fetched_count = len(data_list)
                total_fetched += page_fetched_count
                all_data.extend(data_list)
                cursor = result.get('result', {}).get('nextPageCursor')

                logger.debug(f"Page {page_count}: Fetched {page_fetched_count} records for {category}. Total so far: {total_fetched}. Next cursor: {'Yes' if cursor else 'No'}")

                if not cursor: # Bybit API uses empty string "" or null when no more pages
                    logger.info(f"{Fore.GREEN}Finished summoning. Total {total_fetched} records found for {category} across {page_count} pages.{Style.RESET_ALL}")
                    break # Exit loop if no more pages

                if not data_list and cursor:
                     # This case might indicate an issue or just an empty page before the end? Log it.
                     logger.warning(f"{Fore.YELLOW}Page {page_count} for {category} returned no data but has a next cursor. Continuing pagination...{Style.RESET_ALL}")

                # Respect the API's rhythm - pause *between* successful page fetches
                time.sleep(API_DELAY)

            return all_data
        except ConnectionError as e: # Catch errors raised by _api_call_with_retry
             logger.error(f"{Fore.RED}Failed to summon closed PnL data for {category} due to API/Network issue: {str(e)}{Style.RESET_ALL}")
             return [] # Return empty list on failure
        except Exception as e:
            logger.error(f"{Fore.RED}An unexpected error occurred while summoning PnL for {category}: {str(e)}{Style.RESET_ALL}")
            # Consider logging traceback for debugging: logging.exception("...")
            return [] # Return empty on other unexpected failures

    def process_data(self, data, category):
        """Transmute raw data into structured insights, adding context."""
        if not data:
            logger.info(f"{Fore.YELLOW}No raw data provided for processing (Realm: {category}).{Style.RESET_ALL}")
            return pd.DataFrame()

        logger.info(f"{Fore.CYAN}Weaving {len(data)} raw records from realm '{category}' into the chronicle...{Style.RESET_ALL}")
        try:
            df = pd.DataFrame(data)
            if df.empty:
                logger.info(f"{Fore.YELLOW}DataFrame created from raw data is empty (Realm: {category}).{Style.RESET_ALL}")
                return df

            # Add the category column early for context
            df['category'] = category

            # Ensure essential columns exist before proceeding; log if not
            # orderId is crucial for deduplication
            required_cols = ['createdTime', 'updatedTime', 'symbol', 'side', 'avgEntryPrice', 'avgExitPrice', 'closedSize', 'closedPnl', 'leverage', 'orderId', 'execType']
            present_cols = df.columns.tolist()
            missing_cols = [col for col in required_cols if col not in present_cols]
            if missing_cols:
                logger.warning(f"{Fore.YELLOW}Missing essential columns in data for realm {category}: {missing_cols}. Available: {present_cols}{Style.RESET_ALL}")
                # Add missing required columns with None/NaN value
                for col in missing_cols:
                    df[col] = pd.NA # Use pandas NA for consistency

            # Map closing order side to position direction
            # 'side' in closedPnl indicates the side of the *closing* order(s).
            # If closing order side is 'Sell', it closed a 'Long' position.
            # If closing order side is 'Buy', it closed a 'Short' position.
            df['Position_Direction'] = df['side'].apply(lambda x: 'Long' if x == 'Sell' else ('Short' if x == 'Buy' else 'Unknown'))

            # Select, reorder, and ensure columns exist
            # Ensure 'orderId' is kept for deduplication, even if not in base_cols list initially
            base_cols = ['createdTime', 'updatedTime', 'symbol', 'category', 'Position_Direction', 'avgEntryPrice', 'avgExitPrice', 'closedSize', 'closedPnl', 'leverage', 'orderId', 'execType']
            # Ensure all base_cols that are present in df are included, plus any others (like orderId if missing from base_cols)
            final_cols = [col for col in base_cols if col in df.columns]
            # Add any columns present in df but not in base_cols (like orderId if it wasn't listed)
            final_cols.extend([col for col in df.columns if col not in final_cols])
            # Make sure orderId is present if it exists at all
            if 'orderId' in df.columns and 'orderId' not in final_cols:
                 final_cols.append('orderId')

            df = df[final_cols] # Reorder/select

            # Convert timestamps (ms) to lucid datetime objects, handle potential errors
            for col in ['createdTime', 'updatedTime']:
                 if col in df.columns:
                    # Coerce errors will turn unparsable values into NaT (Not a Time)
                    df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
                    # Optionally, handle NaT values if needed, e.g., fill with a default or log them
                    if df[col].isnull().any():
                        invalid_count = df[col].isnull().sum()
                        logger.warning(f"{Fore.YELLOW}Found {invalid_count} invalid timestamp values in column '{col}' for realm {category}, converted to NaT.{Style.RESET_ALL}")

            # Convert numeric columns, coercing errors to NaN
            numeric_cols = ['avgEntryPrice', 'avgExitPrice', 'closedSize', 'closedPnl', 'leverage']
            for col in numeric_cols:
                if col in df.columns:
                    original_dtype = df[col].dtype
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any() and not pd.api.types.is_numeric_dtype(original_dtype):
                         # Log only if conversion actually introduced NaNs from non-numeric types
                         invalid_count = df[col].isnull().sum() - pd.to_numeric(data[col], errors='coerce').isnull().sum() # Approx original NaNs
                         if invalid_count > 0:
                            logger.warning(f"{Fore.YELLOW}Found {invalid_count} non-numeric values in column '{col}' for realm {category}, converted to NaN.{Style.RESET_ALL}")


            # Ensure orderId is treated as string for consistency in comparisons
            if 'orderId' in df.columns:
                 df['orderId'] = df['orderId'].astype(str)


            # Add empty columns for the scribe's manual notes
            manual_cols = ['Strategy', 'Emotions', 'Lessons_Learned']
            for col in manual_cols:
                if col not in df.columns: # Only add if not already present
                    df[col] = '' # Initialize as empty string

            logger.info(f"{Fore.GREEN}Successfully processed {len(df)} records for realm {category}.{Style.RESET_ALL}")
            return df

        except Exception as e:
            logger.error(f"{Fore.RED}Failed to process data for realm {category}: {str(e)}{Style.RESET_ALL}")
            logging.exception(f"Traceback for data processing error in realm {category}:") # Log full traceback to file
            return pd.DataFrame() # Return empty DataFrame on processing error

    def save_to_csv(self, df_new):
        """Commit the chronicle to the CSV scroll, banishing duplicate echoes."""
        if df_new.empty:
            logger.info(f"{Fore.YELLOW}No new insights to commit to the scroll.{Style.RESET_ALL}")
            return

        if 'orderId' not in df_new.columns:
            logger.error(f"{Back.RED}{Fore.WHITE}Critical Error:{Style.RESET_ALL}{Fore.RED} Cannot save data - 'orderId' column is missing from the new data to be saved.{Style.RESET_ALL}")
            return # Cannot proceed without orderId for deduplication

        # Ensure orderId is string in the new data BEFORE reading existing file
        df_new['orderId'] = df_new['orderId'].astype(str)
        original_new_count = len(df_new)
        df_to_append = pd.DataFrame()

        try:
            # Check if the scroll already exists and is not empty
            if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > 0:
                logger.info(f"{Fore.CYAN}Consulting the existing scroll ({self.csv_file}) to prevent echoes...{Style.RESET_ALL}")
                try:
                    # Read existing data, ensuring orderId is treated as string
                    existing_df = pd.read_csv(self.csv_file, dtype={'orderId': str}, keep_default_na=False, na_values=[''])
                    if 'orderId' not in existing_df.columns:
                        logger.warning(f"{Fore.YELLOW}Existing scroll is missing 'orderId' column. Cannot perform deduplication. Appending all {original_new_count} new records.{Style.RESET_ALL}")
                        df_to_append = df_new
                    else:
                        # Perform deduplication
                        existing_ids = set(existing_df['orderId'].unique())
                        df_to_append = df_new[~df_new['orderId'].isin(existing_ids)].copy() # Use .copy() to avoid SettingWithCopyWarning
                        duplicates_found = original_new_count - len(df_to_append)
                        if duplicates_found > 0:
                            logger.info(f"{Fore.YELLOW}Banished {duplicates_found} duplicate echoes based on orderId.{Style.RESET_ALL}")
                        else:
                            logger.info("No duplicate echoes found based on orderId.")

                except pd.errors.EmptyDataError:
                    logger.warning(f"{Fore.YELLOW}Existing scroll ({self.csv_file}) is empty despite existing. Proceeding to write anew.{Style.RESET_ALL}")
                    df_to_append = df_new
                except Exception as e:
                    logger.error(f"{Fore.RED}Error reading existing scroll {self.csv_file}: {str(e)}. Appending might create duplicates if deduplication fails.{Style.RESET_ALL}")
                    df_to_append = df_new # Proceed cautiously by appending all new data

            else:
                # File doesn't exist or is empty, all new data is good to append
                logger.info(f"Scroll ({self.csv_file}) not found or is empty. Preparing to inscribe {original_new_count} new records.")
                df_to_append = df_new

            # Append only the non-duplicate records
            if not df_to_append.empty:
                is_new_file = not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0
                logger.info(f"{Fore.CYAN}Inscribing {len(df_to_append)} new insights onto the scroll ({self.csv_file})...{Style.RESET_ALL}")

                # Ensure columns match existing file if appending, or use df_to_append's columns if new
                header_to_write = is_new_file
                cols_to_write = df_to_append.columns.tolist()

                # If appending, try to align columns with the existing file to prevent issues
                if not is_new_file and 'existing_df' in locals():
                     existing_cols = existing_df.columns.tolist()
                     # Reorder df_to_append columns to match existing_df, adding missing ones with NA
                     df_to_append = df_to_append.reindex(columns=existing_cols, fill_value=pd.NA)
                     # If df_to_append had *extra* columns, they are now dropped unless added to existing_cols logic
                     # For simplicity, we assume the goal is to match the existing structure when appending.
                     cols_to_write = existing_cols # Use the existing column order

                df_to_append.to_csv(
                    self.csv_file,
                    mode='a',
                    header=header_to_write, # Add header only if file is new or was empty
                    index=False,
                    columns=cols_to_write # Explicitly specify column order
                )
                logger.info(f"{Fore.GREEN}Successfully inscribed {len(df_to_append)} new records.{Style.RESET_ALL}")
            else:
                logger.info(f"{Fore.GREEN}No new, unique records found to inscribe after consulting the scroll.{Style.RESET_ALL}")

        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}Failed to commit insights:{Style.RESET_ALL}{Fore.RED} Error saving to the scroll ({self.csv_file}): {str(e)}{Style.RESET_ALL}")
            logging.exception(f"Traceback for CSV saving error:")

    def backup_csv(self):
        """Create a time-stamped echo of the scroll for safekeeping."""
        if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
            logger.info(f"{Fore.YELLOW}No scroll ({self.csv_file}) found or it's empty. No echo created.{Style.RESET_ALL}")
            return

        try:
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            # Ensure the backup directory exists if you want backups elsewhere
            # backup_dir = "backups"
            # os.makedirs(backup_dir, exist_ok=True)
            # backup_file = os.path.join(backup_dir, f"{BACKUP_PREFIX}{timestamp}_{os.path.basename(self.csv_file)}")
            backup_file = f"{BACKUP_PREFIX}{timestamp}_{os.path.basename(self.csv_file)}"

            # Use simple file copy for backup; avoids pandas reading/writing overhead for just backup
            import shutil
            shutil.copy2(self.csv_file, backup_file) # copy2 preserves metadata like modification time
            logger.info(f"{Fore.GREEN}Created a protective echo: {backup_file}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Failed to create protective echo: {str(e)}{Style.RESET_ALL}")

    def fetch_historical_data(self, start_date_str=HISTORICAL_START_DATE_STR, end_date_dt=None, categories=CATEGORIES):
        """Embark on a quest to retrieve ancient chronicles from specified realms, chunk by chunk."""
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Initiating Historical Quest ---{Style.RESET_ALL}")
        try:
            # Convert start date string to datetime object and then to milliseconds timestamp
            start_date_dt = dt.datetime.strptime(start_date_str, "%Y-%m-%d")
            start_time_ms = int(start_date_dt.timestamp() * 1000)
        except ValueError:
            logger.error(f"{Back.RED}{Fore.WHITE}Invalid HISTORICAL_START_DATE_STR format ({start_date_str}). Use YYYY-MM-DD.{Style.RESET_ALL}")
            return

        # Use current time if end_date is not provided
        if end_date_dt is None:
            end_date_dt = dt.datetime.now()
        end_time_ms = int(end_date_dt.timestamp() * 1000)

        # Ensure start time is before end time
        if start_time_ms >= end_time_ms:
             logger.error(f"{Fore.RED}Historical start time ({start_date_dt}) is not before end time ({end_date_dt}). Aborting quest.{Style.RESET_ALL}")
             return

        start_str = start_date_dt.strftime('%Y-%m-%d')
        end_str = end_date_dt.strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Questing for chronicles from {Fore.CYAN}{start_str}{Style.RESET_ALL} to {Fore.CYAN}{end_str}{Style.RESET_ALL}")

        # Define the temporal chunk size (e.g., 7 days in milliseconds) to avoid overwhelming the API
        # Bybit API max range for this endpoint is often 7 days, check their docs if issues arise
        chunk_size_ms = 7 * 24 * 60 * 60 * 1000
        all_historical_data = []
        total_records_across_realms = 0

        for category in categories:
            logger.info(f"{Fore.BLUE}--- Questing in Realm: {Style.BRIGHT}{category}{Style.RESET_ALL} ---")
            current_start_ms = start_time_ms
            records_this_realm = 0
            chunk_num = 0
            while current_start_ms < end_time_ms:
                chunk_num += 1
                # Calculate the end of the current chunk, ensuring it doesn't exceed the overall end time
                # Bybit intervals are often [startTime, endTime) -> endTime should be start + duration
                current_end_ms = min(current_start_ms + chunk_size_ms, end_time_ms)

                start_chunk_str = dt.datetime.fromtimestamp(current_start_ms / 1000).strftime('%Y-%m-%d %H:%M')
                end_chunk_str = dt.datetime.fromtimestamp(current_end_ms / 1000).strftime('%Y-%m-%d %H:%M')
                logger.info(f"Chunk {chunk_num}: Fetching {Fore.CYAN}{start_chunk_str}{Style.RESET_ALL} to {Fore.CYAN}{end_chunk_str}{Style.RESET_ALL} for {category}")

                # Fetch data for the current chunk
                data = self.fetch_closed_pnl(
                    category=category,
                    start_time=current_start_ms,
                    end_time=current_end_ms # Use end of chunk as end time
                )
                if data:
                    df_chunk = self.process_data(data, category) # Pass category
                    if not df_chunk.empty:
                        all_historical_data.append(df_chunk)
                        records_this_realm += len(df_chunk)
                        logger.debug(f"Chunk {chunk_num}: Processed {len(df_chunk)} records for {category}.")
                    else:
                        logger.info(f"Chunk {chunk_num}: Processing yielded no data for {category}.")
                else:
                    logger.info(f"Chunk {chunk_num}: No raw data found for {category}.")

                # Advance to the next chunk's start time
                current_start_ms = current_end_ms
                # Pause slightly between chunks even if data was empty, to be kind to the API
                time.sleep(API_DELAY)

            logger.info(f"Completed quest in realm {category}. Found {records_this_realm} records.")
            total_records_across_realms += records_this_realm

        # After fetching all chunks for all categories, combine and save
        if all_historical_data:
             logger.info(f"{Fore.CYAN}Combining all fetched historical data ({total_records_across_realms} records across all realms)...{Style.RESET_ALL}")
             combined_df = pd.concat(all_historical_data, ignore_index=True)
             logger.info(f"Total unique historical records to potentially save: {len(combined_df)}")
             self.save_to_csv(combined_df)
        else:
             logger.info(f"{Fore.YELLOW}No historical data found for the specified period and realms.{Style.RESET_ALL}")

        logger.info(f"{Fore.GREEN}{Style.BRIGHT}--- Historical Quest Completed ---{Style.RESET_ALL}")
        self.backup_csv() # Create a backup after the historical fetch is done

    def fetch_daily_data(self, categories=CATEGORIES):
        """Perform the daily ritual to gather recent whispers from the ether."""
        ritual_time = dt.datetime.now()
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Performing Daily Ritual ({ritual_time.strftime('%Y-%m-%d %H:%M:%S')}) ---{Style.RESET_ALL}")

        # Calculate time window for the daily fetch (e.g., last 24 hours)
        end_time_ms = int(ritual_time.timestamp() * 1000)
        start_time_dt = ritual_time - dt.timedelta(hours=DAILY_FETCH_HOURS)
        start_time_ms = int(start_time_dt.timestamp() * 1000)

        start_str = start_time_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_str = ritual_time.strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Gathering whispers from {Fore.CYAN}{start_str}{Style.RESET_ALL} to {Fore.CYAN}{end_str}{Style.RESET_ALL} (approx last {DAILY_FETCH_HOURS} hours).")

        all_new_data = []
        total_records_across_realms = 0
        for category in categories:
            logger.info(f"{Fore.BLUE}--- Listening in Realm: {Style.BRIGHT}{category}{Style.RESET_ALL} ---")
            data = self.fetch_closed_pnl(
                category=category,
                start_time=start_time_ms,
                end_time=end_time_ms
            )
            if data:
                 df_daily = self.process_data(data, category) # Pass category
                 if not df_daily.empty:
                    all_new_data.append(df_daily)
                    records_this_realm = len(df_daily)
                    total_records_across_realms += records_this_realm
                    logger.info(f"Heard {records_this_realm} new whispers in realm {category}.")
                 else:
                     logger.info(f"Whispers heard in {category}, but processing yielded no data.")
            else:
                 logger.info(f"Silence in realm {category} for this period.")
            # Add a small delay between categories even in daily fetch
            time.sleep(API_DELAY / 2) # Smaller delay maybe?

        # Combine data from all categories for this daily run and save
        if all_new_data:
            logger.info(f"{Fore.CYAN}Combining daily whispers from all realms ({total_records_across_realms} total records)...{Style.RESET_ALL}")
            combined_df = pd.concat(all_new_data, ignore_index=True)
            logger.info(f"Total unique new records found during daily ritual: {len(combined_df)}")
            self.save_to_csv(combined_df)
        else:
             logger.info(f"{Fore.YELLOW}No new whispers heard in any realm during this daily ritual.{Style.RESET_ALL}")

        logger.info(f"{Fore.GREEN}{Style.BRIGHT}--- Daily Ritual Completed ---{Style.RESET_ALL}")
        self.backup_csv() # Create a backup after the daily fetch

def main_spell():
    """The main incantation to orchestrate the chronicle."""
    logger.info(f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}--- Pyrmethus's Bybit Journal Automaton Activated ---{Style.RESET_ALL}")

    try:
        journal = TradingJournal()
    except (ValueError, ConnectionError) as e:
        # Error already logged critically in __init__
        logger.critical(f"{Back.RED}{Fore.WHITE}Halting spell: Failed to initialize TradingJournal. Check logs above.{Style.RESET_ALL}")
        return # Stop execution if initialization fails
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}An unexpected critical error occurred during initialization: {e}{Style.RESET_ALL}")
        logging.exception("Traceback for critical initialization error:")
        return # Stop execution

    # --- Choose Your Path ---
    # Path 1: Initial Historical Fetch (Run only ONCE or when needed to backfill)
    # Uncomment the following lines *only* for the first run or to backfill history.
    # logger.info(f"{Fore.YELLOW}{Style.BRIGHT}Preparing for the grand Historical Quest... This may take some time.{Style.RESET_ALL}")
    # try:
    #     journal.fetch_historical_data() # Fetches from HISTORICAL_START_DATE_STR to now
    # except Exception as e:
    #     logger.error(f"{Back.RED}{Fore.WHITE}Historical Quest failed unexpectedly:{Style.RESET_ALL}{Fore.RED} {e}{Style.RESET_ALL}")
    #     logging.exception("Traceback for Historical Quest failure:") # Log details
    # logger.info(f"{Fore.GREEN}Historical Quest finished. Now proceeding to daily rituals.{Style.RESET_ALL}")
    # IMPORTANT: Comment out the journal.fetch_historical_data() call again after the first successful run!

    # Path 2: Scheduled Daily Updates (Standard operation)
    logger.info(f"{Fore.GREEN}Performing initial fetch for the last {DAILY_FETCH_HOURS} hours before scheduling...{Style.RESET_ALL}")
    try:
        # Perform one fetch immediately upon starting to catch up since last run
        journal.fetch_daily_data()
    except Exception as e:
         logger.error(f"{Fore.RED}Initial daily fetch encountered an unexpected rift: {e}{Style.RESET_ALL}")
         logging.exception("Traceback for initial daily fetch failure:") # Log details

    # Schedule the daily ritual
    # Fetches data for the *previous* N hours (defined by DAILY_FETCH_HOURS)
    # Running at 00:05 ensures we capture the full previous day
    schedule_time = "00:05"
    schedule.every().day.at(schedule_time).do(journal.fetch_daily_data)
    # Alternatively, run every X hours:
    # schedule.every(DAILY_FETCH_HOURS).hours.do(journal.fetch_daily_data) # Careful with overlap/timing
    logger.info(f"{Fore.GREEN}Scheduling the daily ritual to commence every day at {schedule_time}.{Style.RESET_ALL}")

    logger.info(f"{Fore.CYAN}The automaton now slumbers, awaiting the scheduled time or a manual interruption (CTRL+C)...{Style.RESET_ALL}")

    while True:
        try:
            schedule.run_pending()
            # Sleep for a while to avoid busy-waiting
            # Check the schedule every minute
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info(f"\n{Fore.YELLOW}{Style.BRIGHT}Spell broken by the user's will. The automaton rests.{Style.RESET_ALL}")
            break
        except Exception as e:
            # Catch unexpected errors in the scheduling loop itself
            logger.error(f"{Back.RED}{Fore.WHITE}An unexpected rift occurred in the main scheduling loop:{Style.RESET_ALL}{Fore.RED} {str(e)}{Style.RESET_ALL}")
            logging.exception("Traceback for main loop error:") # Log details
            logger.info("Attempting to mend the weave and continue slumbering in 60 seconds...")
            time.sleep(60) # Wait before potentially retrying the loop

if __name__ == "__main__":
    main_spell()
