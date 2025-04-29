
"""
Enhanced Trading Journal Module
Handles API interaction, data processing, CSV storage with deduplication,
and backups, all illuminated by Colorama.
"""
import datetime as dt
import logging
import os
import sys
import time
from decimal import Decimal
from typing import Dict, List, Optional, Union

import pandas as pd
import schedule
from colorama import Back, Fore, Style, init
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# Initialize Colorama with autoreset
init(autoreset=True)

# Constants
CSV_FILENAME = 'trades_journal.csv'
LOG_FILENAME = 'trading_journal.log'
BACKUP_PREFIX = 'backup_'
CATEGORIES = ["linear", "inverse"]
HISTORICAL_START_DATE = "2023-01-01"
DAILY_FETCH_HOURS = 24
API_DELAY = 1.0  # Increased for reliability
MAX_RETRIES = 5
RETRY_DELAY = 5

# Configure logging with color
logger = logging.getLogger("trading_journal")
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(LOG_FILENAME)
fh.setLevel(logging.INFO)
fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fh_formatter)

# Console handler with colors
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter(
    f'{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - {Fore.BLUE}%(name)s{Style.RESET_ALL} - '
    f'%(levelname)s - %(message)s'
)
ch.setFormatter(ch_formatter)

logger.addHandler(fh)
logger.addHandler(ch)

class TradingJournal:
    """Trading journal with enhanced error handling and colorized output."""
    
    def __init__(self, csv_file: str = CSV_FILENAME) -> None:
        logger.info(f"{Fore.MAGENTA}Initializing Trading Journal...{Style.RESET_ALL}")
        load_dotenv()
        self.csv_file = csv_file
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            error_msg = f"{Back.RED}{Fore.WHITE}Error: API credentials not found in environment{Style.RESET_ALL}"
            logger.critical(error_msg)
            raise ValueError("API credentials missing")
            
        try:
            self.session = HTTP(
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            self._verify_connection()
            logger.info(f"{Fore.GREEN}Successfully connected to Bybit API{Style.RESET_ALL}")
        except Exception as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Failed to initialize API connection: {e}{Style.RESET_ALL}")
            raise

    def _verify_connection(self) -> None:
        """Verify API connection is working."""
        try:
            self.session.get_api_key_info()
            logger.info(f"{Fore.GREEN}API connection verified{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}API verification failed: {e}{Style.RESET_ALL}")
            raise ConnectionError(f"API verification failed: {e}")

    def fetch_trades(self, category: str = "linear", start_time: Optional[int] = None) -> List[Dict]:
        """Fetch trade history with retry logic."""
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                logger.info(f"{Fore.CYAN}Fetching trades for {category}...{Style.RESET_ALL}")
                trades = self.session.get_closed_pnl(
                    category=category,
                    startTime=start_time,
                    limit=100
                )
                return trades.get('result', {}).get('list', [])
            except Exception as e:
                retry_count += 1
                logger.warning(f"{Fore.YELLOW}Attempt {retry_count}/{MAX_RETRIES} failed: {e}{Style.RESET_ALL}")
                if retry_count == MAX_RETRIES:
                    logger.error(f"{Back.RED}{Fore.WHITE}Failed to fetch trades after {MAX_RETRIES} attempts{Style.RESET_ALL}")
                    return []
                time.sleep(RETRY_DELAY)

    def process_trades(self, trades: List[Dict], category: str) -> pd.DataFrame:
        """Process trade data into DataFrame format."""
        if not trades:
            logger.info(f"{Fore.YELLOW}No trades found for {category}{Style.RESET_ALL}")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(trades)
            df['category'] = category
            df['timestamp'] = pd.to_datetime(df['createdTime'], unit='ms')
            df['pnl'] = pd.to_numeric(df['closedPnl'])
            df['position_value'] = pd.to_numeric(df['closedSize']) * pd.to_numeric(df['avgEntryPrice'])
            
            logger.info(f"{Fore.GREEN}Processed {len(df)} trades for {category}{Style.RESET_ALL}")
            return df
        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}Error processing trades: {e}{Style.RESET_ALL}")
            return pd.DataFrame()

    def save_trades(self, df: pd.DataFrame) -> None:
        """Save trades to CSV with deduplication."""
        if df.empty:
            logger.info(f"{Fore.YELLOW}No new trades to save{Style.RESET_ALL}")
            return

        try:
            mode = 'a' if os.path.exists(self.csv_file) else 'w'
            df.to_csv(self.csv_file, mode=mode, header=(mode == 'w'), index=False)
            logger.info(f"{Fore.GREEN}Saved {len(df)} trades to {self.csv_file}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}Error saving trades: {e}{Style.RESET_ALL}")

    def backup_journal(self) -> None:
        """Create backup of trading journal."""
        if not os.path.exists(self.csv_file):
            return

        try:
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{BACKUP_PREFIX}{timestamp}_{os.path.basename(self.csv_file)}"
            
            import shutil
            shutil.copy2(self.csv_file, backup_file)
            logger.info(f"{Fore.GREEN}Created backup: {backup_file}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Back.RED}{Fore.WHITE}Backup failed: {e}{Style.RESET_ALL}")

    def run_daily_update(self) -> None:
        """Run daily trade update."""
        logger.info(f"{Fore.MAGENTA}Starting daily trade update...{Style.RESET_ALL}")
        
        end_time = int(dt.datetime.now().timestamp() * 1000)
        start_time = end_time - (DAILY_FETCH_HOURS * 3600 * 1000)
        
        all_trades = []
        for category in CATEGORIES:
            trades = self.fetch_trades(category, start_time)
            if trades:
                df = self.process_trades(trades, category)
                all_trades.append(df)
                time.sleep(API_DELAY)
        
        if all_trades:
            combined_df = pd.concat(all_trades, ignore_index=True)
            self.save_trades(combined_df)
            self.backup_journal()
        
        logger.info(f"{Fore.GREEN}Daily update completed{Style.RESET_ALL}")

def main() -> None:
    """Main entry point for the trading journal."""
    logger.info(f"{Back.BLUE}{Fore.WHITE}Trading Journal Started{Style.RESET_ALL}")
    
    try:
        journal = TradingJournal()
        journal.run_daily_update()
        
        # Schedule daily updates
        schedule.every().day.at("00:00").do(journal.run_daily_update)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info(f"{Fore.YELLOW}Journal stopped by user{Style.RESET_ALL}")
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}Fatal error: {e}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()
