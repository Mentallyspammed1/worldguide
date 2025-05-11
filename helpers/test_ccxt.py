# test_ccxt.py
import os

import ccxt
from dotenv import load_dotenv

load_dotenv()
print("Testing basic CCXT connection...")

config = {
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
}
exchange_id = "bybit"
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class(config)

if os.getenv("BYBIT_TESTNET_MODE", "true").lower() == "true":
    print("Setting sandbox mode...")
    exchange.set_sandbox_mode(True)
else:
    print("Using mainnet...")

try:
    print("Attempting to load markets...")
    markets = exchange.load_markets()
    print(f"Successfully loaded {len(markets)} markets.")
    # Optional: Check for your specific symbol
    if "DOT/USDT:USDT" in markets:
        print("DOT/USDT:USDT market found.")
    else:
        print("DOT/USDT:USDT market NOT found.")

except ccxt.NetworkError as e:
    print(f"CCXT NetworkError: {e}")
except ccxt.ExchangeError as e:
    print(f"CCXT ExchangeError: {e}")
except ccxt.AuthenticationError as e:
    print(f"CCXT AuthenticationError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback

    traceback.print_exc()

finally:
    # Attempt to close connection if possible
    if hasattr(exchange, "close") and callable(exchange.close):
        try:
            # close() might be async, wrap if necessary for testing
            # await exchange.close() # If running in async context
            print("Closing exchange connection (sync attempt)...")
            # exchange.close() # Sync close might not exist or work properly
        except Exception as close_err:
            print(f"Error closing connection: {close_err}")
