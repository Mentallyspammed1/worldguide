from dotenv import load_dotenv
import os
import ccxt

# Load API keys from .env file
load_dotenv()
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

# Initialize CCXT with Bybit
exchange = ccxt.bybit(
    {
        "apiKey": api_key,
        "secret": api_secret,
        "options": {
            "defaultType": "linear"  # For linear perpetual contracts like BTCUSDT
        },
    }
)

# Fetch open positions
positions = exchange.fetch_positions()

# Iterate through each position and set trailing stop if in profit
for position in positions:
    # Check if the position is in profit
    if float(position.get("unrealizedPnl", 0)) > 0:
        symbol = position["symbol"]
        side = position["side"]  # 'Buy' for long, 'Sell' for short
        positionIdx = position.get("positionIdx", 0)  # Default to 0 for one-way mode
        markPrice = position["markPrice"]

        # Calculate trailing stop distance (e.g., 1% of markPrice)
        trailingDistance = 0.01 * float(markPrice)

        # Prepare parameters for setting trailing stop
        params = {
            "symbol": symbol,
            "side": side,
            "positionIdx": positionIdx,
            "trailingStop": str(trailingDistance),
            "activePrice": str(markPrice),  # Set active price to current mark price
        }

        # Set trailing stop using Bybit's API
        exchange.privatePostV5PositionTradingStop(params)
        print(f"Set trailing stop for {symbol} {side} with distance {trailingDistance}")
    else:
        print(
            f"Position {position.get('symbol', 'Unknown')} {position.get('side', 'Unknown')} is not in profit, skipping."
        )

print("Process completed.")
