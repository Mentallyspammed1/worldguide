import pandas as pd
from decimal import Decimal

try:
    df = pd.read_csv('bybit_trading_journal.csv')
    df['closedPnl'] = df['closedPnl'].astype(float)
    summary = df.groupby('symbol')['closedPnl'].agg(['sum', 'count']).sort_values(by='sum', ascending=False)
    print(summary)
except Exception as e:
    print(f"Error: {e}")
