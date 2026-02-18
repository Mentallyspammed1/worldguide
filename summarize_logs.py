import os
import re
from decimal import Decimal

log_dir = 'bot_logs'
pnl_pattern = re.compile(r'PnL=([-+]?\d*\.?\d+|N/A)')
closed_pnl_pattern = re.compile(r'Closed .* Net: \$([-+]?\d*\.?\d+)')

stats = {}

for filename in os.listdir(log_dir):
    if not filename.endswith('.log'):
        continue

    symbol = filename.split('_')[0]
    if symbol not in stats:
        stats[symbol] = {'realized_pnl': Decimal('0'), 'trades': 0}

    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            match = closed_pnl_pattern.search(line)
            if match:
                pnl = Decimal(match.group(1))
                stats[symbol]['realized_pnl'] += pnl
                stats[symbol]['trades'] += 1

print(f"{'Symbol':<20} | {'Realized PnL':<15} | {'Trades':<10}")
print("-" * 50)
for symbol, data in sorted(stats.items(), key=lambda x: x[1]['realized_pnl'], reverse=True):
    if data['trades'] > 0:
        print(f"{symbol:<20} | {data['realized_pnl']:<15.4f} | {data['trades']:<10}")
