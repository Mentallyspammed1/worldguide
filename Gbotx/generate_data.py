
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_data(start_date, num_candles, interval_minutes):
    data = []
    current_time = start_date
    price = 100.0  # Starting price

    for i in range(num_candles):
        open_price = price
        close_price = open_price + np.random.uniform(-1, 1) * 0.5
        high_price = max(open_price, close_price) + np.random.uniform(0, 0.2)
        low_price = min(open_price, close_price) - np.random.uniform(0, 0.2)
        volume = np.random.randint(1000, 10000)

        data.append({
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 3),
            'high': round(high_price, 3),
            'low': round(low_price, 3),
            'close': round(close_price, 3),
            'volume': volume
        })

        price = close_price + np.random.uniform(-0.5, 0.5) # Price drift
        current_time += timedelta(minutes=interval_minutes)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    start_date = datetime(2025, 7, 1, 0, 0, 0)
    num_candles = 1000  # Generate 1000 candles
    interval_minutes = 5  # 5-minute interval

    df = generate_dummy_data(start_date, num_candles, interval_minutes)
    df.to_csv('./data/dummy_data.csv', index=False)
    print(f"Generated {num_candles} candles to ./data/dummy_data.csv")
