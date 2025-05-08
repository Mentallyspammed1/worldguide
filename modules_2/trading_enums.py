# File: trading_enums.py
from enum import Enum

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"

class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2
```

```python
