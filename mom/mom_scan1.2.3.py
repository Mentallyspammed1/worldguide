```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Momentum Scanner Trading Bot for Bybit (V5 API) - mom_scan1.2.py

This bot scans multiple symbols based on momentum indicators, Ehlers filters (optional),
and other technical analysis tools to identify potential trading opportunities on Bybit's
Unified Trading Account (primarily linear perpetual contracts). It utilizes both
WebSocket for real-time data and REST API for historical data and order management.

Key Features:
- Multi-Symbol Trading: Monitors and trades multiple configured symbols concurrently.
- Configurable Strategy: Parameters per symbol defined via JSON (periods, thresholds, etc.).
- Momentum Indicators: EMA/SuperSmoother, ROC, RSI, ADX, Bollinger Bands.
- Optional Ehlers Filters: Super Smoother, Instantaneous Trendline (Note: sensitive to tuning).
- Volume Analysis: Volume SMA and rolling percentile for volume surge detection.
- Dynamic Thresholds: Adjusts RSI/ROC thresholds based on market volatility (ATR).
- ATR-Based Risk Management: Position sizing, Stop Loss, and Take Profit calculated using ATR.
- Optional Multi-Timeframe Analysis: Filters trades based on trend confirmation on a higher timeframe.
- WebSocket Integration: Low-latency kline updates with robust reconnection logic and initial data population.
- REST API Fallback: Uses REST API for historical data, initial population, and order management.
- Robust Error Handling: Specific Bybit V5 error code checks and comprehensive exception handling.
- Dry Run Mode: Simulates trading logic without placing real orders.
- Detailed Logging: Colored console output (via Colorama) and clean UTF-8 encoded file output.
- Trade Performance Tracking: Thread-safe metrics logging (P&L, win rate, fees, etc.).
- Graceful Shutdown: Handles SIGINT/SIGTERM signals for clean exit, including optional position closing.
- Comprehensive Configuration Validation: Checks config structure, types, and values on startup.
- Dynamic Kline Limit Calculation: Automatically determines required historical data points based on indicator periods.
- Improved NaN Handling: Rigorous checks and filling of NaN values throughout data processing.
- Clear Action Logging: Distinct logging for trade actions (entry, exit attempts).
- Enhanced WebSocket Management: Dedicated thread for connection monitoring and reconnection.
"""

# --- Standard Library Imports ---
import os
import sys
import time
import logging
import json
import argparse
import threading
import math
import signal
import datetime as dt
from queue import Queue, Empty
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, List, Union

# --- Third-party Library Imports ---
# Attempt to import required libraries and provide helpful