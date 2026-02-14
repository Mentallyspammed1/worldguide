 {
    "name": "ema_alignment",
    "description": "EMA alignment scoring for trend direction",
    "code": "\"ema_short = indicator_calc.calculate_ema(config['ema_short_period'])\\nema_long = indicator_calc.calculate_ema(config['ema_long_period'])\\nalignment = 0\\nif ema_short.iloc[-1] > ema_long.iloc[-1] and df['close'].iloc[-1] > ema_short.iloc[-1]:\\n    alignment = 1\\nelif ema_short.iloc[-1] < ema_long.iloc[-1] and df['close'].iloc[-1] < ema_short.iloc[-1]:\\n    alignment = -1\\nscore = weights['ema_alignment'] * alignment\""
  },
  {
    "name": "stoch_rsi_signal",
    "description": "Stochastic RSI oversold/overbought crossover detection",
    "code": "\"stoch_rsi = indicator_calc.calculate_stoch_rsi()\\nk = stoch_rsi['k'].iloc[-1] if not stoch_rsi.empty else None\\nd = stoch_rsi['d'].iloc[-1] if not stoch_rsi.empty else None\\nsignal = 0\\nif k is not None and d is not None:\\n    if k < config['stoch_rsi_oversold_threshold'] and k > d:\\n        signal = weights['stoch_rsi']  # bullish\\n    elif k > config['stoch_rsi_overbought_threshold'] and k < d:\\n        signal = -weights['stoch_rsi']  # bearish\""
  },
  {
    "name": "volume_confirmation",
    "description": "Volume spike confirmation",
    "code": "\"volume_ma = df['volume'].rolling(config['volume_ma_period']).mean()\\nvolume_now = df['volume'].iloc[-1]\\nvolume_avg = volume_ma.iloc[-1]\\nif volume_avg and volume_now > volume_avg * config['volume_confirmation_multiplier']:\\n    signal_strength = weights['volume_confirmation']\""
  },
  {
    "name": "macd_crossover",
    "description": "MACD bullish/bearish crossover",
    "code": "\"macd_df = indicator_calc.calculate_macd()\\nmacd_line = macd_df['macd'].iloc[-1] if not macd_df.empty else None\\nsignal_line = macd_df['signal'].iloc[-1] if not macd_df.empty else None\\nsignal = 0\\nif macd_line is not None and signal_line is not None:\\n    if macd_line > signal_line and macd_line > 0:\\n        signal = weights['macd']  # bullish\\n    elif macd_line < signal_line and macd_line < 0:\\n        signal = -weights['macd']  # bearish\""
  },
  {
    "name": "order_book_wall_detection",
    "description": "Detect bullish/bearish walls in order book",
    "code": "\"walls = order_book_analyzer.analyze_order_book_walls(order_book, current_price)\\nhas_bullish_wall, has_bearish_wall = walls[0], walls[1]\\nsignal_score = 0\\nif has_bullish_wall:\\n    signal_score += config['order_book_support_confidence_boost']\\nif has_bearish_wall:\\n    signal_score -= config['order_book_resistance_confidence_boost']\""
  },
  {
    "name": "rsi_signal",
    "description": "RSI oversold/overbought levels",
    "code": "\"rsi_series = indicator_calc.calculate_rsi()\\nrsi_val = rsi_series.iloc[-1] if not rsi_series.empty else None\\nsignal_strength = 0\\nif rsi_val:\\n    if rsi_val < 30:\\n        signal_strength = weights['rsi']  # bullish\\n    elif rsi_val > 70:\\n        signal_strength = -weights['rsi']  # bearish\""
  },
  {
    "name": "atr_based_stoploss_takeprofit",
    "description": "Calculate stop loss and take profit based on ATR",
    "code": "\"atr = indicator_calc.calculate_atr().iloc[-1] if not indicator_calc.calculate_atr().empty else 0\\nsl = current_price - (Decimal(atr) * Decimal(config['stop_loss_multiple'])) if signal_type == 'buy' else current_price + (Decimal(atr) * Decimal(config['stop_loss_multiple']))\\ntp = current_price + (Decimal(atr) * Decimal(config['take_profit_multiple'])) if signal_type == 'buy' else current_price - (Decimal(atr) * Decimal(config['take_profit_multiple']))\""
  },
  {
    "name": "momentum_trend_strength",
    "description": "Use momentum moving averages for trend strength",
    "code": "\"momentum = df['close'].diff(config['momentum_period'])\\nmomentum_ma_short = momentum.rolling(config['momentum_ma_short']).mean()\\nmomentum_ma_long = momentum.rolling(config['momentum_ma_long']).mean()\\ntrend = 'neutral'\\nstrength = 0\\nif momentum_ma_short.iloc[-1] > momentum_ma_long.iloc[-1]:\\n    trend = 'uptrend'\\n    strength = abs(momentum_ma_short.iloc[-1] - momentum_ma_long.iloc[-1]) / atr_value\\nelif momentum_ma_short.iloc[-1] < momentum_ma_long.iloc[-1]:\\n    trend = 'downtrend'\\n    strength = abs(momentum_ma_long.iloc[-1] - momentum_ma_short.iloc[-1]) / atr_value\""
  },
  {
    "name": "stochastic_oscillator_signal",
    "description": "Stochastic oscillator crossovers",
    "code": "\"stoch_osc = indicator_calc.calculate_stochastic_oscillator()\\nk = stoch_osc['k'].iloc[-1]\\nd = stoch_osc['d'].iloc[-1]\\nsignal_strength = 0\\nif k < 20 and k > d:\\n    signal_strength = weights['stochastic_oscillator']  # bullish\\nelif k > 80 and k < d:\\n    signal_strength = -weights['stochastic_oscillator']  # bearish\""
  },
  {
    "name": "supertrend_signal",
    "description": "Use supertrend direction as signal",
    "code": "\"supertrend = indicator_calc.calculate_supertrend()\\ndirection = supertrend['direction'].iloc[-1]\\nsignal_strength = 0\\nif direction == 1:\\n    signal_strength = weights['supertrend']  # bullish\\nelif direction == -1:\\n    signal_strength = -weights['supertrend']  # bearish\""
  },
  {
    "name": "trend_confirmation_with_adx",
    "description": "Use ADX to confirm trend strength",
    "code": "\"adx_val = indicator_calc.calculate_adx()\\nif adx_val > 25:\\n    # confirm other trend signals\\n    confidence_boost = 0.1\""
  },
  {
    "name": "bollinger_band_breakout",
    "description": "Price crossing Bollinger Bands upper or lower",
    "code": "\"bb = indicator_calc.calculate_bollinger_bands()\\nprice = df['close'].iloc[-1]\\nif price > bb['upper'].iloc[-1]:\\n    signal_strength = -weights['bollinger_bands']  # bearish breakout\\nelif price < bb['lower'].iloc[-1]:\\n    signal_strength = weights['bollinger_bands']  # bullish breakout\""
  },
  {
    "name": "vortex_indicator_signal",
    "description": "Vortex indicator bullish/bearish cross",
    "code": "\"vortex = indicator_calc.calculate_vortex()\\nif vortex['vi_plus'].iloc[-1] > vortex['vi_minus'].iloc[-1]:\\n    signal_strength = weights['vortex']  # bullish\\nelif vortex['vi_plus'].iloc[-1] < vortex['vi_minus'].iloc[-1]:\\n    signal_strength = -weights['vortex']  # bearish\""
  },
  {
    "name": "fibonacci_retracement_levels",
    "description": "Calculate fib retracement and check if price is near support or resistance",
    "code": "\"fib_levels = sr_analyzer.calculate_fibonacci_retracement(high, low, current_price)\\nfor label, level in fib_levels.items():\\n    if abs(current_price - level) / current_price < 0.005:\\n        if level < current_price:\\n            signal_strength += weights.get('fib_retracement', 0)  # support\\n        else:\\n            signal_strength -= weights.get('fib_retracement', 0)  # resistance\""
  },
  {
    "name": "adx_di_based_signal",
    "description": "Directional Movement Index based signal using +DI and -DI",
    "code": "\"df_adx = pd.DataFrame()\\ndf_adx['+DM'] = ...  # calculated as per indicator_calc.calculate_adx\\ndf_adx['-DM'] = ...\\nplus_di = df_adx['+DI'].iloc[-1]\\nminus_di = df_adx['-DI'].iloc[-1]\\nsignal_strength = 0\\nif plus_di > minus_di:\\n    signal_strength = weights['adx']\\nelif minus_di > plus_di:\\n    signal_strength = -weights['adx']\""
  },
  {
    "name": "risk_reward_calculation",
    "description": "Calculate risk reward ratio based on SL and TP",
    "code": "\"if stop_loss and take_profit and signal_type != SignalType.HOLD:\\n    if signal_type == SignalType.BUY:\\n        risk = abs(current_price - stop_loss)\\n        reward = abs(take_profit - current_price)\\n    else:\\n        risk = abs(stop_loss - current_price)\\n        reward = abs(current_price - take_profit)\\n    risk_reward_ratio = reward / risk if risk > 0 else None\""
  },
  {
    "name": "order_book_liquidity_cluster_detection",
    "description": "Detect price proximity to liquidity clusters",
    "code": "\"clusters = order_book_analyzer.find_liquidity_clusters(order_book)\\nfor p, qty in clusters['bids']:\\n    if (current_price - p) / current_price < 0.002 and current_price > p:\\n        signal_score += 0.4  # bullish\\nfor p, qty in clusters['asks']:\\n    if (p - current_price) / current_price < 0.002 and current_price < p:\\n        signal_score -= 0.4  # bearish\""
  },
  {
    "name": "trailing_stop_loss_update",
    "description": "Update trailing stop loss based on Chandelier Exit",
    "code": "\"ce = indicator_values.get('chandelier_exit', {})\\nif signal_type == SignalType.BUY and ce.get('long'):\n    if trailing_stop is None or ce['long'] > trailing_stop:\n        trailing_stop = ce['long']\nelif signal_type == SignalType.SELL and ce.get('short'):\n    if trailing_stop is None or ce['short'] < trailing_stop:\n        trailing_stop = ce['short']\""
  },
  {
    "name": "macd_divergence_detection",
    "description": "Detect bullish or bearish MACD divergence",
    "code": "\"macd_df = indicator_calc.calculate_macd()\\nprices = df['close']\\nhist = macd_df['histogram']\\nif prices.iloc[-2] > prices.iloc[-1] and hist.iloc[-2] < hist.iloc[-1]:\\n    divergence = 'bullish'\\nelif prices.iloc[-2] < prices.iloc[-1] and hist.iloc[-2] > hist.iloc[-1]:\\n    divergence = 'bearish'\\nelse:\\n    divergence = None\""
  }
]

  {
    "fix": "Add missing NEON_CYAN definition for consistent coloring",
    "code_snippet": "\"NEON_CYAN = Fore.CYAN\""
  },
  {
    "fix": "In DataValidator.validate_dataframe, convert 'start_time' column to datetime if needed",
    "code_snippet": "if 'start_time' in df.columns:\n    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):\n        df['start_time'] = pd.to_datetime(df['start_time'])"
  },
  {
    "fix": "In RiskManager.calculate_position_size align quantity using floor division and rounding down",
    "code_snippet": "final_size = (final_size // qty_step) * qty_step"
  },
  {
    "upgrade": "Add async support in APIClient.make_request for better concurrency",
    "code_snippet": "import asyncio\nimport aiohttp\n\nasync def make_request_async(self, method: str, endpoint: str, params: dict = None):\n    async with aiohttp.ClientSession() as session:\n        # Implement similar logic as synchronous with retries\n        pass"
  },
  {
    "fix": "In TradingAnalyzer.generate_trading_signal, fix rounding error with Decimal by quantizing stop_loss and take_profit",
    "code_snippet": "if stop_loss and take_profit:\n    stop_loss = stop_loss.quantize(Decimal('0.00001'))\n    take_profit = take_profit.quantize(Decimal('0.00001'))"
  },
  {
    "upgrade": "Implement database connection pooling in DatabaseManager to improve performance",
    "code_snippet": "import sqlite3\nfrom sqlite3 import Connection\n\nclass DatabaseManager:\n    def __init__(self, db_path: str):\n        self.db_path = db_path\n        self.conn = sqlite3.connect(db_path, check_same_thread=False)\n        self.conn.execute('PRAGMA journal_mode=WAL;')\n        self._ensure_db_exists()\n    def _get_conn(self) -> Connection:\n        return self.conn"
  },
  {
    "fix": "Handle edge case in IndicatorCalculator.calculate_ema_alignment_series to avoid empty DataFrame",
    "code_snippet": "if ema_short.empty or ema_long.empty:\n    return pd.Series(dtype=float)"
  },
  {
    "fix": "Add exception handling in SignalHistoryTracker.sync_with_exchange to avoid crashing",
    "code_snippet": "try:\n    # existing sync code\nexcept Exception as e:\n    self.logger.error(f\"Error syncing with exchange: {e}\")"
  },
  {
    "upgrade": "Add logging of API rate limit remaining in APIClient.make_request",
    "code_snippet": "if 'X-RateLimit-Remaining' in response.headers:\n    self.logger.debug(f\"API Rate Limit Remaining: {response.headers['X-RateLimit-Remaining']}\")"
  },
  {
    "fix": "In RiskManager.calculate_position_size, correctly apply rounding for final_size with Decimal.quantize and rounding=ROUND_DOWN",
    "code_snippet": "final_size = (final_size / qty_step).quantize(Decimal('1'), rounding=decimal.ROUND_DOWN) * qty_step"
  },
  {
    "upgrade": "Add caching for fetch_fee_rates in APIClient to reduce redundant calls",
    "code_snippet": "def fetch_fee_rates(self, symbol: str) -> tuple[Decimal, Decimal]:\n    if hasattr(self, '_fee_cache') and symbol in self._fee_cache:\n        return self._fee_cache[symbol]\n    res = self.make_request('GET', '/v5/contract/fee-rate', {'symbol': symbol})\n    if res and res.get('retCode') == 0:\n        maker = Decimal(str(res['result'].get('makerFeeRate', '0')))\n        taker = Decimal(str(res['result'].get('takerFeeRate', '0')))\n        self._fee_cache[symbol] = (maker, taker)\n        return maker, taker\n    return Decimal('0'), Decimal('0')"
  },
  {
    "fix": "In TradingAnalyzer._calculate_ema_alignment_series add fillna(0) after comparison to avoid NaNs",
    "code_snippet": "alignment[(ema_short > ema_long) & (self.df['close'] > ema_short)] = 1.0\nalignment = alignment.fillna(0.0)"
  },
  {
    "upgrade": "Add graceful websocket reconnect logic with backoff in utility function reconnect_websocket",
    "code_snippet": "def reconnect_websocket(ws, url, on_message):\n    ws.close()\n    time.sleep(1)\n    backoff = 1\n    while True:\n        try:\n            ws = connect_websocket(url, on_message)\n            return ws\n        except Exception:\n            time.sleep(backoff)\n            backoff = min(backoff * 2, 60)"
  },
  {
    "fix": "Add type hints for all methods missing them to improve readability and static analysis",
    "code_snippet": "def generate_signal(self, indicator_values: dict[str, Any], market_regime: MarketRegime, current_price: Decimal, atr_value: Decimal) -> TradingSignal:"
  },
  {
    "upgrade": "Add method in NotificationSystem to send combined notification (email + webhook + sms) in one call",
    "code_snippet": "def send_combined_notification(self, subject: str, message: str, payload: dict, sms_message: str) -> None:\n    self.send_email(subject, message)\n    self.send_webhook(payload)\n    self.send_sms(sms_message)"
  },
  {
    "fix": "In DataValidator.validate_dataframe, change dropna() to dropna(how='any', inplace=True) to avoid copy warning",
    "code_snippet": "df.dropna(how='any', inplace=True)"
  },
  {
    "upgrade": "Add function to perform database vacuum periodically to optimize database file size",
    "code_snippet": "def vacuum_database(self):\n    conn = sqlite3.connect(self.db_path)\n    conn.execute('VACUUM')\n    conn.close()"
  },
  {
    "fix": "Add check for zero division in SupportResistanceAnalyzer.calculate_fibonacci_retracement before dividing diff",
    "code_snippet": "if diff <= 0:\n    self.logger.warning(f\"{NEON_YELLOW}High less or equal to Low, skipping Fibonacci retracement.{RESET}\")\n    return {}"
  },
  {
    "upgrade": "Enhance SignalGenerator.generate_signal to accept previous indicator values for delta calculations",
    "code_snippet": "def generate_signal(self, indicator_values: dict[str, Any], previous_values: dict[str, Any], market_regime: MarketRegime, current_price: Decimal, atr_value: Decimal) -> TradingSignal:"
  },
  {
    "fix": "Fix potential division by zero in MarketRegimeDetector._calculate_atr",
    "code_snippet": "atr = (np.mean(tr[-self.atr_period:]) if len(tr) >= self.atr_period and np.any(tr[-self.atr_period:] > 0) else np.mean(tr))"
  }
]

      "id": 1,
      "description": "Add a loading spinner during data fetch in UI",
      "code": "def show_loading_spinner():\n    import itertools, sys, threading, time\n    done = False\n    def animate():\n        for c in itertools.cycle(['|', '/', '-', '\\\\']):\n            if done:\n                break\n            sys.stdout.write(f'\\rLoading {c}')\n            sys.stdout.flush()\n            time.sleep(0.1)\n        sys.stdout.write('\\rDone!     \\n')\n    t = threading.Thread(target=animate)\n    t.start()\n    return lambda: setattr(globals(), 'done', True)"
    },
    {
      "id": 2,
      "description": "Format PnL output with color gradients for UI",
      "code": "def format_pnl_output(pnl: Decimal) -> str:\n    if pnl > 0:\n        color = NEON_GREEN\n    elif pnl < 0:\n        color = NEON_RED\n    else:\n        color = NEON_WHITE\n    return f'{color}${pnl:.2f}{RESET}'"
    },
    {
      "id": 3,
      "description": "Upgrade signal display with confidence bar",
      "code": "def confidence_bar(confidence: float, length: int = 20) -> str:\n    filled_length = int(length * confidence)\n    bar = NEON_GREEN + '█' * filled_length + NEON_WHITE + '-' * (length - filled_length) + RESET\n    return bar\n\ndef display_signal_with_confidence(signal: TradingSignal):\n    bar = confidence_bar(signal.confidence)\n    return f'Signal: {signal.signal_type.value.upper()} {bar} Confidence: {signal.confidence:.2f}'"
    },
    {
      "id": 4,
      "description": "Add timestamp formatted in local timezone to outputs",
      "code": "def format_timestamp(timestamp: float, tz: ZoneInfo = TIMEZONE) -> str:\n    dt = datetime.fromtimestamp(timestamp, tz)\n    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')"
    },
    {
      "id": 5,
      "description": "Add JSON output for trading signal with all metadata",
      "code": "def signal_to_json(signal: TradingSignal) -> str:\n    output = {\n        'type': signal.signal_type.value if signal.signal_type else None,\n        'confidence': signal.confidence,\n        'conditions_met': signal.conditions_met,\n        'stop_loss': str(signal.stop_loss) if signal.stop_loss else None,\n        'take_profit': str(signal.take_profit) if signal.take_profit else None,\n        'timestamp': format_timestamp(signal.timestamp),\n        'symbol': signal.symbol,\n        'timeframe': signal.timeframe,\n        'position_size': str(signal.position_size) if signal.position_size else None,\n        'risk_reward_ratio': signal.risk_reward_ratio\n    }\n    return json.dumps(output, indent=2)"
    },
    {
      "id": 6,
      "description": "Add progress percentage output for backtesting",
      "code": "def display_backtest_progress(current: int, total: int) -> None:\n    percent = (current / total) * 100\n    bar_length = 30\n    filled_length = int(bar_length * current // total)\n    bar = NEON_GREEN + '█' * filled_length + NEON_WHITE + '-' * (bar_length - filled_length) + RESET\n    sys.stdout.write(f'\\rBacktesting: |{bar}| {percent:.2f}% Complete')\n    sys.stdout.flush()\n    if current == total:\n        print()"
    },
    {
      "id": 7,
      "description": "Add summarized indicator output with color highlights",
      "code": "def summarized_indicator_output(indicators: dict[str, Any]) -> str:\n    lines = []\n    for name, val in indicators.items():\n        if val is None or (isinstance(val, float) and np.isnan(val)):\n            continue\n        color = NEON_WHITE\n        try:\n            val_float = float(val) if not isinstance(val, dict) else None\n            if val_float is not None:\n                if val_float > 0:\n                    color = NEON_GREEN\n                elif val_float < 0:\n                    color = NEON_RED\n                else:\n                    color = NEON_YELLOW\n            lines.append(f'{name.upper()}: {color}{val}{RESET}')\n        except Exception:\n            lines.append(f'{name.upper()}: {NEON_WHITE}{val}{RESET}')\n    return ' | '.join(lines)"
    },
    {
      "id": 8,
      "description": "Add support/resistance level outputs in UI",
      "code": "def display_support_resistance(supports: list[tuple[str, Decimal]], resistances: list[tuple[str, Decimal]]) -> str:\n    sup_str = ', '.join([f'{label}@${value:.4f}' for label, value in supports]) or 'None'\n    res_str = ', '.join([f'{label}@${value:.4f}' for label, value in resistances]) or 'None'\n    return f'Supports: {NEON_GREEN}{sup_str}{RESET} | Resistances: {NEON_RED}{res_str}{RESET}'"
    },
    {
      "id": 9,
      "description": "Add condition met list with bullet points in outputs",
      "code": "def conditions_to_text(conditions: list[str]) -> str:\n    if not conditions:\n        return 'None'\n    bullet = '\\u2022'\n    return '\\n'.join([f'{bullet} {cond}' for cond in conditions])"
    },
    {
      "id": 10,
      "description": "Add compact indicator summary in json format",
      "code": "def indicators_to_compact_json(indicators: dict[str, Any]) -> dict:\n    output = {}\n    for key, val in indicators.items():\n        if isinstance(val, (float, int, str)):\n            output[key] = val\n        elif isinstance(val, dict):\n            output[key] = {k: v for k, v in val.items() if isinstance(v, (float, int, str))}\n        elif isinstance(val, pd.Series) and not val.empty:\n            output[key] = float(val.iloc[-1])\n    return output"
    },
    {
      "id": 11,
      "description": "Display trailing stop updates with color-coded notifications",
      "code": "def display_trailing_stop_update(symbol: str, old_sl: Decimal, new_sl: Decimal) -> str:\n    if new_sl > old_sl:\n        color = NEON_GREEN\n        direction = 'Increased'\n    elif new_sl < old_sl:\n        color = NEON_RED\n        direction = 'Decreased'\n    else:\n        color = NEON_WHITE\n        direction = 'Unchanged'\n    return f'{color}Trailing Stop {direction} for {symbol}: {old_sl:.4f} -> {new_sl:.4f}{RESET}'"
    },
    {
      "id": 12,
      "description": "Add position size output with rounding to 4 decimals",
      "code": "def format_position_size(size: Decimal) -> str:\n    return f'{size.quantize(Decimal(\"0.0001\"))}'"
    },
    {
      "id": 13,
      "description": "Output open positions list with pnl and risk/reward",
      "code": "def display_open_positions(signals: dict[int, SignalHistory], current_price: Decimal) -> str:\n    lines = []\n    for sid, signal in signals.items():\n        if signal.signal_type == SignalType.BUY:\n            unrealized_pnl = (current_price - signal.entry_price) * signal.quantity\n        else:\n            unrealized_pnl = (signal.entry_price - current_price) * signal.quantity\n        r_str = f'R:R={float(signal.risk_reward_ratio):.2f}' if signal.risk_reward_ratio else 'R:R=N/A'\n        lines.append(f'ID:{sid} {signal.signal_type.value.upper()} {signal.symbol} Qty:{signal.quantity:.4f} Entry:${signal.entry_price:.4f} PnL:${unrealized_pnl:.2f} {r_str}')\n    return '\\n'.join(lines)"
    },
    {
      "id": 14,
      "description": "Add detailed JSON output for active positions including stop loss and take profit",
      "code": "def active_positions_to_json(signals: dict[int, SignalHistory]) -> str:\n    results = []\n    for sid, s in signals.items():\n        results.append({\n            'id': sid,\n            'symbol': s.symbol,\n            'signal_type': s.signal_type.value,\n            'entry_price': str(s.entry_price),\n            'quantity': str(s.quantity),\n            'stop_loss': str(s.stop_loss) if s.stop_loss else None,\n            'take_profit': str(s.take_profit) if s.take_profit else None,\n            'trailing_sl': str(s.trailing_sl) if s.trailing_sl else None,\n            'highest_price': str(s.highest_price) if s.highest_price else None,\n            'lowest_price': str(s.lowest_price) if s.lowest_price else None,\n            'profit_loss': str(s.profit_loss) if s.profit_loss else None,\n            'net_pnl': str(s.net_pnl) if s.net_pnl else None,\n            'exit_reason': s.exit_reason,\n            'market_regime': s.market_regime.value if s.market_regime else None\n        })\n    return json.dumps(results, indent=2)"
    },
    {
      "id": 15,
      "description": "Color-code and format support/resistance nearby levels sorted by proximity",
      "code": "def nearest_levels_ui(current_price: Decimal, supports: list[tuple[str, Decimal]], resistances: list[tuple[str, Decimal]]) -> str:\n    def format_level(label, val):\n        diff = abs((val - current_price) / current_price) * 100\n        color = NEON_GREEN if val < current_price else NEON_RED\n        return f'{color}{label}@${val:.4f} ({diff:.2f}%) {RESET}'\n    sup_lines = [format_level(l, v) for l, v in sorted(supports, key=lambda x: abs((current_price - x[1])/current_price))]\n    res_lines = [format_level(l, v) for l, v in sorted(resistances, key=lambda x: abs((x[1] - current_price)/current_price))]\n    return 'Supports: ' + ', '.join(sup_lines) + '\\nResistances: ' + ', '.join(res_lines)"
    },
    {
      "id": 16,
      "description": "Add compact indicator output for terminal dashboard",
      "code": "def terminal_indicator_dashboard(indicators: dict[str, float]) -> str:\n    parts = []\n    for name in ['rsi', 'mfi', 'cci', 'fve', 'stc', 'cmo']:\n        val = indicators.get(name, None)\n        if val is not None and not pd.isna(val):\n            parts.append(f'{name.upper()}: {val:.2f}')\n    return ' | '.join(parts)"
    },
    {
      "id": 17,
      "description": "Add order book imbalance output with colors",
      "code": "def order_book_imbalance_ui(imbalance: float) -> str:\n    if imbalance > 0.3:\n        color = NEON_GREEN\n        state = \"Strong Buy\"\n    elif imbalance < -0.3:\n        color = NEON_RED\n        state = \"Strong Sell\"\n    else:\n        color = NEON_YELLOW\n        state = \"Neutral\"\n    return f'Order Book Imbalance: {color}{imbalance:.2f} ({state}){RESET}'"
    },
    {
      "id": 18,
      "description": "Show notification summary with key indicators",
      "code": "def notification_summary(signal: TradingSignal, indicators: dict[str, Any]) -> str:\n    parts = [f'Signal: {signal.signal_type.value.upper()} {signal.symbol}', f'Confidence: {signal.confidence:.2f}']\n    key_indicators = ['rsi', 'mfi', 'atr', 'momentum_ma_short', 'momentum_ma_long']\n    for k in key_indicators:\n        v = indicators.get(k, None)\n        if v is not None and not pd.isna(v):\n            parts.append(f'{k.upper()}: {v:.2f}')\n    return ' | '.join(parts)"
    },
    {
      "id": 19,
      "description": "Provide JSON array output of recent trade performance metrics",
      "code": "def performance_metrics_to_json(metrics_list: list[PerformanceMetrics]) -> str:\n    results = []\n    for m in metrics_list:\n        results.append({\n            'total_trades': m.total_trades,\n            'winning_trades': m.winning_trades,\n            'losing_trades': m.losing_trades,\n            'win_rate': m.win_rate,\n            'profit_factor': m.profit_factor,\n            'max_drawdown': m.max_drawdown,\n            'sharpe_ratio': m.sharpe_ratio,\n            'total_profit': str(m.total_profit),\n            'total_loss': str(m.total_loss),\n            'net_profit': str(m.net_profit),\n            'average_win': str(m.average_win),\n            'average_loss': str(m.average_loss)\n        })\n    return json.dumps(results, indent=2)"
    },
    {
      "id": 20,
      "description": "Add clear console output separator",
      "code": "def print_separator():\n    print(f'{NEON_CYAN}{\"-\" * 60}{RESET}')"
    }
  ]