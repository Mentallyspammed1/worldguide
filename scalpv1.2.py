import logging
import os
import time

import ccxt
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

logger = logging.getLogger("ScalpingBot")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler("scalping_bot.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def retry_api_call(max_retries=3, initial_delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded:
                    logger.warning(f"Rate limit exceeded, retrying in {delay} seconds... (Retry {retries + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                except ccxt.NetworkError as e:
                    logger.error(f"Network error during API call: {e}. Retrying in {delay} seconds... (Retry {retries + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error during API call: {e}. (Retry {retries + 1}/{max_retries})")
                    if "Order does not exist" in str(e) or "Order not found" in str(e):
                        logger.warning("Order not found/does not exist, likely filled or cancelled. Returning None.")
                        return None
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                except Exception as e:
                    logger.exception(f"Unexpected error during API call: {e}. (Retry {retries + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
            logger.error(f"Max retries reached for API call {func.__name__}. Aborting.")
            return None

        return wrapper

    return decorator


class ScalpingBot:
    def __init__(self, config_file="config.yaml"):
        self.config = self.load_config(config_file)
        self.validate_config()

        self.exchange = self.initialize_exchange()
        if not self.exchange:
            exit(1)  # Exit if exchange initialization fails

        self.market = self.exchange.market(self.symbol)
        self.open_positions = []
        self.iteration = 0
        self.daily_pnl = 0.0

    def load_config(self, config_file):
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Error loading config file: {e}. Exiting.")
            exit(1)

    def validate_config(self):
        def validate_section(section_name, required_params, param_types={}, param_ranges={}):
            if section_name not in self.config:
                raise ValueError(f"Missing '{section_name}' section in config.yaml")
            for param, param_type in param_types.items():
                if param not in self.config[section_name]:
                    raise ValueError(f"Missing '{param}' in config.yaml {section_name} section")
                if not isinstance(self.config[section_name][param], param_type):
                    raise ValueError(f"'{param}' must be of type {param_type}")
            for param, (min_val, max_val) in param_ranges.items():
                if param not in self.config[section_name]:
                    raise ValueError(f"Missing '{param}' in config.yaml {section_name} section")
                value = self.config[section_name][param]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"'{param}' must be between {min_val} and {max_val}")
            for param in required_params:
                if param not in self.config[section_name]:
                    raise ValueError(f"Missing '{param}' in config.yaml {section_name} section")

        required_sections = ["exchange", "trading", "order_book", "indicators", "risk_management"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing '{section}' section in config.yaml")

        validate_section("exchange", ["exchange_id"], param_types={"exchange_id": str})
        if not self.config["exchange"]["exchange_id"]:
            raise ValueError("'exchange_id' must be a non-empty string")

        validate_section(
            "trading",
            ["symbol", "simulation_mode", "entry_order_type", "limit_order_offset_buy", "limit_order_offset_sell"],
            param_types={"symbol": str, "simulation_mode": bool, "entry_order_type": str},
            param_ranges={
                "limit_order_offset_buy": (0, float('inf')),
                "limit_order_offset_sell": (0, float('inf')),
            },
        )
        if not self.config["trading"]["symbol"]:
            raise ValueError("'symbol' must be a non-empty string")
        if self.config["trading"]["entry_order_type"] not in ["market", "limit"]:
            raise ValueError("'entry_order_type' must be 'market' or 'limit'")


        validate_section(
            "order_book",
            ["depth", "imbalance_threshold"],
            param_types={"depth": int},
            param_ranges={"depth": (1, float('inf')), "imbalance_threshold": (0, float('inf'))},
        )

        validate_section(
            "indicators",
            ["volatility_window", "volatility_multiplier", "ema_period", "rsi_period", "macd_short_period", "macd_long_period", "macd_signal_period", "stoch_rsi_period"],
            param_types={
                "volatility_window": int,
                "volatility_multiplier": (int, float),
                "ema_period": int,
                "rsi_period": int,
                "macd_short_period": int,
                "macd_long_period": int,
                "macd_signal_period": int,
                "stoch_rsi_period": int,
            },
            param_ranges={
                "volatility_window": (1, float('inf')),
                "volatility_multiplier": (0, float('inf')),
                "ema_period": (1, float('inf')),
                "rsi_period": (1, float('inf')),
                "macd_short_period": (1, float('inf')),
                "macd_long_period": (1, float('inf')),
                "macd_signal_period": (1, float('inf')),
                "stoch_rsi_period": (1, float('inf')),
            },
        )
        if self.config["indicators"]["macd_short_period"] >= self.config["indicators"]["macd_long_period"]:
            raise ValueError("'macd_short_period' must be less than 'macd_long_period'")

        validate_section(
            "risk_management",
            ["order_size_percentage", "stop_loss_percentage", "take_profit_percentage", "max_open_positions", "time_based_exit_minutes", "trailing_stop_loss_percentage"],
            param_types={
                "order_size_percentage": (int, float),
                "stop_loss_percentage": (int, float),
                "take_profit_percentage": (int, float),
                "max_open_positions": int,
                "time_based_exit_minutes": int,
                "trailing_stop_loss_percentage": (int, float),
            },
            param_ranges={
                "order_size_percentage": (0, 1),
                "stop_loss_percentage": (0, 1),
                "take_profit_percentage": (0, 1),
                "max_open_positions": (1, float('inf')),
                "time_based_exit_minutes": (1, float('inf')),
                "trailing_stop_loss_percentage": (0, 1),
            },
        )


        if "logging_level" in self.config:
            log_level = self.config["logging_level"].upper()
            if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.setLevel(getattr(logging, log_level))
            else:
                logger.warning(f"Invalid logging level '{log_level}' in config. Using default (DEBUG).")

        # Assign config values to instance variables for easier access
        self.exchange_id = self.config["exchange"]["exchange_id"]
        self.symbol = self.config["trading"]["symbol"]
        self.simulation_mode = self.config["trading"]["simulation_mode"]
        self.entry_order_type = self.config["trading"]["entry_order_type"]
        self.limit_order_offset_buy = self.config["trading"]["limit_order_offset_buy"]
        self.limit_order_offset_sell = self.config["trading"]["limit_order_offset_sell"]
        self.order_book_depth = self.config["order_book"]["depth"]
        self.imbalance_threshold = self.config["order_book"]["imbalance_threshold"]
        self.volatility_window = self.config["indicators"]["volatility_window"]
        self.volatility_multiplier = self.config["indicators"]["volatility_multiplier"]
        self.ema_period = self.config["indicators"]["ema_period"]
        self.rsi_period = self.config["indicators"]["rsi_period"]
        self.macd_short_period = self.config["indicators"]["macd_short_period"]
        self.macd_long_period = self.config["indicators"]["macd_long_period"]
        self.macd_signal_period = self.config["indicators"]["macd_signal_period"]
        self.stoch_rsi_period = self.config["indicators"]["stoch_rsi_period"]
        self.order_size_percentage = self.config["risk_management"]["order_size_percentage"]
        self.stop_loss_pct = self.config["risk_management"]["stop_loss_percentage"]
        self.take_profit_pct = self.config["risk_management"]["take_profit_percentage"]
        self.max_open_positions = self.config["risk_management"]["max_open_positions"]
        self.time_based_exit_minutes = self.config["risk_management"]["time_based_exit_minutes"]
        self.trailing_stop_loss_percentage = self.config["risk_management"]["trailing_stop_loss_percentage"]

        logger.info("Configuration validated successfully.")

    def initialize_exchange(self):
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class(
                {
                    "apiKey": os.getenv("BYBIT_API_KEY"),
                    "secret": os.getenv("BYBIT_API_SECRET"),
                    "options": {"defaultType": "linear"},
                    "enableRateLimit": True,
                    "recvWindow": 10000,
                }
            )
            exchange.load_markets()
            logger.info(f"Connected to {self.exchange_id.upper()} successfully.")
            if self.symbol not in exchange.markets:
                logger.error(f"Symbol {self.symbol} not found on {self.exchange_id}. Available symbols: {list(exchange.markets.keys())}")
                return None  # Return None to indicate failure
            return exchange
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication Error: Please check your API keys. {e}")
            return None
        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange Not Available: {self.exchange_id} might be down or unreachable. {e}")
            return None
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            return None

    @retry_api_call()
    def fetch_market_price(self):
        ticker = self.exchange.fetch_ticker(self.symbol)
        price = ticker.get("last") or ticker.get("close")
        if price is not None:
            logger.debug(f"Fetched market price: {price}")
            return float(price)
        logger.warning(f"Market price ('last' or 'close') unavailable in ticker: {ticker}")
        return None

    @retry_api_call()
    def fetch_order_book(self):
        orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if bids and asks:
            bid_volume = sum(bid[1] for bid in bids)
            ask_volume = sum(ask[1] for ask in asks)
            imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else float("inf") if bid_volume > 0 else 1.0
            logger.debug(f"Order Book - Bid Vol: {bid_volume:.4f}, Ask Vol: {ask_volume:.4f}, Bid/Ask Ratio: {imbalance_ratio:.2f}")
            return imbalance_ratio
        logger.warning("Order book data (bids or asks) unavailable.")
        return None

    @retry_api_call()
    def fetch_historical_data(self, limit=None, timeframe="1m"):
        if limit is None:
            limit = max(
                self.volatility_window,
                self.ema_period,
                self.rsi_period + 1,
                self.macd_long_period + self.macd_signal_period,
                self.stoch_rsi_period + 6,
                100,
            )

        logger.debug(f"Fetching {limit} historical {timeframe} candles for {self.symbol}...")
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)

        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            if len(df) < limit:
                logger.warning(f"Insufficient historical data fetched. Requested {limit}, got {len(df)}.")
                min_required = max(
                    self.volatility_window,
                    self.ema_period,
                    self.rsi_period + 1,
                    self.macd_long_period,
                    self.stoch_rsi_period + 6,
                )
                if len(df) < min_required:
                    logger.error(f"Fetched data ({len(df)}) is less than minimum required ({min_required}). Cannot proceed with calculations.")
                    return None
            logger.debug(f"Historical data fetched successfully. Shape: {df.shape}")
            return df
        logger.warning("Historical OHLCV data unavailable.")
        return None

    def calculate_volatility(self, data: pd.DataFrame):
        if data is None or len(data) < self.volatility_window:
            logger.warning(f"Insufficient data for volatility calculation (need {self.volatility_window}, got {len(data) if data is not None else 0})")
            return None
        log_returns = np.log(data["close"] / data["close"].shift(1))
        volatility = log_returns.rolling(window=self.volatility_window).std().iloc[-1]
        if pd.isna(volatility):
            logger.warning("Volatility calculation resulted in NaN.")
            return None
        logger.debug(f"Calculated volatility: {volatility:.5f}")
        return volatility

    def calculate_ema(self, data: pd.Series, period: int):
        if data is None or len(data) < period:
            logger.warning(f"Insufficient data for EMA calculation (need {period}, got {len(data) if data is not None else 0})")
            return None
        ema = data.ewm(span=period, adjust=False).mean().iloc[-1]
        if pd.isna(ema):
            logger.warning(f"EMA calculation (period {period}) resulted in NaN.")
            return None
        logger.debug(f"Calculated EMA (period {period}): {ema:.4f}")
        return ema

    def calculate_rsi(self, data: pd.Series, period: int):
        if data is None or len(data) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation (need {period + 1}, got {len(data) if data is not None else 0})")
            return None

        delta = data.diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]

        if pd.isna(rsi_value):
            logger.warning("RSI calculation resulted in NaN.")
            return None
        logger.debug(f"Calculated RSI (period {period}): {rsi_value:.2f}")
        return rsi_value

    def calculate_macd(self, data: pd.Series):
        if data is None or len(data) < self.macd_long_period:
            logger.warning(f"Insufficient data for MACD calculation (need {self.macd_long_period}, got {len(data) if data is not None else 0})")
            return None, None, None

        ema_short = data.ewm(span=self.macd_short_period, adjust=False).mean()
        ema_long = data.ewm(span=self.macd_long_period, adjust=False).mean()

        if ema_short.isnull().all() or ema_long.isnull().all():
            logger.warning("EMA calculation for MACD failed.")
            return None, None, None

        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=self.macd_signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        hist_val = histogram.iloc[-1]

        if pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val):
            logger.warning("MACD calculation resulted in NaN.")
            return None, None, None

        logger.debug(f"MACD: {macd_val:.4f}, Signal: {signal_val:.4f}, Histogram: {hist_val:.4f}")
        return macd_val, signal_val, hist_val

    def calculate_stoch_rsi(self, data: pd.Series, period: int):
        if data is None or len(data) < period + 6:
            logger.warning(f"Insufficient data for Stoch RSI calculation (need {period + 6}, got {len(data) if data is not None else 0})")
            return None, None

        delta = data.diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))

        if rsi_series.isnull().all():
            logger.warning("RSI series calculation for Stoch RSI failed.")
            return None, None

        min_rsi = rsi_series.rolling(window=period).min()
        max_rsi = rsi_series.rolling(window=period).max()
        stoch_rsi_k = 100 * (rsi_series - min_rsi) / (max_rsi - min_rsi)

        k = stoch_rsi_k.rolling(window=3).mean()
        d = k.rolling(window=3).mean()

        k_val = k.iloc[-1]
        d_val = d.iloc[-1]

        if pd.isna(k_val) or pd.isna(d_val):
            logger.warning("Stochastic RSI calculation resulted in NaN.")
            return None, None

        logger.debug(f"Calculated Stochastic RSI (period {period}) - K: {k_val:.2f}, D: {d_val:.2f}")
        return k_val, d_val

    @retry_api_call()
    def fetch_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            quote_currency = self.market["quote"]
            free_balance = balance.get("free", {}).get(quote_currency, 0.0)
            logger.debug(f"Fetched free balance for {quote_currency}: {free_balance}")
            return float(free_balance)
        except Exception as e:
            logger.error(f"Could not retrieve {self.market.get('quote', 'quote currency')} balance: {e}")
            return None

    def calculate_order_size(self, price: float, volatility: float):
        balance = self.fetch_balance()
        if balance is None or balance <= 0 or price <= 0:
            logger.warning(f"Could not calculate order size due to invalid balance ({balance}) or price ({price}).")
            return 0

        base_size_quote = balance * self.order_size_percentage

        if volatility is not None and volatility > 0:
            adjustment_factor = 1 / (1 + (volatility * self.volatility_multiplier))
            adjusted_size_quote = base_size_quote * adjustment_factor
            logger.debug(f"Volatility adjustment factor: {adjustment_factor:.4f}")
        else:
            adjusted_size_quote = base_size_quote
            logger.debug("No volatility adjustment applied.")

        max_size_quote = balance * 0.05
        final_size_quote = min(adjusted_size_quote, max_size_quote)

        final_size_base = final_size_quote / price

        amount_precision = self.market.get("precision", {}).get("amount")
        min_amount = self.market.get("limits", {}).get("amount", {}).get("min")

        if amount_precision is not None:
            final_size_base = self.exchange.amount_to_precision(self.symbol, final_size_base)
            if float(final_size_base) <= 0:
                logger.warning(f"Order size became zero after applying amount precision. Initial quote size: {final_size_quote:.4f}")
                return 0

        final_size_base_float = float(final_size_base)
        if min_amount is not None and final_size_base_float < min_amount:
            logger.warning(f"Calculated order size {final_size_base_float} is below minimum {min_amount}. Adjusting to minimum.")
            min_size_quote = min_amount * price
            if min_size_quote <= max_size_quote:
                final_size_base = self.exchange.amount_to_precision(self.symbol, min_amount)
                final_size_base_float = float(final_size_base)
                logger.info(f"Using minimum order size: {final_size_base} {self.market['base']}")
            else:
                logger.warning(f"Minimum order size ({min_amount}) exceeds risk cap. Skipping trade.")
                return 0

        logger.info(
            f"Calculated order size: {final_size_base} {self.market['base']} "
            f"(Quote Value: ~{final_size_base_float * price:.2f} {self.market['quote']}, "
            f"Balance: {balance:.2f}, Volatility: {volatility:.5f if volatility is not None else 'N/A'})"
        )
        return final_size_base_float

    def compute_trade_signal_score(self, price, indicators: dict, orderbook_imbalance):
        score = 0
        reasons = []

        ema = indicators.get("ema")
        rsi = indicators.get("rsi")
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        stoch_rsi_k = indicators.get("stoch_rsi_k")
        stoch_rsi_d = indicators.get("stoch_rsi_d")

        if orderbook_imbalance is not None:
            if orderbook_imbalance > self.imbalance_threshold:
                score += 1
                reasons.append(f"Order book: strong bid pressure (Ratio: {orderbook_imbalance:.2f} > {self.imbalance_threshold:.2f}).")
            elif orderbook_imbalance < (1 / self.imbalance_threshold):
                score -= 1
                reasons.append(f"Order book: strong ask pressure (Ratio: {orderbook_imbalance:.2f} < {1/self.imbalance_threshold:.2f}).")
            else:
                reasons.append(f"Order book: pressure neutral (Ratio: {orderbook_imbalance:.2f}).")

        if ema is not None:
            if price > ema:
                score += 1
                reasons.append(f"Price > EMA ({price:.2f} > {ema:.2f}) (bullish).")
            elif price < ema:
                score -= 1
                reasons.append(f"Price < EMA ({price:.2f} < {ema:.2f}) (bearish).")
            else:
                reasons.append(f"Price == EMA ({price:.2f}).")

        if rsi is not None:
            if rsi < 30:
                score += 1
                reasons.append(f"RSI < 30 ({rsi:.2f}) (oversold).")
            elif rsi > 70:
                score -= 1
                reasons.append(f"RSI > 70 ({rsi:.2f}) (overbought).")
            else:
                reasons.append(f"RSI neutral ({rsi:.2f}).")

        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                score += 1
                reasons.append(f"MACD > Signal ({macd:.4f} > {macd_signal:.4f}) (bullish).")
            elif macd < macd_signal:
                score -= 1
                reasons.append(f"MACD < Signal ({macd:.4f} < {macd_signal:.4f}) (bearish).")
            else:
                reasons.append(f"MACD == Signal ({macd:.4f}).")

        if stoch_rsi_k is not None and stoch_rsi_d is not None:
            if stoch_rsi_k < 20 and stoch_rsi_d < 20:
                score += 1
                reasons.append(f"Stoch RSI < 20 (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}) (oversold).")
            elif stoch_rsi_k > 80 and stoch_rsi_d > 80:
                score -= 1
                reasons.append(f"Stoch RSI > 80 (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}) (overbought).")
            else:
                reasons.append(f"Stoch RSI neutral (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}).")

        return score, reasons

    @retry_api_call()
    def place_order(
        self,
        side: str,
        order_size: float,
        order_type: str = "market",
        price: float = None,
        stop_loss_price: float = None,
        take_profit_price: float = None,
    ):
        try:
            order_size_str = self.exchange.amount_to_precision(self.symbol, order_size)
            price_str = self.exchange.price_to_precision(self.symbol, price) if price else None
            sl_price_str = self.exchange.price_to_precision(self.symbol, stop_loss_price) if stop_loss_price else None
            tp_price_str = self.exchange.price_to_precision(self.symbol, take_profit_price) if take_profit_price else None

            params = {}
            if stop_loss_price:
                params["stopLoss"] = sl_price_str
            if take_profit_price:
                params["takeProfit"] = tp_price_str

            if self.simulation_mode:
                if order_type == "limit" and price is None:
                    logger.error("[SIMULATION] Limit price required for limit orders.")
                    return None

                simulated_fill_price = price if order_type == "limit" else self.fetch_market_price()
                if simulated_fill_price is None and order_type == "market":
                    logger.error("[SIMULATION] Could not fetch market price for simulated market order.")
                    return None

                trade_details = {
                    "id": f"sim_{int(time.time() * 1000)}",
                    "symbol": self.symbol,
                    "status": "closed",
                    "side": side,
                    "type": order_type,
                    "amount": float(order_size_str),
                    "price": simulated_fill_price,
                    "average": simulated_fill_price,
                    "cost": float(order_size_str) * simulated_fill_price,
                    "filled": float(order_size_str),
                    "remaining": 0.0,
                    "timestamp": int(time.time() * 1000),
                    "datetime": pd.to_datetime(time.time(), unit="s").isoformat(),
                    "fee": None,
                    "info": {
                        "stopLoss": sl_price_str,
                        "takeProfit": tp_price_str,
                        "orderType": order_type.capitalize(),
                        "execType": "Trade",
                    },
                }
                log_price = price_str if order_type == "limit" else f"Market (~{simulated_fill_price:.2f})"
                logger.info(
                    f"[SIMULATION] Order Placed: ID: {trade_details['id']}, Type: {trade_details['type']}, "
                    f"Side: {trade_details['side'].upper()}, Size: {trade_details['amount']:.4f} {self.market['base']}, "
                    f"Price: {log_price}, SL: {sl_price_str or 'N/A'}, TP: {tp_price_str or 'N/A'}, Status: {trade_details['status']}"
                )
                return trade_details
            else:  # Live Trading
                order = None
                if order_type == "market":
                    logger.info(f"Placing LIVE Market {side.upper()} order for {order_size_str} {self.market['base']} with params: {params}")
                    order = self.exchange.create_market_order(self.symbol, side, float(order_size_str), params=params)
                elif order_type == "limit":
                    if price is None:
                        logger.error("Limit price required for live limit orders.")
                        return None
                    logger.info(f"Placing LIVE Limit {side.upper()} order for {order_size_str} {self.market['base']} at {price_str} with params: {params}")
                    order = self.exchange.create_limit_order(self.symbol, side, float(order_size_str), float(price_str), params=params)
                else:
                    logger.error(f"Unsupported order type for live trading: {order_type}")
                    return None

                if order:
                    log_price_actual = order.get("average") or order.get("price")
                    log_price_display = f"{log_price_actual:.{self.market['precision']['price']}f}" if log_price_actual else (price_str if order_type == 'limit' else 'Market')

                    logger.info(
                        f"Order Placed Successfully: ID: {order.get('id')}, Type: {order.get('type')}, Side: {order.get('side', '').upper()}, "
                        f"Amount: {order.get('amount'):.{self.market['precision']['amount']}f} {self.market['base']}, Fill Price: {log_price_display}, "
                        f"StopLoss: {params.get('stopLoss', 'N/A')}, TakeProfit: {params.get('takeProfit', 'N/A')}, Status: {order.get('status', 'N/A')}"
                    )
                else:
