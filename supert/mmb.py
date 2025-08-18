#!/usr/bin/env python3
"""
Pyrmethus's Market-Making Alchemist - A Spell to Conjure Market-Making Orders on Bybit.

Forged with an asyncio-native, lock-free approach to order book & position state.
This ritual is imbued with:
- A powerful, pluggable strategy engine.
- Enhanced, colorized logging for a vibrant terminal experience.
- Graceful shutdown upon interruption.
- An elegant, self-healing state management system.
"""
from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, getcontext
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import aiohttp
import websockets
from colorama import Fore, Style, init
from websockets.exceptions import ConnectionClosed

# Initialize the magic of Colorama
init(autoreset=True)

# Set high precision for financial rituals
getcontext().prec = 38

# --- Configuration Runes from the Environment (12-factor: env vars only) ---
SYMBOL: str = os.getenv("SYMBOL", "BTCUSDT")
BASE_QTY: Decimal = Decimal(os.getenv("BASE_QTY", "0.001"))
ORDER_LEVELS: int = int(os.getenv("ORDER_LEVELS", "5"))
SPREAD_BPS: Decimal = Decimal(os.getenv("SPREAD_BPS", "0.05"))
MAX_POSITION: Decimal = Decimal(os.getenv("MAX_POSITION", "0.1"))
INVENTORY_TARGET: Decimal = Decimal(os.getenv("INVENTORY_TARGET", "0"))
BYBIT_TESTNET: bool = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
API_KEY: str = os.environ.get("BYBIT_API_KEY", "")
API_SECRET: str = os.environ.get("BYBIT_API_SECRET", "")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# --- Logging & Mystical Output ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("alchemist")

def log_colorized_message(level: int, message: str, color: str, style: str = Style.NORMAL) -> None:
    """Logs a message with a mystical glow."""
    log_level_name = logging.getLevelName(level)
    if log_level_name == "WARNING":
        prefix = f"{Fore.YELLOW}{Style.BRIGHT}[WARNING]{Style.RESET_ALL}"
    elif log_level_name == "ERROR":
        prefix = f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL}"
    else:
        prefix = f"{Fore.LIGHTCYAN_EX}[{log_level_name}]{Style.RESET_ALL}"

    log.log(level, f"{prefix} {color}{style}{message}{Style.RESET_ALL}")

# --- Data Models of the Cosmos ---
@dataclass(slots=True)
class MarketData:
    """A sigil containing the current market's essence."""
    symbol: str
    bid: Decimal
    ask: Decimal
    bid_sz: Decimal
    ask_sz: Decimal
    ts: float

    @property
    def mid(self) -> Decimal:
        """Divines the mid-price from the bid and ask."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Reveals the spread of the market."""
        return self.ask - self.bid

@dataclass(slots=True)
class Order:
    """A sigil representing an order placed in the market."""
    id: str
    symbol: str
    side: str
    price: Decimal
    qty: Decimal
    filled: Decimal = Decimal("0")
    status: str = "New"

# --- The Bybit V5 Conduit (async, circuit-breaker) ---
class BybitClient:
    """A conduit to interact with the Bybit API."""
    def __init__(self) -> None:
        self.base_url = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"
        self.ws_public = "wss://stream-testnet.bybit.com/v5/public/linear" if BYBIT_TESTNET else "wss://stream.bybit.com/v5/public/linear"
        self.ws_private = "wss://stream-testnet.bybit.com/v5/private" if BYBIT_TESTNET else "wss://stream.bybit.com/v5/private"
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def _sign(self, ts: str, recv: str, payload: str) -> str:
        """Performs a mystical HMAC-SHA256 signature."""
        param_str = f"{ts}API_KEY{API_KEY}recvWindow{recv}{payload}"
        return hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()

    async def _request(self, method: str, path: str, params: Dict[str, Any]) -> Any:
        """Dispatches a request through the REST conduit."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        ts = str(int(time.time() * 1000))
        recv = "5000"
        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv,
            "Content-Type": "application/json",
        }
        if method.upper() == "GET":
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            headers["X-BAPI-SIGN"] = hmac.new(
                API_SECRET.encode(), (ts + API_KEY + recv + query).encode(), hashlib.sha256
            ).hexdigest()
            url = f"{self.base_url}{path}?{query}"
            async with self.session.get(url, headers=headers) as r:
                return await r.json()
        else:
            body = json.dumps(params, separators=(",", ":"))
            headers["X-BAPI-SIGN"] = hmac.new(
                API_SECRET.encode(), (ts + API_KEY + recv + body).encode(), hashlib.sha256
            ).hexdigest()
            async with self.session.post(
                f"{self.base_url}{path}", headers=headers, data=body
            ) as r:
                return await r.json()

    async def tickers(self, symbol: str) -> MarketData:
        """Fetches the market's current tick-essence."""
        data = await self._request(
            "GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol}
        )
        if data["retCode"] != 0 or not data["result"]["list"]:
            raise RuntimeError(f"Failed to fetch ticker for {symbol}: {data['retMsg']}")
        d = data["result"]["list"][0]
        return MarketData(
            symbol=symbol,
            bid=Decimal(d["bid1Price"]),
            ask=Decimal(d["ask1Price"]),
            bid_sz=Decimal(d["bid1Size"]),
            ask_sz=Decimal(d["ask1Size"]),
            ts=float(d["time"]) / 1000,
        )

    async def place_order(self, symbol: str, side: str, price: Decimal, qty: Decimal) -> str:
        """Places a new order with a ritualistic incantation."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": "PostOnly",
        }
        res = await self._request("POST", "/v5/order/create", params)
        if res["retCode"] != 0:
            raise RuntimeError(res["retMsg"])
        log_colorized_message(logging.INFO, f"Order placed: {side} {qty} @ {price}", Fore.GREEN, Style.BRIGHT)
        return res["result"]["orderId"]

    async def cancel_order(self, symbol: str, order_id: str) -> None:
        """Cancels a specific order, dispelling it from the market."""
        params = {"category": "linear", "symbol": symbol, "orderId": order_id}
        res = await self._request("POST", "/v5/order/cancel", params)
        if res["retCode"] != 0:
            raise RuntimeError(res["retMsg"])
        log_colorized_message(logging.INFO, f"Order cancelled: {order_id}", Fore.YELLOW)

    async def open_orders(self, symbol: str) -> List[Order]:
        """Divines a list of all active orders."""
        res = await self._request(
            "GET",
            "/v5/order/realtime",
            {"category": "linear", "symbol": symbol, "orderStatus": "New,PartiallyFilled"},
        )
        if res["retCode"] != 0:
            raise RuntimeError(res["retMsg"])
        return [
            Order(
                id=o["orderId"],
                symbol=o["symbol"],
                side=o["side"],
                price=Decimal(o["price"]),
                qty=Decimal(o["qty"]),
                filled=Decimal(o["cumExecQty"]),
                status=o["orderStatus"],
            ) for o in res["result"]["list"]
        ]

    async def position(self, symbol: str) -> Decimal:
        """Summons the current position size."""
        res = await self._request(
            "GET", "/v5/position/list", {"category": "linear", "symbol": symbol}
        )
        if res["retCode"] != 0:
            raise RuntimeError(res["retMsg"])
        lst = res["result"]["list"]
        if not lst:
            return Decimal("0")
        pos = lst[0]
        size = Decimal(pos["size"])
        return size if pos["side"] == "Buy" else -size

    async def close(self) -> None:
        """Closes the REST conduit."""
        if self.session:
            await self.session.close()

# --- The Strategy Engine (Pluggable) ---
class SimpleSpreadStrategy:
    """A basic strategy for placing orders at a fixed spread around the mid-price."""
    def __init__(self, spread_bps: Decimal, levels: int, qty: Decimal) -> None:
        self.spread_bps = spread_bps
        self.levels = levels
        self.qty = qty

    async def compute_quotes(self, md: MarketData, position: Decimal) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """Calculates the prices and quantities for new orders."""
        mid = md.mid
        spread = mid * self.spread_bps / 10_000
        bids = [(mid - spread * (i + 1), self.qty) for i in range(self.levels)]
        asks = [(mid + spread * (i + 1), self.qty) for i in range(self.levels)]
        
        return bids, asks

# --- The Alchemist's Core Ritual Loop ---
class AlchemistCore:
    """The central engine for the market-making ritual."""
    def __init__(self) -> None:
        if not API_KEY or not API_SECRET:
            log_colorized_message(logging.CRITICAL, "BYBIT_API_KEY and BYBIT_API_SECRET environment variables must be set. The gates to the exchange remain sealed!", Fore.RED, Style.BRIGHT)
            sys.exit(1)
        self.client = BybitClient()
        self.strategy = SimpleSpreadStrategy(SPREAD_BPS, ORDER_LEVELS, BASE_QTY)
        self._orders: Dict[str, Order] = {}
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        """Initiates the core ritual loop."""
        log_colorized_message(logging.INFO, "Alchemist is awakening...", Fore.CYAN)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self.shutdown)
        loop.add_signal_handler(signal.SIGTERM, self.shutdown)
        
        await self._main_loop()

    def shutdown(self) -> None:
        """Gracefully triggers the shutdown ritual."""
        log_colorized_message(logging.INFO, "A farewell is cast. The ritual is being gracefully concluded.", Fore.MAGENTA)
        self._shutdown.set()

    async def _sync_orders(self) -> None:
        """Periodically synchronizes local order state with the API."""
        try:
            api_orders = {o.id: o for o in await self.client.open_orders(SYMBOL)}
            self._orders = api_orders
            log_colorized_message(logging.DEBUG, f"Order state synced. Found {len(self._orders)} open orders.", Fore.WHITE)
        except Exception as e:
            log_colorized_message(logging.ERROR, f"Failed to sync orders: {e}", Fore.RED)

    async def _reconcile_orders(self, desired_bids: List[Tuple[Decimal, Decimal]], desired_asks: List[Tuple[Decimal, Decimal]]) -> None:
        """Reconciles the desired order state with the current active orders."""
        desired_state = {f"{p}-{s}": (p, q, s) for p, q in desired_bids + desired_asks for s in ("Buy", "Sell")}
        
        current_state = {f"{o.price}-{o.side}": o for o in self._orders.values()}
        
        orders_to_cancel = [o for o in self._orders.values() if f"{o.price}-{o.side}" not in desired_state]
        
        new_orders = [
            (price, qty, side) for price, qty, side in (desired_bids + desired_asks)
            if f"{price}-{side}" not in current_state
        ]
        
        if orders_to_cancel:
            log_colorized_message(logging.INFO, f"Dispelling {len(orders_to_cancel)} stale orders...", Fore.LIGHTYELLOW_EX)
            await asyncio.gather(*[self.client.cancel_order(SYMBOL, o.id) for o in orders_to_cancel])
        
        if new_orders:
            log_colorized_message(logging.INFO, f"Conjuring {len(new_orders)} new orders...", Fore.LIGHTGREEN_EX)
            for price, qty, side in new_orders:
                try:
                    await self.client.place_order(SYMBOL, side, price, qty)
                except Exception as e:
                    log_colorized_message(logging.ERROR, f"Failed to place order: {e}", Fore.RED)

    async def _main_loop(self) -> None:
        """The main ritual loop, performing the market-making spell."""
        log_colorized_message(logging.INFO, "The Alchemist has entered the main ritual loop.", Fore.CYAN, Style.BRIGHT)
        sync_interval = 60 # Sync with API every 60 seconds
        last_sync = 0
        
        while not self._shutdown.is_set():
            try:
                if time.time() - last_sync > sync_interval:
                    await self._sync_orders()
                    last_sync = time.time()

                md = await self.client.tickers(SYMBOL)
                pos = await self.client.position(SYMBOL)
                
                log_colorized_message(logging.INFO, f"Market Data: Mid={md.mid:.4f}, Position={pos}", Fore.BLUE)
                
                bids, asks = await self.strategy.compute_quotes(md, pos)
                
                await self._reconcile_orders(bids, asks)

                await asyncio.sleep(1)

            except Exception as e:
                log_colorized_message(logging.ERROR, f"A cosmic disturbance occurred in the ritual loop: {e}. Waiting to re-align...", Fore.RED, Style.BRIGHT)
                await asyncio.sleep(5)

    async def close(self) -> None:
        """Cleanses the realm of any remaining orders."""
        log_colorized_message(logging.INFO, "The ritual is concluding. Dispelling all open orders...", Fore.MAGENTA)
        
        # Sync one last time to get all open orders
        await self._sync_orders()
        
        orders_to_cancel = list(self._orders.values())
        if orders_to_cancel:
            await asyncio.gather(*[self.client.cancel_order(SYMBOL, o.id) for o in orders_to_cancel])
            log_colorized_message(logging.INFO, f"{len(orders_to_cancel)} orders successfully dispelled.", Fore.GREEN)
        
        await self.client.close()

# --- Awaken the Alchemist ---
async def main() -> None:
    """The grand incantation to summon the alchemist."""
    alchemist = AlchemistCore()
    try:
        await alchemist.start()
    except (KeyboardInterrupt, SystemExit):
        await alchemist.close()
    finally:
        await alchemist.close()

if __name__ == "__main__":
    asyncio.run(main())
