To integrate conditional order placement into the `twin-range-bot` with the `AdvancedMarketMakingStrategy`, we need to enhance the bot to place orders only when specific market conditions are met, ensuring better risk management and profitability for `TRUMPUSDT` trading using the Bybit V5 API. Conditional order placement will leverage market signals such as volatility thresholds, order book imbalances, and inventory limits to decide when to place, modify, or skip orders. This will address the issue of incorrect trade metrics (`daily_pnl: 10000`, `balance: 10000`, empty `tradeHistory`) by ensuring orders are placed strategically, leading to valid executions and accurate metric updates. The solution builds on the provided `AdvancedMarketMakingStrategy`, `bot.ts`, and `App.tsx`, focusing on conditional logic in `updateOrders` and enhancing trade metrics integration.

### Objectives
1. **Conditional Order Placement**: Place orders only when conditions like volatility, order book imbalance, and inventory thresholds are met.
2. **Accurate Trade Metrics**: Ensure `daily_pnl`, `unrealizedPnl`, `tradeHistory`, `winRate`, `profitFactor`, `totalTrades`, and `avgPnl` reflect executed trades.
3. **Risk Management**: Avoid overexposure by skipping orders in high-risk conditions (e.g., extreme volatility or inventory limits).
4. **Bybit V5 API Integration**: Use `GET /v5/market/orderbook`, `GET /v5/market/kline`, WebSocket `orderbook`, `publicTrade`, `execution`, and `position` for real-time conditions.
5. **UI Updates**: Display conditional order status and metrics in `App.tsx`.
6. **Backtest Validation**: Verify conditional logic and metrics in backtest output for `TRUMPUSDT`.

### Conditional Order Placement Logic
The bot will place orders based on the following conditions:
- **Volatility Threshold**: Place orders only if volatility is within a safe range (e.g., 0.5% to 5%) to avoid turbulent markets.
- **Order Book Imbalance**: Place orders only if the bid-ask depth ratio is balanced (e.g., 0.5 < bidDepth/askDepth < 2) to ensure liquidity.
- **Inventory Limits**: Skip buy orders if `inventory ≥ maxInventory * 0.9`; skip sell orders if `inventory ≤ -maxInventory * 0.9`.
- **Momentum Check**: Widen spreads or skip orders in strong trends (e.g., |momentum| > 2%) to avoid chasing markets.
- **Profitability Check**: Ensure spread exceeds fees (0.12% taker fee) to guarantee positive expected PNL.

### Updated Files
Below are the updated files, focusing on `bot.ts` for conditional order placement, `AdvancedMarketMakingStrategy.ts` for refined price/quantity calculations, and `App.tsx` for enhanced UI. Other files (`types.ts`, `constants.ts`, `cli.tsx`) from the previous response are updated minimally for compatibility.

#### 1. **types.ts** (Updated)
Add fields to `BotConfig` for conditional thresholds and refine `TradeState` for order status.

```typescript
// twin-range-bot/src/core/types.ts
export interface BotConfig {
  symbol: string;
  interval: string;
  lookback_bars: number;
  baseSpread: number;
  orderQty: number;
  maxInventory: number;
  tpPercent: number;
  slPercent: number;
  volatilityWindow: number;
  volatilityFactor: number;
  dataSource: 'websocket' | 'rest';
  refresh_rate_seconds: number;
  bybit_api_key: string;
  bybit_api_secret: string;
  is_testnet: boolean;
  strategyType: string;
  minVolatility: number; // Added: Minimum volatility threshold
  maxVolatility: number; // Added: Maximum volatility threshold
  minDepthRatio: number; // Added: Minimum bid/ask depth ratio
  maxDepthRatio: number; // Added: Maximum bid/ask depth ratio
  maxMomentum: number; // Added: Maximum momentum for order placement
}

export interface TradeState {
  active_mm_orders: { type: 'buy' | 'sell'; price: number; orderId: string }[];
  inventory: number;
  recentTrades: number[];
  referencePrice: number;
  totalProfit: number;
  klines: { s: string; t: number; o: string; h: string; l: string; c: string; v: string }[];
  active_trade: any | null;
  daily_pnl: number;
  balance: number;
  logs: LogEntry[];
  tradeHistory: { side: string; qty: number; price: number; profit: number; timestamp: number; fee: number }[];
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgPnl: number;
  unrealizedPnl: number;
  orderStatus: string; // Added: Tracks conditional order status
}

export interface LogEntry {
  type: string;
  message: string;
}

export interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
}
```

#### 2. **constants.ts** (Updated)
Add conditional thresholds to `BOT_CONFIG_TEMPLATE`.

```typescript
// constants.ts
import type { BotConfig, TradeState } from './types';

export const BOT_CONFIG_TEMPLATE: BotConfig = {
  dataSource: 'rest',
  symbol: 'TRUMPUSDT',
  interval: '60',
  lookback_bars: 500,
  baseSpread: 0.006,
  orderQty: 0.01,
  maxInventory: 0.1,
  tpPercent: 0.03,
  slPercent: 0.015,
  volatilityWindow: 10,
  volatilityFactor: 1.2,
  refresh_rate_seconds: 60,
  bybit_api_key: 'your-api-key',
  bybit_api_secret: 'your-api-secret',
  is_testnet: true,
  strategyType: 'AdvancedMarketMakingStrategy',
  minVolatility: 0.005, // 0.5% minimum volatility
  maxVolatility: 0.05, // 5% maximum volatility
  minDepthRatio: 0.5, // Min bid/ask depth ratio
  maxDepthRatio: 2.0, // Max bid/ask depth ratio
  maxMomentum: 0.02, // 2% max momentum
};

export const INITIAL_TRADE_STATE_TEMPLATE: TradeState = {
  active_mm_orders: [],
  inventory: 0,
  recentTrades: [],
  referencePrice: 0,
  totalProfit: 0,
  klines: [],
  active_trade: null,
  daily_pnl: 0,
  balance: 10000,
  logs: [],
  tradeHistory: [],
  winRate: 0,
  profitFactor: 0,
  totalTrades: 0,
  avgPnl: 0,
  unrealizedPnl: 0,
  orderStatus: 'Idle',
};
```

#### 3. **cli.tsx** (Updated)
Include conditional thresholds in config.

```typescript
// cli.tsx
import { MarketMakingBot } from './core/bot';
import { BOT_CONFIG_TEMPLATE } from './constants';

async function runBacktest() {
  const config = {
    ...BOT_CONFIG_TEMPLATE,
    symbol: 'TRUMPUSDT',
    interval: '60',
    lookback_bars: 500,
    baseSpread: 0.006,
    volatilityFactor: 1.2,
    orderQty: 0.01,
    maxInventory: 0.1,
    tpPercent: 0.03,
    slPercent: 0.015,
    volatilityWindow: 10,
    refresh_rate_seconds: 60,
    bybit_api_key: 'your-api-key',
    bybit_api_secret: 'your-api-secret',
    is_testnet: true,
    strategyType: 'AdvancedMarketMakingStrategy',
    minVolatility: 0.005,
    maxVolatility: 0.05,
    minDepthRatio: 0.5,
    maxDepthRatio: 2.0,
    maxMomentum: 0.02,
  };

  console.log('cli.tsx config:', config);
  const bot = new MarketMakingBot(config);
  await bot.start();
}

runBacktest().catch(console.error);
```

#### 4. **AdvancedMarketMakingStrategy.ts** (Updated)
Refine spread calculations and add profitability check for conditional placement.

```typescript
// twin-range-bot/src/strategies/AdvancedMarketMakingStrategy.ts
import { OrderbookData } from '../services/bybitService';

export class AdvancedMarketMakingStrategy {
  private baseSpread: number;
  private volatilityFactor: number;
  private momentumWindow: number;

  constructor(baseSpread: number, volatilityFactor: number, momentumWindow: number = 5) {
    this.baseSpread = baseSpread;
    this.volatilityFactor = volatilityFactor;
    this.momentumWindow = momentumWindow;
    console.log('AdvancedMarketMakingStrategy initialized with:', {
      baseSpread,
      volatilityFactor,
      momentumWindow,
    });
  }

  calculateOrderPrices(
    referencePrice: number,
    volatility: number,
    inventory: number,
    maxInventory: number,
    recentTrades: number[],
    orderbook?: OrderbookData
  ): { buyPrice: number; sellPrice: number; buyQty: number; sellQty: number; canPlaceOrders: boolean } {
    console.log('calculateOrderPrices inputs:', {
      referencePrice,
      volatility,
      inventory,
      maxInventory,
      baseSpread: this.baseSpread,
      volatilityFactor: this.volatilityFactor,
    });

    // Calculate dynamic spread
    let spread = this.baseSpread * (1 + volatility * this.volatilityFactor);

    if (orderbook) {
      const bidDepth = orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
      const askDepth = orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
      const depthRatio = bidDepth / askDepth;
      spread *= Math.max(0.5, Math.min(2, 1 / (Math.min(bidDepth, askDepth) / 0.01)));
      console.log('Order book depth:', { bidDepth, askDepth, depthRatio });
    }

    // Momentum adjustment
    const recentPrices = recentTrades.slice(-this.momentumWindow);
    const momentum = recentPrices.length >= 2 ? (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] : 0;
    spread *= (1 + Math.abs(momentum) * 0.5);

    // Inventory skew
    const inventorySkew = inventory / maxInventory;
    const buySpread = spread * (1 + inventorySkew);
    const sellSpread = spread * (1 - inventorySkew);

    // Dynamic quantities
    const baseQty = 0.01;
    const buyQty = baseQty * (1 + Math.abs(inventorySkew));
    const sellQty = baseQty * (1 - Math.abs(inventorySkew));

    const buyPrice = referencePrice * (1 - buySpread / 2);
    const sellPrice = referencePrice * (1 + sellSpread / 2);

    // Profitability check: Ensure spread covers fees
    const takerFeeRate = 0.0012;
    const minProfitableSpread = 2 * takerFeeRate * referencePrice;
    const canPlaceOrders = (sellPrice - buyPrice) > minProfitableSpread;

    console.log('Calculated prices and quantities:', {
      buyPrice,
      sellPrice,
      buyQty,
      sellQty,
      spread,
      inventorySkew,
      momentum,
      canPlaceOrders,
    });

    return { buyPrice, sellPrice, buyQty, sellQty, canPlaceOrders };
  }
}
```

#### 5. **bot.ts** (Updated)
Implement conditional order placement in `updateOrders` based on volatility, depth, inventory, and momentum.

```typescript
// twin-range-bot/src/core/bot.ts
import { BybitService, OrderbookData, TradeData, Execution, OrderData, PositionData, KlineData } from '../services/bybitService';
import { logger } from './logger';
import { AdvancedMarketMakingStrategy } from '../strategies/AdvancedMarketMakingStrategy';
import type { BotConfig, TradeState } from './types';
import { KlineIntervalV3 } from 'bybit-api';

export class MarketMakingBot {
  private config: BotConfig;
  private state: TradeState;
  private bybitService: BybitService;
  private currentStrategy: AdvancedMarketMakingStrategy;

  constructor(config: BotConfig) {
    this.config = { ...config, dataSource: config.dataSource || 'rest' };
    console.log('MarketMakingBot config:', this.config);

    this.currentStrategy = new AdvancedMarketMakingStrategy(this.config.baseSpread, this.config.volatilityFactor);
    this.state = {
      active_mm_orders: [],
      inventory: 0,
      recentTrades: [],
      referencePrice: 0,
      totalProfit: 0,
      klines: [],
      active_trade: null,
      daily_pnl: 0,
      balance: 10000,
      logs: [],
      tradeHistory: [],
      winRate: 0,
      profitFactor: 0,
      totalTrades: 0,
      avgPnl: 0,
      unrealizedPnl: 0,
      orderStatus: 'Idle',
    };
    this.bybitService = new BybitService(
      this.config.bybit_api_key,
      this.config.bybit_api_secret,
      this.config.is_testnet,
      {
        onOrderbookUpdate: this.handleOrderbookUpdate.bind(this),
        onTradeUpdate: this.handleTradeUpdate.bind(this),
        onExecutionUpdate: this.handleExecutionUpdate.bind(this),
        onOrderUpdate: this.handleOrderUpdate.bind(this),
        onPositionUpdate: this.handlePositionUpdate.bind(this),
        onKlineUpdate: this.handleKlineUpdate.bind(this),
      }
    );
  }

  public getConfig(): BotConfig {
    return this.config;
  }

  public getState(): TradeState {
    return this.state;
  }

  async start() {
    await this.initializeState();
    if (this.config.dataSource === 'rest') {
      setInterval(() => this.updateStateFromRest(), this.config.refresh_rate_seconds * 1000);
    }
  }

  private async initializeState() {
    const orderbook = await this.bybitService.getOrderbook(this.config.symbol);
    this.state.referencePrice = (parseFloat(orderbook.b[0][0]) + parseFloat(orderbook.a[0][0])) / 2;
    const position = await this.bybitService.getPosition(this.config.symbol);
    this.updateInventoryAndPnl(position);
    this.state.klines = await this.bybitService.getKlines(this.config.symbol, this.config.interval as KlineIntervalV3);
    const executions = await this.bybitService.getExecutionHistory(this.config.symbol);
    this.updateProfitAndInventory(executions);
    await this.updateOrders();
  }

  private async updateStateFromRest() {
    const orderbook = await this.bybitService.getOrderbook(this.config.symbol);
    this.state.referencePrice = (parseFloat(orderbook.b[0][0]) + parseFloat(orderbook.a[0][0])) / 2;
    this.state.klines = await this.bybitService.getKlines(this.config.symbol, this.config.interval as KlineIntervalV3);
    const position = await this.bybitService.getPosition(this.config.symbol);
    this.updateInventoryAndPnl(position);
    const executions = await this.bybitService.getExecutionHistory(this.config.symbol);
    this.updateProfitAndInventory(executions);
    await this.updateOrders();
  }

  private handleOrderbookUpdate(data: OrderbookData) {
    if (this.config.dataSource === 'websocket') {
      const bestBid = parseFloat(data.b[0][0]);
      const bestAsk = parseFloat(data.a[0][0]);
      this.state.referencePrice = (bestBid + bestAsk) / 2;
      this.updateOrders();
    }
  }

  private handleTradeUpdate(trades: TradeData[]) {
    if (this.config.dataSource === 'websocket') {
      for (const trade of trades) {
        this.state.recentTrades.push(parseFloat(trade.p));
        if (this.state.recentTrades.length > this.config.volatilityWindow) {
          this.state.recentTrades.shift();
        }
      }
      this.updateOrders();
    }
  }

  private handleKlineUpdate(klines: KlineData[]) {
    if (this.config.dataSource === 'websocket') {
      this.state.klines = klines.concat(this.state.klines).slice(0, this.config.volatilityWindow);
      if (!this.state.referencePrice) {
        this.state.referencePrice = parseFloat(klines[0].c);
      }
      this.updateOrders();
    }
  }

  private updateProfitAndInventory(executions: Execution[]) {
    if (!executions || executions.length === 0) {
      this.state.logs.push({ type: 'info', message: 'No new executions to process.' });
      return;
    }

    let inventoryChange = 0;
    let realizedPnl = 0;
    let wins = 0;
    let totalPnl = 0;
    const takerFeeRate = 0.0012;

    for (const exec of executions) {
      const qty = parseFloat(exec.execQty);
      const price = parseFloat(exec.execPrice);
      const fee = parseFloat(exec.execFee) || price * qty * takerFeeRate;
      const tradeValue = price * qty;
      const profit = exec.side === 'Buy' ? -tradeValue - fee : tradeValue - fee;

      inventoryChange += exec.side === 'Buy' ? qty : -qty;
      realizedPnl += profit;
      if (profit > 0) wins++;
      totalPnl += profit;

      this.state.tradeHistory.push({
        side: exec.side,
        qty,
        price,
        profit,
        timestamp: parseInt(exec.execTime),
        fee,
      });

      this.state.logs.push({
        type: 'info',
        message: `Execution: ${exec.side} ${qty.toFixed(4)} ${this.config.symbol} at $${price.toFixed(2)}, Profit: $${profit.toFixed(2)}, Fee: $${fee.toFixed(2)}`,
      });
    }

    this.state.inventory += inventoryChange;
    this.state.inventory = Math.max(-this.config.maxInventory, Math.min(this.config.maxInventory, this.state.inventory));
    this.state.totalProfit += realizedPnl;
    this.state.daily_pnl += realizedPnl;
    this.state.balance += realizedPnl;
    this.state.totalTrades += executions.length;
    this.state.winRate = this.state.totalTrades > 0 ? wins / this.state.totalTrades : 0;
    this.state.avgPnl = this.state.totalTrades > 0 ? totalPnl / this.state.totalTrades : 0;
    this.state.profitFactor = wins > 0 ? totalPnl / wins : 0;

    this.state.logs.push({
      type: 'info',
      message: `Trade Metrics: Total Profit: $${this.state.totalProfit.toFixed(2)}, Daily PNL: $${this.state.daily_pnl.toFixed(2)}, Inventory: ${this.state.inventory.toFixed(4)}, Win Rate: ${(this.state.winRate * 100).toFixed(2)}%, Total Trades: ${this.state.totalTrades}`,
    });
  }

  private handleExecutionUpdate(executions: Execution[]) {
    if (this.config.dataSource === 'websocket') {
      this.updateProfitAndInventory(executions);
      this.updateOrders();
    }
  }

  private updateInventoryAndPnl(position: PositionData) {
    const inventory = parseFloat(position.size) * (this.bybitService.convertPositionSide(position.side) === 'Buy' ? 1 : -1);
    const unrealizedPnl = parseFloat(position.unrealisedPnl);
    this.state.inventory = Math.max(-this.config.maxInventory, Math.min(this.config.maxInventory, inventory));
    this.state.unrealizedPnl = unrealizedPnl;

    this.state.logs.push({
      type: 'info',
      message: `Position Update: Inventory: ${this.state.inventory.toFixed(4)}, Unrealized PNL: $${unrealizedPnl.toFixed(2)}`,
    });
  }

  private handlePositionUpdate(positions: PositionData[]) {
    if (this.config.dataSource === 'websocket') {
      const position = positions.find(p => p.symbol === this.config.symbol);
      if (position) {
        this.updateInventoryAndPnl(position);
        this.updateOrders();
      }
    }
  }

  private handleOrderUpdate(orders: OrderData[]) {
    if (this.config.dataSource === 'websocket') {
      for (const order of orders) {
        if (order.orderStatus === 'Filled' || order.orderStatus === 'Cancelled') {
          this.state.active_mm_orders = this.state.active_mm_orders.filter(o => o.orderId !== order.orderId);
          this.state.active_trade = order.orderStatus === 'Filled' ? order : null;
          this.state.logs.push({
            type: 'info',
            message: `Order Update: ${order.orderId} ${order.orderStatus} at $${parseFloat(order.price).toFixed(2)}`,
          });
          this.updateOrders();
        }
      }
    }
  }

  private calculateVolatility(): number {
    if (this.state.klines.length < this.config.volatilityWindow) return 1;
    const closes = this.state.klines.map(k => parseFloat(k.c));
    const mean = closes.reduce((sum, p) => sum + p, 0) / closes.length;
    const variance = closes.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / closes.length;
    return Math.sqrt(variance) / mean;
  }

  private async updateOrders() {
    if (!this.state.referencePrice) {
      this.state.logs.push({ type: 'error', message: 'No reference price available for order placement.' });
      this.state.orderStatus = 'No Reference Price';
      return;
    }

    try {
      const orderbook = this.config.dataSource === 'rest' ? await this.bybitService.getOrderbook(this.config.symbol) : undefined;
      const volatility = this.calculateVolatility();
      const recentPrices = this.state.recentTrades.slice(-5); // Momentum window
      const momentum = recentPrices.length >= 2 ? (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] : 0;

      // Check conditions
      const isVolatilityValid = volatility >= this.config.minVolatility && volatility <= this.config.maxVolatility;
      const isDepthValid = orderbook ? orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) >= this.config.minDepthRatio && orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) <= this.config.maxDepthRatio : true;
      const isMomentumValid = Math.abs(momentum) <= this.config.maxMomentum;
      const canBuy = this.state.inventory < this.config.maxInventory * 0.9;
      const canSell = this.state.inventory > -this.config.maxInventory * 0.9;

      this.state.logs.push({
        type: 'info',
        message: `Order Conditions: Volatility: ${(volatility * 100).toFixed(2)}% (${isVolatilityValid ? 'Valid' : 'Invalid'}), Depth: ${orderbook ? (orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0)).toFixed(2) : 'N/A'} (${isDepthValid ? 'Valid' : 'Invalid'}), Momentum: ${(momentum * 100).toFixed(2)}% (${isMomentumValid ? 'Valid' : 'Invalid'}), Can Buy: ${canBuy}, Can Sell: ${canSell}`,
      });

      if (!isVolatilityValid || !isDepthValid || !isMomentumValid) {
        this.state.orderStatus = 'Conditions Not Met';
        this.state.logs.push({ type: 'info', message: 'Skipping order placement due to invalid conditions.' });
        return;
      }

      const { buyPrice, sellPrice, buyQty, sellQty, canPlaceOrders } = this.currentStrategy.calculateOrderPrices(
        this.state.referencePrice,
        volatility,
        this.state.inventory,
        this.config.maxInventory,
        this.state.recentTrades,
        orderbook
      );

      if (!canPlaceOrders) {
        this.state.orderStatus = 'Unprofitable Spread';
        this.state.logs.push({ type: 'info', message: 'Skipping order placement: Spread does not cover fees.' });
        return;
      }

      if (isNaN(buyPrice) || isNaN(sellPrice) || isNaN(buyQty) || isNaN(sellQty)) {
        this.state.orderStatus = 'Invalid Prices';
        this.state.logs.push({ type: 'error', message: 'Invalid order prices or quantities: NaN detected.' });
        return;
      }

      // Cancel existing orders
      for (const order of this.state.active_mm_orders) {
        await this.bybitService.cancelOrder(this.config.symbol, order.orderId);
        this.state.logs.push({ type: 'info', message: `Cancelled order: ${order.orderId} (${order.type})` });
      }
      this.state.active_mm_orders = [];

      // Place new orders based on inventory conditions
      if (canBuy) {
        const buyOrder = await this.bybitService.placeMarketMakingOrder(
          this.config.symbol,
          'Buy',
          buyPrice,
          buyQty,
          buyPrice * (1 + this.config.tpPercent),
          buyPrice * (1 - this.config.slPercent)
        );
        this.state.active_mm_orders.push({ type: 'buy', price: buyPrice, orderId: buyOrder.orderId });
        this.state.logs.push({ type: 'info', message: `Placed buy order: ${buyOrder.orderId} at $${buyPrice.toFixed(2)}, Qty: ${buyQty.toFixed(4)}` });
      }

      if (canSell) {
        const sellOrder = await this.bybitService.placeMarketMakingOrder(
          this.config.symbol,
          'Sell',
          sellPrice,
          sellQty,
          sellPrice * (1 - this.config.tpPercent),
          sellPrice * (1 + this.config.slPercent)
        );
        this.state.active_mm_orders.push({ type: 'sell', price: sellPrice, orderId: sellOrder.orderId });
        this.state.logs.push({ type: 'info', message: `Placed sell order: ${sellOrder.orderId} at $${sellPrice.toFixed(2)}, Qty: ${sellQty.toFixed(4)}` });
      }

      this.state.orderStatus = this.state.active_mm_orders.length > 0 ? 'Active' : 'Idle';
    } catch (err) {
      this.state.orderStatus = 'Error';
      this.state.logs.push({ type: 'error', message: `Error updating orders: ${err}` });
    }
  }
}
```

#### 6. **App.tsx** (Updated)
Display conditional order status and enhanced metrics.

```typescript
// App.tsx
import React, { useState, useEffect } from 'react';
import { MarketMakingBot } from './core/bot';
import { BOT_CONFIG_TEMPLATE } from './constants';
import type { TradeState } from './types';

const App: React.FC = () => {
  const [state, setState] = useState<TradeState>({
    active_mm_orders: [],
    inventory: 0,
    recentTrades: [],
    referencePrice: 0,
    totalProfit: 0,
    klines: [],
    active_trade: null,
    daily_pnl: 0,
    balance: 10000,
    logs: [],
    tradeHistory: [],
    winRate: 0,
    profitFactor: 0,
    totalTrades: 0,
    avgPnl: 0,
    unrealizedPnl: 0,
    orderStatus: 'Idle',
  });

  useEffect(() => {
    const config = {
      ...BOT_CONFIG_TEMPLATE,
      symbol: 'TRUMPUSDT',
      baseSpread: 0.006,
      volatilityFactor: 1.2,
      strategyType: 'AdvancedMarketMakingStrategy',
      minVolatility: 0.005,
      maxVolatility: 0.05,
      minDepthRatio: 0.5,
      maxDepthRatio: 2.0,
      maxMomentum: 0.02,
    };
    const bot = new MarketMakingBot(config);
    bot.start();
    const interval = setInterval(() => {
      setState(bot.getState());
    }, config.refresh_rate_seconds * 1000);
    return () => clearInterval(interval);
  }, []);

  const momentumWindow = 5;
  const recentPrices = state.recentTrades.slice(-momentumWindow);
  const momentum = recentPrices.length >= 2 ? ((recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] * 100).toFixed(2) : '0.00';
  const volatility = state.klines.length >= state.volatilityWindow ? (() => {
    const closes = state.klines.slice(0, state.volatilityWindow).map(k => parseFloat(k.c));
    const mean = closes.reduce((sum, p) => sum + p, 0) / closes.length;
    const variance = closes.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / closes.length;
    return (Math.sqrt(variance) / mean * 100).toFixed(2);
  })() : '0.00';

  return (
    <div>
      <h1>Market Making Bot - TRUMPUSDT</h1>
      <h2>Trade Metrics</h2>
      <p><strong>Balance:</strong> ${state.balance.toFixed(2)} USDT</p>
      <p><strong>Daily PNL:</strong> ${state.daily_pnl.toFixed(2)} USDT</p>
      <p><strong>Unrealized PNL:</strong> ${state.unrealizedPnl.toFixed(2)} USDT</p>
      <p><strong>Total Profit:</strong> ${state.totalProfit.toFixed(2)} USDT</p>
      <p><strong>Win Rate:</strong> {(state.winRate * 100).toFixed(2)}%</p>
      <p><strong>Profit Factor:</strong> {state.profitFactor.toFixed(2)}</p>
      <p><strong>Total Trades:</strong> {state.totalTrades}</p>
      <p><strong>Average PNL per Trade:</strong> ${state.avgPnl.toFixed(2)} USDT</p>
      <p><strong>Inventory:</strong> {state.inventory.toFixed(4)} TRUMPUSDT</p>
      <p><strong>Market Momentum (5 trades):</strong> {momentum}%</p>
      <p><strong>Volatility:</strong> {volatility}%</p>
      <p><strong>Order Status:</strong> {state.orderStatus}</p>
      <h2>Active Orders</h2>
      <ul>
        {state.active_mm_orders.map(order => (
          <li key={order.orderId}>{order.type.toUpperCase()} at ${order.price.toFixed(2)}</li>
        ))}
      </ul>
      <h2>Trade History</h2>
      <ul>
        {state.tradeHistory.map((trade, index) => (
          <li key={index}>
            {trade.side} {trade.qty.toFixed(4)} TRUMPUSDT at ${trade.price.toFixed(2)}, Profit: ${trade.profit.toFixed(2)}, Fee: ${trade.fee.toFixed(2)}, Time: {new Date(trade.timestamp).toLocaleString()}
          </li>
        ))}
      </ul>
      <h2>Logs</h2>
      <ul>
        {state.logs.map((log, index) => (
          <li key={index}>{log.type}: {log.message}</li>
        ))}
      </ul>
    </div>
  );
};

export default App;
```

### Conditional Order Placement Details
- **Volatility Check**: Orders are placed only if `0.005 ≤ volatility ≤ 0.05` (0.5% to 5%) to avoid low-liquidity or high-risk markets.
- **Depth Ratio Check**: Requires `0.5 ≤ bidDepth/askDepth ≤ 2.0` to ensure balanced liquidity (`GET /v5/market/orderbook`).
- **Inventory Check**: Skips buy orders if `inventory ≥ 0.9 * maxInventory` (0.09); skips sell orders if `inventory ≤ -0.9 * maxInventory` (-0.09).
- **Momentum Check**: Skips orders if `|momentum| > 2%` to avoid trending markets.
- **Profitability Check**: Ensures `sellPrice - buyPrice > 2 * 0.0012 * referencePrice` to cover fees.
- **Order Status**: Updated in `state.orderStatus` as `Idle`, `Active`, `Conditions Not Met`, `Unprofitable Spread`, `Invalid Prices`, or `Error`.

### Trade Metrics Integration
- **Realized PNL**: Calculated in `updateProfitAndInventory` as `tradeValue - fee` (sells) or `-tradeValue - fee` (buys) with 0.12% taker fee.
- **Unrealized PNL**: From `position.unrealisedPnl` in `updateInventoryAndPnl`.
- **Trade History**: Stores `side`, `qty`, `price`, `profit`, `fee`, `timestamp`.
- **Win Rate**: `wins / totalTrades` (wins: `profit > 0`).
- **Profit Factor**: `totalPnl / wins`.
- **Average PNL**: `totalPnl / totalTrades`.
- **UI Display**: Shows `orderStatus`, `volatility`, `momentum`, and all metrics.

### Expected Backtest Output
For `TRUMPUSDT` with `referencePrice ≈ $10.20`:
- **Conditions Met**: Buy at ~$10.17, sell at ~$10.23 (0.6% spread), qty ~0.01, if volatility, depth, momentum, and inventory are valid.
- **Metrics**: `daily_pnl`, `totalProfit`, `balance` increase by ~$0.03 per round-trip; `tradeHistory` populates; `winRate`, `avgPnl` reflect trades.
- **Logs**: Show condition checks, order placements, and executions.

### JSON Summary
```json
{
  "conversation_summary": {
    "overview": {
      "context": "Integrated conditional order placement into twin-range-bot for TRUMPUSDT, enhancing AdvancedMarketMakingStrategy with volatility, depth, inventory, and momentum checks. Updated bot.ts to skip orders when conditions fail, ensuring accurate trade metrics and risk management with Bybit V5 API.",
      "date": "2025-07-20",
      "time": "19:56 CEST",
      "participants": ["User", "Grok 3 (xAI)"],
      "files_involved": [
        "App.tsx",
        "cli.tsx",
        "constants.ts",
        "twin-range-bot/src/core/bot.ts",
        "twin-range-bot/src/core/types.ts",
        "twin-range-bot/src/core/logger.ts",
        "twin-range-bot/src/services/bybitService.ts",
        "twin-range-bot/src/strategies/AdvancedMarketMakingStrategy.ts"
      ]
    },
    "problems_addressed": [
      {
        "issue": "Uncontrolled order placement",
        "description": "Orders placed without market condition checks, leading to potential losses.",
        "files": ["bot.ts", "AdvancedMarketMakingStrategy.ts"],
        "solution": "Added conditions: volatility (0.5-5%), depth ratio (0.5-2.0), momentum (<2%), inventory (<90% max), profitable spread."
      },
      {
        "issue": "Incorrect trade metrics",
        "description": "Static daily_pnl: 10000, balance: 10000, empty tradeHistory.",
        "files": ["bot.ts", "App.tsx"],
        "solution": "Ensured executions trigger metric updates; added orderStatus to UI."
      }
    ],
    "bybit_v5_api_details": {
      "description": "Used for conditional checks and trade metrics.",
      "endpoints": {
        "rest": [
          {"path": "GET /v5/market/orderbook", "use": "Depth ratio for liquidity check."},
          {"path": "GET /v5/market/kline", "use": "Volatility calculation."},
          {"path": "GET /v5/position/list", "use": "Inventory, unrealizedPnl."},
          {"path": "POST /v5/order/create", "use": "Conditional order placement."},
          {"path": "GET /v5/execution/list", "use": "Trade metrics."}
        ],
        "websocket": [
          {"topic": "orderbook.50.<symbol>", "use": "Real-time depth updates."},
          {"topic": "publicTrade.<symbol>", "use": "Momentum calculation."},
          {"topic": "execution", "use": "Trade metrics updates."},
          {"topic": "position", "use": "Unrealized PNL updates."}
        ]
      }
    },
    "functions_and_implementations": [
      {
        "function": "AdvancedMarketMakingStrategy",
        "file": "twin-range-bot/src/strategies/AdvancedMarketMakingStrategy.ts",
        "description": "Calculates conditional order prices and quantities.",
        "methods": [
          {"name": "calculateOrderPrices", "parameters": ["referencePrice: number", "volatility: number", "inventory: number", "maxInventory: number", "recentTrades: number[]", "orderbook?: OrderbookData"], "description": "Returns prices, quantities, and profitability check."}
        ]
      },
      {
        "function": "MarketMakingBot",
        "file": "twin-range-bot/src/core/bot.ts",
        "description": "Implements conditional order placement.",
        "methods": [
          {"name": "updateOrders", "parameters": [], "description": "Places orders only if conditions (volatility, depth, momentum, inventory) are met."},
          {"name": "updateProfitAndInventory", "parameters": ["executions: Execution[]"], "description": "Updates trade metrics."},
          {"name": "updateInventoryAndPnl", "parameters": ["position: PositionData"], "description": "Updates inventory, unrealizedPnl."}
        ]
      },
      {
        "function": "App",
        "file": "App.tsx",
        "description": "Displays conditional order status and metrics.",
        "methods": [
          {"name": "useEffect", "parameters": [], "description": "Updates UI with bot state."}
        ]
      }
    ],
    "ideas_and_enhancements": [
      {
        "idea": "Conditional Order Placement",
        "description": "Place orders only when volatility, depth, momentum, and inventory conditions are met.",
        "details": ["Volatility: 0.5-5%, Depth: 0.5-2.0, Momentum: <2%, Inventory: <90% max, Spread > 2 * fees."]
      },
      {
        "idea": "Enhanced Metrics",
        "description": "Track orderStatus, volatility, momentum in UI.",
        "details": ["orderStatus: Idle, Active, Conditions Not Met, etc."]
      }
    ],
    "next_steps": [
      "Apply updated files (bot.ts, App.tsx, types.ts, constants.ts, AdvancedMarketMakingStrategy.ts).",
      "Clear cache (rm -rf node_modules/.cache).",
      "Run npm run start:backtest for TRUMPUSDT.",
      "Verify tradeHistory, daily_pnl, orderStatus in logs/UI.",
      "Test on testnet with REST; switch to mainnet for WebSocket.",
      "Tune minVolatility, maxVolatility, maxMomentum."
    ]
  }
}
```

### Next Steps
1. **Apply Updates**: Replace `bot.ts`, `App.tsx`, `types.ts`, `constants.ts`, and `AdvancedMarketMakingStrategy.ts`.
2. **Clear Cache**: Run `rm -rf node_modules/.cache` or `npx tsc --build --clean`.
3. **Run Backtest**: Execute `npm run start:backtest` and check logs for condition checks, order placements, and metrics.
4. **Verify Metrics**: Ensure `tradeHistory`, `daily_pnl`, `winRate`, and `orderStatus` reflect conditional logic.
5. **Test Live**: Run on testnet (`is_testnet: true`) and monitor `App.tsx` UI.
6. **Debug**: Share backtest logs if metrics or orders are incorrect.

If issues persist, provide the latest backtest output or specific errors!
