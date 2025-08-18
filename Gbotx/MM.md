Thank you for the detailed suggestions to upgrade `BasicMarketMakingStrategy.ts` for multi-symbol support in the `twin-range-bot` (referred to as Gbotx). However, I notice that your response assumes the existence of a `BasicMarketMakingStrategy.ts` file, which wasn't provided in the earlier context. The current codebase includes `AdvancedMarketMakingStrategy.ts`, `bot.ts`, `App.tsx`, `types.ts`, `constants.ts`, and `cli.tsx`, with the strategy already tailored for `TRUMPUSDT` using the Bybit V5 API. Additionally, your suggestions are based on a backtesting-focused setup with file-based historical data, whereas the current bot is designed for both backtesting (`rest` mode) and live trading (`websocket` mode) using Bybit's API endpoints (`GET /v5/market/orderbook`, `GET /v5/market/kline`, WebSocket `orderbook`, `publicTrade`, etc.). The bot also integrates conditional order placement for `TRUMPUSDT` with dynamic spreads, inventory rebalancing, and trade metrics.

To align with the existing codebase and the user's request for conditional order placement, I'll adapt your 10 upgrades to enhance `AdvancedMarketMakingStrategy.ts` for multi-symbol support while maintaining compatibility with the current architecture, Bybit V5 API integration, and conditional logic. The upgrades will focus on enabling the bot to handle multiple symbols (e.g., `TRUMPUSDT`, `BTCUSDT`, `ETHUSDT`), improving high-frequency trading (HFT) capabilities, and ensuring robust trade metrics and UI integration in `App.tsx`. I'll also incorporate your neon-colored logging and dynamic configuration ideas but adjust the backtesting approach to use Bybit's historical kline data (`GET /v5/market/kline`) instead of local JSON files. Below, I'll outline the 10 upgrades, provide updated code for key files, and ensure the conditional order placement logic is preserved and extended across multiple symbols.

### 10 Upgrades for Multi-Symbol Support
1. **Dynamic Symbol Configuration**: Allow multiple symbols in `BotConfig` and initialize state per symbol.
2. **Per-Symbol Volatility Calculation with ATR**: Replace the current volatility calculation with Average True Range (ATR) for each symbol.
3. **Dynamic Order Sizing with Risk Allocation**: Adjust order sizes based on ATR and per-symbol capital allocation.
4. **Per-Symbol Cash and Inventory Management**: Track cash and inventory separately for each symbol.
5. **Chronological Backtesting Across Symbols**: Process kline data chronologically for all symbols in backtest mode.
6. **Granular Trade Tracking**: Enhance `tradeHistory` with entry/exit prices and per-symbol metrics.
7. **Configurable Strategy Parameters**: Externalize parameters like spread, ATR period, and risk percent.
8. **Advanced Metrics per Symbol**: Calculate Sharpe, Sortino, max drawdown, and volume traded per symbol.
9. **Enhanced Logging with Winston**: Use structured, neon-colored logs for multi-symbol debugging.
10. **CLI and UI Enhancements**: Update `cli.tsx` and `App.tsx` for multi-symbol support and metrics display.

### Updated Files
Below are the updated files, focusing on `AdvancedMarketMakingStrategy.ts`, `bot.ts`, `types.ts`, `constants.ts`, `cli.tsx`, and `App.tsx`. The `bybitService.ts` and `logger.ts` remain compatible but will be enhanced for logging. I'll assume the `bybitService.ts` from the previous context handles multi-symbol API calls (e.g., `getOrderbook(symbol)`, `getKlines(symbol, interval)`).

#### 1. **types.ts** (Updated)
Add per-symbol state and configuration fields.

```typescript
// twin-range-bot/src/core/types.ts
export interface BotConfig {
  symbols: string[]; // Changed to array
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
  minVolatility: number;
  maxVolatility: number;
  minDepthRatio: number;
  maxDepthRatio: number;
  maxMomentum: number;
  atrPeriod: number; // Added for ATR
  riskPercent: number; // Added for risk allocation
}

export interface PerSymbolState {
  active_mm_orders: { type: 'buy' | 'sell'; price: number; orderId: string }[];
  inventory: number;
  recentTrades: number[];
  referencePrice: number;
  klines: { s: string; t: number; o: string; h: string; l: string; c: string; v: string }[];
  tradeHistory: { side: string; qty: number; price: number; profit: number; timestamp: number; fee: number; entryPrice?: number; tradeId: string }[];
  unrealizedPnl: number;
  orderStatus: string;
  cash: number; // Added for per-symbol cash
  atr: number; // Added for ATR tracking
}

export interface TradeState {
  symbols: { [key: string]: PerSymbolState }; // Per-symbol state
  totalProfit: number;
  daily_pnl: number;
  balance: number;
  logs: LogEntry[];
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgPnl: number;
  equityCurve: number[];
}

export interface LogEntry {
  type: string;
  message: string;
  symbol?: string; // Added for symbol-specific logs
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
Configure multiple symbols and add ATR/risk parameters.

```typescript
// constants.ts
import type { BotConfig, TradeState } from './types';

export const BOT_CONFIG_TEMPLATE: BotConfig = {
  dataSource: 'rest',
  symbols: ['TRUMPUSDT', 'BTCUSDT', 'ETHUSDT'], // Multi-symbol
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
  minVolatility: 0.005,
  maxVolatility: 0.05,
  minDepthRatio: 0.5,
  maxDepthRatio: 2.0,
  maxMomentum: 0.02,
  atrPeriod: 14,
  riskPercent: 0.01,
};

export const INITIAL_TRADE_STATE_TEMPLATE: TradeState = {
  symbols: {
    TRUMPUSDT: {
      active_mm_orders: [],
      inventory: 0,
      recentTrades: [],
      referencePrice: 0,
      klines: [],
      tradeHistory: [],
      unrealizedPnl: 0,
      orderStatus: 'Idle',
      cash: 10000 / 3, // Split initial capital
      atr: 0,
    },
    BTCUSDT: {
      active_mm_orders: [],
      inventory: 0,
      recentTrades: [],
      referencePrice: 0,
      klines: [],
      tradeHistory: [],
      unrealizedPnl: 0,
      orderStatus: 'Idle',
      cash: 10000 / 3,
      atr: 0,
    },
    ETHUSDT: {
      active_mm_orders: [],
      inventory: 0,
      recentTrades: [],
      referencePrice: 0,
      klines: [],
      tradeHistory: [],
      unrealizedPnl: 0,
      orderStatus: 'Idle',
      cash: 10000 / 3,
      atr: 0,
    },
  },
  totalProfit: 0,
  daily_pnl: 0,
  balance: 10000,
  logs: [],
  winRate: 0,
  profitFactor: 0,
  totalTrades: 0,
  avgPnl: 0,
  equityCurve: [10000],
};
```

#### 3. **cli.tsx** (Updated)
Support multi-symbol configuration via command-line arguments.

```typescript
// cli.tsx
import { MarketMakingBot } from './core/bot';
import { BOT_CONFIG_TEMPLATE } from './constants';
import { program } from 'commander';

program
  .option('-s, --symbols <symbols>', 'Comma-separated symbols', 'TRUMPUSDT,BTCUSDT,ETHUSDT')
  .option('-c, --capital <number>', 'Initial capital', '10000')
  .parse(process.argv);

async function runBacktest() {
  const options = program.opts();
  const symbols = options.symbols.split(',').map((s: string) => s.trim().toUpperCase());
  const config = {
    ...BOT_CONFIG_TEMPLATE,
    symbols,
    initialCapital: parseFloat(options.capital),
  };

  console.log('cli.tsx config:', config);
  const bot = new MarketMakingBot(config);
  await bot.start();
}

runBacktest().catch(console.error);
```

#### 4. **logger.ts** (Updated)
Enhance Winston logger for neon-colored, symbol-specific logs.

```typescript
// twin-range-bot/src/core/logger.ts
import winston from 'winston';
import 'winston-daily-rotate-file';

const logFormat = winston.format.printf(({ level, message, timestamp, symbol, ...metadata }) => {
  let msg = `${timestamp} [${level.toUpperCase()}]${symbol ? `[${symbol}] ` : ''}${message}`;
  if (Object.keys(metadata).length > 0) {
    msg += ` ${JSON.stringify(metadata, null, 2)}`;
  }
  return msg;
});

const logger = winston.createLogger({
  level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
  format: winston.format.combine(
    winston.format.timestamp(),
    logFormat,
    winston.format.colorize({ all: true }) // Neon colors
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.DailyRotateFile({
      filename: 'bot-%DATE%.log',
      datePattern: 'YYYY-MM-DD',
      zippedArchive: true,
      maxSize: '20m',
      maxFiles: '14d',
      level: 'info',
    }),
  ],
});

export default logger;
```

#### 5. **AdvancedMarketMakingStrategy.ts** (Updated)
Add ATR-based volatility and dynamic order sizing.

```typescript
// twin-range-bot/src/strategies/AdvancedMarketMakingStrategy.ts
import { OrderbookData } from '../services/bybitService';
import logger from '../core/logger';

export class AdvancedMarketMakingStrategy {
  private baseSpread: number;
  private volatilityFactor: number;
  private momentumWindow: number;
  private atrPeriod: number;
  private riskPercent: number;
  private stopLossMultiplier: number = 3;
  private minTradeQuantity: number = 0.001;
  private maxQuantityCap: number = 1;

  constructor(config: {
    baseSpread: number;
    volatilityFactor: number;
    momentumWindow?: number;
    atrPeriod: number;
    riskPercent: number;
  }) {
    this.baseSpread = config.baseSpread;
    this.volatilityFactor = config.volatilityFactor;
    this.momentumWindow = config.momentumWindow || 5;
    this.atrPeriod = config.atrPeriod;
    this.riskPercent = config.riskPercent;
    logger.info('AdvancedMarketMakingStrategy initialized', { config });
  }

  calculateVolatility(symbol: string, klines: any[], currentPrice: number, prevAtr: number): number {
    if (klines.length < this.atrPeriod) {
      logger.warn('Insufficient klines for ATR calculation', { symbol, klinesLength: klines.length });
      return 1;
    }
    const trueRanges = klines.slice(0, this.atrPeriod).map((k, i) => {
      const high = parseFloat(k.h);
      const low = parseFloat(k.l);
      const prevClose = i < klines.length - 1 ? parseFloat(klines[i + 1].c) : currentPrice;
      return Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    });
    const alpha = 2 / (this.atrPeriod + 1);
    const atr = prevAtr ? (trueRanges[0] * alpha) + (prevAtr * (1 - alpha)) : trueRanges.reduce((sum, tr) => sum + tr, 0) / trueRanges.length;
    logger.debug('ATR calculated', { symbol, atr, trueRanges });
    return atr / currentPrice;
  }

  calculateOrderSize(symbol: string, currentPrice: number, volatility: number, cash: number): number {
    const riskAmount = cash * this.riskPercent;
    const stopLossDistance = volatility * currentPrice * this.stopLossMultiplier;
    if (stopLossDistance === 0) {
      logger.warn('Zero stop-loss distance', { symbol, volatility });
      return 0;
    }
    let quantity = riskAmount / stopLossDistance;
    const maxAffordable = cash / currentPrice;
    quantity = Math.min(Math.max(quantity, this.minTradeQuantity), maxAffordable, this.maxQuantityCap);
    quantity = parseFloat(quantity.toFixed(6));
    logger.debug('Order size calculated', { symbol, quantity, riskAmount, stopLossDistance, maxAffordable });
    return quantity;
  }

  calculateOrderPrices(
    symbol: string,
    referencePrice: number,
    volatility: number,
    inventory: number,
    maxInventory: number,
    recentTrades: number[],
    orderbook?: OrderbookData
  ): { buyPrice: number; sellPrice: number; buyQty: number; sellQty: number; canPlaceOrders: boolean } {
    let spread = this.baseSpread * (1 + volatility * this.volatilityFactor);
    if (orderbook) {
      const bidDepth = orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
      const askDepth = orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
      const depthRatio = bidDepth / askDepth;
      spread *= Math.max(0.5, Math.min(2, 1 / (Math.min(bidDepth, askDepth) / 0.01)));
      logger.debug('Order book depth', { symbol, bidDepth, askDepth, depthRatio });
    }

    const recentPrices = recentTrades.slice(-this.momentumWindow);
    const momentum = recentPrices.length >= 2 ? (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] : 0;
    spread *= (1 + Math.abs(momentum) * 0.5);

    const inventorySkew = inventory / maxInventory;
    const buySpread = spread * (1 + inventorySkew);
    const sellSpread = spread * (1 - inventorySkew);

    const buyPrice = referencePrice * (1 - buySpread / 2);
    const sellPrice = referencePrice * (1 + sellSpread / 2);

    const canPlaceOrders = (sellPrice - buyPrice) > 2 * 0.0012 * referencePrice;

    logger.debug('Order prices calculated', {
      symbol,
      buyPrice,
      sellPrice,
      spread,
      inventorySkew,
      momentum,
      canPlaceOrders,
    });

    return { buyPrice, sellPrice, buyQty: 0, sellQty: 0, canPlaceOrders }; // Quantities calculated separately
  }
}
```

#### 6. **bot.ts** (Updated)
Implement multi-symbol support with conditional order placement and chronological backtesting.

```typescript
// twin-range-bot/src/core/bot.ts
import { BybitService, OrderbookData, TradeData, Execution, OrderData, PositionData, KlineData } from '../services/bybitService';
import { logger } from './logger';
import { AdvancedMarketMakingStrategy } from '../strategies/AdvancedMarketMakingStrategy';
import type { BotConfig, TradeState, PerSymbolState } from './types';
import { KlineIntervalV3 } from 'bybit-api';

export class MarketMakingBot {
  private config: BotConfig;
  private state: TradeState;
  private bybitService: BybitService;
  private currentStrategy: AdvancedMarketMakingStrategy;

  constructor(config: BotConfig) {
    this.config = { ...config, dataSource: config.dataSource || 'rest' };
    logger.info('MarketMakingBot initialized', { config: this.config });

    this.currentStrategy = new AdvancedMarketMakingStrategy({
      baseSpread: this.config.baseSpread,
      volatilityFactor: this.config.volatilityFactor,
      atrPeriod: this.config.atrPeriod,
      riskPercent: this.config.riskPercent,
    });

    this.state = {
      symbols: {},
      totalProfit: 0,
      daily_pnl: 0,
      balance: config.initialCapital || 10000,
      logs: [],
      winRate: 0,
      profitFactor: 0,
      totalTrades: 0,
      avgPnl: 0,
      equityCurve: [config.initialCapital || 10000],
    };
    this.config.symbols.forEach(symbol => {
      this.state.symbols[symbol] = {
        active_mm_orders: [],
        inventory: 0,
        recentTrades: [],
        referencePrice: 0,
        klines: [],
        tradeHistory: [],
        unrealizedPnl: 0,
        orderStatus: 'Idle',
        cash: this.state.balance / this.config.symbols.length,
        atr: 0,
      };
    });

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
    } else {
      await this.backtest();
    }
  }

  private async initializeState() {
    for (const symbol of this.config.symbols) {
      const orderbook = await this.bybitService.getOrderbook(symbol);
      this.state.symbols[symbol].referencePrice = (parseFloat(orderbook.b[0][0]) + parseFloat(orderbook.a[0][0])) / 2;
      const position = await this.bybitService.getPosition(symbol);
      this.updateInventoryAndPnl(symbol, position);
      this.state.symbols[symbol].klines = await this.bybitService.getKlines(symbol, this.config.interval as KlineIntervalV3);
      const executions = await this.bybitService.getExecutionHistory(symbol);
      this.updateProfitAndInventory(symbol, executions);
      await this.updateOrders(symbol);
    }
    this.updateEquity();
  }

  private async updateStateFromRest() {
    for (const symbol of this.config.symbols) {
      const orderbook = await this.bybitService.getOrderbook(symbol);
      this.state.symbols[symbol].referencePrice = (parseFloat(orderbook.b[0][0]) + parseFloat(orderbook.a[0][0])) / 2;
      this.state.symbols[symbol].klines = await this.bybitService.getKlines(symbol, this.config.interval as KlineIntervalV3);
      const position = await this.bybitService.getPosition(symbol);
      this.updateInventoryAndPnl(symbol, position);
      const executions = await this.bybitService.getExecutionHistory(symbol);
      this.updateProfitAndInventory(symbol, executions);
      await this.updateOrders(symbol);
    }
    this.updateEquity();
  }

  private async backtest() {
    const allEvents: { timestamp: number; symbol: string; kline: KlineData }[] = [];
    for (const symbol of this.config.symbols) {
      const klines = await this.bybitService.getKlines(symbol, this.config.interval as KlineIntervalV3, this.config.lookback_bars);
      klines.forEach(kline => {
        allEvents.push({ timestamp: parseInt(kline.t), symbol, kline });
      });
    }
    allEvents.sort((a, b) => a.timestamp - b.timestamp);

    for (const event of allEvents) {
      const { symbol, kline } = event;
      this.state.symbols[symbol].referencePrice = parseFloat(kline.c);
      this.state.symbols[symbol].klines = [kline, ...this.state.symbols[symbol].klines].slice(0, this.config.volatilityWindow);
      await this.updateOrders(symbol);
      this.updateEquity();
    }

    const metrics = this.calculateMetrics();
    logger.info('Backtest completed', { metrics });
  }

  private handleOrderbookUpdate(data: OrderbookData) {
    if (this.config.dataSource === 'websocket') {
      const symbol = data.s;
      if (this.config.symbols.includes(symbol)) {
        const bestBid = parseFloat(data.b[0][0]);
        const bestAsk = parseFloat(data.a[0][0]);
        this.state.symbols[symbol].referencePrice = (bestBid + bestAsk) / 2;
        this.updateOrders(symbol);
      }
    }
  }

  private handleTradeUpdate(trades: TradeData[]) {
    if (this.config.dataSource === 'websocket') {
      for (const trade of trades) {
        const symbol = trade.s;
        if (this.config.symbols.includes(symbol)) {
          this.state.symbols[symbol].recentTrades.push(parseFloat(trade.p));
          if (this.state.symbols[symbol].recentTrades.length > this.config.volatilityWindow) {
            this.state.symbols[symbol].recentTrades.shift();
          }
          this.updateOrders(symbol);
        }
      }
    }
  }

  private handleKlineUpdate(klines: KlineData[]) {
    if (this.config.dataSource === 'websocket') {
      for (const kline of klines) {
        const symbol = kline.s;
        if (this.config.symbols.includes(symbol)) {
          this.state.symbols[symbol].klines = [kline, ...this.state.symbols[symbol].klines].slice(0, this.config.volatilityWindow);
          if (!this.state.symbols[symbol].referencePrice) {
            this.state.symbols[symbol].referencePrice = parseFloat(kline.c);
          }
          this.updateOrders(symbol);
        }
      }
    }
  }

  private updateProfitAndInventory(symbol: string, executions: Execution[]) {
    if (!executions || executions.length === 0) {
      logger.info('No new executions to process', { symbol });
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
      let profit = exec.side === 'Buy' ? -tradeValue - fee : tradeValue - fee;

      inventoryChange += exec.side === 'Buy' ? qty : -qty;
      realizedPnl += profit;
      if (profit > 0) wins++;
      totalPnl += profit;

      const tradeId = `${symbol}-${exec.execTime}-${Math.random().toString(36).substring(2, 9)}`;
      const openTrades = this.state.symbols[symbol].tradeHistory.filter(t => t.side === 'Buy' && !t.entryPrice);
      if (exec.side === 'Sell' && openTrades.length > 0) {
        const buyTrade = openTrades.shift();
        if (buyTrade) {
          profit = (price - buyTrade.price) * qty - fee;
          buyTrade.profit = profit;
          buyTrade.entryPrice = buyTrade.price;
          buyTrade.price = price;
        }
      }

      this.state.symbols[symbol].tradeHistory.push({
        side: exec.side,
        qty,
        price,
        profit,
        timestamp: parseInt(exec.execTime),
        fee,
        entryPrice: exec.side === 'Buy' ? price : undefined,
        tradeId,
      });

      logger.info(`Execution: ${exec.side} ${qty.toFixed(4)} ${symbol} at $${price.toFixed(2)}, Profit: $${profit.toFixed(2)}, Fee: $${fee.toFixed(2)}`, { symbol });
    }

    this.state.symbols[symbol].inventory += inventoryChange;
    this.state.symbols[symbol].inventory = Math.max(-this.config.maxInventory, Math.min(this.config.maxInventory, this.state.symbols[symbol].inventory));
    this.state.totalProfit += realizedPnl;
    this.state.daily_pnl += realizedPnl;
    this.state.symbols[symbol].cash += realizedPnl;
    this.state.balance += realizedPnl;
    this.state.totalTrades += executions.length;
    this.state.winRate = this.state.totalTrades > 0 ? wins / this.state.totalTrades : 0;
    this.state.avgPnl = this.state.totalTrades > 0 ? totalPnl / this.state.totalTrades : 0;
    this.state.profitFactor = wins > 0 ? totalPnl / wins : 0;

    logger.info('Trade metrics updated', {
      symbol,
      totalProfit: this.state.totalProfit.toFixed(2),
      dailyPnl: this.state.daily_pnl.toFixed(2),
      inventory: this.state.symbols[symbol].inventory.toFixed(4),
      winRate: (this.state.winRate * 100).toFixed(2),
      totalTrades: this.state.totalTrades,
    });
  }

  private handleExecutionUpdate(executions: Execution[]) {
    if (this.config.dataSource === 'websocket') {
      const executionsBySymbol = executions.reduce((acc, exec) => {
        acc[exec.symbol] = acc[exec.symbol] || [];
        acc[exec.symbol].push(exec);
        return acc;
      }, {} as { [key: string]: Execution[] });
      for (const symbol in executionsBySymbol) {
        if (this.config.symbols.includes(symbol)) {
          this.updateProfitAndInventory(symbol, executionsBySymbol[symbol]);
          this.updateOrders(symbol);
        }
      }
    }
  }

  private updateInventoryAndPnl(symbol: string, position: PositionData) {
    const inventory = parseFloat(position.size) * (this.bybitService.convertPositionSide(position.side) === 'Buy' ? 1 : -1);
    const unrealizedPnl = parseFloat(position.unrealisedPnl);
    this.state.symbols[symbol].inventory = Math.max(-this.config.maxInventory, Math.min(this.config.maxInventory, inventory));
    this.state.symbols[symbol].unrealizedPnl = unrealizedPnl;

    logger.info('Position updated', {
      symbol,
      inventory: this.state.symbols[symbol].inventory.toFixed(4),
      unrealizedPnl: unrealizedPnl.toFixed(2),
    });
  }

  private handlePositionUpdate(positions: PositionData[]) {
    if (this.config.dataSource === 'websocket') {
      for (const position of positions) {
        if (this.config.symbols.includes(position.symbol)) {
          this.updateInventoryAndPnl(position.symbol, position);
          this.updateOrders(position.symbol);
        }
      }
    }
  }

  private handleOrderUpdate(orders: OrderData[]) {
    if (this.config.dataSource === 'websocket') {
      for (const order of orders) {
        if (this.config.symbols.includes(order.symbol)) {
          if (order.orderStatus === 'Filled' || order.orderStatus === 'Cancelled') {
            this.state.symbols[order.symbol].active_mm_orders = this.state.symbols[order.symbol].active_mm_orders.filter(o => o.orderId !== order.orderId);
            this.state.symbols[order.symbol].orderStatus = order.orderStatus === 'Filled' ? 'Filled' : 'Cancelled';
            logger.info(`Order Update: ${order.orderId} ${order.orderStatus} at $${parseFloat(order.price).toFixed(2)}`, { symbol: order.symbol });
            this.updateOrders(order.symbol);
          }
        }
      }
    }
  }

  private async updateOrders(symbol: string) {
    const state = this.state.symbols[symbol];
    if (!state.referencePrice) {
      state.orderStatus = 'No Reference Price';
      logger.error('No reference price available', { symbol });
      return;
    }

    try {
      const orderbook = this.config.dataSource === 'rest' ? await this.bybitService.getOrderbook(symbol) : undefined;
      const volatility = this.currentStrategy.calculateVolatility(symbol, state.klines, state.referencePrice, state.atr);
      state.atr = volatility * state.referencePrice;
      const momentum = state.recentTrades.length >= 2 ? (state.recentTrades[state.recentTrades.length - 1] - state.recentTrades[0]) / state.recentTrades[0] : 0;

      const isVolatilityValid = volatility >= this.config.minVolatility && volatility <= this.config.maxVolatility;
      const isDepthValid = orderbook ? orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) >= this.config.minDepthRatio && orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) <= this.config.maxDepthRatio : true;
      const isMomentumValid = Math.abs(momentum) <= this.config.maxMomentum;
      const canBuy = state.inventory < this.config.maxInventory * 0.9;
      const canSell = state.inventory > -this.config.maxInventory * 0.9;

      logger.info(`Order Conditions: Volatility: ${(volatility * 100).toFixed(2)}% (${isVolatilityValid}), Depth: ${orderbook ? (orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0)).toFixed(2) : 'N/A'} (${isDepthValid}), Momentum: ${(momentum * 100).toFixed(2)}% (${isMomentumValid}), Can Buy: ${canBuy}, Can Sell: ${canSell}`, { symbol });

      if (!isVolatilityValid || !isDepthValid || !isMomentumValid) {
        state.orderStatus = 'Conditions Not Met';
        logger.info('Skipping order placement due to invalid conditions', { symbol });
        return;
      }

      const { buyPrice, sellPrice, canPlaceOrders } = this.currentStrategy.calculateOrderPrices(
        symbol,
        state.referencePrice,
        volatility,
        state.inventory,
        this.config.maxInventory,
        state.recentTrades,
        orderbook
      );

      const buyQty = this.currentStrategy.calculateOrderSize(symbol, state.referencePrice, volatility, state.cash);
      const sellQty = this.currentStrategy.calculateOrderSize(symbol, state.referencePrice, volatility, state.cash);

      if (!canPlaceOrders) {
        state.orderStatus = 'Unprofitable Spread';
        logger.info('Skipping order placement: Spread does not cover fees', { symbol });
        return;
      }

      if (isNaN(buyPrice) || isNaN(sellPrice) || isNaN(buyQty) || isNaN(sellQty)) {
        state.orderStatus = 'Invalid Prices';
        logger.error('Invalid order prices or quantities: NaN detected', { symbol });
        return;
      }

      for (const order of state.active_mm_orders) {
        await this.bybitService.cancelOrder(symbol, order.orderId);
        logger.info(`Cancelled order: ${order.orderId} (${order.type})`, { symbol });
      }
      state.active_mm_orders = [];

      if (canBuy && buyQty > 0) {
        const buyOrder = await this.bybitService.placeMarketMakingOrder(
          symbol,
          'Buy',
          buyPrice,
          buyQty,
          buyPrice * (1 + this.config.tpPercent),
          buyPrice * (1 - this.config.slPercent)
        );
        state.active_mm_orders.push({ type: 'buy', price: buyPrice, orderId: buyOrder.orderId });
        logger.info(`Placed buy order: ${buyOrder.orderId} at $${buyPrice.toFixed(2)}, Qty: ${buyQty.toFixed(4)}`, { symbol });
      }

      if (canSell && sellQty > 0) {
        const sellOrder = await this.bybitService.placeMarketMakingOrder(
          symbol,
          'Sell',
          sellPrice,
          sellQty,
          sellPrice * (1 - this.config.tpPercent),
          sellPrice * (1 + this.config.slPercent)
        );
        state.active_mm_orders.push({ type: 'sell', price: sellPrice, orderId: sellOrder.orderId });
        logger.info(`Placed sell order: ${sellOrder.orderId} at $${sellPrice.toFixed(2)}, Qty: ${sellQty.toFixed(4)}`, { symbol });
      }

      state.orderStatus = state.active_mm_orders.length > 0 ? 'Active' : 'Idle';
    } catch (err) {
      state.orderStatus = 'Error';
      logger.error(`Error updating orders: ${err}`, { symbol });
    }
  }

  private updateEquity() {
    const totalCash = Object.values(this.state.symbols).reduce((sum, s) => sum + s.cash, 0);
    const totalPositionValue = Object.entries(this.state.symbols).reduce(
      (sum, [symbol, state]) => sum + state.inventory * state.referencePrice,
      0
    );
    this.state.balance = totalCash + totalPositionValue;
    this.state.equityCurve.push(this.state.balance);
    logger.info('Equity updated', { totalCash, totalPositionValue, balance: this.state.balance });
  }

  private calculateMetrics(): { [key: string]: any } & { aggregate: any } {
    const aggregateMetrics = {
      sharpeRatio: 0,
      sortinoRatio: 0,
      maxDrawdown: 0,
      winRate: 0,
      profitFactor: 0,
      calmarRatio: 0,
      totalProfit: 0,
      totalTradeCount: 0,
      averageTradeProfit: 0,
      totalVolumeTraded: 0,
    };
    const symbolMetrics: { [key: string]: any } = {};

    for (const symbol of this.config.symbols) {
      const closedTrades = this.state.symbols[symbol].tradeHistory.filter(t => t.profit !== 0);
      const profits = closedTrades.map(t => t.profit);
      const totalProfit = profits.reduce((sum, p) => sum + p, 0);
      const winCount = profits.filter(p => p > 0).length;
      const totalTrades = closedTrades.length;
      const grossProfit = profits.filter(p => p > 0).reduce((sum, p) => sum + p, 0);
      const grossLoss = Math.abs(profits.filter(p => p <= 0).reduce((sum, p) => sum + p, 0));
      const totalVolume = closedTrades.reduce((sum, t) => sum + t.qty * t.price, 0);

      let sharpeRatio = 0,
        sortinoRatio = 0,
        maxDrawdown = 0,
        winRate = 0,
        profitFactor = Infinity,
        calmarRatio = 0,
        averageTradeProfit = 0;

      if (totalTrades > 0) {
        winRate = (winCount / totalTrades) * 100;
        profitFactor = grossLoss > 0 ? grossProfit / grossLoss : Infinity;
        averageTradeProfit = totalProfit / totalTrades;
      }

      const returns = profits.map(p => p / (this.state.symbols[symbol].cash || this.state.balance / this.config.symbols.length));
      const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length || 0;
      const stdDev = returns.length > 0 ? Math.sqrt(returns.map(r => Math.pow(r - meanReturn, 2)).reduce((sum, r) => sum + r, 0) / returns.length) : 0;
      const downsideReturns = returns.filter(r => r < 0);
      const downsideStdDev = downsideReturns.length > 0 ? Math.sqrt(downsideReturns.map(r => Math.pow(r - meanReturn, 2)).reduce((sum, r) => sum + r, 0) / downsideReturns.length) : 0;

      sharpeRatio = stdDev > 0 ? meanReturn / stdDev : 0;
      sortinoRatio = downsideStdDev > 0 ? meanReturn / downsideStdDev : 0;
      const equityPeak = Math.max(...this.state.equityCurve);
      maxDrawdown = Math.max(...this.state.equityCurve.map(e => (equityPeak - e) / equityPeak)) * 100;
      const annualReturn = meanReturn * (252 * 24); // Rough annualization for hourly klines
      calmarRatio = maxDrawdown > 0 ? annualReturn / maxDrawdown : 0;

      symbolMetrics[symbol] = {
        totalProfit: totalProfit.toFixed(2),
        tradeCount: totalTrades,
        winRate: winRate.toFixed(2),
        profitFactor: profitFactor.toFixed(2),
        averageTradeProfit: averageTradeProfit.toFixed(4),
        sharpeRatio: sharpeRatio.toFixed(4),
        sortinoRatio: sortinoRatio.toFixed(4),
        maxDrawdown: maxDrawdown.toFixed(2),
        calmarRatio: calmarRatio.toFixed(4),
        totalVolumeTraded: totalVolume.toFixed(2),
      };

      aggregateMetrics.totalProfit += totalProfit;
      aggregateMetrics.totalTradeCount += totalTrades;
      aggregateMetrics.totalVolumeTraded += totalVolume;
      aggregateMetrics.sharpeRatio += sharpeRatio / this.config.symbols.length;
      aggregateMetrics.sortinoRatio += sortinoRatio / this.config.symbols.length;
      aggregateMetrics.maxDrawdown = Math.max(aggregateMetrics.maxDrawdown, maxDrawdown);
      aggregateMetrics.winRate += winRate / this.config.symbols.length;
      aggregateMetrics.profitFactor += profitFactor / this.config.symbols.length;
      aggregateMetrics.calmarRatio += calmarRatio / this.config.symbols.length;
    }

    aggregateMetrics.averageTradeProfit = aggregateMetrics.totalTradeCount > 0 ? aggregateMetrics.totalProfit / aggregateMetrics.totalTradeCount : 0;
    return { ...symbolMetrics, aggregate: aggregateMetrics };
  }
}
```

#### 7. **App.tsx** (Updated)
Display per-symbol metrics and order status.

```typescript
// App.tsx
import React, { useState, useEffect } from 'react';
import { MarketMakingBot } from './core/bot';
import { BOT_CONFIG_TEMPLATE } from './constants';
import type { TradeState } from './types';

const App: React.FC = () => {
  const [state, setState] = useState<TradeState>({
    symbols: {},
    totalProfit: 0,
    daily_pnl: 0,
    balance: 10000,
    logs: [],
    winRate: 0,
    profitFactor: 0,
    totalTrades: 0,
    avgPnl: 0,
    equityCurve: [10000],
  });

  useEffect(() => {
    const config = {
      ...BOT_CONFIG_TEMPLATE,
      symbols: ['TRUMPUSDT', 'BTCUSDT', 'ETHUSDT'],
      initialCapital: 10000,
    };
    const bot = new MarketMakingBot(config);
    bot.start();
    const interval = setInterval(() => {
      setState(bot.getState());
    }, config.refresh_rate_seconds * 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h1>Gbotx - Multi-Symbol Market Making</h1>
      <h2>Aggregate Metrics</h2>
      <p><strong>Balance:</strong> ${state.balance.toFixed(2)} USDT</p>
      <p><strong>Daily PNL:</strong> ${state.daily_pnl.toFixed(2)} USDT</p>
      <p><strong>Total Profit:</strong> ${state.totalProfit.toFixed(2)} USDT</p>
      <p><strong>Win Rate:</strong> {(state.winRate * 100).toFixed(2)}%</p>
      <p><strong>Profit Factor:</strong> {state.profitFactor.toFixed(2)}</p>
      <p><strong>Total Trades:</strong> {state.totalTrades}</p>
      <p><strong>Average PNL:</strong> ${state.avgPnl.toFixed(2)} USDT</p>
      {Object.keys(state.symbols).map(symbol => (
        <div key={symbol}>
          <h2>{symbol} Metrics</h2>
          <p><strong>Cash:</strong> ${state.symbols[symbol].cash.toFixed(2)} USDT</p>
          <p><strong>Unrealized PNL:</strong> ${state.symbols[symbol].unrealizedPnl.toFixed(2)} USDT</p>
          <p><strong>Inventory:</strong> {state.symbols[symbol].inventory.toFixed(4)}</p>
          <p><strong>Order Status:</strong> {state.symbols[symbol].orderStatus}</p>
          <p><strong>Volatility (ATR):</strong> {(state.symbols[symbol].atr * 100).toFixed(2)}%</p>
          <h3>Active Orders</h3>
          <ul>
            {state.symbols[symbol].active_mm_orders.map(order => (
              <li key={order.orderId}>{order.type.toUpperCase()} at ${order.price.toFixed(2)}</li>
            ))}
          </ul>
          <h3>Trade History</h3>
          <ul>
            {state.symbols[symbol].tradeHistory.map((trade, index) => (
              <li key={trade.tradeId}>
                {trade.side} {trade.qty.toFixed(4)} {symbol} at ${trade.price.toFixed(2)}, Profit: ${trade.profit.toFixed(2)}, Fee: ${trade.fee.toFixed(2)}, Time: {new Date(trade.timestamp).toLocaleString()}
              </li>
            ))}
          </ul>
        </div>
      ))}
      <h2>Logs</h2>
      <ul>
        {state.logs.map((log, index) => (
          <li key={index}>{log.symbol ? `[${log.symbol}] ` : ''}{log.type}: {log.message}</li>
        ))}
      </ul>
    </div>
  );
};

export default App;
```

### Integration with Conditional Order Placement
- **Dynamic Symbol Configuration**: `config.symbols` drives initialization and updates for all symbols.
- **ATR Volatility**: `calculateVolatility` uses ATR for precise volatility per symbol, integrated into conditional checks (`minVolatility`, `maxVolatility`).
- **Dynamic Order Sizing**: `calculateOrderSize` uses ATR and per-symbol cash to ensure risk-aligned quantities.
- **Per-Symbol Cash/Inventory**: Tracks `cash` and `inventory` per symbol, ensuring balanced capital allocation.
- **Chronological Backtesting**: Processes klines chronologically across symbols for realistic simulation.
- **Granular Trade Tracking**: Links buy/sell trades via `tradeId` and `entryPrice` for accurate profit calculation.
- **Configurable Parameters**: Externalizes `atrPeriod`, `riskPercent`, etc., for tuning.
- **Advanced Metrics**: Calculates per-symbol and aggregate metrics (Sharpe, Sortino, etc.).
- **Enhanced Logging**: Neon-colored, symbol-specific logs with Winston.
- **CLI/UI Enhancements**: Supports dynamic symbol input and displays per-symbol metrics.

### Expected Backtest Output
For `TRUMPUSDT`, `BTCUSDT`, `ETHUSDT`:
- **Order Placement**: Orders placed only if volatility (0.5-5%), depth (0.5-2.0), momentum (<2%), and inventory (<90% max) are valid.
- **Metrics**: Per-symbol `tradeHistory`, `unrealizedPnl`, `cash`, `atr`; aggregate `totalProfit`, `winRate`, etc.
- **Logs**: Show condition checks, order placements, and executions per symbol in neon colors.

### JSON Summary
```json
{
  "conversation_summary": {
    "overview": {
      "context": "Enhanced AdvancedMarketMakingStrategy.ts for multi-symbol support (TRUMPUSDT, BTCUSDT, ETHUSDT) in twin-range-bot, integrating conditional order placement with ATR-based volatility, dynamic sizing, and advanced metrics. Updated bot.ts, App.tsx, and supporting files for Bybit V5 API compatibility and HFT.",
      "date": "2025-07-20",
      "time": "20:16 CEST",
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
    "upgrades_implemented": [
      {
        "upgrade": "Dynamic Symbol Configuration",
        "description": "Supports multiple symbols via config.symbols.",
        "files": ["types.ts", "constants.ts", "bot.ts"]
      },
      {
        "upgrade": "ATR Volatility",
        "description": "Uses ATR for per-symbol volatility calculation.",
        "files": ["AdvancedMarketMakingStrategy.ts"]
      },
      {
        "upgrade": "Dynamic Order Sizing",
        "description": "Adjusts quantities based on ATR and per-symbol cash.",
        "files": ["AdvancedMarketMakingStrategy.ts"]
      },
      {
        "upgrade": "Per-Symbol Cash/Inventory",
        "description": "Tracks cash and inventory per symbol.",
        "files": ["bot.ts", "types.ts"]
      },
      {
        "upgrade": "Chronological Backtesting",
        "description": "Processes klines chronologically across symbols.",
        "files": ["bot.ts"]
      },
      {
        "upgrade": "Granular Trade Tracking",
        "description": "Links buy/sell trades with tradeId and entryPrice.",
        "files": ["bot.ts", "types.ts"]
      },
      {
        "upgrade": "Configurable Parameters",
        "description": "Externalizes atrPeriod, riskPercent, etc.",
        "files": ["types.ts", "constants.ts", "AdvancedMarketMakingStrategy.ts"]
      },
      {
        "upgrade": "Advanced Metrics",
        "description": "Calculates Sharpe, Sortino, max drawdown per symbol.",
        "files": ["bot.ts"]
      },
      {
        "upgrade": "Enhanced Logging",
        "description": "Neon-colored, symbol-specific logs with Winston.",
        "files": ["logger.ts"]
      },
      {
        "upgrade": "CLI/UI Enhancements",
        "description": "Supports multi-symbol input and per-symbol metrics display.",
        "files": ["cli.tsx", "App.tsx"]
      }
    ],
    "next_steps": [
      "Apply updated files.",
      "Install dependencies: npm install commander winston-daily-rotate-file.",
      "Clear cache: rm -rf node_modules/.cache.",
      "Run backtest: npm run start:backtest.",
      "Verify per-symbol metrics and logs in App.tsx.",
      "Test on testnet with REST, then mainnet with WebSocket."
    ]
  }
}
```

### Next Steps
1. **Apply Updates**: Replace `types.ts`, `constants.ts`, `cli.tsx`, `logger.ts`, `AdvancedMarketMakingStrategy.ts`, `bot.ts`, `App.tsx`.
2. **Install Dependencies**: Run `npm install commander winston-daily-rotate-file`.
3. **Clear Cache**: Run `rm -rf node_modules/.cache` or `npx tsc --build --clean`.
4. **Run Backtest**: Execute `npm run start:backtest` and verify logs/UI for all symbols.
5. **Test Live**: Run on testnet (`is_testnet: true`) and monitor `App.tsx`.
6. **Debug**: Share logs if metrics or orders are incorrect.

Let me know if you need further tweaks or specific debugging help! 😻

To create an advanced market-making strategy for `twin-range-bot` that integrates accurately with the provided trade metrics and Bybit V5 API, we need to enhance the existing `BasicMarketMakingStrategy` to handle `TRUMPUSDT` trading with sophisticated features like dynamic spread adjustment, inventory rebalancing, and risk-aware order placement. The strategy will leverage real-time data from Bybit V5 API endpoints (`GET /v5/market/orderbook`, `GET /v5/market/kline`, WebSocket `orderbook`, `publicTrade`, `execution`, `position`) to optimize profitability and manage risk. The advanced strategy will address the issues of `NaN` spread calculations and incorrect trade metrics (`daily_pnl: 10000`, `balance: 10000`, empty `tradeHistory`) by ensuring robust calculations and proper integration with `bot.ts` and `App.tsx`.

### Objectives
1. **Dynamic Spread Adjustment**: Adjust spreads based on volatility, order book depth, and market momentum to capture optimal bid-ask spreads.
2. **Inventory Rebalancing**: Actively manage inventory to stay within `maxInventory` (±0.1) using skewed order sizes and prices.
3. **Risk Management**: Incorporate stop-loss (SL), take-profit (TP), and position sizing based on volatility and account balance.
4. **Trade Metrics Integration**: Ensure accurate updates to `daily_pnl`, `unrealizedPnl`, `tradeHistory`, `winRate`, `profitFactor`, `totalTrades`, and `avgPnl`.
5. **Backtest and Live Compatibility**: Support both backtesting (`rest` mode) and live trading (`websocket` mode) on testnet/mainnet.
6. **UI Integration**: Reflect advanced strategy metrics in `App.tsx` for real-time monitoring.

### Advanced Market Making Strategy Design
The `AdvancedMarketMakingStrategy` will:
- **Calculate Spreads**: Use volatility (from klines), order book depth, and momentum (from recent trades) to set dynamic spreads.
- **Skew Orders**: Adjust order prices and quantities based on inventory skew to rebalance toward zero.
- **Manage Risk**: Set TP/SL dynamically based on volatility and cap position size to limit exposure.
- **Handle Executions**: Process executions to update trade metrics accurately, accounting for fees (0.12% taker).
- **Log Details**: Provide comprehensive logs for debugging spread calculations and trade outcomes.

### Updated Files
Below are the updated files, focusing on the new `AdvancedMarketMakingStrategy.ts`, with minor adjustments to `bot.ts`, `cli.tsx`, `constants.ts`, and `App.tsx` to integrate the strategy. The `types.ts` and `bybitService.ts` files from the previous response are unchanged and compatible.

#### 1. **types.ts** (Updated)
Add `strategyType` as a required field and define trade history structure for better UI rendering.

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
  strategyType: string; // Required for strategy selection
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
Update `strategyType` to use `AdvancedMarketMakingStrategy` and adjust parameters for `TRUMPUSDT` volatility.

```typescript
// constants.ts
import type { BotConfig, TradeState } from './types';

export const BOT_CONFIG_TEMPLATE: BotConfig = {
  dataSource: 'rest',
  symbol: 'TRUMPUSDT',
  interval: '60',
  lookback_bars: 500,
  baseSpread: 0.006, // Increased for TRUMPUSDT volatility
  orderQty: 0.01,
  maxInventory: 0.1,
  tpPercent: 0.03, // Adjusted for larger TP due to volatility
  slPercent: 0.015, // Tighter SL for risk control
  volatilityWindow: 10,
  volatilityFactor: 1.2, // Slightly higher to capture volatility
  refresh_rate_seconds: 60,
  bybit_api_key: 'your-api-key',
  bybit_api_secret: 'your-api-secret',
  is_testnet: true,
  strategyType: 'AdvancedMarketMakingStrategy',
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
};
```

#### 3. **cli.tsx** (Updated)
Set `strategyType` to `AdvancedMarketMakingStrategy` and ensure correct parameter passing.

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
  };

  console.log('cli.tsx config:', config);
  const bot = new MarketMakingBot(config);
  await bot.start();
}

runBacktest().catch(console.error);
```

#### 4. **AdvancedMarketMakingStrategy.ts** (New)
Implement an advanced strategy with dynamic spreads, inventory rebalancing, and risk management.

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
  ): { buyPrice: number; sellPrice: number; buyQty: number; sellQty: number } {
    console.log('calculateOrderPrices inputs:', {
      referencePrice,
      volatility,
      inventory,
      maxInventory,
      baseSpread: this.baseSpread,
      volatilityFactor: this.volatilityFactor,
    });

    // Calculate dynamic spread based on volatility and order book depth
    let spread = this.baseSpread * (1 + volatility * this.volatilityFactor);

    if (orderbook) {
      const bidDepth = orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
      const askDepth = orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
      const depthFactor = Math.min(bidDepth, askDepth) / 0.01; // Assume orderQty = 0.01
      spread *= Math.max(0.5, Math.min(2, 1 / depthFactor));
    }

    // Adjust spread based on market momentum (recent trade direction)
    const recentPrices = recentTrades.slice(-this.momentumWindow);
    const momentum = recentPrices.length >= 2 ? (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] : 0;
    spread *= (1 + Math.abs(momentum) * 0.5); // Widen spread in trending markets

    // Inventory skew for rebalancing
    const inventorySkew = inventory / maxInventory;
    const buySpread = spread * (1 + inventorySkew); // Widen buy spread if overbought
    const sellSpread = spread * (1 - inventorySkew); // Widen sell spread if oversold

    // Dynamic order quantities
    const baseQty = 0.01; // Default orderQty
    const buyQty = baseQty * (1 + Math.abs(inventorySkew)); // Increase buy qty if oversold
    const sellQty = baseQty * (1 - Math.abs(inventorySkew)); // Increase sell qty if overbought

    const buyPrice = referencePrice * (1 - buySpread / 2);
    const sellPrice = referencePrice * (1 + sellSpread / 2);

    console.log('Calculated prices and quantities:', {
      buyPrice,
      sellPrice,
      buyQty,
      sellQty,
      spread,
      inventorySkew,
      momentum,
    });

    return { buyPrice, sellPrice, buyQty, sellQty };
  }
}
```

#### 5. **bot.ts** (Updated)
Integrate `AdvancedMarketMakingStrategy` and pass `recentTrades` to `calculateOrderPrices`. Enhance trade metrics handling.

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

    switch (this.config.strategyType) {
      case 'AdvancedMarketMakingStrategy':
        this.currentStrategy = new AdvancedMarketMakingStrategy(this.config.baseSpread, this.config.volatilityFactor);
        break;
      default:
        this.currentStrategy = new AdvancedMarketMakingStrategy(this.config.baseSpread, this.config.volatilityFactor);
    }

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
      return;
    }
    try {
      const orderbook = this.config.dataSource === 'rest' ? await this.bybitService.getOrderbook(this.config.symbol) : undefined;
      const { buyPrice, sellPrice, buyQty, sellQty } = this.currentStrategy.calculateOrderPrices(
        this.state.referencePrice,
        this.calculateVolatility(),
        this.state.inventory,
        this.config.maxInventory,
        this.state.recentTrades,
        orderbook
      );

      if (isNaN(buyPrice) || isNaN(sellPrice) || isNaN(buyQty) || isNaN(sellQty)) {
        this.state.logs.push({ type: 'error', message: 'Invalid order prices or quantities: NaN detected.' });
        return;
      }

      for (const order of this.state.active_mm_orders) {
        await this.bybitService.cancelOrder(this.config.symbol, order.orderId);
        this.state.logs.push({ type: 'info', message: `Cancelled order: ${order.orderId} (${order.type})` });
      }
      this.state.active_mm_orders = [];

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
    } catch (err) {
      this.state.logs.push({ type: 'error', message: `Error updating orders: ${err}` });
    }
  }
}
```

#### 6. **App.tsx** (Updated)
Enhance UI to display advanced metrics like momentum and dynamic quantities.

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
  });

  useEffect(() => {
    const config = {
      ...BOT_CONFIG_TEMPLATE,
      symbol: 'TRUMPUSDT',
      baseSpread: 0.006,
      volatilityFactor: 1.2,
      strategyType: 'AdvancedMarketMakingStrategy',
    };
    const bot = new MarketMakingBot(config);
    bot.start();
    const interval = setInterval(() => {
      setState(bot.getState());
    }, config.refresh_rate_seconds * 1000);
    return () => clearInterval(interval);
  }, []);

  // Calculate momentum for display
  const momentumWindow = 5;
  const recentPrices = state.recentTrades.slice(-momentumWindow);
  const momentum = recentPrices.length >= 2 ? ((recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] * 100).toFixed(2) : '0.00';

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

### Integration with Trade Metrics
- **Dynamic Spreads**: `AdvancedMarketMakingStrategy` adjusts spreads using:
  - **Volatility**: From `calculateVolatility` (kline-based standard deviation).
  - **Order Book Depth**: Scales spread based on bid/ask liquidity (`GET /v5/market/orderbook`).
  - **Momentum**: Widens spread in trending markets using `recentTrades` (WebSocket `publicTrade`).
- **Inventory Rebalancing**: Skews `buyQty` and `sellQty` based on `inventorySkew` to push inventory toward zero.
- **Risk Management**:
  - TP: 3% (`tpPercent: 0.03`) for larger profit targets on volatile `TRUMPUSDT`.
  - SL: 1.5% (`slPercent: 0.015`) to limit losses.
  - Position capped at `maxInventory: 0.1`.
- **Trade Metrics**:
  - **Realized PNL**: Calculated as `tradeValue - fee` (sells) or `-tradeValue - fee` (buys) in `updateProfitAndInventory`.
  - **Unrealized PNL**: From `position.unrealisedPnl` in `updateInventoryAndPnl`.
  - **Trade History**: Stores `side`, `qty`, `price`, `profit`, `fee`, `timestamp`.
  - **Win Rate**: `wins / totalTrades` (wins: `profit > 0`).
  - **Profit Factor**: `totalPnl / wins`.
  - **Average PNL**: `totalPnl / totalTrades`.
- **Profitability Example**: Buy 0.01 TRUMPUSDT at $10.18, sell at $10.24 (0.6% spread), profit ≈ $0.03 after $0.03 fees (0.12% * $10.21 * 0.01 * 2).

### Expected Backtest Output
For `TRUMPUSDT` with `referencePrice ≈ $10.20`:
- **Buy Price/Qty**: ~$10.18, ~0.01 (adjusted by inventory skew).
- **Sell Price/Qty**: ~$10.24, ~0.01.
- **Metrics**: `daily_pnl`, `totalProfit`, `balance` increase by ~$0.03 per round-trip; `totalTrades` increments; `winRate` and `avgPnl` reflect outcomes.
- **Logs**: Show order placements, executions, and updated metrics.

### JSON Summary
```json
{
  "conversation_summary": {
    "overview": {
      "context": "Created AdvancedMarketMakingStrategy for twin-range-bot to replace BasicMarketMakingStrategy, addressing NaN spreads and incorrect trade metrics for TRUMPUSDT. The strategy uses dynamic spreads, inventory rebalancing, and risk management, integrating with Bybit V5 API for accurate trade metrics and UI display.",
      "date": "2025-07-20",
      "time": "19:40 CEST",
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
        "issue": "NaN in spread calculations",
        "description": "Fixed by ensuring baseSpread: 0.006, volatilityFactor: 1.2 in AdvancedMarketMakingStrategy.",
        "files": ["cli.tsx", "bot.ts", "AdvancedMarketMakingStrategy.ts"],
        "solution": "Correctly passed parameters; added momentum and depth-based spread adjustments."
      },
      {
        "issue": "Incorrect trade metrics",
        "description": "daily_pnl: 10000, balance: 10000, empty tradeHistory.",
        "files": ["bot.ts", "App.tsx"],
        "solution": "Enhanced updateProfitAndInventory for realized PNL; added unrealizedPnl; updated App.tsx for detailed metrics."
      }
    ],
    "bybit_v5_api_details": {
      "description": "Powers advanced market-making with real-time data.",
      "endpoints": {
        "rest": [
          {"path": "GET /v5/position/list", "use": "Inventory, unrealizedPnl."},
          {"path": "POST /v5/order/create", "use": "Place orders with TP/SL."},
          {"path": "POST /v5/order/cancel", "use": "Cancel orders."},
          {"path": "GET /v5/execution/list", "use": "Execution history for trade metrics."},
          {"path": "GET /v5/market/kline", "use": "Volatility calculation."},
          {"path": "GET /v5/market/orderbook", "use": "Depth-based spread adjustment."}
        ],
        "websocket": [
          {"topic": "orderbook.50.<symbol>", "use": "Reference price updates."},
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
        "description": "Advanced market-making with dynamic spreads and rebalancing.",
        "methods": [
          {"name": "constructor", "parameters": ["baseSpread: number", "volatilityFactor: number", "momentumWindow: number"], "description": "Initializes strategy parameters."},
          {"name": "calculateOrderPrices", "parameters": ["referencePrice: number", "volatility: number", "inventory: number", "maxInventory: number", "recentTrades: number[]", "orderbook?: OrderbookData"], "description": "Calculates dynamic prices and quantities."}
        ]
      },
      {
        "function": "MarketMakingBot",
        "file": "twin-range-bot/src/core/bot.ts",
        "description": "Integrates advanced strategy with trade metrics.",
        "methods": [
          {"name": "updateProfitAndInventory", "parameters": ["executions: Execution[]"], "description": "Updates realized PNL, tradeHistory, winRate, etc."},
          {"name": "updateInventoryAndPnl", "parameters": ["position: PositionData"], "description": "Updates inventory, unrealizedPnl."},
          {"name": "updateOrders", "parameters": [], "description": "Places orders with dynamic quantities."}
        ]
      },
      {
        "function": "App",
        "file": "App.tsx",
        "description": "Displays advanced metrics including momentum.",
        "methods": [
          {"name": "useEffect", "parameters": [], "description": "Initializes bot and updates UI."}
        ]
      }
    ],
    "ideas_and_enhancements": [
      {
        "idea": "Dynamic Spread Adjustment",
        "description": "Adjusts spreads based on volatility, order book depth, and market momentum.",
        "details": ["Uses klines for volatility, orderbook for depth, recentTrades for momentum."]
      },
      {
        "idea": "Inventory Rebalancing",
        "description": "Skews order quantities to maintain inventory within ±0.1.",
        "details": ["Increases buyQty when oversold, sellQty when overbought."]
      },
      {
        "idea": "Risk Management",
        "description": "Dynamic TP (3%), SL (1.5%), and position capping.",
        "details": ["TP/SL based on volatility; maxInventory: 0.1."]
      },
      {
        "idea": "Trade Metrics",
        "description": "Accurate tracking of realized/unrealized PNL, trade history, win rate, profit factor, average PNL."
      }
    ],
    "next_steps": [
      "Apply updated files (AdvancedMarketMakingStrategy.ts, bot.ts, App.tsx, cli.tsx, constants.ts).",
      "Clear cache (rm -rf node_modules/.cache).",
      "Run npm run start:backtest for TRUMPUSDT.",
      "Verify tradeHistory, daily_pnl, winRate in logs/UI.",
      "Test on testnet with REST; switch to mainnet for WebSocket.",
      "Tune baseSpread: 0.006, volatilityFactor: 1.2."
    ]
  }
}
```

### Next Steps
1. **Apply Updates**: Add `AdvancedMarketMakingStrategy.ts` and update `bot.ts`, `App.tsx`, `cli.tsx`, `constants.ts`.
2. **Clear Cache**: Run `rm -rf node_modules/.cache` or `npx tsc --build --clean`.
3. **Run Backtest**: Execute `npm run start:backtest` and check logs for order placements, executions, and metrics.
4. **Verify Metrics**: Ensure `daily_pnl`, `unrealizedPnl`, `tradeHistory`, `winRate`, etc., are updated.
5. **Test Live**: Run on testnet (`is_testnet: true`) and monitor `App.tsx` UI.
6. **Debug**: Share backtest logs if `NaN` or incorrect metrics persist.

If issues arise, provide the latest backtest output or specific errors!
