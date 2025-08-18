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
  minVolatility: number; // Added for ATR
  maxVolatility: number; // Added for ATR
  minDepthRatio: number; // Added for ATR
  maxDepthRatio: number; // Added for ATR
  maxMomentum: number; // Added for ATR
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

export interface StrategyConfig {
  symbols: string[];
  dataDirectory?: string;
  initialCapital?: number;
  spread?: number;
  riskPercent?: number;
  atrPeriod?: number;
  stopLossMultiplier?: number;
  minTradeQuantity?: number;
  maxQuantityCap?: number;
}