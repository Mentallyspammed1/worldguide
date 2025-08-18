// twin-range-bot/src/core/types.ts
export interface PerSymbolState {
  cash: number;
  unrealizedPnl: number;
  inventory: number;
  atr: number;
  active_mm_orders: OrderData[];
  tradeHistory: { tradeId: string; side: string; qty: number; price: number; profit: number; fee: number; timestamp: number }[];
  orderbook: OrderbookData | null;
  klines: KlineData[];
  executions: Execution[];
  orderStatus: string;
  referencePrice: number;
  recentTrades: number[]; // Added missing property
}

export interface TradeState {
  symbols: { [key: string]: PerSymbolState };
  totalProfit: number;
  daily_pnl: number;
  balance: number;
  logs: { type: string; message: string; symbol?: string }[];
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgPnl: number;
  equityCurve: number[];
}

export interface BotConfig {
  symbols: string[];
  interval: string;
  initialCapital: number;
  is_testnet: boolean;
  refresh_rate_seconds: number;
  maxInventory: number;
  riskPercent: number;
  atrPeriod: number;
  lookback_bars: number;
  baseSpread: number;
  orderQty: number;
  tpPercent: number;
  slPercent: number;
  volatilityWindow: number;
  volatilityFactor: number;
  dataSource: 'websocket' | 'rest' | 'backtest';
  bybit_api_key: string;
  bybit_api_secret: string;
  strategyType: string;
  minVolatility: number;
  maxVolatility: number;
  minDepthRatio: number;
  maxDepthRatio: number;
  maxMomentum: number;
}

export interface OrderData {
  orderId: string;
  symbol: string;
  side: 'Buy' | 'Sell';
  orderType: string;
  price: number; // Changed from string to number
  qty: string;
  orderStatus: string;
  takeProfit: string;
  stopLoss: string;
  ts: number;
  type: 'buy' | 'sell'; // Added missing property
}

export interface KlineData {
  s: string;
  t: number;
  o: string;
  h: string;
  l: string;
  c: string;
  v: string;
}

export interface OrderbookData {
  s: string;
  b: [string, string][];
  a: [string, string][];
  ts: number;
  u: number;
}

export interface Execution {
  symbol: string;
  orderId: string;
  side: string;
  execPrice: string;
  execQty: string;
  execFee: string;
  execTime: string;
}

export interface PositionData {
  symbol: string;
  side: any;
  size: string;
  avgPrice: string;
  updatedTime: string;
  positionValue: string;
  unrealisedPnl: string;
}

export type BotStatus = 'IDLE' | 'RUNNING' | 'PAUSED' | 'ERROR'; // Exported missing type

export interface ChartDataPoint {
  time: string;
  value: number;
} // Exported missing type
