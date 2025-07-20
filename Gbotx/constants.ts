// constants.ts
import type { BotConfig, TradeState, PerSymbolState } from './types';

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
  bybit_api_key: 'NRrb4Biggi3sO7rKZ1',
  bybit_api_secret: 'TXztLxhYdHIcyzmN6QR2zSc2Dxj0UuQRiMzQ',
  is_testnet: false,
  strategyType: 'AdvancedMarketMakingStrategy',
  minVolatility: 0.005,
  maxVolatility: 0.05,
  minDepthRatio: 0.5,
  maxDepthRatio: 2.0,
  maxMomentum: 0.02,
  atrPeriod: 14,
  riskPercent: 0.01,
  initialCapital: 10000,
};

const initialCapital = BOT_CONFIG_TEMPLATE.initialCapital || 10000;
const symbols = BOT_CONFIG_TEMPLATE.symbols; // Use symbols from BOT_CONFIG_TEMPLATE
const capitalPerSymbol = initialCapital / symbols.length;

const initialPerSymbolState: { [key: string]: PerSymbolState } = {};
symbols.forEach(symbol => {
  initialPerSymbolState[symbol] = {
    active_mm_orders: [],
    inventory: 0,
    recentTrades: [],
    referencePrice: 0,
    klines: [],
    tradeHistory: [],
    unrealizedPnl: 0,
    orderStatus: 'Idle',
    cash: capitalPerSymbol,
    atr: 0,
    orderbook: null,
    executions: [],
  };
});

export const INITIAL_TRADE_STATE_TEMPLATE: TradeState = {
  symbols: initialPerSymbolState,
  totalProfit: 0,
  daily_pnl: 0,
  balance: initialCapital,
  logs: [],
  winRate: 0,
  profitFactor: 0,
  totalTrades: 0,
  avgPnl: 0,
  equityCurve: [initialCapital],
};