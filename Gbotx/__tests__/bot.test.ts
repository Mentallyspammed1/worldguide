// __tests__/bot.test.ts
import { MarketMakingBot } from '../twin-range-bot/src/core/bot';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import BasicMarketMakingStrategy from '../strategies/BasicMarketMakingStrategy';
import MultiSymbolMarketMakingStrategy from '../strategies/MultiSymbolMarketMakingStrategy';

// Mock the Bybit service to avoid actual API calls
vi.mock('../twin-range-bot/src/services/bybitService', () => ({
  BybitService: vi.fn(() => ({
    placeMarketMakingOrder: vi.fn((_symbol, side, price, _qty, _tp, _sl) => Promise.resolve({ orderId: `mock_${side}_${price}` })),
    cancelOrder: vi.fn(() => Promise.resolve()),
    getOrderbook: vi.fn(() => Promise.resolve({ b: [['50000', '1']], a: [['50000', '1']] })),
  })),
}));

// Mock the strategies
vi.mock('../strategies/BasicMarketMakingStrategy');
vi.mock('../strategies/MultiSymbolMarketMakingStrategy');

describe('MarketMakingBot', () => {
  let bot: MarketMakingBot;
  let strategy: BasicMarketMakingStrategy;

  beforeEach(() => {
    strategy = new BasicMarketMakingStrategy();
    bot = new MarketMakingBot({
      symbol: 'BTCUSDT',
      interval: '60',
      lookback_bars: 500,
      baseSpread: 0.005,
      orderQty: 0.01,
      maxInventory: 0.1,
      tpPercent: 0.02,
      slPercent: 0.02,
      volatilityWindow: 10,
      volatilityFactor: 1,
      dataSource: 'rest',
      refresh_rate_seconds: 60,
      strategyType: 'BasicMarketMakingStrategy',
      bybit_api_key: 'test_key',
      bybit_api_secret: 'test_secret',
      is_testnet: true,
      symbols: ['BTCUSDT'],
    }, strategy);
  });

  it('places buy and sell orders based on orderbook', async () => {
    (bot as any).state.referencePrice = 50000;
    const expectedSpread = 50000 * bot.getConfig().baseSpread;
    const expectedBuyPrice = 50000 - expectedSpread / 2;
    const expectedSellPrice = 50000 + expectedSpread / 2;

    (bot as any).state.symbols['BTCUSDT'].referencePrice = 50000;
    await (bot as any).updateOrders('BTCUSDT');

    const buyOrder = bot.getState().symbols['BTCUSDT'].active_mm_orders.find((order) => order.type === 'buy');
    const sellOrder = bot.getState().symbols['BTCUSDT'].active_mm_orders.find((order) => order.type === 'sell');

    expect(buyOrder?.price).toBeCloseTo(expectedBuyPrice);
    expect(sellOrder?.price).toBeCloseTo(expectedSellPrice);
  });
});
