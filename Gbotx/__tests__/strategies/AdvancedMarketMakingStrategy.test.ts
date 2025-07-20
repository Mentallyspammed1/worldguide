import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AdvancedMarketMakingStrategy } from '../../strategies/AdvancedMarketMakingStrategy';
import fs from 'fs/promises';
import path from 'path';

describe('AdvancedMarketMakingStrategy Backtest', () => {
  let strategy: AdvancedMarketMakingStrategy;
  const mockHistoricalData = {
    TRUMPUSDT: [
      { timestamp: "2025-07-20T08:00:00Z", price: 10.00 },
      { timestamp: "2025-07-20T08:00:01Z", price: 10.05 },
      { timestamp: "2025-07-20T08:00:02Z", price: 10.10 },
      { timestamp: "2025-07-20T08:00:03Z", price: 10.08 },
      { timestamp: "2025-07-20T08:00:04Z", price: 10.12 },
      { timestamp: "2025-07-20T08:00:05Z", price: 10.15 },
    ],
    BTCUSDT: [
      { timestamp: "2025-07-20T08:00:00Z", price: 50000 },
      { timestamp: "2025-07-20T08:00:01Z", price: 50010 },
      { timestamp: "2025-07-20T08:00:02Z", price: 50020 },
      { timestamp: "2025-07-20T08:00:03Z", price: 50015 },
      { timestamp: "2025-07-20T08:00:04Z", price: 50025 },
      { timestamp: "2025-07-20T08:00:05Z", price: 50030 },
    ],
  };

  beforeEach(() => {
    vi.spyOn(fs, 'readFile').mockImplementation((filePath) => {
      if (filePath.includes('historical_prices_trumpusdt.json')) {
        return Promise.resolve(JSON.stringify(mockHistoricalData.TRUMPUSDT));
      } else if (filePath.includes('historical_prices_btcusdt.json')) {
        return Promise.resolve(JSON.stringify(mockHistoricalData.BTCUSDT));
      }
      return Promise.reject(new Error('File not found'));
    });

    strategy = new AdvancedMarketMakingStrategy({
      symbols: ['TRUMPUSDT', 'BTCUSDT'],
      initialCapital: 10000,
      spread: 0.006,
      riskPercent: 0.01,
      atrPeriod: 14,
      stopLossMultiplier: 3,
      minTradeQuantity: 0.001,
      maxQuantityCap: 1,
    });
  });

  it('should run backtest and calculate metrics', async () => {
    const metrics = await strategy.runBacktest();

    expect(metrics).toBeDefined();
    expect(metrics.aggregate.totalProfit).toBeDefined();
    expect(metrics.aggregate.finalEquity).toBeDefined();
    expect(metrics.aggregate.totalTradeCount).toBeGreaterThan(0);
    expect(metrics.aggregate.winRate).toBeGreaterThanOrEqual(0);
    expect(metrics.aggregate.profitFactor).toBeGreaterThanOrEqual(0);
    expect(metrics.aggregate.maxDrawdown).toBeGreaterThanOrEqual(0);

    // Check individual symbol metrics
    expect(metrics.TRUMPUSDT).toBeDefined();
    expect(metrics.BTCUSDT).toBeDefined();
  });

  it('should correctly calculate order size', () => {
    // This test needs to be more isolated or mock the calculateVolatility method
    // For now, we'll test with a fixed volatility to ensure the formula is correct
    const mockVolatility = 0.01; // 1% volatility
    const currentPrice = 100;
    const riskAmount = strategy['initialCapital'] * strategy['riskPercent'] / strategy['symbols'].length;
    const stopLossDistance = mockVolatility * strategy['stopLossMultiplier'];
    const expectedQty = riskAmount / stopLossDistance;

    // Temporarily override calculateVolatility for this test
    const originalCalculateVolatility = strategy['calculateVolatility'];
    strategy['calculateVolatility'] = vi.fn(() => mockVolatility);

    const calculatedSize = strategy['calculateOrderSize']('BTCUSDT', currentPrice);

    expect(calculatedSize).toBeCloseTo(Math.min(expectedQty, (strategy['cash']['BTCUSDT'] || 0) / currentPrice, strategy['maxQuantityCap']));

    // Restore original method
    strategy['calculateVolatility'] = originalCalculateVolatility;
  });
});