import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Backtester } from '../../twin-range-bot/src/core/backtest';
import fs from 'fs/promises';
// Mock the logger to prevent console output during tests
vi.mock('../../twin-range-bot/src/core/logger', () => ({
    logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
    },
}));
describe('Backtester', () => {
    let backtester;
    beforeEach(async () => {
        backtester = new Backtester();
        vi.spyOn(fs, 'readFile').mockResolvedValue(JSON.stringify([
            { timestamp: "2025-07-20T08:00:00Z", price: 50000 },
            { timestamp: "2025-07-20T08:00:01Z", price: 50010 },
        ]));
    });
    it('should run backtest and produce a final state', async () => {
        const finalState = await backtester.run();
        expect(finalState.balance).toBeGreaterThan(0);
        expect(finalState.totalProfit).toBeDefined();
    });
});
