// __tests__/services/bybitService.test.ts
import { BybitService } from '../../twin-range-bot/src/services/bybitService';
import { describe, it, expect, beforeEach, vi } from 'vitest';
// Mock the logger to prevent console output during tests
vi.mock('../../twin-range-bot/src/core/logger', () => {
    return {
        logger: {
            info: vi.fn(),
            warn: vi.fn(),
            error: vi.fn(),
            debug: vi.fn(),
        },
    };
});
// Mock the bybit-api library
vi.mock('bybit-api', () => ({
    RestClientV5: vi.fn(() => ({
        submitOrder: vi.fn(() => Promise.resolve({ result: { orderId: 'mock_order_id' } })),
        cancelOrder: vi.fn(() => Promise.resolve()),
    })),
    WebsocketClient: vi.fn(() => ({
        on: vi.fn(),
        subscribe: vi.fn(),
    })),
}));
describe('BybitService', () => {
    let bybitService;
    const mockConfig = {
        symbols: ['BTCUSDT'],
        bybit_api_key: 'test_key',
        bybit_api_secret: 'test_secret',
        is_testnet: true,
    };
    const mockCallbacks = {
        onOrderbookUpdate: vi.fn(),
        onTradeUpdate: vi.fn(),
        onExecutionUpdate: vi.fn(),
        onOrderUpdate: vi.fn(),
        onPositionUpdate: vi.fn(),
        onKlineUpdate: vi.fn(),
    };
    beforeEach(() => {
        bybitService = new BybitService(mockConfig.bybit_api_key, mockConfig.bybit_api_secret, mockConfig.is_testnet, mockConfig, mockCallbacks);
    });
    it('should place a market making order', async () => {
        const orderResponse = await bybitService.placeMarketMakingOrder('BTCUSDT', 'Buy', 50000, 0.01);
        expect(orderResponse.orderId).toBe('mock_order_id');
    });
    it('should cancel an order', async () => {
        await expect(bybitService.cancelOrder('BTCUSDT', 'mock_order_id')).resolves.toBeUndefined();
    });
});
