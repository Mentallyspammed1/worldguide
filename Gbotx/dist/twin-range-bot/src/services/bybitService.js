// twin-range-bot/src/services/bybitService.ts
import { RestClientV5, WebsocketClient } from 'bybit-api';
import { logger } from '../core/logger';
export class BybitService {
    constructor(apiKey, apiSecret, testnet, config, callbacks) {
        this.callbacks = callbacks;
        this.rateLimiter = { lastCall: 0, minInterval: 500 };
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.apiCallMetrics = {};
        this.restClient = new RestClientV5({ key: apiKey, secret: apiSecret, testnet });
        this.wsClient = new WebsocketClient({ key: apiKey, secret: apiSecret, market: 'v5', testnet });
        this.config = config;
        this.setupWebSocket();
        logger.info('BybitService initialized', { testnet, symbols: config.symbols });
    }
    setupWebSocket() {
        this.wsClient.on('error', (error) => { logger.error('WebSocket error', { error: error.message }); this.handleReconnect(); });
        this.wsClient.on('close', () => { logger.warn('WebSocket closed'); this.handleReconnect(); });
        this.wsClient.on('update', (data) => {
            const symbol = data.topic.split('.').slice(-1)[0] || data.data?.s;
            if (data.topic.startsWith('orderbook.50.') && this.config.symbols.includes(symbol))
                this.callbacks.onOrderbookUpdate(data.data);
            else if (data.topic.startsWith('publicTrade.') && this.config.symbols.includes(symbol))
                this.callbacks.onTradeUpdate(data.data);
            else if (data.topic === 'execution')
                this.callbacks.onExecutionUpdate(data.data);
            else if (data.topic === 'order')
                this.callbacks.onOrderUpdate(data.data);
            else if (data.topic === 'position')
                this.callbacks.onPositionUpdate(data.data);
            else if (data.topic.startsWith(`kline.${this.config.interval}.`) && this.config.symbols.includes(symbol))
                this.callbacks.onKlineUpdate(data.data);
        });
        const subscriptions = this.config.symbols.flatMap(s => [
            { topic: `orderbook.50.${s}`, category: 'linear' },
            { topic: `publicTrade.${s}`, category: 'linear' },
            { topic: `kline.${this.config.interval}.${s}`, category: 'linear' },
        ]).concat([{ topic: 'execution', category: 'linear' }, { topic: 'order', category: 'linear' }, { topic: 'position', category: 'linear' }]);
        this.wsClient.subscribe(subscriptions);
        logger.info('WebSocket subscriptions', { subscriptions });
    }
    async handleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            logger.error('Max reconnect attempts reached', { attempts: this.reconnectAttempts });
            return;
        }
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
        logger.info(`Reconnecting in ${delay}ms`, { attempt: this.reconnectAttempts });
        setTimeout(() => {
            this.wsClient = new WebsocketClient({ key: this.restClient.options.key, secret: this.restClient.options.secret, market: 'v5', testnet: this.config.is_testnet });
            this.setupWebSocket();
        }, delay);
    }
    async rateLimit() {
        const elapsed = Date.now() - this.rateLimiter.lastCall;
        if (elapsed < this.rateLimiter.minInterval)
            await new Promise(resolve => setTimeout(resolve, this.rateLimiter.minInterval - elapsed));
        this.rateLimiter.lastCall = Date.now();
    }
    trackApiCall(method, start, success) {
        this.apiCallMetrics[method] = this.apiCallMetrics[method] || { count: 0, totalLatency: 0, errors: 0 };
        this.apiCallMetrics[method].count++;
        this.apiCallMetrics[method].totalLatency += Date.now() - start;
        if (!success)
            this.apiCallMetrics[method].errors++;
        logger.debug('API metrics', { method, avgLatency: (this.apiCallMetrics[method].totalLatency / this.apiCallMetrics[method].count).toFixed(2) });
    }
    async placeMarketMakingOrder(symbol, side, price, qty, takeProfit, stopLoss) {
        await this.rateLimit();
        const start = Date.now();
        try {
            if (!this.config.symbols.includes(symbol))
                throw new Error(`Invalid symbol: ${symbol}`);
            const response = await this.restClient.submitOrder({ category: 'linear', symbol, side, orderType: 'Limit', qty: qty.toFixed(6), price: price.toFixed(2), timeInForce: 'GTC', takeProfit: takeProfit?.toFixed(2), stopLoss: stopLoss?.toFixed(2), tpTriggerBy: 'LastPrice', slTriggerBy: 'LastPrice' });
            if (!response.result.orderId)
                throw new Error('Missing orderId');
            this.trackApiCall('placeMarketMakingOrder', start, true);
            logger.info(`Order placed: ${side} ${qty.toFixed(6)} ${symbol} at $${price.toFixed(2)}`, { symbol });
            return response.result;
        }
        catch (err) {
            this.trackApiCall('placeMarketMakingOrder', start, false);
            logger.error('Order placement failed', { symbol, error: err.message });
            throw err;
        }
    }
    async placeBatchOrders(orders) {
        await this.rateLimit();
        const start = Date.now();
        try {
            const validOrders = orders.filter(o => this.config.symbols.includes(o.symbol));
            const responses = await Promise.all(validOrders.map(o => this.restClient.submitOrder({ category: 'linear', symbol: o.symbol, side: o.side, orderType: 'Limit', qty: o.qty.toFixed(6), price: o.price.toFixed(2), timeInForce: 'GTC', takeProfit: o.takeProfit?.toFixed(2), stopLoss: o.stopLoss?.toFixed(2), tpTriggerBy: 'LastPrice', slTriggerBy: 'LastPrice' })));
            this.trackApiCall('placeBatchOrders', start, true);
            logger.info('Batch orders placed', { count: validOrders.length });
            return responses.map(r => r.result);
        }
        catch (err) {
            this.trackApiCall('placeBatchOrders', start, false);
            logger.error('Batch order failed', { error: err.message });
            throw err;
        }
    }
    async cancelOrder(symbol, orderId) {
        await this.rateLimit();
        const start = Date.now();
        try {
            if (!this.config.symbols.includes(symbol))
                throw new Error(`Invalid symbol: ${symbol}`);
            await this.restClient.cancelOrder({ category: 'linear', symbol, orderId });
            this.trackApiCall('cancelOrder', start, true);
            logger.info(`Order cancelled: ${orderId}`, { symbol });
        }
        catch (err) {
            this.trackApiCall('cancelOrder', start, false);
            logger.error('Cancel order failed', { symbol, orderId, error: err.message });
            throw err;
        }
    }
    async cancelBatchOrders(orders) {
        await this.rateLimit();
        const start = Date.now();
        try {
            const validOrders = orders.filter(o => this.config.symbols.includes(o.symbol));
            await Promise.all(validOrders.map(o => this.restClient.cancelOrder({ category: 'linear', symbol: o.symbol, orderId: o.orderId })));
            this.trackApiCall('cancelBatchOrders', start, true);
            logger.info('Batch orders cancelled', { count: validOrders.length });
        }
        catch (err) {
            this.trackApiCall('cancelBatchOrders', start, false);
            logger.error('Batch cancel failed', { error: err.message });
            throw err;
        }
    }
    async getActiveOrders(symbol) {
        await this.rateLimit();
        const start = Date.now();
        try {
            if (!this.config.symbols.includes(symbol))
                throw new Error(`Invalid symbol: ${symbol}`);
            const response = await this.restClient.getActiveOrders({ category: 'linear', symbol });
            const orders = response.result.list.map((o) => ({ orderId: o.orderId, symbol: o.symbol, side: o.side, orderType: o.orderType, price: o.price, qty: o.qty, orderStatus: o.orderStatus, takeProfit: o.takeProfit, stopLoss: o.stopLoss, ts: o.updatedTime }));
            this.trackApiCall('getActiveOrders', start, true);
            logger.debug(`Fetched ${orders.length} orders`, { symbol });
            return orders;
        }
        catch (err) {
            this.trackApiCall('getActiveOrders', start, false);
            logger.error('Fetch orders failed', { symbol, error: err.message });
            throw err;
        }
    }
    async getKlines(symbol, interval, limit = 200) {
        await this.rateLimit();
        const start = Date.now();
        try {
            if (!this.config.symbols.includes(symbol))
                throw new Error(`Invalid symbol: ${symbol}`);
            const response = await this.restClient.getKline({ category: 'linear', symbol, interval, limit });
            if (!response.result.list)
                return [];
            const klines = response.result.list.map((k) => ({ s: symbol, t: parseInt(k[0]), o: k[1], h: k[2], l: k[3], c: k[4], v: k[5] }));
            this.trackApiCall('getKlines', start, true);
            logger.debug(`Fetched ${klines.length} klines`, { symbol });
            return klines;
        }
        catch (err) {
            this.trackApiCall('getKlines', start, false);
            logger.error('Fetch klines failed', { symbol, error: err.message });
            throw err;
        }
    }
    async getExecutionHistory(symbol, orderId) {
        await this.rateLimit();
        const start = Date.now();
        try {
            if (!this.config.symbols.includes(symbol))
                throw new Error(`Invalid symbol: ${symbol}`);
            const response = await this.restClient.getExecutionList({ category: 'linear', symbol, orderId });
            this.trackApiCall('getExecutionHistory', start, true);
            logger.debug(`Fetched ${response.result.list?.length || 0} executions`, { symbol });
            return response.result.list || [];
        }
        catch (err) {
            this.trackApiCall('getExecutionHistory', start, false);
            logger.error('Fetch executions failed', { symbol, error: err.message });
            throw err;
        }
    }
    async getOrderbook(symbol, depth = 50) {
        await this.rateLimit();
        const start = Date.now();
        try {
            if (!this.config.symbols.includes(symbol))
                throw new Error(`Invalid symbol: ${symbol}`);
            const response = await this.restClient.getOrderbook({ category: 'linear', symbol, limit: depth });
            if (!response.result.b || !response.result.a)
                throw new Error('Invalid orderbook');
            const orderbook = { s: symbol, b: response.result.b, a: response.result.a, ts: response.time, u: response.result.u };
            this.trackApiCall('getOrderbook', start, true);
            logger.debug('Orderbook fetched', { symbol, bids: orderbook.b.length });
            return orderbook;
        }
        catch (err) {
            this.trackApiCall('getOrderbook', start, false);
            logger.error('Fetch orderbook failed', { symbol, error: err.message });
            throw err;
        }
    }
    async getPositions(symbol) {
        await this.rateLimit();
        const start = Date.now();
        try {
            if (!this.config.symbols.includes(symbol))
                throw new Error(`Invalid symbol: ${symbol}`);
            const response = await this.restClient.getPositionInfo({ category: 'linear', symbol });
            const positions = response.result.list.map((p) => ({ symbol: p.symbol, side: p.side, size: p.size, avgPrice: p.avgPrice, updatedTime: p.updatedTime, positionValue: p.positionValue, unrealisedPnl: p.unrealisedPnl }));
            this.trackApiCall('getPositions', start, true);
            logger.debug(`Fetched ${positions.length} positions`, { symbol });
            return positions.length ? positions : [{ symbol, side: '', size: '0', avgPrice: '0', updatedTime: Date.now().toString(), positionValue: '0', unrealisedPnl: '0' }];
        }
        catch (err) {
            this.trackApiCall('getPositions', start, false);
            logger.error('Fetch positions failed', { symbol, error: err.message });
            throw err;
        }
    }
    convertPositionSide(side) {
        return side === '' ? 'None' : side;
    }
    getApiMetrics() {
        return Object.entries(this.apiCallMetrics).reduce((acc, [m, v]) => ({ ...acc, [m]: { count: v.count, avgLatency: v.count ? v.totalLatency / v.count : 0, errorRate: v.count ? v.errors / v.count : 0 } }), {});
    }
}
