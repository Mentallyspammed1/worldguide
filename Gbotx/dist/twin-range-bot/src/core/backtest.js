// twin-range-bot/src/core/backtest.ts
import { MarketMakingBot } from './bot';
import logger from '../../../logger';
import { BOT_CONFIG_TEMPLATE } from '../../../constants';
export class Backtester {
    constructor(config = {}) {
        this.config = {
            ...BOT_CONFIG_TEMPLATE,
            symbols: ['TRUMPUSDT', 'BTCUSDT', 'ETHUSDT'],
            interval: '60',
            initialCapital: 10000,
            is_testnet: false,
            refresh_rate_seconds: 60,
            ...config,
        };
        this.bot = new MarketMakingBot(this.config);
        this.mockData = this.generateMockData();
        this.currentTimestamp = Math.min(...Object.values(this.mockData.klines).flatMap(klines => klines.map(k => k.t)));
        logger.info('Backtester initialized', { symbols: this.config.symbols });
    }
    generateMockData() {
        const startTime = Date.now() - 24 * 60 * 60 * 1000;
        const klines = {};
        const orderbooks = {};
        this.config.symbols.forEach(symbol => {
            klines[symbol] = [];
            orderbooks[symbol] = [];
            const basePrice = symbol === 'TRUMPUSDT' ? 10 : symbol === 'BTCUSDT' ? 50000 : 2000;
            for (let i = 0; i < 24; i++) {
                const t = startTime + i * 60 * 60 * 1000;
                const price = basePrice * (1 + (Math.random() - 0.5) * 0.02);
                klines[symbol].push({
                    s: symbol,
                    t,
                    o: price.toFixed(2),
                    h: (price * 1.01).toFixed(2),
                    l: (price * 0.99).toFixed(2),
                    c: price.toFixed(2),
                    v: (Math.random() * 100).toFixed(2),
                });
                orderbooks[symbol].push({
                    s: symbol,
                    b: [[(price * 0.998).toFixed(2), '10'], [(price * 0.996).toFixed(2), '10']],
                    a: [[(price * 1.002).toFixed(2), '10'], [(price * 1.004).toFixed(2), '10']],
                    ts: t,
                    u: i,
                });
            }
        });
        return { klines, orderbooks };
    }
    simulateExecution(symbol, order) {
        const orderbook = this.mockData.orderbooks[symbol].find(ob => ob.ts >= order.ts);
        if (!orderbook)
            return null;
        const price = parseFloat(order.side === 'Buy' ? orderbook.a[0][0] : orderbook.b[0][0]);
        if ((order.side === 'Buy' && parseFloat(order.price) >= price) || (order.side === 'Sell' && parseFloat(order.price) <= price)) {
            const exec = {
                symbol,
                orderId: order.orderId,
                side: order.side,
                execPrice: price.toFixed(2),
                execQty: order.qty,
                execFee: (parseFloat(order.qty) * price * 0.0006).toFixed(6),
                execTime: order.ts.toString(),
            };
            logger.info(`Execution: ${exec.side} ${exec.execQty} ${symbol} at $${exec.execPrice}`, { symbol });
            return exec;
        }
        return null;
    }
    async run() {
        const endTime = Math.max(...Object.values(this.mockData.klines).flatMap(klines => klines.map(k => k.t)));
        while (this.currentTimestamp <= endTime) {
            for (const symbol of this.config.symbols) {
                const kline = this.mockData.klines[symbol].find(k => k.t >= this.currentTimestamp);
                const orderbook = this.mockData.orderbooks[symbol].find(ob => ob.ts >= this.currentTimestamp);
                if (kline && orderbook) {
                    this.bot.handleKlineUpdate([kline]);
                    this.bot.handleOrderbookUpdate(orderbook);
                    await this.bot.updateOrders(symbol);
                    const orders = await this.bot.getActiveOrders(symbol);
                    const executions = orders.map(order => this.simulateExecution(symbol, order)).filter((e) => e !== null);
                    if (executions.length)
                        this.bot.handleExecutionUpdate(executions);
                    const position = {
                        symbol,
                        side: Math.random() > 0.5 ? 'Buy' : 'Sell',
                        size: (Math.random() * 0.1).toFixed(4),
                        avgPrice: parseFloat(kline.c).toFixed(2),
                        updatedTime: this.currentTimestamp.toString(),
                        positionValue: (parseFloat(kline.c) * Math.random() * 0.1).toFixed(2),
                        unrealisedPnl: ((Math.random() - 0.5) * 10).toFixed(2),
                    };
                    this.bot.handlePositionUpdate([position]);
                }
            }
            this.currentTimestamp += this.config.refresh_rate_seconds * 1000;
        }
        const state = this.bot.getState();
        logger.info('Backtest done', { balance: state.balance, profit: state.totalProfit });
        return state;
    }
    getState() {
        return this.bot.getState();
    }
}
