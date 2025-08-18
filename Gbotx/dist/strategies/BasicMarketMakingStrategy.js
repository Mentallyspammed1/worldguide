import { logger } from '../logger';
import fs from 'fs/promises';
class BasicMarketMakingStrategy {
    constructor() {
        this.quantity = 0.01;
        this.position = 0;
        this.cash = 10000;
        this.trades = [];
        this.equityCurve = [10000];
    }
    async loadHistoricalData() {
        try {
            const data = await fs.readFile('data/historical_prices.json', 'utf-8');
            return JSON.parse(data);
        }
        catch (error) {
            logger.warn('Could not read historical_prices.json', { error });
            return [];
        }
    }
    simulateOrder(side, price) {
        if (side === 'Buy' && this.cash >= price * this.quantity) {
            this.cash -= price * this.quantity;
            this.position += this.quantity;
            this.trades.push({ timestamp: new Date().toISOString(), price, side, profit: 0 });
            this.updateEquity(price);
        }
        else if (side === 'Sell' && this.position >= this.quantity) {
            this.cash += price * this.quantity;
            this.position -= this.quantity;
            const lastBuy = this.trades.filter(t => t.side === 'Buy').pop();
            if (lastBuy) {
                const profit = (price - lastBuy.price) * this.quantity;
                const tradeIndex = this.trades.findIndex(t => t === lastBuy);
                if (tradeIndex !== -1) {
                    this.trades[tradeIndex].profit = profit;
                }
            }
            this.trades.push({ timestamp: new Date().toISOString(), price, side, profit: 0 });
            this.updateEquity(price);
        }
    }
    updateEquity(currentPrice) {
        const currentEquity = this.cash + (this.position * currentPrice);
        this.equityCurve.push(currentEquity);
    }
    calculateVolatility(symbol, klines, referencePrice, atr) {
        return 0.01; // Dummy value
    }
    calculateMetrics() {
        const profits = this.trades.filter(t => t.profit !== 0).map(t => t.profit);
        if (profits.length === 0) {
            return { totalProfit: 0, finalEquity: this.equityCurve[this.equityCurve.length - 1], tradeCount: 0 };
        }
        const totalProfit = profits.reduce((sum, p) => sum + p, 0);
        return { totalProfit, finalEquity: this.equityCurve[this.equityCurve.length - 1], tradeCount: this.trades.length };
    }
    async backtest() {
        const data = await this.loadHistoricalData();
        if (data.length === 0) {
            return { totalProfit: 0, finalEquity: this.equityCurve[this.equityCurve.length - 1], tradeCount: 0 };
        }
        for (const { price } of data) {
            this.simulateOrder('Buy', price);
            this.simulateOrder('Sell', price);
        }
        return this.calculateMetrics();
    }
}
export default BasicMarketMakingStrategy;
