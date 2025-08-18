import { logger } from '../logger';
import fs from 'fs/promises';

class BasicMarketMakingStrategy {
  private quantity: number = 0.01;
  private position: number = 0;
  private cash: number = 10000;
  public trades: { timestamp: string; price: number; side: string; profit: number }[] = [];
  private equityCurve: number[] = [10000];

  async loadHistoricalData() {
    try {
      const data = await fs.readFile('data/historical_prices.json', 'utf-8');
      return JSON.parse(data) as { timestamp: string; price: number }[];
    } catch (error) {
      logger.warn('Could not read historical_prices.json', { error });
      return [];
    }
  }

  private simulateOrder(side: 'Buy' | 'Sell', price: number) {
    if (side === 'Buy' && this.cash >= price * this.quantity) {
      this.cash -= price * this.quantity;
      this.position += this.quantity;
      this.trades.push({ timestamp: new Date().toISOString(), price, side, profit: 0 });
      this.updateEquity(price);
    } else if (side === 'Sell' && this.position >= this.quantity) {
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

  private updateEquity(currentPrice: number) {
    const currentEquity = this.cash + (this.position * currentPrice);
    this.equityCurve.push(currentEquity);
  }

  public calculateVolatility(symbol: string, klines: any[], referencePrice: number, atr: number): number {
    return 0.01; // Dummy value
  }

  public calculateMetrics() {
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
