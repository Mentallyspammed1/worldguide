import { logger } from '../logger';
import fs from 'fs/promises';
import path from 'path'; // Import path for better file handling
class MultiSymbolMarketMakingStrategy {
    constructor(config) {
        this.cash = {}; // Cash allocated per symbol
        this.symbolCapitalAllocation = {}; // Percentage or fixed amount of total capital per symbol
        this.positions = {};
        this.trades = {}; // Organized by symbol
        this.equityCurve = [];
        this.basePrices = {};
        this.previousPrices = {};
        this.symbols = config.symbols;
        this.dataDirectory = config.dataDirectory || 'data';
        this.initialCapital = config.initialCapital || 10000;
        this.spread = config.spread || 50;
        this.riskPercent = config.riskPercent || 0.01;
        this.atrPeriod = config.atrPeriod || 14;
        this.stopLossMultiplier = config.stopLossMultiplier || 3;
        this.minTradeQuantity = config.minTradeQuantity || 0.001;
        this.maxQuantityCap = config.maxQuantityCap || 1;
        const capitalPerSymbol = this.initialCapital / this.symbols.length; // Simple equal split
        this.symbols.forEach(symbol => {
            this.positions[symbol] = 0;
            this.trades[symbol] = [];
            this.basePrices[symbol] = 0;
            this.cash[symbol] = capitalPerSymbol; // Initialize cash for each symbol
            this.symbolCapitalAllocation[symbol] = capitalPerSymbol; // Keep track of initial allocation
        });
        this.equityCurve = [this.initialCapital]; // Equity curve still tracks total
    }
    async loadHistoricalData() {
        const data = {};
        for (const symbol of this.symbols) {
            try {
                const filePath = path.join(this.dataDirectory, `historical_prices_${symbol.toLowerCase()}.json`);
                const fileContent = await fs.readFile(filePath, 'utf-8');
                data[symbol] = JSON.parse(fileContent);
                logger.info(`Loaded historical data for ${symbol} from ${filePath}`);
            }
            catch (error) {
                logger.warning(`No data file for ${symbol} found at ${path.join(this.dataDirectory, `historical_prices_${symbol.toLowerCase()}.json`)}, initializing with empty dataset.`, { error });
                data[symbol] = [];
            }
        }
        return data;
    }
    calculateVolatility(symbol, currentPrice) {
        const prevPriceInfo = this.previousPrices[symbol] || { high: currentPrice, low: currentPrice, close: currentPrice };
        const trueRange = Math.max(currentPrice - prevPriceInfo.low, Math.abs(currentPrice - prevPriceInfo.close), Math.abs(prevPriceInfo.close - prevPriceInfo.high));
        if (!this.previousPrices[symbol]) {
            this.previousPrices[symbol] = { high: currentPrice, low: currentPrice, close: currentPrice };
            return trueRange;
        }
        else {
            const alpha = 2 / (this.atrPeriod + 1);
            const prevAtr = this.previousPrices[symbol].atr || trueRange;
            const currentAtr = (trueRange * alpha) + (prevAtr * (1 - alpha));
            this.previousPrices[symbol].atr = currentAtr;
            this.previousPrices[symbol] = { high: currentPrice, low: currentPrice, close: currentPrice, atr: currentAtr };
            return currentAtr;
        }
    }
    calculateOrderSize(symbol, currentPrice) {
        const riskAmountPerSymbol = (this.initialCapital * this.riskPercent) / this.symbols.length;
        const volatility = this.calculateVolatility(symbol, currentPrice);
        const stopLossDistance = volatility * this.stopLossMultiplier;
        if (stopLossDistance === 0) {
            logger.warn(`Zero stop-loss distance for ${symbol}. Cannot calculate order size.`, { symbol, currentPrice, volatility });
            return 0;
        }
        let quantityFromRisk = riskAmountPerSymbol / stopLossDistance;
        const maxAffordableQuantity = (this.cash[symbol] || 0) / currentPrice; // Use per-symbol cash
        let calculatedQuantity = Math.min(Math.max(quantityFromRisk, 0), maxAffordableQuantity);
        calculatedQuantity = parseFloat(calculatedQuantity.toFixed(6));
        const minTradeQuantity = this.minTradeQuantity;
        if (calculatedQuantity < minTradeQuantity) {
            logger.debug(`Calculated quantity for ${symbol} below minimum trade quantity, setting to 0.`, { symbol, calculatedQuantity, minTradeQuantity });
            return 0;
        }
        const maxQuantityCap = this.maxQuantityCap;
        calculatedQuantity = Math.min(calculatedQuantity, maxQuantityCap);
        logger.debug(`Calculated order size for ${symbol}`, {
            symbol,
            currentPrice,
            riskAmountPerSymbol,
            volatility,
            stopLossDistance,
            quantityFromRisk,
            maxAffordableQuantity,
            calculatedQuantity,
        });
        return calculatedQuantity;
    }
    simulateOrder(symbol, side, triggerPrice, orderPrice) {
        if ((side === 'Buy' && this.basePrices[symbol] <= triggerPrice) || (side === 'Sell' && this.basePrices[symbol] >= triggerPrice)) {
            const quantity = this.calculateOrderSize(symbol, this.basePrices[symbol]);
            const executedPrice = orderPrice;
            if (quantity === 0) {
                logger.debug(`Order quantity for ${symbol} is zero, skipping trade.`, { symbol, side, triggerPrice, orderPrice });
                return;
            }
            if (side === 'Buy') {
                const cost = executedPrice * quantity;
                if ((this.cash[symbol] || 0) >= cost) {
                    this.cash[symbol] -= cost;
                    this.positions[symbol] += quantity;
                    this.trades[symbol].push({
                        timestamp: new Date().toISOString(),
                        symbol: symbol,
                        side: side,
                        entryPrice: executedPrice,
                        quantity: quantity,
                        profit: 0,
                        status: 'Open',
                        tradeId: `${symbol}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
                    });
                    this.updateEquity();
                    logger.info(`Backtest ${side} executed for ${symbol}`, { price: executedPrice, quantity, cash: this.cash[symbol], position: this.positions[symbol] });
                }
                else {
                    logger.debug(`Insufficient cash for ${symbol} buy order. Needed: ${cost.toFixed(2)}, Available: ${(this.cash[symbol] || 0).toFixed(2)}`, { symbol, price: executedPrice, quantity });
                }
            }
            else if (side === 'Sell') {
                if (this.positions[symbol] >= quantity) {
                    const proceeds = executedPrice * quantity;
                    this.cash[symbol] += proceeds;
                    this.positions[symbol] -= quantity;
                    const openBuyTrades = this.trades[symbol].filter(t => t.side === 'Buy' && t.status === 'Open');
                    if (openBuyTrades.length > 0) {
                        const tradeToClose = openBuyTrades[0];
                        const profit = (executedPrice - tradeToClose.entryPrice) * quantity;
                        tradeToClose.profit += profit;
                        tradeToClose.exitPrice = executedPrice;
                        tradeToClose.status = 'Closed';
                        logger.info(`Backtest ${side} executed for ${symbol} (closing trade)`, {
                            price: executedPrice,
                            quantity,
                            cash: this.cash[symbol],
                            position: this.positions[symbol],
                            profit,
                            closedTradeId: tradeToClose.tradeId
                        });
                    }
                    else {
                        this.trades[symbol].push({
                            timestamp: new Date().toISOString(),
                            symbol: symbol,
                            side: side,
                            entryPrice: executedPrice,
                            quantity: quantity,
                            profit: 0,
                            status: 'Open',
                            tradeId: `${symbol}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
                        });
                        logger.info(`Backtest ${side} executed for ${symbol} (standalone)`, { price: executedPrice, quantity, cash: this.cash[symbol], position: this.positions[symbol] });
                    }
                    this.updateEquity();
                }
                else {
                    logger.debug(`Insufficient position for ${symbol} sell order. Needed: ${quantity.toFixed(6)}, Available: ${this.positions[symbol].toFixed(6)}`, { symbol, price: executedPrice, quantity });
                }
            }
        }
    }
    updateEquity() {
        const totalCash = Object.values(this.cash).reduce((sum, c) => sum + c, 0);
        const totalPositionValue = Object.entries(this.positions).reduce((sum, [symbol, pos]) => sum + (pos * (this.basePrices[symbol] || 0)), 0);
        const currentEquity = totalCash + totalPositionValue;
        this.equityCurve.push(currentEquity);
        logger.info('Equity updated', { totalCash, positions: this.positions, equity: currentEquity });
    }
    async backtest() {
        const allSymbolData = await this.loadHistoricalData();
        const allEvents = [];
        for (const symbol of this.symbols) {
            if (allSymbolData[symbol]?.length) {
                allSymbolData[symbol].forEach(dataPoint => {
                    allEvents.push({ timestamp: dataPoint.timestamp, symbol, price: dataPoint.price });
                });
            }
            else {
                logger.warn(`No historical data found for symbol: ${symbol}. Skipping this symbol in backtest.`);
            }
        }
        allEvents.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        if (allEvents.length === 0) {
            logger.error('No events to backtest. Ensure historical data files are populated.');
            return;
        }
        logger.info(`Starting backtest with ${allEvents.length} events across ${this.symbols.length} symbols.`);
        for (const event of allEvents) {
            const { timestamp, symbol, price } = event;
            this.basePrices[symbol] = price;
            this.simulateOrder(symbol, 'Buy', price - this.spread / 2, price - this.spread / 2);
            this.simulateOrder(symbol, 'Sell', price + this.spread / 2, price + this.spread / 2);
        }
        const metrics = this.calculateMetrics(allEvents);
        logger.info('Backtest completed with multi-symbol support', { metrics, finalEquity: this.equityCurve[this.equityCurve.length - 1] });
        return metrics;
    }
    calculateMetrics(allEvents) {
        const aggregateMetrics = {
            sharpeRatio: 0, sortinoRatio: 0, maxDrawdown: 0, winRate: 0, profitFactor: 0,
            calmarRatio: 0, totalProfit: 0, finalEquity: 0, totalTradeCount: 0,
            averageTradeProfit: 0, totalVolumeTraded: 0
        };
        const symbolMetrics = {};
        for (const symbol of this.symbols) {
            const closedTrades = this.trades[symbol].filter(t => t.status === 'Closed');
            const profits = closedTrades.map(t => t.profit);
            const totalSymbolProfit = profits.reduce((sum, p) => sum + p, 0);
            const winCount = profits.filter(p => p > 0).length;
            const lossCount = profits.filter(p => p <= 0).length;
            const totalTrades = winCount + lossCount;
            const grossProfit = profits.filter(p => p > 0).reduce((a, b) => a + b, 0);
            const grossLoss = Math.abs(profits.filter(p => p <= 0).reduce((a, b) => a + b, 0));
            const totalVolume = closedTrades.reduce((sum, t) => sum + (t.quantity * t.entryPrice), 0);
            let sharpeRatio = 0, sortinoRatio = 0, maxDrawdown = 0, winRate = 0, profitFactor = Infinity, calmarRatio = 0, averageTradeProfit = 0;
            if (totalTrades > 0) {
                winRate = (winCount / totalTrades) * 100;
                profitFactor = grossLoss > 0 ? grossProfit / grossLoss : Infinity;
                averageTradeProfit = totalSymbolProfit / totalTrades;
            }
            const equityPeak = Math.max(...this.equityCurve);
            const drawdowns = this.equityCurve.map(e => (equityPeak - e) / equityPeak);
            maxDrawdown = Math.max(...drawdowns) * 100;
            const returnsFromTrades = profits.map(p => p / (this.symbolCapitalAllocation[symbol] || this.initialCapital / this.symbols.length));
            const meanReturn = returnsFromTrades.reduce((a, b) => a + b, 0) / returnsFromTrades.length || 0;
            const stdDev = Math.sqrt(returnsFromTrades.map(r => Math.pow(r - meanReturn, 2)).reduce((a, b) => a + b, 0) / returnsFromTrades.length) || 0;
            const downsideReturns = returnsFromTrades.filter(r => r < 0);
            const downsideStdDev = Math.sqrt(downsideReturns.map(r => Math.pow(r - meanReturn, 2)).reduce((a, b) => a + b, 0) / downsideReturns.length) || 0;
            sharpeRatio = stdDev > 0 ? meanReturn / stdDev : 0;
            sortinoRatio = downsideStdDev > 0 ? meanReturn / downsideStdDev : 0;
            const annualReturn = meanReturn * (252 * 60 * 60 / (allEvents.length / totalTrades));
            calmarRatio = maxDrawdown > 0 ? annualReturn / maxDrawdown : 0;
            symbolMetrics[symbol] = {
                totalProfit: totalSymbolProfit,
                tradeCount: totalTrades,
                winRate: parseFloat(winRate.toFixed(2)),
                profitFactor: parseFloat(profitFactor.toFixed(2)),
                averageTradeProfit: parseFloat(averageTradeProfit.toFixed(4)),
                sharpeRatio: parseFloat(sharpeRatio.toFixed(4)),
                sortinoRatio: parseFloat(sortinoRatio.toFixed(4)),
                maxDrawdown: parseFloat(maxDrawdown.toFixed(2)),
                calmarRatio: parseFloat(calmarRatio.toFixed(4)),
                totalVolumeTraded: parseFloat(totalVolume.toFixed(2)),
            };
            aggregateMetrics.sharpeRatio += symbolMetrics[symbol].sharpeRatio / this.symbols.length;
            aggregateMetrics.sortinoRatio += symbolMetrics[symbol].sortinoRatio / this.symbols.length;
            aggregateMetrics.maxDrawdown = Math.max(aggregateMetrics.maxDrawdown, symbolMetrics[symbol].maxDrawdown);
            aggregateMetrics.winRate += symbolMetrics[symbol].winRate / this.symbols.length;
            aggregateMetrics.profitFactor += symbolMetrics[symbol].profitFactor / this.symbols.length;
            aggregateMetrics.calmarRatio += symbolMetrics[symbol].calmarRatio / this.symbols.length;
            aggregateMetrics.totalProfit += symbolMetrics[symbol].totalProfit;
            aggregateMetrics.totalTradeCount += symbolMetrics[symbol].tradeCount;
            aggregateMetrics.totalVolumeTraded += symbolMetrics[symbol].totalVolumeTraded;
        }
        aggregateMetrics.finalEquity = this.equityCurve[this.equityCurve.length - 1];
        aggregateMetrics.averageTradeProfit = aggregateMetrics.totalTradeCount > 0 ? aggregateMetrics.totalProfit / aggregateMetrics.totalTradeCount : 0;
        return { ...symbolMetrics, aggregate: aggregateMetrics };
    }
}
export default MultiSymbolMarketMakingStrategy;
