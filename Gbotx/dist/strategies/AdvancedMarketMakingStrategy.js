import { logger } from '../logger';
import fs from 'fs/promises';
import path from 'path'; // Import path for better file handling
export class AdvancedMarketMakingStrategy {
    constructor(config) {
        // Backtesting specific properties
        this.basePrices = {}; // Current price per symbol
        this.cash = {}; // Cash allocated per symbol
        this.symbolCapitalAllocation = {}; // Percentage or fixed amount of total capital per symbol
        this.positions = {}; // Current position per symbol
        this.trades = {}; // Organized by symbol
        this.equityCurve = [];
        // ATR calculation properties
        this.previousPrices = {};
        this.symbols = config.symbols;
        this.dataDirectory = config.dataDirectory || 'data';
        this.initialCapital = config.initialCapital || 10000;
        this.baseSpread = config.spread || 0.006; // Default spread
        this.volatilityFactor = config.volatilityFactor || 1.2; // Default volatility factor
        this.momentumWindow = 5; // Default momentum window
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
        this.equityCurve = [this.initialCapital];
        console.log('AdvancedMarketMakingStrategy initialized with:', config);
    }
    // Load historical data for all symbols
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
                logger.warn(`No data file for ${symbol} found at ${path.join(this.dataDirectory, `historical_prices_${symbol.toLowerCase()}.json`)}, initializing with empty dataset.`, { error });
                data[symbol] = [];
            }
        }
        return data;
    }
    // Calculate Average True Range (ATR) for a specific symbol
    calculateVolatility(symbol, currentPrice) {
        // For backtesting, we'll simplify by using the current price for high/low/close
        // In a live environment, you'd feed actual OHLCV data.
        const prevPriceInfo = this.previousPrices[symbol] || { high: currentPrice, low: currentPrice, close: currentPrice };
        const trueRange = Math.max(currentPrice - prevPriceInfo.low, Math.abs(currentPrice - prevPriceInfo.close), Math.abs(prevPriceInfo.close - prevPriceInfo.high));
        // Initialize or update the Exponential Moving Average of True Ranges
        // This is a simplified ATR for backtesting with only 'price' data
        // In a real scenario, you'd use a proper ATR calculation over a series of OHLC data.
        if (!this.previousPrices[symbol] || !this.previousPrices[symbol].atr) {
            this.previousPrices[symbol] = { high: currentPrice, low: currentPrice, close: currentPrice, atr: trueRange };
            return trueRange; // First ATR is just the first true range
        }
        else {
            // A simplified exponential moving average for true range (ATR)
            // For a more accurate backtest, you would need historical high/low/close for each timestamp.
            const alpha = 2 / (this.atrPeriod + 1);
            const prevAtr = this.previousPrices[symbol].atr || trueRange; // Assume 'atr' property on previousPrices for tracking
            const currentAtr = (trueRange * alpha) + (prevAtr * (1 - alpha));
            this.previousPrices[symbol].atr = currentAtr; // Store for next iteration
            this.previousPrices[symbol] = { high: currentPrice, low: currentPrice, close: currentPrice, atr: currentAtr }; // Update for next iteration
            return currentAtr;
        }
    }
    // Determine order size based on risk and volatility for a symbol
    calculateOrderSize(symbol, currentPrice) {
        const riskAmountPerSymbol = (this.initialCapital * this.riskPercent) / this.symbols.length; // Split initial risk
        const volatility = this.calculateVolatility(symbol, currentPrice);
        const stopLossDistance = volatility * this.stopLossMultiplier; // Use 3x ATR as stop loss, adjustable parameter
        if (stopLossDistance === 0) {
            logger.warn(`Zero stop-loss distance for ${symbol}. Cannot calculate order size.`, { symbol, currentPrice, volatility });
            return 0;
        }
        // Quantity based on risk management
        let quantityFromRisk = riskAmountPerSymbol / stopLossDistance;
        // Maximum affordable quantity based on available cash for this symbol's allocation
        // Assuming cash is managed globally, this is total cash / number of symbols
        // A more advanced approach would be per-symbol cash allocation.
        const maxAffordableQuantity = (this.cash[symbol] || 0) / currentPrice;
        // Ensure quantity is positive and not NaN/Infinity
        let calculatedQuantity = Math.min(Math.max(quantityFromRisk, 0), maxAffordableQuantity);
        // Round to a reasonable precision for crypto, e.g., 6 decimal places or 0.001
        calculatedQuantity = parseFloat(calculatedQuantity.toFixed(6));
        // Minimum trade quantity (e.g., 0.001 BTC or 0.01 ETH)
        const minTradeQuantity = this.minTradeQuantity; // This should ideally be market-specific or configurable
        if (calculatedQuantity < minTradeQuantity) {
            logger.debug(`Calculated quantity for ${symbol} below minimum trade quantity, setting to 0.`, { symbol, calculatedQuantity, minTradeQuantity });
            return 0; // Do not trade if quantity is too small
        }
        // Cap at a reasonable maximum to prevent excessively large orders
        const maxQuantityCap = this.maxQuantityCap; // E.g., don't trade more than 1 BTC/ETH at a time
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
    calculateOrderPrices(referencePrice, volatility, inventory, maxInventory, recentTrades, orderbook) {
        console.log('calculateOrderPrices inputs:', {
            referencePrice,
            volatility,
            inventory,
            maxInventory,
            baseSpread: this.baseSpread,
            volatilityFactor: this.volatilityFactor,
        });
        // Calculate dynamic spread based on volatility and order book depth
        let spread = this.baseSpread * (1 + volatility * this.volatilityFactor);
        if (orderbook) {
            const bidDepth = orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
            const askDepth = orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
            const depthRatio = bidDepth / askDepth;
            spread *= Math.max(0.5, Math.min(2, 1 / (Math.min(bidDepth, askDepth) / 0.01)));
            console.log('Order book depth:', { bidDepth, askDepth, depthRatio });
        }
        // Momentum adjustment
        const recentPrices = recentTrades.slice(-this.momentumWindow);
        const momentum = recentPrices.length >= 2 ? (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] : 0;
        spread *= (1 + Math.abs(momentum) * 0.5);
        // Inventory skew
        const inventorySkew = inventory / maxInventory;
        const buySpread = spread * (1 + inventorySkew);
        const sellSpread = spread * (1 - inventorySkew);
        // Profitability check: Ensure spread covers fees
        const takerFeeRate = 0.0012;
        const minProfitableSpread = 2 * takerFeeRate * referencePrice;
        const canPlaceOrders = (sellPrice - buyPrice) > minProfitableSpread;
        // Dynamic quantities - these will be overridden by calculateOrderSize in backtest
        const baseQty = 0.01;
        const buyQty = baseQty * (1 + Math.abs(inventorySkew));
        const sellQty = baseQty * (1 - Math.abs(inventorySkew));
        const buyPrice = referencePrice * (1 - buySpread / 2);
        const sellPrice = referencePrice * (1 + sellSpread / 2);
        console.log('Calculated prices and quantities:', {
            buyPrice,
            sellPrice,
            buyQty,
            sellQty,
            spread,
            inventorySkew,
            momentum,
            canPlaceOrders,
        });
        return { buyPrice, sellPrice, buyQty, sellQty, canPlaceOrders };
    }
    // Backtesting specific methods
    getTrades(symbol) {
        if (symbol) {
            return this.trades[symbol] || [];
        }
        return Object.values(this.trades).flat();
    }
    getCash(symbol) {
        if (symbol) {
            return this.cash[symbol];
        }
        return Object.values(this.cash).reduce((sum, c) => sum + c, 0);
    }
    getPosition(symbol) {
        if (symbol) {
            return this.positions[symbol];
        }
        return Object.values(this.positions).reduce((sum, p) => sum + p, 0);
    }
    simulateOrder(symbol, side, price, qty) {
        const feeRate = 0.0012; // Taker fee
        const fee = price * qty * feeRate;
        if (side === 'Buy') {
            if ((this.cash[symbol] || 0) >= price * qty) {
                this.cash[symbol] = (this.cash[symbol] || 0) - (price * qty) - fee;
                this.positions[symbol] = (this.positions[symbol] || 0) + qty;
                this.trades[symbol].push({
                    timestamp: new Date().toISOString(),
                    symbol,
                    side,
                    entryPrice: price,
                    quantity: qty,
                    profit: 0,
                    status: 'Open',
                    tradeId: `${symbol}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
                    fee,
                });
                this.updateEquity();
                logger.info(`Backtest Buy executed for ${symbol}: ${qty.toFixed(4)} at ${price.toFixed(2)}. Cash: ${this.cash[symbol]?.toFixed(2)}, Position: ${this.positions[symbol]?.toFixed(4)}`);
            }
            else {
                logger.debug(`Insufficient cash for ${symbol} buy order. Needed: ${(price * qty).toFixed(2)}, Available: ${this.cash[symbol]?.toFixed(2)}`);
            }
        }
        else { // Sell
            if ((this.positions[symbol] || 0) >= qty) {
                const proceeds = price * qty;
                this.cash[symbol] = (this.cash[symbol] || 0) + proceeds - fee;
                this.positions[symbol] = (this.positions[symbol] || 0) - qty;
                // Find the corresponding buy trade to calculate profit (simplified for market making)
                const openBuyTrades = this.trades[symbol].filter(t => t.side === 'Buy' && t.status === 'Open');
                let profit = 0;
                if (openBuyTrades.length > 0) {
                    const tradeToClose = openBuyTrades[0]; // Simplistic FIFO
                    profit = (proceeds - (tradeToClose.entryPrice * tradeToClose.quantity)) - fee; // Profit for the portion closed
                    tradeToClose.profit += profit; // Add profit to the original buy trade
                    tradeToClose.exitPrice = price;
                    tradeToClose.status = 'Closed'; // Mark as closed
                }
                this.trades[symbol].push({
                    timestamp: new Date().toISOString(),
                    symbol,
                    side,
                    entryPrice: price, // Treat sell as entry price for shorting, if applicable
                    quantity: qty,
                    profit,
                    status: 'Closed', // Mark as closed for simplicity in backtest
                    tradeId: `${symbol}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
                    fee,
                });
                this.updateEquity();
                logger.info(`Backtest Sell executed for ${symbol}: ${qty.toFixed(4)} at ${price.toFixed(2)}. Cash: ${this.cash[symbol]?.toFixed(2)}, Position: ${this.positions[symbol]?.toFixed(4)}, Profit: ${profit.toFixed(2)}`);
            }
            else {
                logger.debug(`Insufficient position for ${symbol} sell order. Needed: ${qty.toFixed(4)}, Available: ${this.positions[symbol]?.toFixed(4)}`);
            }
        }
    }
    updateEquity() {
        const totalCash = Object.values(this.cash).reduce((sum, c) => sum + c, 0);
        const totalPositionValue = Object.entries(this.positions).reduce((sum, [symbol, pos]) => sum + (pos * (this.basePrices[symbol] || 0)), 0);
        const currentEquity = totalCash + totalPositionValue;
        this.equityCurve.push(currentEquity);
        logger.info('Equity updated', { totalCash: totalCash.toFixed(2), positions: JSON.stringify(this.positions), equity: currentEquity.toFixed(2) });
    }
    calculateMetrics() {
        const aggregateMetrics = {
            sharpeRatio: 0, sortinoRatio: 0, maxDrawdown: 0, winRate: 0, profitFactor: 0,
            calmarRatio: 0, totalProfit: 0, finalEquity: 0, totalTradeCount: 0,
            averageTradeProfit: 0, totalVolumeTraded: 0
        };
        const symbolMetrics = {};
        for (const symbol of this.symbols) {
            const closedTrades = this.trades[symbol].filter(t => t.status === 'Closed'); // Only consider closed trades for profit metrics
            const profits = closedTrades.map(t => t.profit);
            const totalSymbolProfit = profits.reduce((sum, p) => sum + p, 0);
            const winCount = profits.filter(p => p > 0).length;
            const lossCount = profits.filter(p => p <= 0).length; // Including zero profit as non-winning
            const totalTrades = winCount + lossCount;
            const grossProfit = profits.filter(p => p > 0).reduce((a, b) => a + b, 0);
            const grossLoss = Math.abs(profits.filter(p => p <= 0).reduce((a, b) => a + b, 0));
            const totalVolume = closedTrades.reduce((sum, t) => sum + (t.quantity * t.entryPrice), 0); // Volume in USD/USDT
            let sharpeRatio = 0, sortinoRatio = 0, maxDrawdown = 0, winRate = 0, profitFactor = Infinity, calmarRatio = 0, averageTradeProfit = 0;
            if (totalTrades > 0) {
                winRate = (winCount / totalTrades) * 100;
                profitFactor = grossLoss > 0 ? grossProfit / grossLoss : Infinity;
                averageTradeProfit = totalSymbolProfit / totalTrades;
            }
            // Equity curve for specific symbol (requires tracking per-symbol equity, which is complex)
            // For simplicity, max drawdown is still calculated on total equity.
            const equityPeak = Math.max(...this.equityCurve);
            const drawdowns = this.equityCurve.map(e => (equityPeak - e) / equityPeak);
            maxDrawdown = Math.max(...drawdowns) * 100;
            // For Sharpe/Sortino, you'd ideally need a return series.
            // Here, we'll use a simplified approach based on trade profits for illustration.
            const returnsFromTrades = profits.map(p => p / (this.symbolCapitalAllocation[symbol] || this.initialCapital / this.symbols.length));
            const meanReturn = returnsFromTrades.reduce((a, b) => a + b, 0) / returnsFromTrades.length || 0;
            const stdDev = Math.sqrt(returnsFromTrades.map(r => Math.pow(r - meanReturn, 2)).reduce((a, b) => a + b, 0) / returnsFromTrades.length) || 0;
            const downsideReturns = returnsFromTrades.filter(r => r < 0);
            const downsideStdDev = Math.sqrt(downsideReturns.map(r => Math.pow(r - meanReturn, 2)).reduce((a, b) => a + b, 0) / downsideReturns.length) || 0;
            sharpeRatio = stdDev > 0 ? meanReturn / stdDev : 0;
            sortinoRatio = downsideStdDev > 0 ? meanReturn / downsideStdDev : 0;
            // Calmar ratio needs annual return and max drawdown. Simplified here.
            // const annualReturn = meanReturn * (252 * 60 * 60 / (allEvents.length / totalTrades)); // Rough annualization
            // calmarRatio = maxDrawdown > 0 ? annualReturn / maxDrawdown : 0;
            symbolMetrics[symbol] = {
                totalProfit: parseFloat(totalSymbolProfit.toFixed(2)),
                tradeCount: totalTrades,
                winRate: parseFloat(winRate.toFixed(2)),
                profitFactor: parseFloat(profitFactor.toFixed(2)),
                averageTradeProfit: parseFloat(averageTradeProfit.toFixed(4)),
                sharpeRatio: parseFloat(sharpeRatio.toFixed(4)),
                sortinoRatio: parseFloat(sortinoRatio.toFixed(4)),
                maxDrawdown: parseFloat(maxDrawdown.toFixed(2)),
                calmarRatio: parseFloat(calmarRatio.toFixed(4)),
                totalVolumeTraded: parseFloat(totalVolume.toFixed(2)),
                // Add more metrics as needed
            };
            // Aggregate (simple average for demonstration, but weighted averages might be better)
            aggregateMetrics.sharpeRatio += symbolMetrics[symbol].sharpeRatio / this.symbols.length;
            aggregateMetrics.sortinoRatio += symbolMetrics[symbol].sortinoRatio / this.symbols.length;
            aggregateMetrics.maxDrawdown = Math.max(aggregateMetrics.maxDrawdown, symbolMetrics[symbol].maxDrawdown); // Max of all symbols
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
    // Backtest with multi-symbol support, processing data chronologically
    async runBacktest(historicalDataInput) {
        this.cash = {}; // Reset cash for each symbol
        this.positions = {}; // Reset positions for each symbol
        this.trades = {}; // Reset trades for each symbol
        this.equityCurve = [this.initialCapital]; // Reset equity curve
        const allSymbolData = historicalDataInput || await this.loadHistoricalData();
        const allEvents = [];
        // Flatten all historical data into a single, time-ordered list of events
        for (const symbol of this.symbols) {
            this.cash[symbol] = this.initialCapital / this.symbols.length; // Re-initialize cash per symbol
            this.positions[symbol] = 0;
            this.trades[symbol] = [];
            if (allSymbolData[symbol]?.length) {
                allSymbolData[symbol].forEach(dataPoint => {
                    allEvents.push({ timestamp: dataPoint.timestamp, symbol, price: dataPoint.price });
                });
            }
            else {
                logger.warn(`No historical data found for symbol: ${symbol}. Skipping this symbol in backtest.`);
            }
        }
        // Sort all events by timestamp to simulate chronological market updates
        allEvents.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        if (allEvents.length === 0) {
            logger.error('No events to backtest. Ensure historical data files are populated.');
            return;
        }
        logger.info(`Starting backtest with ${allEvents.length} events across ${this.symbols.length} symbols.`);
        for (const event of allEvents) {
            const { timestamp, symbol, price } = event;
            this.basePrices[symbol] = price; // Update the current price for the symbol
            const qty = this.calculateOrderSize(symbol, price); // Dynamic quantity per symbol
            // Simulate market making by trying to buy below and sell above current price
            const buyPrice = price * (1 - this.baseSpread / 2);
            const sellPrice = price * (1 + this.baseSpread / 2);
            this.simulateOrder(symbol, 'Buy', buyPrice, qty);
            this.simulateOrder(symbol, 'Sell', sellPrice, qty);
        }
        const metrics = this.calculateMetrics();
        logger.info('Backtest completed with multi-symbol support', { metrics });
        return metrics;
    }
}
