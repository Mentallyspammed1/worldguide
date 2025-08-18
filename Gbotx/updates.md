Got it! You're looking for 10 upgrades to your BasicMarketMakingStrategy.ts for multi-symbol support, building on the provided code. This is an awesome enhancement for your Bybit trading bot, Gbotx, especially for HFT and conditional order sizing across multiple assets. As a Python and JS coder who loves complete, enhanced code, and neon colorization, I'll provide detailed, enhanced upgrades to make your bot even more robust. Let's make this Gbotx shine with these upgrades! 😻
Here are 10 significant upgrades to your BasicMarketMakingStrategy.ts for multi-symbol support, along with explanations and enhanced code where applicable:
1. Dynamic Symbol Configuration and Management
Instead of hardcoding symbols, let's make them configurable. This allows for easier expansion and testing with different asset pairs without code changes.
// strategies/BasicMarketMakingStrategy.ts
import logger from '../logger';
import fs from 'fs/promises';
import path from 'path'; // Import path for better file handling

interface HistoricalDataPoint {
  timestamp: string;
  price: number;
}

interface HistoricalData {
  [key: string]: HistoricalDataPoint[];
}

// ... inside BasicMarketMakingStrategy class

private symbols: string[]; // No longer hardcoded
private dataDirectory: string; // To configure data path

constructor(symbols: string[] = ['BTCUSDT', 'ETHUSDT'], dataDirectory: string = 'data') {
  this.symbols = symbols;
  this.dataDirectory = dataDirectory;
  this.symbols.forEach(symbol => {
    this.positions[symbol] = 0;
    this.trades[symbol] = [];
    this.basePrices[symbol] = 0; // Initialize base prices
  });
}

// Load historical data for all symbols
async loadHistoricalData(): Promise<HistoricalData> {
  const data: HistoricalData = {};
  for (const symbol of this.symbols) {
    try {
      const filePath = path.join(this.dataDirectory, `historical_prices_${symbol.toLowerCase()}.json`);
      const fileContent = await fs.readFile(filePath, 'utf-8');
      data[symbol] = JSON.parse(fileContent);
      logger.info(`Loaded historical data for ${symbol} from ${filePath}`);
    } catch (error) {
      logger.warning(`No data file for ${symbol} found at ${path.join(this.dataDirectory, `historical_prices_${symbol.toLowerCase()}.json`)}, initializing with empty dataset.`, { error });
      data[symbol] = [];
    }
  }
  return data;
}

Enhancement:
 * symbols array as a constructor parameter: This makes the strategy more flexible.
 * dataDirectory parameter: Allows specifying where historical data files are located.
 * path.join: Ensures cross-platform compatibility for file paths.
 * Improved Logging: More descriptive messages during data loading.
2. Robust Volatility Calculation with ATR
Your current calculateVolatility is a very basic true range. Let's implement a more standard Average True Range (ATR) calculation for a more robust volatility measure, which is crucial for HFT strategies.
// strategies/BasicMarketMakingStrategy.ts
// ... (imports and other class members)

// Add a property to store previous prices for ATR calculation
private previousPrices: { [key: string]: { high: number; low: number; close: number } } = {};
private atrPeriod: number = 14; // Common ATR period

// Calculate Average True Range (ATR) for a specific symbol
private calculateVolatility(symbol: string, currentPrice: number): number {
  // For backtesting, we'll simplify by using the current price for high/low/close
  // In a live environment, you'd feed actual OHLCV data.
  const prevPriceInfo = this.previousPrices[symbol] || { high: currentPrice, low: currentPrice, close: currentPrice };

  const trueRange = Math.max(
    currentPrice - prevPriceInfo.low,
    Math.abs(currentPrice - prevPriceInfo.close),
    Math.abs(prevPriceInfo.close - prevPriceInfo.high)
  );

  // Initialize or update the Exponential Moving Average of True Ranges
  // This is a simplified ATR for backtesting with only 'price' data
  // In a real scenario, you'd use a proper ATR calculation over a series of OHLC data.
  if (!this.previousPrices[symbol]) {
    this.previousPrices[symbol] = { high: currentPrice, low: currentPrice, close: currentPrice };
    return trueRange; // First ATR is just the first true range
  } else {
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

// ... (rest of the class)

Enhancement:
 * ATR Concept: Introduces the idea of Average True Range for a more accurate volatility measure.
 * Simplified ATR Calculation for Backtest: Acknowledges the limitation of price only data and provides a simplified EMA-like ATR. Note: For a truly accurate ATR in backtesting, your historical data would need high, low, and close prices for each timestamp.
3. Advanced Order Sizing with Dynamic Stop Loss
Your current calculateOrderSize is a good start. Let's enhance it to use the new ATR for stop-loss distance and ensure it considers available capital more dynamically.
// strategies/BasicMarketMakingStrategy.ts
// ... (imports and other class members)

// Determine order size based on risk and volatility for a symbol
private calculateOrderSize(symbol: string, currentPrice: number): number {
  const riskAmountPerSymbol = (this.initialCapital * this.riskPercent) / this.symbols.length; // Split initial risk
  const volatility = this.calculateVolatility(symbol, currentPrice);
  const stopLossDistance = volatility * 3; // Use 3x ATR as stop loss, adjustable parameter

  if (stopLossDistance === 0) {
    logger.warn(`Zero stop-loss distance for ${symbol}. Cannot calculate order size.`, { symbol, currentPrice, volatility });
    return 0;
  }

  // Quantity based on risk management
  let quantityFromRisk = riskAmountPerSymbol / stopLossDistance;

  // Maximum affordable quantity based on available cash for this symbol's allocation
  // Assuming cash is managed globally, this is total cash / number of symbols
  // A more advanced approach would be per-symbol cash allocation.
  const maxAffordableQuantity = (this.cash / this.symbols.length) / currentPrice;

  // Ensure quantity is positive and not NaN/Infinity
  let calculatedQuantity = Math.min(Math.max(quantityFromRisk, 0), maxAffordableQuantity);

  // Round to a reasonable precision for crypto, e.g., 6 decimal places or 0.001
  calculatedQuantity = parseFloat(calculatedQuantity.toFixed(6));

  // Minimum trade quantity (e.g., 0.001 BTC or 0.01 ETH)
  const minTradeQuantity = 0.001; // This should ideally be market-specific or configurable
  if (calculatedQuantity < minTradeQuantity) {
    logger.debug(`Calculated quantity for ${symbol} below minimum trade quantity, setting to 0.`, { symbol, calculatedQuantity, minTradeQuantity });
    return 0; // Do not trade if quantity is too small
  }

  // Cap at a reasonable maximum to prevent excessively large orders
  const maxQuantityCap = 1; // E.g., don't trade more than 1 BTC/ETH at a time
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

Enhancement:
 * ATR-based Stop Loss: Uses the calculated ATR for stop loss distance, making it more dynamic.
 * Explicit Risk per Symbol: Clarifies how riskPercent is distributed.
 * Minimum and Maximum Quantity Caps: Prevents tiny or excessively large trades.
 * Rounding: Ensures quantities are practical for crypto exchanges.
 * Robustness: Checks for zero stopLossDistance.
 * Enhanced Logging: Detailed debug logs for order size calculation.
4. Per-Symbol Cash Allocation (Optional but Recommended)
For more precise risk management in a multi-symbol setup, consider allocating a portion of the initialCapital to each symbol, rather than just splitting the riskPercent. This requires a more significant refactor but offers better control.
// strategies/BasicMarketMakingStrategy.ts
// ... (imports and other class members)

private cash: { [key: string]: number } = {}; // Cash allocated per symbol
private symbolCapitalAllocation: { [key: string]: number } = {}; // Percentage or fixed amount of total capital per symbol

constructor(symbols: string[] = ['BTCUSDT', 'ETHUSDT'], dataDirectory: string = 'data') {
  this.symbols = symbols;
  this.dataDirectory = dataDirectory;
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

// Update simulateOrder to use per-symbol cash
private simulateOrder(symbol: string, side: 'Buy' | 'Sell', triggerPrice: number, orderPrice: number) {
  if ((side === 'Buy' && this.basePrices[symbol] <= triggerPrice) || (side === 'Sell' && this.basePrices[symbol] >= triggerPrice)) {
    const quantity = this.calculateOrderSize(symbol, this.basePrices[symbol]);
    const executedPrice = orderPrice;

    if (quantity === 0) {
      logger.debug(`Order quantity for ${symbol} is zero, skipping trade.`, { symbol, side, triggerPrice, orderPrice });
      return;
    }

    if (side === 'Buy') {
      const cost = executedPrice * quantity;
      if (this.cash[symbol] >= cost) {
        this.cash[symbol] -= cost;
        this.positions[symbol] += quantity;
        this.trades[symbol].push({ timestamp: new Date().toISOString(), price: executedPrice, side, profit: 0, quantity });
        this.updateEquity();
        logger.info(`Backtest ${side} executed for ${symbol}`, { price: executedPrice, quantity, cash: this.cash[symbol], position: this.positions[symbol] });
      } else {
        logger.debug(`Insufficient cash for ${symbol} buy order. Needed: ${cost.toFixed(2)}, Available: ${this.cash[symbol].toFixed(2)}`, { symbol, price: executedPrice, quantity });
      }
    } else if (side === 'Sell') {
      if (this.positions[symbol] >= quantity) {
        const proceeds = executedPrice * quantity;
        this.cash[symbol] += proceeds;
        this.positions[symbol] -= quantity;

        // Find the corresponding buy trade to calculate profit (simplified for market making)
        // In a real scenario, you'd manage individual trade lots or average cost.
        // For simplicity here, we assume profit is calculated against the average entry price or simply the last buy.
        let profit = 0;
        const buyTradesForSymbol = this.trades[symbol].filter(t => t.side === 'Buy' && t.profit === 0);
        if (buyTradesForSymbol.length > 0) {
            // Very simplistic: assume the sell closes the oldest open buy position.
            // A proper FIFO/LIFO or average cost tracking would be more accurate.
            const correspondingBuy = buyTradesForSymbol.shift(); // Takes the oldest "unclosed" buy
            if (correspondingBuy) {
                profit = proceeds - (correspondingBuy.price * correspondingBuy.quantity);
                correspondingBuy.profit = profit; // Update the profit on the "closed" buy trade
            }
        }
        this.trades[symbol].push({ timestamp: new Date().toISOString(), price: executedPrice, side, profit, quantity }); // Record the sell trade
        this.updateEquity();
        logger.info(`Backtest ${side} executed for ${symbol}`, { price: executedPrice, quantity, cash: this.cash[symbol], position: this.positions[symbol], profit });
      } else {
        logger.debug(`Insufficient position for ${symbol} sell order. Needed: ${quantity.toFixed(6)}, Available: ${this.positions[symbol].toFixed(6)}`, { symbol, price: executedPrice, quantity });
      }
    }
  }
}

// Update equity to reflect total cash across all symbols
private updateEquity() {
  const totalCash = Object.values(this.cash).reduce((sum, c) => sum + c, 0);
  const totalPositionValue = Object.entries(this.positions).reduce((sum, [symbol, pos]) => sum + (pos * (this.basePrices[symbol] || 0)), 0);
  const currentEquity = totalCash + totalPositionValue;
  this.equityCurve.push(currentEquity);
  logger.info('Equity updated', { totalCash, positions: this.positions, equity: currentEquity });
}

Enhancement:
 * Per-Symbol Cash: this.cash becomes an object {[symbol: string]: number}.
 * Dynamic Capital Allocation: initialCapital is split among symbols at initialization.
 * Clearer Trade Logic: More robust checks for quantity being zero and clearer logging for insufficient funds/positions.
 * Profit Calculation Adjustment: A very simplified profit calculation for market making when closing positions (ideally, you'd track individual buy/sell lots).
5. Improved Backtesting Loop and Event Handling
The current backtest loop iterates through symbols then their data. A more realistic backtest would process data chronologically across all symbols, simulating real-time market events. This is a significant refactor.
// strategies/BasicMarketMakingStrategy.ts
// ... (imports and other class members)

// Backtest with multi-symbol support, processing data chronologically
async backtest() {
  const allSymbolData = await this.loadHistoricalData();
  const allEvents: { timestamp: string; symbol: string; price: number }[] = [];

  // Flatten all historical data into a single, time-ordered list of events
  for (const symbol of this.symbols) {
    if (allSymbolData[symbol]?.length) {
      allSymbolData[symbol].forEach(dataPoint => {
        allEvents.push({ timestamp: dataPoint.timestamp, symbol, price: dataPoint.price });
      });
    } else {
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

    // Implement your market-making logic for the current symbol and price
    // Place bid (buy) order
    this.simulateOrder(symbol, 'Buy', price - this.spread / 2, price - this.spread / 2); // Bid at (current price - spread/2)

    // Place ask (sell) order
    this.simulateOrder(symbol, 'Sell', price + this.spread / 2, price + this.spread / 2); // Ask at (current price + spread/2)

    // Update equity after processing orders at this timestamp across all symbols that might have traded
    // (Note: updateEquity is called within simulateOrder, so it reflects per-trade changes)
  }

  const metrics = this.calculateMetrics();
  logger.info('Backtest completed with multi-symbol support', { metrics, finalEquity: this.equityCurve[this.equityCurve.length - 1] });
  return metrics;
}

Enhancement:
 * Chronological Event Processing: Creates a single sorted list of all market events across all symbols.
 * Realistic Simulation: This ensures that market-making decisions are based on the correct chronological order of price updates, which is vital for HFT.
 * Clearer Backtest Flow: The loop now clearly processes events one by one.
6. Granular Trade Tracking with Entry/Exit Prices
The current trades array tracks some details, but for more advanced metric calculation, tracking entry and exit prices and linking them is crucial. This is particularly important for profit calculation and understanding individual trade performance.
// strategies/BasicMarketMakingStrategy.ts
// ... (imports and other class members)

interface TradeRecord {
  timestamp: string;
  symbol: string; // Add symbol to trade record
  side: 'Buy' | 'Sell';
  entryPrice: number;
  exitPrice?: number; // Optional exit price for closed trades
  quantity: number;
  profit: number; // Profit specific to this trade instance
  status: 'Open' | 'Closed'; // Track if trade is open or closed
  tradeId: string; // Unique ID for each trade instance
}

private trades: { [key: string]: TradeRecord[] } = {}; // Organized by symbol

// ... (inside simulateOrder)

// When a BUY order is executed:
this.trades[symbol].push({
  timestamp: new Date().toISOString(),
  symbol: symbol,
  side: side,
  entryPrice: executedPrice,
  quantity: quantity,
  profit: 0,
  status: 'Open',
  tradeId: `${symbol}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}` // Unique ID
});

// When a SELL order is executed:
// Instead of just adding a sell trade, let's try to 'close' an open buy trade
const openBuyTrades = this.trades[symbol].filter(t => t.side === 'Buy' && t.status === 'Open');
if (openBuyTrades.length > 0) {
    // Find the oldest open buy trade to close (simplistic FIFO for market making)
    const tradeToClose = openBuyTrades[0];
    const profit = (executedPrice - tradeToClose.entryPrice) * quantity; // Profit for the portion closed
    tradeToClose.profit += profit; // Add profit to the original buy trade
    tradeToClose.exitPrice = executedPrice;
    tradeToClose.status = 'Closed'; // Mark as closed

    logger.info(`Backtest ${side} executed for ${symbol} (closing trade)`, {
        price: executedPrice,
        quantity,
        cash: this.cash[symbol],
        position: this.positions[symbol],
        profit,
        closedTradeId: tradeToClose.tradeId
    });
} else {
    // If no open buy trades, this might be a short sell (not currently handled) or an erroneous state.
    // For a pure market maker, sells should ideally close prior buys.
    // We'll still record it as a standalone sell for now.
    this.trades[symbol].push({
        timestamp: new Date().toISOString(),
        symbol: symbol,
        side: side,
        entryPrice: executedPrice, // Treat sell as entry price for shorting, if applicable
        quantity: quantity,
        profit: 0, // Profit will be calculated when covered
        status: 'Open', // Treat as open short position if not closing a long
        tradeId: `${symbol}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
    });
    logger.info(`Backtest ${side} executed for ${symbol} (standalone)`, { price: executedPrice, quantity, cash: this.cash[symbol], position: this.positions[symbol] });
}

Enhancement:
 * TradeRecord Interface: Defines a more detailed structure for each trade.
 * status and tradeId: Helps in managing and linking trades.
 * Improved Profit Calculation: Attempts to link sell orders to buy orders for more accurate per-trade profit. This is still simplistic for a true market maker (which might have many open bids/asks), but it's an improvement. For complex market making, you'd need a more sophisticated order book simulation.
7. Configuration for Strategy Parameters
Hardcoding parameters like spread, riskPercent, atrPeriod, etc., makes optimization difficult. Externalize them for easier tuning.
// strategies/BasicMarketMakingStrategy.ts
// ... (imports and other class members)

interface StrategyConfig {
  symbols: string[];
  dataDirectory?: string;
  initialCapital?: number;
  spread?: number;
  riskPercent?: number;
  atrPeriod?: number;
  stopLossMultiplier?: number;
  minTradeQuantity?: number;
  maxQuantityCap?: number;
}

class BasicMarketMakingStrategy {
  // ... existing private members

  private spread: number;
  private riskPercent: number;
  private atrPeriod: number;
  private stopLossMultiplier: number;
  private minTradeQuantity: number;
  private maxQuantityCap: number;

  constructor(config: StrategyConfig) {
    this.symbols = config.symbols;
    this.dataDirectory = config.dataDirectory || 'data';
    this.initialCapital = config.initialCapital || 10000;
    this.spread = config.spread || 50;
    this.riskPercent = config.riskPercent || 0.01;
    this.atrPeriod = config.atrPeriod || 14;
    this.stopLossMultiplier = config.stopLossMultiplier || 3;
    this.minTradeQuantity = config.minTradeQuantity || 0.001;
    this.maxQuantityCap = config.maxQuantityCap || 1;

    this.symbols.forEach(symbol => {
      this.positions[symbol] = 0;
      this.trades[symbol] = [];
      this.basePrices[symbol] = 0;
      // Initialize cash per symbol based on initialCapital or specified allocation
      this.cash[symbol] = this.initialCapital / this.symbols.length;
      this.symbolCapitalAllocation[symbol] = this.initialCapital / this.symbols.length;
    });
    this.equityCurve = [this.initialCapital];
  }

  // ... (update calculateOrderSize to use this.stopLossMultiplier, etc.)
  private calculateOrderSize(symbol: string, currentPrice: number): number {
    // ...
    const stopLossDistance = volatility * this.stopLossMultiplier; // Use configurable multiplier
    // ...
    const minTradeQuantity = this.minTradeQuantity;
    // ...
    const maxQuantityCap = this.maxQuantityCap;
    // ...
  }
}

Enhancement:
 * StrategyConfig Interface: Defines a clear structure for strategy parameters.
 * Constructor Accepts Config: All parameters are passed through a config object, making the strategy highly configurable.
 * Default Values: Provides sensible defaults if parameters are not provided.
8. Enhanced Metrics Calculation
The metrics calculation can be improved by ensuring robustness (e.g., handling zero trades) and adding more relevant HFT-specific metrics like average trade duration or profitability per unit of volume.
// strategies/BasicMarketMakingStrategy.ts
// ... (imports and other class members)

// Calculate advanced metrics per symbol and aggregate
private calculateMetrics(): { [key: string]: any } & { aggregate: any } {
  const aggregateMetrics = {
    sharpeRatio: 0, sortinoRatio: 0, maxDrawdown: 0, winRate: 0, profitFactor: 0,
    calmarRatio: 0, totalProfit: 0, finalEquity: 0, totalTradeCount: 0,
    averageTradeProfit: 0, totalVolumeTraded: 0
  };
  const symbolMetrics: { [key: string]: any } = {};

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
    const annualReturn = meanReturn * (252 * 60 * 60 / (allEvents.length / totalTrades)); // Rough annualization
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

Enhancement:
 * Focus on Closed Trades: Metrics like profit factor and win rate are more meaningful for closed trades.
 * Robustness for Zero Trades: Handles cases where there are no trades to avoid division by zero errors.
 * totalVolumeTraded: A useful metric for market makers.
 * Precision Formatting: Rounds metrics to a reasonable number of decimal places for cleaner output.
 * Simplified Annualization: Provides a very basic annualization for Sharpe/Calmar in a backtest context.
9. Comprehensive Logging with Winston
Your existing Winston logger is great. Let's ensure it's utilized effectively with clear, structured logs for multi-symbol operations, especially for debugging.
// logger.ts (Ensure your logger setup includes transport for file and console)
import winston from 'winston';
import 'winston-daily-rotate-file'; // For rotating log files

const logFormat = winston.format.printf(({ level, message, timestamp, ...metadata }) => {
  let msg = `${timestamp} [${level.toUpperCase()}] ${message}`;
  if (Object.keys(metadata).length > 0) {
    msg += ` ${JSON.stringify(metadata, null, 2)}`; // Pretty print metadata
  }
  return msg;
});

const logger = winston.createLogger({
  level: process.env.NODE_ENV === 'production' ? 'info' : 'debug', // Default level
  format: winston.format.combine(
    winston.format.timestamp(),
    logFormat,
    winston.format.colorize({ all: true }) // Neon colorization for console output
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.DailyRotateFile({
      filename: 'bot-%DATE%.log',
      datePattern: 'YYYY-MM-DD',
      zippedArchive: true,
      maxSize: '20m', // Rotate after 20MB
      maxFiles: '14d', // Keep logs for 14 days
      level: 'info' // File logs at info level or higher
    })
  ],
});

export default logger;

Enhancement:
 * winston-daily-rotate-file: Essential for managing log file size, especially for HFT.
 * colorize({ all: true }): Explicitly enables neon colorization for all log levels in the console, as you like!
 * Structured Metadata: JSON.stringify(metadata, null, 2) for pretty-printing context.
 * Log Levels: Sets debug for development and info for production, giving you fine-grained control.
10. cli.tsx and setup.sh Updates for Dynamic Symbol Use
Finally, ensure your CLI can leverage the new configurable strategy and that your setup script can handle multiple data files more generically.
// cli.tsx (or cli.ts if not using React components)
import BasicMarketMakingStrategy from './strategies/BasicMarketMakingStrategy';
import logger from './logger';
import { program } from 'commander'; // For better CLI argument parsing

program
  .option('-s, --symbols <symbols>', 'Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)', 'BTCUSDT,ETHUSDT')
  .option('-c, --capital <number>', 'Initial capital for the strategy', parseFloat)
  .option('-d, --data-dir <path>', 'Directory for historical data files', 'data')
  .option('-p, --spread <number>', 'Spread for market making', parseFloat)
  .option('-r, --risk-percent <number>', 'Percentage of capital at risk per trade', parseFloat)
  .parse(process.argv);

const options = program.opts();

async function main() {
  logger.info('Starting Gbotx Backtesting Module');

  try {
    const symbols = options.symbols.split(',').map((s: string) => s.trim().toUpperCase());
    const config = {
      symbols: symbols,
      dataDirectory: options.dataDir,
      initialCapital: options.capital,
      spread: options.spread,
      riskPercent: options.riskPercent,
      // Add other configurable parameters here
    };

    const strategy = new BasicMarketMakingStrategy(config);
    logger.info('Strategy initialized with configuration:', config);

    const metrics = await strategy.backtest();
    logger.info('Backtest Results:', { metrics });
  } catch (error) {
    logger.error('An error occurred during backtesting:', { error: error.message, stack: error.stack });
  }
}

main();

# setup.sh
setup_backtest_data() {
  log_info "Setting up backtest data..."
  if [ ! -d data ]; then
    mkdir data
    log_info "Created data directory."
  fi

  # Define symbols to create data for
  SYMBOLS=("btcusdt" "ethusdt" "solusdt") # Added SOLUSDT as an example

  for symbol in "${SYMBOLS[@]}"; do
    if [ ! -f "data/historical_prices_${symbol}.json" ]; then
      log_info "Creating sample historical_prices_${symbol}.json..."
      case "$symbol" in
        "btcusdt")
          echo '[
            {"timestamp": "2025-07-20T08:00:00Z", "price": 50000},
            {"timestamp": "2025-07-20T08:00:01Z", "price": 50010},
            {"timestamp": "2025-07-20T08:00:02Z", "price": 50020},
            {"timestamp": "2025-07-20T08:00:03Z", "price": 50015},
            {"timestamp": "2025-07-20T08:00:04Z", "price": 50025},
            {"timestamp": "2025-07-20T08:00:05Z", "price": 50030}
          ]' > "data/historical_prices_${symbol}.json"
          ;;
        "ethusdt")
          echo '[
            {"timestamp": "2025-07-20T08:00:00Z", "price": 3000},
            {"timestamp": "2025-07-20T08:00:01Z", "price": 3005},
            {"timestamp": "2025-07-20T08:00:02Z", "price": 3010},
            {"timestamp": "2025-07-20T08:00:03Z", "price": 3007},
            {"timestamp": "2025-07-20T08:00:04Z", "price": 3012},
            {"timestamp": "2025-07-20T08:00:05Z", "price": 3015}
          ]' > "data/historical_prices_${symbol}.json"
          ;;
        "solusdt")
          echo '[
            {"timestamp": "2025-07-20T08:00:00Z", "price": 150},
            {"timestamp": "2025-07-20T08:00:01Z", "price": 150.5},
            {"timestamp": "2025-07-20T08:00:02Z", "price": 151},
            {"timestamp": "2025-07-20T08:00:03Z", "price": 150.8},
            {"timestamp": "2025-07-20T08:00:04Z", "price": 151.2},
            {"timestamp": "2025-07-20T08:00:05Z", "price": 151.5}
          ]' > "data/historical_prices_${symbol}.json"
          ;;
        *)
          log_warning "No specific sample data for $symbol, creating generic empty file."
          echo '[]' > "data/historical_prices_${symbol}.json"
          ;;
      esac
      log_success "Sample historical_prices_${symbol}.json created. Replace with real data as needed."
    else
      log_info "Historical data for ${symbol} already exists. Skipping creation."
    fi
  done
}

Enhancement:
 * commander for CLI: Provides a more professional and extensible way to handle command-line arguments.
 * Dynamic Symbol Input: Users can now specify symbols directly from the command line.
 * Configurable Parameters: Pass capital, spread, etc., directly via CLI.
 * Generic setup.sh: Uses an array of SYMBOLS and a case statement to handle data creation for different symbols more cleanly.
 * Added SOLUSDT: As an example to show how easy it is to add more symbols.
Remember to install commander and winston-daily-rotate-file:
npm install commander winston-daily-rotate-file
These upgrades significantly enhance the multi-symbol backtesting capabilities of your Gbotx, making it more flexible, robust, and capable of generating more insightful metrics. Let me know if you want to dive deeper into any of these, or if you have more ideas for Gbotx! 😻
