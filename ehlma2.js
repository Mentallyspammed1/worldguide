const ccxt = require('ccxt');
require('dotenv').config();
const { performance } = require('perf_hooks'); // For more precise timing if needed

// --- Configuration ---
const config = {
    exchange: 'bybit', // Or 'binance', 'kucoin', etc. (ensure compatibility)
    symbol: 'BTC/USDT:USDT', // Unified symbol format for derivatives
    timeframe: '1m', // 1-minute candles
    tradeAmountQuote: 10, // Amount to trade in Quote currency (e.g., 10 USDT)
    leverage: 5, // Set leverage

    // Ehlers Super Smoother Periods
    fastMaPeriod: 10,
    slowMaPeriod: 20,

    // ATR Configuration
    atrPeriod: 14,
    atrSmoothPeriod: 10, // Period for smoothing ATR

    // Trailing Stop Configuration
    trailingStopPercent: 0.5, // Trailing stop activation percentage (Client-side)

    // Order Book Analysis Configuration
    orderBookDepth: 50, // How many levels deep to fetch and analyze (increased for wall detection)
    imbalanceDepth: 10, // Levels for simple imbalance calculation
    imbalanceThreshold: 0.2, // Threshold for simple imbalance signal
    weightedImbalanceDepth: 20, // Levels for weighted imbalance (increased for deeper analysis)
    maxSpreadPercent: 0.05, // Maximum allowed spread % ((ask-bid)/mid) for market orders (0.05% = 5 basis points)
    wallDetectDepth: 20, // How many levels near BBO to check for walls (increased depth)
    wallSizeThresholdMultiplier: 5, // Order size must be X times the average size in the wallDetectDepth to be considered a wall

    // Order Placement Strategy
    useLimitOrders: true, // Try to use limit orders for better entry?
    limitOrderPriceOffsetTicks: 1, // How many ticks inside the spread to place limit orders (0 = at BBO, 1 = 1 tick better)

    // Bot Control
    maxBufferSize: 200,
    logLevel: 'info', // 'debug', 'info', 'warn', 'error'
    rateLimitBufferMs: 500,
};

// --- Logging Utility ---
const logger = {
    debug: (message, ...args) => config.logLevel === 'debug' && console.debug(`[DEBUG] ${message}`, ...args),
    info: (message, ...args) => ['debug', 'info'].includes(config.logLevel) && console.log(`[INFO] ${message}`, ...args),
    warn: (message, ...args) => ['debug', 'info', 'warn'].includes(config.logLevel) && console.warn(`[WARN] ${message}`, ...args),
    error: (message, ...args) => console.error(`[ERROR] ${message}`, ...args),
};

// --- Indicator Calculations ---
/**
 * Calculates the Ehlers Two-Pole Super Smoother Filter.
 * @param {number[]} prices - Array of prices.
 * @param {number} period - The filter period.
 * @returns {number[]} - Array of smoothed values.
 */
function calculateEhlersSuperSmoother(prices, period) {
    if (period < 2) {
        logger.warn(`EhlersSuperSmoother period must be >= 2. Received ${period}. Returning input prices.`);
        return [...prices];
    }
    if (prices.length < 2) {
        return Array(prices.length).fill(null);
    }

    const result = Array(prices.length).fill(null);
    // Note: Original Ehlers might use specific constants, this follows common implementations
    const a1 = Math.exp(-Math.sqrt(2) * Math.PI / period);
    const coeff2 = 2 * a1 * Math.cos(Math.sqrt(2) * Math.PI / period); // Using radians
    const coeff3 = -a1 * a1;
    const coeff1 = 1 - coeff2 - coeff3;

    // Initialization requires care
    result[0] = prices[0];
    if (prices.length > 1) {
        // Initialize second point carefully, e.g., simple average or repeat first value adjusted
        result[1] = (coeff1 / 2) * (prices[1] + prices[0]) + coeff2 * (result[0] || prices[0]); // Example initialization
    }


    for (let i = 2; i < prices.length; i++) {
        // Ensure previous values are numbers; fallback if needed (though should be populated by this point if length >= 2)
        const prev1 = result[i-1] ?? prices[i-1]; // Use previous price as fallback if smoother value is null
        const prev2 = result[i-2] ?? prices[i-2];
        result[i] = (coeff1 * (prices[i] + prices[i - 1]) / 2) + (coeff2 * prev1) + (coeff3 * prev2);
    }
    return result;
}

/** Calculates True Range (TR) */
function calculateTR(highs, lows, closes) {
     if (!highs || !lows || !closes || highs.length !== lows.length || lows.length !== closes.length) {
        logger.error('TR Calculation: Input arrays must exist and have the same length.');
        return [];
    }
    if (highs.length === 0) return [];

    // First TR calculation needs careful handling if only 1 data point exists
    const firstTR = highs.length > 0 ? highs[0] - lows[0] : 0; // Or handle differently if needed
    const tr = [Math.max(firstTR, 0)]; // Ensure TR is non-negative

    for (let i = 1; i < highs.length; i++) {
        const high = highs[i];
        const low = lows[i];
        const prevClose = closes[i - 1]; // This is safe due to loop starting at 1
        tr.push(Math.max(
            high - low,
            Math.abs(high - prevClose),
            Math.abs(low - prevClose)
        ));
    }
    return tr;
}


// --- Enhanced Order Book Analysis ---

/**
 * Analyzes the order book for deeper insights.
 * @param {object} orderbook - CCXT order book structure.
 * @param {object} market - CCXT market structure.
 * @param {object} cfg - The configuration object.
 * @returns {object|null} - Analysis results or null if data is insufficient.
 */
function analyzeOrderBook(orderbook, market, cfg) {
    // Added robustness checks
    if (!market || !market.precision) {
        logger.error('Market data or precision missing for order book analysis.');
        return null;
    }
     if (!orderbook || !orderbook.bids || !orderbook.asks || !orderbook.bids.length || !orderbook.asks.length) {
        logger.debug('Order book data missing or empty for analysis.');
        return null;
    }

    const bestBidPrice = orderbook.bids[0][0];
    const bestAskPrice = orderbook.asks[0][0];

    // Avoid division by zero or invalid prices
    if (bestBidPrice <= 0 || bestAskPrice <= 0 || bestBidPrice >= bestAskPrice) {
        logger.warn(`Invalid best bid/ask prices: Bid=${bestBidPrice}, Ask=${bestAskPrice}. Skipping analysis.`);
        return null;
    }

    const midPrice = (bestBidPrice + bestAskPrice) / 2;
    const spread = bestAskPrice - bestBidPrice;
    const spreadPercent = (spread / midPrice) * 100;

    // --- Simple Imbalance (Top N levels) ---
    const topBids = orderbook.bids.slice(0, cfg.imbalanceDepth);
    const topAsks = orderbook.asks.slice(0, cfg.imbalanceDepth);
    const topBidVolume = topBids.reduce((sum, [_, vol]) => sum + vol, 0);
    const topAskVolume = topAsks.reduce((sum, [_, vol]) => sum + vol, 0);
    const totalTopVolume = topBidVolume + topAskVolume;
    const simpleImbalance = totalTopVolume > 0 ? (topBidVolume - topAskVolume) / totalTopVolume : 0;

    // --- Weighted Imbalance ---
    let weightedBidVolume = 0;
    let weightedAskVolume = 0;
    const bidsForWeighted = orderbook.bids.slice(0, cfg.weightedImbalanceDepth);
    const asksForWeighted = orderbook.asks.slice(0, cfg.weightedImbalanceDepth);

    // Weighting logic - inverse distance
    for (const [price, volume] of bidsForWeighted) {
        const distance = Math.max(midPrice - price, 1e-9); // Avoid division by zero
        const weight = 1 / distance;
        weightedBidVolume += volume * weight;
    }
    for (const [price, volume] of asksForWeighted) {
        const distance = Math.max(price - midPrice, 1e-9); // Avoid division by zero
        const weight = 1 / distance;
        weightedAskVolume += volume * weight;
    }
    const totalWeightedVolume = weightedBidVolume + weightedAskVolume;
    const weightedImbalance = totalWeightedVolume > 0 ? (weightedBidVolume - weightedAskVolume) / totalWeightedVolume : 0;


    // --- Wall Detection ---
    const bidWallCandidates = orderbook.bids.slice(0, cfg.wallDetectDepth);
    const askWallCandidates = orderbook.asks.slice(0, cfg.wallDetectDepth);
    const avgBidSize = bidWallCandidates.length > 0 ? bidWallCandidates.reduce((sum, [_, vol]) => sum + vol, 0) / bidWallCandidates.length : 0;
    const avgAskSize = askWallCandidates.length > 0 ? askWallCandidates.reduce((sum, [_, vol]) => sum + vol, 0) / askWallCandidates.length : 0;
    const bidWallThreshold = avgBidSize * cfg.wallSizeThresholdMultiplier;
    const askWallThreshold = avgAskSize * cfg.wallSizeThresholdMultiplier;

    const bidWalls = bidWallThreshold > 0 ? bidWallCandidates.filter(([_, vol]) => vol >= bidWallThreshold).map(([price, size]) => ({ price, size })) : [];
    const askWalls = askWallThreshold > 0 ? askWallCandidates.filter(([_, vol]) => vol >= askWallThreshold).map(([price, size]) => ({ price, size })) : [];

    // --- Results ---
    const analysis = {
        bestBid: bestBidPrice,
        bestAsk: bestAskPrice,
        midPrice: midPrice,
        spread: spread,
        spreadPercent: spreadPercent,
        isSpreadTooWide: spreadPercent > cfg.maxSpreadPercent,
        simpleImbalance: simpleImbalance,
        weightedImbalance: weightedImbalance,
        bidWalls: bidWalls,
        askWalls: askWalls,
        timestamp: orderbook.timestamp || Date.now(),
    };

    // Conditional Debug Logging
    if (config.logLevel === 'debug') {
        const priceFmt = market.precision?.price ?? 2; // Use ?? for nullish coalescing
        const amountFmt = market.precision?.amount ?? 4;
        logger.debug(`OB Analysis: Spread=${spread.toFixed(priceFmt)} (${spreadPercent.toFixed(3)}%), SImb=${simpleImbalance.toFixed(3)}, WImb=${weightedImbalance.toFixed(3)}, BidWalls=${bidWalls.length}, AskWalls=${askWalls.length}`);
        if (analysis.isSpreadTooWide) {
            logger.debug(`Spread (${spreadPercent.toFixed(3)}%) exceeds threshold (${cfg.maxSpreadPercent}%)`);
        }
        if (bidWalls.length > 0) logger.debug(`Bid Walls Found: ${bidWalls.map(w => `${w.size.toFixed(amountFmt)}@${w.price.toFixed(priceFmt)}`).join(', ')}`);
        if (askWalls.length > 0) logger.debug(`Ask Walls Found: ${askWalls.map(w => `${w.size.toFixed(amountFmt)}@${w.price.toFixed(priceFmt)}`).join(', ')}`);
    }

    return analysis;
}


// --- Main Trading Bot Logic ---
async function tradingBot() {
    logger.info('Starting trading bot with enhanced order book analysis...');
    logger.info('Configuration:', config);

    // --- Initialization ---
    const apiKey = process.env.BYBIT_API_KEY;
    const secret = process.env.BYBIT_SECRET;
    if (!apiKey || !secret) { logger.error('API key and secret required in .env file'); return; }

    let exchange;
    try {
        exchange = new ccxt[config.exchange]({
             apiKey,
             secret,
             enableRateLimit: true,
             options: {
                 defaultType: 'swap',
                 adjustForTimeDifference: true,
                 recvWindow: 10000
            }
        });
        await exchange.loadMarkets();
        logger.info(`Loaded markets from ${config.exchange}.`);
         if (exchange.has['setLeverage']) {
             try {
                const markets = await exchange.fetchMarkets(); // Fetch fresh markets info
                if (markets.some(m => m.symbol === config.symbol)) {
                    await exchange.setLeverage(config.leverage, config.symbol);
                    logger.info(`Leverage set to ${config.leverage}x for ${config.symbol}`);
                } else {
                     logger.warn(`Symbol ${config.symbol} not found in fetched markets, cannot set leverage.`);
                }
             } catch (e) { logger.warn(`Could not set leverage (maybe already set or market type mismatch): ${e.message}`); }
        } else { logger.warn(`Exchange ${config.exchange} does not support setting leverage via setLeverage().`); }
    } catch (e) { logger.error(`Exchange initialization failed: ${e.message}`); return; }

    const market = exchange.market(config.symbol);
    if (!market) { logger.error(`Symbol ${config.symbol} not found on ${config.exchange}.`); return; }
    if (!market.contract) { logger.warn(`${config.symbol} might not be a contract/derivative market based on CCXT info.`); }

    const amountPrecision = market.precision?.amount;
    const pricePrecision = market.precision?.price;
    const tickSize = market.precision?.price ? Math.pow(10, -market.precision.price) : undefined;

    if (amountPrecision === undefined || pricePrecision === undefined) {
        logger.warn(`Could not determine precision for ${config.symbol}. Order sizing/pricing might be inaccurate.`);
    }
     if (tickSize === undefined && config.useLimitOrders && config.limitOrderPriceOffsetTicks > 0) {
        logger.warn(`Cannot determine tick size for ${config.symbol}, disabling limit order price offset. Limit orders will be placed at BBO.`);
        config.limitOrderPriceOffsetTicks = 0;
    }


    // --- Data Buffers & State ---
    let timestamps = [], opens = [], highs = [], lows = [], closes = [], volumes = [];
    let position = null;
    let isFetchingData = false;
    let isProcessingSignal = false;
    let lastOrderBookAnalysis = null;

    // --- Main Loop ---
    const runCycle = async () => {
        if (isFetchingData || isProcessingSignal) return;

        isFetchingData = true;
        let currentPrice = closes.length > 0 ? closes[closes.length - 1] : null;

        try {
            const startTime = performance.now();
            // logger.debug('Watching OHLCV and OrderBook...'); // Can be noisy

            const [ohlcvResult, orderbookResult] = await Promise.allSettled([
                exchange.watchOHLCV(config.symbol, config.timeframe, undefined, 1),
                exchange.watchOrderBook(config.symbol, config.orderBookDepth)
            ]);

            const endTime = performance.now();
            // logger.debug(`Data fetch took: ${(endTime - startTime).toFixed(2)} ms`); // Can be noisy

            // --- Process OHLCV ---
            let newCandleReceived = false;
            if (ohlcvResult.status === 'fulfilled' && ohlcvResult.value.length > 0) {
                const latestCandle = ohlcvResult.value[ohlcvResult.value.length - 1];
                 if (timestamps.length === 0 || latestCandle[0] > timestamps[timestamps.length - 1]) {
                    timestamps.push(latestCandle[0]); opens.push(latestCandle[1]); highs.push(latestCandle[2]); lows.push(latestCandle[3]); closes.push(latestCandle[4]); volumes.push(latestCandle[5]);
                    newCandleReceived = true;
                    currentPrice = latestCandle[4];
                    logger.debug(`New candle: ${new Date(latestCandle[0]).toISOString()} C:${latestCandle[4]}`);
                 } else if (latestCandle[0] === timestamps[timestamps.length - 1]) {
                     const lastIndex = timestamps.length - 1;
                     highs[lastIndex] = Math.max(highs[lastIndex], latestCandle[2]);
                     lows[lastIndex] = Math.min(lows[lastIndex], latestCandle[3]);
                     closes[lastIndex] = latestCandle[4];
                     volumes[lastIndex] = latestCandle[5];
                     currentPrice = latestCandle[4];
                     // logger.debug(`Candle update: ${new Date(latestCandle[0]).toISOString()} C:${latestCandle[4]}`);
                 }
            } else if (ohlcvResult.status === 'rejected') {
                logger.warn('Failed to fetch OHLCV:', ohlcvResult.reason?.message || ohlcvResult.reason);
            }

            // --- Process Order Book ---
            if (orderbookResult.status === 'fulfilled') {
                lastOrderBookAnalysis = analyzeOrderBook(orderbookResult.value, market, config);
                if (lastOrderBookAnalysis && currentPrice === null) {
                     currentPrice = lastOrderBookAnalysis.midPrice;
                     logger.debug(`Using order book mid-price (${currentPrice}) as current price.`);
                }
            } else if (orderbookResult.status === 'rejected') {
                logger.warn('Failed to fetch OrderBook:', orderbookResult.reason?.message || orderbookResult.reason);
                lastOrderBookAnalysis = null;
            }

            isFetchingData = false;

            // --- Buffer Management ---
            while (closes.length > config.maxBufferSize) {
                 timestamps.shift(); opens.shift(); highs.shift(); lows.shift(); closes.shift(); volumes.shift();
            }

             // --- Signal Processing ---
             const minCandlesNeeded = Math.max(config.slowMaPeriod, config.atrPeriod + config.atrSmoothPeriod) + 2; // Need buffer
             if (closes.length >= minCandlesNeeded && !isProcessingSignal && currentPrice !== null) {
                 isProcessingSignal = true;
                 const processStartTime = performance.now();

                try {
                    // --- Calculate Indicators ---
                    const fastMA = calculateEhlersSuperSmoother(closes, config.fastMaPeriod);
                    const slowMA = calculateEhlersSuperSmoother(closes, config.slowMaPeriod);
                    const tr = calculateTR(highs, lows, closes);
                    const validTR = tr.filter(v => typeof v === 'number');
                    const atr = calculateEhlersSuperSmoother(validTR, config.atrPeriod);
                    const validAtr = atr.filter(v => typeof v === 'number');
                    const smoothedAtr = calculateEhlersSuperSmoother(validAtr, config.atrSmoothPeriod);

                    // Get latest values with checks
                    const lastFastMA = fastMA.length > 0 ? fastMA[fastMA.length - 1] : null;
                    const lastSlowMA = slowMA.length > 0 ? slowMA[slowMA.length - 1] : null;
                    // Corrected typo here
                    const prevFastMA = fastMA.length > 1 ? fastMA[fastMA.length - 2] : null;
                    const prevSlowMA = slowMA.length > 1 ? slowMA[slowMA.length - 2] : null;
                    const lastAtr = atr.length > 0 ? atr[atr.length - 1] : null;
                    const lastSmoothedAtr = smoothedAtr.length > 0 ? smoothedAtr[smoothedAtr.length - 1] : null;

                    const indicatorsValid = [lastFastMA, lastSlowMA, prevFastMA, prevSlowMA, lastAtr, lastSmoothedAtr].every(v => typeof v === 'number');

                    if (!indicatorsValid) {
                         logger.debug('Indicator calculation resulted in null/invalid values, skipping signal check.');
                         isProcessingSignal = false;
                         return;
                    }

                    const priceFmt = pricePrecision ?? 2;
                    logger.debug(`Indicators: Px=${currentPrice.toFixed(priceFmt)}, FastMA=${lastFastMA.toFixed(priceFmt)}, SlowMA=${lastSlowMA.toFixed(priceFmt)}, ATR=${lastAtr?.toFixed(4)}, SmATR=${lastSmoothedAtr?.toFixed(4)}`);

                    // --- Position Management & Trading Logic ---

                    // 1. Update Trailing Stop
                     if (position) {
                        if (position.side === 'long') {
                            position.highestPrice = Math.max(position.highestPrice ?? position.entryPrice, currentPrice);
                            const potentialStop = position.highestPrice * (1 - config.trailingStopPercent / 100);
                            position.stopPrice = Math.max(position.stopPrice ?? position.entryPrice * (1 - config.trailingStopPercent / 100), potentialStop);
                        } else if (position.side === 'short') {
                            position.lowestPrice = Math.min(position.lowestPrice ?? position.entryPrice, currentPrice);
                            const potentialStop = position.lowestPrice * (1 + config.trailingStopPercent / 100);
                            position.stopPrice = Math.min(position.stopPrice ?? position.entryPrice * (1 + config.trailingStopPercent / 100), potentialStop);
                        }
                         // logger.debug(`${position.side} Pos Update: TrailStop=${position.stopPrice?.toFixed(priceFmt)}`); // Can be noisy
                    }

                     // 2. Check Exit Conditions
                     let exitSignal = null;
                     if (position) {
                        const stopPriceHit = (position.side === 'long' && position.stopPrice && currentPrice <= position.stopPrice) ||
                                            (position.side === 'short' && position.stopPrice && currentPrice >= position.stopPrice);
                        const maCrossExit = (position.side === 'long' && lastFastMA < lastSlowMA && prevFastMA >= prevSlowMA) ||
                                            (position.side === 'short' && lastFastMA > lastSlowMA && prevFastMA <= prevSlowMA);

                         if (stopPriceHit) exitSignal = 'Trailing Stop Hit';
                         else if (maCrossExit) exitSignal = 'MA Crossover Exit';

                         if (exitSignal) {
                             logger.info(`Exit Signal (${position.side}): ${exitSignal}. Price: ${currentPrice.toFixed(priceFmt)}, Stop: ${position.stopPrice?.toFixed(priceFmt)}`);
                             try {
                                 const closeSide = position.side === 'long' ? 'sell' : 'buy';
                                 const order = await exchange.createOrder(config.symbol, 'market', closeSide, position.amount, undefined, { 'reduceOnly': true });
                                 logger.info(`Position closed via Market Order (${position.side} ${position.amount} ${market.base}). Order ID: ${order.id}`);
                                 position = null;
                             } catch (e) {
                                 logger.error(`Error closing ${position.side} position:`, e);
                                 // Position remains, will try again next cycle
                            }
                         }
                     }

                     // 3. Check Entry Conditions
                     if (!position && !exitSignal) {
                         const isBullishCross = prevFastMA <= prevSlowMA && lastFastMA > lastSlowMA;
                         const isBearishCross = prevFastMA >= prevSlowMA && lastFastMA < lastSlowMA;
                         const isVolatile = lastAtr > lastSmoothedAtr;

                         let entrySignalReason = null;
                         let entrySide = null;
                         let obFactorsAllowEntry = false;
                         let preferredOrderType = 'market';
                         let limitPrice = null;

                         if (lastOrderBookAnalysis) {
                            const ob = lastOrderBookAnalysis;
                             if (ob.isSpreadTooWide && !config.useLimitOrders) {
                                 logger.info(`Skipping entry: Spread (${ob.spreadPercent.toFixed(3)}%) too wide for market order.`);
                             } else {
                                 if (isBullishCross && isVolatile) {
                                     const obConfirmsBuy = ob.simpleImbalance > config.imbalanceThreshold || ob.weightedImbalance > 0.05;
                                     const noImmediateAskWall = !(ob.askWalls.length > 0 && ob.askWalls[0].price <= ob.bestAsk + (tickSize ?? 0) * 3);
                                     if (obConfirmsBuy && noImmediateAskWall) {
                                         entrySignalReason = 'Long Entry: MA Cross + Volatility + OB Confirm';
                                         entrySide = 'buy';
                                         obFactorsAllowEntry = true;
                                         if (config.useLimitOrders && !ob.isSpreadTooWide && tickSize !== undefined) {
                                             preferredOrderType = 'limit';
                                             limitPrice = exchange.priceToPrecision(config.symbol, ob.bestBid + tickSize * config.limitOrderPriceOffsetTicks);
                                         } else if (config.useLimitOrders && ob.isSpreadTooWide) {
                                              logger.debug("Spread too wide, falling back to market order for long.");
                                              preferredOrderType = 'market';
                                         }
                                     } else { logger.debug(`Skipping Long: MA+Vol OK. OB Confirm=${obConfirmsBuy}, No Ask Wall=${noImmediateAskWall}`); }
                                 } else if (isBearishCross && isVolatile) {
                                     const obConfirmsSell = ob.simpleImbalance < -config.imbalanceThreshold || ob.weightedImbalance < -0.05;
                                     const noImmediateBidWall = !(ob.bidWalls.length > 0 && ob.bidWalls[0].price >= ob.bestBid - (tickSize ?? 0) * 3);
                                     if (obConfirmsSell && noImmediateBidWall) {
                                         entrySignalReason = 'Short Entry: MA Cross + Volatility + OB Confirm';
                                         entrySide = 'sell';
                                         obFactorsAllowEntry = true;
                                         if (config.useLimitOrders && !ob.isSpreadTooWide && tickSize !== undefined) {
                                             preferredOrderType = 'limit';
                                             limitPrice = exchange.priceToPrecision(config.symbol, ob.bestAsk - tickSize * config.limitOrderPriceOffsetTicks);
                                         } else if (config.useLimitOrders && ob.isSpreadTooWide) {
                                             logger.debug("Spread too wide, falling back to market order for short.");
                                             preferredOrderType = 'market';
                                         }
                                     } else { logger.debug(`Skipping Short: MA+Vol OK. OB Confirm=${obConfirmsSell}, No Bid Wall=${noImmediateBidWall}`); }
                                 }
                             }
                         } else {
                             logger.debug("Skipping entry check: No recent order book analysis available.");
                         }

                         // --- Execute Entry Order ---
                         if (entrySignalReason && entrySide && obFactorsAllowEntry) {
                             logger.info(`Entry Signal: ${entrySignalReason}. Type: ${preferredOrderType}${limitPrice ? ` @ ${limitPrice}` : ''}`);
                             try {
                                 const amountInBase = exchange.amountToPrecision(config.symbol, config.tradeAmountQuote / currentPrice);
                                 logger.info(`Attempting to ${entrySide} ${amountInBase} ${market.base} (${config.tradeAmountQuote} ${market.quote})`);

                                 if (market.limits?.amount?.min && parseFloat(amountInBase) < market.limits.amount.min) {
                                     logger.error(`Order amount ${amountInBase} is below market minimum ${market.limits.amount.min}. Skipping order.`);
                                     throw new Error("Order amount too small");
                                 }

                                 let order = null;
                                 const orderParams = {};

                                 if (preferredOrderType === 'limit' && limitPrice) {
                                     order = await exchange.createOrder(config.symbol, 'limit', entrySide, amountInBase, limitPrice, orderParams);
                                     logger.info(`Limit order placed: ${order.side} ${order.amount} @ ${order.price}. ID: ${order.id}`);
                                 } else {
                                     order = await exchange.createOrder(config.symbol, 'market', entrySide, amountInBase, undefined, orderParams);
                                     logger.info(`Market order placed: ${order.side} ${order.amount}. Avg Price: ${order.average || 'N/A'}, ID: ${order.id}`);
                                 }

                                 // --- Update Position State (Simplified) ---
                                 const entryPrice = (preferredOrderType === 'limit' ? order.price : order.average) || currentPrice;
                                 const filledAmount = order.filled || order.amount;

                                 if (filledAmount > 0) {
                                     position = {
                                         side: entrySide === 'buy' ? 'long' : 'short',
                                         entryPrice: entryPrice,
                                         amount: filledAmount,
                                         highestPrice: entryPrice,
                                         lowestPrice: entryPrice,
                                         stopPrice: null
                                     };
                                     position.stopPrice = position.side === 'long'
                                        ? position.entryPrice * (1 - config.trailingStopPercent / 100)
                                        : position.entryPrice * (1 + config.trailingStopPercent / 100);
                                     logger.info(`Position opened: ${position.side}, Entry: ${position.entryPrice.toFixed(priceFmt)}, Amount: ${position.amount}, Initial Stop: ${position.stopPrice?.toFixed(priceFmt)}`);
                                 } else {
                                     logger.warn(`Order placed (ID: ${order.id}, Type: ${preferredOrderType}) but filled amount is 0 or unavailable. Position state not updated.`);
                                 }
                             } catch (e) {
                                 logger.error(`Error placing ${entrySide} ${preferredOrderType} order: ${e.message}`, e);
                                 if (e instanceof ccxt.InsufficientFunds) logger.error("Insufficient funds.");
                                 else if (e instanceof ccxt.InvalidOrder) logger.error(`Invalid order parameters: ${e.message}.`);
                             }
                         }
                     }

                } catch (processingError) {
                    logger.error('Error during signal processing:', processingError);
                } finally {
                     const processEndTime = performance.now();
                     // logger.debug(`Signal processing took: ${(processEndTime - processStartTime).toFixed(2)} ms`); // Can be noisy
                     isProcessingSignal = false;
                }
            } else if (closes.length < minCandlesNeeded) {
                // logger.info(`Waiting for more data... Have ${closes.length}/${minCandlesNeeded} candles.`); // Can be noisy
            } else if (currentPrice === null){
                 logger.debug("Waiting for current price data...");
            }

        } catch (e) {
            logger.error('Error in main trading loop:', e);
            isFetchingData = false;
            isProcessingSignal = false;
             if (e instanceof ccxt.NetworkError || e instanceof ccxt.ExchangeNotAvailable || e instanceof ccxt.RequestTimeout) {
                logger.warn(`Network/Exchange issue: ${e.message}. Retrying after delay...`);
                await new Promise(resolve => setTimeout(resolve, 5000));
            } else if (e instanceof ccxt.AuthenticationError) {
                logger.error("Authentication failed! Check API keys. Stopping bot."); return;
            } else if (e instanceof ccxt.RateLimitExceeded) {
                 logger.warn(`Rate limit exceeded: ${e.message}. Waiting longer...`);
                 await new Promise(resolve => setTimeout(resolve, (exchange.rateLimit || 1000) * 2 + config.rateLimitBufferMs)); // Use default rateLimit if undefined
            } else {
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    };

    // --- Initial Data Fetch ---
    try {
        logger.info(`Fetching initial ${config.maxBufferSize} candles for ${config.symbol}...`);
        const initialOhlcv = await exchange.fetchOHLCV(config.symbol, config.timeframe, undefined, config.maxBufferSize);
        initialOhlcv.forEach(c => {
             timestamps.push(c[0]); opens.push(c[1]); highs.push(c[2]); lows.push(c[3]); closes.push(c[4]); volumes.push(c[5]);
        });
        logger.info(`Fetched ${initialOhlcv.length} initial candles. Last candle time: ${new Date(timestamps[timestamps.length - 1]).toISOString()}`);
        currentPrice = closes.length > 0 ? closes[closes.length - 1] : null;
    } catch (e) {
        logger.error(`Failed to fetch initial OHLCV data: ${e.message}. Starting with WebSocket data only.`);
    }

    // --- Start Loop & Graceful Shutdown ---
    logger.info("Starting main execution cycle...");
    const intervalTimeMs = 1000;
    const intervalId = setInterval(runCycle, intervalTimeMs);

    process.on('SIGINT', async () => {
        logger.info("SIGINT received. Initiating graceful shutdown...");
        clearInterval(intervalId);

        if (config.useLimitOrders) {
            try {
                logger.info("Attempting to cancel open limit orders...");
                const openOrders = await exchange.fetchOpenOrders(config.symbol);
                let cancelledCount = 0;
                for (const order of openOrders.filter(o => o.type === 'limit')) { // Filter for limit orders
                    try {
                        await exchange.cancelOrder(order.id, config.symbol);
                        logger.info(`Cancelled open limit order ${order.id}`);
                        cancelledCount++;
                        await new Promise(resolve => setTimeout(resolve, 300)); // Small delay
                    } catch (cancelError) {
                         logger.error(`Failed to cancel order ${order.id}: ${cancelError.message}`);
                    }
                }
                 logger.info(`Cancelled ${cancelledCount} open limit orders.`);
            } catch (e) {
                 logger.error(`Error fetching or cancelling open orders during shutdown: ${e.message}`);
            }
        }

        if (position) {
            logger.warn(`Closing open ${position.side} position via market order before shutdown...`);
            try {
                 const closeSide = position.side === 'long' ? 'sell' : 'buy';
                 await exchange.createOrder(config.symbol, 'market', closeSide, position.amount, undefined, { 'reduceOnly': true });
                 logger.info("Position closed successfully.");
            } catch (e) {
                 logger.error(`EMERGENCY: Failed to close position on shutdown: ${e.message}. Manual intervention may be required!`);
            }
        } else {
            logger.info("No open position to close.");
        }

        logger.info("Shutdown complete.");
        process.exit(0);
    });
}

// --- Run the Bot ---
tradingBot().catch(e => {
    logger.error("Unhandled critical error during bot execution:", e);
    process.exit(1);
});
