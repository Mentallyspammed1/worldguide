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

// --- Indicator Calculations --- (EhlersSuperSmoother, calculateTR - unchanged from previous version)
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
    const Deg2Rad = Math.PI / 180.0;
    const a1 = Math.exp(-Math.sqrt(2) * Math.PI / period);
    const coeff2 = 2 * a1 * Math.cos(Math.sqrt(2) * Math.PI / period);
    const coeff3 = -a1 * a1;
    const coeff1 = 1 - coeff2 - coeff3;

    result[0] = prices[0];
    if (prices.length > 1) {
        result[1] = (coeff1 / 2) * (prices[1] + prices[0]) + coeff2 * (result[0] || prices[0]);
    }

    for (let i = 2; i < prices.length; i++) {
        const priceTerm = coeff1 * (prices[i] + prices[i - 1]) / 2;
        const prev1 = result[i-1] ?? prices[i-1];
        const prev2 = result[i-2] ?? prices[i-2];
        result[i] = priceTerm + coeff2 * prev1 + coeff3 * prev2;
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
    const tr = [Math.max(highs[0] - lows[0], 0)]; // First TR is simply High - Low or 0 if length is 1
    for (let i = 1; i < highs.length; i++) {
        const high = highs[i];
        const low = lows[i];
        const prevClose = closes[i - 1];
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
    if (!orderbook || !orderbook.bids || !orderbook.asks || !orderbook.bids.length || !orderbook.asks.length) {
        logger.debug('Order book data missing or empty for analysis.');
        return null;
    }

    const bestBidPrice = orderbook.bids[0][0];
    const bestAskPrice = orderbook.asks[0][0];
    const midPrice = (bestBidPrice + bestAskPrice) / 2;
    const spread = bestAskPrice - bestBidPrice;
    const spreadPercent = (spread / midPrice) * 100;

    // --- Simple Imbalance (Top N levels) ---
    const topBids = orderbook.bids.slice(0, cfg.imbalanceDepth);
    const topAsks = orderbook.asks.slice(0, cfg.imbalanceDepth);
    const topBidVolume = topBids.reduce((sum, [_, vol]) => sum + vol, 0);
    const topAskVolume = topAsks.reduce((sum, [_, vol]) => sum + vol, 0);
    const simpleImbalance = (topBidVolume + topAskVolume > 0)
        ? (topBidVolume - topAskVolume) / (topBidVolume + topAskVolume)
        : 0;

    // --- Weighted Imbalance (Deeper levels, weighted by distance from mid-price) ---
    let weightedBidVolume = 0;
    let weightedAskVolume = 0;
    let totalBidWeight = 0;
    let totalAskWeight = 0;
    const bidsForWeighted = orderbook.bids.slice(0, cfg.weightedImbalanceDepth);
    const asksForWeighted = orderbook.asks.slice(0, cfg.weightedImbalanceDepth);

    for (const [price, volume] of bidsForWeighted) {
        const distance = midPrice - price;
        const weight = 1 / (1 + distance); // Simple inverse distance weighting
        weightedBidVolume += volume * weight;
        totalBidWeight += weight;
    }
    for (const [price, volume] of asksForWeighted) {
        const distance = price - midPrice;
        const weight = 1 / (1 + distance);
        weightedAskVolume += volume * weight;
        totalAskWeight += weight;
    }
    const weightedImbalance = (weightedBidVolume + weightedAskVolume > 0)
        ? (weightedBidVolume - weightedAskVolume) / (weightedBidVolume + weightedAskVolume) // Normalize? Might be better without for magnitude: (weightedBidVolume / totalBidWeight) - (weightedAskVolume / totalAskWeight)
        : 0;


    // --- Wall Detection (Significant Volume Clusters) ---
    const bidWallCandidates = orderbook.bids.slice(0, cfg.wallDetectDepth);
    const askWallCandidates = orderbook.asks.slice(0, cfg.wallDetectDepth);
    const avgBidSize = bidWallCandidates.reduce((sum, [_, vol]) => sum + vol, 0) / (bidWallCandidates.length || 1);
    const avgAskSize = askWallCandidates.reduce((sum, [_, vol]) => sum + vol, 0) / (askWallCandidates.length || 1);
    const bidWallThreshold = avgBidSize * cfg.wallSizeThresholdMultiplier;
    const askWallThreshold = avgAskSize * cfg.wallSizeThresholdMultiplier;

    const bidWalls = bidWallCandidates.filter(([_, vol]) => vol >= bidWallThreshold).map(([price, size]) => ({ price, size }));
    const askWalls = askWallCandidates.filter(([_, vol]) => vol >= askWallThreshold).map(([price, size]) => ({ price, size }));

    // --- Results ---
    const analysis = {
        bestBid: bestBidPrice,
        bestAsk: bestAskPrice,
        midPrice: midPrice,
        spread: spread,
        spreadPercent: spreadPercent,
        isSpreadTooWide: spreadPercent > cfg.maxSpreadPercent,
        simpleImbalance: simpleImbalance, // Quick check
        weightedImbalance: weightedImbalance, // Deeper view
        bidWalls: bidWalls, // [{ price, size }, ...]
        askWalls: askWalls, // [{ price, size }, ...]
        timestamp: orderbook.timestamp || Date.now(),
    };

    logger.debug(`OB Analysis: Spread=${spread.toFixed(market.precision.price)} (${spreadPercent.toFixed(3)}%), SImb=${simpleImbalance.toFixed(3)}, WImb=${weightedImbalance.toFixed(3)}, BidWalls=${bidWalls.length}, AskWalls=${askWalls.length}`);
    if (analysis.isSpreadTooWide) {
        logger.debug(`Spread (${spreadPercent.toFixed(3)}%) exceeds threshold (${cfg.maxSpreadPercent}%)`);
    }
    if (bidWalls.length > 0) logger.debug(`Bid Walls Found: ${bidWalls.map(w => `${w.size.toFixed(market.precision.amount)}@${w.price.toFixed(market.precision.price)}`).join(', ')}`);
    if (askWalls.length > 0) logger.debug(`Ask Walls Found: ${askWalls.map(w => `${w.size.toFixed(market.precision.amount)}@${w.price.toFixed(market.precision.price)}`).join(', ')}`);


    return analysis;
}


// --- Main Trading Bot Logic ---
async function tradingBot() {
    logger.info('Starting trading bot with enhanced order book analysis...');
    logger.info('Configuration:', config);

    // --- Initialization (API Keys, Exchange, Market, Leverage - unchanged) ---
    const apiKey = process.env.BYBIT_API_KEY;
    const secret = process.env.BYBIT_SECRET;
    if (!apiKey || !secret) { logger.error('API key and secret required'); return; }

    let exchange;
    try {
        exchange = new ccxt[config.exchange]({ apiKey, secret, enableRateLimit: true, options: { defaultType: 'swap', adjustForTimeDifference: true, recvWindow: 10000 } });
        await exchange.loadMarkets();
        logger.info(`Loaded markets from ${config.exchange}.`);
         if (exchange.has['setLeverage']) {
             try {
                await exchange.setLeverage(config.leverage, config.symbol);
                logger.info(`Leverage set to ${config.leverage}x for ${config.symbol}`);
             } catch (e) { logger.warn(`Could not set leverage: ${e.message}`); }
        } else { logger.warn(`Exchange does not support setting leverage via setLeverage().`); }
    } catch (e) { logger.error(`Exchange init failed: ${e.message}`); return; }

    const market = exchange.market(config.symbol);
    if (!market) { logger.error(`Symbol ${config.symbol} not found.`); return; }
    if (!market.contract) { logger.error(`${config.symbol} is not a contract market.`); /* return; */ } // Allow non-contracts if needed

    const amountPrecision = market.precision?.amount;
    const pricePrecision = market.precision?.price;
    const tickSize = market.precision?.price ? (1 / Math.pow(10, market.precision.price)) : undefined; // Calculate tick size

    if (amountPrecision === undefined || pricePrecision === undefined) {
        logger.warn(`Could not determine precision for ${config.symbol}.`);
    }
     if (tickSize === undefined && config.useLimitOrders && config.limitOrderPriceOffsetTicks > 0) {
        logger.warn(`Cannot determine tick size for ${config.symbol}, disabling limit order price offset.`);
        config.limitOrderPriceOffsetTicks = 0; // Fallback to BBO
    }


    // --- Data Buffers & State (unchanged) ---
    let timestamps = [], opens = [], highs = [], lows = [], closes = [], volumes = [];
    let position = null; // { side, entryPrice, amount, highestPrice?, lowestPrice?, stopPrice? }
    let isFetchingData = false;
    let isProcessingSignal = false;
    let lastOrderBookAnalysis = null; // Store the latest analysis

    // --- Main Loop ---
    const runCycle = async () => {
        if (isFetchingData || isProcessingSignal) return;

        isFetchingData = true;
        let currentPrice = closes.length > 0 ? closes[closes.length - 1] : null; // Use last close as current price initially

        try {
            const startTime = performance.now();
            logger.debug('Watching OHLCV and OrderBook...');

            // Fetch concurrently
            const [ohlcvResult, orderbookResult] = await Promise.allSettled([
                exchange.watchOHLCV(config.symbol, config.timeframe, undefined, 1),
                exchange.watchOrderBook(config.symbol, config.orderBookDepth) // Fetch sufficient depth
            ]);

            const endTime = performance.now();
            logger.debug(`Data fetch took: ${(endTime - startTime).toFixed(2)} ms`);

            // --- Process OHLCV (unchanged) ---
            let newCandleReceived = false;
            if (ohlcvResult.status === 'fulfilled' && ohlcvResult.value.length > 0) {
                const latestCandle = ohlcvResult.value[ohlcvResult.value.length - 1];
                 if (timestamps.length === 0 || latestCandle[0] > timestamps[timestamps.length - 1]) {
                    timestamps.push(latestCandle[0]); opens.push(latestCandle[1]); highs.push(latestCandle[2]); lows.push(latestCandle[3]); closes.push(latestCandle[4]); volumes.push(latestCandle[5]);
                    newCandleReceived = true;
                    currentPrice = latestCandle[4]; // Update current price
                    logger.debug(`New candle: ${new Date(latestCandle[0]).toISOString()} C:${latestCandle[4]}`);
                 } else if (latestCandle[0] === timestamps[timestamps.length - 1]) {
                     highs[highs.length - 1] = Math.max(highs[highs.length - 1], latestCandle[2]);
                     lows[lows.length - 1] = Math.min(lows[lows.length - 1], latestCandle[3]);
                     closes[closes.length - 1] = latestCandle[4];
                     volumes[volumes.length - 1] = latestCandle[5];
                     currentPrice = latestCandle[4]; // Update current price
                     logger.debug(`Candle update: ${new Date(latestCandle[0]).toISOString()} C:${latestCandle[4]}`);
                 }
            } else if (ohlcvResult.status === 'rejected') {
                logger.warn('Failed to fetch OHLCV:', ohlcvResult.reason?.message || ohlcvResult.reason);
            }

            // --- Process Order Book ---
            if (orderbookResult.status === 'fulfilled') {
                // Perform the detailed analysis
                lastOrderBookAnalysis = analyzeOrderBook(orderbookResult.value, market, config);
                if (lastOrderBookAnalysis && !currentPrice) {
                     currentPrice = lastOrderBookAnalysis.midPrice; // Use mid-price if candle data is lagging
                }
            } else if (orderbookResult.status === 'rejected') {
                logger.warn('Failed to fetch OrderBook:', orderbookResult.reason?.message || orderbookResult.reason);
                lastOrderBookAnalysis = null; // Invalidate old analysis on fetch failure
            }

            isFetchingData = false;

            // --- Buffer Management (unchanged) ---
            if (closes.length > config.maxBufferSize) {
                 timestamps.shift(); opens.shift(); highs.shift(); lows.shift(); closes.shift(); volumes.shift();
            }

             // --- Signal Processing ---
             const minCandlesNeeded = Math.max(config.slowMaPeriod, config.atrPeriod + config.atrSmoothPeriod);
             if (closes.length >= minCandlesNeeded && !isProcessingSignal && currentPrice !== null) {
                 isProcessingSignal = true;
                 const processStartTime = performance.now();

                try {
                    // --- Calculate Indicators (unchanged) ---
                    const fastMA = calculateEhlersSuperSmoother(closes, config.fastMaPeriod);
                    const slowMA = calculateEhlersSuperSmoother(closes, config.slowMaPeriod);
                    const tr = calculateTR(highs, lows, closes);
                    const atr = calculateEhlersSuperSmoother(tr.filter(v => v !== null), config.atrPeriod); // Filter nulls from TR if any
                    const smoothedAtr = calculateEhlersSuperSmoother(atr.filter(v => v !== null), config.atrSmoothPeriod);

                    // Get latest values
                    const lastFastMA = fastMA[fastMA.length - 1];
                    const lastSlowMA = slowMA[slowMA.length - 1];
                    const prevFastMA = fastMA[fastFastMA.length - 2];
                    const prevSlowMA = slowMA[slowMA.length - 2];
                    const lastAtr = atr[atr.length - 1];
                    const lastSmoothedAtr = smoothedAtr[smoothedAtr.length - 1];

                    if ([lastFastMA, lastSlowMA, prevFastMA, prevSlowMA, lastAtr, lastSmoothedAtr].some(v => v === null)) {
                         logger.debug('Indicator calculation resulted in null values, skipping signal check.');
                         isProcessingSignal = false;
                         return;
                    }

                    logger.debug(`Indicators: Price=${currentPrice.toFixed(pricePrecision)}, FastMA=${lastFastMA.toFixed(pricePrecision)}, SlowMA=${lastSlowMA.toFixed(pricePrecision)}, ATR=${lastAtr?.toFixed(4)}, SmoothedATR=${lastSmoothedAtr?.toFixed(4)}`);

                    // --- Position Management & Trading Logic ---

                    // 1. Update Trailing Stop / Position High/Low (unchanged)
                     if (position) {
                        if (position.side === 'long') {
                            position.highestPrice = Math.max(position.highestPrice ?? position.entryPrice, currentPrice);
                            const potentialStop = position.highestPrice * (1 - config.trailingStopPercent / 100);
                            position.stopPrice = Math.max(position.stopPrice ?? -Infinity, potentialStop);
                            logger.debug(`Long Pos Update: High=${position.highestPrice.toFixed(pricePrecision)}, TrailStop=${position.stopPrice.toFixed(pricePrecision)}`);
                        } else if (position.side === 'short') {
                            position.lowestPrice = Math.min(position.lowestPrice ?? position.entryPrice, currentPrice);
                            const potentialStop = position.lowestPrice * (1 + config.trailingStopPercent / 100);
                            position.stopPrice = Math.min(position.stopPrice ?? Infinity, potentialStop);
                            logger.debug(`Short Pos Update: Low=${position.lowestPrice.toFixed(pricePrecision)}, TrailStop=${position.stopPrice.toFixed(pricePrecision)}`);
                        }
                    }

                     // 2. Check Exit Conditions (unchanged - prioritize exit signals)
                     let exitSignal = null;
                     if (position) {
                        if (position.side === 'long' && position.stopPrice && currentPrice <= position.stopPrice) exitSignal = 'Trailing Stop Hit';
                        else if (position.side === 'long' && lastFastMA < lastSlowMA && prevFastMA >= prevSlowMA) exitSignal = 'MA Bearish Cross';
                        else if (position.side === 'short' && position.stopPrice && currentPrice >= position.stopPrice) exitSignal = 'Trailing Stop Hit';
                        else if (position.side === 'short' && lastFastMA > lastSlowMA && prevFastMA <= prevSlowMA) exitSignal = 'MA Bullish Cross';


                         if (exitSignal) {
                             logger.info(`Exit Signal (${position.side}): ${exitSignal}. Price: ${currentPrice.toFixed(pricePrecision)}, Stop: ${position.stopPrice?.toFixed(pricePrecision)}`);
                             try {
                                 const closeSide = position.side === 'long' ? 'sell' : 'buy';
                                 const order = await exchange.createOrder(config.symbol, 'market', closeSide, position.amount, undefined, { 'reduceOnly': true });
                                 logger.info(`Position closed (${position.side} ${position.amount} ${market.base}). Order ID: ${order.id}`);
                                 position = null;
                             } catch (e) { logger.error(`Error closing ${position.side} position:`, e); }
                         }
                     }

                     // 3. Check Entry Conditions (only if not in a position and no exit signal)
                     if (!position && !exitSignal) {
                         const isBullishCross = prevFastMA <= prevSlowMA && lastFastMA > lastSlowMA;
                         const isBearishCross = prevFastMA >= prevSlowMA && lastFastMA < lastSlowMA;
                         const isVolatile = lastAtr > lastSmoothedAtr; // Volatility expansion

                         let entrySignalReason = null;
                         let entrySide = null;

                         // --- Apply Order Book Analysis to Entry ---
                         let obFactorsAllowEntry = false;
                         let preferredOrderType = 'market'; // Default
                         let limitPrice = null;

                         if (lastOrderBookAnalysis) {
                            const ob = lastOrderBookAnalysis;
                             if (ob.isSpreadTooWide && !config.useLimitOrders) {
                                 logger.info(`Skipping potential entry: Spread (${ob.spreadPercent.toFixed(3)}%) too wide for market order.`);
                             } else {
                                 // Base technical signal
                                 if (isBullishCross && isVolatile) {
                                     // Check order book confirmation
                                     if (ob.simpleImbalance > config.imbalanceThreshold || ob.weightedImbalance > 0) { // Require some buy pressure
                                         // Check for immediate large ask walls
                                         const immediateAskWall = ob.askWalls.length > 0 && ob.askWalls[0].price <= ob.bestAsk + (tickSize || 0) * 5; // Wall very close
                                         if (!immediateAskWall) {
                                             entrySignalReason = 'Long Entry: MA Cross + Volatility + OB Confirm';
                                             entrySide = 'buy';
                                             obFactorsAllowEntry = true;
                                             if (config.useLimitOrders && !ob.isSpreadTooWide && tickSize) {
                                                 preferredOrderType = 'limit';
                                                 limitPrice = exchange.priceToPrecision(config.symbol, ob.bestBid + tickSize * config.limitOrderPriceOffsetTicks);
                                             } else if (config.useLimitOrders && ob.isSpreadTooWide) {
                                                 logger.debug("Spread too wide, falling back to market order despite useLimitOrders=true");
                                                 preferredOrderType = 'market'; // Fallback
                                             }
                                         } else { logger.info(`Skipping Long: MA Cross + Volatility ok, but immediate ask wall detected at ${ob.askWalls[0].price}`); }
                                     } else { logger.debug('Skipping Long: MA Cross + Volatility ok, but OB imbalance not supportive.'); }

                                 } else if (isBearishCross && isVolatile) {
                                     // Check order book confirmation
                                     if (ob.simpleImbalance < -config.imbalanceThreshold || ob.weightedImbalance < 0) { // Require some sell pressure
                                          // Check for immediate large bid walls
                                         const immediateBidWall = ob.bidWalls.length > 0 && ob.bidWalls[0].price >= ob.bestBid - (tickSize || 0) * 5; // Wall very close
                                         if (!immediateBidWall) {
                                             entrySignalReason = 'Short Entry: MA Cross + Volatility + OB Confirm';
                                             entrySide = 'sell';
                                             obFactorsAllowEntry = true;
                                              if (config.useLimitOrders && !ob.isSpreadTooWide && tickSize) {
                                                 preferredOrderType = 'limit';
                                                 limitPrice = exchange.priceToPrecision(config.symbol, ob.bestAsk - tickSize * config.limitOrderPriceOffsetTicks);
                                             } else if (config.useLimitOrders && ob.isSpreadTooWide) {
                                                 logger.debug("Spread too wide, falling back to market order despite useLimitOrders=true");
                                                 preferredOrderType = 'market'; // Fallback
                                             }
                                         } else { logger.info(`Skipping Short: MA Cross + Volatility ok, but immediate bid wall detected at ${ob.bidWalls[0].price}`); }
                                     } else { logger.debug('Skipping Short: MA Cross + Volatility ok, but OB imbalance not supportive.'); }
                                 }
                             }
                         } else {
                             logger.debug("Skipping entry check: No recent order book analysis available.");
                         }


                         // --- Execute Entry Order ---
                         if (entrySignalReason && entrySide && obFactorsAllowEntry) {
                             logger.info(`Entry Signal: ${entrySignalReason}. Preferred type: ${preferredOrderType}${limitPrice ? ` @ ${limitPrice}` : ''}`);
                             try {
                                 // Calculate amount (unchanged)
                                 const amountInBase = exchange.amountToPrecision(config.symbol, config.tradeAmountQuote / currentPrice);
                                 logger.info(`Attempting to ${entrySide} ${amountInBase} ${market.base} (${config.tradeAmountQuote} ${market.quote})`);

                                 let order = null;
                                 if (preferredOrderType === 'limit' && limitPrice) {
                                     order = await exchange.createOrder(config.symbol, 'limit', entrySide, amountInBase, limitPrice);
                                     logger.info(`Limit order placed: ${order.side} ${order.amount} @ ${order.price}. ID: ${order.id}`);
                                     // Note: Limit orders might not fill immediately.
                                     // A more complex bot would track open orders and potentially cancel/replace them.
                                     // For simplicity, we assume it fills or the logic handles it next cycle.
                                 } else {
                                     // Market Order (if !useLimitOrders, spread too wide, or other fallback)
                                     order = await exchange.createOrder(config.symbol, 'market', entrySide, amountInBase);
                                     logger.info(`Market order placed: ${order.side} ${order.amount}. Avg Price: ${order.average || 'N/A'}, ID: ${order.id}`);
                                 }


                                 // --- Update Position State ---
                                 // Use order details if possible, otherwise estimate
                                 // CRITICAL: For market orders, `order.average` is the best price. For limit orders that *might* have filled,
                                 // you'd ideally fetch execution details. Here, we'll use the limit price for limit orders
                                 // and average/current for market, acknowledging this simplification.
                                 const entryPrice = (preferredOrderType === 'limit' ? order.price : order.average) || currentPrice;
                                 const filledAmount = order.filled || order.amount; // Use filled amount if available

                                 // Only create position state IF the order seems valid (e.g., amount > 0)
                                 // A proper implementation would wait for fill confirmation via websockets/polling.
                                 if (filledAmount > 0) {
                                     position = {
                                         side: entrySide === 'buy' ? 'long' : 'short',
                                         entryPrice: entryPrice,
                                         amount: filledAmount,
                                         highestPrice: entryPrice,
                                         lowestPrice: entryPrice,
                                         stopPrice: null // Initialize stop price (will be set by trailing logic)
                                     };
                                     // Calculate initial stop based on entry (can be refined)
                                      position.stopPrice = position.side === 'long'
                                        ? entryPrice * (1 - config.trailingStopPercent / 100)
                                        : entryPrice * (1 + config.trailingStopPercent / 100);

                                     logger.info(`Position opened: ${position.side}, Entry: ${position.entryPrice.toFixed(pricePrecision)}, Amount: ${position.amount}, Initial Stop: ${position.stopPrice?.toFixed(pricePrecision)}`);
                                 } else {
                                    logger.warn(`Order placed (ID: ${order.id}) but filled amount is 0 or unavailable. Position state not updated.`);
                                     // This might happen if a limit order was placed but not yet filled.
                                     // The bot will likely re-evaluate on the next tick.
                                 }


                             } catch (e) {
                                 logger.error(`Error placing ${entrySide} ${preferredOrderType} order:`, e);
                                 if (e instanceof ccxt.InsufficientFunds) {
                                    logger.error("Insufficient funds.");
                                 }
                                 // Handle other errors (e.g., invalid order size)
                             }
                         }
                     }

                } catch (processingError) {
                    logger.error('Error during signal processing:', processingError);
                } finally {
                     const processEndTime = performance.now();
                     logger.debug(`Signal processing took: ${(processEndTime - processStartTime).toFixed(2)} ms`);
                     isProcessingSignal = false;
                }
            } else if (closes.length < minCandlesNeeded) {
                logger.info(`Waiting for more data... Have ${closes.length}/${minCandlesNeeded} candles.`);
            } else if (isProcessingSignal) {
                logger.debug("Signal processing already in progress, skipping cycle.");
            } else if (currentPrice === null){
                 logger.debug("Waiting for price data...");
            }

        } catch (e) {
            logger.error('Error in main trading loop:', e);
            isFetchingData = false; // Reset flags on error
            isProcessingSignal = false;
             if (e instanceof ccxt.NetworkError || e instanceof ccxt.ExchangeNotAvailable || e instanceof ccxt.RequestTimeout) {
                logger.warn(`Network/Exchange issue: ${e.message}. Retrying after delay...`);
                await new Promise(resolve => setTimeout(resolve, 5000));
            } else if (e instanceof ccxt.AuthenticationError) {
                logger.error("Authentication failed! Check API keys."); return; // Stop
            } else {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
    };

    // --- Initial Data Fetch (unchanged) ---
    try {
        logger.info(`Fetching initial ${config.maxBufferSize} candles...`);
        const initialOhlcv = await exchange.fetchOHLCV(config.symbol, config.timeframe, undefined, config.maxBufferSize);
        initialOhlcv.forEach(c => { timestamps.push(c[0]); opens.push(c[1]); highs.push(c[2]); lows.push(c[3]); closes.push(c[4]); volumes.push(c[5]); });
        logger.info(`Fetched ${initialOhlcv.length} initial candles.`);
        currentPrice = closes.length > 0 ? closes[closes.length - 1] : null;
    } catch (e) { logger.error(`Failed to fetch initial OHLCV: ${e.message}.`); }

    // --- Start Loop & Shutdown (unchanged) ---
    const intervalId = setInterval(runCycle, 1000); // Check frequently

    process.on('SIGINT', async () => {
        logger.info("SIGINT received. Shutting down...");
        clearInterval(intervalId);
        if (position) {
            logger.warn(`Closing open ${position.side} position before shutdown...`);
            try {
                 const closeSide = position.side === 'long' ? 'sell' : 'buy';
                 await exchange.createOrder(config.symbol, 'market', closeSide, position.amount, undefined, { 'reduceOnly': true });
                 logger.info("Position closed.");
            } catch (e) { logger.error(`Failed to close position on shutdown: ${e.message}`); }
        }
        // Add logic here to cancel any open limit orders if necessary
        logger.info("Shutdown complete.");
        process.exit(0);
    });
}

// --- Run the Bot ---
tradingBot().catch(e => {
    logger.error("Unhandled critical error:", e);
    process.exit(1);
});
