// src/services/strategyService.js
const { calculateIndicators } = require('../utils/indicators');
const {
    fetchOHLCV,
    fetchBalance,
    setLeverage,
    fetchPosition,
    createMarketOrder,
    createOrder, // For potential future limit-based SL/TP orders
    getMarketDetails,
    getBybit, // Needed to check initialization status
} = require('./bybitService');
require('dotenv').config();

// --- State Variables ---
let tradingIntervalId = null; // Holds the ID for the setInterval loop
let isTradingEnabled = false; // Flag indicating if the main trading loop is active
let currentConfig = {}; // Holds the active trading configuration
let tradeLogs = []; // Simple in-memory log storage
let lastExecutionTime = 0; // Timestamp of the last strategy run start
let isProcessingTrade = false; // Flag to prevent concurrent strategy runs

// --- Constants ---
const MAX_LOGS = 250; // Maximum number of log entries to keep in memory
const MIN_INTERVAL_MS = 5000; // Minimum time (ms) between strategy executions
const DEFAULT_OHLCV_LIMIT = 300; // How many candles to fetch

// --- Initialization ---
const initializeStrategyConfig = () => {
    currentConfig = {
        symbol: process.env.DEFAULT_SYMBOL || "BTC/USDT:USDT",
        interval: process.env.DEFAULT_INTERVAL || "5m",
        leverage: parseInt(process.env.DEFAULT_LEVERAGE || "10", 10),
        riskPerTrade: parseFloat(process.env.RISK_PER_TRADE || "0.005"),
        atrPeriod: parseInt(process.env.ATR_PERIOD || "14", 10),
        atrSlMult: parseFloat(process.env.ATR_SL_MULT || "1.5"),
        atrTpMult: parseFloat(process.env.ATR_TP_MULT || "1.5"),
        indicatorPeriod: parseInt(process.env.INDICATOR_PERIOD || "14", 10),
        ehlersMaPeriod: parseInt(process.env.EHLERS_MA_PERIOD || "10", 10),
        stochRsiK: parseInt(process.env.STOCH_RSI_K || "3", 10),
        stochRsiD: parseInt(process.env.STOCH_RSI_D || "3", 10),
        stochRsiLength: parseInt(process.env.STOCH_RSI_LENGTH || "14", 10),
        stochRsiStochLength: parseInt(process.env.STOCH_RSI_STOCH_LENGTH || "14", 10),
        strategy: process.env.STRATEGY_NAME || "STOCH_RSI_EHLERS_MA",
    };
    logTrade("Strategy configuration initialized from environment variables.", "INFO");
};

// Initialize config when the module loads
initializeStrategyConfig();


// --- Logging Utility ---
const logTrade = (message, level = 'INFO') => {
    const timestamp = new Date().toISOString();
    // Pad level for alignment
    const levelStr = `[${level.padEnd(7)}]`; // e.g., [INFO   ]
    const logEntry = `[${timestamp}] ${levelStr} ${message}`;
    console.log(logEntry); // Log to console
    tradeLogs.push(logEntry);
    // Trim logs if exceeding max size
    if (tradeLogs.length > MAX_LOGS) {
        tradeLogs.shift(); // Remove the oldest log entry
    }
};

// --- Formatting Utilities ---
// Formats a price based on market precision rules
const formatPrice = (price, market) => {
    if (typeof price !== 'number' || isNaN(price)) return price; // Return original if not a valid number
    if (!market?.precision?.price) {
        // console.warn(`[Strategy] Missing price precision for ${market?.symbol}. Using default formatting.`);
        // Attempt a reasonable default based on magnitude? Or fixed decimal places.
        return parseFloat(price.toFixed(4)); // Example default
    }
    return parseFloat(price.toFixed(market.precision.price));
};

// Formats an amount based on market precision rules (using floor to avoid exceeding limits)
const formatAmount = (amount, market) => {
    if (typeof amount !== 'number' || isNaN(amount)) return amount;
    if (!market?.precision?.amount) {
        // console.warn(`[Strategy] Missing amount precision for ${market?.symbol}. Using default formatting.`);
        return parseFloat(amount.toFixed(6)); // Example default
    }
    // Use floor to ensure we don't round up over exchange limits
    const factor = Math.pow(10, market.precision.amount);
    const flooredAmount = Math.floor(amount * factor) / factor;
    return parseFloat(flooredAmount.toFixed(market.precision.amount)); // Ensure correct decimal places after floor
};

// --- Core Strategy Logic ---
const runStrategy = async () => {
    const now = Date.now();
    // Prevent concurrent executions and enforce minimum interval
    if (isProcessingTrade) {
        logTrade("Skipping execution: Previous run still processing.", "DEBUG");
        return;
    }
     if (now - lastExecutionTime < MIN_INTERVAL_MS) {
        // logTrade(`Skipping execution: Too soon since last run (executed ${Math.round((now - lastExecutionTime)/1000)}s ago).`, "DEBUG");
        return;
    }

    isProcessingTrade = true;
    lastExecutionTime = now;
    logTrade(`--- Running Strategy Check [${currentConfig.strategy}] for ${currentConfig.symbol} (${currentConfig.interval}) ---`, "INFO");

    try {
        // 0. Pre-checks: Ensure Bybit service is ready
        if (!getBybit()) {
            throw new Error("Bybit service is not initialized. Cannot run strategy.");
        }

        // 1. Get Market Details (Precision, Limits) - Crucial for orders
        const market = await getMarketDetails(currentConfig.symbol);
        if (!market || !market.limits || !market.precision) {
            // This is critical, stop trading if we can't get market info
            logTrade(`Market details unavailable for ${currentConfig.symbol}. Cannot proceed. Stopping trading.`, "ERROR");
            stopTrading();
            isProcessingTrade = false;
            return;
        }
        // logTrade(`Market details fetched: Price Precision=${market.precision.price}, Amount Precision=${market.precision.amount}, Min Amount=${market.limits.amount?.min}`, "DEBUG");

        // 2. Fetch OHLCV Data
        const ohlcv = await fetchOHLCV(currentConfig.symbol, currentConfig.interval, DEFAULT_OHLCV_LIMIT);
        const requiredCandles = Math.max(currentConfig.indicatorPeriod, currentConfig.atrPeriod, currentConfig.ehlersMaPeriod, 50); // Min required candles
        if (!ohlcv || ohlcv.length < requiredCandles) {
            logTrade(`Insufficient historical data (${ohlcv?.length || 0} candles) for ${currentConfig.symbol}. Need at least ${requiredCandles}. Skipping check.`, "WARN");
            isProcessingTrade = false;
            return;
        }
        const lastCandle = ohlcv[ohlcv.length - 1];
        const currentPrice = lastCandle.close;
        logTrade(`Current Price (${currentConfig.symbol}): ${formatPrice(currentPrice, market)}`, "DEBUG");

        // 3. Calculate Indicators
        const indicators = calculateIndicators(ohlcv, currentConfig);
        // Check for essential indicators needed by the strategy
        if (!indicators.atr || !indicators.stochRsi || !indicators.ehlersMa) {
            logTrade("Failed to calculate required indicators (ATR, StochRSI, EhlersMA). Check data or config. Skipping check.", "WARN");
            isProcessingTrade = false;
            return;
        }
         logTrade(`Indicators: ATR=${indicators.atr?.toFixed(5)}, StochK=${indicators.stochRsi?.k?.toFixed(2)}, StochD=${indicators.stochRsi?.d?.toFixed(2)}, MA=${indicators.ehlersMa?.toFixed(5)}`, "DEBUG");


        // 4. Fetch Current State (Balance, Position)
        const balance = await fetchBalance('USDT'); // Assuming USDT margin
        const position = await fetchPosition(currentConfig.symbol);
        const hasOpenPosition = position !== null;
        const positionSide = hasOpenPosition ? (position.side === 'long' ? 'buy' : 'sell') : null; // 'buy' or 'sell'
        const positionSizeContracts = hasOpenPosition ? parseFloat(position.contracts || position.info?.size || 0) : 0; // Size in contracts/base currency

        logTrade(`Balance: ${formatPrice(balance, { precision: { price: 2 } })} USDT. Position: ${hasOpenPosition ? `${position.side.toUpperCase()} ${formatAmount(positionSizeContracts, market)} ${market.base} @ ${formatPrice(position.entryPrice, market)}` : 'None'}`, "DEBUG");


        // --- Strategy Implementation: STOCH_RSI_EHLERS_MA ---
        let entrySignal = null; // 'buy' or 'sell'
        let exitSignal = false;

        // Validate indicator values before use
        if (typeof indicators.stochRsi?.k !== 'number' || typeof indicators.stochRsi?.d !== 'number' || typeof indicators.ehlersMa !== 'number') {
            logTrade("Invalid indicator values received (NaN or null). Skipping strategy logic.", "WARN");
            isProcessingTrade = false;
            return;
        }

        const { k: stochK, d: stochD } = indicators.stochRsi;
        const maValue = indicators.ehlersMa;
        const priceAboveMa = currentPrice > maValue;
        const priceBelowMa = currentPrice < maValue;

        // --- Entry Conditions ---
        if (!hasOpenPosition) {
            // Long Entry: Stoch %K crosses above %D (fast > slow), both below 25 (oversold), and price is above the MA (trend confirmation)
            if (stochK > stochD && stochK < 25 && priceAboveMa) {
                entrySignal = 'buy';
                logTrade(`LONG ENTRY Signal: StochK(${stochK.toFixed(2)}) > StochD(${stochD.toFixed(2)}) [Both < 25], Price(${formatPrice(currentPrice, market)}) > MA(${formatPrice(maValue, market)})`);
            }
            // Short Entry: Stoch %K crosses below %D (fast < slow), both above 75 (overbought), and price is below the MA (trend confirmation)
            else if (stochK < stochD && stochK > 75 && priceBelowMa) {
                entrySignal = 'sell';
                logTrade(`SHORT ENTRY Signal: StochK(${stochK.toFixed(2)}) < StochD(${stochD.toFixed(2)}) [Both > 75], Price(${formatPrice(currentPrice, market)}) < MA(${formatPrice(maValue, market)})`);
            }
        }

        // --- Exit Conditions ---
        if (hasOpenPosition) {
            // Long Exit: Stoch %K crosses below %D (fast < slow) while above 75 (overbought exit)
            if (positionSide === 'buy' && stochK < stochD && stochK > 75) {
                exitSignal = true;
                logTrade(`LONG EXIT Signal: StochK(${stochK.toFixed(2)}) < StochD(${stochD.toFixed(2)}) [Both > 75]`);
            }
            // Short Exit: Stoch %K crosses above %D (fast > slow) while below 25 (oversold exit)
            else if (positionSide === 'sell' && stochK > stochD && stochK < 25) {
                exitSignal = true;
                logTrade(`SHORT EXIT Signal: StochK(${stochK.toFixed(2)}) > StochD(${stochD.toFixed(2)}) [Both < 25]`);
            }
            // Add SL/TP exit logic here IF NOT using exchange-native SL/TP orders
            // Example: Check if currentPrice hit calculated SL/TP based on entry + ATR
            // const entryPrice = parseFloat(position.entryPrice);
            // const atrValue = indicators.atr;
            // if (positionSide === 'buy' && currentPrice <= (entryPrice - atrValue * currentConfig.atrSlMult)) { exitSignal = true; logTrade("STOP LOSS hit for LONG position.", "WARN"); }
            // else if (positionSide === 'sell' && currentPrice >= (entryPrice + atrValue * currentConfig.atrSlMult)) { exitSignal = true; logTrade("STOP LOSS hit for SHORT position.", "WARN"); }
        }

        // --- Execute Trades ---
        if (entrySignal && !hasOpenPosition) {
            logTrade(`Attempting to execute ${entrySignal.toUpperCase()} entry for ${currentConfig.symbol}...`, "INFO");

            if (balance <= 0) {
                logTrade("Insufficient balance (0 USDT). Cannot open position.", "ERROR");
                isProcessingTrade = false;
                return;
            }

            // Calculate Position Size based on Risk & ATR Stop Loss
            const atrValue = indicators.atr;
            if (typeof atrValue !== 'number' || atrValue <= 0) {
                logTrade(`Invalid ATR value (${atrValue}). Cannot calculate position size.`, "ERROR");
                isProcessingTrade = false;
                return;
            }

            const slDistancePoints = atrValue * currentConfig.atrSlMult; // Stop distance in price points
            const riskAmountQuote = balance * currentConfig.riskPerTrade; // Risk amount in USDT

            // Stop Loss Price Calculation
            const stopLossPrice = entrySignal === 'buy' ? currentPrice - slDistancePoints : currentPrice + slDistancePoints;
            const formattedSL = formatPrice(stopLossPrice, market);

            // Take Profit Price Calculation (Optional - Bybit might require TP to be set with SL)
            const tpDistancePoints = atrValue * currentConfig.atrTpMult;
            const takeProfitPrice = entrySignal === 'buy' ? currentPrice + tpDistancePoints : currentPrice - tpDistancePoints;
            const formattedTP = formatPrice(takeProfitPrice, market);

            // Calculate position size in BASE currency (e.g., BTC for BTC/USDT)
            // Size = Risk Amount (Quote) / Stop Distance (Quote per Base)
            if (slDistancePoints <= 0) {
                 logTrade(`Stop Loss distance is zero or negative (${slDistancePoints.toFixed(5)}). Cannot calculate position size. Check ATR/Multiplier.`, "ERROR");
                 isProcessingTrade = false;
                 return;
            }
            let positionSizeBase = riskAmountQuote / slDistancePoints;

            // --- Apply Market Limits and Precision ---
            const minAmount = market.limits?.amount?.min;
            const maxAmount = market.limits?.amount?.max;

            logTrade(`Trade Calculation: Balance=${balance.toFixed(2)}, Risk=${(currentConfig.riskPerTrade * 100).toFixed(2)}%, ATR=${atrValue.toFixed(5)}, SL Dist=${slDistancePoints.toFixed(5)}`, "DEBUG");
            logTrade(`Calculated Size (Base): ${positionSizeBase.toFixed(8)}, Min Allowed: ${minAmount ?? 'N/A'}, Max Allowed: ${maxAmount ?? 'N/A'}`, "DEBUG");


            if (typeof minAmount === 'number' && positionSizeBase < minAmount) {
                logTrade(`Calculated size (${positionSizeBase.toFixed(8)}) is below minimum (${minAmount}). Adjusting to minimum.`, "WARN");
                positionSizeBase = minAmount;
            }
            if (typeof maxAmount === 'number' && positionSizeBase > maxAmount) {
                logTrade(`Calculated size (${positionSizeBase.toFixed(8)}) exceeds maximum (${maxAmount}). Adjusting to maximum.`, "WARN");
                positionSizeBase = maxAmount;
            }

            // Format final size according to market precision AFTER applying limits
            const finalPositionSize = formatAmount(positionSizeBase, market);

            // Final validation of size after formatting and limits
            if (finalPositionSize <= 0 || (typeof minAmount === 'number' && finalPositionSize < minAmount)) {
                logTrade(`Final position size (${finalPositionSize}) is invalid or below minimum after adjustments. Cannot place order.`, "ERROR");
                isProcessingTrade = false;
                return;
            }

            logTrade(`Final Size (Formatted): ${finalPositionSize} ${market.base}`, "INFO");
            logTrade(`Target Entry: ~${formatPrice(currentPrice, market)}, SL: ${formattedSL}, TP: ${formattedTP}`, "INFO");

            // Set Leverage before placing the order (best practice)
            try {
                await setLeverage(currentConfig.symbol, currentConfig.leverage);
            } catch (leverageError) {
                // Log warning but potentially continue if leverage setting isn't critical or already set
                logTrade(`Failed to set leverage to ${currentConfig.leverage}x. Continuing with existing leverage. Error: ${leverageError.message}`, "WARN");
            }

            // --- Place Market Order with SL/TP ---
            // Construct order parameters. Parameter names depend heavily on the exchange (Bybit V5 example).
            // Check CCXT documentation and Bybit API V5 docs for exact parameter names.
            const orderParams = {
                // 'stopLoss': formattedSL, // Price for SL
                // 'takeProfit': formattedTP, // Price for TP
                // 'slTriggerPrice': formattedSL, // Trigger price might be needed
                // 'tpTriggerPrice': formattedTP, // Trigger price might be needed
                // 'tpslMode': 'Full', // Or 'Partial' - if you want SL/TP to close the whole position
                // 'slTriggerBy': 'LastPrice', // Or MarkPrice, IndexPrice
                // 'tpTriggerBy': 'LastPrice',
                // Add more params as needed, e.g., timeInForce, clientOrderId
            };
            // Filter out undefined params to avoid sending empty values
            Object.keys(orderParams).forEach(key => (orderParams[key] === undefined || orderParams[key] === null) && delete orderParams[key]);

            try {
                // Use createMarketOrder helper
                const orderResult = await createMarketOrder(currentConfig.symbol, entrySignal, finalPositionSize, orderParams);
                logTrade(`Market ${entrySignal.toUpperCase()} order placed for ${finalPositionSize} ${market.base}. Order ID: ${orderResult.id}. Params sent: ${JSON.stringify(orderParams)}`, "SUCCESS");
                // Optional: Add logic here to monitor order fill status if needed immediately
            } catch (orderError) {
                 logTrade(`Failed to place market ${entrySignal.toUpperCase()} order: ${orderError.message}`, "ERROR");
                 // Consider specific error handling (e.g., insufficient margin)
                 if (orderError.message.includes('insufficient balance') || orderError.message.includes('margin')) {
                     logTrade("Order failed likely due to insufficient margin/balance.", "ERROR");
                 }
            }

        } else if (exitSignal && hasOpenPosition) {
            logTrade(`Attempting to execute ${positionSide.toUpperCase()} EXIT for ${currentConfig.symbol} due to strategy signal...`, "INFO");
            const closeSide = positionSide === 'buy' ? 'sell' : 'buy'; // Opposite side to close
            const amountToClose = Math.abs(positionSizeContracts); // Use the actual size from the fetched position

            // Format amount according to market precision
            const formattedAmountToClose = formatAmount(amountToClose, market);

            if (formattedAmountToClose <= 0) {
                logTrade(`Invalid amount to close (${formattedAmountToClose}). Cannot place close order.`, "ERROR");
                isProcessingTrade = false;
                return;
            }

            // Place Market Order to close the position
            // Use 'reduceOnly' parameter to ensure it only reduces/closes the position
            const closeParams = {
                 'reduceOnly': true // CRITICAL for safely closing positions
            };

            try {
                const closeOrderResult = await createMarketOrder(currentConfig.symbol, closeSide, formattedAmountToClose, closeParams);
                logTrade(`Market ${closeSide.toUpperCase()} (CLOSE) order placed for ${formattedAmountToClose} ${market.base}. Order ID: ${closeOrderResult.id}.`, "SUCCESS");
                 // Optional: Monitor fill status
            } catch (orderError) {
                 logTrade(`Failed to place market ${closeSide.toUpperCase()} (CLOSE) order: ${orderError.message}`, "ERROR");
                 // Handle potential errors during close (e.g., already closed, network issue)
            }
        } else {
             logTrade("No entry or exit conditions met.", "DEBUG");
        }

    } catch (error) {
        // Catch errors from API calls (fetchOHLCV, fetchPosition etc.) or logic errors
        logTrade(`Strategy Execution Error: ${error.message}`, "ERROR");
        console.error("[Strategy] Full error stack trace:", error);
        // Consider adding logic to stop trading on repeated critical errors
        // if (error.message.includes("authentication") || error.message.includes("API key")) {
        //    logTrade("Potential API key issue detected. Stopping trading.", "ERROR");
        //    stopTrading();
        // }
    } finally {
        isProcessingTrade = false; // Release the lock
        logTrade("--- Strategy Check Complete ---", "DEBUG");
    }
};

// --- Control Functions ---

// Function to parse interval string (e.g., "5m", "1h") into milliseconds
const parseIntervalToMs = (intervalString) => {
    if (!intervalString || typeof intervalString !== 'string') {
        throw new Error("Invalid interval string format.");
    }
    const value = parseInt(intervalString.slice(0, -1), 10);
    const unit = intervalString.slice(-1).toLowerCase();

    if (isNaN(value) || value <= 0) {
        throw new Error(`Invalid interval value: ${value}`);
    }

    switch (unit) {
        case 'm': return value * 60 * 1000;
        case 'h': return value * 60 * 60 * 1000;
        case 'd': return value * 24 * 60 * 60 * 1000;
        case 'w': return value * 7 * 24 * 60 * 60 * 1000;
        default: throw new Error(`Unsupported interval unit: ${unit} (use m, h, d, w)`);
    }
};


const startTrading = (configUpdate = {}) => {
    if (tradingIntervalId) {
        logTrade("Trading loop is already active.", "WARN");
        return { success: false, message: "Trading loop already running." };
    }

    isTradingEnabled = true;
    // Update config with any provided overrides, falling back to current/initial config
    currentConfig = { ...currentConfig, ...configUpdate };
    logTrade(`--- Starting Trading Loop [${currentConfig.strategy}] ---`, "INFO");
    logTrade(`Applied Configuration: ${JSON.stringify(currentConfig)}`, "INFO");

    if (process.env.USE_SANDBOX !== 'true') {
         logTrade("!!! LIVE TRADING MODE ACTIVE - MONITOR CLOSELY !!!", "WARN");
    } else {
         logTrade("Sandbox Mode Active.", "INFO");
    }

    let intervalMs;
    try {
        intervalMs = parseIntervalToMs(currentConfig.interval);
        // Enforce minimum interval to prevent excessive API calls / rate limiting
        if (intervalMs < MIN_INTERVAL_MS) {
            logTrade(`Requested interval (${currentConfig.interval}) is below minimum (${MIN_INTERVAL_MS}ms). Adjusting to minimum.`, "WARN");
            intervalMs = MIN_INTERVAL_MS;
        }
        logTrade(`Setting trading interval to ${intervalMs} ms (${currentConfig.interval})`, "INFO");
    } catch (e) {
        logTrade(`Invalid interval string: "${currentConfig.interval}". Stopping trading start. Error: ${e.message}`, "ERROR");
        isTradingEnabled = false; // Ensure state reflects failure
        return { success: false, message: `Invalid interval: ${e.message}` };
    }

    // Run immediately after a short delay (allow for setup/logging)
    // Check isTradingEnabled again in case stopTrading was called quickly
    const initialRunTimeout = setTimeout(() => {
        if (isTradingEnabled) {
             logTrade("Performing initial strategy run...", "INFO");
             runStrategy();
        }
    }, 2000); // 2-second delay before first run

    // Set up the interval timer
    tradingIntervalId = setInterval(runStrategy, intervalMs);

    return { success: true, message: `Trading loop started for ${currentConfig.symbol} (${currentConfig.interval}).` };
};

const stopTrading = () => {
    if (!tradingIntervalId) {
        logTrade("Trading loop is not currently running.", "WARN");
        return { success: false, message: "Trading loop not running." };
    }

    clearInterval(tradingIntervalId);
    tradingIntervalId = null;
    isTradingEnabled = false;
    isProcessingTrade = false; // Reset processing flag
    lastExecutionTime = 0; // Reset timer
    logTrade("--- Trading loop stopped ---", "INFO");
    return { success: true, message: "Trading loop stopped successfully." };
};

const updateConfig = (newConfig) => {
    if (typeof newConfig !== 'object' || newConfig === null) {
        logTrade("Invalid configuration update data received.", "ERROR");
        return { success: false, message: "Invalid configuration data." };
    }

    const wasTrading = isTradingEnabled;
    if (wasTrading) {
        logTrade("Stopping trading loop to apply new configuration...", "INFO");
        stopTrading();
    }

    // Merge new config, ensuring type conversions if necessary (e.g., strings to numbers)
    // Basic merge - add validation/type checking as needed
    currentConfig = { ...currentConfig, ...newConfig };
    // Re-validate/convert numeric types potentially passed as strings from JSON
    currentConfig.leverage = parseInt(currentConfig.leverage, 10);
    currentConfig.riskPerTrade = parseFloat(currentConfig.riskPerTrade);
    // ... add for other numeric fields ...

    logTrade(`Configuration updated: ${JSON.stringify(currentConfig)}`, "INFO");

    // Restart trading if it was running before the update
    if (wasTrading) {
        logTrade("Restarting trading loop with new configuration...", "INFO");
        // Small delay before restarting
        setTimeout(() => startTrading(), 1000); // Restart uses the updated currentConfig
    }

    return { success: true, config: currentConfig };
};


const getStatus = async () => {
    let balance = null;
    let position = null;
    let statusErrorMsg = null;
    let market = null;

    try {
        // Ensure Bybit is initialized before fetching status data
        if (!getBybit()) {
             throw new Error("Trading service (Bybit) not ready.");
        }
        // Fetch data concurrently for speed
        [balance, position, market] = await Promise.all([
            fetchBalance('USDT'), // Assuming USDT margin
            fetchPosition(currentConfig.symbol),
            getMarketDetails(currentConfig.symbol) // Fetch market details for formatting position
        ]).catch(fetchError => {
             // Catch errors during the Promise.all fetch
             logTrade(`Error fetching status data: ${fetchError.message}`, "ERROR");
             statusErrorMsg = `Failed to fetch status: ${fetchError.message}`;
             // Return partial data if possible, or nulls
             return [balance, position, market]; // Return whatever might have resolved before error
        });

        // Format position data using market details if available
        if (position && market) {
             position.entryPriceFormatted = formatPrice(position.entryPrice, market);
             position.markPriceFormatted = formatPrice(position.markPrice, market);
             position.liquidationPriceFormatted = formatPrice(position.liquidationPrice, market);
             position.contractsFormatted = formatAmount(position.contracts || position.info?.size || 0, market);
             // Add PNL formatting if needed
        }

    } catch (error) {
        // Catch errors outside the Promise.all (e.g., getBybit failure)
        logTrade(`Error in getStatus function: ${error.message}`, "ERROR");
        statusErrorMsg = `Failed to get status: ${error.message}`;
    }

    return {
        isTradingEnabled,
        config: currentConfig,
        logs: [...tradeLogs].reverse(), // Return newest logs first for UI
        balance,
        position, // Contains formatted fields if market data was available
        error: statusErrorMsg, // Include error message if fetch failed
        lastUpdate: new Date().toISOString(),
        lastStrategyRun: lastExecutionTime > 0 ? new Date(lastExecutionTime).toISOString() : null,
    };
};

module.exports = {
    startTrading,
    stopTrading,
    getStatus,
    updateConfig,
};
