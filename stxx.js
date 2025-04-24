
#!/usr/bin/env node

// ███████╗██╗   ██╗███████╗ ██████╗ ███████╗███████╗██╗   ██╗███████╗
// ██╔════╝██║   ██║██╔════╝██╔════╝ ██╔════╝██╔════╝██║   ██║██╔════╝
// ███████╗██║   ██║███████╗██║  ███╗███████╗███████╗██║   ██║███████╗
// ╚════██║██║   ██║╚════██║██║   ██║╚════██║╚════██║██║   ██║╚════██║
// ███████║╚██████╔╝███████║╚██████╔╝███████║███████║╚██████╔╝███████║
// ╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚══════╝
// Pyrmethus Roofed/Fisher/SuperTrend/ADX Bot - Enhanced Invocation v1.1

// --- Arcane Libraries & Modules ---
import ccxt from 'ccxt';
import dotenv from 'dotenv';
import { exec } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import {
    cyan, green, yellow, red, magenta, gray, bold, dim, reset
} from 'nanocolors';
import { EMA, ATR, ADX, InverseFisherTransform } from 'technicalindicators';

dotenv.config(); // Awaken configuration runes

// --- Configuration Glyphs (Loaded from .env) ---
const config = {
    BYBIT_API_KEY: process.env.BYBIT_API_KEY,
    BYBIT_API_SECRET: process.env.BYBIT_API_SECRET,
    SYMBOL: process.env.SYMBOL,
    LEVERAGE: process.env.LEVERAGE,
    TIMEFRAME: process.env.TIMEFRAME,
    ORDER_AMOUNT_USD: process.env.ORDER_AMOUNT_USD,
    ROOF_FAST_EMA: process.env.ROOF_FAST_EMA,
    ROOF_SLOW_EMA: process.env.ROOF_SLOW_EMA,
    ST_EMA_PERIOD: process.env.ST_EMA_PERIOD,
    ST_ATR_PERIOD: process.env.ST_ATR_PERIOD,
    ST_MULTIPLIER: process.env.ST_MULTIPLIER,
    FISHER_PERIOD: process.env.FISHER_PERIOD,
    ADX_PERIOD: process.env.ADX_PERIOD,
    MIN_ADX_LEVEL: process.env.MIN_ADX_LEVEL,
    MAX_ADX_LEVEL: process.env.MAX_ADX_LEVEL,
    RANGE_ATR_PERIOD: process.env.RANGE_ATR_PERIOD,
    MIN_ATR_PERCENTAGE: process.env.MIN_ATR_PERCENTAGE,
    TSL_ATR_MULTIPLIER: process.env.TSL_ATR_MULTIPLIER,
    INITIAL_SL_ATR_MULTIPLIER: process.env.INITIAL_SL_ATR_MULTIPLIER,
    POLL_INTERVAL_MS: process.env.POLL_INTERVAL_MS,
    PERSISTENCE_FILE: process.env.PERSISTENCE_FILE || 'trading_state_roof_fisher.json',
    LOG_LEVEL: process.env.LOG_LEVEL || 'INFO',
    DRY_RUN: process.env.DRY_RUN === 'true',
    TERMUX_NOTIFY: process.env.TERMUX_NOTIFY === 'true',
    NOTIFICATION_PHONE_NUMBER: process.env.NOTIFICATION_PHONE_NUMBER,
};

// --- Pathfinding & Constants ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const persistenceFilePath = path.join(__dirname, config.PERSISTENCE_FILE);
const LOG_LEVELS = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
const currentLogLevel = LOG_LEVELS[config.LOG_LEVEL.toUpperCase()] ?? LOG_LEVELS.INFO;

// --- The Oracle's Voice (Logger) ---
const logger = {
    debug: (message) => currentLogLevel <= LOG_LEVELS.DEBUG && console.log(dim(gray(`[DEBUG] ${new Date().toISOString()} ${message}`))),
    info: (message) => currentLogLevel <= LOG_LEVELS.INFO && console.log(cyan(`[INFO]  ${new Date().toISOString()} ${message}`)),
    warn: (message) => currentLogLevel <= LOG_LEVELS.WARN && console.log(yellow(`[WARN]  ${new Date().toISOString()} ${message}`)),
    error: (message) => currentLogLevel <= LOG_LEVELS.ERROR && console.log(red(bold(`[ERROR] ${new Date().toISOString()} ${message}`))),
    trade: (message) => console.log(green(bold(`[TRADE] ${new Date().toISOString()} ${message}`))),
    dryRun: (message) => config.DRY_RUN && console.log(magenta(`[DRY RUN] ${new Date().toISOString()} ${message}`))
};

// --- The Bot's Memory (State) ---
let state = {
    positionSide: null, // 'long', 'short', or null
    entryPrice: null,   // number | null
    positionAmount: null, // number | null
    currentTSL: null,   // { price: number, orderId: string | null } | null
    lastSignal: null,   // 'long' or 'short' to prevent immediate flip-flop entries
    cycleCount: 0       // number
};

// --- State Scroll Management ---

/**
 * Loads the bot's state from the persistence file.
 * Initializes state if the file doesn't exist.
 */
async function loadState() {
    try {
        const data = await fs.readFile(persistenceFilePath, 'utf8');
        state = JSON.parse(data);
        logger.info(`Resuming consciousness from scroll: ${persistenceFilePath}`);
        state.cycleCount = state.cycleCount || 0; // Ensure cycleCount exists
    } catch (error) {
        if (error.code === 'ENOENT') {
            logger.warn(`State scroll not found: ${persistenceFilePath}. Conjuring fresh memory.`);
        } else {
            logger.error(`Error deciphering state scroll: ${error.message}. Starting fresh.`);
        }
        // Initialize state if loading fails or file not found
        state = { positionSide: null, entryPrice: null, positionAmount: null, currentTSL: null, lastSignal: null, cycleCount: 0 };
    }
}

/**
 * Saves the current bot state to the persistence file.
 */
async function saveState() {
    try {
        await fs.writeFile(persistenceFilePath, JSON.stringify(state, null, 2), 'utf8');
        logger.debug(`Memory inscribed upon ${persistenceFilePath}`);
    } catch (error) {
        logger.error(`Error inscribing state scroll: ${error.message}`);
    }
}

// --- Termux SMS Dispatch ---

/**
 * Sends an SMS notification using Termux API if enabled.
 * @param {string} message The message content to send.
 */
function sendTermuxSms(message) {
    if (!config.TERMUX_NOTIFY || !config.NOTIFICATION_PHONE_NUMBER) {
        logger.debug('Termux SMS dormant.');
        return;
    }
    const command = `termux-sms-send -n "${config.NOTIFICATION_PHONE_NUMBER}" "${message}"`;
    logger.info(`Dispatching Termux SMS: ${message}`);
    exec(command, (error, stdout, stderr) => {
        if (error) {
            logger.error(`Termux SMS dispatch failed: ${error.message}`);
            return;
        }
        if (stderr) {
            logger.warn(`Termux SMS stderr whispers: ${stderr}`);
        }
        logger.info(`Termux SMS dispatch attempted.`);
        if (currentLogLevel <= LOG_LEVELS.DEBUG) {
            logger.debug(`Termux SMS stdout echoes: ${stdout}`);
        }
    });
}

// --- Exchange Citadel Connection ---
const exchange = new ccxt.bybit({
    apiKey: config.BYBIT_API_KEY,
    secret: config.BYBIT_API_SECRET,
    options: { defaultType: 'linear' } // Ensure linear contracts
});

if (config.DRY_RUN) {
    exchange.setSandboxMode(true);
    logger.warn(bold('DRY RUN MODE: Operating in simulation plane. No real orders will be placed.'));
}

// --- Market Divination (Indicator Calculation) ---

/**
 * Calculates all necessary technical indicators.
 * @param {Array<Array<number>>} candles - OHLCV candle data [[timestamp, open, high, low, close, volume], ...]
 * @returns {Promise<object|null>} An object containing the latest indicator values, or null if calculation fails.
 */
async function calculateIndicators(candles) {
    const requiredCandlesMin = Math.max(
        parseInt(config.ROOF_SLOW_EMA), // Roofing needs the longest EMA
        parseInt(config.ST_EMA_PERIOD) + parseInt(config.ST_ATR_PERIOD), // SuperTrend combination
        parseInt(config.FISHER_PERIOD),
        parseInt(config.ADX_PERIOD),
        parseInt(config.RANGE_ATR_PERIOD)
    ) + 1; // +1 for Fisher signal lag and general buffer

    if (!candles || candles.length < requiredCandlesMin) {
        logger.warn(`Insufficient candle data (${candles?.length}) for divination (need at least ${requiredCandlesMin}).`);
        return null;
    }

    // Prepare input arrays
    const high = candles.map(c => c[2]);
    const low = candles.map(c => c[3]);
    const close = candles.map(c => c[4]);
    const hlc3 = candles.map(c => (c[2] + c[3] + c[4]) / 3); // Typical price for roofing

    try {
        // ---===[ Ehlers Roofing Filter Calculation ]===---
        // Purpose: Smooth price data and remove high-frequency noise. Output is Fast EMA - Slow EMA of HLC3.
        logger.debug(`Calculating Roofing Filter (Fast EMA ${config.ROOF_FAST_EMA}, Slow EMA ${config.ROOF_SLOW_EMA}) on HLC3...`);
        const roofFastPeriod = parseInt(config.ROOF_FAST_EMA);
        const roofSlowPeriod = parseInt(config.ROOF_SLOW_EMA);

        const emaFastRoof = EMA.calculate({ period: roofFastPeriod, values: hlc3 });
        const emaSlowRoof = EMA.calculate({ period: roofSlowPeriod, values: hlc3 });

        if (!emaFastRoof || !emaSlowRoof || emaFastRoof.length === 0 || emaSlowRoof.length === 0) {
            logger.warn("Roofing Filter EMA calculation failed or yielded zero length."); return null;
        }

        // Align and calculate the filter output (Fast EMA - Slow EMA)
        const roofLength = Math.min(emaFastRoof.length, emaSlowRoof.length);
        const roofFilteredPrice = [];
        const emaFastRoofAligned = emaFastRoof.slice(-roofLength);
        const emaSlowRoofAligned = emaSlowRoof.slice(-roofLength);
        for (let i = 0; i < roofLength; i++) {
            roofFilteredPrice.push(emaFastRoofAligned[i] - emaSlowRoofAligned[i]);
        }
        logger.debug(`Roofing Filter applied. Output length: ${roofFilteredPrice.length}`);
        // ---===========================================---

        // 1. Ehlers-Inspired SuperTrend (Basis uses Roofed Price)
        // Purpose: Determine trend direction using ATR bands around a smoothed price (the roofed price).
        logger.debug(`Calculating EMA(${config.ST_EMA_PERIOD}) on Roofed Price for ST Basis...`);
        // Smooth the *output* of the roofing filter to use as the SuperTrend basis
        const stBasisSmoothed = EMA.calculate({ period: parseInt(config.ST_EMA_PERIOD), values: roofFilteredPrice });

        logger.debug(`Calculating ATR(${config.ST_ATR_PERIOD}) for SuperTrend (using regular HLC)...`);
        const atrSuperTrendArr = ATR.calculate({ period: parseInt(config.ST_ATR_PERIOD), high, low, close });

        if (!stBasisSmoothed || !atrSuperTrendArr || stBasisSmoothed.length === 0 || atrSuperTrendArr.length === 0) {
             logger.warn('Failed to calculate Smoothed Basis or ATR for SuperTrend.'); return null;
        }

        // Align all ST inputs based on the shortest array length after calculations
        const stInputLength = Math.min(stBasisSmoothed.length, atrSuperTrendArr.length, high.length, low.length, close.length);
        if (stInputLength === 0) {
            logger.warn('SuperTrend input alignment resulted in zero length.'); return null;
        }

        const superTrendInput = {
            high: high.slice(-stInputLength),
            low: low.slice(-stInputLength),
            close: close.slice(-stInputLength), // Use actual close for trend flip check
            atr: atrSuperTrendArr.slice(-stInputLength),
            basis: stBasisSmoothed.slice(-stInputLength), // Using the smoothed, roofed price difference as basis
            multiplier: parseFloat(config.ST_MULTIPLIER),
        };
        logger.debug('Invoking custom SuperTrend calculation with roofed basis...');
        const stResult = calculateCustomSuperTrend(superTrendInput);
        if (!stResult || stResult.length === 0) {
             logger.warn('Custom SuperTrend calculation failed or yielded zero length.'); return null;
        }


        // 2. Ehlers Fisher Transform (on High/Low)
        // Purpose: Identify potential turning points by normalizing price movement. Uses High/Low.
        logger.debug(`Calculating Fisher Transform(${config.FISHER_PERIOD})...`);
        const fisherInput = { high: high, low: low, period: parseInt(config.FISHER_PERIOD) };
        const fisherResult = InverseFisherTransform.calculate(fisherInput);
        if (!fisherResult || fisherResult.length === 0) {
             logger.warn('Fisher Transform calculation failed or yielded zero length.'); return null;
        }
        const fisherSignal = [NaN, ...fisherResult.slice(0, -1)]; // 1-period lagged signal line

        // 3. ADX (Average Directional Index) Calculation
        // Purpose: Measure trend strength (not direction). Uses HLC.
        logger.debug(`Calculating ADX(${config.ADX_PERIOD})...`);
        const adxResult = ADX.calculate({ period: parseInt(config.ADX_PERIOD), high, low, close });
         if (!adxResult || adxResult.length === 0) {
             logger.warn('ADX calculation failed or yielded zero length.'); return null;
        }

        // 4. ATR for Range Filter
        // Purpose: Measure volatility relative to price for range filtering. Uses HLC.
        logger.debug(`Calculating ATR(${config.RANGE_ATR_PERIOD}) for Range Filter...`);
        const atrRangeResult = ATR.calculate({ period: parseInt(config.RANGE_ATR_PERIOD), high, low, close });
         if (!atrRangeResult || atrRangeResult.length === 0) {
             logger.warn('ATR Range calculation failed or yielded zero length.'); return null;
        }

        // --- Extract Latest Values ---
        // Find the latest valid index across all potentially different length arrays resulting from calculations
        const lastValidIdx = Math.min(
            stResult.length,
            fisherResult.length,
            fisherSignal.length, // Make sure signal is included
            adxResult.length,
            atrRangeResult.length,
            atrSuperTrendArr.length // ATR used for ST/TSL
        ) - 1;

        if (lastValidIdx < 0) {
            logger.warn("Could not get a valid common index for latest indicator values after calculations.");
            return null;
        }

        // Safely access the latest values using the calculated common index
        const latestST = stResult[lastValidIdx];
        const latestFisher = fisherResult[lastValidIdx];
        const latestFisherSignal = fisherSignal[lastValidIdx];
        const prevFisher = lastValidIdx > 0 ? fisherResult[lastValidIdx - 1] : NaN;
        const prevFisherSignal = lastValidIdx > 0 ? fisherSignal[lastValidIdx - 1] : NaN;
        const latestAdx = adxResult[lastValidIdx];
        const latestAtrRange = atrRangeResult[lastValidIdx];
        const latestAtrST = atrSuperTrendArr[lastValidIdx]; // ATR for ST/TSL

        // Final check for NaN values in critical indicators
        if (!latestST || typeof latestST.value !== 'number' || isNaN(latestST.value) ||
            !latestAdx || typeof latestAdx.adx !== 'number' || isNaN(latestAdx.adx) ||
            isNaN(latestFisher) || isNaN(latestFisherSignal) ||
            isNaN(latestAtrRange) || isNaN(latestAtrST))
        {
            logger.warn("Could not divine all latest indicator values (some might be NaN or undefined after indexing).");
            if (currentLogLevel <= LOG_LEVELS.DEBUG) {
                 logger.debug(`Indexes: LastValid=${lastValidIdx}`);
                 logger.debug(`Values: ST=${JSON.stringify(latestST)}, Fisher=${latestFisher}, FisherSig=${latestFisherSignal}, ADX=${JSON.stringify(latestAdx)}, ATR_Range=${latestAtrRange}, ATR_ST=${latestAtrST}`);
            }
            return null;
        }

        const latestClosePrice = close[close.length - 1]; // Use the very latest close price

        logger.debug(`Latest Indicators: Price=${latestClosePrice.toFixed(4)}, ST=${latestST.value.toFixed(4)} (${latestST.direction}), Fisher=${latestFisher.toFixed(3)}, Sig=${latestFisherSignal.toFixed(3)}, ADX=${latestAdx.adx.toFixed(2)}, ATR_Range=${latestAtrRange.toFixed(4)}, ATR_ST=${latestAtrST.toFixed(4)}`);

        return {
            superTrendValue: latestST.value,
            superTrendDirection: latestST.direction,
            fisherValue: latestFisher,
            fisherSignalValue: latestFisherSignal,
            prevFisherValue: prevFisher,
            prevFisherSignalValue: prevFisherSignal,
            adx: latestAdx.adx,
            pdi: latestAdx.pdi, // Positive Directional Indicator
            mdi: latestAdx.mdi, // Negative Directional Indicator
            atrRange: latestAtrRange, // ATR for range filter
            atrSuperTrend: latestAtrST, // ATR used in SuperTrend calc (also for SL/TSL)
            price: latestClosePrice
        };
    } catch (error) {
        logger.error(`Error during indicator divination: ${error.message}`);
        console.error(error); // Log stack trace for debugging
        return null;
    }
}

// --- Custom SuperTrend Spell ---

/**
 * Calculates SuperTrend based on provided basis, ATR, and multiplier.
 * @param {object} input - Input data for SuperTrend calculation.
 * @param {number[]} input.high - Array of high prices.
 * @param {number[]} input.low - Array of low prices.
 * @param {number[]} input.close - Array of close prices.
 * @param {number[]} input.atr - Array of ATR values.
 * @param {number[]} input.basis - Array of basis values (e.g., smoothed roofed price).
 * @param {number} input.multiplier - ATR multiplier.
 * @returns {Array<{value: number, direction: string}>} Array of SuperTrend objects with value and direction.
 */
function calculateCustomSuperTrend(input) {
    const { high, low, close, atr, basis, multiplier } = input;
    let trend = [];
    let direction = 'up'; // Initial assumption
    let stValue = NaN;

    for (let i = 0; i < basis.length; i++) {
        // Skip if essential inputs are NaN for the current candle
        if (isNaN(basis[i]) || isNaN(atr[i]) || isNaN(close[i])) {
            trend.push({ value: NaN, direction: direction });
            continue;
        }

        const upperBandBasic = basis[i] + multiplier * atr[i];
        const lowerBandBasic = basis[i] - multiplier * atr[i];

        let prevStValue = (i > 0 && trend[i - 1] && !isNaN(trend[i - 1].value)) ? trend[i - 1].value : NaN;
        let prevDirection = (i > 0 && trend[i - 1]) ? trend[i - 1].direction : direction;

        // Determine current ST value based on previous direction and bands
        if (prevDirection === 'up') {
            stValue = Math.max(prevStValue, lowerBandBasic); // Trail stop up using the lower band
            if (close[i] < stValue) { // Price crossed below trailing stop
                direction = 'down';
                stValue = upperBandBasic; // Start new down-trend stop at upper band
            }
        } else { // direction === 'down'
            stValue = Math.min(prevStValue, upperBandBasic); // Trail stop down using the upper band
            if (close[i] > stValue) { // Price crossed above trailing stop
                direction = 'up';
                stValue = lowerBandBasic; // Start new up-trend stop at lower band
            }
        }

        trend.push({ value: stValue, direction: direction });
    }
    return trend;
}


// --- The Art of Trading (Core Logic) ---

/**
 * Checks entry/exit conditions based on indicators and manages orders.
 * @param {object} market - The market object from ccxt.
 * @param {object} indicators - The calculated indicator values.
 */
async function checkAndPlaceOrder(market, indicators) {
    const {
        superTrendValue, superTrendDirection,
        fisherValue, fisherSignalValue, prevFisherValue, prevFisherSignalValue,
        adx, atrRange, atrSuperTrend, price
    } = indicators;

    // --- Filters ---
    const adxFilterPassed = adx >= parseFloat(config.MIN_ADX_LEVEL) && adx <= parseFloat(config.MAX_ADX_LEVEL);
    const atrValuePercent = (atrRange / price) * 100;
    const atrFilterPassed = atrValuePercent >= parseFloat(config.MIN_ATR_PERCENTAGE);

    let filterReason = '';
    if (!adxFilterPassed) filterReason += `ADX (${adx.toFixed(2)}) outside [${config.MIN_ADX_LEVEL}-${config.MAX_ADX_LEVEL}]`;
    if (!atrFilterPassed) filterReason += `${filterReason ? '; ' : ''}ATR (${atrValuePercent.toFixed(3)}%) < ${config.MIN_ATR_PERCENTAGE}%`;

    if (!adxFilterPassed || !atrFilterPassed) {
        logger.info(`Filters block action: ${filterReason}.`);
        // Reset lastSignal if filters are blocking potential entry
        if (!state.positionSide && state.lastSignal) {
             logger.debug(`Clearing lastSignal (${state.lastSignal}) due to filter block.`);
             state.lastSignal = null;
        }
        return; // Stop processing if filters fail
    }
    logger.info(green(`Filters Passed: ADX=${adx.toFixed(2)}, ATR%=${atrValuePercent.toFixed(3)}%`));

    // --- Signal Conditions ---
    // Fisher Bullish Cross: Fisher line crosses above its signal line
    const fisherBullishCross = !isNaN(fisherValue) && !isNaN(fisherSignalValue) && !isNaN(prevFisherValue) && !isNaN(prevFisherSignalValue) &&
                               fisherValue > fisherSignalValue && prevFisherValue <= prevFisherSignalValue;
    // Fisher Bearish Cross: Fisher line crosses below its signal line
    const fisherBearishCross = !isNaN(fisherValue) && !isNaN(fisherSignalValue) && !isNaN(prevFisherValue) && !isNaN(prevFisherSignalValue) &&
                               fisherValue < fisherSignalValue && prevFisherValue >= prevFisherSignalValue;

    // Entry Conditions: SuperTrend direction aligned with Fisher cross
    const longCondition = superTrendDirection === 'up' && fisherBullishCross;
    const shortCondition = superTrendDirection === 'down' && fisherBearishCross;

    // Exit Conditions: SuperTrend flips direction (primary exit signal)
    const closeLongCondition = superTrendDirection === 'down'; // Exit long if ST turns down
    const closeShortCondition = superTrendDirection === 'up'; // Exit short if ST turns up

    // Logging current signal status
    logger.info(`Signals: Px=${magenta(price.toFixed(market.precision.price))}, ST=${magenta(superTrendValue.toFixed(market.precision.price))} ${superTrendDirection === 'up' ? green('Up') : red('Down')}`);
    logger.info(`Fisher: Val=${magenta(fisherValue.toFixed(3))}, Sig=${magenta(fisherSignalValue.toFixed(3))} | Cross: ${fisherBullishCross ? green('BULL') : fisherBearishCross ? red('BEAR') : gray('NONE')}`);

    const currentSide = state.positionSide;
    // Calculate order amount based on fixed USD value and current price
    const orderAmount = parseFloat((parseFloat(config.ORDER_AMOUNT_USD) / price).toFixed(market.precision.amount));

    // --- Position Management ---

    // 1. Check for Exits based on SuperTrend flip first
    if (currentSide === 'long' && closeLongCondition) {
        logger.info('SuperTrend flipped DOWN. Closing LONG.');
        await closePosition(market, 'sell', `SuperTrend flipped DOWN`);
        return; // Exit processed, end cycle check here
    } else if (currentSide === 'short' && closeShortCondition) {
        logger.info('SuperTrend flipped UP. Closing SHORT.');
        await closePosition(market, 'buy', `SuperTrend flipped UP`);
        return; // Exit processed, end cycle check here
    }

    // 2. Check for Entries if flat
    if (!state.positionSide) { // If flat
        if (longCondition && state.lastSignal !== 'long') {
            await openPosition(market, 'buy', orderAmount, price, atrSuperTrend);
            state.lastSignal = 'long'; // Set last signal to prevent immediate re-entry if conditions flicker
        } else if (shortCondition && state.lastSignal !== 'short') {
            await openPosition(market, 'sell', orderAmount, price, atrSuperTrend);
            state.lastSignal = 'short'; // Set last signal
        } else {
             // If no entry condition met, ensure lastSignal is cleared if it was previously set
             if (state.lastSignal) {
                 logger.debug(`No entry condition met. Clearing lastSignal (${state.lastSignal}).`);
                 state.lastSignal = null;
             }
        }
    } else {
        // If already in a position, ensure lastSignal is cleared (it's only for entry filtering)
        if (state.lastSignal) {
            logger.debug(`In position (${state.positionSide}). Clearing lastSignal (${state.lastSignal}).`);
            state.lastSignal = null;
        }
        // TSL update happens elsewhere (in runBot loop after indicator calculation)
    }
}


// --- Opening a New Position ---

/**
 * Opens a new long or short position.
 * @param {object} market - The market object from ccxt.
 * @param {'buy' | 'sell'} side - The side of the order ('buy' for long, 'sell' for short).
 * @param {number} amount - The amount of the asset to trade.
 * @param {number} entryPrice - The approximate current price for logging and SL calculation.
 * @param {number} atrValue - The current ATR value (from SuperTrend calc) for initial SL calculation.
 */
async function openPosition(market, side, amount, entryPrice, atrValue) {
    const positionSide = side === 'buy' ? 'long' : 'short';
    logger.trade(`${bold(positionSide.toUpperCase())} signal confirmed by Fisher. Attempting entry.`);

    // Calculate Initial Stop Loss based on ATR
    const slMultiplier = parseFloat(config.INITIAL_SL_ATR_MULTIPLIER);
    let initialSlPrice;
    if (positionSide === 'long') {
        initialSlPrice = entryPrice - slMultiplier * atrValue;
    } else { // short
        initialSlPrice = entryPrice + slMultiplier * atrValue;
    }
    initialSlPrice = parseFloat(exchange.priceToPrecision(config.SYMBOL, initialSlPrice));

    // Prepare order parameters with stop loss
    const params = {
        stopLoss: {
            triggerPrice: initialSlPrice
            // type: 'market' // Bybit default is market SL trigger
        }
        // takeProfit: {} // Could add TP logic here if needed
    };

    logger.info(`Placing ${positionSide.toUpperCase()} order: Amount=${amount.toFixed(market.precision.amount)}, Entry~=${entryPrice.toFixed(market.precision.price)}, Initial SL=${initialSlPrice} (ATR_ST=${atrValue.toFixed(4)})`);

    if (!config.DRY_RUN) {
        try {
            // Use createMarketOrderWithCost for Bybit to specify cost in quote currency (USD)
            const orderResponse = await exchange.createMarketOrderWithCost(config.SYMBOL, side, parseFloat(config.ORDER_AMOUNT_USD), undefined, params);

            logger.info(`Market order sent: ID ${orderResponse.id}, Side: ${orderResponse.side}, Cost: ${orderResponse.cost}`);
            logger.info("Waiting briefly for position confirmation...");
            await sleep(5000); // Wait 5 seconds for the position to likely appear

            // Verify position opened correctly
            const position = await fetchCurrentPosition(market.id);
            if (position && position.side === positionSide && parseFloat(position.contracts) > 0) {
                // Update state with actual position details
                state.positionSide = positionSide;
                state.entryPrice = parseFloat(position.entryPrice);
                state.positionAmount = parseFloat(position.contracts); // Use actual contracts amount
                logger.info(`Position Confirmed: Side=${green(state.positionSide.toUpperCase())}, Entry=${magenta(state.entryPrice)}, Amount=${state.positionAmount}`);
                logger.info("Setting initial TSL ward...");
                // Set the initial TSL based on the *initial* SL price. The update function will then trail it.
                // Note: Bybit's createOrder SL might create an actual SL order. We manage our TSL separately.
                // We will immediately try to set our own TSL based on the current price and ATR.
                await updateTrailingStopLoss(market, atrValue); // Initialize TSL
            } else {
                logger.error(`Failed confirmation for ${positionSide} ${config.SYMBOL} entry. Position Data: ${JSON.stringify(position)}`);
                sendTermuxSms(`BOT ALERT: Failed confirmation ${positionSide} ${config.SYMBOL} entry.`);
                state.positionSide = null; // Reset state as entry failed
            }
        } catch (e) {
            logger.error(`Error placing ${positionSide} order: ${e.message}`);
            console.error(e);
            sendTermuxSms(`BOT ERROR: Failed ${positionSide} order ${config.SYMBOL}. ${e.message}`);
            return; // Stop further processing on error
        }
    } else { // Dry Run
        state.positionSide = positionSide;
        state.entryPrice = entryPrice;
        state.positionAmount = amount;
        // Simulate the SL being set (we'll manage TSL simulation in updateTrailingStopLoss)
        state.currentTSL = { price: initialSlPrice, orderId: `dry_sl_${Date.now()}` }; // Simulate initial SL as TSL start
        logger.dryRun(`Simulated ${positionSide.toUpperCase()} entry: Amt=${amount.toFixed(market.precision.amount)}, Entry=${entryPrice.toFixed(market.precision.price)}, Initial SL=${initialSlPrice}`);
        sendTermuxSms(`BOT DRY RUN: Simulated ${positionSide} entry ${config.SYMBOL} @ ${entryPrice.toFixed(market.precision.price)}`);
    }
    await saveState();
}


// --- Closing the Current Position ---

/**
 * Closes the currently open position.
 * @param {object} market - The market object from ccxt.
 * @param {'buy' | 'sell'} side - The side of the closing order ('buy' to close short, 'sell' to close long).
 * @param {string} reason - The reason for closing the position (for logging).
 */
async function closePosition(market, side, reason) {
    if (!state.positionSide || !state.positionAmount) {
        logger.warn("Close attempt, but no position in memory.");
        return;
    }

    const closeSide = state.positionSide === 'long' ? 'sell' : 'buy'; // Determine correct closing side
    const amount = state.positionAmount; // Use the stored position amount

    logger.trade(`Closing ${state.positionSide.toUpperCase()} position. Reason: ${reason}`);
    logger.info(`Placing CLOSE order: Side=${closeSide}, Amount=${amount}`);

    // Cancel existing TSL order before closing position
    if (state.currentTSL && state.currentTSL.orderId) {
        await cancelOrder(state.currentTSL.orderId, 'TSL before closing');
        state.currentTSL = null; // Clear TSL state immediately
    }

    if (!config.DRY_RUN) {
        try {
            // Place a market order to close the position
            const params = { reduceOnly: true }; // Ensure it only reduces/closes the position
            const orderResponse = await exchange.createMarketOrder(config.SYMBOL, closeSide, amount, undefined, params);
            logger.info(`Close order sent: ID ${orderResponse.id}, Side: ${orderResponse.side}, Amount: ${orderResponse.amount}`);
            sendTermuxSms(`BOT TRADE: Closed ${state.positionSide} ${config.SYMBOL}. Reason: ${reason}`);
        } catch (e) {
            logger.error(`Error closing ${state.positionSide} position: ${e.message}`);
            console.error(e);
            // Attempt to send notification even on error
            sendTermuxSms(`BOT ERROR: Failed closing ${state.positionSide} ${config.SYMBOL}. ${e.message}`);
            // Don't reset state here, as the position might still be open. State reconciliation should handle it.
            return;
        }
    } else { // Dry Run
        logger.dryRun(`Simulated CLOSE ${state.positionSide.toUpperCase()}: Side=${closeSide}, Amt=${amount}. Reason: ${reason}`);
        sendTermuxSms(`BOT DRY RUN: Simulated closing ${state.positionSide} ${config.SYMBOL}. Reason: ${reason}`);
    }

    // Reset state after successful close (or simulated close)
    logger.info(`Position ${state.positionSide.toUpperCase()} closed. Resetting state.`);
    state.positionSide = null;
    state.entryPrice = null;
    state.positionAmount = null;
    state.currentTSL = null; // Ensure TSL is cleared
    state.lastSignal = null; // Clear last signal after closing
    await saveState();
}


// --- The Guardian Ward (Trailing Stop Loss) ---

/**
 * Updates the Trailing Stop Loss (TSL) order based on current price and ATR.
 * @param {object} market - The market object from ccxt.
 * @param {number} atrValue - The current ATR value (from SuperTrend calc) to use for TSL calculation.
 */
async function updateTrailingStopLoss(market, atrValue) {
    if (!state.positionSide || !state.entryPrice || !state.positionAmount) {
        logger.debug("TSL ward sleeps (no position).");
        return;
    }
    if (isNaN(atrValue) || atrValue <= 0) {
        logger.warn(`Invalid ATR (${atrValue}) provided for TSL calculation. Skipping TSL update.`);
        return;
    }

    // Fetch the latest price for TSL calculation
    const candles = await safeFetchOHLCV(config.TIMEFRAME, 2); // Need last closed candle price
    if (!candles || candles.length < 1) {
        logger.warn("Could not fetch current price for TSL update.");
        return;
    }
    const currentPrice = candles[candles.length - 1][4]; // Use the close of the last candle

    const multiplier = parseFloat(config.TSL_ATR_MULTIPLIER);
    let potentialNewTslPrice;

    // Calculate potential new TSL price based on current price and ATR
    if (state.positionSide === 'long') {
        potentialNewTslPrice = currentPrice - atrValue * multiplier;
    } else { // short
        potentialNewTslPrice = currentPrice + atrValue * multiplier;
    }

    let newTslPrice;
    // Determine the actual new TSL price: must be better (higher for long, lower for short) than the current TSL
    if (state.positionSide === 'long') {
        // If TSL exists, new price must be higher than current TSL price. Otherwise, use potential price.
        newTslPrice = state.currentTSL?.price ? Math.max(state.currentTSL.price, potentialNewTslPrice) : potentialNewTslPrice;
    } else { // short
        // If TSL exists, new price must be lower than current TSL price. Otherwise, use potential price.
        newTslPrice = state.currentTSL?.price ? Math.min(state.currentTSL.price, potentialNewTslPrice) : potentialNewTslPrice;
    }

    // Ensure price is formatted correctly for the exchange
    newTslPrice = parseFloat(exchange.priceToPrecision(config.SYMBOL, newTslPrice));

    // Check if TSL price actually needs updating
    if (state.currentTSL && state.currentTSL.price === newTslPrice) {
        logger.debug(`TSL price (${newTslPrice}) unchanged.`);
        return;
    }

    // Sanity check: Ensure TSL doesn't move backwards (should be covered by Math.max/min logic, but belt-and-suspenders)
    if (state.currentTSL) {
        if (state.positionSide === 'long' && newTslPrice < state.currentTSL.price) {
            logger.warn(`TSL Calculation Error (Long): New ${newTslPrice} < Current ${state.currentTSL.price}. Skipping update.`);
            return;
        }
        if (state.positionSide === 'short' && newTslPrice > state.currentTSL.price) {
            logger.warn(`TSL Calculation Error (Short): New ${newTslPrice} > Current ${state.currentTSL.price}. Skipping update.`);
            return;
        }
    }

    // Prevent setting TSL too close to current price (e.g., less than 0.25 * ATR away)
    const minDistance = atrValue * 0.25; // Minimum distance threshold
    if (state.positionSide === 'long' && (currentPrice - newTslPrice) < minDistance) {
        logger.info(`New TSL ${newTslPrice} too close to current price ${currentPrice} (Dist: ${(currentPrice - newTslPrice).toFixed(4)}, Min: ${minDistance.toFixed(4)}). Holding current TSL.`);
        return;
    }
    if (state.positionSide === 'short' && (newTslPrice - currentPrice) < minDistance) {
        logger.info(`New TSL ${newTslPrice} too close to current price ${currentPrice} (Dist: ${(newTslPrice - currentPrice).toFixed(4)}, Min: ${minDistance.toFixed(4)}). Holding current TSL.`);
        return;
    }


    logger.info(`Updating TSL ward for ${state.positionSide.toUpperCase()}.`);
    logger.info(`Current Px=${currentPrice.toFixed(market.precision.price)}, ATR_ST=${atrValue.toFixed(4)}, New Trigger=${magenta(newTslPrice.toFixed(market.precision.price))}`);

    // Cancel the previous TSL order if it exists
    if (state.currentTSL && state.currentTSL.orderId) {
        await cancelOrder(state.currentTSL.orderId, 'existing TSL');
    }

    // Place the new TSL order
    const tslOrderSide = state.positionSide === 'long' ? 'sell' : 'buy'; // Opposite side to close position
    // Bybit uses createOrder with stopPrice for stop-market orders
    const params = {
        stopPrice: newTslPrice, // The trigger price for the stop order
        reduceOnly: true,       // Ensure it only closes the position
        closeOnTrigger: true    // Often redundant with reduceOnly, but explicit
        // basePrice: currentPrice // Sometimes needed for Bybit stop orders to calculate trigger distance
    };

    if (!config.DRY_RUN) {
        try {
            // Create a STOP_MARKET order
            const newTslOrderResponse = await exchange.createOrder(
                config.SYMBOL,
                'market', // Order type becomes market when triggered
                tslOrderSide,
                state.positionAmount,
                undefined, // Price is not needed for market order
                params      // Contains stopPrice, reduceOnly, etc.
            );
            logger.info(`Placed new TSL ward: ID ${newTslOrderResponse.id}, Trigger=${newTslPrice}`);
            // Update state with the new TSL price and order ID
            state.currentTSL = { price: newTslPrice, orderId: newTslOrderResponse.id };
        } catch (e) {
            logger.error(`Error placing TSL ward: ${e.message}`);
            console.error(e);
            state.currentTSL = null; // Clear TSL state on failure
            sendTermuxSms(`BOT CRITICAL: Failed TSL placement ${config.SYMBOL} @ ${newTslPrice}. ${e.message}`);
        }
    } else { // Dry Run
        const simulatedOrderId = `dry_tsl_${Date.now()}`;
        state.currentTSL = { price: newTslPrice, orderId: simulatedOrderId };
        logger.dryRun(`Simulated TSL ward: ID ${simulatedOrderId}, Trigger=${newTslPrice}`);
    }
    await saveState(); // Save state after TSL update attempt
}


// --- Banishing Orders (Cancellation) ---

/**
 * Cancels an order by its ID.
 * @param {string} orderId - The ID of the order to cancel.
 * @param {string} orderTypeLabel - A label for the order type (e.g., 'TSL') for logging.
 */
async function cancelOrder(orderId, orderTypeLabel) {
    // Don't try to cancel simulated dry run orders or null IDs
    if (!orderId || orderId.startsWith('dry_')) {
        logger.debug(`Skipping cancel ${orderTypeLabel}: No real ID or dry run ID (${orderId})`);
        return;
    }

    logger.info(`Attempting banishment of ${orderTypeLabel} order: ${orderId}`);
    if (!config.DRY_RUN) {
        try {
            await exchange.cancelOrder(orderId, config.SYMBOL);
            logger.info(`${orderTypeLabel} order ${orderId} banished.`);
        } catch (e) {
            // Handle cases where the order might have already been filled or cancelled
            if (e instanceof ccxt.OrderNotFound) {
                logger.warn(`${orderTypeLabel} order ${orderId} already vanished or filled.`);
            } else if (e instanceof ccxt.InvalidOrder) {
                 logger.warn(`${orderTypeLabel} order ${orderId} likely already closed/filled: ${e.message}`);
            } else {
                // Log other errors
                logger.error(`Error banishing ${orderTypeLabel} ${orderId}: ${e.message}`);
                console.error(e);
                sendTermuxSms(`BOT ALERT: Failed cancelling ${orderTypeLabel} ${orderId} ${config.SYMBOL}. ${e.message}`);
            }
        }
    } else { // Dry Run
        logger.dryRun(`Simulated banishment of ${orderTypeLabel} order: ${orderId}`);
    }
}

// --- Utility Charms ---

/**
 * Safely fetches OHLCV data with error handling.
 * @param {string} timeframe - The timeframe string (e.g., '1h', '15m').
 * @param {number} limit - The number of candles to fetch.
 * @returns {Promise<Array<Array<number>>|null>} OHLCV data or null on failure.
 */
async function safeFetchOHLCV(timeframe, limit) {
    try {
        // Fetch a bit more than required to ensure indicator calculations have enough lead-in data
        const fetchLimit = limit + 50; // Add buffer for indicator warm-up
        const candles = await exchange.fetchOHLCV(config.SYMBOL, timeframe, undefined, fetchLimit);
        if (!candles || candles.length < limit) {
            logger.warn(`Fetched insufficient candles (${candles?.length}) for ${config.SYMBOL} ${timeframe}, needed ${limit}. Might be a new market or connection issue.`);
            return null;
        }
        // Return only the required number of most recent candles
        return candles.slice(-limit);
    } catch (e) {
        logger.error(`Error fetching OHLCV for ${config.SYMBOL} ${timeframe}: ${e.message}`);
        if (e instanceof ccxt.RateLimitExceeded) {
            logger.warn("Rate limit exceeded. Pausing briefly...");
            await sleep(10000); // Wait longer for rate limits
        } else if (e instanceof ccxt.NetworkError) {
             logger.warn(`Network error fetching candles: ${e.message}. Retrying next cycle.`);
        } else {
            console.error(e); // Log other errors fully
        }
        return null;
    }
}

/**
 * Fetches the current position for the configured symbol.
 * Handles dry run simulation.
 * @param {string} symbolId - The exchange-specific symbol ID.
 * @returns {Promise<object|null>} The position object (ccxt format) or null if no position.
 */
async function fetchCurrentPosition(symbolId) {
    if (config.DRY_RUN) {
        // Simulate fetching position based on current state
        if (state.positionSide) {
            return {
                symbol: symbolId,
                side: state.positionSide,
                entryPrice: state.entryPrice,
                contracts: state.positionAmount, // 'contracts' is common field name
                // Add other fields ccxt might return if needed by reconciliation logic
                leverage: parseInt(config.LEVERAGE),
                unrealizedPnl: 0, // Mock PnL
                initialMargin: 0, // Mock margin
                maintMargin: 0,   // Mock margin
            };
        }
        return null; // No position in state
    }

    // Live mode: Fetch actual position
    try {
        const positions = await exchange.fetchPositions([symbolId]);
        // Find the position matching the symbol that has a non-zero size
        const position = positions.find(p =>
            p.symbol === symbolId &&
            p.contracts !== undefined && parseFloat(p.contracts) !== 0 &&
            p.entryPrice !== undefined && p.entryPrice !== null
        );

        if (currentLogLevel <= LOG_LEVELS.DEBUG) {
            logger.debug(`Fetched position ${symbolId}: ${position ? `${position.side} ${position.contracts} @ ${position.entryPrice}` : 'None'}`);
        }
        return position || null; // Return found position or null
    } catch (e) {
        logger.error(`Error fetching position for ${symbolId}: ${e.message}`);
        console.error(e);
        // Decide how to handle this - returning null might cause incorrect reconciliation
        // Maybe throw or return a specific error state? For now, return null.
        return null;
    }
}

/**
 * Pauses execution for a specified duration.
 * @param {number} ms - Milliseconds to sleep.
 * @returns {Promise<void>}
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Sets the leverage for the symbol.
 * @param {object} market - The market object from ccxt.
 * @returns {Promise<void>}
 */
async function setLeverage(market) {
     const leverageValue = parseInt(config.LEVERAGE);
     if (isNaN(leverageValue) || leverageValue <= 0) {
         logger.warn("Leverage configuration is invalid or missing. Skipping leverage setting.");
         return;
     }

     if (config.DRY_RUN) {
         logger.dryRun(`Simulated setting leverage to ${leverageValue}x for ${config.SYMBOL}`);
         return;
     }

     try {
         logger.info(`Attempting to set leverage for ${config.SYMBOL} to ${leverageValue}x`);
         await exchange.setLeverage(leverageValue, config.SYMBOL);
         logger.info(`Leverage for ${config.SYMBOL} confirmed set to ${leverageValue}x.`);
     } catch (e) {
         // Check if the error message indicates leverage was already set
         if (e.message.includes('leverage not modified') || e.message.includes('Leverage is not changed')) {
             logger.warn(`Leverage for ${config.SYMBOL} already set to ${leverageValue}x.`);
         } else {
             logger.error(`Failed to set leverage for ${config.SYMBOL} to ${leverageValue}x: ${e.message}`);
             console.error(e);
             // This might be critical, consider halting if leverage can't be set.
             throw new Error(`Could not set leverage for ${config.SYMBOL}. Halting initialization.`);
         }
     }
}

/**
 * Validates the configuration loaded from .env.
 * Logs errors and exits if critical configuration is missing or invalid.
 */
function validateConfig() {
    const requiredKeys = ['BYBIT_API_KEY', 'BYBIT_API_SECRET', 'SYMBOL', 'TIMEFRAME', 'ORDER_AMOUNT_USD', 'LEVERAGE'];
    const numericKeys = [
        'LEVERAGE', 'ORDER_AMOUNT_USD', 'ROOF_FAST_EMA', 'ROOF_SLOW_EMA',
        'ST_EMA_PERIOD', 'ST_ATR_PERIOD', 'ST_MULTIPLIER', 'FISHER_PERIOD',
        'ADX_PERIOD', 'MIN_ADX_LEVEL', 'MAX_ADX_LEVEL', 'RANGE_ATR_PERIOD',
        'MIN_ATR_PERCENTAGE', 'TSL_ATR_MULTIPLIER', 'INITIAL_SL_ATR_MULTIPLIER',
        'POLL_INTERVAL_MS'
    ];
    let errors = [];

    // Check required keys
    for (const key of requiredKeys) {
        if (!config[key]) {
            errors.push(`Missing required environment variable: ${key}`);
        }
    }

    // Check numeric keys
    for (const key of numericKeys) {
        if (config[key] !== undefined) { // Only check if key exists
            const value = Number(config[key]);
            if (isNaN(value)) {
                errors.push(`Environment variable ${key} must be a valid number, but received: "${config[key]}"`);
            } else {
                // Add specific range checks if necessary (e.g., periods > 0)
                if ((key.includes('PERIOD') || key.includes('EMA')) && value <= 0) {
                     errors.push(`${key} must be greater than 0.`);
                }
                if (key === 'LEVERAGE' && value <= 0) {
                    errors.push(`LEVERAGE must be greater than 0.`);
                }
                 if (key === 'ORDER_AMOUNT_USD' && value <= 0) {
                    errors.push(`ORDER_AMOUNT_USD must be greater than 0.`);
                }
                if ((key.includes('MULTIPLIER') || key.includes('PERCENTAGE')) && value < 0) {
                     errors.push(`${key} cannot be negative.`);
                }
            }
        } else if (requiredKeys.includes(key)) {
             // This case is already covered by the required keys check, but good to be explicit
             // errors.push(`Missing required numeric environment variable: ${key}`);
        }
    }

    // Check specific logic
    if (Number(config.ROOF_FAST_EMA) >= Number(config.ROOF_SLOW_EMA)) {
         errors.push("ROOF_FAST_EMA must be strictly less than ROOF_SLOW_EMA.");
    }
     if (Number(config.MIN_ADX_LEVEL) >= Number(config.MAX_ADX_LEVEL)) {
         errors.push("MIN_ADX_LEVEL must be less than MAX_ADX_LEVEL.");
    }

    // Check Termux config if enabled
    if (config.TERMUX_NOTIFY && !config.NOTIFICATION_PHONE_NUMBER) {
        errors.push("TERMUX_NOTIFY is true, but NOTIFICATION_PHONE_NUMBER is missing.");
    }

    if (errors.length > 0) {
        logger.error(red(bold("Configuration Errors Found:")));
        errors.forEach(err => logger.error(red(`- ${err}`)));
        logger.error(red(bold("Please check your .env file. Halting bot.")));
        process.exit(1); // Exit if config is invalid
    }

    logger.info("Configuration validated successfully.");
}


// --- The Grand Ritual Loop ---

/**
 * The main execution loop of the trading bot.
 * Fetches data, calculates indicators, checks state, manages positions and TSL.
 */
async function runBot() {
    state.cycleCount++;
    logger.info(`\n${cyan(bold(`----- Ritual Cycle ${state.cycleCount} Started (${new Date().toISOString()}) -----`))}`);

    try {
        // --- Load State & Market Data ---
        await loadState(); // Load latest state from file
        const markets = await exchange.loadMarkets(); // Ensure markets are loaded
        const market = exchange.market(config.SYMBOL);
        if (!market || !market.linear) { // Check if market exists and is linear (for USDT/USD margined)
            logger.error(red(bold(`Market ${config.SYMBOL} is invalid or not a Linear contract!`)));
            logger.error(red(bold("Halting cycle. Please check SYMBOL configuration.")));
            // Consider stopping the bot entirely here if market is fundamentally wrong
            return;
        }
        logger.debug(`Market details loaded for ${config.SYMBOL}. Precision: Price=${market.precision.price}, Amount=${market.precision.amount}`);

        // --- State Synchronization ---
        const livePosition = await fetchCurrentPosition(market.id);

        // Scenario 1: State has position, but exchange reports flat.
        if (state.positionSide && !livePosition) {
            logger.warn(yellow(`State/Reality Mismatch: Memory had ${state.positionSide} position, but exchange reports FLAT.`));
            logger.warn(yellow(`This could be due to manual closure, TSL/SL execution, or liquidation.`));
            sendTermuxSms(`BOT ALERT: Position mismatch ${config.SYMBOL}. Bot thought ${state.positionSide}, exchange FLAT. State reset.`);
            // Cancel any potentially dangling TSL order associated with the non-existent position
            if (state.currentTSL && state.currentTSL.orderId) {
                await cancelOrder(state.currentTSL.orderId, 'dangling TSL from mismatch');
            }
            // Reset state to flat
            logger.warn("Resetting internal state to FLAT.");
            state = { ...state, positionSide: null, entryPrice: null, positionAmount: null, currentTSL: null, lastSignal: null };
            await saveState();
        }
        // Scenario 2: State is flat, but exchange reports an open position. CRITICAL MISMATCH.
        else if (!state.positionSide && livePosition) {
            logger.error(red(bold(`CRITICAL STATE MISMATCH: Bot memory is FLAT, but exchange reports OPEN ${livePosition.side} ${config.SYMBOL}!`)));
            logger.error(red(bold(` -> Exchange Position Details: Entry=${livePosition.entryPrice}, Size=${livePosition.contracts}`)));
            logger.error(red(bold(" -> This indicates a potential major issue or external interference.")));
            logger.error(red(bold(" -> Manual intervention REQUIRED. Halting bot cycles to prevent conflicts.")));
            sendTermuxSms(`BOT CRITICAL: State mismatch ${config.SYMBOL}. Exchange OPEN ${livePosition.side}, bot FLAT. Manual check REQUIRED! Bot halted.`);
            // Halt the bot loop by not scheduling the next run
            // Consider process.exit(1) for immediate stop if preferred
            return; // Stop further cycles
        }
        // Scenario 3: State and exchange both report a position. Check for consistency.
        else if (state.positionSide && livePosition) {
            const stateAmount = parseFloat(state.positionAmount);
            const liveAmount = parseFloat(livePosition.contracts);
            // Use a small tolerance for amount comparison due to potential floating point inaccuracies
            const amountDiff = Math.abs(stateAmount - liveAmount);
            const tolerance = stateAmount * 0.005; // 0.5% tolerance, adjust as needed

            if (state.positionSide !== livePosition.side || amountDiff > tolerance) {
                logger.warn(yellow(`State/Reality Drift Detected: Updating bot memory from live exchange data.`));
                logger.warn(yellow(` -> Memory: ${state.positionSide} ${stateAmount.toFixed(market.precision.amount)} @ ${state.entryPrice?.toFixed(market.precision.price)}`));
                logger.warn(yellow(` -> Reality: ${livePosition.side} ${liveAmount.toFixed(market.precision.amount)} @ ${livePosition.entryPrice?.toFixed(market.precision.price)}`));

                // Update state with live data
                state.positionSide = livePosition.side;
                state.entryPrice = parseFloat(livePosition.entryPrice);
                state.positionAmount = liveAmount;

                logger.info('Re-evaluating TSL due to state reconciliation.');
                // Cancel potentially incorrect TSL and recalculate/replace it
                if (state.currentTSL && state.currentTSL.orderId) {
                    await cancelOrder(state.currentTSL.orderId, 'TSL during reconciliation');
                    state.currentTSL = null; // Clear TSL state before potential update
                }
                // Fetch fresh ATR to update TSL correctly based on reconciled state
                const freshCandlesForTsl = await safeFetchOHLCV(config.TIMEFRAME, Math.max(parseInt(config.ST_ATR_PERIOD), 50)); // Fetch enough for ATR calc
                 if (freshCandlesForTsl) {
                     const high = freshCandlesForTsl.map(c => c[2]);
                     const low = freshCandlesForTsl.map(c => c[3]);
                     const close = freshCandlesForTsl.map(c => c[4]);
                     const atrValues = ATR.calculate({ period: parseInt(config.ST_ATR_PERIOD), high, low, close });
                     const latestAtrForTsl = atrValues.length > 0 ? atrValues[atrValues.length - 1] : NaN;
                     if (!isNaN(latestAtrForTsl)) {
                         await updateTrailingStopLoss(market, latestAtrForTsl);
                     } else {
                         logger.warn("Could not calculate fresh ATR for TSL reconciliation. TSL might be missing.");
                     }
                 } else {
                     logger.warn("Could not fetch fresh candles for TSL reconciliation. TSL might be missing.");
                 }
                 await saveState(); // Save reconciled state
            } else {
                // State and reality match for the open position
                logger.debug("Memory/reality position match verified.");
            }
        }
        // Scenario 4: State and exchange both report flat.
        else { // !state.positionSide && !livePosition
            logger.debug("Memory/reality align: Bot is FLAT and exchange confirms no position.");
        }

        // --- Gather Ingredients & Perform Divination ---
        // Determine candle limit needed based on longest indicator period + buffer
        const candleLimit = Math.max(
             parseInt(config.ROOF_SLOW_EMA),
             parseInt(config.ST_EMA_PERIOD) + parseInt(config.ST_ATR_PERIOD), // SuperTrend lookback
             parseInt(config.FISHER_PERIOD),
             parseInt(config.ADX_PERIOD),
             parseInt(config.RANGE_ATR_PERIOD)
        ) + 100; // Add generous buffer for warm-up and potential gaps
        logger.debug(`Fetching ${candleLimit} candles for ${config.TIMEFRAME}...`);
        const candles = await safeFetchOHLCV(config.TIMEFRAME, candleLimit);
        if (!candles) {
            logger.warn("No candle data fetched. Skipping strategy execution this cycle.");
            return; // Skip rest of the cycle if no data
        }

        logger.debug("Calculating indicators...");
        const indicators = await calculateIndicators(candles);
        if (!indicators) {
            logger.warn("Indicator calculation failed. Skipping strategy execution this cycle.");
            return; // Skip rest of the cycle if indicators fail
        }

        // --- Log Current State & Indicators ---
        const positionStatus = state.positionSide
            ? `${state.positionSide === 'long' ? green(state.positionSide.toUpperCase()) : red(state.positionSide.toUpperCase())} (Entry: ${magenta(state.entryPrice?.toFixed(market.precision.price))}, Amt: ${state.positionAmount?.toFixed(market.precision.amount)})`
            : bold('FLAT');
        logger.info(`Position: ${positionStatus}`);
        logger.info(`Indicators: Px=${magenta(indicators.price.toFixed(market.precision.price))}, ST=${magenta(indicators.superTrendValue.toFixed(market.precision.price))} ${indicators.superTrendDirection === 'up' ? green('Up') : red('Down')}, ADX=${magenta(indicators.adx.toFixed(2))}`);
        logger.info(`Fisher: Val=${magenta(indicators.fisherValue.toFixed(3))}, Sig=${magenta(indicators.fisherSignalValue.toFixed(3))}, ATR_ST=${magenta(indicators.atrSuperTrend.toFixed(4))}`); // Using ATR_ST for TSL
        if (state.currentTSL) {
            logger.info(`Active TSL: Trigger=${magenta(state.currentTSL.price.toFixed(market.precision.price))}, ID=${gray(state.currentTSL.orderId || 'N/A')}`);
        }

        // --- Enact Strategy ---
        if (state.positionSide) {
            // If in a position, primarily manage the exit (TSL and ST flip)
            logger.debug("Position active. Updating TSL & checking for SuperTrend exit signal...");
            // Update TSL first based on the latest ATR
            await updateTrailingStopLoss(market, indicators.atrSuperTrend); // Use the ATR calculated for ST
            // Check if SuperTrend has flipped against the position
            await checkAndPlaceOrder(market, indicators); // This will trigger closePosition if ST flips
        } else {
            // If flat, check for entry signals
            logger.debug("Position flat. Scanning for entry (SuperTrend trend + Fisher cross)...");
            await checkAndPlaceOrder(market, indicators); // This will trigger openPosition if entry conditions met
        }

        await saveState(); // Save state at the end of a successful cycle part
        logger.info(cyan(bold(`----- Ritual Cycle ${state.cycleCount} Completed -----`)));

    } catch (e) {
        // --- Global Error Handling for the Cycle ---
        logger.error(red(bold(`Unhandled Exception during Cycle ${state.cycleCount}:`)));
        logger.error(red(e.message));
        console.error(e); // Log the full stack trace for debugging

        // Handle specific CCXT errors or general errors
        if (e instanceof ccxt.NetworkError) {
            logger.warn(`Network Error detected: ${e.message}. Bot will retry next cycle.`);
        } else if (e instanceof ccxt.ExchangeError) {
            // Includes RateLimitExceeded, AuthenticationError, InsufficientFunds, etc.
            logger.error(`Exchange Error encountered: ${e.message}.`);
            sendTermuxSms(`BOT ERROR: Exchange Err ${config.SYMBOL} Cycle ${state.cycleCount}. ${e.message}`);
            // Depending on the error, might need specific handling (e.g., pause on rate limit)
            if (e instanceof ccxt.RateLimitExceeded) {
                 logger.warn("Rate limit exceeded during cycle. Increasing pause before next cycle.");
                 await sleep(15000); // Longer pause after rate limit
            } else if (e instanceof ccxt.AuthenticationError) {
                logger.error(red(bold("CRITICAL: Authentication failed! Check API Keys. Halting bot.")));
                sendTermuxSms(`BOT CRITICAL: Auth Error ${config.SYMBOL}. Check API Keys! Bot halted.`);
                return; // Stop the bot loop
            }
        } else {
            // General unexpected errors
            logger.error("An unexpected critical error occurred.");
            sendTermuxSms(`BOT CRITICAL FAILURE Cycle ${state.cycleCount}. Check logs! ${e.message}`);
            // Consider halting for unknown errors: process.exit(1);
        }
        // Even after errors, try to save state if possible (might contain partial updates)
        await saveState();

    } finally {
        // --- Schedule Next Cycle ---
        const intervalSeconds = parseInt(config.POLL_INTERVAL_MS) / 1000;
        logger.debug(`Awaiting ${intervalSeconds}s until next cycle...`);
        // Use setTimeout to schedule the next run, ensuring loop continues even after errors (unless explicitly halted)
        setTimeout(runBot, parseInt(config.POLL_INTERVAL_MS));
    }
}


// --- The Awakening Ritual ---

/**
 * Initializes the bot, validates configuration, sets up the exchange connection, and starts the main loop.
 */
async function initialize() {
    console.log(cyan(bold("\nInitializing Pyrmethus Roofed/Fisher/ST/ADX Bot...")));
    console.log(cyan(bold("=======================================================")));

    try {
        // 1. Validate Configuration
        validateConfig(); // Exits if critical config is bad

        // 2. Log Configuration Summary
        logger.info(bold("--- Configuration Summary ---"));
        logger.info(` Exchange: ${cyan('Bybit (Linear)')}`);
        logger.info(` Locus (Symbol): ${cyan(config.SYMBOL)} | Timeframe: ${cyan(config.TIMEFRAME)}`);
        logger.info(` Cycle Interval: ${cyan(parseInt(config.POLL_INTERVAL_MS) / 1000)}s | Log Level: ${cyan(config.LOG_LEVEL)}`);
        logger.info(` Order Size (USD): ${cyan(config.ORDER_AMOUNT_USD)} | Leverage: ${cyan(config.LEVERAGE)}x`);
        logger.info(bold("--- Strategy Parameters ---"));
        logger.info(` Roofing Filter EMAs: ${cyan(config.ROOF_FAST_EMA)} / ${cyan(config.ROOF_SLOW_EMA)} (on HLC3)`);
        logger.info(` SuperTrend: Basis EMA(${cyan(config.ST_EMA_PERIOD)}) of Roofed Price, ATR(${cyan(config.ST_ATR_PERIOD)}), Multiplier(${cyan(config.ST_MULTIPLIER)})`);
        logger.info(` Confirmation: Fisher Transform (${cyan(config.FISHER_PERIOD)}) Crossover (on High/Low)`);
        logger.info(` Filters: ADX(${cyan(config.ADX_PERIOD)}) Range [${cyan(config.MIN_ADX_LEVEL)}-${cyan(config.MAX_ADX_LEVEL)}], ATR(${cyan(config.RANGE_ATR_PERIOD)}) Volatility >= ${cyan(config.MIN_ATR_PERCENTAGE)}%`);
        logger.info(` Risk Management: Initial SL (${cyan(config.INITIAL_SL_ATR_MULTIPLIER)}x ATR_ST), Trailing SL (${cyan(config.TSL_ATR_MULTIPLIER)}x ATR_ST)`);
        logger.info(bold("--- Settings ---"));
        if (config.DRY_RUN) logger.warn(magenta(bold(" DRY RUN MODE ACTIVE - No real trades will be executed.")));
        else logger.info(green(bold(" LIVE TRADING MODE ACTIVE")));
        if (config.TERMUX_NOTIFY) logger.info(` Termux SMS Notifications: ${green('Enabled')} for ${cyan(config.NOTIFICATION_PHONE_NUMBER)}`);
        else logger.info(" Termux SMS Notifications: Disabled");
        logger.info(` State Persistence File: ${cyan(config.PERSISTENCE_FILE)}`);
        console.log(cyan(bold("=======================================================")));

        // 3. Load Initial State
        await loadState();

        // 4. Connect to Exchange & Set Leverage
        logger.info("Connecting to exchange and loading market definitions...");
        await exchange.loadMarkets(); // Load markets to get details like precision
        const market = exchange.market(config.SYMBOL);
        if (!market) {
            throw new Error(`Market ${config.SYMBOL} not found on exchange.`);
        }
        await setLeverage(market); // Set leverage (includes dry run check)

    } catch (e) {
        logger.error(red(bold(`Fatal Initialization Error: ${e.message}`)));
        console.error(e);
        process.exit(1); // Stop the bot if initialization fails critically
    }

    // 5. Start the Main Loop
    logger.info(green(bold("Initialization complete. Starting the main ritual loop...")));
    runBot(); // Start the first cycle
}

// Begin the enchantment...
initialize();




SYMBOL=BCH/USDT:USDT
    LEVERAGE=5

    # --- Strategy Parameters ---
    TIMEFRAME=15m
    ORDER_AMOUNT_USD=50

    # --- Ehlers Roofing Filter (for ST Basis Smoothing) ---
    ROOF_FAST_EMA=10   # Fast EMA period for Roofing Filter
    ROOF_SLOW_EMA=40   # Slow EMA period for Roofing Filter (Must be > Fast)

    # --- Ehlers SuperTrend (Basis now uses Roofed Price) & Filters ---
    ST_EMA_PERIOD=10   # EMA Period for *Roofed Price* Smoothing (Basis)
    ST_ATR_PERIOD=10   # ATR Period for SuperTrend calculation (uses regular HLC)
    ST_MULTIPLIER=3    # SuperTrend Multiplier

    # --- Ehlers Fisher Transform ---
    FISHER_PERIOD=10

    # --- ADX Filter (Trend Strength) ---
    ADX_PERIOD=14
    MIN_ADX_LEVEL=20
    MAX_ADX_LEVEL=60

    # --- ATR Range Filter (Volatility) ---
    RANGE_ATR_PERIOD=14
    MIN_ATR_PERCENTAGE=0.2

    # --- Risk Management ---
    TSL_ATR_MULTIPLIER=1.5
    INITIAL_SL_ATR_MULTIPLIER=2.0

    # --- Bot Settings ---
    POLL_INTERVAL_MS=30000
    PERSISTENCE_FILE=trading_state_roof_fisher.json # New state file name
    LOG_LEVEL=INFO
    DRY_RUN=true
    TERMUX_NOTIFY=false
    NOTIFICATION_PHONE_NUMBER=+1234567890
#!/usr/bin/env node
// -*- coding: utf-8 -*-

/**
 * Enhanced Trading Bot using CCXT for Bybit Futures/Swaps (Linear Contracts) - Node.js Version.
 *
 * Strategy: Dual Enhanced Ehlers SuperTrend confirmation, Volume Spike filter, Order Book Pressure filter.
 * Enhancements:
 * - Enhanced Ehlers Gaussian Filtered Supertrend for improved signal quality.
 * - Robust input validation and error handling in indicator functions.
 * - Improved logging (async file logging using file handle, detailed messages) and error messages for better debugging.
 * - Clearer code structure and comments (JSDoc) for maintainability (indicators and exchange helpers extracted).
 * - Enhanced state management with validation on load and reliable atomic saving.
 * - More robust order placement logic using CCXT unified methods with specific Bybit V5 parameters and error handling (incl. conditional orders).
 * - Improved handling of CCXT exceptions and refined retry logic with exponential backoff.
 * - Consistent use of market precision for amounts and prices in calculations and order placement.
 * - Refined Dry Run simulation for orders, state updates, and balance.
 * - Robust Trailing Stop Loss (TSL) implementation by modifying the existing SL order (with Cancel & Replace fallback).
 * - Graceful shutdown procedure ensuring orders are cancelled and positions potentially closed.
 * - Retains all features from the base bot_st.js: Risk Management, ATR-based SL/TP/TSL,
 *   Configuration via .env, Termux SMS Notifications (Optional), API Call Retries,
 *   Persistent State Management, Dry Run Mode, Colorized Logging, Specific error handling,
 *   Data Caching, Bybit V5 API considerations.
 *
 * Original Supertrend Enhancement & Gaussian Filter logic adapted from: alternatest.txt & bot_stx.js
 * Base bot structure and features from: bot_st.js
 */

// --- Core Node.js Modules ---
const os = require('os');
const fs = require('fs').promises; // Use promise-based fs for async operations
const path = require('path');
const { exec, execSync } = require('child_process'); // For running shell commands (e.g., Termux SMS)
const { inspect } = require('util'); // For detailed object logging

// --- Third-party Libraries ---
const ccxt = require('ccxt');
const dotenv = require('dotenv');
const c = require('nanocolors'); // Import nanocolors (conventionally assigned to 'c')

// --- Constants ---

// Define standard sides for orders
const Side = Object.freeze({
    BUY: "buy",
    SELL: "sell"
});

// Define potential position states
const PositionSide = Object.freeze({
    LONG: "long",
    SHORT: "short",
    NONE: "none"
});

// Define the structure and indices for OHLCV data arrays returned by CCXT
const OHLCV_SCHEMA = Object.freeze(["timestamp", "open", "high", "low", "close", "volume"]);
const OHLCV_INDEX = Object.freeze(OHLCV_SCHEMA.reduce((acc, name, i) => {
    acc[name.toUpperCase()] = i; // Use uppercase keys for easier access
    return acc;
}, {}));

// Default configuration values used if not specified in .env
const DEFAULTS = Object.freeze({
    SYMBOL: "BTC/USDT:USDT",        // Default trading pair (Bybit linear swap format)
    TIMEFRAME: "1m",                // Default candlestick timeframe
    LEVERAGE: 10.0,                 // Default leverage
    RISK_PER_TRADE: 0.01,           // Default risk percentage (1%) -> 0.01 means 1% of balance
    SL_ATR_MULT: 1.5,               // Default Stop Loss ATR multiplier
    TP_ATR_MULT: 2.0,               // Default Take Profit ATR multiplier
    TRAILING_STOP_MULT: 1.5,        // Default Trailing Stop ATR multiplier (Set <= 0 to disable)
    SHORT_ST_PERIOD: 7,             // Default short SuperTrend period
    LONG_ST_PERIOD: 14,             // Default long SuperTrend period (also used for main ATR)
    ST_MULTIPLIER: 2.0,             // Default SuperTrend ATR multiplier
    GAUSSIAN_FILTER_LENGTH: 5,      // Default Gaussian filter length for Supertrend enhancement
    VOLUME_SPIKE_THRESHOLD: 1.5,    // Default volume ratio threshold (Short VMA / Long VMA)
    OB_PRESSURE_THRESHOLD: 0.6,     // Default order book buy pressure threshold (0.0 to 1.0) - 0.6 means need >= 60% buy volume in observed book depth
    LOGGING_LEVEL: "INFO",          // Default logging level (DEBUG, INFO, WARN, ERROR)
    MAX_RETRIES: 3,                 // Default max retries for API calls (0 means one attempt, no retries)
    RETRY_DELAY: 5,                 // Default initial delay (seconds) between retries (uses exponential backoff)
    CURRENCY: "USDT",               // Default quote currency for balance/risk
    EXCHANGE_TYPE: "swap",          // Default market type (swap/future/spot) - ensure matches SYMBOL
    ORDER_TRIGGER_PRICE_TYPE: "LastPrice", // Default trigger price for SL/TP (Bybit options: LastPrice, MarkPrice, IndexPrice)
    TIME_IN_FORCE: "GoodTillCancel",// Default Time-in-Force for conditional orders (GTC, IOC, FOK)
    VOLUME_SHORT_PERIOD: 5,         // Default short MA period for volume
    VOLUME_LONG_PERIOD: 20,         // Default long MA period for volume
    ORDER_BOOK_DEPTH: 10,           // Default depth for order book fetching
    CACHE_TTL: 30,                  // Default cache Time-To-Live (seconds) for frequently used API data
    STATE_FILE: "trading_bot_state_enhanced.json", // Default filename for persistent state
    LOG_FILE_ENABLED: "true",       // Default: enable file logging
    LOG_DIR: "logs",                // Default directory for log files
    DRY_RUN: "true",                // Default: dry run ENABLED (safer default)
    SMS_ENABLED: "false",           // Default: SMS disabled
    SMS_RECIPIENT_NUMBER: "",       // Default: Empty SMS recipient number
    CLOSE_POSITION_ON_SHUTDOWN: "true", // Default: Close position on shutdown
    BYBIT_ACCOUNT_TYPE: "CONTRACT", // Default: Bybit V5 account type (CONTRACT, UNIFIED, FUND)
});

// List of valid timeframes supported by CCXT/Bybit (adjust if needed)
const VALID_TIMEFRAMES = Object.freeze(["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"]);

// --- Global Variables ---
let botInstance = null; // Holds the main TradingBot instance for access by shutdown handlers
let isShuttingDown = false; // Flag to prevent multiple shutdown executions

// --- Logging Setup ---
// Simple logger using nanocolors for console and basic async file appending.
const logLevels = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
let currentLogLevel = logLevels.INFO; // Default level, updated by Config
let logFilePath = null; // Path to the current log file, null if disabled
let logFileHandle = null; // Async file handle for efficient appending

/**
 * Initializes logging based on configuration (sets level and file path/handle).
 * Should be called after Config is initialized.
 * @param {Config} config - The validated configuration object.
 * @returns {Promise<void>}
 */
async function setupLogging(config) {
    const levelName = config.logging_level.toUpperCase();
    currentLogLevel = logLevels[levelName] ?? logLevels.INFO; // Default to INFO if invalid
    if (logLevels[levelName] === undefined) {
        // Use console directly as logger might not be fully set up
        console.warn(c.yellow(`[WARN] Invalid LOGGING_LEVEL '${config.logging_level}'. Defaulting to INFO.`));
    }

    if (config.log_file_enabled) {
        try {
            // Ensure log directory exists
            await fs.mkdir(config.log_dir, { recursive: true });
            // Create a unique log file name with timestamp
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            logFilePath = path.join(config.log_dir, `trading_bot_${timestamp}.log`);
            // Open file handle for writing in append mode (async)
            logFileHandle = await fs.open(logFilePath, 'a');
            // Log the file path being used (only to console initially)
            console.log(c.cyan(`Logging ${levelName} and higher level messages to file: ${logFilePath}`));
        } catch (err) {
            console.error(c.red(`[ERROR] Failed to create log directory or open log file: ${err.message}. File logging disabled.`));
            logFilePath = null; // Disable file logging if setup fails
            logFileHandle = null;
        }
    } else {
        console.log(c.gray("File logging is disabled in configuration."));
        logFilePath = null;
        logFileHandle = null;
    }
}

// Logger object with methods for different levels
const logger = {
    /**
     * Logs a message to console and optionally to file.
     * @param {'DEBUG'|'INFO'|'WARN'|'ERROR'} level - The log level.
     * @param {...any} args - The message parts to log. Objects are inspected.
     */
    log(level, ...args) {
        // Skip logging if the message level is below the configured level
        if (logLevels[level] < currentLogLevel) return;

        const timestamp = new Date().toISOString();
        // Define colors for each log level
        const levelColor = {
            DEBUG: c.gray,
            INFO: c.cyan,
            WARN: c.yellow,
            ERROR: c.red,
        }[level] || c.white; // Default to white if level is unknown

        // Format message for console output (with colors)
        // Use util.inspect for objects to get better formatting and control depth/colors
        const consoleMessage = `${c.dim(timestamp)} [${levelColor(c.bold(level))}] ${args.map(arg =>
            typeof arg === 'object' ? inspect(arg, { depth: 3, colors: true }) : String(arg)
        ).join(' ')}`;

        // Output to console using appropriate method (log, warn, error)
        const consoleMethod = level === 'WARN' ? console.warn : level === 'ERROR' ? console.error : console.log;
        consoleMethod(consoleMessage);

        // Append to log file if handle is open
        if (logFileHandle) {
            // Format message for file (without colors, potentially deeper inspection)
            const fileMessage = `${timestamp} [${level}] ${args.map(arg =>
                typeof arg === 'object' ? inspect(arg, { depth: 4, colors: false }) : String(arg)
            ).join(' ')}\n`;

            // Asynchronously write to the file using the handle, handle potential errors
            // Use a fire-and-forget approach for performance, but log errors if they occur
            logFileHandle.write(fileMessage).catch(err => {
                // Log error to console only, to avoid infinite loop if file writing fails repeatedly
                console.error(c.red(`[ERROR] Failed to write to log file '${logFilePath}': ${err.message}`));
                // Optional: Consider disabling file logging after repeated errors
                // logger.warn(c.yellow("Disabling file logging due to repeated write errors."));
                // logFileHandle.close().catch(e => console.error("Error closing log file handle:", e));
                // logFileHandle = null;
            });
        }
    },
    // Convenience methods for each level
    debug(...args) { this.log('DEBUG', ...args); },
    info(...args) { this.log('INFO', ...args); },
    warn(...args) { this.log('WARN', ...args); },
    error(...args) { this.log('ERROR', ...args); },

    /**
     * Closes the asynchronous log file handle during shutdown.
     * @returns {Promise<void>}
     */
    async closeLogFile() {
        if (logFileHandle) {
            const handleToClose = logFileHandle; // Capture handle locally
            logFileHandle = null; // Prevent further writes immediately
            console.log(c.yellow("Closing log file handle..."));
            try {
                await handleToClose.close();
                console.log(c.green("Log file handle closed."));
            } catch (err) {
                console.error(c.red(`Error closing log file handle: ${err.message}`));
            }
        }
    }
};

// --- Utility Functions ---

/**
 * Simple asynchronous sleep function.
 * @param {number} ms - Milliseconds to sleep.
 * @returns {Promise<void>}
 */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Parses a timeframe string (like '1m', '1h', '1d') into milliseconds using CCXT's parser.
 * @param {string} timeframeString - The timeframe string.
 * @returns {number} The timeframe in milliseconds, or 0 if parsing fails.
 */
function parseTimeframeToMs(timeframeString) {
    try {
        // CCXT's parseTimeframe returns seconds, convert to milliseconds
        const seconds = ccxt.parseTimeframe(timeframeString);
        if (seconds && seconds > 0) {
            return seconds * 1000;
        }
        logger.warn(`Could not parse timeframe string '${timeframeString}' into seconds. Returning 0ms.`);
        return 0;
    } catch (e) {
        logger.error(`Error parsing timeframe string '${timeframeString}': ${e.message}`);
        return 0;
    }
}


/**
 * Wraps an async function call with retry logic for specific exceptions using exponential backoff.
 * @param {Function} func - The async function to execute.
 * @param {number} maxRetries - Maximum number of retry attempts (0 means one attempt, no retries).
 * @param {number} initialDelaySeconds - Initial delay between retries in seconds.
 * @param {Array<Error>} [allowedExceptions] - Array of CCXT/Error classes that trigger a retry. Defaults to common network/availability errors.
 * @param {string} [funcName] - Name of the function being called (for logging). Defaults to function's name or 'anonymous function'.
 * @returns {Promise<any>} - The result of the function if successful.
 * @throws {Error} - The last exception if all retries fail or a non-retryable error occurs.
 */
async function retryOnException(
    func,
    maxRetries,
    initialDelaySeconds,
    allowedExceptions = [ // Default retryable CCXT network/availability errors
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
    ],
    funcName = func.name || 'anonymous function'
) {
    let attempts = 0;
    while (attempts <= maxRetries) {
        attempts++;
        try {
            return await func(); // Attempt the function call
        } catch (e) {
            // Check if the error is an instance of any allowed exception types
            // Also check specific Bybit messages that might indicate temporary issues
            const isRetryable = allowedExceptions.some(excType => e instanceof excType) ||
                                (e instanceof ccxt.ExchangeError && (
                                    e.message.includes('busy') ||
                                    e.message.includes('try again later') ||
                                    e.message.includes('operation timed out') || // Some exchanges use this
                                    e.message.includes('ret_code=10006') || // request frequency too high
                                    e.message.includes('ret_code=10016') || // service unavailable / internal error
                                    (e.httpStatus === 503) || // Service Unavailable
                                    (e.httpStatus === 504) // Gateway Timeout
                                    // Add more specific Bybit error codes here if known (e.g., from info.retCode)
                                ));

            if (isRetryable && attempts <= maxRetries) {
                // Calculate delay with exponential backoff (e.g., 5s, 7.5s, 11.3s, ...)
                // Add random jitter (e.g., +/- 10%) to prevent thundering herd
                const jitter = (Math.random() * 0.2) - 0.1; // Range: -0.1 to +0.1
                const delay = initialDelaySeconds * 1000 * Math.pow(1.5, attempts - 1) * (1 + jitter);
                const delaySec = (delay / 1000).toFixed(1);
                logger.warn(`[Retry] Attempt ${attempts}/${maxRetries + 1} for ${c.yellow(funcName)} failed: ${c.red(e.constructor.name)} - ${e.message}. Retrying in ${delaySec}s...`);
                await sleep(delay); // Wait before next retry
            } else {
                // If it's not a retryable error OR max retries are exhausted
                const reason = isRetryable ? `failed after ${attempts} attempts` : 'non-retryable error';
                logger.error(`[Error] ${c.yellow(funcName)} ${reason}: ${c.red(e.constructor.name)} - ${e.message}`);
                // Log stack trace for better debugging
                if (e instanceof Error) logger.debug(`${funcName} Error Stack:`, e.stack);
                throw e; // Re-throw the error to be handled by the caller
            }
        }
    }
    // This point should theoretically not be reached if maxRetries >= 0
    const unexpectedError = new Error(`${funcName} retry loop completed unexpectedly after ${attempts} attempts.`);
    logger.error(c.red(unexpectedError.message));
    throw unexpectedError;
}

// --- Configuration Class ---
/**
 * Loads, validates, and provides access to bot configuration settings from .env file and defaults.
 */
class Config {
    constructor() {
        // Initial log before full logger setup
        console.log(c.blue("Loading and validating configuration from .env file..."));
        dotenv.config(); // Load variables from .env into process.env

        // --- Load API Credentials ---
        this.bybit_api_key = process.env.BYBIT_API_KEY || null;
        this.bybit_api_secret = process.env.BYBIT_API_SECRET || null;

        // --- Load Trading Parameters ---
        this.symbol = process.env.SYMBOL || DEFAULTS.SYMBOL;
        this.leverage = parseFloat(process.env.LEVERAGE || DEFAULTS.LEVERAGE);
        this.risk_per_trade = parseFloat(process.env.RISK_PER_TRADE || DEFAULTS.RISK_PER_TRADE);
        this.sl_atr_mult = parseFloat(process.env.SL_ATR_MULT || DEFAULTS.SL_ATR_MULT);
        this.tp_atr_mult = parseFloat(process.env.TP_ATR_MULT || DEFAULTS.TP_ATR_MULT);
        this.trailing_stop_mult = parseFloat(process.env.TRAILING_STOP_MULT || DEFAULTS.TRAILING_STOP_MULT);
        this.timeframe = process.env.TIMEFRAME || DEFAULTS.TIMEFRAME;

        // --- Load Strategy Parameters ---
        this.short_st_period = parseInt(process.env.SHORT_ST_PERIOD || DEFAULTS.SHORT_ST_PERIOD, 10);
        this.long_st_period = parseInt(process.env.LONG_ST_PERIOD || DEFAULTS.LONG_ST_PERIOD, 10);
        this.st_multiplier = parseFloat(process.env.ST_MULTIPLIER || DEFAULTS.ST_MULTIPLIER);
        this.gaussian_filter_length = parseInt(process.env.GAUSSIAN_FILTER_LENGTH || DEFAULTS.GAUSSIAN_FILTER_LENGTH, 10);
        this.volume_short_period = parseInt(process.env.VOLUME_SHORT_PERIOD || DEFAULTS.VOLUME_SHORT_PERIOD, 10);
        this.volume_long_period = parseInt(process.env.VOLUME_LONG_PERIOD || DEFAULTS.VOLUME_LONG_PERIOD, 10);
        this.volume_spike_threshold = parseFloat(process.env.VOLUME_SPIKE_THRESHOLD || DEFAULTS.VOLUME_SPIKE_THRESHOLD);
        this.order_book_depth = parseInt(process.env.ORDER_BOOK_DEPTH || DEFAULTS.ORDER_BOOK_DEPTH, 10);
        this.ob_pressure_threshold = parseFloat(process.env.OB_PRESSURE_THRESHOLD || DEFAULTS.OB_PRESSURE_THRESHOLD);

        // --- Load Bot Behavior Parameters ---
        this.dry_run = (process.env.DRY_RUN || DEFAULTS.DRY_RUN).toLowerCase() !== 'false';
        this.logging_level = (process.env.LOGGING_LEVEL || DEFAULTS.LOGGING_LEVEL).toUpperCase();
        this.log_file_enabled = (process.env.LOG_FILE_ENABLED || DEFAULTS.LOG_FILE_ENABLED).toLowerCase() === 'true';
        this.log_dir = process.env.LOG_DIR || DEFAULTS.LOG_DIR;
        this.max_retries = parseInt(process.env.MAX_RETRIES || DEFAULTS.MAX_RETRIES, 10);
        this.retry_delay = parseInt(process.env.RETRY_DELAY || DEFAULTS.RETRY_DELAY, 10);
        this.cache_ttl = parseInt(process.env.CACHE_TTL || DEFAULTS.CACHE_TTL, 10);
        this.state_file = process.env.STATE_FILE || DEFAULTS.STATE_FILE;
        this.close_position_on_shutdown = (process.env.CLOSE_POSITION_ON_SHUTDOWN || DEFAULTS.CLOSE_POSITION_ON_SHUTDOWN).toLowerCase() === 'true';

        // --- Load Exchange/Order Parameters ---
        this.currency = process.env.CURRENCY || DEFAULTS.CURRENCY;
        this.exchange_type = process.env.EXCHANGE_TYPE || DEFAULTS.EXCHANGE_TYPE;
        // Normalize trigger price type to match Bybit's V5 API expected values (case-sensitive)
        this.order_trigger_price_type = this._normalizeTriggerPriceType(process.env.ORDER_TRIGGER_PRICE_TYPE || DEFAULTS.ORDER_TRIGGER_PRICE_TYPE);
        // Normalize Time-in-Force
        this.time_in_force = this._normalizeTimeInForce(process.env.TIME_IN_FORCE || DEFAULTS.TIME_IN_FORCE);
        this.bybit_account_type = (process.env.BYBIT_ACCOUNT_TYPE || DEFAULTS.BYBIT_ACCOUNT_TYPE).toUpperCase(); // CONTRACT, UNIFIED, FUND

        // --- Load Notification Parameters ---
        this.sms_enabled = (process.env.SMS_ENABLED || DEFAULTS.SMS_ENABLED).toLowerCase() === 'true';
        this.sms_recipient_number = process.env.SMS_RECIPIENT_NUMBER || DEFAULTS.SMS_RECIPIENT_NUMBER;

        // --- Internal State ---
        this.termux_sms_available = false; // Determined during validation

        // --- Perform Validation and Checks ---
        this._validate(); // Validate loaded parameters
        this._checkTermuxSms(); // Check Termux SMS capability after validation

        // --- Finalize Logging Setup ---
        // Call setupLogging *after* validation ensures LOGGING_LEVEL is valid
        setupLogging(this).catch(err => console.error(c.red("Error during async logging setup:"), err));

        // Log the final configuration summary (masking the secret)
        const configSummary = { ...this };
        configSummary.bybit_api_secret = configSummary.bybit_api_secret ? '******' : null;
        logger.info(c.green(`Configuration loaded successfully. Dry Run: ${c.bold(this.dry_run)}`));
        logger.debug("Full Config (Secret Masked):", configSummary);
    }

    /**
     * Normalizes ORDER_TRIGGER_PRICE_TYPE to Bybit V5 expected values.
     * @param {string} type - Input trigger price type string.
     * @returns {'LastPrice'|'MarkPrice'|'IndexPrice'} Normalized type.
     * @private
     */
    _normalizeTriggerPriceType(type) {
        const lowerType = String(type || '').toLowerCase().replace(/[\s_-]/g, ""); // Normalize further
        if (lowerType.includes('last')) return 'LastPrice';
        if (lowerType.includes('mark')) return 'MarkPrice';
        if (lowerType.includes('index')) return 'IndexPrice';
        logger.warn(c.yellow(`Unknown ORDER_TRIGGER_PRICE_TYPE '${type}'. Defaulting to 'LastPrice'.`));
        return 'LastPrice'; // Default
    }

    /**
     * Normalizes TIME_IN_FORCE to CCXT/Bybit expected values.
     * @param {string} tif - Input Time-in-Force string.
     * @returns {'GTC'|'IOC'|'FOK'|'PostOnly'} Normalized TIF.
     * @private
     */
    _normalizeTimeInForce(tif) {
        const lowerTif = String(tif || '').toLowerCase().replace(/[\s_-]/g, "");
        if (lowerTif === 'goodtillcancel' || lowerTif === 'gtc') return 'GTC';
        if (lowerTif === 'immediateorcancel' || lowerTif === 'ioc') return 'IOC';
        if (lowerTif === 'fillorkill' || lowerTif === 'fok') return 'FOK';
        if (lowerTif === 'postonly') return 'PostOnly';
        logger.warn(c.yellow(`Unknown TIME_IN_FORCE '${tif}'. Defaulting to 'GTC'.`));
        return 'GTC'; // Default
    }

    /**
     * Validates all loaded configuration parameters.
     * Throws an error if validation fails.
     * @private
     * @throws {Error} If configuration is invalid.
     */
    _validate() {
        logger.debug("Validating configuration parameters...");
        const errors = [];

        // Check API keys only if not in dry run mode
        if (!this.dry_run && (!this.bybit_api_key || !this.bybit_api_secret)) {
            errors.push("BYBIT_API_KEY and BYBIT_API_SECRET environment variables are required when DRY_RUN is false.");
        }

        // Validate numerical parameters
        if (isNaN(this.leverage) || this.leverage <= 0) errors.push("LEVERAGE must be a positive number.");
        if (isNaN(this.risk_per_trade) || !(this.risk_per_trade > 0 && this.risk_per_trade <= 1.0)) {
            errors.push("RISK_PER_TRADE must be a positive number between 0 (exclusive) and 1 (inclusive) (e.g., 0.01 for 1%).");
        }
        // Warnings for high risk
        if (this.risk_per_trade > 0.05 && this.risk_per_trade <= 0.1) logger.warn(c.yellow(`High Risk Setting: RISK_PER_TRADE (${(this.risk_per_trade * 100).toFixed(1)}%) is > 5%. Ensure this is intended.`));
        if (this.risk_per_trade > 0.1) logger.warn(c.red(`EXTREME Risk Setting: RISK_PER_TRADE (${(this.risk_per_trade * 100).toFixed(1)}%) is > 10%. High chance of liquidation!`));

        if (!VALID_TIMEFRAMES.includes(this.timeframe)) errors.push(`Invalid TIMEFRAME: '${this.timeframe}'. Must be one of: ${VALID_TIMEFRAMES.join(', ')}`);

        // Helper validation functions
        const checkPositiveNumber = (key) => { if (isNaN(this[key]) || this[key] <= 0) errors.push(`${key.toUpperCase()} must be a positive number. Got: ${this[key]}`); };
        const checkNonNegativeNumber = (key) => { if (isNaN(this[key]) || this[key] < 0) errors.push(`${key.toUpperCase()} must be a non-negative number. Got: ${this[key]}`); };
        const checkPositiveInteger = (key) => { if (!Number.isInteger(this[key]) || this[key] <= 0) errors.push(`${key.toUpperCase()} must be a positive integer. Got: ${this[key]}`); };
        const checkNonNegativeInteger = (key) => { if (!Number.isInteger(this[key]) || this[key] < 0) errors.push(`${key.toUpperCase()} must be a non-negative integer. Got: ${this[key]}`); };
        const checkNumberRange = (key, min, max) => { if (isNaN(this[key]) || this[key] < min || this[key] > max) errors.push(`${key.toUpperCase()} must be a number between ${min} and ${max}. Got: ${this[key]}`); };

        checkPositiveNumber('sl_atr_mult');
        checkPositiveNumber('tp_atr_mult');
        checkNonNegativeNumber('trailing_stop_mult'); // Can be 0 if disabled
        if (this.trailing_stop_mult > 0 && isNaN(this.trailing_stop_mult)) errors.push(`TRAILING_STOP_MULT must be a positive number if enabled (or 0 to disable). Got: ${this.trailing_stop_mult}`);

        checkPositiveInteger('short_st_period');
        checkPositiveInteger('long_st_period');
        checkPositiveNumber('st_multiplier');
        checkPositiveInteger('gaussian_filter_length');
        checkPositiveNumber('volume_spike_threshold');
        checkPositiveInteger('volume_short_period');
        checkPositiveInteger('volume_long_period');
        checkPositiveInteger('order_book_depth');
        checkNonNegativeInteger('max_retries');
        checkPositiveInteger('retry_delay');
        checkPositiveInteger('cache_ttl');
        checkNumberRange('ob_pressure_threshold', 0, 1);

        if (this.volume_short_period >= this.volume_long_period) errors.push("VOLUME_SHORT_PERIOD must be less than VOLUME_LONG_PERIOD.");

        // Validate SMS settings
        if (this.sms_enabled && (!this.sms_recipient_number || !String(this.sms_recipient_number).trim())) {
            logger.warn(c.yellow("SMS_ENABLED is true, but SMS_RECIPIENT_NUMBER is not set. Disabling SMS notifications."));
            this.sms_enabled = false;
        }

        // Validate Exchange/Order parameters
        if (!['swap', 'future', 'spot'].includes(this.exchange_type)) errors.push(`Invalid EXCHANGE_TYPE: '${this.exchange_type}'. Must be 'swap', 'future', or 'spot'.`);
        // Validation for normalized types
        const valid_trigger_types = ['LastPrice', 'MarkPrice', 'IndexPrice'];
        if (!valid_trigger_types.includes(this.order_trigger_price_type)) errors.push(`Invalid/Unnormalized ORDER_TRIGGER_PRICE_TYPE: '${this.order_trigger_price_type}'. Must normalize to: ${valid_trigger_types.join(', ')}.`);
        const valid_tif = ['GTC', 'IOC', 'FOK', 'PostOnly'];
        if (!valid_tif.includes(this.time_in_force)) errors.push(`Invalid/Unnormalized TIME_IN_FORCE: '${this.time_in_force}'. Must normalize to: ${valid_tif.join(', ')}.`);
        const valid_account_types = ['CONTRACT', 'UNIFIED', 'FUND'];
        if (!valid_account_types.includes(this.bybit_account_type)) errors.push(`Invalid BYBIT_ACCOUNT_TYPE: '${this.bybit_account_type}'. Must be one of: ${valid_account_types.join(', ')}`);


        // Logging level is handled by setupLogging, but ensure key exists
        if (!logLevels.hasOwnProperty(this.logging_level)) {
            // Use console here as logger might not be ready
            console.warn(c.yellow(`Invalid LOGGING_LEVEL '${this.logging_level}'. Defaulting to INFO.`));
            this.logging_level = "INFO"; // Correct the value for setupLogging
        }

        // --- Final Error Check ---
        if (errors.length > 0) {
            const errorMessage = "Configuration validation failed:\n" + errors.map(e => `- ${e}`).join('\n');
            // Use console.error as logger might not be fully ready if setupLogging fails
            console.error(c.red(errorMessage));
            throw new Error(errorMessage);
        }
        // Logger should be available after setupLogging is called following validation
        // logger.debug(c.green("Configuration validation successful.")); // Logged after logger is ready
    }

    /**
     * Checks if Termux SMS sending is likely possible.
     * Updates `this.termux_sms_available` and potentially disables `this.sms_enabled`.
     * @private
     */
    _checkTermuxSms() {
        if (!this.sms_enabled) {
            this.termux_sms_available = false;
            // Use console as logger might not be ready yet
            console.log(c.gray("SMS is disabled in config."));
            return;
        }

        // Check if running in Termux environment (more reliable checks)
        // Check TERMUX_VERSION env var OR existence of typical Termux home dir
        // Use try-catch for fs.existsSync for robustness
        let termuxDirExists = false;
        try {
             // fs.existsSync is synchronous, use fs.promises.stat instead for async check
             fs.stat('/data/data/com.termux/files/home')
                 .then(stats => { termuxDirExists = stats.isDirectory(); })
                 .catch(() => { termuxDirExists = false; }); // Ignore errors like EPERM or ENOENT
        } catch (e) {
             console.warn(c.yellow("Could not check Termux directory existence synchronously:"), e.message);
        }
        const isTermux = !!process.env.TERMUX_VERSION || termuxDirExists;

        if (!isTermux) {
             console.log(c.yellow("Not running in a recognizable Termux environment. Disabling SMS feature."));
             this.sms_enabled = false; // Disable SMS if not in Termux
             this.termux_sms_available = false;
             return;
        }

        // Check if termux-sms-send command exists and is executable
        try {
             // Use `which` to check if the command exists in PATH and is executable
             // Stdio 'ignore' prevents output, throws error if not found/executable
             execSync('which termux-sms-send', { stdio: 'ignore', timeout: 2000 }); // Add short timeout
             this.termux_sms_available = true;
             console.log(c.green("Termux environment and 'termux-sms-send' command found. SMS enabled."));
        } catch (error) {
             console.warn(c.yellow("Termux environment detected, but 'termux-sms-send' command not found or not executable in PATH. Disabling SMS. Ensure Termux:API app is installed and 'pkg install termux-api' was run. Error:"), error.message);
             this.sms_enabled = false; // Disable SMS if command is missing
             this.termux_sms_available = false;
        }
    }
}

// --- Notification Service ---
/**
 * Handles sending notifications, currently only Termux SMS.
 */
class NotificationService {
    /**
     * Sends an SMS message using the Termux API command asynchronously.
     * @param {string} message - The message content.
     * @param {Config} config - The bot's configuration object.
     */
    sendSms(message, config) {
        // Check if SMS should be sent
        if (!config.sms_enabled || !config.termux_sms_available || !config.sms_recipient_number) {
            logger.debug(c.gray(`SMS sending skipped (Enabled: ${config.sms_enabled}, Available: ${config.termux_sms_available}, Recipient Set: ${!!config.sms_recipient_number}): ${message.substring(0, 80)}...`));
            return;
        }

        try {
            // Basic sanitization to prevent issues with shell characters
            // Remove common problematic shell meta-characters. This is NOT foolproof security.
            // Escape single quotes properly for shell command
            let sanitizedMessage = String(message) // Ensure message is a string
                .replace(/[`$!\\;&|<>*?()#~]/g, "") // Remove potentially dangerous characters
                .replace(/"/g, '\\"') // Escape double quotes
                .replace(/'/g, "'\\''"); // Escape single quotes for shell embedding

            const maxSmsLength = 160; // Standard SMS length limit
            if (sanitizedMessage.length > maxSmsLength) {
                logger.warn(c.yellow(`Truncating long SMS message (${sanitizedMessage.length} chars) to ${maxSmsLength}.`));
                sanitizedMessage = sanitizedMessage.substring(0, maxSmsLength - 3) + "...";
            }

            // Construct the command carefully escaping the message and number
            // Quote recipient number and message content to handle spaces/special chars allowed after sanitization
            const command = `termux-sms-send -n "${config.sms_recipient_number}" "${sanitizedMessage}"`;

            logger.debug(`Executing SMS command: termux-sms-send -n "${config.sms_recipient_number}" "..."`);

            // Execute the command asynchronously with a timeout
            exec(command, { timeout: 30000 }, (error, stdout, stderr) => { // 30 second timeout
                if (error) {
                    // Log detailed error information
                    logger.error(c.red(`SMS command failed: ${error.message}. Code: ${error.code}, Signal: ${error.signal}`));
                    if (stderr) logger.error(c.red(`SMS stderr: ${stderr.trim()}`));
                    if (stdout) logger.error(c.red(`SMS stdout (error context): ${stdout.trim()}`));
                    // Potentially disable SMS temporarily if it keeps failing?
                    // config.sms_enabled = false; // Be careful modifying config state here
                    // logger.warn(c.yellow("Disabling SMS temporarily due to sending failure."));
                    return;
                }
                // Log success and any output/stderr (usually empty on success)
                logger.info(c.green(`SMS potentially sent successfully to ${config.sms_recipient_number}: "${message.substring(0, 80)}..."`));
                if (stderr && stderr.trim()) logger.debug(`termux-sms-send stderr: ${stderr.trim()}`);
                if (stdout && stdout.trim()) logger.debug(`termux-sms-send stdout: ${stdout.trim()}`);
            });
        } catch (e) {
            // Catch synchronous errors during command preparation (less likely)
            logger.error(c.red(`SMS sending failed with unexpected synchronous error: ${e.message}`), e.stack);
        }
    }
}

// --- Exchange Manager ---
/**
 * Handles CCXT initialization, market loading, data fetching, caching, and exchange interactions.
 */
class ExchangeManager {
    /**
     * @param {Config} config - The validated bot configuration.
     */
    constructor(config) {
        this.config = config;
        /** @type {ccxt.Exchange | null} */
        this.exchange = null; // CCXT exchange instance
        // In-memory cache store
        this._caches = {
            ohlcv: { key: null, data: null, timestamp: 0, ttl: this.config.cache_ttl },
            order_book: { key: null, data: null, timestamp: 0, ttl: Math.max(1, Math.min(this.config.cache_ttl, 10)) }, // Shorter TTL for OB
            ticker: { key: null, data: null, timestamp: 0, ttl: Math.max(1, Math.min(this.config.cache_ttl, 5)) },   // Shorter TTL for ticker
            balance: { key: null, data: null, timestamp: 0, ttl: this.config.cache_ttl },
            position: { key: null, data: null, timestamp: 0, ttl: Math.max(5, Math.min(this.config.cache_ttl, 15)) }, // Moderate TTL for position
        };
    }

    /**
     * Initializes the CCXT exchange instance, loads markets, validates, and sets leverage.
     * @returns {Promise<void>}
     * @throws {Error} If initialization fails critically.
     */
    async initialize() {
        this.exchange = setupBybitExchange(this.config);
        await loadMarketsAndValidate(this.exchange, this.config); // config.symbol might be updated here
        if (!this.config.dry_run && ['swap', 'future'].includes(this.config.exchange_type)) {
            await setLeverage(this.exchange, this.config);
        }
        logger.info(c.green(`Exchange Manager initialized for ${c.bold(this.exchange.id)}, Symbol: ${c.bold(this.config.symbol)}, Type: ${c.bold(this.config.exchange_type)}`));
    }

    /**
     * Retrieves data from the specified cache if the key matches and data is not expired.
     * @param {string} cacheName - The name of the cache (e.g., 'ohlcv', 'ticker').
     * @param {string} key - A unique key identifying the specific data within the cache (e.g., symbol_timeframe_limit).
     * @returns {any | null} The cached data (deep copy) or null if not found or expired.
     * @private
     */
    _getCache(cacheName, key) {
        const cache = this._caches[cacheName];
        // Check if cache exists, key matches, and data is present
        if (!cache || cache.key !== key || cache.data === null) {
            return null;
        }
        // Check if cache entry has expired
        const age = (Date.now() / 1000) - cache.timestamp; // Age in seconds
        if (age < cache.ttl) {
            logger.debug(c.gray(`CACHE HIT: Using cached data for '${cacheName}' (Key: ${key}, Age: ${age.toFixed(1)}s < TTL: ${cache.ttl}s)`));
            // Return deep copy to prevent accidental modification of cached data
            try {
                return JSON.parse(JSON.stringify(cache.data));
            } catch (e) {
                logger.warn(c.yellow(`Failed to deep copy cached data for ${cacheName}. Returning null. Error: ${e.message}`));
                return null;
            }
        } else {
            logger.debug(c.gray(`CACHE EXPIRED: Data for '${cacheName}' (Key: ${key}, Age: ${age.toFixed(1)}s >= TTL: ${cache.ttl}s)`));
            // Clear expired data
            cache.data = null;
            cache.key = null;
            return null; // Cache expired
        }
    }

    /**
     * Stores data in the specified cache with the given key and current timestamp.
     * Stores a deep copy of the data.
     * @param {string} cacheName - The name of the cache.
     * @param {string} key - The unique key for the data.
     * @param {any} data - The data to be stored.
     * @private
     */
    _setCache(cacheName, key, data) {
        if (this._caches[cacheName]) {
            try {
                // Store deep copy to prevent modification issues
                this._caches[cacheName].key = key;
                this._caches[cacheName].data = data !== null ? JSON.parse(JSON.stringify(data)) : null;
                this._caches[cacheName].timestamp = Date.now() / 1000; // Store timestamp in seconds
                logger.debug(c.gray(`CACHE SET: Updated cache for '${cacheName}' (Key: ${key})`));
            } catch (e) {
                logger.warn(c.yellow(`Failed to deep copy data for caching ${cacheName}. Cache not set. Error: ${e.message}`));
                 this._caches[cacheName].data = null; // Ensure cache is cleared if copy fails
                 this._caches[cacheName].key = null;
            }
        } else {
            logger.warn(c.yellow(`Attempted to set unknown cache: ${cacheName}`));
        }
    }

    // --- Data Fetching Methods (delegate to helper functions using this instance) ---
    async fetchOhlcv(limit = 100) { return fetchOHLCV(this, limit); }
    async fetchOrderBook() { return fetchOrderBookData(this); }
    async getPosition() { return getExchangePosition(this); }
    async fetchBalance() { return getAccountBalance(this); }
    async fetchTicker() { return getTickerData(this); }
}

// --- Indicator Calculation Functions (Extracted for Clarity & Robustness) ---

/**
 * Applies a simple Gaussian filter (approximated by two cascaded SMAs).
 * Robustly handles input validation and calculation errors.
 * @param {number[]} data - Array of numerical data.
 * @param {number} length - Lookback period for the moving averages. Must be a positive integer.
 * @param {object} logger - Logger instance.
 * @returns {number[]} - Array of filtered data, same length as input, or empty array on critical failure. Early values will be less accurate or NaN.
 */
function gaussianFilter(data, length, logger) {
    // --- Input Validation ---
    if (!Array.isArray(data) || data.length === 0) {
        logger.error(c.red(c.bold("Error (gaussianFilter):")) + c.yellow(" Invalid input. 'data' must be a non-empty array."));
        return [];
    }
    if (!Number.isInteger(length) || length <= 0) {
        logger.error(c.red(c.bold("Error (gaussianFilter):")) + c.yellow(` Invalid input. 'length' must be a positive integer. Received: ${length}`));
        return [];
    }
    if (length > data.length) {
        logger.warn(c.yellow(c.bold("Warning (gaussianFilter):")) + ` Filter length (${length}) is greater than data length (${data.length}). Result will have significant lag and potential inaccuracies.`);
        // Proceed, but be aware the result might not be useful initially.
    }

    try {
        // --- First SMA Pass ---
        const sma1 = calculateSMA(data, length, logger); // Use the robust SMA function
        if (sma1.length !== data.length) {
            logger.error(c.red(c.bold("Error (gaussianFilter):")) + c.yellow(" First SMA pass failed or returned incorrect length."));
            return [];
        }

        // --- Second SMA Pass (on the result of the first pass) ---
        const sma2 = calculateSMA(sma1, length, logger); // Apply SMA again
        if (sma2.length !== data.length) {
            logger.error(c.red(c.bold("Error (gaussianFilter):")) + c.yellow(" Second SMA pass failed or returned incorrect length."));
            return [];
        }

        return sma2; // Return the result of the second SMA pass

    } catch (error) {
        logger.error(c.red(c.bold("Error (gaussianFilter):")) + c.yellow(` Unexpected calculation error. ${error.message}`), error.stack);
        return []; // Return empty array on unexpected errors
    }
}


/**
 * Calculates Average True Range (ATR) using Simple Moving Average (SMA) of True Range.
 * Robustly handles input validation and calculation errors.
 * @param {number[]} high - Array of high prices.
 * @param {number[]} low - Array of low prices.
 * @param {number[]} close - Array of closing prices.
 * @param {number} period - Lookback period for ATR calculation. Must be a positive integer.
 * @param {object} logger - Logger instance.
 * @returns {number[]} - Array of ATR values, same length as input, or empty array on critical failure. Early values will be less accurate or NaN.
 */
function calculateATR(high, low, close, period, logger) {
    const n = high?.length;
    // --- Input Validation ---
    if (!Array.isArray(high) || !Array.isArray(low) || !Array.isArray(close) || !n || n !== low.length || n !== close.length) {
        logger.error(c.red(c.bold("Error (calculateATR):")) + c.yellow(" Invalid input. Requires non-empty 'high', 'low', 'close' arrays of equal length."));
        return [];
    }
    if (!Number.isInteger(period) || period <= 0) {
        logger.error(c.red(c.bold("Error (calculateATR):")) + c.yellow(` Invalid input. 'period' must be a positive integer. Received: ${period}`));
        return [];
    }
    if (period >= n) { // Use >= because index starts at 0
        logger.warn(c.yellow(c.bold("Warning (calculateATR):")) + ` ATR period (${period}) >= data length (${n}). Result may be inaccurate or primarily based on initial TR values.`);
        // Proceed, but understand the limitation.
    }

    try {
        // --- Calculate True Range (TR) ---
        const tr = new Array(n).fill(NaN); // Initialize with NaN
        for (let i = 0; i < n; i++) {
            const h = high[i], l = low[i], c = close[i];
            const cPrev = i > 0 ? close[i - 1] : null; // Previous close

            // Validate price data for this index
            if (typeof h !== 'number' || !Number.isFinite(h) ||
                typeof l !== 'number' || !Number.isFinite(l) || l > h || // Basic check low <= high
                typeof c !== 'number' || !Number.isFinite(c) ||
                (i > 0 && (typeof cPrev !== 'number' || !Number.isFinite(cPrev)))) {
                // logger.warn(c.yellow(c.bold("Warning (calculateATR):")) + c.yellow(` Invalid/non-finite price encountered at index ${i}. Skipping TR calculation for this index.`)); // Too noisy
                continue; // Keep tr[i] as NaN
            }

            const highLow = h - l;
            const highClosePrev = (i > 0 && cPrev !== null) ? Math.abs(h - cPrev) : highLow; // Use highLow for the first bar
            const lowClosePrev = (i > 0 && cPrev !== null) ? Math.abs(l - cPrev) : highLow;  // Use highLow for the first bar

            tr[i] = Math.max(highLow, highClosePrev, lowClosePrev);
            if (!Number.isFinite(tr[i])) {
                 // logger.warn(c.yellow(c.bold("Warning (calculateATR):")) + ` Calculated TR at index ${i} is non-finite.`); // Too noisy
                 tr[i] = NaN; // Ensure it's NaN if calculation fails
            }
        }

        // --- Calculate SMA of TR ---
        // Use the robust calculateSMA function
        const atr = calculateSMA(tr, period, logger);

        // Final check if SMA calculation succeeded
        if (atr.length !== n) {
            logger.error(c.red(c.bold("Error (calculateATR):")) + c.yellow(" SMA calculation for ATR failed or returned incorrect length."));
            return [];
        }

        return atr;

    } catch (error) {
        logger.error(c.red(c.bold("Error (calculateATR):")) + c.yellow(` Unexpected calculation error. ${error.message}`), error.stack);
        return [];
    }
}


/**
 * @typedef {object} SupertrendResult - Contains the trend direction and line value for a single point.
 * @property {number} trend - The trend direction: 1 for bullish, -1 for bearish, 0 for undefined/error.
 * @property {number} value - The calculated Supertrend line value (NaN if calculation failed for this point).
 */

/**
 * Calculates the enhanced Ehlers Supertrend indicator.
 * Uses Gaussian filtered high/low prices for ATR calculation and band centering.
 * Robustly handles input validation and calculation errors.
 * @param {number[]} high - Array of high prices.
 * @param {number[]} low - Array of low prices.
 * @param {number[]} close - Array of closing prices.
 * @param {number} period - Period for ATR and Supertrend. Must be a positive integer.
 * @param {number} multiplier - Multiplier for ATR band calculation. Must be a positive number.
 * @param {number} gaussianLength - Length for Gaussian filter applied to high/low. Must be a positive integer.
 * @param {object} logger - Logger instance.
 * @returns {SupertrendResult[]} - Array of Supertrend result objects, same length as input, or empty array on critical failure.
 */
function ehlersSupertrend(high, low, close, period, multiplier, gaussianLength, logger) {
    const n = high?.length;
    // --- Input Validation ---
    if (!Array.isArray(high) || !Array.isArray(low) || !Array.isArray(close) || !n || n !== low.length || n !== close.length) {
        logger.error(c.red(c.bold("Error (ehlersSupertrend):")) + c.yellow(" Invalid input. Requires non-empty 'high', 'low', 'close' arrays of equal length.")); return [];
    }
    if (!Number.isInteger(period) || period <= 0) { logger.error(c.red(c.bold("Error (ehlersSupertrend):")) + c.yellow(` Invalid period: ${period}. Must be a positive integer.`)); return []; }
    if (typeof multiplier !== 'number' || !Number.isFinite(multiplier) || multiplier <= 0) { logger.error(c.red(c.bold("Error (ehlersSupertrend):")) + c.yellow(` Invalid multiplier: ${multiplier}. Must be a positive number.`)); return []; }
    if (!Number.isInteger(gaussianLength) || gaussianLength <= 0) { logger.error(c.red(c.bold("Error (ehlersSupertrend):")) + c.yellow(` Invalid gaussianLength: ${gaussianLength}. Must be a positive integer.`)); return []; }
    if (period >= n) { logger.warn(c.yellow(c.bold("Warning (ehlersSupertrend):")) + ` Supertrend period (${period}) >= data length (${n}). Results may be inaccurate.`);}
    if (gaussianLength >= n) { logger.warn(c.yellow(c.bold("Warning (ehlersSupertrend):")) + ` Gaussian length (${gaussianLength}) >= data length (${n}). Filtered values may be inaccurate.`);}


    logger.debug(c.dim(`Calculating Ehlers Supertrend (P:${period}, M:${multiplier}, G:${gaussianLength})...`));

    try {
        // Step 1: Gaussian Filter High/Low
        const filteredHigh = gaussianFilter(high, gaussianLength, logger);
        const filteredLow = gaussianFilter(low, gaussianLength, logger);
        if (filteredHigh.length !== n || filteredLow.length !== n) {
            logger.error(c.red(c.bold("Error (ehlersSupertrend):")) + c.yellow(" Gaussian filtering failed or returned incorrect length. Cannot proceed.")); return [];
        }

        // Step 2: Calculate ATR on Filtered H/L and Original Close
        const atrValues = calculateATR(filteredHigh, filteredLow, close, period, logger);
        if (atrValues.length !== n) {
            logger.error(c.red(c.bold("Error (ehlersSupertrend):")) + c.yellow(" ATR calculation failed or returned incorrect length. Cannot proceed.")); return [];
        }

        // Step 3: Calculate Basic Upper/Lower Bands using FILTERED high/low center
        const upperBandBasic = new Array(n).fill(NaN);
        const lowerBandBasic = new Array(n).fill(NaN);
        for (let i = 0; i < n; i++) {
            const fh = filteredHigh[i], fl = filteredLow[i], atr = atrValues[i];
            // Check if intermediate values are valid
            if (!Number.isFinite(fh) || !Number.isFinite(fl) || !Number.isFinite(atr)) {
                // logger.warn(c.yellow(c.bold("Warning (ehlersSupertrend):")) + c.yellow(` Invalid intermediate value (Filtered H/L or ATR) at index ${i}. Skipping band calculation for this index.`)); // Too noisy
                continue; // Keep bands as NaN
            }
            const center = (fh + fl) / 2;
            upperBandBasic[i] = center + multiplier * atr;
            lowerBandBasic[i] = center - multiplier * atr;
            if (!Number.isFinite(upperBandBasic[i]) || !Number.isFinite(lowerBandBasic[i])) {
                 // logger.warn(c.yellow(c.bold("Warning (ehlersSupertrend):")) + ` Calculated basic band at index ${i} is non-finite.`); // Too noisy
                 // Mark as NaN if calculation failed
                 upperBandBasic[i] = NaN;
                 lowerBandBasic[i] = NaN;
            }
        }

        // Step 4: Calculate Final Supertrend Line and Trend (Stateful)
        const supertrendResults = new Array(n);
        let currentTrend = 0; // Start neutral until first valid calculation
        let currentSupertrendLine = NaN; // Final line value

        for (let i = 0; i < n; i++) {
            const currentClose = close[i];
            const currentUpperBasic = upperBandBasic[i]; // Upper band based on filtered center
            const currentLowerBasic = lowerBandBasic[i]; // Lower band based on filtered center

            // Get previous state, handling the first iteration and previous errors
            let previousTrend = (i > 0 && supertrendResults[i - 1]) ? supertrendResults[i - 1].trend : 0; // Use 0 if no valid previous trend
            let previousSupertrendLine = (i > 0 && supertrendResults[i - 1]) ? supertrendResults[i - 1].value : NaN;

            // Check if inputs for this step are valid
            if (typeof currentClose !== 'number' || !Number.isFinite(currentClose) ||
                !Number.isFinite(currentUpperBasic) || !Number.isFinite(currentLowerBasic)) {
                // logger.warn(c.yellow(c.bold("Warning (ehlersSupertrend):")) + c.yellow(` Invalid input value (Close/Bands) at index ${i}. Cannot determine trend/value for this index. Carrying over previous state.`)); // Too noisy
                supertrendResults[i] = { trend: previousTrend, value: previousSupertrendLine }; // Carry over previous state on error
                continue;
            }

            // --- Supertrend Core Logic ---
            // Determine potential trend flip based on close vs basic bands
            let nextPotentialTrend = previousTrend;
            if (currentClose > currentUpperBasic) {
                nextPotentialTrend = 1; // Bullish signal
            } else if (currentClose < currentLowerBasic) {
                nextPotentialTrend = -1; // Bearish signal
            }
            // If close is between bands, potential trend remains the same

            // Update the confirmed currentTrend state based on the potential flip
            if (nextPotentialTrend === 1 && previousTrend <= 0) { // Flip to Bullish or first bullish signal
                currentTrend = 1;
            } else if (nextPotentialTrend === -1 && previousTrend >= 0) { // Flip to Bearish or first bearish signal
                currentTrend = -1;
            }
            // Else, currentTrend remains unchanged (could be 1, -1, or still 0 if never triggered)

            // Determine the final Supertrend line value based on the confirmed currentTrend
            let finalSupertrendLine = NaN;
            if (currentTrend === 1) { // If confirmed bullish
                finalSupertrendLine = currentLowerBasic;
                // Ensure line doesn't decrease if continuing bullish
                if (previousTrend === 1 && Number.isFinite(previousSupertrendLine)) {
                    finalSupertrendLine = Math.max(finalSupertrendLine, previousSupertrendLine);
                }
            } else if (currentTrend === -1) { // If confirmed bearish
                 finalSupertrendLine = currentUpperBasic;
                 // Ensure line doesn't increase if continuing bearish
                 if (previousTrend === -1 && Number.isFinite(previousSupertrendLine)) {
                     finalSupertrendLine = Math.min(finalSupertrendLine, previousSupertrendLine);
                 }
            } else { // Trend is still neutral (0)
                finalSupertrendLine = NaN;
            }

            // Final check for finiteness of the calculated line
            if (!Number.isFinite(finalSupertrendLine)) {
                // logger.warn(c.yellow(c.bold("Warning (ehlersSupertrend):")) + c.yellow(` Calculated Supertrend line at index ${i} is non-finite. Setting to NaN.`)); // Too noisy
                finalSupertrendLine = NaN; // Ensure it's NaN
                // If line is NaN, trend should probably also be neutral?
                // currentTrend = 0; // Optional: Reset trend if line calculation fails? Let's keep trend based on band cross.
            }

            // Store result for this index
            supertrendResults[i] = { trend: currentTrend, value: finalSupertrendLine };
        }

        logger.debug(c.dim("Ehlers Supertrend calculation complete."));
        return supertrendResults;

    } catch (error) {
         logger.error(c.red(c.bold("Error (ehlersSupertrend):")) + c.yellow(` Unexpected calculation error. ${error.message}`), error.stack);
         return [];
    }
}

/**
 * Calculates Simple Moving Average (SMA).
 * Robustly handles input validation and calculation errors, including non-finite values in data.
 * @param {number[]} data - Array of numerical data.
 * @param {number} period - Period for SMA calculation. Must be a positive integer.
 * @param {object} logger - Logger instance.
 * @returns {number[]} - Array of SMA values, same length as input, or empty array on critical failure. Early values or those affected by NaNs will be NaN.
 */
function calculateSMA(data, period, logger) {
    // --- Input Validation ---
    if (!Array.isArray(data) || data.length === 0) { logger.error(c.red(c.bold("Error (calculateSMA):")) + c.yellow(" Invalid input data array.")); return []; }
    if (!Number.isInteger(period) || period <= 0) { logger.error(c.red(c.bold("Error (calculateSMA):")) + c.yellow(` Invalid period: ${period}. Must be a positive integer.`)); return []; }
    if (period > data.length) { logger.warn(c.yellow(c.bold("Warning (calculateSMA):")) + ` Period (${period}) > data length (${data.length}). SMA will be calculated based on available data.`); }

    const smaValues = new Array(data.length).fill(NaN); // Initialize with NaN
    let sum = 0;
    let validCountInWindow = 0; // Track valid numbers in the current window

    try {
        for (let i = 0; i < data.length; i++) {
            const value = data[i];
            const isValueValid = typeof value === 'number' && Number.isFinite(value);

            // Remove the value falling out of the window (if it was valid)
            if (i >= period) {
                const oldValue = data[i - period];
                if (typeof oldValue === 'number' && Number.isFinite(oldValue)) {
                    sum -= oldValue;
                    validCountInWindow--;
                }
            }

            // Add the new value if it's valid
            if (isValueValid) {
                 sum += value;
                 validCountInWindow++;
            } else {
                 // Only warn if the window should be full, otherwise initial NaNs are expected
                 // if (i >= period -1) { logger.warn(...) } // Reduce log noise
                 // Keep smaValues[i] as NaN
            }

            // Calculate SMA if there are valid numbers in the window
            // Ensure we only calculate if the effective window size is at least 1
            if (validCountInWindow > 0 && i >= Math.min(period - 1, data.length - 1)) {
                 // Average over only the valid points found within the effective window
                 const currentWindowSize = Math.min(i + 1, period); // Actual number of elements considered
                 if (validCountInWindow > 0) { // Double check
                    smaValues[i] = sum / validCountInWindow; // Average over valid points
                    if (!Number.isFinite(smaValues[i])) {
                        // logger.warn(c.yellow(c.bold("Warning (calculateSMA):")) + ` Resulting SMA value at index ${i} is non-finite despite valid inputs. Marking NaN.`); // Too noisy
                        smaValues[i] = NaN; // Ensure non-finite results are NaN
                    }
                 }
            } // else Keep smaValues[i] as NaN

        }
        return smaValues;
    } catch (error) {
        logger.error(c.red(c.bold("Error (calculateSMA):")) + c.yellow(` Unexpected calculation error. ${error.message}`), error.stack);
        return [];
    }
}

/**
 * Calculates Exponential Moving Average (EMA).
 * Robustly handles input validation and calculation errors, including non-finite values in data.
 * Uses SMA for initial seeding.
 * @param {number[]} data - Array of numerical data.
 * @param {number} period - Period for EMA calculation. Must be a positive integer.
 * @param {object} logger - Logger instance.
 * @returns {number[]} - Array of EMA values, same length as input, or empty array on critical failure. Values affected by NaNs will be NaN.
 */
function calculateEMA(data, period, logger) {
    // --- Input Validation ---
    if (!Array.isArray(data) || data.length === 0) { logger.error(c.red(c.bold("Error (calculateEMA):")) + c.yellow(" Invalid input data array.")); return []; }
    if (!Number.isInteger(period) || period <= 0) { logger.error(c.red(c.bold("Error (calculateEMA):")) + c.yellow(` Invalid period: ${period}. Must be a positive integer.`)); return []; }
    if (period > data.length) { logger.warn(c.yellow(c.bold("Warning (calculateEMA):")) + ` Period (${period}) > data length (${data.length}). EMA results might be less meaningful initially.`); }

    const emaValues = new Array(data.length).fill(NaN); // Initialize with NaN
    const smoothingFactor = 2 / (period + 1);
    let emaPrev = NaN; // Stores the previous valid EMA value
    let initialSmaCalculated = false;

    try {
        // Calculate initial SMA for seeding if enough data exists
        const initialSma = calculateSMA(data.slice(0, period), period, logger);
        if (initialSma.length > 0 && Number.isFinite(initialSma[initialSma.length - 1])) {
            emaPrev = initialSma[initialSma.length - 1];
            emaValues[period - 1] = emaPrev; // Set the first EMA value at the end of the SMA period
            initialSmaCalculated = true;
            logger.debug(`EMA seeded with SMA(${period}) value: ${emaPrev}`);
        } else {
            logger.debug(`Could not seed EMA with SMA(${period}). Will seed with first valid data point.`);
        }

        const startIndex = initialSmaCalculated ? period : 0;

        for (let i = startIndex; i < data.length; i++) {
            const value = data[i];
            const isValueValid = typeof value === 'number' && Number.isFinite(value);

            if (isValueValid) {
                // Initialize the first EMA value if SMA seeding failed or previous was NaN
                if (!Number.isFinite(emaPrev)) {
                    emaValues[i] = value; // Seed with the current valid data point
                } else {
                    // Calculate EMA using the standard formula
                    emaValues[i] = (value - emaPrev) * smoothingFactor + emaPrev;
                }

                // Check if the calculated EMA is finite
                if (!Number.isFinite(emaValues[i])) {
                     // logger.warn(c.yellow(c.bold("Warning (calculateEMA):")) + ` Resulting EMA value at index ${i} is non-finite. Resetting EMA for next calculation.`); // Too noisy
                     emaValues[i] = NaN; // Ensure it's NaN
                     emaPrev = NaN; // Reset emaPrev so next valid point restarts calculation
                } else {
                    // Update emaPrev for the next iteration ONLY if the current calculation was successful
                    emaPrev = emaValues[i];
                }
            } else {
                 // logger.warn(...) // Reduce log noise
                 emaValues[i] = NaN; // Ensure output is NaN if input is invalid
                 // Previous EMA (emaPrev) remains unchanged, calculation will continue from there if next value is valid
            }
        }
        return emaValues;
    } catch (error) {
        logger.error(c.red(c.bold("Error (calculateEMA):")) + c.yellow(` Unexpected calculation error. ${error.message}`), error.stack);
        return [];
    }
}


/**
 * Calculates Volume Moving Averages (VMA) using SMA.
 * @param {number[]} volumeData - Array of volume data.
 * @param {number} shortPeriod - Period for short-term VMA. Must be positive integer.
 * @param {number} longPeriod - Period for long-term VMA. Must be positive integer, > shortPeriod.
 * @param {object} logger - Logger instance.
 * @returns {{vmaShort: number[], vmaLong: number[]}} - Object containing VMA arrays, or empty arrays on critical failure.
 */
function calculateVMA(volumeData, shortPeriod, longPeriod, logger) {
    const defaultReturn = { vmaShort: [], vmaLong: [] };
    // --- Input Validation ---
    if (!Array.isArray(volumeData)) { logger.error(c.red(c.bold("Error (calculateVMA):")) + c.yellow(" Invalid volume data array.")); return defaultReturn; }
    if (!Number.isInteger(shortPeriod) || shortPeriod <= 0) { logger.error(c.red(c.bold("Error (calculateVMA):")) + c.yellow(` Invalid shortPeriod: ${shortPeriod}. Must be a positive integer.`)); return defaultReturn; }
    if (!Number.isInteger(longPeriod) || longPeriod <= 0) { logger.error(c.red(c.bold("Error (calculateVMA):")) + c.yellow(` Invalid longPeriod: ${longPeriod}. Must be a positive integer.`)); return defaultReturn; }
    if (shortPeriod >= longPeriod) { logger.error(c.red(c.bold("Error (calculateVMA):")) + c.yellow(" shortPeriod must be less than longPeriod.")); return defaultReturn; }
    if (longPeriod > volumeData.length) { logger.warn(c.yellow(c.bold("Warning (calculateVMA):")) + ` longPeriod (${longPeriod}) > data length (${volumeData.length}). Results may be inaccurate.`); }

    // Calculate SMAs for volume using the robust SMA function
    const vmaShort = calculateSMA(volumeData, shortPeriod, logger);
    const vmaLong = calculateSMA(volumeData, longPeriod, logger);

    // Check if SMA calculations were successful (returned arrays of expected length)
    if (vmaShort.length !== volumeData.length || vmaLong.length !== volumeData.length) {
        logger.error(c.red(c.bold("Error (calculateVMA):")) + c.yellow(" SMA calculation for VMA failed or returned unexpected length."));
        return defaultReturn; // Return empty arrays if SMA failed
    }

    return { vmaShort, vmaLong };
}

/**
 * Calculates Order Book Pressure based on cumulative bid/ask volumes within the fetched depth.
 * @param {ccxt.OrderBook | null} orderBook - Order book data from exchange.fetchOrderBook(). Expected structure: { bids: [[price, volume], ...], asks: [[price, volume], ...] }.
 * @param {object} logger - Logger instance.
 * @returns {number} - Order book pressure (0 to 1, higher = more buy-side volume pressure). Returns 0.5 (neutral) if order book is invalid, empty, or calculation fails.
 */
function calculateOrderBookPressure(orderBook, logger) {
    // --- Input Validation ---
    if (!orderBook || typeof orderBook !== 'object' || !Array.isArray(orderBook.bids) || !Array.isArray(orderBook.asks)) {
        logger.warn(c.yellow("Invalid or incomplete order book structure received. Cannot calculate pressure. Returning neutral (0.5)."));
        return 0.5; // Neutral pressure if structure is wrong
    }
    if (orderBook.bids.length === 0 && orderBook.asks.length === 0) {
         logger.debug("Order book is empty (no bids or asks). Returning neutral pressure (0.5).");
         return 0.5; // Neutral if book is empty
    }

    let totalBidVolume = 0;
    let totalAskVolume = 0;

    try {
        // Sum bid volumes
        for (const bid of orderBook.bids) { // Expected format: [price, volume, ...]
            if (Array.isArray(bid) && bid.length >= 2 && typeof bid[1] === 'number' && Number.isFinite(bid[1]) && bid[1] > 0) {
                totalBidVolume += bid[1];
            } // else { logger.debug(...) } // Reduce log noise
        }
        // Sum ask volumes
        for (const ask of orderBook.asks) { // Expected format: [price, volume, ...]
            if (Array.isArray(ask) && ask.length >= 2 && typeof ask[1] === 'number' && Number.isFinite(ask[1]) && ask[1] > 0) {
                totalAskVolume += ask[1];
            } // else { logger.debug(...) } // Reduce log noise
        }
    } catch (e) {
        logger.error(c.red(c.bold("Error (calculateOrderBookPressure):")) + c.yellow(` Failed during volume summation. ${e.message}`), orderBook);
        return 0.5; // Return neutral on unexpected error
    }

    const totalVolume = totalBidVolume + totalAskVolume;

    // Handle case where total volume is zero (e.g., only invalid entries found)
    if (totalVolume <= 0) {
        logger.debug("Total valid bid/ask volume is zero or negative. Returning neutral pressure (0.5).");
        return 0.5;
    }

    // Calculate pressure: Ratio of bid volume to total volume
    const buyPressure = totalBidVolume / totalVolume;

    // Final check for sanity
    if (!Number.isFinite(buyPressure)) {
        logger.warn(c.yellow(c.bold("Warning (calculateOrderBookPressure):")) + ` Calculated pressure is non-finite (${buyPressure}). Returning neutral (0.5).`);
        return 0.5;
    }

    logger.debug(`Order Book Pressure: Bids Vol=${totalBidVolume.toFixed(4)}, Asks Vol=${totalAskVolume.toFixed(4)}, Total Vol=${totalVolume.toFixed(4)}, Pressure=${buyPressure.toFixed(3)}`);
    return buyPressure; // Value between 0 and 1
}


// --- Helper Functions for Exchange Interaction (Extracted from ExchangeManager) ---

/**
 * Sets up and configures the Bybit CCXT exchange instance.
 * @param {Config} config - The bot's configuration.
 * @returns {ccxt.Exchange} - The configured CCXT exchange instance.
 * @throws {Error} If setup fails critically (e.g., missing keys, CCXT class not found).
 */
function setupBybitExchange(config) {
    logger.info(c.blue(`Initializing Bybit ${config.exchange_type} exchange connection...`));
    let apiKey, apiSecret;

    if (config.dry_run) {
        logger.info(c.magenta("Dry Run mode enabled. Using dummy API keys. Real connection might still be attempted for market data."));
        // Provide unique dummy keys to satisfy CCXT constructor, though they won't authenticate
        apiKey = "DRY_RUN_API_KEY_" + Date.now() + Math.random();
        apiSecret = "DRY_RUN_API_SECRET_" + Date.now() + Math.random();
    } else {
        if (!config.bybit_api_key || !config.bybit_api_secret) {
            // This error should ideally be caught by config validation, but double-check
            throw new Error("CRITICAL: API Key and Secret are required for live trading (DRY_RUN=false). Check .env file.");
        }
        apiKey = config.bybit_api_key;
        apiSecret = config.bybit_api_secret;
    }

    try {
        // Ensure the exchange class exists in CCXT
        if (!ccxt.hasOwnProperty('bybit')) {
            throw new Error("CCXT 'bybit' exchange class not found. Ensure CCXT library is installed correctly (`npm install ccxt`).");
        }

        const exchangeOptions = {
            apiKey: apiKey,
            secret: apiSecret,
            enableRateLimit: true, // Enable CCXT's built-in rate limiter
            options: {
                defaultType: config.exchange_type, // 'swap', 'future', 'spot' - Helps CCXT select endpoints
                adjustForTimeDifference: true, // Auto-sync client time with server time
                recvWindow: 15000, // Increased request timeout window (milliseconds)
                // Bybit V5 specific options:
                fetchPositionsRequiresSymbol: true, // Bybit V5 requires symbol for fetchPositions
                // It's generally safer to pass 'category' in individual API call params rather than globally
                // Bybit V5 Unified Margin requires specific header for HTTP requests if using UM account
                // 'unifiedMargin': config.bybit_account_type === 'UNIFIED', // Might be handled by CCXT
            }
        };

        // Instantiate the Bybit exchange
        const exchange = new ccxt.bybit(exchangeOptions);

        // Set sandbox mode if needed (requires separate API keys for testnet)
        if (process.env.BYBIT_SANDBOX_MODE === 'true') {
             exchange.setSandboxMode(true);
             logger.info(c.yellow("Sandbox mode enabled via BYBIT_SANDBOX_MODE=true. Using testnet endpoints. Ensure testnet API keys are set."));
        }

        logger.info(`CCXT ${c.bold(exchange.id)} instance created (Version: ${c.dim(exchange.version || 'N/A')}). Default Type: ${config.exchange_type}`);
        return exchange;

    } catch (e) {
        logger.error(c.red(`FATAL: Unexpected error during CCXT exchange setup: ${e.message}`), e.stack);
        throw new Error(`Exchange setup failed: ${e.message}`); // Re-throw as a critical error
    }
}

/**
 * Loads market data from the exchange, validates the configured symbol, type, and contract details.
 * Updates the config object with the standardized symbol from the exchange.
 * @param {ccxt.Exchange} exchange - The initialized CCXT exchange instance.
 * @param {Config} config - The bot's configuration (will be updated with standardized symbol).
 * @returns {Promise<void>}
 * @throws {Error} If markets fail to load, symbol is invalid, type mismatch, or contract validation fails.
 */
async function loadMarketsAndValidate(exchange, config) {
    if (!exchange) throw new Error("Exchange not initialized before loading markets.");

    try {
        logger.info(`Loading exchange markets for ${exchange.id} (this may take a moment)...`);
        // Wrap the loadMarkets call in retry logic
        const loadMarketsFunc = async () => await exchange.loadMarkets();
        await retryOnException(loadMarketsFunc, config.max_retries, config.retry_delay, undefined, 'loadMarkets'); // Use default retryable errors
        logger.info(c.green(`Successfully loaded ${Object.keys(exchange.markets || {}).length} markets from ${exchange.id}.`));

        // --- Symbol and Type Validation ---
        const symbol = config.symbol; // Use the symbol from config initially
        const exchangeType = config.exchange_type; // 'swap', 'future', 'spot'
        logger.info(`Validating symbol '${c.bold(symbol)}' and type '${c.bold(exchangeType)}'...`);

        let market;
        try {
            // Retrieve market data for the symbol
            market = exchange.market(symbol);
            if (!market) {
                // Throw a specific error if market() returns undefined/null
                throw new ccxt.BadSymbol(`Market data not found for symbol '${symbol}' on ${exchange.id}.`);
            }
        } catch (e) {
            // Handle CCXT's BadSymbol error specifically
            if (e instanceof ccxt.BadSymbol) {
                // Try to find similar symbols for helpful logging
                const availableSymbolsSample = Object.keys(exchange.markets || {})
                    .filter(s => exchange.markets[s]?.type === exchangeType) // Filter by configured type
                    .slice(0, 15); // Show a small sample
                logger.error(c.red(`Symbol Error: ${e.message}`));
                logger.error(c.yellow(`Please ensure the SYMBOL ('${symbol}') is correct for ${exchange.id} and matches the EXCHANGE_TYPE ('${exchangeType}').`));
                if (availableSymbolsSample.length > 0) {
                    logger.error(`Available symbols sample for type '${exchangeType}': ${availableSymbolsSample.join(', ')}...`);
                } else {
                    logger.error(`Could not find any available symbols matching type '${exchangeType}'. Check EXCHANGE_TYPE or market loading.`);
                }
                // Re-throw a more user-friendly error
                throw new Error(`Symbol '${symbol}' not found or invalid for the configured exchange type '${exchangeType}'.`);
            } else {
                // Handle other potential errors during market retrieval
                logger.error(c.red(`Unexpected error retrieving market data for ${symbol}: ${e.message}`), e.stack);
                throw new Error(`Failed to get market data for validation: ${e.message}`);
            }
        }

        // --- Type Validation ---
        // Check if the market's type matches the configured type
        if (market.type !== exchangeType) {
            throw new Error(`Market Type Mismatch: Symbol ${symbol}'s actual type ('${market.type}') does not match configured EXCHANGE_TYPE ('${exchangeType}').`);
        }

        // --- Contract Type Validation (Only for swap/future) ---
        if (exchangeType === 'swap' || exchangeType === 'future') {
            // Check if it's a LINEAR contract (settled in Quote currency, e.g., USDT)
             if (!market.linear) {
                 // If CCXT doesn't explicitly mark it linear, double-check the settle currency
                 if (!market.settle || market.settle.toUpperCase() !== config.currency.toUpperCase()) {
                    throw new Error(`Contract Type Error: Symbol ${symbol} is not a LINEAR contract settling in ${config.currency}. Detected Settle Currency: ${market.settle ?? 'Unknown'}. This bot only supports linear contracts.`);
                 } else {
                     // If settle currency matches but linear flag is false/missing, issue a warning but proceed cautiously
                     logger.warn(c.yellow(`Market ${symbol} lacks explicit 'linear=true' flag, but settles in ${config.currency}. Proceeding, but verify market data if issues arise.`));
                 }
             } else if (market.settle && market.settle.toUpperCase() !== config.currency.toUpperCase()) {
                 // If marked linear but settles in the wrong currency (data inconsistency)
                 logger.warn(c.yellow(`Market ${symbol} is marked linear but settles in ${market.settle}. Expected ${config.currency}. This might indicate incorrect market data from the exchange.`));
                 throw new Error(`Contract Settle Currency Mismatch: Symbol ${symbol} is linear but settles in ${market.settle}, expected ${config.currency}.`);
             }
             logger.debug(`Contract type validated for ${symbol}: Linear, Settle=${market.settle}`);
        }

        // --- Standardize Symbol ---
        // Use the symbol format provided by the exchange market data
        const standardizedSymbol = market.symbol;
        if (standardizedSymbol !== symbol) {
            logger.info(`Standardizing symbol format from '${symbol}' to '${c.bold(standardizedSymbol)}' based on exchange market data.`);
            config.symbol = standardizedSymbol; // Update config in place
        }

        // --- Log Precision and Limits ---
        logger.debug(`Market Precision for ${config.symbol}: Price=${market.precision?.price}, Amount=${market.precision?.amount}, Base=${market.precision?.base}, Quote=${market.precision?.quote}`);
        logger.debug(`Market Limits for ${config.symbol}: Amount(Min=${market.limits?.amount?.min}, Max=${market.limits?.amount?.max}), Price(Min=${market.limits?.price?.min}, Max=${market.limits?.price?.max}), Cost(Min=${market.limits?.cost?.min}, Max=${market.limits?.cost?.max})`);
        logger.debug(`Market Tick Size: ${market.info?.tickSize || 'N/A'}`); // Bybit specific tick size

        // Check for required precision/limits
        if (market.precision?.price === undefined || market.precision?.amount === undefined || market.limits?.amount?.min === undefined) {
             logger.warn(c.yellow(`Market ${config.symbol} is missing required precision (price/amount) or minimum amount limit information. Order placement might fail or be inaccurate.`));
             // Consider throwing an error if these are absolutely critical
             // throw new Error(`Market ${config.symbol} is missing critical precision/limit information.`);
        }

        logger.info(c.green(`Symbol '${c.bold(config.symbol)}' validated successfully (Type: ${market.type}, Linear: ${market.linear ?? 'N/A'}, Settle: ${market.settle ?? 'N/A'}).`));

    } catch (e) {
        // Catch specific errors from the validation process or retries
        if (e instanceof ccxt.AuthenticationError) {
            logger.error(c.red(`Exchange Authentication Failed during market load/validation: ${e.message}. Check API keys/permissions.`), e.stack);
            throw new Error(`Authentication Failed: ${e.message}`); // Critical
        } else if (e instanceof ccxt.NetworkError || e instanceof ccxt.ExchangeNotAvailable || e instanceof ccxt.RequestTimeout || e instanceof ccxt.DDoSProtection) {
            logger.error(c.red(`Failed to connect or load markets after retries: ${e.message}`), e.stack);
            throw new Error(`Exchange connection/market load failed: ${e.message}`); // Critical
        } else if (e instanceof Error && (e.message.includes('Symbol') || e.message.includes('Market Type') || e.message.includes('Contract Type') || e.message.includes('Settle Currency'))) {
            // Catch the specific validation errors thrown above
            logger.error(c.red(`Configuration Validation Failed: ${e.message}`));
            throw e; // Re-throw validation errors
        } else {
            // Catch any other unexpected errors
            logger.error(c.red(`Unexpected error during market loading/validation: ${e.message}`), e.stack);
            throw new Error(`Market loading/validation failed unexpectedly: ${e.message}`); // Critical
        }
    }
}

/**
 * Sets the leverage for the configured symbol on the exchange.
 * Handles potential errors and warnings gracefully.
 * @param {ccxt.Exchange} exchange - The initialized CCXT exchange instance.
 * @param {Config} config - The bot's configuration.
 * @returns {Promise<void>}
 * @throws {Error} If leverage setting fails critically (e.g., auth error, invalid value).
 */
async function setLeverage(exchange, config) {
    const symbol = config.symbol;
    const leverage = config.leverage;

    // Pre-checks
    if (!exchange) { logger.warn(c.yellow("Cannot set leverage: Exchange not initialized.")); return; }
    if (!exchange.has['setLeverage']) { logger.warn(c.yellow(`Exchange ${exchange.id} does not natively support the setLeverage() method via CCXT. Skipping leverage setting.`)); return; }
    if (config.exchange_type === 'spot') { logger.info("Leverage setting is not applicable for spot markets. Skipping."); return; }

    logger.info(`Attempting to set leverage for ${c.bold(symbol)} to ${c.bold(leverage)}x...`);
    try {
        // Prepare parameters, especially for Bybit V5 which requires category
        // and often setting buy/sell leverage together.
        const params = {};
        if (exchange.id === 'bybit' && ['swap', 'future'].includes(config.exchange_type)) {
            params.category = 'linear'; // Essential for Bybit V5 linear contracts
            params.buyLeverage = leverage;
            params.sellLeverage = leverage;
            logger.debug(`Using Bybit V5 params for setLeverage: ${JSON.stringify(params)}`);
        }

        // Define the async function to call
        const setLeverageFunc = async () => await exchange.setLeverage(leverage, symbol, params);

        // Execute with retry logic
        const response = await retryOnException(
            setLeverageFunc,
            config.max_retries,
            config.retry_delay,
            undefined, // Use default retryable errors
            'setLeverage'
        );

        // Log success (response might be minimal or verbose depending on exchange)
        // Check Bybit V5 response structure for confirmation if possible
        let confirmationMsg = "(No detailed response provided by exchange)";
        if (exchange.id === 'bybit' && response?.info?.retCode === 0) {
            confirmationMsg = c.green(`Confirmed by Bybit V5 (retCode=0).`);
        } else if (response) {
             confirmationMsg = `Raw Response: ${inspect(response, {depth: 1})}`;
        }
        logger.info(c.green(`Leverage set command successfully executed for ${symbol}. ${confirmationMsg}`));

    } catch (e) {
        // Handle specific errors after retries have failed
        if (e instanceof ccxt.ExchangeError) {
            const msgLower = e.message.toLowerCase();
            // Common benign messages indicating leverage is already set
            if (msgLower.includes("leverage not modified") || msgLower.includes("same leverage") || e.message.includes("ret_code=30036")) { // 30036: Bybit code for leverage not modified
                logger.info(`Leverage for ${symbol} is already set to ${leverage}x.`);
            }
            // Cannot change leverage with open position/orders
            else if (msgLower.includes("position exists") || msgLower.includes("open position") || msgLower.includes("open order") || msgLower.includes("active order") || e.message.includes("ret_code=30018") || e.message.includes("ret_code=30019")) { // Bybit codes for position/order exists
                logger.warn(c.yellow(`Cannot modify leverage for ${symbol} to ${leverage}x because of an existing position or active orders. Using current exchange leverage setting.`));
                // Consider fetching current position leverage here if needed for verification?
            }
            // Insufficient margin (less common for just setting leverage, but possible)
            else if (msgLower.includes("insufficient margin") || msgLower.includes("margin is insufficient")) {
                logger.error(c.red(`Failed to set leverage for ${symbol} due to insufficient margin: ${e.message}. Check account balance.`));
                // This might be critical depending on strategy assumptions
            }
            // Invalid leverage value or outside allowed range
            else if (msgLower.includes("invalid leverage") || msgLower.includes("leverage limit") || msgLower.includes("parameter error") || e.message.includes("ret_code=30037")) { // 30037: Invalid leverage
                logger.error(c.red(`Invalid leverage value (${leverage}) or value is outside the allowed limits for ${symbol}: ${e.message}`));
                throw new Error(`Invalid leverage configuration for ${symbol}: ${e.message}`); // Critical config error
            }
            // Other exchange-specific errors
            else {
                logger.error(c.red(`Failed to set leverage for ${symbol} due to exchange error: ${e.constructor.name} - ${e.message}`), e.stack);
                // Decide if this is critical or just a warning
                // Potentially throw error to halt bot if leverage setting is crucial and fails unexpectedly
                throw new Error(`Failed to set leverage: ${e.message}`);
            }
        } else if (e instanceof ccxt.AuthenticationError) {
            logger.error(c.red(`Authentication failed while setting leverage: ${e.message}. Check API key permissions (Trade permissions needed).`));
            throw new Error(`Authentication failed setting leverage: ${e.message}`); // Critical
        } else if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout) {
            // If retries failed due to network issues
            logger.warn(c.yellow(`Could not set leverage due to persistent connection issue after retries: ${e.message}. The bot will proceed, assuming the current leverage setting on the exchange is acceptable.`));
        } else {
            // Unexpected errors
            logger.error(c.red(`Unexpected error setting leverage: ${e.constructor.name} - ${e.message}`), e.stack);
            throw new Error(`Unexpected error setting leverage: ${e.message}`); // Treat as critical
        }
    }
}

/**
 * Fetches OHLCV (candlestick) data using the ExchangeManager's cache and retry logic.
 * Validates the structure of the returned data.
 * @param {ExchangeManager} exchangeManager - Instance of ExchangeManager containing the CCXT instance and config.
 * @param {number} limit - The maximum number of candles to fetch.
 * @returns {Promise<Array<Array<number>> | null>} OHLCV data as an array of arrays, or null on failure or if no data is returned.
 */
async function fetchOHLCV(exchangeManager, limit) {
    const { config, exchange } = exchangeManager;
    const cacheName = "ohlcv";
    const cacheKey = `${config.symbol}_${config.timeframe}_${limit}`; // Unique key for this request

    // Try fetching from cache first
    const cachedData = exchangeManager._getCache(cacheName, cacheKey);
    if (cachedData) return cachedData; // Return cached data if valid

    logger.debug(`Fetching ${limit} OHLCV candles for ${config.symbol} (${config.timeframe})...`);

    // Define the async function for the API call
    const fetchFunc = async () => {
        if (!exchange || !exchange.has['fetchOHLCV']) {
            throw new ccxt.NotSupported("Exchange object not initialized or does not support fetchOHLCV method.");
        }
        // Prepare parameters, especially Bybit V5 category
        const params = {};
        if (exchange.id === 'bybit' && ['swap', 'future'].includes(config.exchange_type)) {
            params.category = 'linear'; // Required for Bybit V5 linear contracts
        }
        // Call fetchOHLCV with symbol, timeframe, since (undefined), limit, and params
        return await exchange.fetchOHLCV(config.symbol, config.timeframe, undefined, limit, params);
    };

    try {
        // Execute the fetch function with retry logic
        const ohlcvData = await retryOnException(fetchFunc, config.max_retries, config.retry_delay, undefined, 'fetchOHLCV');

        // --- Validate the returned data ---
        if (!Array.isArray(ohlcvData)) {
            logger.warn(c.yellow(`fetchOHLCV returned non-array data type: ${typeof ohlcvData}. Expected an array.`), ohlcvData);
            return null; // Return null if data is not an array
        }
        // Check if array is empty (valid response, but no data)
        if (ohlcvData.length === 0) {
            logger.debug("fetchOHLCV returned 0 candles.");
            // Cache the empty result to avoid refetching immediately
            exchangeManager._setCache(cacheName, cacheKey, ohlcvData);
            return ohlcvData; // Return the empty array
        }
        // Check the structure of the first candle (should be an array with at least 6 elements)
        if (!Array.isArray(ohlcvData[0]) || ohlcvData[0].length < OHLCV_SCHEMA.length) {
            logger.warn(c.yellow(`Received malformed OHLCV data structure. Expected array of arrays with >= ${OHLCV_SCHEMA.length} elements (Timestamp, O, H, L, C, V). Received:`), ohlcvData[0]);
            return null; // Return null if structure is wrong
        }
        // Basic check for valid numbers in the last candle (more robust than checking all)
        const lastCandle = ohlcvData[ohlcvData.length - 1];
        if (lastCandle.some((val, idx) => idx < OHLCV_SCHEMA.length && (typeof val !== 'number' || !Number.isFinite(val)))) {
             logger.warn(c.yellow(`Last candle in fetched OHLCV data contains non-finite numbers:`), lastCandle);
             // Decide whether to return null or proceed cautiously
             // return null;
        }

        // --- Cache and Return ---
        exchangeManager._setCache(cacheName, cacheKey, ohlcvData); // Cache the valid data
        const lastCandleTime = new Date(ohlcvData[ohlcvData.length - 1][OHLCV_INDEX.TIMESTAMP]).toISOString();
        logger.debug(`Fetched ${ohlcvData.length} OHLCV candles successfully. Last candle time: ${lastCandleTime}`);
        return ohlcvData;

    } catch (e) {
        // Log errors after retries have failed
        logger.error(c.red(`Failed to fetch OHLCV data for ${config.symbol} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
        return null; // Return null on failure
    }
}

/**
 * Fetches Order Book data using the ExchangeManager's cache and retry logic.
 * Validates the structure of the returned data.
 * @param {ExchangeManager} exchangeManager - Instance of ExchangeManager.
 * @returns {Promise<ccxt.OrderBook | null>} Order book object { bids: [[price, amount], ...], asks: [[price, amount], ...], ... } or null on failure.
 */
async function fetchOrderBookData(exchangeManager) {
    const { config, exchange } = exchangeManager;
    const cacheName = "order_book";
    const depth = config.order_book_depth;
    const cacheKey = `${config.symbol}_${depth}`;

    // Try cache first
    const cachedData = exchangeManager._getCache(cacheName, cacheKey);
    if (cachedData) return cachedData;

    logger.debug(`Fetching order book for ${config.symbol} (depth: ${depth})...`);

    // Define the async function for the API call
    const fetchFunc = async () => {
        if (!exchange || !exchange.has['fetchOrderBook']) {
            throw new ccxt.NotSupported("Exchange object not initialized or does not support fetchOrderBook method.");
        }
        // Prepare parameters (e.g., Bybit V5 category)
        const params = {};
        if (exchange.id === 'bybit' && ['swap', 'future'].includes(config.exchange_type)) {
            params.category = 'linear'; // Required for Bybit V5
        }
        // Call fetchOrderBook with symbol, depth, and params
        return await exchange.fetchOrderBook(config.symbol, depth, params);
    };

    try {
        // Execute with retry logic
        const orderBookData = await retryOnException(fetchFunc, config.max_retries, config.retry_delay, undefined, 'fetchOrderBook');

        // --- Validate the returned data ---
        // Basic structure check
        if (!orderBookData || typeof orderBookData !== 'object' || !Array.isArray(orderBookData.bids) || !Array.isArray(orderBookData.asks) || typeof orderBookData.timestamp !== 'number') {
            logger.warn(c.yellow("fetchOrderBook returned invalid, incomplete, or malformed data structure."), orderBookData);
            return null;
        }
        // Optional: Check format of bids/asks arrays if needed:
        const isValidEntry = (entry) => Array.isArray(entry) && entry.length >= 2 && typeof entry[0] === 'number' && typeof entry[1] === 'number';
        if ((orderBookData.bids.length > 0 && !isValidEntry(orderBookData.bids[0])) ||
            (orderBookData.asks.length > 0 && !isValidEntry(orderBookData.asks[0]))) {
            logger.warn(c.yellow("Order book bids/asks entries have incorrect format [price, amount]."), orderBookData);
            return null;
        }

        // --- Cache and Return ---
        exchangeManager._setCache(cacheName, cacheKey, orderBookData);
        logger.debug(`Fetched order book successfully: ${orderBookData.bids.length} bids, ${orderBookData.asks.length} asks. Timestamp: ${new Date(orderBookData.timestamp).toISOString()}`);
        return orderBookData;

    } catch (e) {
        logger.error(c.red(`Failed to fetch order book for ${config.symbol} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
        return null;
    }
}

/**
 * @typedef {object} PositionState - Represents the bot's view of the current position.
 * @property {PositionSide} side - The side of the position ('long', 'short', 'none').
 * @property {number} size - The absolute size of the position (positive number).
 * @property {number} entryPrice - The average entry price of the position.
 */

/**
 * Fetches current position data using the ExchangeManager's cache and retry logic.
 * Handles aggregation for exchanges that might return multiple position entries (e.g., hedge mode).
 * Returns a standardized position object.
 * @param {ExchangeManager} exchangeManager - Instance of ExchangeManager.
 * @returns {Promise<PositionState>} Position details. Returns default state ({ side: 'none', size: 0, entryPrice: 0 }) on failure or no position.
 */
async function getExchangePosition(exchangeManager) {
    const { config, exchange } = exchangeManager;
    const cacheName = "position";
    const cacheKey = config.symbol; // Cache key based on symbol

    // Default return value representing no position
    const defaultReturn = { side: PositionSide.NONE, size: 0.0, entryPrice: 0.0 };

    // Try cache first
    const cachedData = exchangeManager._getCache(cacheName, cacheKey);
    // Basic validation of cached data structure
    if (cachedData && typeof cachedData === 'object' &&
        cachedData.side !== undefined && cachedData.size !== undefined && cachedData.entryPrice !== undefined &&
        Object.values(PositionSide).includes(cachedData.side) &&
        typeof cachedData.size === 'number' && typeof cachedData.entryPrice === 'number') {
        return cachedData; // Return valid cached data
    } else if (cachedData) {
        logger.warn(c.yellow("Invalid position data found in cache. Refetching."));
        exchangeManager._setCache(cacheName, cacheKey, null); // Clear invalid cache entry
    }

    // Handle Dry Run mode
    if (config.dry_run) {
        // In dry run, we rely entirely on the bot's internal state (`this.currentPosition`).
        // Fetching position doesn't make sense. We return the *default* state here,
        // expecting the calling function (like `updateCurrentPositionState`) to handle the dry run logic
        // by using the bot's internal state instead of this fetched (default) value.
        logger.debug(c.magenta("DRY RUN: Skipping actual position fetch. Returning default state (no position). Bot logic should use internal state."));
        return defaultReturn;
    }

    logger.debug(`Fetching position for ${config.symbol}...`);

    // Define the async function for the API call
    const fetchFunc = async () => {
        if (!exchange) throw new Error("Exchange object not initialized.");

        // Prepare parameters (e.g., Bybit V5 category)
        const params = {};
        if (exchange.id === 'bybit' && ['swap', 'future'].includes(config.exchange_type)) {
            params.category = 'linear'; // Required for V5 linear contracts
        }

        // Use fetchPositions if available (preferred, handles multiple positions/hedge mode)
        // Bybit V5 requires symbol(s) for fetchPositions
        if (exchange.has['fetchPositions'] && exchange.hasFetchPositionsRequiresSymbol !== false) {
            // Fetch for the specific symbol
            const positions = await exchange.fetchPositions([config.symbol], params);
            // Filter results just in case the exchange returns more than requested
            return positions.filter(p => p && p.symbol === config.symbol);
        }
        // Fallback to fetchPosition if fetchPositions isn't suitable or available
        else if (exchange.has['fetchPosition']) {
             logger.debug("Using fetchPosition(symbol, params) fallback...");
             const position = await exchange.fetchPosition(config.symbol, params);
             // Wrap it in an array for consistent processing below
             return (position && position.symbol === config.symbol) ? [position] : [];
        } else {
            throw new ccxt.NotSupported("Exchange does not support fetchPositions or fetchPosition methods required to get position data.");
        }
    };

    try {
        // Execute with retry logic
        const relevantPositions = await retryOnException(fetchFunc, config.max_retries, config.retry_delay, undefined, 'getPosition');

        // --- Process the returned position data ---
        if (!Array.isArray(relevantPositions)) {
             logger.warn(c.yellow(`Position fetch returned non-array data: ${typeof relevantPositions}. Assuming no position.`), relevantPositions);
             exchangeManager._setCache(cacheName, cacheKey, defaultReturn);
             return defaultReturn;
        }
        if (relevantPositions.length === 0) {
            logger.debug(`No open positions found for symbol ${config.symbol}.`);
            exchangeManager._setCache(cacheName, cacheKey, defaultReturn);
            return defaultReturn;
        }

        // --- Aggregate position info (handles multiple entries, e.g., hedge mode) ---
        // For Bybit V5 linear in One-Way mode, there should only be one entry per symbol.
        let netSize = 0.0;
        let totalValue = 0.0; // Sum of (abs(size) * entryPrice)
        let totalAbsSize = 0.0; // Sum of abs(size)
        const market = exchange.market(config.symbol); // Get market info for precision/tolerance
        const sizeTolerance = Math.max(1e-9, market?.limits?.amount?.min / 10 || 1e-9);

        for (const pos of relevantPositions) {
            // Extract data, preferring 'info' field for exchange-specific details if available
            const info = pos.info || {};
            // Bybit V5 fields: info.size, info.avgPrice, info.side ('Buy'/'Sell' maps to long/short)
            // CCXT standard fields: pos.contracts, pos.entryPrice, pos.side ('long'/'short')
            const sizeStr = info.size ?? pos.contracts ?? pos.contractSize ?? '0'; // Prefer info.size
            const entryPriceStr = info.avgPrice ?? pos.entryPrice ?? '0'; // Prefer info.avgPrice
            // Determine side: Bybit V5 'Buy'/'Sell' or CCXT 'long'/'short'
            let side = (info.side || pos.side || '').toLowerCase();
            if (side === 'buy') side = 'long';
            if (side === 'sell') side = 'short';

            let sizeNum = 0.0, entryPriceNum = 0.0;
            try {
                // Parse values carefully
                sizeNum = parseFloat(sizeStr);
                entryPriceNum = parseFloat(entryPriceStr);
                if (!Number.isFinite(sizeNum) || !Number.isFinite(entryPriceNum)) {
                    throw new Error(`Parsed size (${sizeStr}) or entry price (${entryPriceStr}) is non-finite.`);
                }
            } catch (parseError) {
                logger.warn(c.yellow(`Could not parse size ('${sizeStr}') or entry price ('${entryPriceStr}') for a position entry. Skipping this entry. Error: ${parseError.message}. Raw Entry:`), pos);
                continue; // Skip this position entry
            }

            const absSize = Math.abs(sizeNum);

            // Skip negligible "dust" positions
            if (absSize < sizeTolerance) {
                 logger.debug(`Skipping negligible position entry: Size=${sizeNum}, Side=${side}`);
                 continue;
            }

            // Accumulate net size and weighted value
            if (side === 'long') {
                netSize += sizeNum;
                totalValue += absSize * entryPriceNum;
                totalAbsSize += absSize;
            } else if (side === 'short') {
                // Note: Bybit V5 size for Sell side is positive. Net size calculation needs direction.
                netSize -= sizeNum; // Short position contributes negatively to net size
                totalValue += absSize * entryPriceNum;
                totalAbsSize += absSize;
            } else {
                logger.warn(c.yellow(`Skipping position entry with unrecognized side: '${side}'. Raw Entry:`), pos);
            }
        } // End loop through position entries

        // --- Determine final aggregated state ---
        let finalSide = PositionSide.NONE;
        let finalSize = 0.0;
        let finalEntryPrice = 0.0;

        // Check net size against tolerance
        if (netSize > sizeTolerance) {
            finalSide = PositionSide.LONG;
            finalSize = netSize; // Net size is positive for long
        } else if (netSize < -sizeTolerance) {
            finalSide = PositionSide.SHORT;
            finalSize = Math.abs(netSize); // Report absolute size for short
        }
        // If netSize is within tolerance, side remains NONE, size remains 0.0

        // Calculate average entry price if there's a position
        if (finalSide !== PositionSide.NONE && totalAbsSize > sizeTolerance) {
            finalEntryPrice = totalValue / totalAbsSize;
            // Ensure entry price is positive
            if (finalEntryPrice <= 0) {
                logger.warn(c.yellow(`Calculated average entry price (${finalEntryPrice}) is non-positive. Using 0.0 instead.`));
                finalEntryPrice = 0.0;
            }
        }

        // --- Construct result and cache it ---
        const aggregatedPosition = {
            side: finalSide,
            size: finalSize,
            entryPrice: finalEntryPrice,
        };

        logger.debug(`Aggregated position for ${config.symbol}: Side=${finalSide}, Size=${finalSize.toFixed(market?.precision?.amount || 8)}, EntryPrice=${finalEntryPrice.toFixed(market?.precision?.price || 8)}`);
        exchangeManager._setCache(cacheName, cacheKey, aggregatedPosition); // Cache the result
        return aggregatedPosition;

    } catch (e) {
        logger.error(c.red(`Failed to fetch or process position for ${config.symbol} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
        // Cache the default state on error to prevent repeated failed lookups within TTL
        exchangeManager._setCache(cacheName, cacheKey, defaultReturn);
        return defaultReturn; // Return default state on error
    }
}

/**
 * Fetches account balance for the configured quote currency using the ExchangeManager's cache and retry logic.
 * Handles different structures returned by exchanges (especially Bybit V5).
 * Prioritizes available balance suitable for placing new trades.
 * @param {ExchangeManager} exchangeManager - Instance of ExchangeManager.
 * @returns {Promise<number | null>} Account balance (available/wallet balance) in quote currency, or null on failure.
 */
async function getAccountBalance(exchangeManager) {
    const { config, exchange } = exchangeManager;
    const currency = config.currency; // e.g., USDT
    const cacheName = "balance";
    const cacheKey = `${currency}_${config.bybit_account_type}`; // Cache by currency and account type

    // Try cache first
    const cachedBalance = exchangeManager._getCache(cacheName, cacheKey);
    if (cachedBalance !== null && typeof cachedBalance === 'number' && Number.isFinite(cachedBalance)) {
        return cachedBalance; // Return valid cached number
    }

    // Handle Dry Run
    if (config.dry_run) {
        // Retrieve simulated balance from state if available, otherwise use default
        const simulatedBalance = exchangeManager._caches.balance.data ?? 10000.0;
        logger.debug(c.magenta(`DRY RUN: Using simulated account balance of ${simulatedBalance}.`));
        // Ensure the simulated balance is cached
        exchangeManager._setCache(cacheName, cacheKey, simulatedBalance);
        return simulatedBalance;
    }

    logger.debug(`Fetching account balance for ${currency} (Account Type: ${config.bybit_account_type})...`);

    // Define the async function for the API call
    const fetchFunc = async () => {
        if (!exchange || !exchange.has['fetchBalance']) {
            throw new ccxt.NotSupported("Exchange object not initialized or does not support fetchBalance method.");
        }
        // Prepare parameters, especially for Bybit V5
        const params = {};
        if (exchange.id === 'bybit') {
            // Bybit V5 requires accountType for specific balance views
            params.accountType = config.bybit_account_type; // Use configured type
            logger.debug(`Using accountType='${params.accountType}' for fetchBalance.`);
            // Optionally specify the coin for V5, though fetchBalance usually returns all
            // params.coin = currency;
        }

        // Fetch the balance data
        const balanceData = await exchange.fetchBalance(params);

        // --- Parse the balance data ---
        // We want the balance available for opening *new* trades (available margin/wallet balance).
        // CCXT standard structure: balanceData[CURRENCY].free
        // Bybit V5: Needs careful parsing based on accountType.

        let balanceValueStr = null;

        // 1. Try standard CCXT structure first (prioritize 'free')
        if (balanceData[currency]) {
            balanceValueStr = balanceData[currency].free ?? balanceData[currency].total; // Fallback to total if free is missing
            if (balanceValueStr !== undefined && balanceValueStr !== null) {
                 logger.debug(`Found balance via standard CCXT structure: ${balanceValueStr} ${currency} (${balanceData[currency].free === undefined ? 'using total' : 'using free'})`);
            }
        }

        // 2. If standard structure failed or yielded null/undefined, try parsing Bybit V5 'info' field
        if (balanceValueStr === null || balanceValueStr === undefined) {
            logger.debug("Standard CCXT balance structure not found or missing value. Attempting Bybit V5 structure parsing...");
            if (exchange.id === 'bybit' && balanceData.info?.result?.list?.length > 0) {
                const accountInfo = balanceData.info.result.list[0]; // Usually only one item for specific accountType

                // Check if accountType matches the response
                if (accountInfo?.accountType !== config.bybit_account_type) {
                    logger.warn(c.yellow(`Bybit fetchBalance response accountType ('${accountInfo?.accountType}') does not match requested type ('${config.bybit_account_type}'). Parsing may be incorrect.`));
                }

                // --- V5 CONTRACT Account ---
                if (accountInfo?.accountType === 'CONTRACT' && Array.isArray(accountInfo.coin)) {
                    const coinInfo = accountInfo.coin.find(c => c.coin === currency);
                    if (coinInfo) {
                        // Prioritize availableBalance (available margin), then walletBalance for CONTRACT
                        balanceValueStr = coinInfo.availableToBorrow ?? coinInfo.availableBalance ?? coinInfo.walletBalance;
                        let source = coinInfo.availableToBorrow ? 'availableToBorrow' : (coinInfo.availableBalance ? 'availableBalance' : 'walletBalance');
                        logger.debug(`Found balance via Bybit V5 CONTRACT structure: ${balanceValueStr} ${currency} (Source: ${source})`);
                    }
                }
                // --- V5 UNIFIED Account ---
                else if (accountInfo?.accountType === 'UNIFIED') {
                    // Use totalAvailableBalance (available margin in USDT equivalent)
                    balanceValueStr = accountInfo.totalAvailableBalance;
                    if (balanceValueStr !== undefined && balanceValueStr !== null) {
                         logger.debug(`Found balance via Bybit V5 UNIFIED structure: ${balanceValueStr} ${currency} (Source: totalAvailableBalance - Assuming USDT equivalent)`);
                    }
                }
                 // --- V5 FUND Account (less relevant for trading) ---
                 else if (accountInfo?.accountType === 'FUND') {
                      logger.warn(c.yellow(`Fetched balance for FUND account type. This is usually not used for derivatives trading.`));
                      const coinInfo = accountInfo.coin?.find(c => c.coin === currency);
                      if (coinInfo) {
                          balanceValueStr = coinInfo.free ?? coinInfo.total ?? coinInfo.availableBalance; // Check multiple fields
                          logger.debug(`Found balance via Bybit V5 FUND structure: ${balanceValueStr} ${currency}`);
                      }
                 }
            }
        }

        // --- Final Check and Parsing ---
        if (balanceValueStr === null || balanceValueStr === undefined) {
            throw new Error(`Could not find balance for currency '${currency}' and account type '${config.bybit_account_type}'. Check exchange response structure or if currency exists in the account. Raw response (limited depth): ${inspect(balanceData, {depth: 2})}`);
        }

        const balanceNum = parseFloat(balanceValueStr);
         if (isNaN(balanceNum) || !Number.isFinite(balanceNum)) {
             throw new Error(`Parsed balance value '${balanceValueStr}' for currency '${currency}' is not a valid finite number.`);
         }
         if (balanceNum < 0) {
              logger.warn(c.yellow(`Fetched balance for ${currency} is negative (${balanceNum}). This might indicate debt or issues. Risk calculation may fail.`));
         }

        return balanceNum; // Return the successfully parsed balance
    };

    try {
        // Execute fetch function with retry logic
        const balance = await retryOnException(fetchFunc, config.max_retries, config.retry_delay, undefined, 'fetchBalance');

        // Cache the successful result
        exchangeManager._setCache(cacheName, cacheKey, balance);
        logger.debug(`Fetched balance successfully: ${balance.toFixed(4)} ${currency}`);
        return balance;

    } catch (e) {
        logger.error(c.red(`Failed to fetch balance for ${currency} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
        // Cache null on error to prevent hammering API on persistent issues
        exchangeManager._setCache(cacheName, cacheKey, null);
        return null; // Return null on failure
    }
}

/**
 * Fetches ticker data (last price, bid, ask etc.) using the ExchangeManager's cache and retry logic.
 * Validates the structure, ensuring at least a 'last' price is present.
 * @param {ExchangeManager} exchangeManager - Instance of ExchangeManager.
 * @returns {Promise<ccxt.Ticker | null>} Ticker object or null on failure.
 */
async function getTickerData(exchangeManager) {
    const { config, exchange } = exchangeManager;
    const cacheName = "ticker";
    const cacheKey = config.symbol; // Cache by symbol

    // Try cache first
    const cachedTicker = exchangeManager._getCache(cacheName, cacheKey);
    // Add basic validation for cached ticker
    if (cachedTicker && typeof cachedTicker === 'object' && typeof cachedTicker.last === 'number' && Number.isFinite(cachedTicker.last)) {
        return cachedTicker;
    } else if (cachedTicker) {
         logger.warn(c.yellow("Invalid ticker data found in cache. Refetching."));
         exchangeManager._setCache(cacheName, cacheKey, null); // Clear invalid cache
    }

    logger.debug(`Fetching ticker for ${config.symbol}...`);

    // Define the async function for the API call
    const fetchFunc = async () => {
        if (!exchange || !exchange.has['fetchTicker']) {
            throw new ccxt.NotSupported("Exchange object not initialized or does not support fetchTicker method.");
        }
        // Prepare parameters (e.g., Bybit V5 category)
        const params = {};
        if (exchange.id === 'bybit' && ['swap', 'future'].includes(config.exchange_type)) {
            params.category = 'linear'; // Required for Bybit V5
        }
        return await exchange.fetchTicker(config.symbol, params);
    };

    try {
        // Execute with retry logic (shorter delay for ticker is reasonable)
        const ticker = await retryOnException(fetchFunc, config.max_retries, config.retry_delay / 2, undefined, 'fetchTicker');

        // --- Validate the returned data ---
        if (!ticker || typeof ticker !== 'object' || typeof ticker.last !== 'number' || !Number.isFinite(ticker.last) || ticker.last <= 0) {
            logger.warn(c.yellow("fetchTicker returned invalid, incomplete data, or non-finite/non-positive 'last' price."), ticker);
            return null; // Critical if 'last' price is missing or invalid
        }
        // Optional: Check for bid/ask if needed for specific logic
        if (typeof ticker.bid !== 'number' || typeof ticker.ask !== 'number' || ticker.bid <= 0 || ticker.ask <= 0 || ticker.bid >= ticker.ask) {
             logger.warn(c.yellow("fetchTicker returned invalid or potentially crossed bid/ask prices."), ticker);
        }

        // --- Cache and Return ---
        exchangeManager._setCache(cacheName, cacheKey, ticker);
        logger.debug(`Fetched ticker successfully: Last=${ticker.last}, Bid=${ticker.bid ?? 'N/A'}, Ask=${ticker.ask ?? 'N/A'}`);
        return ticker;

    } catch (e) {
        logger.error(c.red(`Failed to fetch ticker for ${config.symbol} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
        // Cache null on error
        exchangeManager._setCache(cacheName, cacheKey, null);
        return null;
    }
}


// --- Trading Bot Class ---
/**
 * The main class orchestrating the trading strategy, state management, and exchange interactions.
 */
class TradingBot {
    /**
     * Initializes the TradingBot instance.
     */
    constructor() {
        // --- Configuration & Services (Initialized in initialize()) ---
        /** @type {Config | null} */
        this.config = null;
        /** @type {ExchangeManager | null} */
        this.exchangeManager = null;
        /** @type {NotificationService | null} */
        this.notifier = null;
        /** @type {ccxt.Market | null} */
        this.marketInfo = null; // Cache market details (precision, limits) fetched once

        // --- Bot State (Managed via load/saveState) ---
        /** @type {{lastOrderSide: PositionSide, lastProcessedCandleTime: number | null}} */
        this.state = { // Persistent state saved to file
            lastOrderSide: PositionSide.NONE, // Tracks the side of the last *closed* position or successful entry
            lastProcessedCandleTime: null,   // Timestamp of the last fully processed candle
        };
        /** @type {PositionState} */
        this.currentPosition = { // Live position state, updated frequently, also saved
             side: PositionSide.NONE,
             size: 0.0,
             entryPrice: 0.0
        };
        /** @type {{stopLoss: {id: string, price: number} | null, takeProfit: {id: string, price: number} | null}} */
        this.activeOrders = { // Tracks IDs and prices of *bot-placed* SL/TP orders, also saved
            stopLoss: null,
            takeProfit: null,
        };
        /** @type {number} */
        this.simulatedBalance = 10000.0; // Internal balance for dry run simulation

        // --- Internal Transient State (Reset on start) ---
        /** @type {Object.<string, Array<Array<number>>>} */
        this.historicalOHLCV = {}; // Stores fetched OHLCV data { [timeframe]: [...] }
        /** @type {{short: SupertrendResult | null, long: SupertrendResult | null}} */
        this.lastSupertrendSignals = { short: null, long: null }; // Last calculated ST values
        /** @type {boolean} */
        this.lastVolumeSpike = false;                       // Last volume spike status
        /** @type {number} */
        this.lastOrderBookPressure = 0.5;                   // Last OB pressure value (neutral default)
        /** @type {boolean} */
        this._isRunning = false;                            // Controls the main loop execution
        /** @type {boolean} */
        this._stop_requested = false;                       // Flag for graceful shutdown signal
        /** @type {NodeJS.Timeout | null} */
        this._mainLoopTimeoutId = null;                     // Holds the timeout ID for the main loop
    }

    /**
     * Initializes the bot's core components: Config, Notifier, ExchangeManager.
     * Fetches initial market information.
     * @returns {Promise<void>}
     * @throws {Error} If any critical initialization step fails.
     */
    async initialize() {
        logger.info(c.blue("--- Initializing Trading Bot ---"));
        // Order is important: Config -> Notifier -> ExchangeManager (uses Config)
        this.config = new Config(); // Loads and validates .env
        // Logging setup happens *after* config validation within Config constructor

        this.notifier = new NotificationService(); // Sets up notification service
        this.exchangeManager = new ExchangeManager(this.config); // Creates manager, passes config
        await this.exchangeManager.initialize(); // Initializes CCXT, loads markets, sets leverage

        // Fetch and cache market info (precision, limits) after exchange is ready
        this.marketInfo = this.exchangeManager.exchange.market(this.config.symbol);
        if (!this.marketInfo || this.marketInfo.precision?.amount === undefined || this.marketInfo.precision?.price === undefined || this.marketInfo.limits?.amount?.min === undefined) {
            // This should ideally be caught during market validation, but double-check
            throw new Error(`Failed to load essential market info (precision/limits) for ${this.config.symbol} after initialization.`);
        }
        // Add tickSize if available (useful for price comparisons/buffers)
        this.marketInfo.tickSize = parseFloat(this.marketInfo.info?.tickSize) || Math.pow(10, -(this.marketInfo.precision.price || 8)); // Use tickSize from info or derive from precision
        logger.info(`Market info cached for ${c.bold(this.config.symbol)}: Price Precision=${this.marketInfo.precision?.price}, Amount Precision=${this.marketInfo.precision?.amount}, Min Amount=${this.marketInfo.limits?.amount?.min}, Tick Size=${this.marketInfo.tickSize}`);

        // Log validation success now that logger is fully ready
        logger.debug(c.green("Configuration validation successful."));
        logger.info(c.green("--- Trading Bot Initialized Successfully ---"));
    }


    // --- State Management ---

    /**
     * Loads the bot's state from the JSON file specified in the config.
     * Performs validation on the loaded state structure and values.
     * Verifies the loaded state against the exchange after loading.
     * @returns {Promise<void>}
     */
    async loadState() {
        const stateFilePath = this.config.state_file;
        try {
            logger.info(`Attempting to load state from ${stateFilePath}...`);
            const stateData = await fs.readFile(stateFilePath, 'utf8');
            const loadedState = JSON.parse(stateData);

            // --- Validate loaded state ---
            if (typeof loadedState !== 'object' || loadedState === null) {
                throw new Error("Loaded state is not a valid object.");
            }

            // Restore state components if they exist and have the correct basic type/value
            // Persistent state fields
            if (loadedState.state && typeof loadedState.state === 'object') {
                this.state.lastOrderSide = Object.values(PositionSide).includes(loadedState.state.lastOrderSide)
                    ? loadedState.state.lastOrderSide
                    : PositionSide.NONE;
                this.state.lastProcessedCandleTime = typeof loadedState.state.lastProcessedCandleTime === 'number' && Number.isFinite(loadedState.state.lastProcessedCandleTime)
                    ? loadedState.state.lastProcessedCandleTime
                    : null;
            } else {
                 this.state = { lastOrderSide: PositionSide.NONE, lastProcessedCandleTime: null }; // Default if missing/invalid
                 logger.warn(c.yellow("Loaded state missing 'state' object or invalid. Resetting persistent state fields."));
            }

            // Current position state
            if (loadedState.currentPosition && typeof loadedState.currentPosition === 'object') {
                 this.currentPosition.side = Object.values(PositionSide).includes(loadedState.currentPosition.side)
                     ? loadedState.currentPosition.side
                     : PositionSide.NONE;
                 this.currentPosition.size = typeof loadedState.currentPosition.size === 'number' && Number.isFinite(loadedState.currentPosition.size)
                     ? loadedState.currentPosition.size
                     : 0.0;
                 this.currentPosition.entryPrice = typeof loadedState.currentPosition.entryPrice === 'number' && Number.isFinite(loadedState.currentPosition.entryPrice)
                     ? loadedState.currentPosition.entryPrice
                     : 0.0;
                 // Ensure size/entry are non-negative
                 if (this.currentPosition.size < 0) this.currentPosition.size = 0;
                 if (this.currentPosition.entryPrice < 0) this.currentPosition.entryPrice = 0;
                 // If side is NONE, ensure size/entry are 0
                 if (this.currentPosition.side === PositionSide.NONE) {
                     this.currentPosition.size = 0.0;
                     this.currentPosition.entryPrice = 0.0;
                 }
            } else {
                 this.currentPosition = { side: PositionSide.NONE, size: 0.0, entryPrice: 0.0 }; // Default if missing/invalid
                 logger.warn(c.yellow("Loaded state missing 'currentPosition' object or invalid. Resetting current position."));
            }

            // Active orders state
            if (loadedState.activeOrders && typeof loadedState.activeOrders === 'object') {
                 const isValidOrderState = (order) => order && typeof order.id === 'string' && order.id.length > 0 && typeof order.price === 'number' && Number.isFinite(order.price) && order.price > 0;
                 this.activeOrders.stopLoss = isValidOrderState(loadedState.activeOrders.stopLoss) ? loadedState.activeOrders.stopLoss : null;
                 this.activeOrders.takeProfit = isValidOrderState(loadedState.activeOrders.takeProfit) ? loadedState.activeOrders.takeProfit : null;
            } else {
                 this.activeOrders = { stopLoss: null, takeProfit: null }; // Default if missing/invalid
                 logger.warn(c.yellow("Loaded state missing 'activeOrders' object or invalid. Resetting active orders."));
            }

            // Load simulated balance if present and in dry run mode
            if (this.config.dry_run && typeof loadedState.simulatedBalance === 'number' && Number.isFinite(loadedState.simulatedBalance)) {
                 this.simulatedBalance = loadedState.simulatedBalance;
                 // Cache the loaded simulated balance
                 this.exchangeManager._setCache('balance', `${this.config.currency}_${this.config.bybit_account_type}`, this.simulatedBalance);
                 logger.info(c.magenta(`DRY RUN: Loaded simulated balance: ${this.simulatedBalance}`));
            } else if (this.config.dry_run) {
                 logger.info(c.magenta(`DRY RUN: Using default simulated balance: ${this.simulatedBalance}`));
                 this.exchangeManager._setCache('balance', `${this.config.currency}_${this.config.bybit_account_type}`, this.simulatedBalance);
            }


            logger.info(c.green("Bot state loaded successfully."));
            logger.debug("Loaded State Details:", { state: this.state, currentPosition: this.currentPosition, activeOrders: this.activeOrders, simulatedBalance: this.config.dry_run ? this.simulatedBalance : 'N/A' });

            // IMPORTANT: Verify loaded state against actual exchange state, especially after restarts.
            await this.verifyStateWithExchange();

        } catch (error) {
            if (error.code === 'ENOENT') {
                logger.warn(c.yellow(`State file (${stateFilePath}) not found. Starting with default initial state.`));
                this.resetState(false); // Initialize with defaults if file doesn't exist
            } else if (error instanceof SyntaxError) {
                 logger.error(c.red(`Error parsing state file (${stateFilePath}): ${error.message}. File might be corrupted. Starting with default state.`));
                 this.resetState();
            }
             else {
                logger.error(c.red(`Error loading state file (${stateFilePath}): ${error.message}. Starting with default state.`), error.stack);
                this.resetState(); // Initialize with defaults on other errors
            }
            // Ensure default simulated balance is set if state load fails in dry run
            if (this.config.dry_run) {
                 this.simulatedBalance = 10000.0;
                 this.exchangeManager._setCache('balance', `${this.config.currency}_${this.config.bybit_account_type}`, this.simulatedBalance);
            }
        }
    }

    /**
     * Saves the bot's current state (persistent parts and simulated balance) to the JSON file atomically.
     * @returns {Promise<void>}
     */
    async saveState() {
        const stateFilePath = this.config.state_file;
        const tempFilePath = `${stateFilePath}.tmp_${Date.now()}`;
        try {
            // Consolidate the state to be saved
            const stateToSave = {
                state: this.state,
                currentPosition: this.currentPosition,
                activeOrders: this.activeOrders,
                simulatedBalance: this.simulatedBalance, // Always save simulated balance
                timestamp: new Date().toISOString() // Add timestamp for reference
            };

            const stateString = JSON.stringify(stateToSave, null, 2); // Pretty print JSON

            // Atomic write: Write to temp file first
            await fs.writeFile(tempFilePath, stateString, 'utf8');
            // Rename temp file to actual state file (atomic on most systems)
            await fs.rename(tempFilePath, stateFilePath);

            logger.debug(c.gray(`Bot state saved successfully to ${stateFilePath}.`));
        } catch (error) {
            logger.error(c.red(`Error saving state file (${stateFilePath}): ${error.message}`), error.stack);
            // Attempt to clean up temp file if it exists
            try { await fs.unlink(tempFilePath); } catch (cleanupError) { /* Ignore cleanup error */ }
            // Send an alert if state saving fails, as it's critical for recovery
            this.notifier.sendSms(`CRITICAL ERROR: Failed to save state file ${stateFilePath}! Bot may not recover correctly after restart. Reason: ${error.message.substring(0,100)}`, this.config);
        }
    }

    /**
     * Resets the bot's state variables to their default initial values.
     * @param {boolean} [logMessage=true] - Whether to log the reset action.
     */
    resetState(logMessage = true) {
        if (logMessage) logger.info("Resetting bot state to default initial values.");
        this.state = {
            lastOrderSide: PositionSide.NONE,
            lastProcessedCandleTime: null,
        };
        this.currentPosition = { side: PositionSide.NONE, size: 0.0, entryPrice: 0.0 };
        this.activeOrders = { stopLoss: null, takeProfit: null };
        this.simulatedBalance = 10000.0; // Reset simulated balance too
        // Reset transient indicator states as well
        this.lastSupertrendSignals = { short: null, long: null };
        this.lastVolumeSpike = false;
        this.lastOrderBookPressure = 0.5;
        this.historicalOHLCV = {}; // Clear fetched data
    }

    /**
     * Verifies the loaded/internal state (position, active orders) against the actual state on the exchange.
     * Corrects internal state if discrepancies are found.
     * Cancels any unexpected open conditional orders for the symbol found on the exchange.
     * @returns {Promise<void>}
     */
    async verifyStateWithExchange() {
         logger.info("Verifying internal state against exchange...");

         if (this.config.dry_run) {
             logger.info(c.magenta("DRY RUN: Skipping exchange state verification."));
             return;
         }

         let stateWasCorrected = false; // Track if changes were made

         try {
             // --- 1. Verify Position ---
             this.exchangeManager._setCache('position', this.config.symbol, null); // Clear cache
             const actualPosition = await this.exchangeManager.getPosition(); // Fetch fresh
             if (!actualPosition) {
                  logger.error(c.red("Failed to fetch actual position during state verification. Cannot verify position state."));
             } else {
                  const statePos = this.currentPosition;
                  const sizeTolerance = Math.max(1e-9, this.marketInfo?.limits?.amount?.min / 10 || 1e-9);
                  const sizeDiff = Math.abs(statePos.size - actualPosition.size);

                  if (statePos.side !== actualPosition.side || sizeDiff > sizeTolerance) {
                       logger.warn(c.yellow(`Position State Mismatch: Internal=${statePos.side}/${statePos.size.toFixed(8)}, Exchange=${actualPosition.side}/${actualPosition.size.toFixed(8)}. Updating internal state.`));
                       this.currentPosition = actualPosition; // Correct internal state
                       if (actualPosition.side !== PositionSide.NONE && statePos.side === PositionSide.NONE) {
                           logger.warn(c.yellow("Position exists on exchange but not in loaded state. Clearing potentially invalid active orders from state."));
                           this.activeOrders = { stopLoss: null, takeProfit: null };
                       } else if (actualPosition.side === PositionSide.NONE && statePos.side !== PositionSide.NONE) {
                           logger.warn(c.yellow("Position found in loaded state but not on exchange. Clearing internal position and active orders."));
                           this.activeOrders = { stopLoss: null, takeProfit: null };
                       }
                       stateWasCorrected = true;
                  } else {
                       logger.info(c.green("Position state matches exchange."));
                  }
             }

             // --- 2. Verify Active Orders (SL/TP) ---
             logger.debug("Fetching open conditional orders for verification...");
             const params = { 'category': 'linear' }; // Bybit V5 param
             let openOrders = [];
             try {
                 // Fetch orders with status 'open' or 'untriggered' (or similar relevant statuses)
                 // CCXT's fetchOpenOrders should handle this, but Bybit might have nuances.
                 // We might need fetchOrders with specific status filters if fetchOpenOrders is insufficient.
                 openOrders = await retryOnException(
                     async () => await this.exchangeManager.exchange.fetchOpenOrders(this.config.symbol, undefined, undefined, params),
                     this.config.max_retries, this.config.retry_delay, undefined, 'fetchOpenOrders (Verification)'
                 );
             } catch (e) {
                 logger.error(c.red(`Failed to fetch open orders during verification: ${e.message}. Skipping order verification.`));
                 // Save state if position was corrected before this error
                 if (stateWasCorrected) await this.saveState();
                 return; // Cannot verify orders if fetch fails
             }

             // Filter for potential SL/TP orders: Conditional (stop/takeProfit), reduceOnly, correct symbol.
             const relevantOpenOrders = openOrders.filter(o =>
                 (o.type?.toLowerCase().includes('stop') || o.type?.toLowerCase().includes('takeprofit')) && // Check type
                 (o.params?.reduceOnly === true || o.reduceOnly === true) && // Check reduceOnly flag
                 (o.status === 'open' || o.status === 'untriggered') // Check status
             );
             logger.debug(`Found ${relevantOpenOrders.length} potentially relevant open conditional orders on exchange.`);

             const activeStateOrders = [this.activeOrders.stopLoss, this.activeOrders.takeProfit].filter(o => o !== null);
             const activeStateOrderIds = new Set(activeStateOrders.map(o => o.id));
             const openExchangeOrderIds = new Set(relevantOpenOrders.map(o => o.id));

             // Check 1: Orders in state but NOT on exchange -> Remove from state
             for (const stateOrder of activeStateOrders) {
                  if (!openExchangeOrderIds.has(stateOrder.id)) {
                      logger.warn(c.yellow(`Order Mismatch: Order ID ${stateOrder.id} (Price: ${stateOrder.price}) found in state but not active/open on exchange. Removing from internal state.`));
                      if (this.activeOrders.stopLoss?.id === stateOrder.id) this.activeOrders.stopLoss = null;
                      if (this.activeOrders.takeProfit?.id === stateOrder.id) this.activeOrders.takeProfit = null;
                      stateWasCorrected = true;
                  }
             }

             // Check 2: Orders on exchange but NOT in state -> Cancel them (unexpected/orphaned orders)
             for (const exchangeOrder of relevantOpenOrders) {
                  if (!activeStateOrderIds.has(exchangeOrder.id)) {
                       logger.warn(c.yellow(`Order Mismatch: Unexpected open conditional order found on exchange (ID: ${exchangeOrder.id}, Type: ${exchangeOrder.type}, Price: ${exchangeOrder.triggerPrice || exchangeOrder.price}, Side: ${exchangeOrder.side}). Attempting cancellation.`));
                       try {
                           await retryOnException(
                               async () => await this.exchangeManager.exchange.cancelOrder(exchangeOrder.id, this.config.symbol, params),
                               this.config.max_retries, this.config.retry_delay / 2, undefined, `cancelOrder (Orphaned ${exchangeOrder.id})`
                           );
                           logger.info(`Successfully cancelled unexpected order ${exchangeOrder.id}.`);
                       } catch (cancelError) {
                            if (cancelError instanceof ccxt.OrderNotFound || (cancelError instanceof ccxt.ExchangeError && (cancelError.message.includes("Order does not exist") || cancelError.message.includes("already closed") || cancelError.message.includes("has been filled") || cancelError.message.includes("canceled") || cancelError.message.includes("ret_code=30034") || cancelError.message.includes("ret_code=10001")))) {
                                logger.info(`Unexpected order ${exchangeOrder.id} was already gone.`);
                            } else {
                                logger.error(c.red(`Failed to cancel unexpected order ${exchangeOrder.id}: ${cancelError.message}`));
                            }
                       }
                       // Don't mark state corrected here, as the order wasn't part of our expected state.
                       // The goal is just cleanup.
                  }
             }

             if (!stateWasCorrected) {
                 logger.info(c.green("Active order state matches exchange."));
             } else {
                 logger.info("State verification resulted in corrections.");
             }

             // Save state if any corrections were made to our internal representation
             if (stateWasCorrected) {
                 await this.saveState();
             }

         } catch (error) {
              logger.error(c.red(`Error during state verification with exchange: ${error.message}`), error.stack);
              // If verification fails, proceed cautiously, state might be inaccurate.
         }
    }


    // --- Indicator Calculation and Signal Generation ---

    /**
     * Fetches necessary market data (OHLCV, Order Book) and calculates all required indicators.
     * Updates the bot's internal state with the latest indicator values.
     * Prevents reprocessing of the same candle data.
     * @returns {Promise<boolean>} True if new indicators were calculated successfully, false otherwise.
     */
    async calculateIndicators() {
        logger.debug("Calculating indicators...");

        // --- Fetch Required Data ---
        const requiredCandles = Math.max(
            this.config.long_st_period, this.config.short_st_period,
            this.config.gaussian_filter_length, this.config.volume_long_period
        ) + 2; // Buffer for lookbacks and SMA/EMA seeding
        const ohlcvLimit = requiredCandles + 100; // Fetch extra for stability and indicator calculation buffer
        const ohlcvData = await this.exchangeManager.fetchOhlcv(ohlcvLimit);
        const orderBook = await this.exchangeManager.fetchOrderBook(); // Fetch latest order book

        // --- Validate Fetched Data ---
        if (!ohlcvData || ohlcvData.length < requiredCandles) {
            logger.warn(`Insufficient OHLCV data fetched (${ohlcvData?.length || 0} candles). Need at least ${requiredCandles}. Skipping indicator calculation.`);
            return false;
        }
        if (!orderBook) {
             logger.warn(`Failed to fetch order book data. Order book pressure filter will use neutral value (0.5).`);
             this.lastOrderBookPressure = 0.5; // Reset to neutral
        }

        // --- Check for New Candle Data ---
        const lastCandleTimestamp = ohlcvData[ohlcvData.length - 1][OHLCV_INDEX.TIMESTAMP];
        if (lastCandleTimestamp === this.state.lastProcessedCandleTime) {
            logger.debug(c.gray(`Skipping indicator calculation: Candle data timestamp (${new Date(lastCandleTimestamp).toISOString()}) hasn't changed.`));
            return false; // No new data
        }

        // Store latest OHLCV data internally
        this.historicalOHLCV[this.config.timeframe] = ohlcvData;

        // --- Extract Data Arrays for Indicators ---
        let validCandleCount = 0;
        const highPrices = [], lowPrices = [], closePrices = [], volumeData = [];
        for (const c of ohlcvData) {
             if (c && c.length >= OHLCV_SCHEMA.length && c.slice(0, OHLCV_SCHEMA.length).every(v => typeof v === 'number' && Number.isFinite(v))) {
                  highPrices.push(c[OHLCV_INDEX.HIGH]);
                  lowPrices.push(c[OHLCV_INDEX.LOW]);
                  closePrices.push(c[OHLCV_INDEX.CLOSE]);
                  volumeData.push(c[OHLCV_INDEX.VOLUME]);
                  validCandleCount++;
             } else {
                  logger.warn(c.yellow(`Filtering out null, incomplete, or non-finite candle: ${inspect(c)}`));
                  // Pad with NaNs to keep array lengths consistent for indicators if needed,
                  // but robust indicator functions should handle internal NaNs.
                  // Let's assume the indicator functions handle gaps or filter internally.
             }
        }
        if (validCandleCount < requiredCandles) {
             logger.warn(`Insufficient valid OHLCV data points after filtering (${validCandleCount} points). Need ${requiredCandles}. Skipping calculation.`);
             return false;
        }

        // --- Calculate Indicators using Standalone Functions ---
        let calculationSuccess = true;
        const supertrendShortResult = ehlersSupertrend(highPrices, lowPrices, closePrices, this.config.short_st_period, this.config.st_multiplier, this.config.gaussian_filter_length, logger);
        const supertrendLongResult = ehlersSupertrend(highPrices, lowPrices, closePrices, this.config.long_st_period, this.config.st_multiplier, this.config.gaussian_filter_length, logger);
        const vmaResults = calculateVMA(volumeData, this.config.volume_short_period, this.config.volume_long_period, logger);
        const volumeSpike = (vmaResults.vmaShort.length > 0 && vmaResults.vmaLong.length > 0)
                          ? this.detectVolumeSpike(volumeData, vmaResults.vmaShort, vmaResults.vmaLong)
                          : false;
        const obPressure = orderBook ? calculateOrderBookPressure(orderBook, logger) : this.lastOrderBookPressure;

        // --- Check Indicator Results & Update Bot State ---
        // Helper to get the last valid *object* or primitive from an indicator result array
        const getLastValidResult = (resultArray) => {
             if (!Array.isArray(resultArray) || resultArray.length === 0) return null;
             for (let i = resultArray.length - 1; i >= 0; i--) {
                  const item = resultArray[i];
                  // Check if item is a valid result (not null/undefined)
                  // For objects (like SupertrendResult), ensure required properties are valid
                  if (item !== null && item !== undefined) {
                      if (typeof item === 'object') {
                          // Example validation for SupertrendResult
                          if (item.hasOwnProperty('trend') && item.hasOwnProperty('value') && Number.isFinite(item.trend)) {
                              return item;
                          }
                          // Add checks for other object types if needed
                      } else if (typeof item === 'number' || typeof item === 'boolean') {
                          // For primitive results like VMA or volumeSpike
                          if (Number.isFinite(item) || typeof item === 'boolean') {
                             return item; // Return the primitive value itself
                          }
                      }
                  }
             }
             return null; // No valid result found
        }

        this.lastSupertrendSignals.short = getLastValidResult(supertrendShortResult);
        this.lastSupertrendSignals.long = getLastValidResult(supertrendLongResult);
        this.lastVolumeSpike = volumeSpike; // Already a boolean
        this.lastOrderBookPressure = obPressure; // Already a number

        // Check if critical indicators failed (result is null or trend is 0/NaN)
        if (!this.lastSupertrendSignals.short || this.lastSupertrendSignals.short.trend === 0) { logger.error(c.red("Short Ehlers Supertrend calculation failed or is neutral.")); calculationSuccess = false; }
        if (!this.lastSupertrendSignals.long || this.lastSupertrendSignals.long.trend === 0) { logger.error(c.red("Long Ehlers Supertrend calculation failed or is neutral.")); calculationSuccess = false; }
        if (vmaResults.vmaShort.length === 0 || vmaResults.vmaLong.length === 0) { logger.warn(c.yellow("VMA calculation failed. Volume filter might be inaccurate.")); } // Non-critical?

        if (!calculationSuccess) {
             logger.warn(c.yellow("One or more critical indicator calculations failed. Signal generation will be skipped."));
             this.state.lastProcessedCandleTime = lastCandleTimestamp; // Mark candle processed even on failure
             // Don't save state here, just mark as processed for this run
             return false; // Indicate failure
        }

        // Mark this candle timestamp as processed and save state
        this.state.lastProcessedCandleTime = lastCandleTimestamp;
        await this.saveState(); // Save state after successful indicator calculation

        logger.debug("Indicators calculated.", {
            shortST: this.lastSupertrendSignals.short ? `T:${this.lastSupertrendSignals.short.trend} V:${this.lastSupertrendSignals.short.value?.toFixed(this.marketInfo?.precision?.price || 4)}` : 'null',
            longST: this.lastSupertrendSignals.long ? `T:${this.lastSupertrendSignals.long.trend} V:${this.lastSupertrendSignals.long.value?.toFixed(this.marketInfo?.precision?.price || 4)}` : 'null',
            volSpike: this.lastVolumeSpike,
            obPressure: this.lastOrderBookPressure.toFixed(3)
        });
        return true; // Indicate success
    }

    /**
     * Detects if a volume spike occurred on the most recent candle.
     * Conditions: Short VMA > Long VMA * Threshold AND Last Volume > Short VMA.
     * Handles potential NaN values in inputs.
     * @param {number[]} volumeData - Array of raw volume data.
     * @param {number[]} vmaShort - Array of short-term VMA values.
     * @param {number[]} vmaLong - Array of long-term VMA values.
     * @returns {boolean} - True if a volume spike is detected on the last candle, false otherwise.
     */
    detectVolumeSpike(volumeData, vmaShort, vmaLong) {
        const n = volumeData.length;
        if (n === 0 || n > vmaShort.length || n > vmaLong.length || n < 1) {
            logger.warn(c.yellow("Invalid input arrays for volume spike detection."));
            return false;
        }

        const lastVolume = volumeData[n - 1];
        const currentVmaShort = vmaShort[n - 1];
        const currentVmaLong = vmaLong[n - 1];
        const threshold = this.config.volume_spike_threshold;

        if (!Number.isFinite(lastVolume) || !Number.isFinite(currentVmaShort) || !Number.isFinite(currentVmaLong) || currentVmaLong <= 0) {
            logger.debug(c.gray(`Cannot calculate volume spike due to non-finite values or zero/negative long VMA. Vol=${lastVolume}, Short=${currentVmaShort}, Long=${currentVmaLong}`));
            return false;
        }

        const volumeRatio = currentVmaShort / currentVmaLong;
        const isSpike = volumeRatio >= threshold && lastVolume > currentVmaShort;

        logger.debug(`Volume Spike Check: LastVol=${lastVolume.toFixed(2)}, ShortVMA=${currentVmaShort.toFixed(2)}, LongVMA=${currentVmaLong.toFixed(2)}, Ratio=${volumeRatio.toFixed(2)}, Threshold=${threshold}, Spike=${isSpike}`);
        return isSpike;
    }

    /**
     * Generates a trading signal (LONG, SHORT, NONE) based on the latest calculated indicator states.
     * Requires confirmation from both Supertrends and filters (Volume Spike, OB Pressure).
     * @returns {PositionSide} - The trading signal (PositionSide.LONG, PositionSide.SHORT, PositionSide.NONE).
     */
    generateSignal() {
        // --- Check if all required indicators are available and valid ---
        if (!this.lastSupertrendSignals.short || !this.lastSupertrendSignals.long ||
            !Number.isFinite(this.lastSupertrendSignals.short.trend) || !Number.isFinite(this.lastSupertrendSignals.long.trend) ||
            this.lastSupertrendSignals.short.trend === 0 || this.lastSupertrendSignals.long.trend === 0) {
            logger.debug(c.gray("Supertrend signals not available, non-finite, or neutral. Signal=NONE."));
            return PositionSide.NONE;
        }
        if (!Number.isFinite(this.lastOrderBookPressure)) {
            logger.warn(c.yellow("Order book pressure is non-finite. Signal=NONE."));
            return PositionSide.NONE;
        }

        // --- Extract latest indicator values ---
        const shortStTrend = this.lastSupertrendSignals.short.trend;
        const longStTrend = this.lastSupertrendSignals.long.trend;
        const volumeConfirm = this.lastVolumeSpike;
        const obPressure = this.lastOrderBookPressure;
        const obConfirmThreshold = this.config.ob_pressure_threshold;
        const obSellThreshold = 1.0 - obConfirmThreshold;

        let signal = PositionSide.NONE; // Default signal

        // --- Define Signal Conditions ---
        const isLongSignal = shortStTrend === 1 && longStTrend === 1 && volumeConfirm && obPressure >= obConfirmThreshold;
        const isShortSignal = shortStTrend === -1 && longStTrend === -1 && volumeConfirm && obPressure <= obSellThreshold;

        if (isLongSignal) {
            signal = PositionSide.LONG;
        } else if (isShortSignal) {
            signal = PositionSide.SHORT;
        }

        logger.info(`Signal Check: ShortST=${shortStTrend}, LongST=${longStTrend}, VolConfirm=${volumeConfirm}, OBPressure=${obPressure.toFixed(3)} (BuyThr>=${obConfirmThreshold.toFixed(3)}, SellThr<=${obSellThreshold.toFixed(3)}) ==> Signal=${c.bold(signal)}`);
        return signal;
    }

    // --- Risk Management ---

    /**
     * Calculates the position size in base currency based on risk percentage, account balance, and stop loss distance.
     * Adjusts the size according to the market's precision and limits.
     * Updates simulated balance in dry run mode.
     * @param {number} entryPrice - The intended or estimated entry price.
     * @param {number} stopLossPrice - The calculated stop loss price.
     * @param {number | null} [atrValue=null] - The current ATR value (optional, for logging/reference).
     * @returns {Promise<number | null>} The calculated position size in base currency, or null if calculation fails or size is invalid/too small.
     */
    async calculatePositionSize(entryPrice, stopLossPrice, atrValue = null) {
        logger.debug(`Calculating position size: Entry=${entryPrice}, SL=${stopLossPrice}, ATR=${atrValue?.toFixed(this.marketInfo?.precision?.price || 4) ?? 'N/A'}`);

        // --- Get Required Data ---
        let balance;
        if(this.config.dry_run) {
            balance = this.simulatedBalance; // Use internal simulated balance
            logger.debug(c.magenta(`DRY RUN: Using simulated balance ${balance} for size calculation.`));
        } else {
             balance = await this.exchangeManager.fetchBalance(); // Fetches (potentially cached) live balance
        }
        const riskFraction = this.config.risk_per_trade;

        // --- Validate Inputs ---
        if (balance === null || isNaN(balance) || balance <= 0) {
            logger.error(c.red(`Cannot calculate position size: Invalid or non-positive balance fetched/simulated (${balance}).`));
            return null;
        }
        if (isNaN(entryPrice) || isNaN(stopLossPrice) || entryPrice <= 0 || stopLossPrice <= 0) {
             logger.error(c.red(`Cannot calculate position size: Invalid entry (${entryPrice}) or stop loss (${stopLossPrice}) price.`));
             return null;
        }
        const priceDiff = Math.abs(entryPrice - stopLossPrice); // Risk per unit of base currency
        if (priceDiff <= 0) {
            logger.error(c.red(`Cannot calculate position size: Entry price (${entryPrice}) and Stop Loss price (${stopLossPrice}) are identical or invalid, resulting in zero risk distance.`));
            return null;
        }
        if (!this.marketInfo || this.marketInfo.precision?.amount === undefined || this.marketInfo.limits?.amount?.min === undefined) {
             logger.error(c.red("Market info (amount precision/min limit) not available. Cannot accurately calculate or validate position size."));
             return null;
        }

        // --- Calculate Size ---
        const riskAmount = balance * riskFraction; // Amount to risk in quote currency (e.g., USDT)
        // For Linear Contracts: Size (in Base Currency) = Risk Amount (Quote) / Price Difference (Quote per Base)
        const positionSizeBase = riskAmount / priceDiff;

        logger.debug(`Risk Amount: ${riskAmount.toFixed(2)} ${this.config.currency}, Price Difference: ${priceDiff.toFixed(this.marketInfo.precision.price || 2)}, Calculated Size (Raw): ${positionSizeBase}`);

        // --- Apply Exchange Precision and Limits ---
        const minAmount = this.marketInfo.limits.amount.min;
        const maxAmount = this.marketInfo.limits.amount.max; // Might be undefined
        let adjustedSize;
        try {
             adjustedSize = parseFloat(this.exchangeManager.exchange.amountToPrecision(this.config.symbol, positionSizeBase));
        } catch (e) {
             logger.error(c.red(`Error applying amount precision: ${e.message}. Raw size: ${positionSizeBase}`));
             return null;
        }

        // Check against minimum order size
        if (minAmount !== undefined && adjustedSize < minAmount) {
            logger.warn(c.yellow(`Calculated position size (${adjustedSize}) is below the minimum required order size (${minAmount}) for ${this.config.symbol}. Cannot place trade.`));
            return null; // Cannot place trade if below minimum
        }

        // Check against maximum order size
        if (maxAmount !== undefined && adjustedSize > maxAmount) {
            logger.warn(c.yellow(`Calculated position size (${adjustedSize}) exceeds the maximum allowed order size (${maxAmount}). Capping size to maximum.`));
            try {
                 adjustedSize = parseFloat(this.exchangeManager.exchange.amountToPrecision(this.config.symbol, maxAmount)); // Adjust max amount to precision too
            } catch (e) {
                 logger.error(c.red(`Error applying amount precision to max amount: ${e.message}. Max size: ${maxAmount}`));
                 return null;
            }
            const actualRiskAmount = adjustedSize * priceDiff;
            const actualRiskFraction = balance > 0 ? actualRiskAmount / balance : 0;
            logger.warn(c.yellow(`Position size capped to maximum. Actual risk: ${actualRiskAmount.toFixed(2)} ${this.config.currency} (${(actualRiskFraction * 100).toFixed(2)}% of balance).`));
        }

        // Final sanity check for the adjusted size
        if (isNaN(adjustedSize) || adjustedSize <= 0) {
             logger.error(c.red(`Final calculated position size (${adjustedSize}) is invalid or zero after adjustments.`));
             return null;
        }

        logger.info(`Calculated Position Size: ${c.bold(adjustedSize)} ${this.marketInfo.base} (Risk: ${riskAmount.toFixed(2)} ${this.config.currency} / ${(riskFraction * 100).toFixed(2)}%)`);
        return adjustedSize;
    }

    /**
     * Calculates Stop Loss and/or Take Profit prices based on the entry price and current ATR.
     * Applies market price precision.
     * @param {PositionSide} side - The side of the potential trade (LONG or SHORT).
     * @param {number} entryPrice - The estimated or actual entry price.
     * @returns {Promise<{stopLoss: number | null, takeProfit: number | null, atrValue: number | null}>} Object containing formatted SL/TP prices and the ATR value used, or nulls if calculation fails.
     */
    async calculateSLTP(side, entryPrice) {
         // --- Get Current ATR ---
         const ohlcvData = this.historicalOHLCV[this.config.timeframe];
         const atrPeriod = this.config.long_st_period; // Use the longer period ATR for SL/TP stability
         if (!ohlcvData || ohlcvData.length < atrPeriod) {
              logger.warn(c.yellow(`Insufficient historical OHLCV data (${ohlcvData?.length || 0} candles, need ${atrPeriod}) to calculate ATR for SL/TP.`));
              return { stopLoss: null, takeProfit: null, atrValue: null };
         }

         const highPrices = ohlcvData.map(c => c[OHLCV_INDEX.HIGH]);
         const lowPrices = ohlcvData.map(c => c[OHLCV_INDEX.LOW]);
         const closePrices = ohlcvData.map(c => c[OHLCV_INDEX.CLOSE]);
         const atrValues = calculateATR(highPrices, lowPrices, closePrices, atrPeriod, logger);
         if (atrValues.length === 0 || atrValues.length !== ohlcvData.length) {
              logger.error(c.red("Failed to calculate ATR or got unexpected result length. Cannot determine SL/TP."));
              return { stopLoss: null, takeProfit: null, atrValue: null };
         }

         // Get the latest valid ATR value
         let currentAtr = null;
         for(let i = atrValues.length - 1; i >= 0; i--) {
             if(Number.isFinite(atrValues[i]) && atrValues[i] > 0) {
                 currentAtr = atrValues[i];
                 break;
             }
         }
         if (currentAtr === null) {
              logger.error(c.red(`Could not find a valid positive ATR value in the recent period. Cannot set SL/TP.`));
              return { stopLoss: null, takeProfit: null, atrValue: null };
         }
         logger.debug(`Using ATR(${atrPeriod}) for SL/TP calculation: ${currentAtr.toFixed(this.marketInfo.precision.price || 4)}`);

         // --- Calculate Raw SL/TP Prices ---
         const slDistance = currentAtr * this.config.sl_atr_mult;
         const tpDistance = currentAtr * this.config.tp_atr_mult;
         let rawStopLossPrice, rawTakeProfitPrice;

         if (side === PositionSide.LONG) {
             rawStopLossPrice = entryPrice - slDistance;
             rawTakeProfitPrice = entryPrice + tpDistance;
         } else if (side === PositionSide.SHORT) {
             rawStopLossPrice = entryPrice + slDistance;
             rawTakeProfitPrice = entryPrice - tpDistance;
         } else {
             logger.error(c.red(`Cannot calculate SL/TP for invalid side '${side}'.`));
             return { stopLoss: null, takeProfit: null, atrValue: currentAtr };
         }

         // --- Apply Price Precision ---
         if (this.marketInfo?.precision?.price === undefined) {
             logger.error(c.red("Market info (price precision) not available. Cannot format SL/TP prices accurately."));
             return { stopLoss: null, takeProfit: null, atrValue: currentAtr };
         }
         let stopLossPrice, takeProfitPrice;
         try {
              stopLossPrice = parseFloat(this.exchangeManager.exchange.priceToPrecision(this.config.symbol, rawStopLossPrice));
              takeProfitPrice = parseFloat(this.exchangeManager.exchange.priceToPrecision(this.config.symbol, rawTakeProfitPrice));
         } catch(e) {
              logger.error(c.red(`Error applying price precision to SL/TP: ${e.message}. Raw SL=${rawStopLossPrice}, Raw TP=${rawTakeProfitPrice}`));
              return { stopLoss: null, takeProfit: null, atrValue: currentAtr };
         }

         // --- Validate Calculated Prices ---
         if (isNaN(stopLossPrice) || isNaN(takeProfitPrice) || stopLossPrice <= 0 || takeProfitPrice <= 0) {
              logger.error(c.red(`Calculated SL (${stopLossPrice}) or TP (${takeProfitPrice}) price is invalid or non-positive after precision adjustment.`));
              return { stopLoss: null, takeProfit: null, atrValue: currentAtr };
         }
         if ((side === PositionSide.LONG && (stopLossPrice >= entryPrice || takeProfitPrice <= entryPrice)) ||
             (side === PositionSide.SHORT && (stopLossPrice <= entryPrice || takeProfitPrice >= entryPrice))) {
             logger.error(c.red(`Calculated SL/TP prices are illogical relative to entry price. Entry=${entryPrice}, SL=${stopLossPrice}, TP=${takeProfitPrice}, Side=${side}. ATR=${currentAtr}. Check ATR multipliers or market conditions.`));
             return { stopLoss: null, takeProfit: null, atrValue: currentAtr };
         }

         logger.info(`Calculated SL=${c.bold(stopLossPrice)}, TP=${c.bold(takeProfitPrice)} (Based on Entry=${entryPrice}, ATR=${currentAtr.toFixed(this.marketInfo?.precision?.price || 4)}, Side=${side})`);
         return { stopLoss: stopLossPrice, takeProfit: takeProfitPrice, atrValue: currentAtr };
    }


    // --- Order Management ---

    /**
     * Places a market order to enter a position. Updates internal state on success (or simulation).
     * Waits briefly and confirms position update from exchange in live mode.
     * Updates simulated balance in dry run.
     * @param {PositionSide} side - The desired position side (LONG or SHORT).
     * @param {number} size - The position size in base currency, adjusted for precision.
     * @returns {Promise<ccxt.Order | object | null>} The created order object from CCXT, a simulated object in dry run, or null on failure.
     */
    async placeEntryOrder(side, size) {
        const orderSide = side === PositionSide.LONG ? Side.BUY : Side.SELL;

        // --- Dry Run Simulation ---
        if (this.config.dry_run) {
            logger.info(c.magenta(`DRY RUN: Simulating place ${c.bold(side)} market order for ${c.bold(size)} ${this.marketInfo.base}...`));
            const ticker = await this.exchangeManager.fetchTicker(); // Use cached ticker if available
            const simulatedEntryPrice = (ticker?.bid && ticker?.ask && ticker.bid > 0 && ticker.ask > 0) ? (orderSide === Side.BUY ? ticker.ask : ticker.bid) : (ticker?.last || 0);
            if (simulatedEntryPrice <= 0) logger.warn(c.yellow("DRY RUN: Could not fetch ticker or bid/ask, using 0 as simulated entry price."));
            const formattedSimEntryPrice = parseFloat(this.exchangeManager.exchange.priceToPrecision(this.config.symbol, simulatedEntryPrice));
            const simulatedCost = size * formattedSimEntryPrice;
            const simulatedFee = simulatedCost * (this.marketInfo?.taker || 0.0006); // Use market taker fee or a default

            // Update internal state as if the order filled
            this.currentPosition = { side, size, entryPrice: formattedSimEntryPrice };
            this.state.lastOrderSide = side; // Track last intended order direction
            this.simulatedBalance -= simulatedFee; // Deduct simulated fee
            logger.info(c.magenta(`DRY RUN: Simulated fee: ${simulatedFee.toFixed(4)} ${this.config.currency}. New simulated balance: ${this.simulatedBalance.toFixed(4)}`));
            await this.saveState(); // Save simulated state (including balance)
            const message = `DRY RUN: Entered ${side} ${size} ${this.config.symbol} @ ~${formattedSimEntryPrice}`;
            logger.info(c.magenta(message));
            this.notifier.sendSms(message, this.config);

            // Return a simulated CCXT order structure
            return { id: `dry_entry_${Date.now()}`, clientOrderId: `dry_entry_coid_${Date.now()}`, timestamp: Date.now(), datetime: new Date().toISOString(), symbol: this.config.symbol, type: 'market', side: orderSide, amount: size, filled: size, remaining: 0, price: formattedSimEntryPrice, average: formattedSimEntryPrice, cost: simulatedCost, status: 'closed', fee: { cost: simulatedFee, currency: this.config.currency }, info: { simulated: true, entryPrice: formattedSimEntryPrice } };
        }

        // --- Live Trading ---
        logger.info(`Attempting to place ${c.bold(side)} (${orderSide}) market order for ${c.bold(size)} ${this.marketInfo.base}...`);
        try {
            const params = {
                'category': 'linear',
                'reduceOnly': false,
                'positionIdx': 0 // 0 for one-way mode
            };

            const placeOrderFunc = async () => await this.exchangeManager.exchange.createMarketOrder(this.config.symbol, orderSide, size, undefined, params);

            const order = await retryOnException( placeOrderFunc, this.config.max_retries, this.config.retry_delay, [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection, ccxt.InsufficientFunds], 'createMarketOrder (Entry)' );

            logger.info(c.green(`Market ${side} (${orderSide}) order placed successfully. Order ID: ${order.id}. Status: ${order.status}`));
            logger.debug("Entry Order Details:", order);

            logger.info("Waiting briefly for order fill and position update...");
            await sleep(5000); // Wait 5 seconds (adjust as needed)

            await this.updateCurrentPositionState(true); // Force fetch latest state

            // Verify if the position state reflects the entry
            const sizeTolerance = Math.max(1e-9, this.marketInfo?.limits?.amount?.min / 10 || 1e-9);
            if (this.currentPosition.side === side && this.currentPosition.size >= size - sizeTolerance) {
                logger.info(c.green(`Position confirmed after entry: Side=${this.currentPosition.side}, Size=${this.currentPosition.size}, Entry=${this.currentPosition.entryPrice}`));
                this.state.lastOrderSide = side; // Track successful order direction
                await this.saveState(); // Save state after confirmed entry
                const message = `LIVE: Entered ${this.currentPosition.side} ${this.currentPosition.size.toFixed(this.marketInfo?.precision?.amount || 8)} ${this.config.symbol} @ ~${this.currentPosition.entryPrice.toFixed(this.marketInfo?.precision?.price || 8)}`;
                this.notifier.sendSms(message, this.config);
                return order; // Return the original order object
            } else {
                logger.error(c.red(`Position state does not reflect successful ${side} entry after order placement and wait. Current state: Side=${this.currentPosition.side}, Size=${this.currentPosition.size}. Order ID: ${order.id}, Status: ${order.status}. Manual check recommended.`));
                try {
                     const updatedOrder = await this.exchangeManager.exchange.fetchOrder(order.id, this.config.symbol, { 'category': 'linear' });
                     logger.error(c.red("Updated Order Status:"), updatedOrder);
                     if (updatedOrder?.info?.orderStatus === 'Rejected') {
                         logger.error(c.red(`Reason for rejection (if available): ${updatedOrder.info?.rejectReason}`));
                     }
                } catch (fetchErr) {
                     logger.error(c.red(`Could not fetch status for order ${order.id}: ${fetchErr.message}`));
                }
                this.notifier.sendSms(`WARNING: ${side} entry order for ${this.config.symbol} placed (ID: ${order.id}) but position state unconfirmed. Check exchange!`, this.config);
                // Reset internal state if entry failed to reflect reality
                if (this.currentPosition.side !== side && this.currentPosition.side !== PositionSide.NONE) {
                     logger.warn(c.yellow("Resetting internal position to NONE as entry confirmation failed."));
                     this.currentPosition = { side: PositionSide.NONE, size: 0.0, entryPrice: 0.0 };
                     this.state.lastOrderSide = PositionSide.NONE;
                     await this.saveState();
                }
                return null; // Indicate potential failure
            }

        } catch (e) {
            if (e instanceof ccxt.InsufficientFunds) {
                logger.error(c.red(`Failed to place ${side} market entry order: Insufficient Funds. Message: ${e.message}`));
            } else if (e instanceof ccxt.InvalidOrder) {
                 logger.error(c.red(`Failed to place ${side} market entry order: Invalid Order. Message: ${e.message}`), e.stack);
            } else if (e instanceof ccxt.ExchangeError) {
                 logger.error(c.red(`Failed to place ${side} market entry order due to ExchangeError: ${e.message}`), e.stack);
            }
            else {
                logger.error(c.red(`Failed to place ${side} market entry order after retries: ${e.constructor.name} - ${e.message}`), e.stack);
            }
            this.notifier.sendSms(`ERROR: Failed to place ${side} entry order for ${this.config.symbol}. Reason: ${e.message.substring(0,100)}`, this.config);
            // Ensure internal state is consistent (should still be flat)
            await this.updateCurrentPositionState(true); // Verify actual state
            if (this.currentPosition.side !== PositionSide.NONE) {
                logger.warn(c.yellow("Position state is not NONE after failed entry attempt. Resetting internal state to NONE."));
                this.currentPosition = { side: PositionSide.NONE, size: 0.0, entryPrice: 0.0 };
                this.state.lastOrderSide = PositionSide.NONE;
                await this.saveState();
            }
            return null; // Indicate failure
        }
    }

    /**
     * Places Stop Loss and/or Take Profit orders using CCXT's `createOrder`.
     * Cancels existing bot-managed SL/TP orders before placing new ones.
     * Updates `activeOrders` state.
     * @param {PositionSide} positionSide - The side of the current position (LONG or SHORT).
     * @param {number} positionSize - The size of the current position (must be positive).
     * @param {number | null} stopLossPrice - The SL trigger price. If null, SL order is not placed.
     * @param {number | null} takeProfitPrice - The TP trigger price. If null, TP order is not placed.
     * @returns {Promise<{slOrder: ccxt.Order | object | null, tpOrder: ccxt.Order | object | null}>} Object containing placed order details (or simulation), or nulls on failure.
     */
    async placeSLTPOrders(positionSide, positionSize, stopLossPrice, takeProfitPrice) {
        if (positionSide === PositionSide.NONE || positionSize <= 0) {
             logger.error(c.red("Cannot place SL/TP orders: Invalid position side or size provided."), { positionSide, positionSize });
             return { slOrder: null, tpOrder: null };
        }
        if (stopLossPrice === null && takeProfitPrice === null) {
             logger.info("Both Stop Loss and Take Profit prices are null. No orders to place.");
             return { slOrder: null, tpOrder: null };
        }
        if ((stopLossPrice !== null && (typeof stopLossPrice !== 'number' || !Number.isFinite(stopLossPrice) || stopLossPrice <= 0)) ||
            (takeProfitPrice !== null && (typeof takeProfitPrice !== 'number' || !Number.isFinite(takeProfitPrice) || takeProfitPrice <= 0))) {
             logger.error(c.red("Cannot place SL/TP orders: Invalid SL or TP price provided."), { stopLossPrice, takeProfitPrice });
             return { slOrder: null, tpOrder: null };
        }

        const orderSide = positionSide === PositionSide.LONG ? Side.SELL : Side.BUY; // Opposite side to close

        // --- Dry Run Simulation ---
        if (this.config.dry_run) {
             logger.info(c.magenta(`DRY RUN: Simulating place SL @ ${stopLossPrice ?? 'N/A'} and TP @ ${takeProfitPrice ?? 'N/A'} for ${positionSide} ${positionSize} ${this.marketInfo.base}.`));
             let simSlOrder = null, simTpOrder = null;
             await this.cancelAllSLTPOrders("Placing new SL/TP (Dry Run)"); // Simulate cancellation
             if (stopLossPrice !== null) {
                 simSlOrder = { id: `dry_sl_${Date.now()}`, price: stopLossPrice, triggerPrice: stopLossPrice, status: 'open', info: { simulated: true, triggerPrice: stopLossPrice } };
                 this.activeOrders.stopLoss = { id: simSlOrder.id, price: stopLossPrice };
             } else {
                 this.activeOrders.stopLoss = null; // Clear if not placing
             }
             if (takeProfitPrice !== null) {
                 simTpOrder = { id: `dry_tp_${Date.now()}`, price: takeProfitPrice, triggerPrice: takeProfitPrice, status: 'open', info: { simulated: true, triggerPrice: takeProfitPrice } };
                 this.activeOrders.takeProfit = { id: simTpOrder.id, price: takeProfitPrice };
             } else {
                 this.activeOrders.takeProfit = null; // Clear if not placing
             }
             await this.saveState(); // Save simulated state
             const message = `DRY RUN: Set SL=${stopLossPrice ?? 'N/A'}, TP=${takeProfitPrice ?? 'N/A'} for ${this.config.symbol}`;
             logger.info(c.magenta(message));
             this.notifier.sendSms(message, this.config);
             return { slOrder: simSlOrder, tpOrder: simTpOrder };
        }

        // --- Live Trading ---
        logger.info(`Attempting to place SL @ ${c.bold(stopLossPrice ?? 'N/A')} and TP @ ${c.bold(takeProfitPrice ?? 'N/A')} for ${positionSide} ${positionSize} ${this.marketInfo.base}...`);

        let slOrderResult = null;
        let tpOrderResult = null;

        try {
            // --- Cancel Existing SL/TP Orders First ---
            await this.cancelAllSLTPOrders("Placing new SL/TP");

            // --- Common Parameters for Bybit V5 Conditional Orders ---
            // Note: CCXT generally handles mapping unified params to exchange-specific ones.
            // For Bybit V5 Stop/TakeProfit market orders triggered by price:
            const baseParams = {
                'category': 'linear',
                'positionIdx': 0, // 0 for one-way mode
                'reduceOnly': true, // SL/TP orders must only reduce/close the position
                'triggerBy': this.config.order_trigger_price_type, // e.g., 'LastPrice', 'MarkPrice'
                'timeInForce': this.config.time_in_force, // GTC usually for SL/TP
                // 'slTriggerBy': this.config.order_trigger_price_type, // Bybit specific alternative
                // 'tpTriggerBy': this.config.order_trigger_price_type, // Bybit specific alternative
            };

            // --- Place Stop Loss Order ---
            if (stopLossPrice !== null) {
                 // For Bybit V5 via CCXT: Use createOrder with type 'Stop' (maps to Stop Market typically)
                 // and provide stopLossPrice in params.
                const slParams = {
                    ...baseParams,
                    'stopLossPrice': stopLossPrice, // CCXT unified parameter for stop loss trigger price
                    'triggerDirection': positionSide === PositionSide.LONG ? 2 : 1, // 2=Falling (Sell SL), 1=Rising (Buy SL) for Bybit
                };
                logger.debug(`Placing Stop Loss (${orderSide}) order. Size=${positionSize}, Trigger=${stopLossPrice}, Params:`, slParams);

                 slOrderResult = await retryOnException(
                     async () => await this.exchangeManager.exchange.createOrder(
                         this.config.symbol,
                         'Stop', // CCXT order type for Stop Market/Limit
                         orderSide,
                         positionSize,
                         undefined, // Price (not needed for market stop)
                         slParams
                     ),
                     this.config.max_retries, this.config.retry_delay, undefined, 'createOrder (StopLoss)'
                 );
                 const actualSlTrigger = slOrderResult.stopLossPrice || slOrderResult.triggerPrice || slOrderResult.info?.triggerPrice || stopLossPrice;
                logger.info(c.green(`Stop Loss order placed successfully. ID: ${slOrderResult.id}, Trigger Price: ${actualSlTrigger}`));
                this.activeOrders.stopLoss = { id: slOrderResult.id, price: actualSlTrigger };
            }

            // --- Place Take Profit Order ---
            if (takeProfitPrice !== null) {
                // For Bybit V5 via CCXT: Use createOrder with type 'TakeProfit' (maps to Take Profit Market typically)
                // and provide takeProfitPrice in params.
                const tpParams = {
                    ...baseParams,
                    'takeProfitPrice': takeProfitPrice, // CCXT unified parameter for take profit trigger price
                    'triggerDirection': positionSide === PositionSide.LONG ? 1 : 2, // 1=Rising (Sell TP), 2=Falling (Buy TP) for Bybit
                };
                 logger.debug(`Placing Take Profit (${orderSide}) order. Size=${positionSize}, Trigger=${takeProfitPrice}, Params:`, tpParams);

                 tpOrderResult = await retryOnException(
                     async () => await this.exchangeManager.exchange.createOrder(
                         this.config.symbol,
                         'TakeProfit', // CCXT order type for Take Profit Market/Limit
                         orderSide,
                         positionSize,
                         undefined, // Price (not needed for market TP)
                         tpParams
                     ),
                     this.config.max_retries, this.config.retry_delay, undefined, 'createOrder (TakeProfit)'
                 );
                 const actualTpTrigger = tpOrderResult.takeProfitPrice || tpOrderResult.triggerPrice || tpOrderResult.info?.triggerPrice || takeProfitPrice;
                logger.info(c.green(`Take Profit order placed successfully. ID: ${tpOrderResult.id}, Trigger Price: ${actualTpTrigger}`));
                this.activeOrders.takeProfit = { id: tpOrderResult.id, price: actualTpTrigger };
            }

            // --- Save state and Notify ---
            await this.saveState(); // Save the updated active order IDs/prices
            const message = `LIVE: Set SL=${this.activeOrders.stopLoss?.price ?? 'N/A'}, TP=${this.activeOrders.takeProfit?.price ?? 'N/A'} for ${this.config.symbol}`;
            this.notifier.sendSms(message, this.config);
            return { slOrder: slOrderResult, tpOrder: tpOrderResult };

        } catch (e) {
             if (e instanceof ccxt.InvalidOrder && (e.message.includes("trigger price") || e.message.includes("too close") || e.message.includes("ret_code=30021") || e.message.includes("ret_code=30071") || e.message.includes("ret_code=110013"))) { // 110013: TP/SL price invalid
                  logger.error(c.red(`Failed to place SL/TP order: Invalid trigger price or too close/far from market price. SL=${stopLossPrice}, TP=${takeProfitPrice}. Message: ${e.message}`), e.stack);
             } else if (e instanceof ccxt.ExchangeError && (e.message.includes("position size is zero") || e.message.includes("ret_code=30010") || e.message.includes("ret_code=30067") || e.message.includes("ret_code=30041") || e.message.includes("ret_code=110043"))) { // 110043: Position size is zero
                  logger.error(c.red(`Failed to place SL/TP order: Exchange error suggests issue with position, parameters, or limits. Message: ${e.message}`));
                  await this.updateCurrentPositionState(true); // Force position update
             }
             else {
                 logger.error(c.red(`Failed to place SL/TP orders: ${e.constructor.name} - ${e.message}`), e.stack);
             }
            this.notifier.sendSms(`ERROR: Failed to set SL/TP for ${this.config.symbol}. Reason: ${e.message.substring(0,100)}. Check logs!`, this.config);
            logger.error(c.red("CRITICAL: Failed to place SL/TP orders. Position may be unprotected. Manual intervention might be required."));
            // Save state even on failure, as activeOrders might have been partially updated or cleared
            await this.saveState();
            return { slOrder: null, tpOrder: null }; // Indicate failure
        }
    }

    /**
     * Closes the current open position using a market order.
     * Fetches the latest position state first and cancels existing SL/TP orders.
     * Updates simulated balance in dry run.
     * @param {string} reason - A short reason for closing the position (for logging and notification).
     * @returns {Promise<ccxt.Order | object | null>} The closing order object, simulation object, or null on failure or if no position exists.
     */
    async closePosition(reason) {
        // 1. Get the most up-to-date position state (force fetch)
        await this.updateCurrentPositionState(true);
        const { side: currentSide, size: currentSize, entryPrice: currentEntry } = this.currentPosition;

        // 2. Check if there's actually a position to close
        if (currentSide === PositionSide.NONE || currentSize <= 0) {
            logger.info(`No open position found for ${this.config.symbol}. Cannot close. Reason given: ${reason}`);
            await this.cancelAllSLTPOrders(`Position already closed/none (${reason})`);
            return null;
        }

        const orderSide = currentSide === PositionSide.LONG ? Side.SELL : Side.BUY; // Opposite side to close

        // --- Dry Run Simulation ---
        if (this.config.dry_run) {
            logger.info(c.magenta(`DRY RUN: Simulating close ${c.bold(currentSide)} position of ${c.bold(currentSize)} ${this.marketInfo.base}. Reason: ${reason}`));
            const closedSide = currentSide; const closedSize = currentSize; const closedEntry = currentEntry;
            const ticker = await this.exchangeManager.fetchTicker();
            const simulatedExitPrice = (ticker?.bid && ticker?.ask && ticker.bid > 0 && ticker.ask > 0) ? (orderSide === Side.BUY ? ticker.ask : ticker.bid) : (ticker?.last || 0);
            const formattedSimExitPrice = parseFloat(this.exchangeManager.exchange.priceToPrecision(this.config.symbol, simulatedExitPrice));
            const simulatedCost = closedSize * formattedSimExitPrice;
            const simulatedFee = simulatedCost * (this.marketInfo?.taker || 0.0006);
            const pnl = (closedSide === PositionSide.LONG)
                        ? (formattedSimExitPrice - closedEntry) * closedSize - simulatedFee
                        : (closedEntry - formattedSimExitPrice) * closedSize - simulatedFee;

            this.currentPosition = { side: PositionSide.NONE, size: 0.0, entryPrice: 0.0 };
            this.state.lastOrderSide = PositionSide.NONE;
            await this.cancelAllSLTPOrders(`Position Closed (Dry Run): ${reason}`); // Simulates cancellation
            this.simulatedBalance += pnl; // Update simulated balance with PnL
            logger.info(c.magenta(`DRY RUN: Simulated Exit @ ~${formattedSimExitPrice}. PnL: ${pnl.toFixed(4)} ${this.config.currency}. Fee: ${simulatedFee.toFixed(4)}. New Balance: ${this.simulatedBalance.toFixed(4)}`));
            await this.saveState(); // Save simulated state
            const message = `DRY RUN: Closed ${closedSide} ${closedSize} ${this.config.symbol}. Reason: ${reason}. PnL: ${pnl.toFixed(2)}`;
            logger.info(c.magenta(message));
            this.notifier.sendSms(message, this.config);
            return { id: `dry_close_${Date.now()}`, symbol: this.config.symbol, type: 'market', side: orderSide, amount: closedSize, filled: closedSize, remaining: 0, price: formattedSimExitPrice, average: formattedSimExitPrice, cost: simulatedCost, status: 'closed', fee: { cost: simulatedFee, currency: this.config.currency }, info: { simulated: true, reason: reason, pnl: pnl } };
        }

        // --- Live Trading ---
        logger.info(`Attempting to close ${c.bold(currentSide)} position of ${c.bold(currentSize)} ${this.marketInfo.base} via market order. Reason: ${reason}`);

        try {
            // 3. Cancel existing SL/TP orders *before* placing the closing market order
            await this.cancelAllSLTPOrders(`Closing position (${reason})`);
            // cancelAllSLTPOrders already saves state after clearing orders

            // 4. Place the closing market order
            const params = {
                'reduceOnly': true, // CRITICAL: Ensures order only closes position, doesn't open opposite
                'category': 'linear',
                'positionIdx': 0,
            };
            logger.debug(`Placing closing market order (${orderSide}). Size=${currentSize}, Params:`, params);

            const placeCloseOrderFunc = async () => await this.exchangeManager.exchange.createMarketOrder( this.config.symbol, orderSide, currentSize, undefined, params );

            const order = await retryOnException( placeCloseOrderFunc, this.config.max_retries, this.config.retry_delay, [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection], 'createMarketOrder (Close)' );

            logger.info(c.green(`Position close market order placed successfully. Order ID: ${order.id}. Status: ${order.status}. Reason: ${reason}`));
            logger.debug("Close Order Details:", order);

            // 5. Update internal state optimistically and save
            const closedSide = this.currentPosition.side; const closedSize = this.currentPosition.size;
            this.currentPosition = { side: PositionSide.NONE, size: 0.0, entryPrice: 0.0 };
            this.state.lastOrderSide = PositionSide.NONE;
            await this.saveState(); // Save state after successful close order placement

            // 6. Notify
            // PnL calculation is complex without fetching execution details, just notify closure.
            const message = `LIVE: Closed ${closedSide} ${closedSize.toFixed(this.marketInfo?.precision?.amount || 8)} ${this.config.symbol}. Reason: ${reason}`;
            this.notifier.sendSms(message, this.config);

            // 7. Optional: Wait and verify closure (next cycle's updateCurrentPositionState usually handles this)
            // await sleep(3000);
            // await this.updateCurrentPositionState(true);
            // ... verification logic ...

            return order; // Return the closing order object

        } catch (e) {
            if (e instanceof ccxt.ExchangeError && (e.message.includes("reduce-only") || e.message.includes("position size is zero") || e.message.includes("ret_code=30017") || e.message.includes("ret_code=30041") || e.message.includes("ret_code=110043") || e.message.includes("ret_code=110025"))) { // 110025: Position idx not match position mode
                 logger.warn(c.yellow(`Failed to place closing order, possibly because position was already closed, size mismatch, or reduceOnly issue. Message: ${e.message}. Verifying position state...`));
                 await this.updateCurrentPositionState(true); // Re-check actual position
                 if(this.currentPosition.side !== PositionSide.NONE) {
                      logger.error(c.red("Position still exists after failed close attempt and verification. Manual intervention likely needed."));
                      this.notifier.sendSms(`CRITICAL ERROR: Failed to CLOSE position for ${this.config.symbol} but position still exists. MANUAL INTERVENTION NEEDED.`, this.config);
                 } else {
                      logger.info(c.green("Position is confirmed closed after failed close attempt."));
                      // Ensure state reflects closure
                      this.state.lastOrderSide = PositionSide.NONE;
                      await this.saveState();
                 }
            } else if (e instanceof ccxt.InsufficientFunds && params.reduceOnly) {
                 logger.error(c.red(`Failed to place closing order due to Insufficient Funds, even with reduceOnly=true? Unexpected error. Message: ${e.message}`), e.stack);
                 this.notifier.sendSms(`CRITICAL ERROR: Failed to CLOSE position for ${this.config.symbol} due to INSUFFICIENT FUNDS (unexpected with reduceOnly). Check margin/account.`, this.config);
            }
            else {
                logger.error(c.red(`Failed to close position: ${e.constructor.name} - ${e.message}`), e.stack);
                this.notifier.sendSms(`ERROR: Failed to CLOSE position for ${this.config.symbol}. Reason: ${e.message.substring(0,100)}. MANUAL INTERVENTION MAY BE NEEDED.`, this.config);
            }
            // Save state even on failure to capture any partial updates (like cleared orders)
            await this.saveState();
            return null; // Indicate failure
        }
    }

    /**
     * Cancels active Stop Loss and Take Profit orders associated with the bot,
     * based on the order IDs stored in `this.activeOrders`. Clears relevant `activeOrders` entries.
     * @param {string} reason - Reason for cancellation (for logging).
     * @returns {Promise<void>}
     */
    async cancelAllSLTPOrders(reason) {
        logger.debug(`Attempting to cancel active SL/TP orders. Reason: ${reason}`);

        const ordersToCancel = [];
        if (this.activeOrders.stopLoss?.id) ordersToCancel.push({ type: 'SL', id: this.activeOrders.stopLoss.id });
        if (this.activeOrders.takeProfit?.id) ordersToCancel.push({ type: 'TP', id: this.activeOrders.takeProfit.id });

        if (ordersToCancel.length === 0) {
            logger.debug("No active bot-managed SL/TP orders found in state to cancel.");
            return;
        }

        // --- Dry Run Simulation ---
        if (this.config.dry_run) {
            logger.info(c.magenta(`DRY RUN: Simulating cancellation of ${ordersToCancel.length} orders. Reason: ${reason}`));
            ordersToCancel.forEach(order => logger.info(c.magenta(` - Simulating cancel ${order.type} order ${order.id}`)));
            this.activeOrders.stopLoss = null;
            this.activeOrders.takeProfit = null;
            // No need to save state here, caller will save if necessary
            return;
        }

        // --- Live Trading ---
        const params = { 'category': 'linear' };
        let stateChanged = false;

        // Use Promise.allSettled to attempt cancellation of all orders even if one fails
        const cancelPromises = ordersToCancel.map(order => {
            return (async () => {
                try {
                    logger.info(`Cancelling ${order.type} order ${order.id}... Reason: ${reason}`);
                    const cancelFunc = async () => await this.exchangeManager.exchange.cancelOrder(order.id, this.config.symbol, params);
                    await retryOnException( cancelFunc, Math.max(0, this.config.max_retries - 1), this.config.retry_delay / 2, undefined, `cancelOrder (${order.type} ID: ${order.id})` );
                    logger.info(c.green(`${order.type} order ${order.id} cancelled successfully.`));
                    return { status: 'fulfilled', type: order.type, id: order.id };
                } catch (e) {
                    // Treat OrderNotFound or similar errors (already closed/cancelled) as success for state clearing
                    if (e instanceof ccxt.OrderNotFound || (e instanceof ccxt.ExchangeError && (e.message.includes("Order does not exist") || e.message.includes("already closed") || e.message.includes("has been filled") || e.message.includes("canceled") || e.message.includes("ret_code=30034") || e.message.includes("ret_code=10001") || e.message.includes("ret_code=110001")))) { // 110001: Order not found or finished
                        logger.warn(c.yellow(`${order.type} order ${order.id} not found or already closed/cancelled on exchange. Assuming gone.`));
                        return { status: 'fulfilled', type: order.type, id: order.id }; // Treat as success for state clearing
                    } else {
                        logger.error(c.red(`Failed to cancel ${order.type} order ${order.id}: ${e.constructor.name} - ${e.message}`), e.stack);
                        this.notifier.sendSms(`ERROR: Failed to cancel ${order.type} order ${order.id} for ${this.config.symbol}. Reason: ${reason}. Check exchange!`, this.config);
                        return { status: 'rejected', type: order.type, id: order.id, reason: e }; // Indicate failure
                    }
                }
            })();
        });

        const results = await Promise.allSettled(cancelPromises);

        // --- Clear the cancelled orders from internal state ---
        results.forEach(result => {
            // If cancellation succeeded or order was not found, clear the state
            if (result.status === 'fulfilled' && result.value) {
                 const { type, id } = result.value;
                 if (type === 'SL' && this.activeOrders.stopLoss?.id === id) {
                     logger.debug(`Clearing SL order ${id} from internal state.`);
                     this.activeOrders.stopLoss = null;
                     stateChanged = true;
                 }
                 if (type === 'TP' && this.activeOrders.takeProfit?.id === id) {
                     logger.debug(`Clearing TP order ${id} from internal state.`);
                     this.activeOrders.takeProfit = null;
                     stateChanged = true;
                 }
            } else if (result.status === 'rejected') {
                 // Log the failure but do not clear the state - manual intervention might be needed
                 // Extract details safely from the rejection reason
                 const rejectionInfo = result.reason || {};
                 const orderType = rejectionInfo.type || 'UnknownType';
                 const orderId = rejectionInfo.id || 'UnknownID';
                 const errorReason = rejectionInfo.reason || result.reason;
                 logger.error(c.red(`Cancellation failed for order (Type: ${orderType}, ID: ${orderId}). State not cleared. Reason:`), errorReason);
            }
        });

        // Save state immediately if orders were cleared
        if (stateChanged) {
            await this.saveState();
        }
        logger.debug("Finished attempting to cancel active orders.");
    }

    /**
     * Fetches the current position from the exchange and updates the internal `this.currentPosition` state.
     * Uses the cached `getExchangePosition` method unless forced. Handles Dry Run.
     * @param {boolean} [forceFetch=false] - If true, bypasses the cache for this fetch.
     * @returns {Promise<void>}
     */
    async updateCurrentPositionState(forceFetch = false) {
        logger.debug(`Updating current position state...${forceFetch ? c.yellow(' (Forced Fetch)') : ''}`);

        let fetchedPosition = null;
        if (this.config.dry_run) {
             // In dry run, the "state" is purely internal. No fetching.
             logger.debug(c.magenta("DRY RUN: Using internal position state."));
             fetchedPosition = this.currentPosition; // Use the current internal state directly
        } else {
             // Live mode: Fetch from exchange
             if (forceFetch) {
                  logger.debug("Clearing position cache before forced fetch.");
                  this.exchangeManager._setCache('position', this.config.symbol, null);
             }
             fetchedPosition = await this.exchangeManager.getPosition(); // Uses cache & retry internally
        }

        if (fetchedPosition) {
            const previousPosition = { ...this.currentPosition }; // Shallow copy previous state
            const stateChanged = JSON.stringify(previousPosition) !== JSON.stringify(fetchedPosition); // Simple deep compare

            this.currentPosition = fetchedPosition; // Update the internal state

            if (stateChanged) {
                 logger.info(`Position state updated: Side=${c.bold(this.currentPosition.side)}, Size=${c.bold(this.currentPosition.size.toFixed(this.marketInfo?.precision?.amount || 8))}, Entry=${this.currentPosition.entryPrice.toFixed(this.marketInfo?.precision?.price || 8)} (Previous: Side=${previousPosition.side}, Size=${previousPosition.size.toFixed(this.marketInfo?.precision?.amount || 8)}, Entry=${previousPosition.entryPrice.toFixed(this.marketInfo?.precision?.price || 8)})`);
                 // If position closed unexpectedly, ensure active orders are cleared
                 if (this.currentPosition.side === PositionSide.NONE && previousPosition.side !== PositionSide.NONE) {
                      logger.warn(c.yellow("Position appears to have closed unexpectedly (e.g., SL/TP hit). Clearing active orders from state."));
                      await this.cancelAllSLTPOrders("Position closed unexpectedly"); // Attempt cancellation, then clear state
                 }
                 await this.saveState(); // Save state immediately on change
            } else {
                logger.debug(`Fetched position matches internal state: Side=${this.currentPosition.side}, Size=${this.currentPosition.size}`);
            }
        } else if (!this.config.dry_run) {
            // If getPosition returns null (due to fetch error after retries) in LIVE mode
            logger.error(c.red("Failed to fetch current position to update state after retries. Internal position state may be inaccurate. Keeping previous state."));
        }
    }


    // --- Trailing Stop Loss Logic ---

    /**
     * Manages the Trailing Stop Loss (TSL) by modifying the existing SL order if price moves favorably.
     * @returns {Promise<void>}
     */
    async manageTrailingStop() {
        // Check if TSL is enabled and we have a position and an active SL order
        if (this.config.trailing_stop_mult <= 0 || this.currentPosition.side === PositionSide.NONE || !this.activeOrders.stopLoss?.id || !this.activeOrders.stopLoss?.price) {
            return; // Skip if TSL disabled, no position, or no active SL
        }

        const { side: positionSide, size: positionSize } = this.currentPosition;
        const currentSlPrice = this.activeOrders.stopLoss.price;
        const currentSlId = this.activeOrders.stopLoss.id;

        logger.debug(`Managing Trailing Stop for ${positionSide} position... Current SL ID: ${currentSlId}, Price: ${currentSlPrice}`);

        try {
            // --- Get Current Price & ATR ---
            const ticker = await this.exchangeManager.fetchTicker();
            if (!ticker || !ticker.last || !Number.isFinite(ticker.last) || ticker.last <= 0) {
                logger.warn(c.yellow("Cannot manage TSL: Failed to fetch valid current price from ticker."));
                return;
            }
            const currentPrice = ticker.last; // Use last price as reference for trailing

            const ohlcvData = this.historicalOHLCV[this.config.timeframe];
            const atrPeriod = this.config.long_st_period;
            if (!ohlcvData || ohlcvData.length < atrPeriod) {
                 logger.warn(c.yellow("Cannot manage TSL: Insufficient historical data for ATR calculation."));
                 return;
            }
            const highPrices = ohlcvData.map(c => c[OHLCV_INDEX.HIGH]);
            const lowPrices = ohlcvData.map(c => c[OHLCV_INDEX.LOW]);
            const closePrices = ohlcvData.map(c => c[OHLCV_INDEX.CLOSE]);
            const atrValues = calculateATR(highPrices, lowPrices, closePrices, atrPeriod, logger);
             let currentAtr = null;
             for(let i = atrValues.length - 1; i >= 0; i--) { if(Number.isFinite(atrValues[i]) && atrValues[i] > 0) { currentAtr = atrValues[i]; break; } }
            if (currentAtr === null) { logger.warn(c.yellow(`Cannot manage TSL: Could not find valid ATR value.`)); return; }

            // --- Calculate New TSL Price ---
            const tslDistance = currentAtr * this.config.trailing_stop_mult;
            let potentialTslPrice;
            if (positionSide === PositionSide.LONG) { potentialTslPrice = currentPrice - tslDistance; }
            else { potentialTslPrice = currentPrice + tslDistance; } // SHORT

            // Apply price precision
            try { potentialTslPrice = parseFloat(this.exchangeManager.exchange.priceToPrecision(this.config.symbol, potentialTslPrice)); }
            catch (e) { logger.error(c.red(`Error applying price precision to TSL price: ${e.message}. Raw TSL=${potentialTslPrice}`)); return; }
            if (isNaN(potentialTslPrice) || potentialTslPrice <= 0) { logger.warn(c.yellow(`Cannot manage TSL: Calculated potential TSL price (${potentialTslPrice}) is invalid.`)); return; }

            // --- Compare with Current SL and Update if Better ---
            let shouldUpdateSl = false;
            // Define a buffer relative to the current price to avoid placing the SL too aggressively
            // E.g., ensure SL is at least 0.1 * ATR away from the current price
            const priceBuffer = Math.max(this.marketInfo?.tickSize * 5, currentAtr * 0.1);

            if (positionSide === PositionSide.LONG) {
                 // New SL must be higher than old SL AND lower than current price minus buffer
                 if (potentialTslPrice > currentSlPrice && potentialTslPrice < currentPrice - priceBuffer) { shouldUpdateSl = true; }
            } else if (positionSide === PositionSide.SHORT) {
                 // New SL must be lower than old SL AND higher than current price plus buffer
                 if (potentialTslPrice < currentSlPrice && potentialTslPrice > currentPrice + priceBuffer) { shouldUpdateSl = true; }
            }

            if (shouldUpdateSl) {
                logger.info(c.blue(`Trailing Stop Update Triggered (${positionSide}): New TSL Price ${potentialTslPrice} is better than Current SL ${currentSlPrice}. Attempting to modify SL order...`));
                this.notifier.sendSms(`TSL Update (${positionSide} ${this.config.symbol}): Moving SL from ${currentSlPrice} to ${potentialTslPrice}`, this.config);

                // --- Execute SL Update (Modify Order) ---
                // Bybit V5 supports modifying conditional orders (SL/TP trigger price)
                if (this.exchangeManager.exchange.has['editOrder']) {
                    try {
                        const orderSide = positionSide === PositionSide.LONG ? Side.SELL : Side.BUY;
                        const editParams = {
                            'category': 'linear',
                            'stopLossPrice': potentialTslPrice, // New trigger price (CCXT unified)
                            // 'triggerPrice': potentialTslPrice, // Bybit might require triggerPrice as well for conditional orders
                            // CCXT might handle this mapping; test if stopLossPrice alone works.
                            // If not, uncomment and test with triggerPrice.
                            // Ensure other params are NOT included unless necessary (like size, which shouldn't change)
                        };
                        logger.debug(`Attempting to modify SL order ${currentSlId} with new price ${potentialTslPrice}... Params:`, editParams);
                        const editOrderFunc = async () => await this.exchangeManager.exchange.editOrder(currentSlId, this.config.symbol, 'Stop', orderSide, undefined, undefined, editParams); // Pass undefined for amount/price

                        const editedOrder = await retryOnException(editOrderFunc, this.config.max_retries, this.config.retry_delay / 2, undefined, 'editOrder (TSL)');

                        const actualNewSl = editedOrder.stopLossPrice || editedOrder.triggerPrice || editedOrder.info?.triggerPrice || potentialTslPrice;
                        const newOrderId = editedOrder.id || currentSlId; // Use new ID if returned, else assume old ID
                        logger.info(c.green(`Successfully modified SL order ${currentSlId} -> ${newOrderId} for TSL. New Trigger Price: ${actualNewSl}`));
                        // Update internal state
                        this.activeOrders.stopLoss = { id: newOrderId, price: actualNewSl };
                        await this.saveState();

                    } catch (e) {
                         // Handle specific errors if Bybit rejects modification (e.g., price too close, order filled)
                         if (e instanceof ccxt.OrderNotFound || (e instanceof ccxt.ExchangeError && (e.message.includes("Order does not exist") || e.message.includes("already closed") || e.message.includes("has been filled") || e.message.includes("canceled") || e.message.includes("ret_code=110001") || e.message.includes("ret_code=30034")))) {
                              logger.warn(c.yellow(`Cannot modify SL order ${currentSlId} for TSL: Order already closed/cancelled. Clearing state.`));
                              this.activeOrders.stopLoss = null;
                              await this.saveState();
                         } else if (e instanceof ccxt.InvalidOrder && (e.message.includes("modify price") || e.message.includes("too close") || e.message.includes("ret_code=110015") || e.message.includes("ret_code=110017") || e.message.includes("ret_code=110064"))) { // 110015: Modify margin error, 110017: Modify price error, 110064: Modify TP/SL error
                              logger.error(c.red(`Failed to modify SL order ${currentSlId}: Invalid modification parameters (e.g., price too close). Message: ${e.message}. Retrying with Cancel & Replace.`));
                              await this.fallbackCancelReplaceSL(positionSide, positionSize, potentialTslPrice);
                         } else {
                              logger.error(c.red(`Failed to modify SL order ${currentSlId} for TSL update: ${e.constructor.name} - ${e.message}. Falling back to Cancel & Replace.`), e.stack);
                              // Fallback to Cancel & Replace if modify fails for other reasons
                              await this.fallbackCancelReplaceSL(positionSide, positionSize, potentialTslPrice);
                         }
                    }
                } else {
                     logger.warn(c.yellow(`Exchange ${this.exchangeManager.exchange.id} does not support editOrder via CCXT. Using Cancel & Replace for TSL.`));
                     await this.fallbackCancelReplaceSL(positionSide, positionSize, potentialTslPrice);
                }
            } else {
                logger.debug(`TSL check (${positionSide}): Potential TSL (${potentialTslPrice}) not better than current SL (${currentSlPrice}) or too close to current price (${currentPrice}). No update needed.`);
            }
        } catch (error) {
            logger.error(c.red("Error occurred during Trailing Stop Loss management:"), error.stack);
            this.notifier.sendSms(`ERROR during TSL management for ${this.config.symbol}: ${error.message.substring(0,100)}. Check logs!`, this.config);
        }
    }

    /**
     * Helper function to perform Cancel & Replace for SL update (TSL fallback).
     * @param {PositionSide} positionSide - The side of the position.
     * @param {number} positionSize - The size of the position.
     * @param {number} newSlPrice - The new Stop Loss price to set.
     * @returns {Promise<void>}
     * @private
     */
    async fallbackCancelReplaceSL(positionSide, positionSize, newSlPrice) {
        logger.info(`Performing Cancel & Replace for SL update to ${newSlPrice}...`);
        const currentTpPrice = this.activeOrders.takeProfit?.price ?? null; // Keep existing TP if any

        // placeSLTPOrders handles cancellation of old orders and placement of new ones
        const { slOrder: newSlOrder } = await this.placeSLTPOrders(positionSide, positionSize, newSlPrice, currentTpPrice);

        if (newSlOrder && this.activeOrders.stopLoss?.id) {
            logger.info(c.green(`Successfully updated Stop Loss via Cancel & Replace. New SL ID: ${this.activeOrders.stopLoss.id}, Price: ${this.activeOrders.stopLoss.price}`));
        } else {
            logger.error(c.red(`Failed to place new Stop Loss order at ${newSlPrice} during TSL fallback (Cancel & Replace). Position might be unprotected!`));
            this.notifier.sendSms(`CRITICAL ERROR: Failed to place new SL after TSL update fallback for ${this.config.symbol}. POSITION UNPROTECTED!`, this.config);
        }
    }


    // --- Main Trading Cycle ---

    /**
     * Executes a single trading cycle: fetches data, calculates indicators, generates signals, manages position.
     * @returns {Promise<void>}
     */
    async runTradingCycle() {
        if (this._stop_requested || !this._isRunning) {
            logger.info(`Stop requested or bot not running, skipping trading cycle.`);
            this._isRunning = false; // Ensure loop stops
            return;
        }

        const cycleStartTime = Date.now();
        logger.info(c.blue(`--- Starting Trading Cycle [${new Date(cycleStartTime).toISOString()}] ---`));

        try {
            // 1. Update Position State from Exchange
            await this.updateCurrentPositionState();
            if (this._stop_requested) return;

            // 2. Calculate Indicators
            // This also saves state if indicators are calculated successfully
            const indicatorsUpdated = await this.calculateIndicators();
            if (this._stop_requested) return;

            // 3. Generate Trading Signal (only if indicators updated successfully)
            const signal = indicatorsUpdated ? this.generateSignal() : PositionSide.NONE;
            if (this._stop_requested) return;

            // --- 4. Position Management Logic ---

            // === CASE A: Currently IN a Position ===
            if (this.currentPosition.side !== PositionSide.NONE) {
                logger.info(`Currently in a ${c.bold(this.currentPosition.side)} position. Size: ${this.currentPosition.size}, Entry: ${this.currentPosition.entryPrice}`);

                // A.1: Check for Exit Signal based on Indicators (e.g., opposite ST confirmation)
                let exitReason = null;
                if (indicatorsUpdated && this.lastSupertrendSignals.short && this.lastSupertrendSignals.long) {
                    const shortStTrend = this.lastSupertrendSignals.short.trend;
                    const longStTrend = this.lastSupertrendSignals.long.trend;
                    if (this.currentPosition.side === PositionSide.LONG && (shortStTrend === -1 || longStTrend === -1)) {
                        exitReason = `Exit Long: ST Flip (${shortStTrend}/${longStTrend})`;
                    } else if (this.currentPosition.side === PositionSide.SHORT && (shortStTrend === 1 || longStTrend === 1)) {
                        exitReason = `Exit Short: ST Flip (${shortStTrend}/${longStTrend})`;
                    }
                }

                if (exitReason) {
                    logger.info(c.yellow(`Exit condition met: ${exitReason}`));
                    await this.closePosition(exitReason);
                    // Position is now closed, cycle ends for position management part
                }
                // A.2: No Exit Signal -> Manage Trailing Stop Loss
                else {
                    logger.debug("No exit signal detected. Managing Trailing Stop Loss...");
                    await this.manageTrailingStop();
                    // TSL management might update state, which is saved internally
                }
            }

            // === CASE B: Currently OUT of a Position (Flat) ===
            else { // currentPosition.side === PositionSide.NONE
                logger.info("Currently flat. Checking for entry signals...");

                // B.1: Check for Valid Entry Signal (LONG or SHORT)
                if (signal === PositionSide.LONG || signal === PositionSide.SHORT) {
                    logger.info(`Entry signal received: ${c.bold(signal)}. Preparing entry...`);

                    // B.1.1: Calculate SL/TP based on potential entry price
                    const ticker = await this.exchangeManager.fetchTicker();
                    // Use ask for long entry, bid for short entry for more realistic price
                    const potentialEntryPrice = (ticker?.bid && ticker?.ask && ticker.bid > 0 && ticker.ask > 0)
                        ? (signal === PositionSide.LONG ? ticker.ask : ticker.bid)
                        : ticker?.last;
                    if (!potentialEntryPrice || !Number.isFinite(potentialEntryPrice) || potentialEntryPrice <= 0) {
                        logger.error(c.red("Could not get valid current price from ticker for calculations. Aborting entry."));
                        return; // Abort this cycle iteration
                    }
                    const formattedPotEntry = parseFloat(this.exchangeManager.exchange.priceToPrecision(this.config.symbol, potentialEntryPrice));
                    logger.debug(`Using potential entry price for calculations: ${formattedPotEntry}`);

                    const { stopLoss, takeProfit, atrValue } = await this.calculateSLTP(signal, formattedPotEntry);
                    if (stopLoss === null || takeProfit === null) { // Both should be calculated successfully
                         logger.error(c.red("Failed to calculate valid SL/TP prices. Aborting entry attempt."));
                         return; // Abort this cycle iteration
                    }

                    // B.1.2: Calculate Position Size
                    const positionSize = await this.calculatePositionSize(formattedPotEntry, stopLoss, atrValue);
                    if (positionSize === null || positionSize <= 0) {
                         logger.error(c.red("Failed to calculate valid position size. Aborting entry attempt."));
                         return; // Abort this cycle iteration
                    }

                    // B.1.3: Place Entry Order
                    const entryOrder = await this.placeEntryOrder(signal, positionSize);

                    // B.1.4: Place SL/TP Orders *after* confirming entry
                    // Check internal state `this.currentPosition` which was updated by `placeEntryOrder`
                    const sizeTolerance = Math.max(1e-9, this.marketInfo?.limits?.amount?.min / 10 || 1e-9);
                    if (entryOrder && this.currentPosition.side === signal && this.currentPosition.size >= positionSize - sizeTolerance) {
                        logger.info(`Entry successful. Placing SL/TP orders for ${this.currentPosition.side} ${this.currentPosition.size}...`);
                        // Use actual position size and recalculate SL/TP based on actual entry price for accuracy
                        const actualEntryPrice = this.currentPosition.entryPrice;
                        const { stopLoss: finalSL, takeProfit: finalTP } = await this.calculateSLTP(signal, actualEntryPrice);
                        if (finalSL === null || finalTP === null) {
                             logger.error(c.red(`CRITICAL: Failed to recalculate SL/TP based on actual entry price ${actualEntryPrice}. Closing position for safety!`));
                             await this.closePosition("Emergency Close: Failed post-entry SL/TP calculation");
                        } else {
                             await this.placeSLTPOrders(this.currentPosition.side, this.currentPosition.size, finalSL, finalTP);
                        }
                    } else {
                         logger.error(c.red(`Entry order for ${signal} failed or position state unconfirmed. SL/TP orders will not be placed.`));
                    }
                }
                // B.2: No Entry Signal
                else {
                    logger.info("No valid entry signal detected this cycle.");
                }
            } // End of Position Management Logic (A or B)

        } catch (error) {
            // --- Catch unexpected errors during the main cycle ---
            logger.error(c.red(`--- !!! UNEXPECTED ERROR in Trading Cycle !!! ---`), error);
            if (error instanceof Error) logger.debug("Cycle Error Stack:", error.stack);
            this.notifier.sendSms(`CRITICAL ERROR in trading cycle: ${error.message.substring(0,100)}. Bot may be unstable. Check logs!`, this.config);
            // Decide if the bot should stop or just wait before the next cycle
            // For now, let it continue to the next cycle after a delay
            logger.error(c.red(`Cycle encountered an error. Waiting before next cycle...`));
        } finally {
            // --- Schedule the next cycle ---
            const cycleEndTime = Date.now();
            const cycleDuration = cycleEndTime - cycleStartTime;
            const timeframeMs = parseTimeframeToMs(this.config.timeframe);
            const delay = Math.max(1000, timeframeMs - cycleDuration); // Ensure minimum 1s delay
            logger.info(c.blue(`--- Trading Cycle Finished [Duration: ${(cycleDuration / 1000).toFixed(2)}s]. Waiting ${c.dim((delay / 1000).toFixed(1))}s for next cycle ---`));

            if (this._isRunning && !this._stop_requested) {
                this._mainLoopTimeoutId = setTimeout(() => this.runTradingCycle(), delay);
            } else {
                logger.info("Bot run loop is stopping or has been stopped. Not scheduling next cycle.");
                this._isRunning = false; // Explicitly set to false
            }
        }
    }

    /**
     * Starts the main trading loop.
     */
    async start() {
        if (this._isRunning) {
            logger.warn("Bot is already running.");
            return;
        }
        logger.info(c.green.bold(">>> Starting Trading Bot <<<"));
        this._isRunning = true;
        this._stop_requested = false;
        // Initial run immediate, subsequent runs scheduled in finally block of runTradingCycle
        await this.runTradingCycle();
    }

    /**
     * Signals the bot to stop the main trading loop gracefully.
     * @param {string} [reason="Stop requested"] - Reason for stopping.
     */
    async stop(reason = "Stop requested") {
        if (this._stop_requested) {
            logger.info("Stop already requested.");
            return;
        }
        logger.info(c.yellow.bold(`>>> Initiating Graceful Stop: ${reason} <<<`));
        this._stop_requested = true;
        this._isRunning = false; // Prevent scheduling new cycles

        // Clear any pending timeout for the next cycle
        if (this._mainLoopTimeoutId) {
            clearTimeout(this._mainLoopTimeoutId);
            this._mainLoopTimeoutId = null;
            logger.info("Cleared pending trading cycle timeout.");
        }

        // Shutdown procedure will be called by the main execution block or signal handler
        // after the current cycle (if any) finishes or immediately if loop was idle.
    }

    /**
     * Performs graceful shutdown procedures: cancels orders, optionally closes position, closes log file.
     * @returns {Promise<void>}
     */
    async shutdown() {
        logger.info(c.yellow.bold("--- Executing Shutdown Procedure ---"));
        isShuttingDown = true; // Use global flag as well

        // 1. Cancel All Active SL/TP Orders
        logger.info("Cancelling any active bot-managed orders...");
        await this.cancelAllSLTPOrders("Bot Shutdown");

        // 2. Optionally Close Open Position
        if (this.config.close_position_on_shutdown) {
            logger.info("Close position on shutdown is ENABLED. Checking for open position...");
            // Force fetch latest position state before attempting close
            await this.updateCurrentPositionState(true);
            if (this.currentPosition.side !== PositionSide.NONE) {
                
