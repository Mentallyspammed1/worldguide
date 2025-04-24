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

