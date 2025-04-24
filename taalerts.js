/**
 * == Market Analysis Bot Enhanced ==
 *
 * Description:
 * A Node.js script designed for market analysis using the CCXT library. It fetches
 * OHLCV data, calculates various technical indicators (Pivots, MAs, EMAs, ATR, RSI,
 * StochRSI, MACD, Bollinger Bands, Ehlers Super Smoother, Ichimoku Cloud, AO, PSAR, VWAP),
 * generates alerts based on configurable conditions, and sends SMS notifications via Termux.
 *
 * Requirements:
 * - Node.js (LTS recommended)
 * - npm/yarn
 * - Run: `npm install ccxt chalk` or `yarn add ccxt chalk`
 * - Termux Environment (for SMS functionality)
 * - Termux:API package (`pkg install termux-api`)
 * - Termux SMS permissions granted (`termux-setup-storage` might be needed first)
 * - Exchange API Keys (optional, for private endpoints/higher rate limits):
 *   Set as environment variables, e.g.,
 *   `export BYBIT_API_KEY='YOUR_KEY'`
 *   `export BYBIT_API_SECRET='YOUR_SECRET'`
 *   (Replace BYBIT with your exchange's name in uppercase)
 *
 * Configuration:
 * - Creates and uses `config.json` in the same directory.
 * - Edit `config.json` to customize parameters, symbols, alerts, etc.
 *
 * Usage:
 * - `node market_bot_enhanced.js`
 */

// --- Core Dependencies ---
const ccxt = require('ccxt');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const chalk = require('chalk'); // Recommended: chalk@4 for commonjs `require`

// --- Constants for Indicator Parameters ---
const RSI_LENGTH = 14;
const STOCH_K_LENGTH = 14;
const STOCH_D_LENGTH = 3;
const STOCH_SMOOTH_K = 3;
const MACD_FAST_LENGTH = 12;
const MACD_SLOW_LENGTH = 26;
const MACD_SIGNAL_LENGTH = 9;
const BB_LENGTH = 20;
const BB_MULT = 2;
const ICHIMOKU_TENKAN_LEN = 9;
const ICHIMOKU_KIJUN_LEN = 26;
const ICHIMOKU_SENKOU_B_LEN = 52;
const ICHIMOKU_CHIKOU_OFFSET = -26; // Offset backwards
const ICHIMOKU_SENKOU_OFFSET = 26; // Plot forwards
const AO_FAST_LEN = 5;
const AO_SLOW_LEN = 34;
const PSAR_START = 0.02;
const PSAR_INCREMENT = 0.02;
const PSAR_MAX = 0.2;
const VWAP_LENGTH_DEFAULT = 14;
const MIN_MA_LENGTH = 2; // Absolute minimum length for any MA/EMA calculation

// --- Configuration Management ---
const CONFIG_FILE = path.join(__dirname, 'config.json');
const DEFAULT_CONFIG = {
    // --- General Settings ---
    "exchange": "bybit",            // Exchange ID (lowercase) from CCXT (e.g., 'binance', 'bybit', 'kucoin')
    "symbol": "BTC/USDT",           // Trading pair symbol in CCXT format
    "timeframe": "1m",              // Chart timeframe (e.g., '1m', '5m', '1h', '4h', '1d')
    "limit": 150,                   // Number of candles to fetch (ensure > longest indicator lookback, e.g., Ichimoku B: 52)
    "phoneNumber": "",              // Your phone number for SMS alerts (e.g., "+11234567890") - LEAVE EMPTY TO DISABLE SMS
    "testnet": false,               // Use exchange's testnet environment (if supported by CCXT)
    "retryFetchDelaySeconds": 30,   // Delay (seconds) before retrying a failed data fetch
    "loopTargetSeconds": 60,        // Target duration (seconds) for each analysis loop

    // --- Pivot Point Settings ---
    "pivotTimeframe": "D",          // Timeframe for pivot calculation ('D', 'W', 'M') - Currently uses PREVIOUS candle of main 'timeframe'
    "ppMethod": "HLC/3",            // Pivot Point calculation method ("HLC/3" or "HLCO/4")
    "rangeMethod": "ATR",           // Range calculation for R/S levels ("ATR", "High-Low", "Average H-L & ATR")
    "atrLength": 14,                // Lookback period for ATR calculation
    "volMALength": 20,              // Lookback period for Volume Moving Average
    "volInfluence": 0.5,            // Influence of volume on pivot range (0 = none)
    "volFactorMinClamp": 0.3,       // Minimum clamp for the volume factor
    "volFactorMaxClamp": 3.0,       // Maximum clamp for the volume factor
    "fibRatio1": 0.382,             // Fibonacci ratio for R1/S1
    "fibRatio2": 0.618,             // Fibonacci ratio for R2/S2
    "fibRatio3": 1.000,             // Fibonacci ratio for R3/S3
    "showCPR": false,               // Calculate and display Central Pivot Range (BC/TC)

    // --- Dynamic Momentum MA Settings ---
    "showMomentumMAs": true,        // Enable calculation and display of dynamic MAs
    "momentumMaLength": 20,         // Base lookback period for dynamic SMA
    "momentumEmaLength": 10,        // Base lookback period for dynamic EMA
    "momentumRocLength": 14,        // Lookback period for Rate of Change (ROC) used for adaptation
    "momentumSensitivity": 0.3,     // Sensitivity of MA length adjustment to ROC (0=none, 1=max)
    "momentumRocNormRange": 20.0,   // Expected ROC range for normalization (-range to +range)
    "momentumMinLength": 5,         // Minimum allowed length for dynamic MAs
    "momentumMaxLength": 100,       // Maximum allowed length for dynamic MAs

    // --- Fixed MA Settings ---
    "showFixedMAs": false,          // Enable calculation and display of fixed-length MAs
    "fixedMaLength": 50,            // Lookback period for fixed SMA
    "fixedEmaLength": 21,           // Lookback period for fixed EMA

    // --- Ehlers Super Smoother Settings ---
    "showEhlers": true,             // Enable calculation and display of Ehlers Super Smoother
    "ehlersLength": 20,             // Lookback period for Ehlers filter
    "ehlersSrc": "close",           // Source data for Ehlers ('close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4')

    // --- Ichimoku Cloud Settings ---
    "showIchimoku": true,           // Enable calculation and display of Ichimoku Cloud
    // Note: Ichimoku lengths are fixed constants (Tenkan: 9, Kijun: 26, Senkou B: 52)

    // --- Awesome Oscillator (AO) Settings ---
    "showAO": true,                 // Enable calculation and display of Awesome Oscillator
    // Note: AO lengths are fixed constants (Fast: 5, Slow: 34)

    // --- Parabolic SAR (PSAR) Settings ---
    "showPSAR": true,               // Enable calculation and display of Parabolic SAR
    // Note: PSAR parameters are fixed constants (Start: 0.02, Inc: 0.02, Max: 0.2)

    // --- Volume Weighted Average Price (VWAP) Settings ---
    "showVWAP": true,               // Enable calculation and display of rolling VWAP
    "vwapLength": VWAP_LENGTH_DEFAULT, // Lookback period for rolling VWAP calculation

    // --- Visual Settings ---
    "colorBars": true,              // Colorize console price output based on Ehlers trend
    "colorBG": false,                // Colorize console background based on Ehlers slope (can be distracting)

    // --- Alert Settings - General ---
    "alertOnHighMomVol": true,      // Alert on high Momentum Volume

    // --- Alert Settings - Pivots & CPR ---
    "alertOnPPCross": true,         // Alert on price crossing the Pivot Point
    "alertOnR1Cross": true,         // Alert on price crossing Resistance 1
    "alertOnR2Cross": false,        // Alert on price crossing Resistance 2
    "alertOnR3Cross": false,        // Alert on price crossing Resistance 3
    "alertOnS1Cross": true,         // Alert on price crossing Support 1
    "alertOnS2Cross": false,        // Alert on price crossing Support 2
    "alertOnS3Cross": false,        // Alert on price crossing Support 3
    "alertOnCPREnterExit": false,   // Alert when price enters/exits the Central Pivot Range (BC/TC)

    // --- Alert Settings - Ehlers ---
    "alertOnEhlersCross": true,     // Alert on price crossing the Ehlers Super Smoother line
    "alertOnEhlersSlope": false,    // Alert on change in Ehlers slope direction

    // --- Alert Settings - Momentum MAs ---
    "alertOnMomMACross": true,      // Alert on price crossing the dynamic Momentum SMA
    "alertOnMomEMACross": true,     // Alert on price crossing the dynamic Momentum EMA
    "alertOnMomMAvsEMACross": false, // Alert when dynamic Momentum SMA/EMA cross each other

    // --- Alert Settings - RSI / StochRSI ---
    "alertOnStochRsiOverbought": true, // Alert when StochRSI K & D are overbought
    "alertOnStochRsiOversold": true, // Alert when StochRSI K & D are oversold
    "alertOnRsiOverbought": true,   // Alert when RSI is overbought
    "alertOnRsiOversold": true,     // Alert when RSI is oversold

    // --- Alert Settings - MACD ---
    "alertOnMacdBullishCross": true, // Alert on MACD line crossing above Signal line (above zero optionally)
    "alertOnMacdBearishCross": true, // Alert on MACD line crossing below Signal line (below zero optionally)
    "macdCrossThreshold": 0,        // Value MACD/Signal must be above/below for cross alert (0 = crosses anywhere)

    // --- Alert Settings - Bollinger Bands ---
    "alertOnBBBreakoutUpper": true, // Alert when price breaks significantly above the Upper BB
    "alertOnBBBreakoutLower": true, // Alert when price breaks significantly below the Lower BB

    // --- Alert Settings - Ichimoku ---
    "alertOnPriceVsKijun": true,    // Alert on price crossing the Kijun-sen (Base Line)
    "alertOnPriceVsKumo": true,     // Alert on price entering/exiting the Kumo (Cloud)
    "alertOnTKCross": true,         // Alert on Tenkan-sen / Kijun-sen cross
    "alertOnChikouPriceCross": false, // Alert on Chikou Span crossing price (requires careful interpretation)

    // --- Alert Settings - AO ---
    "alertOnAOCrossZero": true,     // Alert on Awesome Oscillator crossing the zero line

    // --- Alert Settings - PSAR ---
    "alertOnPSARFlip": true,        // Alert when Parabolic SAR flips direction

    // --- Alert Settings - VWAP ---
    "alertOnPriceVsVWAP": false,    // Alert on price crossing the rolling VWAP (can be noisy)

    // --- Alert Thresholds ---
    "highMomVolThreshold": 1500,          // Threshold value for the 'High Momentum Volume' alert
    "stochRsiOverboughtThreshold": 80,    // StochRSI level considered overbought
    "stochRsiOversoldThreshold": 20,      // StochRSI level considered oversold
    "rsiOverboughtThreshold": 70,         // RSI level considered overbought
    "rsiOversoldThreshold": 30,           // RSI level considered oversold
    "bbBreakoutThresholdMultiplier": 1.005 // Price must be > UpperBB * mult (or < LowerBB / mult) for breakout alert (e.g., 1.005 = 0.5% breakout)
};

/**
 * Validates the loaded configuration object.
 * @param {object} cfg The configuration object to validate.
 */
function validateConfig(cfg) {
    const errors = [];
    const checkInt = (key, min = -Infinity, max = Infinity) => { if (typeof cfg[key] !== 'number' || !Number.isInteger(cfg[key]) || cfg[key] < min || cfg[key] > max) errors.push(`'${key}' must be an integer between ${min}-${max}. Found: ${cfg[key]}`); };
    const checkFloat = (key, min = -Infinity, max = Infinity) => { if (typeof cfg[key] !== 'number' || cfg[key] < min || cfg[key] > max) errors.push(`'${key}' must be a number between ${min}-${max}. Found: ${cfg[key]}`); };
    const checkBool = (key) => { if (typeof cfg[key] !== 'boolean') errors.push(`'${key}' must be true or false. Found: ${cfg[key]}`); };
    const checkString = (key, allowEmpty = false, allowed = null) => {
        if (typeof cfg[key] !== 'string') errors.push(`'${key}' must be a string. Found: ${cfg[key]}`);
        else if (!allowEmpty && cfg[key].trim() === '') errors.push(`'${key}' cannot be empty.`);
        else if (allowed && !allowed.includes(cfg[key])) errors.push(`'${key}' must be one of [${allowed.join(', ')}]. Found: ${cfg[key]}`);
    };

    // Validate key settings
    checkString('exchange', false, ccxt.exchanges); // Check against actual CCXT exchanges
    checkString('symbol');
    checkString('timeframe'); // Could add validation against exchange.timeframes if needed
    checkInt('limit', ICHIMOKU_SENKOU_B_LEN, 1000); // Ensure limit covers longest lookback
    checkString('phoneNumber', true); // Allow empty string to disable SMS
    checkBool('testnet');
    checkInt('retryFetchDelaySeconds', 5);
    checkInt('loopTargetSeconds', 10);

    // Pivots
    checkString('pivotTimeframe', false, ['D', 'W', 'M']);
    checkString('ppMethod', false, ['HLC/3', 'HLCO/4']);
    checkString('rangeMethod', false, ['ATR', 'High-Low', 'Average H-L & ATR']);
    checkInt('atrLength', 1);
    checkInt('volMALength', 1);
    checkFloat('volInfluence', 0);
    checkFloat('volFactorMinClamp', 0);
    checkFloat('volFactorMaxClamp', cfg.volFactorMinClamp); // Max >= Min
    checkFloat('fibRatio1', 0); checkFloat('fibRatio2', cfg.fibRatio1); checkFloat('fibRatio3', cfg.fibRatio2); // Ratios should increase
    checkBool('showCPR');

    // Momentum MAs
    checkBool('showMomentumMAs');
    checkInt('momentumMaLength', MIN_MA_LENGTH);
    checkInt('momentumEmaLength', MIN_MA_LENGTH);
    checkInt('momentumRocLength', 1);
    checkFloat('momentumSensitivity', 0, 1);
    checkFloat('momentumRocNormRange', 1);
    checkInt('momentumMinLength', MIN_MA_LENGTH);
    checkInt('momentumMaxLength', cfg.momentumMinLength); // Max >= Min

    // Fixed MAs
    checkBool('showFixedMAs');
    checkInt('fixedMaLength', MIN_MA_LENGTH);
    checkInt('fixedEmaLength', MIN_MA_LENGTH);

    // Ehlers
    checkBool('showEhlers');
    checkInt('ehlersLength', 3); // Ehlers requires min 3
    checkString('ehlersSrc', false, ['close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4']);

    // Other Indicators
    checkBool('showIchimoku'); checkBool('showAO'); checkBool('showPSAR'); checkBool('showVWAP');
    checkInt('vwapLength', 1);

    // Visuals
    checkBool('colorBars'); checkBool('colorBG');

    // Alerts (check all boolean flags)
    Object.keys(DEFAULT_CONFIG).forEach(key => { if(key.startsWith('alertOn')) checkBool(key); });

    // Thresholds
    checkFloat('highMomVolThreshold', 0);
    checkInt('stochRsiOverboughtThreshold', 50, 100);
    checkInt('stochRsiOversoldThreshold', 0, 50);
    if (cfg.stochRsiOversoldThreshold >= cfg.stochRsiOverboughtThreshold) errors.push('stochRsiOversoldThreshold must be less than stochRsiOverboughtThreshold');
    checkInt('rsiOverboughtThreshold', 50, 100);
    checkInt('rsiOversoldThreshold', 0, 50);
    if (cfg.rsiOversoldThreshold >= cfg.rsiOverboughtThreshold) errors.push('rsiOversoldThreshold must be less than rsiOverboughtThreshold');
    checkFloat('bbBreakoutThresholdMultiplier', 1.0); // Must be >= 1.0
    checkFloat('macdCrossThreshold'); // Can be positive or negative

    if (errors.length > 0) {
        console.error(chalk.red.bold('\n# --- Invalid Configuration --- #'));
        errors.forEach(err => console.error(chalk.red(`- ${err}`)));
        console.error(chalk.red.bold('# --- Please correct config.json and restart --- #\n'));
        process.exit(1); // Exit if config is invalid
    }
    // console.log(chalk.green('# Configuration validated successfully.'));
}


/**
 * Loads configuration from config.json, merging with defaults.
 * Creates config.json with defaults if it doesn't exist.
 * Validates the final configuration.
 * @returns {object} The loaded and validated configuration object.
 */
function loadConfig() {
    let configToLoad = { ...DEFAULT_CONFIG }; // Start with defaults
    try {
        if (fs.existsSync(CONFIG_FILE)) {
            const rawData = fs.readFileSync(CONFIG_FILE);
            const loadedConfig = JSON.parse(rawData);
            // Merge loaded config over defaults
            configToLoad = { ...DEFAULT_CONFIG, ...loadedConfig };
            console.log(chalk.blue(`# Configuration loaded from ${CONFIG_FILE}`));
        } else {
            console.log(chalk.yellow(`# Configuration file not found. Creating ${CONFIG_FILE} with defaults.`));
            saveConfig(configToLoad); // Save default config on first run
        }
    } catch (error) {
        console.error(chalk.red(`# Error loading or parsing ${CONFIG_FILE}:`), error);
        console.log(chalk.yellow("# Loading default configuration as fallback."));
        configToLoad = { ...DEFAULT_CONFIG }; // Ensure defaults on error
    }

    // Set defaults for fixed MAs if not provided in the loaded file
    configToLoad.fixedMaLength = configToLoad.fixedMaLength ?? DEFAULT_CONFIG.fixedMaLength;
    configToLoad.fixedEmaLength = configToLoad.fixedEmaLength ?? DEFAULT_CONFIG.fixedEmaLength;

    validateConfig(configToLoad); // Validate after loading/merging/defaulting
    return configToLoad;
}

/**
 * Saves the configuration object to config.json.
 * @param {object} configToSave The configuration object to save.
 */
function saveConfig(configToSave) {
    try {
        fs.writeFileSync(CONFIG_FILE, JSON.stringify(configToSave, null, 4)); // Pretty print JSON
        console.log(chalk.blue(`# Configuration saved to ${CONFIG_FILE}`));
    } catch (error) {
        console.error(chalk.red(`# Error saving configuration to ${CONFIG_FILE}:`), error);
    }
}

// --- Global Configuration Variable ---
const config = loadConfig(); // Load and validate configuration at the start

// --- Utility Functions ---
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const sum = (arr) => arr.reduce((acc, val) => acc + (Number.isFinite(val) ? val : 0), 0);
const avg = (arr) => { const v = arr.filter(Number.isFinite); return v.length > 0 ? sum(v) / v.length : NaN; };
const nz = (value, replacement = 0) => (value === null || value === undefined || !Number.isFinite(value)) ? replacement : value;
const highest = (arr, len) => {
    if (len <= 0 || !Array.isArray(arr)) return NaN;
    const slice = arr.slice(-len); // Get the last 'len' elements
    const finiteSlice = slice.filter(Number.isFinite);
    return finiteSlice.length > 0 ? Math.max(...finiteSlice) : NaN;
};
const lowest = (arr, len) => {
    if (len <= 0 || !Array.isArray(arr)) return NaN;
    const slice = arr.slice(-len);
    const finiteSlice = slice.filter(Number.isFinite);
    return finiteSlice.length > 0 ? Math.min(...finiteSlice) : NaN;
};

/**
 * Safely gets the last element of an array, ensuring it's a finite number.
 * @param {Array<number>} arr The array.
 * @returns {number|NaN} The last finite number or NaN.
 */
const getSafeLast = (arr) => {
    if (!Array.isArray(arr) || arr.length === 0) return NaN;
    const lastVal = arr[arr.length - 1];
    return Number.isFinite(lastVal) ? lastVal : NaN;
};

/**
 * Safely gets the second to last element of an array, ensuring it's a finite number.
 * @param {Array<number>} arr The array.
 * @returns {number|NaN} The second-to-last finite number or NaN.
 */
const getSafePrev = (arr) => {
    if (!Array.isArray(arr) || arr.length < 2) return NaN;
    const prevVal = arr[arr.length - 2];
    return Number.isFinite(prevVal) ? prevVal : NaN;
};

/**
 * Gets the appropriate source data array based on the configuration string.
 * @param {string} configSrc Configured source ('close', 'hl2', etc.)
 * @param {object} data Object containing open, high, low, close arrays.
 * @returns {Array<number>} The selected source data array.
 */
function getSourceData(configSrc, data) {
    const len = data.close ? data.close.length : 0;
    if (len === 0) return []; // Return empty if no data

    switch(configSrc?.toLowerCase()) {
        case 'open': return data.open || new Array(len).fill(NaN);
        case 'high': return data.high || new Array(len).fill(NaN);
        case 'low': return data.low || new Array(len).fill(NaN);
        case 'hl2': return (data.high && data.low) ? data.high.map((h, i) => Number.isFinite(h) && Number.isFinite(data.low[i]) ? (h + data.low[i]) / 2 : NaN) : new Array(len).fill(NaN);
        case 'hlc3': return (data.high && data.low && data.close) ? data.high.map((h, i) => Number.isFinite(h) && Number.isFinite(data.low[i]) && Number.isFinite(data.close[i]) ? (h + data.low[i] + data.close[i]) / 3 : NaN) : new Array(len).fill(NaN);
        case 'ohlc4': return (data.open && data.high && data.low && data.close) ? data.open.map((o, i) => Number.isFinite(o) && Number.isFinite(data.high[i]) && Number.isFinite(data.low[i]) && Number.isFinite(data.close[i]) ? (o + data.high[i] + data.low[i] + data.close[i]) / 4 : NaN) : new Array(len).fill(NaN);
        case 'close':
        default: return data.close || new Array(len).fill(NaN);
    }
}


// --- Indicator Calculation Functions ---

/** Calculates Simple Moving Average (SMA) */
function calculateSma(src, length) {
    length = Math.max(MIN_MA_LENGTH, Math.round(length));
    if (!src || src.length < length) return new Array(src ? src.length : 0).fill(NaN);

    const smaValues = new Array(src.length).fill(NaN);
    let currentSum = 0;
    let validCount = 0;

    // Initial window calculation
    for (let i = 0; i < length; i++) {
        if (Number.isFinite(src[i])) {
            currentSum += src[i];
            validCount++;
        }
    }
    if (validCount === length) {
        smaValues[length - 1] = currentSum / length;
    }

    // Rolling window calculation
    for (let i = length; i < src.length; i++) {
        const enteringVal = src[i];
        const exitingVal = src[i - length];
        let sumChanged = false;

        if (Number.isFinite(exitingVal)) {
            currentSum -= exitingVal;
            validCount--;
            sumChanged = true;
        }
        if (Number.isFinite(enteringVal)) {
            currentSum += enteringVal;
            validCount++;
            sumChanged = true;
        }

        if (validCount === length) {
            smaValues[i] = currentSum / length;
        } else {
            smaValues[i] = NaN;
             // If the window became invalid due to NaNs entering/exiting, reset sum to avoid carrying invalid state
            if (sumChanged) {
                currentSum = NaN; // Mark sum as invalid
            }
        }
        // If sum became NaN, recalculate for the current window to potentially recover
        if (!Number.isFinite(currentSum) && i >= length -1) {
             const slice = src.slice(i - length + 1, i + 1);
             const currentFinite = slice.filter(Number.isFinite);
             validCount = currentFinite.length;
             if(validCount === length) {
                 currentSum = sum(currentFinite);
                 smaValues[i] = currentSum / length;
             }
        }

    }
    return smaValues;
}

/** Calculates Exponential Moving Average (EMA) */
function calculateEma(src, length) {
    length = Math.max(MIN_MA_LENGTH, Math.round(length));
    if (!src || src.length < length) return new Array(src ? src.length : 0).fill(NaN);

    const alpha = 2 / (length + 1);
    const emaValues = new Array(src.length).fill(NaN);
    let prevEma = NaN;

    // Find the first valid SMA to seed the EMA
    const initialSma = calculateSma(src, length);
    let seedIndex = -1;
    for (let i = length - 1; i < src.length; i++) {
        if (Number.isFinite(initialSma[i])) {
            prevEma = initialSma[i];
            emaValues[i] = prevEma;
            seedIndex = i;
            break;
        }
    }

    // If no valid seed could be found, return NaNs
    if (seedIndex === -1) return emaValues;

    // Calculate subsequent EMAs
    for (let i = seedIndex + 1; i < src.length; i++) {
        const currentSrc = src[i];
        if (Number.isFinite(currentSrc)) {
            prevEma = (currentSrc - prevEma) * alpha + prevEma;
            emaValues[i] = prevEma;
        } else {
            // Carry forward the previous EMA if current source is NaN
            emaValues[i] = prevEma;
        }
    }
    return emaValues;
}

/** Calculates Average True Range (ATR) */
function calculateAtr(high, low, close, length) {
    length = Math.max(1, Math.round(length)); // ATR length can be 1
    const n = close ? close.length : 0;
    if (!high || !low || !close || high.length !== low.length || high.length !== close.length || n < 1) {
        return new Array(n).fill(NaN);
    }
    const trValues = new Array(n);
    for (let i = 0; i < n; i++) {
        const h_i = high[i], l_i = low[i];
        const pc = i > 0 ? close[i - 1] : NaN; // Previous close

        if (!Number.isFinite(h_i) || !Number.isFinite(l_i)) {
            trValues[i] = NaN;
            continue;
        }
        const highLow = h_i - l_i;
        const highPrevClose = Number.isFinite(pc) ? Math.abs(h_i - pc) : NaN;
        const lowPrevClose = Number.isFinite(pc) ? Math.abs(l_i - pc) : NaN;

        if (i === 0) {
            trValues[i] = highLow; // First TR is just High - Low
        } else {
            trValues[i] = Math.max(highLow, highPrevClose, lowPrevClose);
        }
         // Ensure TR is finite
        if (!Number.isFinite(trValues[i])) trValues[i] = NaN;

    }
    // Use Wilder's smoothing (RMA), equivalent to EMA with alpha = 1/length
    // return calculateWilderEma(trValues, length); // If you prefer Wilder's
    return calculateEma(trValues, length); // Using standard EMA here
}

/*
// Optional: Wilder's EMA / RMA for ATR if preferred over standard EMA
function calculateWilderEma(src, length) {
    length = Math.max(1, Math.round(length));
    if (!src || src.length < length) return new Array(src ? src.length : 0).fill(NaN);
    const alpha = 1 / length;
    const emaValues = new Array(src.length).fill(NaN);
    let prevEma = NaN;
    let sumForInit = 0;
    let validCount = 0;
    // Initial SMA-like calculation for the first value
    for(let i = 0; i < length; i++) {
        if(Number.isFinite(src[i])) {
            sumForInit += src[i];
            validCount++;
        }
    }
    if (validCount === length) { // Only initialize if full data
        prevEma = sumForInit / length;
        emaValues[length - 1] = prevEma;
    }

    // Subsequent Wilder smoothing
    for (let i = length; i < src.length; i++) {
        const currentVal = src[i];
        if (!Number.isFinite(prevEma)) { // If previous was NaN, cannot continue Wilder's
             emaValues[i] = NaN;
             // Try to re-seed? Difficult with Wilder's state. NaN is safest.
        } else if (Number.isFinite(currentVal)) {
            prevEma = (currentVal - prevEma) * alpha + prevEma; // Wilder's formula
            emaValues[i] = prevEma;
        } else {
            emaValues[i] = prevEma; // Carry forward if current is NaN
        }
    }
    return emaValues;
}
*/

/** Calculates Rate of Change (ROC) */
function calculateRoc(src, length) {
    length = Math.max(1, Math.round(length));
    if (!src || src.length < length) return new Array(src ? src.length : 0).fill(NaN);
    return src.map((val, i) => {
        if (i >= length) {
            const prevVal = src[i - length];
            if (Number.isFinite(val) && Number.isFinite(prevVal) && prevVal !== 0) {
                return ((val - prevVal) / prevVal) * 100;
            }
        }
        return NaN;
    });
}

/** Calculates Ehlers Super Smoother Filter */
function calculateEhlersSmoother(src, length) {
    length = Math.max(3, Math.round(length)); // Ehlers needs min length 3
    if (!src || src.length < 3) return new Array(src ? src.length : 0).fill(NaN);

    const a1_param = Math.sqrt(2.0) * Math.PI / length;
    const b1_param = Math.sqrt(2.0) * Math.PI / length; // Original uses 1.414, sqrt(2) is more precise

    const a1 = Math.exp(-a1_param);
    const b1 = 2.0 * a1 * Math.cos(b1_param);
    const c2 = b1;
    const c3 = -a1 * a1;
    const c1 = 1.0 - c2 - c3;
    const ehlersValues = new Array(src.length).fill(NaN);

    // Initialization requires care
    let filt_1 = NaN; // Represents filt[i-1]
    let filt_2 = NaN; // Represents filt[i-2]

    for (let i = 0; i < src.length; i++) {
        const currentSrc = src[i];
        const prevSrc = i > 0 ? src[i-1] : NaN; // Previous source value

        if (!Number.isFinite(currentSrc)) {
             // If current source is NaN, carry forward the filter value
             ehlersValues[i] = filt_1; // Carry forward the last calculated value
             // Update history, but keep it as the carried-forward value
             filt_2 = filt_1;
             // filt_1 remains unchanged (as the carried-forward value)
             continue;
        }

        const srcAvg = (currentSrc + nz(prevSrc, currentSrc)) / 2.0; // Average current and previous (or just current if prev is NaN)

        if (i < 2 || !Number.isFinite(filt_1) || !Number.isFinite(filt_2)) {
            // Initial values or if history is invalid, use the source average
            ehlersValues[i] = srcAvg;
        } else {
            // Apply the filter formula
            ehlersValues[i] = c1 * srcAvg + c2 * filt_1 + c3 * filt_2;
        }

        // Update history for the next iteration
        filt_2 = filt_1;
        filt_1 = ehlersValues[i]; // The newly calculated or assigned value
    }
    return ehlersValues;
}

/** Calculates Pivot Points and Support/Resistance Levels */
function calculatePivots(pdH, pdL, pdC, pdO, pdATR, pdVol, pdVolMA, ppMeth, rangeMeth, volInfl, volMinC, volMaxC, fib1, fib2, fib3) {
    // Requires Previous Day High, Low, Close, and ATR
    if (!Number.isFinite(pdH) || !Number.isFinite(pdL) || !Number.isFinite(pdC) || !Number.isFinite(pdATR)) {
        return { pp: NaN, r1: NaN, r2: NaN, r3: NaN, s1: NaN, s2: NaN, s3: NaN, tc: NaN, bc: NaN };
    }

    const useO = ppMeth === "HLCO/4";
    const _pp = useO
        ? (pdH + pdL + pdC + nz(pdO, pdC)) / 4.0 // Use Close if Open is NaN
        : (pdH + pdL + pdC) / 3.0;

    const _pdRangeHL = pdH - pdL;
    let _baseRange;
    switch (rangeMeth) {
        case "High-Low": _baseRange = _pdRangeHL; break;
        case "Average H-L & ATR": _baseRange = (_pdRangeHL + pdATR) / 2.0; break;
        default: _baseRange = pdATR; // Default to ATR
    }
    _baseRange = Math.max(0, _baseRange); // Ensure range is non-negative

    let _volumeFactor = 1.0;
    const isValidVolMA = Number.isFinite(pdVolMA) && pdVolMA > 0;
    if (isValidVolMA && volInfl > 0) {
        const _volumeRatio = nz(pdVol, 0) / pdVolMA; // Use 0 if pdVol is NaN
        _volumeFactor = 1.0 + volInfl * (_volumeRatio - 1.0);
        _volumeFactor = Math.max(volMinC, Math.min(volMaxC, _volumeFactor)); // Clamp factor
    }

    const _dynamicRange = _baseRange * _volumeFactor;
    const _bc = (pdH + pdL) / 2.0; // Central Pivot Range Bottom
    const _tc = (_pp - _bc) + _pp; // Central Pivot Range Top

    return {
        pp: _pp,
        r1: _pp + fib1 * _dynamicRange, r2: _pp + fib2 * _dynamicRange, r3: _pp + fib3 * _dynamicRange,
        s1: _pp - fib1 * _dynamicRange, s2: _pp - fib2 * _dynamicRange, s3: _pp - fib3 * _dynamicRange,
        tc: _tc, bc: _bc
    };
}

/** Calculates Momentum Volume */
function calculateMomentumVolume(close, volume, length) {
    length = Math.max(1, Math.round(length));
    const n = close ? close.length : 0;
    if (!close || !volume || close.length !== volume.length || n < length) {
        return new Array(n).fill(NaN);
    }
    const volSma = calculateSma(volume, config.volMALength); // Use volMALength from config for consistency? Or use 'length'? Using 'length' for now.
    // const volSma = calculateSma(volume, length);

    return close.map((c, i) => {
        // Need 'length' bars lookback for ROC calculation
        if (i >= length) {
            const prevClose = close[i - length]; // Close from 'length' bars ago
            const vol_i = volume[i];
            const sma_i = volSma[i];
            if (Number.isFinite(c) && Number.isFinite(prevClose) && Number.isFinite(vol_i) && Number.isFinite(sma_i) && sma_i !== 0) {
                // (Price Change) * (Volume / Avg Volume)
                 return (c - prevClose) * (vol_i / sma_i);
            }
        }
        return NaN;
    });
}

/** Calculates Relative Strength Index (RSI) using Wilder's Smoothing */
function calculateRsi(src, length) {
    length = Math.max(1, Math.round(length));
    if (!src || src.length <= length) return new Array(src ? src.length : 0).fill(NaN);

    const rsiValues = new Array(src.length).fill(NaN);
    let avgGain = NaN;
    let avgLoss = NaN;

    // Calculate initial average gain/loss
    let initialGains = 0;
    let initialLosses = 0;
    let validChangesCount = 0;
    for (let i = 1; i <= length; i++) {
        const change = src[i] - src[i - 1];
        if (Number.isFinite(change)) {
            initialGains += Math.max(0, change);
            initialLosses += Math.max(0, -change);
            validChangesCount++;
        }
    }

    // Only initialize if we have a full window of valid changes
    if (validChangesCount === length) {
        avgGain = initialGains / length;
        avgLoss = initialLosses / length;

        if (avgLoss === 0) {
            rsiValues[length] = 100;
        } else {
            const rs = avgGain / avgLoss;
            rsiValues[length] = 100 - (100 / (1 + rs));
        }
    } // else: avgGain/avgLoss remain NaN, first RSI is NaN

    // Calculate subsequent RSI using Wilder's smoothing
    for (let i = length + 1; i < src.length; i++) {
        const change = src[i] - src[i - 1];

        // If change is NaN, or previous averages are NaN, carry forward NaN for RSI
        if (!Number.isFinite(change) || !Number.isFinite(avgGain) || !Number.isFinite(avgLoss)) {
            rsiValues[i] = NaN;
             // Reset averages if they were valid but change is now NaN? Maybe safer.
             if(Number.isFinite(avgGain)) avgGain = NaN;
             if(Number.isFinite(avgLoss)) avgLoss = NaN;
            continue;
        }

        const gain = Math.max(0, change);
        const loss = Math.max(0, -change);

        avgGain = (avgGain * (length - 1) + gain) / length;
        avgLoss = (avgLoss * (length - 1) + loss) / length;

        if (avgLoss === 0) {
            rsiValues[i] = 100;
        } else {
            const rs = avgGain / avgLoss;
            rsiValues[i] = 100 - (100 / (1 + rs));
        }
    }
    return rsiValues;
}

/** Calculates Stochastic RSI */
function calculateStochRsi(rsiValues, kLength, dLength, smoothK) {
    kLength = Math.max(1, Math.round(kLength));
    dLength = Math.max(1, Math.round(dLength));
    smoothK = Math.max(1, Math.round(smoothK));
    const n = rsiValues ? rsiValues.length : 0;
    if (!rsiValues || n < kLength) return { k: new Array(n).fill(NaN), d: new Array(n).fill(NaN) };

    const stochKRaw = new Array(n).fill(NaN);
    for (let i = kLength - 1; i < n; i++) {
        const rsiSlice = rsiValues.slice(i - kLength + 1, i + 1);
        const validRsiSlice = rsiSlice.filter(Number.isFinite);

        if (validRsiSlice.length === 0) {
            stochKRaw[i] = NaN; // Not enough data in the window
            continue;
        }

        const minRsi = Math.min(...validRsiSlice);
        const maxRsi = Math.max(...validRsiSlice);
        const currentRsi = rsiValues[i];

        if (!Number.isFinite(currentRsi)) {
             stochKRaw[i] = NaN; // Use NaN if current RSI is NaN
        } else if (maxRsi === minRsi) {
            // If range is zero, find the midpoint or carry forward last value?
            // Let's try carrying forward the last valid StochKRaw
            let lastValidStoch = NaN;
             for(let j=i-1; j >= kLength -1; j--) {
                if(Number.isFinite(stochKRaw[j])) {
                    lastValidStoch = stochKRaw[j];
                    break;
                }
             }
            stochKRaw[i] = Number.isFinite(lastValidStoch) ? lastValidStoch : 50; // Default to 50 if no prior valid
        } else {
            stochKRaw[i] = ((currentRsi - minRsi) / (maxRsi - minRsi)) * 100;
        }
    }

    const smoothedK = calculateSma(stochKRaw, smoothK); // %K line (smoothed raw stoch)
    const stochD = calculateSma(smoothedK, dLength);   // %D line (SMA of %K)

    return { k: smoothedK, d: stochD };
}

/** Calculates MACD (Moving Average Convergence Divergence) */
function calculateMacd(src, fastLength, slowLength, signalLength) {
    fastLength = Math.max(MIN_MA_LENGTH, Math.round(fastLength));
    slowLength = Math.max(fastLength + 1, Math.round(slowLength)); // Slow > Fast
    signalLength = Math.max(1, Math.round(signalLength));
    if (!src || src.length < slowLength) return { macd: [], signal: [], histogram: [] };

    const fastEma = calculateEma(src, fastLength);
    const slowEma = calculateEma(src, slowLength);

    const macdLine = fastEma.map((fast, i) => Number.isFinite(fast) && Number.isFinite(slowEma[i]) ? fast - slowEma[i] : NaN);
    const signalLine = calculateEma(macdLine, signalLength);
    const histogram = macdLine.map((macd, i) => Number.isFinite(macd) && Number.isFinite(signalLine[i]) ? macd - signalLine[i] : NaN);

    return { macd: macdLine, signal: signalLine, histogram: histogram };
}

/** Calculates Bollinger Bands */
function calculateBollingerBands(src, length, mult) {
    length = Math.max(MIN_MA_LENGTH, Math.round(length));
    mult = Math.max(0.1, mult);
    if (!src || src.length < length) return { upper: [], lower: [], middle: [] };

    const middleBand = calculateSma(src, length); // Middle Band is SMA
    const upperBand = new Array(src.length).fill(NaN);
    const lowerBand = new Array(src.length).fill(NaN);

    for (let i = length - 1; i < src.length; i++) {
        if (Number.isFinite(middleBand[i])) {
            const slice = src.slice(i - length + 1, i + 1);
            const finiteSlice = slice.filter(Number.isFinite);

            // Need at least MIN_MA_LENGTH points to calculate a meaningful standard deviation
            if (finiteSlice.length >= MIN_MA_LENGTH) {
                 // Use population standard deviation (divide by N) which is common in trading platforms for BBands
                const variance = sum(finiteSlice.map(x => Math.pow(x - middleBand[i], 2))) / length; // Divide by 'length' (N)
                const stdDev = Math.sqrt(variance);
                upperBand[i] = middleBand[i] + mult * stdDev;
                lowerBand[i] = middleBand[i] - mult * stdDev;
            } // else: Bands remain NaN if not enough finite points
        }
    }
    return { upper: upperBand, lower: lowerBand, middle: middleBand };
}

/** Calculates Ichimoku Cloud components */
function calculateIchimoku(high, low, close) {
    const n = close ? close.length : 0;
    // Need enough data for the longest lookback component (Senkou Span B)
    if (!high || !low || n < ICHIMOKU_SENKOU_B_LEN) {
        console.warn(chalk.yellow(`Ichimoku calculation requires at least ${ICHIMOKU_SENKOU_B_LEN} candles, found ${n}. Skipping.`));
        return { tenkan: [], kijun: [], senkouA: [], senkouB: [], chikou: [] };
    }

    const tenkanSen = new Array(n).fill(NaN);
    const kijunSen = new Array(n).fill(NaN);
    const senkouSpanA_raw = new Array(n).fill(NaN); // Calculated at current time
    const senkouSpanB_raw = new Array(n).fill(NaN); // Calculated at current time
    const chikouSpan_raw = new Array(n).fill(NaN); // Calculated relative to current time

    for (let i = 0; i < n; i++) {
        // Tenkan-sen (Conversion Line)
        if (i >= ICHIMOKU_TENKAN_LEN - 1) {
            const h = highest(high.slice(0, i + 1), ICHIMOKU_TENKAN_LEN); // Pass sliced array for correct indexing with highest/lowest
            const l = lowest(low.slice(0, i + 1), ICHIMOKU_TENKAN_LEN);
            if (Number.isFinite(h) && Number.isFinite(l)) tenkanSen[i] = (h + l) / 2;
        }
        // Kijun-sen (Base Line)
        if (i >= ICHIMOKU_KIJUN_LEN - 1) {
            const h = highest(high.slice(0, i + 1), ICHIMOKU_KIJUN_LEN);
            const l = lowest(low.slice(0, i + 1), ICHIMOKU_KIJUN_LEN);
            if (Number.isFinite(h) && Number.isFinite(l)) kijunSen[i] = (h + l) / 2;
        }
        // Senkou Span A (raw calculation)
        if (Number.isFinite(tenkanSen[i]) && Number.isFinite(kijunSen[i])) {
             senkouSpanA_raw[i] = (tenkanSen[i] + kijunSen[i]) / 2;
        }
        // Senkou Span B (raw calculation)
        if (i >= ICHIMOKU_SENKOU_B_LEN - 1) {
            const h = highest(high.slice(0, i + 1), ICHIMOKU_SENKOU_B_LEN);
            const l = lowest(low.slice(0, i + 1), ICHIMOKU_SENKOU_B_LEN);
             if (Number.isFinite(h) && Number.isFinite(l)) senkouSpanB_raw[i] = (h + l) / 2;
        }
        // Chikou Span (Lagging Span) - Store close price at current index
        chikouSpan_raw[i] = close[i];
    }

    // Apply offsets for plotting
    const senkouSpanA_final = new Array(n).fill(NaN);
    const senkouSpanB_final = new Array(n).fill(NaN);
    const chikouSpan_final = new Array(n).fill(NaN);

    for (let i = 0; i < n; i++) {
        // Plot Senkou A forward
        if (i + ICHIMOKU_SENKOU_OFFSET < n && Number.isFinite(senkouSpanA_raw[i])) {
            senkouSpanA_final[i + ICHIMOKU_SENKOU_OFFSET] = senkouSpanA_raw[i];
        }
        // Plot Senkou B forward
        if (i + ICHIMOKU_SENKOU_OFFSET < n && Number.isFinite(senkouSpanB_raw[i])) {
            senkouSpanB_final[i + ICHIMOKU_SENKOU_OFFSET] = senkouSpanB_raw[i];
        }
         // Plot Chikou backward (value from `i` is plotted at `i + offset`)
        const plotIndex = i + ICHIMOKU_CHIKOU_OFFSET;
         if (plotIndex >= 0 && plotIndex < n && Number.isFinite(chikouSpan_raw[i])) {
             chikouSpan_final[plotIndex] = chikouSpan_raw[i];
         }
    }

    return {
        tenkan: tenkanSen,           // Conversion Line
        kijun: kijunSen,             // Base Line
        senkouA: senkouSpanA_final,  // Leading Span A (plotted ahead)
        senkouB: senkouSpanB_final,  // Leading Span B (plotted ahead)
        chikou: chikouSpan_final     // Lagging Span (plotted behind)
    };
}

/** Calculates Awesome Oscillator (AO) */
function calculateAO(high, low) {
    const n = high ? high.length : 0;
    if (!high || !low || n < AO_SLOW_LEN) {
         console.warn(chalk.yellow(`AO calculation requires at least ${AO_SLOW_LEN} candles, found ${n}. Skipping.`));
         return new Array(n).fill(NaN);
    }

    const midPoint = high.map((h, i) => Number.isFinite(h) && Number.isFinite(low[i]) ? (h + low[i]) / 2 : NaN);
    const smaFast = calculateSma(midPoint, AO_FAST_LEN);
    const smaSlow = calculateSma(midPoint, AO_SLOW_LEN);

    return smaFast.map((fast, i) => Number.isFinite(fast) && Number.isFinite(smaSlow[i]) ? fast - smaSlow[i] : NaN);
}

/** Calculates Parabolic SAR (PSAR) */
function calculatePSAR(high, low, close) {
    const n = high ? high.length : 0;
    if (!high || !low || !close || n < 2) { // Need at least 2 bars
         return new Array(n).fill(NaN);
    }

    const psarValues = new Array(n).fill(NaN);
    let trend = 0; // 1 for uptrend, -1 for downtrend, 0 initial
    let ep = NaN; // Extreme point
    let sar = NaN; // Stop And Reverse value
    let af = PSAR_START; // Acceleration Factor

    // Initialization (using the second bar to determine initial trend)
    if (Number.isFinite(close[1]) && Number.isFinite(close[0])) {
        if (close[1] > close[0]) { // Initial uptrend guess
            trend = 1;
            ep = high[1];
            sar = low[1]; // Start SAR below price for uptrend
        } else { // Initial downtrend guess
            trend = -1;
            ep = low[1];
            sar = high[1]; // Start SAR above price for downtrend
        }
        psarValues[1] = sar; // Set SAR for the second bar
    } else {
        // Cannot determine initial trend, cannot calculate PSAR
        return psarValues;
    }

    // Calculate PSAR from the third bar onwards
    for (let i = 2; i < n; i++) {
        if (!Number.isFinite(high[i]) || !Number.isFinite(low[i]) || !Number.isFinite(high[i-1]) || !Number.isFinite(low[i-1]) || !Number.isFinite(high[i-2]) || !Number.isFinite(low[i-2]) ) {
            // If essential data is missing, cannot calculate, carry forward? No, SAR depends on H/L. Mark NaN.
             psarValues[i] = NaN;
             // Reset state? Might be needed if NaNs persist.
             trend = 0; ep = NaN; sar = NaN; af = PSAR_START;
             continue;
        }
         if(trend === 0) { // If state was reset due to NaN
             psarValues[i] = NaN;
             // Try re-initializing based on current bar vs previous
             if(Number.isFinite(close[i]) && Number.isFinite(close[i-1])) {
                 if (close[i] > close[i-1]) { trend = 1; ep = high[i]; sar = low[i];}
                 else { trend = -1; ep = low[i]; sar = high[i]; }
                 af = PSAR_START;
                 psarValues[i] = sar; // Set current SAR
             }
             continue; // Move to next bar after re-initialization attempt
         }


        const prevSar = sar;
        const prevTrend = trend;
        const prevAf = af;
        const prevEp = ep;
        let currentSar = NaN;

        // Calculate potential SAR for the current period *before* checking for reversal
        if (prevTrend === 1) { // If previous trend was UP
            currentSar = prevSar + prevAf * (prevEp - prevSar);
            // SAR cannot be higher than the lowest of the previous two periods' lows
            currentSar = Math.min(currentSar, low[i-1], low[i-2]);
        } else { // If previous trend was DOWN
            currentSar = prevSar - prevAf * (prevEp - prevSar);
            // SAR cannot be lower than the highest of the previous two periods' highs
            currentSar = Math.max(currentSar, high[i-1], high[i-2]);
        }

        // Check for trend reversal
        let reversed = false;
        if (prevTrend === 1 && low[i] < currentSar) { // Uptrend reverses to downtrend
            trend = -1;
            sar = prevEp; // SAR becomes the high EP of the prior uptrend
            ep = low[i]; // New EP is the current low
            af = PSAR_START; // Reset AF
            reversed = true;
        } else if (prevTrend === -1 && high[i] > currentSar) { // Downtrend reverses to uptrend
            trend = 1;
            sar = prevEp; // SAR becomes the low EP of the prior downtrend
            ep = high[i]; // New EP is the current high
            af = PSAR_START; // Reset AF
            reversed = true;
        }

        // If no reversal, continue the trend and update SAR, EP, AF
        if (!reversed) {
            sar = currentSar; // Use the calculated SAR for this period
            trend = prevTrend; // Trend remains the same

            // Update EP and AF if a new extreme is made in the direction of the trend
            if (trend === 1 && high[i] > prevEp) {
                ep = high[i];
                af = Math.min(PSAR_MAX, prevAf + PSAR_INCREMENT);
            } else if (trend === -1 && low[i] < prevEp) {
                ep = low[i];
                af = Math.min(PSAR_MAX, prevAf + PSAR_INCREMENT);
            } else {
                // No new extreme, keep EP and AF the same
                ep = prevEp;
                af = prevAf;
            }
        }
        // Store the final SAR value for the current bar (either the reversal SAR or the calculated continuing SAR)
        psarValues[i] = sar;
    }
    return psarValues;
}

/** Calculates rolling Volume Weighted Average Price (VWAP) */
function calculateRollingVWAP(close, volume, length) {
    length = Math.max(1, Math.round(length));
    const n = close ? close.length : 0;
    if (!close || !volume || close.length !== volume.length || n < length) {
        return new Array(n).fill(NaN);
    }

    const vwapValues = new Array(n).fill(NaN);
    let sumPV = 0; // Sum of Price * Volume
    let sumVol = 0; // Sum of Volume
    let validCount = 0; // Count of candles with valid price and volume > 0

    // Calculate initial window
    for (let i = 0; i < length; i++) {
        const c = close[i], v = volume[i];
        if (Number.isFinite(c) && Number.isFinite(v) && v > 0) {
            sumPV += c * v;
            sumVol += v;
            validCount++;
        }
    }
    // Calculate first VWAP if the window is fully valid and volume exists
    if (validCount === length && sumVol > 0) {
        vwapValues[length - 1] = sumPV / sumVol;
    }

    // Calculate rolling window
    for (let i = length; i < n; i++) {
        const enterC = close[i], enterV = volume[i];
        const exitC = close[i-length], exitV = volume[i-length];
        let windowChanged = false;

        // Subtract exiting candle's data if it was valid
        if (Number.isFinite(exitC) && Number.isFinite(exitV) && exitV > 0) {
            sumPV -= exitC * exitV;
            sumVol -= exitV;
            validCount--;
            windowChanged = true;
        }
        // Add entering candle's data if it is valid
        if (Number.isFinite(enterC) && Number.isFinite(enterV) && enterV > 0) {
            sumPV += enterC * enterV;
            sumVol += enterV;
            validCount++;
            windowChanged = true;
        }

        // Calculate VWAP if the window is still valid
        if (validCount === length && sumVol > 0) {
            vwapValues[i] = sumPV / sumVol;
        } else {
            vwapValues[i] = NaN;
            // If the window became invalid, reset state to potentially recover later
            if(windowChanged) {
                 sumPV = NaN; sumVol = NaN; validCount = 0;
            }
        }
         // Try to recover state if it became NaN
         if (!Number.isFinite(sumPV) && i >= length -1) {
             const sliceC = close.slice(i - length + 1, i + 1);
             const sliceV = volume.slice(i - length + 1, i + 1);
             sumPV = 0; sumVol = 0; validCount = 0;
             for(let k = 0; k < length; k++){
                  if(Number.isFinite(sliceC[k]) && Number.isFinite(sliceV[k]) && sliceV[k] > 0){
                       sumPV += sliceC[k] * sliceV[k];
                       sumVol += sliceV[k];
                       validCount++;
                  }
             }
              if (validCount === length && sumVol > 0) {
                vwapValues[i] = sumPV / sumVol;
             }
         }
    }
    return vwapValues;
}

// --- Main Indicator Calculation Orchestrator ---

/**
 * Calculates all configured technical indicators.
 * @param {object} config The configuration object.
 * @param {object} data OHLCV data object with arrays: timestamps, open, high, low, close, volume.
 * @returns {object|null} An object containing all calculated indicator results, or null if insufficient data.
 */
function calculateIndicators(config, data) {
    const { high, low, close, open, volume } = data;
    const n = close ? close.length : 0;

    // Determine minimum required data length based on enabled indicators
    let minRequiredLength = MIN_MA_LENGTH; // Absolute minimum
    if (config.showIchimoku) minRequiredLength = Math.max(minRequiredLength, ICHIMOKU_SENKOU_B_LEN);
    if (config.showAO) minRequiredLength = Math.max(minRequiredLength, AO_SLOW_LEN);
    if (config.showMomentumMAs) minRequiredLength = Math.max(minRequiredLength, config.momentumRocLength + 1); // ROC needs lookback
    minRequiredLength = Math.max(minRequiredLength, config.limit, config.volMALength, config.atrLength, config.momentumMaLength, config.momentumEmaLength, config.fixedMaLength, config.fixedEmaLength, config.ehlersLength, config.vwapLength, BB_LENGTH, MACD_SLOW_LENGTH, RSI_LENGTH+1, STOCH_K_LENGTH+1 );

    if (n < minRequiredLength) {
        console.warn(chalk.yellow(`Warning: Insufficient data (${n} candles) for full indicator calculation (longest lookback requires ~${minRequiredLength}). Results may be inaccurate or incomplete.`));
        if (n < 2) return null; // Cannot calculate pivots/previous values without at least 2 candles
    }

    // --- Base Indicators ---
    const atrValues = calculateAtr(high, low, close, config.atrLength);
    const volMaValues = calculateSma(volume, config.volMALength);

    // --- Pivot Calculation ---
    // Safely get previous candle's data (index n-2)
    const pdH = getSafePrev(high);
    const pdL = getSafePrev(low);
    const pdC = getSafePrev(close);
    const pdO = getSafePrev(open);
    const pdATR = getSafePrev(atrValues);
    const pdVol = getSafePrev(volume);
    const pdVolMA = getSafePrev(volMaValues);

    const pivotLevels = calculatePivots(pdH, pdL, pdC, pdO, pdATR, pdVol, pdVolMA, config.ppMethod, config.rangeMethod,
        config.volInfluence, config.volFactorMinClamp, config.volFactorMaxClamp,
        config.fibRatio1, config.fibRatio2, config.fibRatio3);

    // --- Ehlers ---
    const ehlersSrcData = getSourceData(config.ehlersSrc, data);
    const ehlersTrendlineValues = config.showEhlers ? calculateEhlersSmoother(ehlersSrcData, config.ehlersLength) : new Array(n).fill(NaN);

    // --- Dynamic Momentum MAs ---
    let momentumMaValues = new Array(n).fill(NaN);
    let momentumEmaValues = new Array(n).fill(NaN);
    let rocValues = new Array(n).fill(NaN);
    let adjustedMaLength = config.momentumMaLength; // Start with base length
    let adjustedEmaLength = config.momentumEmaLength; // Start with base length

    if (config.showMomentumMAs && n > config.momentumRocLength) {
        rocValues = calculateRoc(close, config.momentumRocLength);
        const lastRoc = getSafeLast(rocValues);

        if (Number.isFinite(lastRoc) && config.momentumRocNormRange !== 0) {
            const normalizedRoc = Math.max(-1.0, Math.min(1.0, lastRoc / config.momentumRocNormRange));
            const adjustmentFactor = 1.0 - normalizedRoc * config.momentumSensitivity;
            // Calculate adjusted lengths, ensuring they stay within configured min/max and MA func requirements
            adjustedEmaLength = Math.round(Math.max(config.momentumMinLength, Math.min(config.momentumMaxLength, config.momentumEmaLength * adjustmentFactor)));
            adjustedMaLength = Math.round(Math.max(config.momentumMinLength, Math.min(config.momentumMaxLength, config.momentumMaLength * adjustmentFactor)));
            adjustedEmaLength = Math.max(MIN_MA_LENGTH, adjustedEmaLength);
            adjustedMaLength = Math.max(MIN_MA_LENGTH, adjustedMaLength);
        }
        // Calculate MAs using the potentially adjusted lengths
        momentumEmaValues = calculateEma(close, adjustedEmaLength);
        momentumMaValues = calculateSma(close, adjustedMaLength);
    }

    // --- Other Standard Indicators ---
    const momVolume = calculateMomentumVolume(close, volume, config.momentumRocLength); // Use momentum ROC length for consistency
    const rsiValues = calculateRsi(close, RSI_LENGTH);
    const stochRsiValues = calculateStochRsi(rsiValues, STOCH_K_LENGTH, STOCH_D_LENGTH, STOCH_SMOOTH_K);
    const macdValues = calculateMacd(close, MACD_FAST_LENGTH, MACD_SLOW_LENGTH, MACD_SIGNAL_LENGTH);
    const bbValues = calculateBollingerBands(close, BB_LENGTH, BB_MULT);

    // --- Fixed MAs ---
    const fixedMaValues = config.showFixedMAs ? calculateSma(close, config.fixedMaLength) : new Array(n).fill(NaN);
    const fixedEmaValues = config.showFixedMAs ? calculateEma(close, config.fixedEmaLength) : new Array(n).fill(NaN);

    // --- Advanced Indicators ---
    const ichimokuValues = config.showIchimoku ? calculateIchimoku(high, low, close) : { tenkan: [], kijun: [], senkouA: [], senkouB: [], chikou: [] };
    const aoValues = config.showAO ? calculateAO(high, low) : new Array(n).fill(NaN);
    const psarValues = config.showPSAR ? calculatePSAR(high, low, close) : new Array(n).fill(NaN);
    const vwapValues = config.showVWAP ? calculateRollingVWAP(close, volume, config.vwapLength) : new Array(n).fill(NaN);

    // --- Return all results ---
    return {
        pivots: pivotLevels,
        ehlersTrendline: ehlersTrendlineValues,
        momentumMa: momentumMaValues,
        momentumEma: momentumEmaValues,
        adjustedMaLength: adjustedMaLength, // Include the calculated lengths
        adjustedEmaLength: adjustedEmaLength,
        fixedMa: fixedMaValues,
        fixedEma: fixedEmaValues,
        atr: atrValues,
        roc: rocValues,
        momentumVolume: momVolume,
        rsi: rsiValues,
        stochRsi: stochRsiValues,
        macd: macdValues,
        bb: bbValues,
        ichimoku: ichimokuValues,
        ao: aoValues,
        psar: psarValues,
        vwap: vwapValues
    };
}


// --- Termux SMS Integration ---

/**
 * Sends an SMS message using the termux-sms-send command.
 * @param {string} message The message content.
 * @param {string} phoneNumber The recipient's phone number.
 * @returns {Promise<{success: boolean, message?: string}>} Promise resolving with success status.
 */
function sendSms(message, phoneNumber) {
    return new Promise((resolve) => {
        // Basic validation
        if (!phoneNumber || typeof phoneNumber !== 'string' || phoneNumber.trim().length < 5) {
            console.warn(chalk.yellow("# Phone number invalid or not configured. SMS not sent."));
            return resolve({ success: false, message: 'Invalid or missing phone number' });
        }

        // Check if running in Termux
        if (typeof process === 'undefined' || !process.env || !process.env.TERMUX_VERSION) {
             console.warn(chalk.yellow("# Not running in Termux environment. SMS sending skipped."));
             return resolve({ success: false, message: 'Not in Termux environment' });
        }

        // Escape potential shell special characters in the message
        const escapedMessage = message.replace(/["`$!]/g, '\\$&'); // Escape ", `, $, !
        const command = `termux-sms-send -n "${phoneNumber}" "${escapedMessage}"`;

        exec(command, (error, stdout, stderr) => {
            if (error) {
                console.error(chalk.red(`# SMS Error executing command: ${error.message}`));
                console.error(chalk.red(`# Command: ${command}`)); // Log command for debugging
                if (stderr) console.error(chalk.red(`stderr: ${stderr}`));
                resolve({ success: false, message: `Termux command failed: ${error.message}` });
            } else {
                // Limit logged message length
                const logMessage = message.length > 60 ? message.substring(0, 57) + '...' : message;
                console.log(chalk.green(`# SMS Sent to ${phoneNumber}: "${logMessage}"`));
                resolve({ success: true });
            }
        });
    });
}


// --- CCXT Data Fetching ---

/**
 * Fetches OHLCV data from the exchange with retry logic.
 * @param {ccxt.Exchange} exchange Initialized CCXT exchange instance.
 * @param {string} symbol Trading pair symbol.
 * @param {string} timeframe Chart timeframe.
 * @param {number} limit Number of candles to fetch.
 * @param {number} retries Maximum number of retry attempts.
 * @returns {Promise<object|null>} OHLCV data object or null on failure.
 */
async function fetchOhlcvData(exchange, symbol, timeframe, limit, retries = 3) {
    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            // console.log(chalk.blue(`# Attempt ${attempt}: Fetching ${limit} ${timeframe} candles for ${symbol}...`));
            const ohlcv = await exchange.fetchOHLCV(symbol, timeframe, undefined, limit);

            // Basic validation of response
            if (!Array.isArray(ohlcv) || ohlcv.length === 0) {
                console.warn(chalk.yellow(`# Attempt ${attempt}: No OHLCV data returned for ${symbol} from ${exchange.id}.`));
                 if (attempt === retries) return null;
                 await sleep(config.retryFetchDelaySeconds * 1000 * attempt); // Wait before retry
                 continue;
            }
            if (ohlcv.length < MIN_MA_LENGTH) { // Need at least 2 for prev calculations
                console.warn(chalk.yellow(`# Attempt ${attempt}: Insufficient OHLCV data (${ohlcv.length} candles) for ${symbol}. Need at least ${MIN_MA_LENGTH}.`));
                if (attempt === retries) return null;
                await sleep(config.retryFetchDelaySeconds * 1000 * attempt);
                continue;
            }
             if (ohlcv.length < limit && attempt === 1) { // Warn only on first attempt if limit not met
                 console.warn(chalk.yellow(`# Received ${ohlcv.length}/${limit} candles for ${symbol}. Calculations might be less accurate if lookbacks aren't met.`));
             }


            // Data seems okay, format and return
            return {
                timestamps: ohlcv.map(c => c[0]),
                open:       ohlcv.map(c => c[1]),
                high:       ohlcv.map(c => c[2]),
                low:        ohlcv.map(c => c[3]),
                close:      ohlcv.map(c => c[4]),
                volume:     ohlcv.map(c => c[5])
            };
        } catch (e) {
            console.error(chalk.red(`# Attempt ${attempt}: Error fetching ${symbol}:`), e.constructor.name);
            console.error(chalk.red(`Message: ${e.message}`));

            if (e instanceof ccxt.AuthenticationError) {
                console.error(chalk.red.bold(`# Authentication Error! Check API keys for ${exchange.id}. Script cannot continue.`));
                process.exit(1); // Exit immediately on auth error
            } else if (e instanceof ccxt.BadSymbol) {
                console.error(chalk.red.bold(`# Bad Symbol Error! Symbol '${symbol}' is likely invalid on ${exchange.id}. Script cannot continue.`));
                 process.exit(1); // Exit immediately on bad symbol
            } else if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeNotAvailable) {
                console.warn(chalk.yellow(`# Network/Timeout/Exchange issue. ` + (attempt < retries ? `Waiting ${config.retryFetchDelaySeconds * attempt}s before retry...` : `Max retries reached.`)));
                if (attempt === retries) return null; // Give up after last attempt
                await sleep(config.retryFetchDelaySeconds * 1000 * attempt); // Exponential backoff
            } else if (e instanceof ccxt.RateLimitExceeded) {
                 console.warn(chalk.yellow(`# Rate Limit Exceeded. Waiting longer before retry...`));
                 if (attempt === retries) return null;
                 await sleep(config.retryFetchDelaySeconds * 1000 * attempt * 2); // Wait longer for rate limits
            }
            else {
                // For other errors (e.g., generic ExchangeError), log but might not be retryable
                console.error(chalk.red(`# Unhandled CCXT Error (${e.constructor.name}). Check exchange status or error details.`));
                 // Optionally log the full error: console.error(e);
                 if (attempt === retries) return null; // Give up after last attempt if designated retryable
                 await sleep(config.retryFetchDelaySeconds * 1000 * attempt); // Standard wait
            }
        }
    }
    console.error(chalk.red(`# Data fetch failed for ${symbol} after ${retries} attempts.`));
    return null; // Explicitly return null if all retries fail
}

// --- Main Application Logic ---

/**
 * The main analysis and alerting loop.
 */
async function mainLoop() {
    const exchangeId = config.exchange.toLowerCase();
    // Validation already checked if exchangeId is in ccxt.exchanges

    const exchangeClass = ccxt[exchangeId];
    const apiKey = process.env[`${exchangeId.toUpperCase()}_API_KEY`] || process.env['EXCHANGE_API_KEY'];
    const apiSecret = process.env[`${exchangeId.toUpperCase()}_API_SECRET`] || process.env['EXCHANGE_API_SECRET'];

    if (!apiKey || !apiSecret) {
        console.warn(chalk.yellow(`# Warning: API Key/Secret not found in environment variables for ${exchangeId.toUpperCase()}.`));
        console.warn(chalk.yellow(`# Using public endpoints only. Set ${exchangeId.toUpperCase()}_API_KEY and ${exchangeId.toUpperCase()}_API_SECRET for private access/higher rate limits.`));
    }

    const exchange = new exchangeClass({
        apiKey: apiKey,
        secret: apiSecret,
        enableRateLimit: true, // CCXT built-in rate limiting
        options: {
            adjustForTimeDifference: true, // Helps with timestamp sync issues
            // Add exchange-specific options if needed, e.g.,
            // 'defaultType': 'spot', // Or 'swap', 'future' for exchanges like Bybit
        }
    });

    // --- Load Markets & Validate Symbol ---
    try {
        await exchange.loadMarkets();
        console.log(chalk.green(`# Loaded ${Object.keys(exchange.markets).length} markets from ${exchange.name}.`));
        if (!exchange.markets[config.symbol]) {
             console.error(chalk.red.bold(`# Error: Symbol '${config.symbol}' not found on ${exchange.name}.`));
             const availableSymbols = Object.keys(exchange.markets);
             console.log(`# Available symbols sample: ${availableSymbols.slice(0, 10).join(', ')}...`);
             process.exit(1); // Exit if symbol is invalid for the exchange
        }
         // Optional: Validate timeframe against exchange capabilities
         if (exchange.timeframes && !exchange.timeframes[config.timeframe]) {
            console.warn(chalk.yellow(`# Warning: Timeframe '${config.timeframe}' might not be natively supported by ${exchange.name}. CCXT might emulate it. Available: ${Object.keys(exchange.timeframes || {}).join(', ')}`));
         }
    } catch (e) {
        console.error(chalk.red.bold(`# Failed to load markets from ${exchange.name}: ${e.message}`));
        console.error(e); // Log full error for debugging
        process.exit(1); // Cannot proceed without markets
    }

    // --- Testnet Configuration ---
    if (config.testnet) {
        if (exchange.urls.test) {
            try {
                 // Some exchanges require setting the sandboxMode option
                 if ('sandboxMode' in exchange.options) {
                     exchange.options['sandboxMode'] = true;
                 } else {
                     // Fallback: try setting the API URL directly
                     exchange.urls.api = typeof exchange.urls.api === 'string'
                        ? exchange.urls.test
                        : exchange.urls.api = exchange.urls.test[Object.keys(exchange.urls.test)[0]]; // Use first testnet URL if it's an object
                 }
                 console.log(chalk.yellow(`# Using Testnet endpoint for ${exchange.name}. URL: ${JSON.stringify(exchange.urls.api)}`));
                 // Re-load markets for testnet if necessary (some exchanges require this)
                 // await exchange.loadMarkets();
            } catch (testnetError) {
                 console.error(chalk.red(`# Error switching to Testnet for ${exchange.name}: ${testnetError.message}`));
                 console.warn(chalk.yellow(`# Proceeding with default endpoint.`));
            }

        } else {
            console.warn(chalk.yellow(`# Testnet requested but no specific testnet URL found for ${exchange.name} in CCXT. Using default endpoint.`));
        }
    }


    console.log(chalk.cyan('\n# --- Initializing Market Analysis Loop --- #'));
    console.log(chalk.cyan(`Exchange: ${exchange.name} | Symbol: ${config.symbol} | Timeframe: ${config.timeframe}`));
    console.log(chalk.cyan(`SMS Alerts: ${config.phoneNumber ? `Enabled (${config.phoneNumber})` : 'Disabled'}`));
    console.log(chalk.cyan(`Testnet Mode: ${config.testnet}`));
    console.log(chalk.cyan('---------------------------------------------'));


    let alertStates = {}; // Persists across loop iterations { alertKey: boolean }
    let lastDataTimestamp = null; // Track the timestamp of the last processed candle

    // --- Main Loop ---
    while (true) {
        const loopStartTime = Date.now();
        let data = null;
        let indicators = null;

        try { // Wrap entire iteration
            // 1. Fetch Data
            data = await fetchOhlcvData(exchange, config.symbol, config.timeframe, config.limit);

            if (!data) {
                console.warn(chalk.yellow("# Data fetch failed after retries. Waiting for next cycle..."));
                await sleep(config.loopTargetSeconds * 1000);
                continue;
            }

             // Check if data is new
             const currentTimestamp = getSafeLast(data.timestamps);
             if (currentTimestamp === lastDataTimestamp) {
                 // console.log(chalk.grey("No new candle data. Waiting briefly..."));
                 await sleep(Math.max(1000, config.loopTargetSeconds * 1000 / 4)); // Shorter wait if no new data
                 continue;
             }
             if (currentTimestamp) { // Only update if we got a valid timestamp
                 lastDataTimestamp = currentTimestamp;
             }


            // 2. Calculate Indicators
            indicators = calculateIndicators(config, data);

            if (!indicators) {
                console.warn(chalk.yellow("# Indicator calculation failed or returned null. Waiting..."));
                await sleep(config.loopTargetSeconds * 1000);
                continue;
            }

            // 3. Extract Latest Values & Perform Analysis/Alerting
            const n = data.close.length;
            const latestO = getSafeLast(data.open);
            const latestH = getSafeLast(data.high);
            const latestL = getSafeLast(data.low);
            const latestC = getSafeLast(data.close);
            const latestV = getSafeLast(data.volume);
            const prevC = getSafePrev(data.close);

            // --- Indicator Values ---
            const pivots    = indicators.pivots; // Object: {pp, r1, s1, ... tc, bc}
            const ehlers    = getSafeLast(indicators.ehlersTrendline);
            const prevEhlers= getSafePrev(indicators.ehlersTrendline);
            const momMa     = getSafeLast(indicators.momentumMa);
            const prevMomMa = getSafePrev(indicators.momentumMa);
            const momEma    = getSafeLast(indicators.momentumEma);
            const prevMomEma= getSafePrev(indicators.momentumEma);
            const momVol    = getSafeLast(indicators.momentumVolume);
            const rsi       = getSafeLast(indicators.rsi);
            const stochK    = getSafeLast(indicators.stochRsi.k);
            const stochD    = getSafeLast(indicators.stochRsi.d);
            const macdLine  = getSafeLast(indicators.macd.macd);
            const signalLine= getSafeLast(indicators.macd.signal);
            const macdHist  = getSafeLast(indicators.macd.histogram);
            const prevMacdLine = getSafePrev(indicators.macd.macd);
            const prevSignalLine= getSafePrev(indicators.macd.signal);
            const bbUpper   = getSafeLast(indicators.bb.upper);
            const bbMiddle  = getSafeLast(indicators.bb.middle);
            const bbLower   = getSafeLast(indicators.bb.lower);
            const tenkan    = getSafeLast(indicators.ichimoku.tenkan);
            const kijun     = getSafeLast(indicators.ichimoku.kijun);
            const senkouA   = getSafeLast(indicators.ichimoku.senkouA); // Value plotted ahead, matches current index
            const senkouB   = getSafeLast(indicators.ichimoku.senkouB); // Value plotted ahead, matches current index
            const chikou    = getSafeLast(indicators.ichimoku.chikou); // Value from past plotted at current index
            const prevTenkan = getSafePrev(indicators.ichimoku.tenkan);
            const prevKijun = getSafePrev(indicators.ichimoku.kijun);
            const ao        = getSafeLast(indicators.ao);
            const prevAo    = getSafePrev(indicators.ao);
            const psar      = getSafeLast(indicators.psar);
            const prevPsar  = getSafePrev(indicators.psar);
            const vwap      = getSafeLast(indicators.vwap);

             // --- Determine Kumo (Cloud) ---
             const isPriceAboveKumo = Number.isFinite(senkouA) && Number.isFinite(senkouB) && latestC > Math.max(senkouA, senkouB);
             const isPriceBelowKumo = Number.isFinite(senkouA) && Number.isFinite(senkouB) && latestC < Math.min(senkouA, senkouB);
             const wasPriceAboveKumo = Number.isFinite(getSafePrev(indicators.ichimoku.senkouA)) && Number.isFinite(getSafePrev(indicators.ichimoku.senkouB)) && prevC > Math.max(getSafePrev(indicators.ichimoku.senkouA), getSafePrev(indicators.ichimoku.senkouB));
             const wasPriceBelowKumo = Number.isFinite(getSafePrev(indicators.ichimoku.senkouA)) && Number.isFinite(getSafePrev(indicators.ichimoku.senkouB)) && prevC < Math.min(getSafePrev(indicators.ichimoku.senkouA), getSafePrev(indicators.ichimoku.senkouB));


            // --- Alerting Logic ---
            const currentAlertsTriggered = new Set(); // Track alerts triggered THIS cycle

            /** Checks condition and sends SMS if alert state changes from false to true */
            const checkAlert = async (key, condition, message) => {
                if (condition) {
                    currentAlertsTriggered.add(key); // Mark as active this cycle
                    if (!alertStates[key]) { // Only send if state was previously false (or unset)
                        // console.log(chalk.yellow(`DEBUG: Triggering alert '${key}'`)); // Debug log
                        try {
                            const result = await sendSms(`[${config.symbol}] ${message}`, config.phoneNumber);
                            if(result.success) {
                                alertStates[key] = true; // Set state to true only AFTER successful send (or if SMS disabled)
                            } else if (result.message === 'Invalid or missing phone number') {
                                 alertStates[key] = true; // Treat as 'sent' if SMS is disabled
                            }
                             // else: Keep state false if SMS failed to send
                        } catch (smsError) { /* Error logged in sendSms, state remains false */ }
                    }
                }
                // Implicit else: If condition is false, the state will be reset later if it wasn't triggered this cycle.
            };


            // --- Define Alert Conditions ---

            // Pivot Alerts
            if (Number.isFinite(pivots.pp)) {
                await checkAlert("pp_cross_above", config.alertOnPPCross && prevC < pivots.pp && latestC > pivots.pp, `Price ${latestC.toFixed(2)} crossed ABOVE PP: ${pivots.pp.toFixed(2)}`);
                await checkAlert("pp_cross_below", config.alertOnPPCross && prevC > pivots.pp && latestC < pivots.pp, `Price ${latestC.toFixed(2)} crossed BELOW PP: ${pivots.pp.toFixed(2)}`);
            }
            // R/S Level Alerts (Example R1/S1)
            if (config.alertOnR1Cross && Number.isFinite(pivots.r1)) {
                 await checkAlert("r1_cross_above", prevC < pivots.r1 && latestC > pivots.r1, `Price ${latestC.toFixed(2)} crossed ABOVE R1: ${pivots.r1.toFixed(2)}`);
                 // await checkAlert("r1_cross_below", prevC > pivots.r1 && latestC < pivots.r1, `Price ${latestC.toFixed(2)} crossed BELOW R1: ${pivots.r1.toFixed(2)}`); // Below R1 isn't usually a signal itself
            }
            // ... Add checks for R2, R3 similarly if enabled ...
             if (config.alertOnS1Cross && Number.isFinite(pivots.s1)) {
                // await checkAlert("s1_cross_above", prevC < pivots.s1 && latestC > pivots.s1, `Price ${latestC.toFixed(2)} crossed ABOVE S1: ${pivots.s1.toFixed(2)}`); // Above S1 isn't usually a signal itself
                await checkAlert("s1_cross_below", prevC > pivots.s1 && latestC < pivots.s1, `Price ${latestC.toFixed(2)} crossed BELOW S1: ${pivots.s1.toFixed(2)}`);
            }
            // ... Add checks for S2, S3 similarly if enabled ...
            // CPR Alert
            if (config.alertOnCPREnterExit && Number.isFinite(pivots.tc) && Number.isFinite(pivots.bc)) {
                const isInCPR = latestC < pivots.tc && latestC > pivots.bc;
                const wasInCPR = prevC < pivots.tc && prevC > pivots.bc; // Assuming TC/BC don't change mid-bar
                await checkAlert("cpr_enter", !wasInCPR && isInCPR, `Price ${latestC.toFixed(2)} ENTERED CPR (${pivots.bc.toFixed(2)} - ${pivots.tc.toFixed(2)})`);
                await checkAlert("cpr_exit", wasInCPR && !isInCPR, `Price ${latestC.toFixed(2)} EXITED CPR (${pivots.bc.toFixed(2)} - ${pivots.tc.toFixed(2)})`);
            }


            // Ehlers Alerts
            if (config.showEhlers && Number.isFinite(ehlers) && Number.isFinite(prevEhlers)) {
                await checkAlert("ehlers_cross_above", config.alertOnEhlersCross && prevC < prevEhlers && latestC > ehlers, `Price ${latestC.toFixed(2)} crossed ABOVE Ehlers: ${ehlers.toFixed(3)}`);
                await checkAlert("ehlers_cross_below", config.alertOnEhlersCross && prevC > prevEhlers && latestC < ehlers, `Price ${latestC.toFixed(2)} crossed BELOW Ehlers: ${ehlers.toFixed(3)}`);
                // Slope Alert (requires third point)
                 const prevPrevEhlers = getSafePrev(indicators.ehlersTrendline.slice(0,-1)); // Ehlers value from 2 bars ago
                 if(config.alertOnEhlersSlope && Number.isFinite(prevPrevEhlers)) {
                    const slopeUp = ehlers > prevEhlers;
                    const prevSlopeUp = prevEhlers > prevPrevEhlers;
                     await checkAlert("ehlers_slope_up", slopeUp && !prevSlopeUp, `Ehlers slope turned UP at ${ehlers.toFixed(3)}`);
                     await checkAlert("ehlers_slope_down", !slopeUp && prevSlopeUp, `Ehlers slope turned DOWN at ${ehlers.toFixed(3)}`);
                 }
            }

            // Momentum MA Alerts
            if (config.showMomentumMAs && Number.isFinite(momMa) && Number.isFinite(prevMomMa)) {
                await checkAlert("mom_ma_cross_above", config.alertOnMomMACross && prevC < prevMomMa && latestC > momMa, `Price ${latestC.toFixed(2)} crossed ABOVE MomMA(${indicators.adjustedMaLength}): ${momMa.toFixed(3)}`);
                await checkAlert("mom_ma_cross_below", config.alertOnMomMACross && prevC > prevMomMa && latestC < momMa, `Price ${latestC.toFixed(2)} crossed BELOW MomMA(${indicators.adjustedMaLength}): ${momMa.toFixed(3)}`);
            }
            if (config.showMomentumMAs && Number.isFinite(momEma) && Number.isFinite(prevMomEma)) {
                await checkAlert("mom_ema_cross_above", config.alertOnMomEMACross && prevC < prevMomEma && latestC > momEma, `Price ${latestC.toFixed(2)} crossed ABOVE MomEMA(${indicators.adjustedEmaLength}): ${momEma.toFixed(3)}`);
                await checkAlert("mom_ema_cross_below", config.alertOnMomEMACross && prevC > prevMomEma && latestC < momEma, `Price ${latestC.toFixed(2)} crossed BELOW MomEMA(${indicators.adjustedEmaLength}): ${momEma.toFixed(3)}`);
            }
            if (config.showMomentumMAs && config.alertOnMomMAvsEMACross && Number.isFinite(momMa) && Number.isFinite(momEma) && Number.isFinite(prevMomMa) && Number.isFinite(prevMomEma)) {
                await checkAlert("mom_ma_ema_cross_bull", prevMomMa < prevMomEma && momMa > momEma, `MomMA(${indicators.adjustedMaLength}) ${momMa.toFixed(3)} crossed ABOVE MomEMA(${indicators.adjustedEmaLength}) ${momEma.toFixed(3)}`);
                await checkAlert("mom_ma_ema_cross_bear", prevMomMa > prevMomEma && momMa < momEma, `MomMA(${indicators.adjustedMaLength}) ${momMa.toFixed(3)} crossed BELOW MomEMA(${indicators.adjustedEmaLength}) ${momEma.toFixed(3)}`);
            }


            // High Volume Alert
            await checkAlert("high_mom_vol", config.alertOnHighMomVol && Number.isFinite(momVol) && momVol > config.highMomVolThreshold, `High Momentum Volume: ${momVol.toFixed(2)} (Threshold: ${config.highMomVolThreshold})`);

            // RSI/StochRSI Alerts
            if (Number.isFinite(stochK) && Number.isFinite(stochD)) {
                await checkAlert("stoch_rsi_ob", config.alertOnStochRsiOverbought && stochK > config.stochRsiOverboughtThreshold && stochD > config.stochRsiOverboughtThreshold, `Stoch RSI Overbought: K=${stochK.toFixed(1)}, D=${stochD.toFixed(1)}`);
                await checkAlert("stoch_rsi_os", config.alertOnStochRsiOversold && stochK < config.stochRsiOversoldThreshold && stochD < config.stochRsiOversoldThreshold, `Stoch RSI Oversold: K=${stochK.toFixed(1)}, D=${stochD.toFixed(1)}`);
                // Could add Stoch K/D crossover alerts here too if desired
            }
            if (Number.isFinite(rsi)) {
                await checkAlert("rsi_ob", config.alertOnRsiOverbought && rsi > config.rsiOverboughtThreshold, `RSI Overbought: ${rsi.toFixed(1)}`);
                await checkAlert("rsi_os", config.alertOnRsiOversold && rsi < config.rsiOversoldThreshold, `RSI Oversold: ${rsi.toFixed(1)}`);
            }

            // MACD Alerts
            if (Number.isFinite(macdLine) && Number.isFinite(signalLine) && Number.isFinite(prevMacdLine) && Number.isFinite(prevSignalLine)) {
                 const bullCrossCond = prevMacdLine < prevSignalLine && macdLine > signalLine && macdLine > config.macdCrossThreshold && signalLine > config.macdCrossThreshold;
                 const bearCrossCond = prevMacdLine > prevSignalLine && macdLine < signalLine && macdLine < config.macdCrossThreshold && signalLine < config.macdCrossThreshold;
                await checkAlert("macd_bull_cross", config.alertOnMacdBullishCross && bullCrossCond, `MACD Bullish Cross: MACD=${macdLine.toFixed(4)}, Signal=${signalLine.toFixed(4)}`);
                await checkAlert("macd_bear_cross", config.alertOnMacdBearishCross && bearCrossCond, `MACD Bearish Cross: MACD=${macdLine.toFixed(4)}, Signal=${signalLine.toFixed(4)}`);
                // Could also alert on histogram crossing zero
            }

            // Bollinger Band Alerts
            if (Number.isFinite(latestC) && Number.isFinite(bbUpper) && Number.isFinite(bbLower)) {
                const upperBreakoutPrice = bbUpper * config.bbBreakoutThresholdMultiplier;
                 // For lower breakout, use division: Price < LowerBB / Multiplier
                 // Example: If Multiplier is 1.005 (0.5%), we check if price is below LowerBB / 1.005
                 const lowerBreakoutPrice = bbLower / config.bbBreakoutThresholdMultiplier;

                await checkAlert("bb_breakout_upper", config.alertOnBBBreakoutUpper && latestC > upperBreakoutPrice, `Price ${latestC.toFixed(2)} broke > BB Upper Threshold: ${upperBreakoutPrice.toFixed(2)} (BB: ${bbUpper.toFixed(2)})`);
                await checkAlert("bb_breakout_lower", config.alertOnBBBreakoutLower && latestC < lowerBreakoutPrice, `Price ${latestC.toFixed(2)} broke < BB Lower Threshold: ${lowerBreakoutPrice.toFixed(2)} (BB: ${bbLower.toFixed(2)})`);
            }

             // Ichimoku Alerts
            if (config.showIchimoku) {
                if (Number.isFinite(latestC) && Number.isFinite(kijun) && Number.isFinite(prevC) && Number.isFinite(prevKijun)) {
                    await checkAlert("price_kijun_cross_a", config.alertOnPriceVsKijun && prevC < prevKijun && latestC > kijun, `Price ${latestC.toFixed(2)} crossed ABOVE Kijun: ${kijun.toFixed(2)}`);
                    await checkAlert("price_kijun_cross_b", config.alertOnPriceVsKijun && prevC > prevKijun && latestC < kijun, `Price ${latestC.toFixed(2)} crossed BELOW Kijun: ${kijun.toFixed(2)}`);
                }
                if (config.alertOnPriceVsKumo) {
                     await checkAlert("price_enter_kumo_a", !wasPriceAboveKumo && !wasPriceBelowKumo && isPriceAboveKumo, `Price ${latestC.toFixed(2)} entered Kumo from below`); // Entered bottom edge
                     await checkAlert("price_enter_kumo_b", !wasPriceAboveKumo && !wasPriceBelowKumo && isPriceBelowKumo, `Price ${latestC.toFixed(2)} entered Kumo from above`); // Entered top edge
                     await checkAlert("price_exit_kumo_a", (wasPriceAboveKumo || wasPriceBelowKumo) && !isPriceAboveKumo && !isPriceBelowKumo && latestC > Math.max(senkouA, senkouB), `Price ${latestC.toFixed(2)} exited Kumo upwards`); // Exited top
                     await checkAlert("price_exit_kumo_b", (wasPriceAboveKumo || wasPriceBelowKumo) && !isPriceAboveKumo && !isPriceBelowKumo && latestC < Math.min(senkouA, senkouB), `Price ${latestC.toFixed(2)} exited Kumo downwards`); // Exited bottom
                }
                 if (Number.isFinite(tenkan) && Number.isFinite(kijun) && Number.isFinite(prevTenkan) && Number.isFinite(prevKijun)) {
                     await checkAlert("tk_cross_bull", config.alertOnTKCross && prevTenkan < prevKijun && tenkan > kijun, `Tenkan ${tenkan.toFixed(2)} crossed ABOVE Kijun ${kijun.toFixed(2)} (Bullish TK Cross)`);
                     await checkAlert("tk_cross_bear", config.alertOnTKCross && prevTenkan > prevKijun && tenkan < kijun, `Tenkan ${tenkan.toFixed(2)} crossed BELOW Kijun ${kijun.toFixed(2)} (Bearish TK Cross)`);
                 }
                 // Chikou alerts are complex - require comparing chikou value (price from past) with price at chikou's plotted index
                 // Example: if (config.alertOnChikouPriceCross && Number.isFinite(chikou) && Number.isFinite(data.close[n-1 - ICHIMOKU_CHIKOU_OFFSET])) ...
            }

            // Awesome Oscillator Alerts
            if (config.showAO && Number.isFinite(ao) && Number.isFinite(prevAo)) {
                 await checkAlert("ao_cross_zero_bull", config.alertOnAOCrossZero && prevAo < 0 && ao > 0, `Awesome Oscillator crossed ABOVE zero: ${ao.toFixed(4)}`);
                 await checkAlert("ao_cross_zero_bear", config.alertOnAOCrossZero && prevAo > 0 && ao < 0, `Awesome Oscillator crossed BELOW zero: ${ao.toFixed(4)}`);
                 // Could add alerts for AO peaks/troughs or divergences
            }

             // Parabolic SAR Alerts
             if (config.showPSAR && Number.isFinite(psar) && Number.isFinite(prevPsar)) {
                 // Check if SAR flipped below/above price
                 const flippedUp = latestC > psar && prevC < prevPsar; // Price crosses above SAR
                 const flippedDown = latestC < psar && prevC > prevPsar; // Price crosses below SAR
                 await checkAlert("psar_flip_up", config.alertOnPSARFlip && flippedUp, `PSAR flipped UP (below price ${latestC.toFixed(2)}), SAR: ${psar.toFixed(2)}`);
                 await checkAlert("psar_flip_down", config.alertOnPSARFlip && flippedDown, `PSAR flipped DOWN (above price ${latestC.toFixed(2)}), SAR: ${psar.toFixed(2)}`);
             }

             // VWAP Alerts
             if (config.showVWAP && Number.isFinite(latestC) && Number.isFinite(vwap) && Number.isFinite(prevC) && Number.isFinite(getSafePrev(indicators.vwap))) {
                 const prevVwap = getSafePrev(indicators.vwap);
                 await checkAlert("price_vwap_cross_a", config.alertOnPriceVsVWAP && prevC < prevVwap && latestC > vwap, `Price ${latestC.toFixed(2)} crossed ABOVE VWAP: ${vwap.toFixed(2)}`);
                 await checkAlert("price_vwap_cross_b", config.alertOnPriceVsVWAP && prevC > prevVwap && latestC < vwap, `Price ${latestC.toFixed(2)} crossed BELOW VWAP: ${vwap.toFixed(2)}`);
             }


            // --- Reset alert states for conditions that are no longer true ---
            const activeKeys = Object.keys(alertStates);
            for (const key of activeKeys) {
                 if (alertStates[key] && !currentAlertsTriggered.has(key)) {
                      // console.log(chalk.grey(`DEBUG: Resetting alert state for '${key}'`)); // Debug log
                      alertStates[key] = false;
                 }
            }

            // 4. Console Output
            console.log(chalk.bold(`\n# --- ${config.symbol} | ${config.timeframe} | ${new Date(currentTimestamp).toLocaleString()} --- #`));

            // Price & Volume
             const priceChange = Number.isFinite(latestC) && Number.isFinite(prevC) ? latestC - prevC : NaN;
             const priceColor = !config.colorBars || !Number.isFinite(ehlers) || !Number.isFinite(prevEhlers)
                 ? chalk.white
                 : latestC > ehlers ? chalk.green : chalk.red; // Color based on price vs Ehlers
            console.log(`Price: ${priceColor.bold(latestC.toFixed(4))} (O:${latestO?.toFixed(4)} H:${latestH?.toFixed(4)} L:${latestL?.toFixed(4)}) Chg: ${chalk.grey(isNaN(priceChange) ? 'N/A' : priceChange.toFixed(4))} | Vol: ${chalk.grey(latestV?.toFixed(2) || 'N/A')}`);

            // Pivots & CPR
             if (Number.isFinite(pivots.pp)) {
                 let pivotStr = `Pivots: PP: ${pivots.pp.toFixed(3)} | R: ${pivots.r1.toFixed(3)} / ${pivots.r2.toFixed(3)} / ${pivots.r3.toFixed(3)} | S: ${pivots.s1.toFixed(3)} / ${pivots.s2.toFixed(3)} / ${pivots.s3.toFixed(3)}`;
                 if(config.showCPR) pivotStr += ` | CPR: ${pivots.bc.toFixed(3)}-${pivots.tc.toFixed(3)}`;
                 console.log(chalk.yellow(pivotStr));
             }

             // Ehlers & MAs
            let maLine = '';
            if(config.showEhlers) maLine += `Ehlers(${config.ehlersLength}): ${chalk.magenta(ehlers?.toFixed(4) || 'N/A')} | `;
             if(config.showMomentumMAs) maLine += `MomMA(${indicators.adjustedMaLength}): ${chalk.cyan(momMa?.toFixed(4) || 'N/A')} MomEMA(${indicators.adjustedEmaLength}): ${chalk.cyan(momEma?.toFixed(4) || 'N/A')} | `;
             if(config.showFixedMAs) maLine += `FixMA(${config.fixedMaLength}): ${chalk.blue(getSafeLast(indicators.fixedMa)?.toFixed(4) || 'N/A')} FixEMA(${config.fixedEmaLength}): ${chalk.blue(getSafeLast(indicators.fixedEma)?.toFixed(4) || 'N/A')} | `;
             if(maLine) console.log(maLine.slice(0, -3)); // Remove trailing ' | '

             // Oscillators & Volume
             console.log(`RSI(${RSI_LENGTH}): ${chalk.blue(rsi?.toFixed(2) || 'N/A')} | StochK(${STOCH_K_LENGTH},${STOCH_SMOOTH_K}): ${chalk.blue(stochK?.toFixed(2) || 'N/A')} | StochD(${STOCH_D_LENGTH}): ${chalk.blue(stochD?.toFixed(2) || 'N/A')} | MomVol: ${chalk.grey(momVol?.toFixed(2) || 'N/A')}`);

             // MACD & AO
             let macdAoLine = `MACD(${MACD_FAST_LENGTH},${MACD_SLOW_LENGTH}): ${chalk.green(macdLine?.toFixed(5) || 'N/A')} | Signal(${MACD_SIGNAL_LENGTH}): ${chalk.red(signalLine?.toFixed(5) || 'N/A')} | Hist: ${chalk.yellow(macdHist?.toFixed(5) || 'N/A')}`;
             if(config.showAO) macdAoLine += ` | AO: ${chalk.magenta(ao?.toFixed(4) || 'N/A')}`;
             console.log(macdAoLine);

             // BBands, PSAR, VWAP
             let bandsLine = `BBands(${BB_LENGTH},${BB_MULT}): L:${chalk.red(bbLower?.toFixed(3) || 'N/A')} M:${chalk.grey(bbMiddle?.toFixed(3) || 'N/A')} U:${chalk.red(bbUpper?.toFixed(3) || 'N/A')}`;
             if(config.showPSAR) bandsLine += ` | PSAR: ${chalk.yellow(psar?.toFixed(3) || 'N/A')}`;
             if(config.showVWAP) bandsLine += ` | VWAP(${config.vwapLength}): ${chalk.cyan(vwap?.toFixed(3) || 'N/A')}`;
             console.log(bandsLine);

              // Ichimoku Cloud
             if (config.showIchimoku) {
                 console.log(`Ichimoku: Tenkan: ${chalk.blue(tenkan?.toFixed(3) || 'N/A')} Kijun: ${chalk.red(kijun?.toFixed(3) || 'N/A')} | Kumo: ${chalk.green(senkouA?.toFixed(3) || 'N/A')} - ${chalk.green(senkouB?.toFixed(3) || 'N/A')} | Chikou: ${chalk.magenta(chikou?.toFixed(3) || 'N/A')}`);
             }

             // Display triggered alerts for this cycle
             const alertsNow = Array.from(currentAlertsTriggered);
             if (alertsNow.length > 0) {
                 console.log(chalk.yellow.bold("Alerts Triggered: ") + chalk.yellow(alertsNow.join(', ')));
             }


        } catch (error) {
            console.error(chalk.red.bold('\n# --- UNHANDLED ERROR IN MAIN LOOP --- #'));
            console.error(error);
            console.error(chalk.red.bold('# --- Attempting to continue... --- #\n'));
            // Don't exit, try to recover on the next loop iteration
        } finally {
            // 5. Sleep
            const loopEndTime = Date.now();
            const loopDuration = loopEndTime - loopStartTime;
            const sleepDuration = Math.max(500, (config.loopTargetSeconds * 1000) - loopDuration); // Ensure minimum sleep
            // console.log(chalk.grey(`Loop took ${loopDuration}ms, sleeping for ${sleepDuration}ms`)); // Optional debug
            await sleep(sleepDuration);
        }
    } // End while(true) loop
}

// --- Script Entry Point ---
async function start() {
    console.log(chalk.green.bold("\n===== Market Analysis Bot Initializing ====="));
    // Optional: Add pre-checks (e.g., Node version) if needed

    await mainLoop(); // Start the main execution loop
}

// Run the application and catch top-level errors
start().catch(error => {
    console.error(chalk.red.bold("\n# --- CRITICAL UNHANDLED ERROR --- #"));
    console.error(error);
    console.error(chalk.red.bold("# --- Script terminated unexpectedly --- #"));
    process.exit(1); // Exit with error code
});

// Optional: Handle graceful shutdown (e.g., on Ctrl+C)
process.on('SIGINT', () => {
    console.log(chalk.yellow.bold("\n# --- SIGINT received. Shutting down gracefully... --- #"));
    // Add any cleanup logic here if needed
    process.exit(0);
});
