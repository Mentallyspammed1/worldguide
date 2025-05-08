const fs = require('fs').promises;
const path = require('path');
const { performance } = require('perf_hooks');
const { Decimal } = require('decimal.js');
const dotenv = require('dotenv');
const nc = require('nanocolors');
const winston = require('winston');
const ccxt = require('ccxt');
const dfd = require('danfojs-node');
const { EMA, ATR } = require('technicalindicators');
const { DateTime } = require('luxon');

// --- Initialize Environment ---
Decimal.set({ precision: 28, rounding: Decimal.ROUND_DOWN });
dotenv.config();

// --- Constants ---
const BOT_VERSION = '1.3.3-js'; // <<<< Version Updated >>>>
const API_KEY = process.env.BYBIT_API_KEY;
const API_SECRET = process.env.BYBIT_API_SECRET;
const CONFIG_FILE = 'config.json';
const LOG_DIRECTORY = 'bot_logs';
const DEFAULT_TIMEZONE = 'America/Chicago';
const TIMEZONE = process.env.TIMEZONE || DEFAULT_TIMEZONE;
const MAX_API_RETRIES = 3;
const RETRY_DELAY_SECONDS = 5;
const POSITION_CONFIRM_DELAY_SECONDS = 8; // Default, override in config
const LOOP_DELAY_SECONDS = 15; // Default, override in config
const BYBIT_API_KLINE_LIMIT = 1000;
const DEFAULT_FETCH_LIMIT = 750; // Default, override in config
const MAX_DF_LEN = 2000; // Limit DataFrame size in memory
// Default strategy parameters (can be overridden in config.json per symbol)
const DEFAULT_VT_LENGTH = 40;
const DEFAULT_VT_ATR_PERIOD = 200;
const DEFAULT_VT_VOL_EMA_LENGTH = 950;
const DEFAULT_VT_ATR_MULTIPLIER = 3.0;
const DEFAULT_OB_SOURCE = 'Wicks'; // 'Wicks' or 'Body'
const DEFAULT_PH_LEFT = 10;
const DEFAULT_PH_RIGHT = 10;
const DEFAULT_PL_LEFT = 10;
const DEFAULT_PL_RIGHT = 10;
const DEFAULT_OB_EXTEND = true;
const DEFAULT_OB_MAX_BOXES = 50;
const DEFAULT_OB_ENTRY_PROXIMITY_FACTOR = 1.005; // e.g., 1.005 = 0.5% proximity
const DEFAULT_OB_EXIT_PROXIMITY_FACTOR = 1.001; // e.g., 1.001 = 0.1% proximity

// Global State
let QUOTE_CURRENCY = 'USDT'; // Default, updated from config/market info
let shutdownRequested = false;
const marketInfoCache = new Map();

// Timeframe Mapping
const VALID_INTERVALS = ['1', '3', '5', '15', '30', '60', '120', '240', 'D', 'W', 'M'];
const CCXT_INTERVAL_MAP = {
    '1': '1m', '3': '3m', '5': '5m', '15': '15m', '30': '30m',
    '60': '1h', '120': '2h', '240': '4h', 'D': '1d', 'W': '1w', 'M': '1M'
};

// Nanocolors Shortcuts
const { green, cyan, magenta, yellow, red, blue, gray, bold, dim, italic } = nc;

// Ensure Log Directory
(async () => {
    try { await fs.mkdir(LOG_DIRECTORY, { recursive: true }); }
    catch (e) { console.error(red(bold(`FATAL: Could not create log directory '${LOG_DIRECTORY}': ${e}`))); process.exit(1); }
})();

// --- Type Definitions (JSDoc) ---
/** @typedef {import('ccxt').Exchange} Exchange */
/** @typedef {import('danfojs-node').DataFrame} DataFrame */
/** @typedef {import('decimal.js').Decimal} DecimalInstance */
/** @typedef {import('ccxt').Order} Order */
/** @typedef {import('ccxt').Position} CcxtPosition */

/**
 * @typedef {object} OrderBlock Represents a detected Order Block zone.
 * @property {string} id Unique ID (e.g., "B_1678886400000" for Bearish, "L_..." for Bullish)
 * @property {'bull' | 'bear'} type Type of Order Block.
 * @property {number} timestamp Timestamp (ms UTC) of the candle defining the OB pivot.
 * @property {DecimalInstance} top The highest price level of the OB zone.
 * @property {DecimalInstance} bottom The lowest price level of the OB zone.
 * @property {boolean} active Whether the OB is currently considered valid (not violated).
 * @property {boolean} violated Whether the price has decisively closed beyond the OB.
 * @property {number | null} violation_ts Timestamp (ms UTC) when the violation occurred.
 * @property {number | null} extended_to_ts Timestamp (ms UTC) the OB currently extends to (if extension enabled).
 */

/**
 * @typedef {object} StrategyAnalysisResults Contains the output of the strategy calculation.
 * @property {DataFrame | null} dataframe The DataFrame containing calculated indicators (potentially trimmed).
 * @property {DecimalInstance | null} last_close Closing price of the most recent candle.
 * @property {boolean | null} current_trend_up True if the calculated trend is up, False if down, null if undetermined.
 * @property {boolean} trend_just_changed True if the trend flipped on the very last candle.
 * @property {OrderBlock[]} active_bull_boxes List of currently active bullish Order Blocks, sorted newest first.
 * @property {OrderBlock[]} active_bear_boxes List of currently active bearish Order Blocks, sorted newest first.
 * @property {number | null} vol_norm_int Normalized volume indicator value (integer) for the last candle.
 * @property {DecimalInstance | null} atr Average True Range (ATR) value for the last candle.
 * @property {DecimalInstance | null} upper_band Volumatic Trend upper band value for the last candle.
 * @property {DecimalInstance | null} lower_band Volumatic Trend lower band value for the last candle.
 * @property {'BUY' | 'SELL' | 'EXIT_LONG' | 'EXIT_SHORT' | 'NONE'} signal The final trading signal generated.
 */

/**
 * @typedef {object} MarketInfo CCXT market info augmented with Decimal types and flags for easier use.
 * @property {string} id Exchange-specific market ID (e.g., 'BTCUSDT').
 * @property {string} symbol Standardized symbol (e.g., 'BTC/USDT').
 * @property {string} base Base currency (e.g., 'BTC').
 * @property {string} quote Quote currency (e.g., 'USDT').
 * @property {string} type Market type ('spot', 'swap', 'future').
 * @property {boolean} active Is the market currently active for trading?
 * @property {boolean} is_contract Is it a derivative contract (swap/future)?
 * @property {boolean} is_linear Is it a linear contract?
 * @property {boolean} is_inverse Is it an inverse contract?
 * @property {string} contract_type_str "linear", "inverse", "spot", or "unknown".
 * @property {DecimalInstance | null} amount_precision_step_decimal Step size for order amount (quantity), as Decimal.
 * @property {DecimalInstance | null} price_precision_step_decimal Step size for order price, as Decimal.
 * @property {DecimalInstance} contract_size_decimal The value of one contract (usually 1 for linear, or base amount for inverse).
 * @property {DecimalInstance | null} min_amount_decimal Minimum order size (in base currency or contracts).
 * @property {DecimalInstance | null} max_amount_decimal Maximum order size.
 * @property {DecimalInstance | null} min_cost_decimal Minimum order cost (in quote currency).
 * @property {DecimalInstance | null} max_cost_decimal Maximum order cost.
 * @property {object} info Raw info object from ccxt `market`.
 */

/**
 * @typedef {object} PositionInfo Standardized position information derived from ccxt, with Decimal types and bot state.
 * @property {string | null} id Exchange-specific position ID (may not always be available).
 * @property {string} symbol Market symbol (e.g., 'BTC/USDT').
 * @property {'long' | 'short' | null} side Position side ('long' or 'short').
 * @property {DecimalInstance} size_decimal Position size (always positive, use 'side' for direction).
 * @property {DecimalInstance} entryPrice Average entry price of the position.
 * @property {string | null} stopLossPrice Current Stop Loss price set on the exchange (as string from API).
 * @property {string | null} takeProfitPrice Current Take Profit price set on the exchange (as string from API).
 * @property {string | null} trailingStopLoss Current Trailing Stop Loss distance set (as string from API, '0' if inactive).
 * @property {string | null} tslActivationPrice Current Trailing Stop Loss activation price set (as string from API).
 * @property {object} info Raw info object from ccxt `fetchPositions`.
 * @property {boolean} be_activated Bot state flag: Has Break-Even been applied *by the bot* during the *current active position*? (Needs careful state management if bot restarts). **Currently reset each cycle in `analyzeAndTradeSymbol`.**
 * @property {boolean} tsl_activated Bot state flag: Is Trailing Stop Loss considered active *by the bot* (based on API or bot action)? **Currently reset each cycle.**
 */

// --- Utility Functions ---
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

/** Safely converts a value to a Decimal, handling null, empty strings, and ensuring positivity if required. */
function safeMarketDecimal(value, fieldName = 'value', allowZero = true) {
    if (value == null || value === '') return null;
    try {
        const dVal = new Decimal(String(value).trim());
        if (dVal.isNaN()) {
            // logger?.warn(`safeMarketDecimal: NaN encountered for field '${fieldName}', value: '${value}'`); // Add logger if needed here
            return null;
        }
        if (!allowZero && dVal.isZero()) {
            // logger?.warn(`safeMarketDecimal: Zero value not allowed for field '${fieldName}'`);
            return null;
        }
        // Generally allow negative for things like PnL, but not for prices/sizes unless specifically needed.
        // Most core values like price steps, sizes should be positive.
        // Let's be strict by default for market constraints.
        if (['amount_precision_step_decimal', 'price_precision_step_decimal', 'contract_size_decimal', 'min_amount_decimal', 'max_amount_decimal', 'min_cost_decimal', 'max_cost_decimal', 'entryPrice', 'slPrice', 'tpPrice', 'tslDistance', 'tslActivation'].includes(fieldName)) {
             if (dVal.isNegative() && !(allowZero && dVal.isZero())) {
                 // logger?.warn(`safeMarketDecimal: Negative value invalid for field '${fieldName}': ${dVal.toFixed()}`);
                 return null;
             }
        }
        return dVal;
    } catch (e) {
        // logger?.error(`safeMarketDecimal: Error converting field '${fieldName}', value: '${value}': ${e.message}`);
        return null;
    }
}

/** Formats a price Decimal to the correct precision string for the exchange/symbol. Returns null on failure or invalid input. */
function formatPrice(exchange, symbol, price) {
    try {
        if (!(price instanceof Decimal)) {
            // logger?.warn(`formatPrice: Input price is not a Decimal instance for ${symbol}`);
            return null;
        }
        if (price.isNaN()) {
            // logger?.warn(`formatPrice: Input price is NaN for ${symbol}`);
            return null;
        }
        // Allow formatting '0' (e.g., to clear SL/TP)
        if (price.isNegative() && !price.isZero()) {
            // logger?.warn(`formatPrice: Input price ${price.toFixed()} is negative for ${symbol}`);
            return null;
        }

        // Use exchange.priceToPrecision to get the correctly formatted string
        const formattedStr = exchange.priceToPrecision(symbol, price.toNumber());

        // Optional: Double-check the formatted string can be parsed back without issues
        const checkDecimal = new Decimal(formattedStr);
        if (checkDecimal.isNaN() || (checkDecimal.isNegative() && !checkDecimal.isZero())) {
             // logger?.error(`formatPrice: Formatted string '${formattedStr}' is invalid for ${symbol}`);
             return null;
        }

        return formattedStr;
    } catch (e) {
        // logger?.error(`formatPrice: Error formatting price ${price?.toFixed()} for ${symbol}: ${e.message}`);
        return null;
    }
}

/** Extracts Bybit V5 error codes from error messages. */
function extractBybitErrorCode(error) {
    if (!error || !error.message) return null;
    // Look for common patterns like 'retCode: 12345', 'ErrCode:12345', 'error code: 12345' etc.
    const match = error.message.match(/(?:ErrCode|retCode|error code)\s*[:=]\s*(\d+)/i);
    return match ? match[1] : null;
}

// --- Logging Setup ---
const loggers = {};
function setupLogger(name) {
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_');
    const loggerName = `pyrmethus_${safeName}`;
    if (loggers[loggerName]) return loggers[loggerName];

    const logFilename = path.join(LOG_DIRECTORY, `${loggerName}.log`);
    const consoleFormat = winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.label({ label: safeName }),
        winston.format.printf(({ level, message, timestamp, label, stack, ...meta }) => {
            const ts = DateTime.fromISO(timestamp).setZone(TIMEZONE).toFormat('HH:mm:ss.SSS'); // Added milliseconds
            let color = blue;
            if (level === 'error') color = red; else if (level === 'warn') color = yellow;
            else if (level === 'info') color = cyan; else if (level === 'debug') color = gray;
            let levelString = level.toUpperCase().padEnd(8);
            if (level === 'error') levelString = bold(levelString); if (level === 'debug') levelString = dim(levelString);
            let redactedMessage = String(message);
            if (API_KEY) redactedMessage = redactedMessage.replace(new RegExp(API_KEY, 'g'), '***API_KEY***');
            if (API_SECRET) redactedMessage = redactedMessage.replace(new RegExp(API_SECRET, 'g'), '***API_SECRET***');
            const stackString = stack ? `\n${gray(stack)}` : '';
            // Only include meta if it's not empty and not just the error object itself
            const metaKeys = Object.keys(meta).filter(key => key !== 'error' && key !== 'stack' && key !== 'message' && key !== 'level' && key !== 'timestamp' && key !== 'label');
            const metaString = metaKeys.length > 0 ? ` ${gray(JSON.stringify(meta))}` : '';
            return `${blue(ts)} ${color(levelString)} ${magenta(`[${label}]`)} ${redactedMessage}${metaString}${stackString}`;
        })
    );
    const fileFormat = winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.label({ label: safeName }),
        winston.format.printf(({ level, message, timestamp, label, stack, ...meta }) => {
            let redactedMessage = String(message);
            if (API_KEY) redactedMessage = redactedMessage.replace(new RegExp(API_KEY, 'g'), '***API_KEY***');
            if (API_SECRET) redactedMessage = redactedMessage.replace(new RegExp(API_SECRET, 'g'), '***API_SECRET***');
            const stackString = stack ? `\n${stack}` : '';
            const metaKeys = Object.keys(meta).filter(key => key !== 'error' && key !== 'stack' && key !== 'message' && key !== 'level' && key !== 'timestamp' && key !== 'label');
            const metaString = metaKeys.length > 0 ? ` ${JSON.stringify(meta)}` : '';
            return `${timestamp} ${level.toUpperCase().padEnd(8)} [${label}] ${redactedMessage}${metaString}${stackString}`;
        })
    );

    const logger = winston.createLogger({
        level: process.env.LOG_LEVEL || 'info',
        format: winston.format.json(), // Default format, overridden by transports
        transports: [
            new winston.transports.Console({
                format: consoleFormat,
                handleExceptions: true, // Log uncaught exceptions
            }),
            new winston.transports.File({
                filename: logFilename,
                format: fileFormat,
                maxsize: 5 * 1024 * 1024, // 5MB
                maxFiles: 3,
                tailable: true,
                handleExceptions: true, // Log uncaught exceptions to file
            })
        ],
        exitOnError: false // Don't exit on handled exceptions
    });
    loggers[loggerName] = logger;
    return logger;
}

const initLogger = setupLogger('init');
initLogger.info(magenta(bold(`Pyrmethus Volumatic Bot v${BOT_VERSION} (JavaScript) Initializing...`)));
initLogger.info(`Using Timezone: ${TIMEZONE}`);
initLogger.info(`Detected Log Level: ${process.env.LOG_LEVEL || 'info'}`);

// --- Configuration Loading ---
let CONFIG = {};

/** Loads and validates the configuration from config.json */
async function loadConfig() {
    try {
        initLogger.info(`Loading configuration from ${CONFIG_FILE}...`);
        const configData = await fs.readFile(CONFIG_FILE, 'utf8');
        CONFIG = JSON.parse(configData);
        initLogger.info("Configuration loaded successfully.");
        ensureConfigKeys(CONFIG); // Validate essential keys
        // Set global quote currency if defined in config
        if (CONFIG.global_quote_currency) {
            QUOTE_CURRENCY = CONFIG.global_quote_currency.toUpperCase();
            initLogger.info(`Global Quote Currency set to: ${QUOTE_CURRENCY}`);
        }
        // Apply log level from config if present
        if (CONFIG.log_level && winston.config.npm.levels[CONFIG.log_level] !== undefined) {
             initLogger.level = CONFIG.log_level;
             // Update existing loggers
             Object.values(loggers).forEach(l => l.level = CONFIG.log_level);
             initLogger.info(`Log level updated to: ${CONFIG.log_level} from config.`);
        }
        return CONFIG;
    } catch (error) {
        initLogger.error(red(bold(`FATAL: Failed to load or parse ${CONFIG_FILE}: ${error.message}`)));
        if (error instanceof SyntaxError) initLogger.error(red("Hint: Check for JSON syntax errors (e.g., trailing commas)."));
        process.exit(1);
    }
}

/** Ensures essential configuration keys exist. */
function ensureConfigKeys(config) {
    const requiredTopLevel = ['symbols', 'enable_trading', 'risk_per_trade'];
    const requiredSymbolLevel = ['interval', 'leverage']; // Add more as needed
    const requiredProtection = ['enable_break_even', 'enable_trailing_stop']; // If protection object exists

    for (const key of requiredTopLevel) {
        if (config[key] === undefined) throw new Error(`Missing required top-level key in config.json: '${key}'`);
    }
    if (!Array.isArray(config.symbols) || config.symbols.length === 0) throw new Error("'symbols' must be a non-empty array in config.json");
    for (const symbolConfig of config.symbols) {
        if (typeof symbolConfig !== 'object' || symbolConfig === null) throw new Error("Each item in 'symbols' must be an object.");
        if (!symbolConfig.name || typeof symbolConfig.name !== 'string') throw new Error("Each symbol config must have a 'name' (string).");
        for (const key of requiredSymbolLevel) {
            if (symbolConfig[key] === undefined) throw new Error(`Missing required key in config for symbol '${symbolConfig.name}': '${key}'`);
        }
        if (symbolConfig.protection && typeof symbolConfig.protection === 'object') {
            for (const key of requiredProtection) {
                 if (symbolConfig.protection[key] === undefined) throw new Error(`Missing required key in 'protection' config for symbol '${symbolConfig.name}': '${key}'`);
            }
        }
    }
    initLogger.info("Essential configuration keys verified.");
}


// --- Exchange Setup ---
/** Initializes the CCXT exchange instance. */
async function initializeExchange() {
    initLogger.info("Initializing Bybit exchange connection...");
    if (!API_KEY || !API_SECRET) {
        initLogger.error(red(bold("FATAL: Bybit API Key or Secret not found in environment variables.")));
        initLogger.error(red("Ensure BYBIT_API_KEY and BYBIT_API_SECRET are set in your .env file or environment."));
        process.exit(1);
    }
    try {
        const exchange = new ccxt.bybit({
            apiKey: API_KEY,
            secret: API_SECRET,
            enableRateLimit: true,
            options: {
                defaultType: 'swap', // Or 'future', 'spot' - adjust as needed, can be overridden
                adjustForTimeDifference: true,
                // v5: { testnet: true } // Uncomment for Bybit Testnet V5
            }
        });
        // exchange.setSandboxMode(true); // Use this for the general CCXT sandbox (if supported by Bybit integration)

        // Test connection - fetchBalance is a good candidate
        initLogger.info("Testing API connection...");
        await exchange.fetchBalance({ accountType: 'UNIFIED' }); // Try fetching unified balance
        initLogger.info(green("Bybit exchange connection successful."));
        return exchange;
    } catch (e) {
        initLogger.error(red(bold(`FATAL: Failed to initialize Bybit exchange: ${e.message}`)));
        if (e instanceof ccxt.AuthenticationError) initLogger.error(red("Hint: Check API key/secret validity and permissions."));
        else if (e instanceof ccxt.NetworkError) initLogger.error(red("Hint: Check network connectivity to Bybit API."));
        else initLogger.error(red("Error details:"), e);
        process.exit(1);
    }
}


// --- CCXT Data Fetching Helpers ---

/** Fetches account balance for a specific currency with retries and error handling. */
async function fetchBalance(exchange, currency, logger) {
    logger.debug(`Fetching balance for currency: ${currency}...`);
    const upperCurrency = currency.toUpperCase();

    for (let attempt = 0; attempt <= MAX_API_RETRIES; attempt++) {
        try {
            // Bybit V5 uses accountType parameter. Try common ones.
            const accountTypes = ['UNIFIED', 'CONTRACT', undefined]; // Undefined might hit default/spot sometimes
            for (const accType of accountTypes) {
                try {
                    const params = accType ? { accountType: accType } : {};
                    logger.debug(`Fetching balance with params: ${JSON.stringify(params)} (Attempt ${attempt + 1}, AccType: ${accType || 'Default'})`);
                    const balanceInfo = await exchange.fetchBalance(params);
                    logger.debug(`Raw balance response (AccType: ${accType || 'Default'}): ${JSON.stringify(balanceInfo)}`);

                    // 1. Try standard ccxt 'free' field
                    let balanceDecimal = safeMarketDecimal(balanceInfo?.[upperCurrency]?.free, `balance_${upperCurrency}_free`, true);
                    if (balanceDecimal) {
                        logger.debug(`Found balance for ${upperCurrency} in standard 'free' field (Type: ${accType || 'Default'}): ${balanceDecimal.toFixed()}`);
                        return balanceDecimal;
                    }

                    // 2. Try Bybit V5 structure (info.result.list)
                    const bybitList = balanceInfo?.info?.result?.list;
                    if (Array.isArray(bybitList)) {
                        const coinEntry = bybitList.find(entry => entry?.coin?.toUpperCase() === upperCurrency);
                        if (coinEntry) {
                            // Prefer 'availableToWithdraw' or 'walletBalance' ? 'availableToWithdraw' seems safer for available equity.
                            const availableBalance = coinEntry.availableToWithdraw ?? coinEntry.availableBalance ?? coinEntry.walletBalance;
                            balanceDecimal = safeMarketDecimal(availableBalance, `balance_${upperCurrency}_bybit`, true);
                            if (balanceDecimal) {
                                logger.debug(`Found balance for ${upperCurrency} in Bybit V5 list (Type: ${accType}): ${balanceDecimal.toFixed()}`);
                                return balanceDecimal;
                            }
                        }
                    }

                    // 3. Try ccxt 'total' field as a fallback (less ideal for available margin)
                    balanceDecimal = safeMarketDecimal(balanceInfo?.[upperCurrency]?.total, `balance_${upperCurrency}_total`, true);
                    if (balanceDecimal) {
                        logger.debug(`Found balance for ${upperCurrency} in standard 'total' field (Type: ${accType || 'Default'}): ${balanceDecimal.toFixed()}`);
                        return balanceDecimal; // Use total if free/available isn't found
                    }

                } catch (innerError) {
                    const bybitCode = extractBybitErrorCode(innerError);
                    // Ignore errors related to invalid account types and try the next one
                    if (accType && (bybitCode === '10001' || /account type.*?exist|invalid account type/i.test(innerError.message))) {
                        logger.debug(`Account type '${accType}' might not be applicable or exist. Trying next... (Code: ${bybitCode})`);
                    } else {
                        throw innerError; // Re-throw other errors
                    }
                }
            }
            // If loop finishes without finding the currency
            throw new Error(`Balance for currency '${upperCurrency}' not found in any checked account type or structure.`);

        } catch (e) {
            const bybitCode = extractBybitErrorCode(e);
            logger.warn(yellow(`Fetch balance attempt ${attempt + 1} failed: ${e.message} ${bybitCode ? `(Code: ${bybitCode})` : ''}`));
            if (e instanceof ccxt.AuthenticationError) { logger.error(red("Authentication error fetching balance. Check API keys.")); return null; }
            if (e instanceof ccxt.RateLimitExceeded) { const wait = RETRY_DELAY_SECONDS * 2 * 1000; logger.warn(yellow(`Rate limit hit. Waiting ${wait / 1000}s...`)); await delay(wait); continue; }
            if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeError) {
                if (attempt >= MAX_API_RETRIES) { logger.error(red(`Max retries exceeded fetching balance for ${upperCurrency}.`)); break; } // Break loop
                await delay(RETRY_DELAY_SECONDS * (attempt + 1) * 1000);
            } else {
                logger.error(red(`Non-retryable error fetching balance: ${e.message}`), { error: e });
                return null; // Non-retryable error
            }
        }
    }
    logger.error(red(`Failed to fetch balance for ${upperCurrency} after all attempts.`));
    return null; // Return null if all retries fail
}

/** Retrieves and caches market information with retries, converting precision/limits to Decimal. */
async function getMarketInfo(exchange, symbol, logger) {
    if (marketInfoCache.has(symbol)) {
        // logger.debug(`Using cached market info for ${symbol}.`);
        return marketInfoCache.get(symbol);
    }
    logger.debug(`Fetching market details for symbol: ${symbol}...`);

    for (let attempt = 0; attempt <= MAX_API_RETRIES; attempt++) {
        try {
            let market = null;
            let forceReload = false;
            // Check if markets are loaded and contain the symbol
            if (!exchange.markets || !exchange.markets[symbol]) {
                logger.info(`Market '${symbol}' not loaded or map is empty. Forcing market map refresh...`);
                forceReload = true;
            }
            // Force reload on retries or if initially missing
            if (forceReload || attempt > 0) {
                logger.info(`Refreshing market map (Attempt ${attempt + 1})...`);
                await exchange.loadMarkets(true); // Force reload
                marketInfoCache.clear(); // Clear cache after reload
                logger.info("Market map refreshed.");
            }
            // Try getting the market again
            market = exchange.market(symbol);
            if (!market) {
                throw new ccxt.BadSymbol(`Market ${symbol} not found after reload.`);
            }

            // --- Parse Market Info ---
            const infoRaw = market.info || {};
            const type = market.type; // 'spot', 'swap', 'future'
            const isContract = market.contract || type === 'swap' || type === 'future';
            const isLinear = market.linear || (infoRaw.contractType === 'LinearPerpetual');
            const isInverse = market.inverse || (infoRaw.contractType === 'InversePerpetual') || (infoRaw.contractType === 'InverseFutures');
            let contractTypeStr = 'unknown';
            if (isLinear) contractTypeStr = 'linear';
            else if (isInverse) contractTypeStr = 'inverse';
            else if (type === 'spot') contractTypeStr = 'spot';

            // Use safeMarketDecimal for robust parsing
            const amountStep = safeMarketDecimal(market.precision?.amount, 'amount_precision_step_decimal', false); // Amount step cannot be zero
            const priceStep = safeMarketDecimal(market.precision?.price, 'price_precision_step_decimal', false); // Price step cannot be zero
            const contractSize = safeMarketDecimal(market.contractSize, 'contract_size_decimal', false); // Contract size cannot be zero
            const minAmount = safeMarketDecimal(market.limits?.amount?.min, 'min_amount_decimal', true);
            const maxAmount = safeMarketDecimal(market.limits?.amount?.max, 'max_amount_decimal', true);
            const minCost = safeMarketDecimal(market.limits?.cost?.min, 'min_cost_decimal', true);
            const maxCost = safeMarketDecimal(market.limits?.cost?.max, 'max_cost_decimal', true);

            // Bybit V5 specific overrides/checks if standard fields are missing
            let finalAmountStep = amountStep;
            if (!finalAmountStep && infoRaw.lotSizeFilter?.qtyStep) {
                 finalAmountStep = safeMarketDecimal(infoRaw.lotSizeFilter.qtyStep, 'amount_precision_step_decimal', false);
                 if (finalAmountStep) logger.debug(`Used Bybit V5 qtyStep for amount precision (${symbol}).`);
            }
            let finalPriceStep = priceStep;
            if (!finalPriceStep && infoRaw.priceFilter?.tickSize) {
                 finalPriceStep = safeMarketDecimal(infoRaw.priceFilter.tickSize, 'price_precision_step_decimal', false);
                 if (finalPriceStep) logger.debug(`Used Bybit V5 tickSize for price precision (${symbol}).`);
            }
            let finalMinAmount = minAmount;
            if (!finalMinAmount && infoRaw.lotSizeFilter?.minOrderQty) {
                 finalMinAmount = safeMarketDecimal(infoRaw.lotSizeFilter.minOrderQty, 'min_amount_decimal', true);
                 if (finalMinAmount) logger.debug(`Used Bybit V5 minOrderQty for min amount (${symbol}).`);
            }
             let finalMaxAmount = maxAmount;
            if (!finalMaxAmount && infoRaw.lotSizeFilter?.maxOrderQty) {
                 finalMaxAmount = safeMarketDecimal(infoRaw.lotSizeFilter.maxOrderQty, 'max_amount_decimal', true);
                 if (finalMaxAmount) logger.debug(`Used Bybit V5 maxOrderQty for max amount (${symbol}).`);
            }
            // Bybit V5 min cost? Sometimes 'minOrderIv' - needs verification if it maps to cost
            // let finalMinCost = minCost; // Example if needed


            const processedInfo = {
                id: market.id,
                symbol: market.symbol,
                base: market.base,
                quote: market.quote,
                type: type,
                active: market.active ?? (infoRaw.status === 'Trading'), // Check Bybit status if ccxt field missing
                is_contract: isContract,
                is_linear: isLinear,
                is_inverse: isInverse,
                contract_type_str: contractTypeStr,
                amount_precision_step_decimal: finalAmountStep,
                price_precision_step_decimal: finalPriceStep,
                contract_size_decimal: contractSize || new Decimal(1), // Default to 1 if missing (common for linear)
                min_amount_decimal: finalMinAmount,
                max_amount_decimal: finalMaxAmount,
                min_cost_decimal: minCost, // Keep standard ccxt for now, adjust if Bybit V5 source found
                max_cost_decimal: maxCost,
                info: infoRaw
            };

            // --- Validation ---
            if (!processedInfo.amount_precision_step_decimal || !processedInfo.price_precision_step_decimal) {
                throw new Error(`Market ${symbol} missing critical precision info (AmountStep or PriceStep) after parsing.`);
            }
            if (!processedInfo.contract_size_decimal) {
                 logger.warn(yellow(`Market ${symbol} missing contract size. Defaulting to 1. This might be incorrect for Inverse contracts.`));
                 processedInfo.contract_size_decimal = new Decimal(1);
            }
            if (!processedInfo.active) {
                logger.warn(yellow(`Market ${symbol} is marked as inactive.`));
                // Decide whether to proceed or fail based on requirements
            }
            // Update global quote currency if needed (e.g., for the first symbol processed)
             if (processedInfo.quote && QUOTE_CURRENCY !== processedInfo.quote.toUpperCase()) {
                 logger.info(`Updating global QUOTE_CURRENCY based on ${symbol} market info to: ${processedInfo.quote.toUpperCase()}`);
                 QUOTE_CURRENCY = processedInfo.quote.toUpperCase();
             }

            logger.debug(`Market info processed successfully for ${symbol}. Type: ${processedInfo.contract_type_str}`);
            marketInfoCache.set(symbol, processedInfo);
            return processedInfo;

        } catch (e) {
            const bybitCode = extractBybitErrorCode(e);
            logger.warn(yellow(`Get market info attempt ${attempt + 1} failed: ${e.message} ${bybitCode ? `(Code: ${bybitCode})` : ''}`));
            if (e instanceof ccxt.BadSymbol) { logger.error(red(`Market symbol '${symbol}' is invalid or not supported by Bybit.`)); return null; }
            if (e instanceof ccxt.AuthenticationError) { logger.error(red("Authentication error fetching market info.")); return null; }
            if (e instanceof ccxt.RateLimitExceeded) { const wait = RETRY_DELAY_SECONDS * 2 * 1000; logger.warn(yellow(`Rate limit hit. Waiting ${wait / 1000}s...`)); await delay(wait); continue; }
            if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeError) {
                if (attempt >= MAX_API_RETRIES) { logger.error(red(`Max retries exceeded fetching market info for ${symbol}.`)); break; }
                await delay(RETRY_DELAY_SECONDS * (attempt + 1) * 1000);
            } else {
                logger.error(red(`Non-retryable error fetching market info: ${e.message}`), { error: e });
                return null; // Non-retryable error
            }
        }
    }
    logger.error(red(`Failed to retrieve market info for ${symbol} after all retries.`));
    return null;
}

/** Fetches OHLCV data with multi-request logic, retries, and converts to Danfo DataFrame. */
async function fetchKlinesCcxt(exchange, symbol, timeframe, limit, logger) {
    const ccxtTimeframe = CCXT_INTERVAL_MAP[timeframe] || timeframe; // Allow passing '1h' etc. directly
    logger.info(cyan(`Fetching klines for ${symbol} | TF: ${ccxtTimeframe} | Limit: ${limit}`));

    const marketInfo = await getMarketInfo(exchange, symbol, logger);
    if (!marketInfo) return null; // Cannot fetch without market info

    const marketId = marketInfo.id; // Use exchange-specific ID
    const category = marketInfo.contract_type_str.toLowerCase(); // 'linear', 'inverse', 'spot'

    let allKlines = [];
    let remainingLimit = limit;
    let endTime = undefined; // Start with latest
    const maxPerRequest = BYBIT_API_KLINE_LIMIT; // Exchange limit per request

    logger.debug(`Starting kline fetch loop. Total needed: ${limit}, Max per request: ${maxPerRequest}`);

    while (remainingLimit > 0) {
        const currentLimit = Math.min(remainingLimit, maxPerRequest);
        logger.debug(`Fetching segment: Limit=${currentLimit}, EndTime=${endTime ? new Date(endTime).toISOString() : 'Latest'}`);

        let fetchedSegment = null;
        for (let attempt = 0; attempt <= MAX_API_RETRIES; attempt++) {
            try {
                const params = { category };
                if (endTime) params.until = endTime; // Bybit uses 'until' for end timestamp (exclusive)

                fetchedSegment = await exchange.fetchOHLCV(marketId, ccxtTimeframe, undefined, currentLimit, params); // 'since' is undefined to get latest

                if (fetchedSegment && fetchedSegment.length > 0) {
                    logger.debug(`Fetched ${fetchedSegment.length} candles in this segment.`);
                    break; // Success
                } else {
                    logger.debug("Fetched segment is empty or null. Assuming no more data in this range.");
                    fetchedSegment = []; // Ensure it's an array
                    break; // Stop fetching if no data returned
                }
            } catch (e) {
                const bybitCode = extractBybitErrorCode(e);
                logger.warn(yellow(`Kline fetch attempt ${attempt + 1} failed: ${e.message} ${bybitCode ? `(Code: ${bybitCode})` : ''}`));
                if (e instanceof ccxt.AuthenticationError) { logger.error(red("Auth error fetching klines.")); return null; }
                if (e instanceof ccxt.RateLimitExceeded) { const wait = RETRY_DELAY_SECONDS * 3 * 1000; logger.warn(yellow(`Rate limit hit. Waiting ${wait / 1000}s...`)); await delay(wait); continue; }
                if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeError) {
                     if (attempt >= MAX_API_RETRIES) { logger.error(red(`Max retries exceeded fetching klines for ${symbol}.`)); fetchedSegment = null; break; } // Break inner loop
                     await delay(RETRY_DELAY_SECONDS * (attempt + 1) * 1000);
                } else { logger.error(red(`Non-retryable error fetching klines: ${e.message}`), { error: e }); return null; }
            }
        } // End retry loop

        if (fetchedSegment === null) {
            logger.error(red("Failed to fetch a kline segment after retries. Returning potentially incomplete data."));
            break; // Exit outer loop if a segment fails completely
        }
        if (fetchedSegment.length === 0) {
            logger.debug("No more klines returned by API. Stopping fetch.");
            break; // Exit outer loop if API returns empty
        }

        allKlines = fetchedSegment.concat(allKlines); // Prepend older data
        remainingLimit -= fetchedSegment.length;
        endTime = fetchedSegment[0][0] - 1; // Set end time for next request to before the first candle of this batch

        logger.debug(`Total klines collected: ${allKlines.length}, Remaining needed: ${remainingLimit}, Next EndTime: ${new Date(endTime).toISOString()}`);

        if (fetchedSegment.length < currentLimit) {
            logger.debug("API returned fewer candles than requested. Assuming reached end of available history.");
            break; // Stop if API returns less than asked, likely hit history limit
        }
        if (remainingLimit <= 0) break; // Exit if we have enough
        await delay(500); // Small delay between fetches to be kind to the API
    } // End while loop

    logger.info(`Finished fetching klines. Total collected: ${allKlines.length} for ${symbol}.`);
    if (allKlines.length === 0) { logger.warn(yellow(`No kline data returned for ${symbol}.`)); return null; }

    // --- Convert to Danfo DataFrame ---
    try {
        // Ensure unique timestamps (sometimes duplicates can occur)
        const uniqueKlinesMap = new Map();
        allKlines.forEach(k => uniqueKlinesMap.set(k[0], k));
        const uniqueKlines = Array.from(uniqueKlinesMap.values());
        uniqueKlines.sort((a, b) => a[0] - b[0]); // Sort by timestamp ascending

        const data = uniqueKlines.map(k => ({
            timestamp: k[0],
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5]) || 0 // Handle potential null/undefined volume
        }));

        let df = new dfd.DataFrame(data);
        df.setIndex({ column: "timestamp", inplace: true, drop: true });
        // Convert columns to appropriate types (Danfo often defaults to float64)
        df = df.astype({ column: "open", dtype: "float32" });
        df = df.astype({ column: "high", dtype: "float32" });
        df = df.astype({ column: "low", dtype: "float32" });
        df = df.astype({ column: "close", dtype: "float32" });
        df = df.astype({ column: "volume", dtype: "float32" });

        // Trim DataFrame if it exceeds max length
        if (df.shape[0] > MAX_DF_LEN) {
            logger.debug(`Trimming DataFrame from ${df.shape[0]} to ${MAX_DF_LEN} rows.`);
            df = df.tail(MAX_DF_LEN); // Keep the most recent data
        }

        logger.debug(`Created DataFrame for ${symbol}. Shape: [${df.shape[0]}, ${df.shape[1]}]. Index type: ${df.index[0] ? typeof df.index[0] : 'N/A'}`);
        return df;
    } catch (e) {
        logger.error(red(`Error creating/processing DataFrame for ${symbol}: ${e.message}`), { error: e });
        return null;
    }
}


/** Retrieves open position details using fetchPositions, standardizes the output, and uses Decimal.js. */
async function getOpenPosition(exchange, symbol, logger) {
    logger.debug(`Checking for open position for symbol: ${symbol}...`);
    const marketInfo = await getMarketInfo(exchange, symbol, logger);
    if (!marketInfo) { logger.error(red(`Cannot get position for ${symbol}: Failed to get market info.`)); return null; }

    // Position checks only make sense for contracts
    if (!marketInfo.is_contract) {
        logger.debug(`Position check skipped for ${symbol}: Not a contract market.`);
        return null;
    }

    const category = marketInfo.contract_type_str.toLowerCase(); // 'linear' or 'inverse'
    const marketId = marketInfo.id; // Use exchange-specific ID

    for (let attempt = 0; attempt <= MAX_API_RETRIES; attempt++) {
        try {
            // Bybit V5 requires category and often symbol for fetchPositions
            const params = { category: category, symbol: marketId };
            logger.debug(`Fetching positions with params: ${JSON.stringify(params)} (Attempt ${attempt + 1})`);

            // fetchPositions might return positions for *all* symbols in the category if symbol isn't specified or fully effective.
            // We need to filter specifically for the market we're interested in.
            const allPositionsRaw = await exchange.fetchPositions(undefined, params); // Pass undefined for symbols list to use params
            logger.debug(`Fetched ${allPositionsRaw.length} total positions for category ${category}. Raw: ${JSON.stringify(allPositionsRaw)}`);

            // Filter based on symbol AND ensuring size is non-zero (within precision)
            const relevantPositions = allPositionsRaw.filter(p =>
                (p.symbol === symbol || p.info?.symbol === marketId) // Match standard symbol or specific ID
            );
            logger.debug(`Filtered down to ${relevantPositions.length} positions matching symbol ${symbol}/${marketId}.`);

            if (relevantPositions.length === 0) {
                logger.info(`No position entry found for ${symbol}.`);
                return null;
            }

            // Find the position with a significant size. Bybit might return entries with size '0' if recently closed.
            let activePositionRaw = null;
            const sizeThreshold = marketInfo.amount_precision_step_decimal ? marketInfo.amount_precision_step_decimal.mul('0.001') : new Decimal('1e-9'); // A small fraction of the step size

            for (const pos of relevantPositions) {
                // Determine size: check info.size, then contracts, then contractSize
                const sizeStr = pos.info?.size ?? pos.contracts ?? pos.contractSize ?? '0';
                const sizeDecimal = safeMarketDecimal(sizeStr, 'pos_size_raw', true); // Allow zero temporarily

                if (sizeDecimal && sizeDecimal.abs().gt(sizeThreshold)) {
                    // Found a position with a non-negligible size
                    activePositionRaw = pos;
                    activePositionRaw.parsed_size_decimal = sizeDecimal.abs(); // Store absolute size
                    logger.debug(`Found candidate active position for ${symbol}: Size=${sizeDecimal.toFixed()}`);
                    break; // Take the first one found with size > threshold
                } else {
                     logger.debug(`Skipping position entry for ${symbol}: Size ${sizeDecimal?.toFixed() ?? 'N/A'} is zero or below threshold ${sizeThreshold.toFixed()}.`);
                }
            }

            if (activePositionRaw) {
                const info = activePositionRaw.info || {};
                const sizeDecimal = activePositionRaw.parsed_size_decimal; // Absolute size

                // Determine side: Check info.side first, then ccxt standardized side, fallback to size sign if needed
                let side = info.side?.toLowerCase(); // Bybit: 'Buy' -> long, 'Sell' -> short
                if (side === 'buy') side = 'long';
                else if (side === 'sell') side = 'short';
                else if (activePositionRaw.side?.toLowerCase() === 'long' || activePositionRaw.side?.toLowerCase() === 'short') {
                    side = activePositionRaw.side.toLowerCase();
                     logger.debug(`Used ccxt standardized side: ${side}`);
                } else {
                    // Fallback: infer from unrealized PnL sign? Risky. Let's rely on Bybit's 'side'.
                    logger.warn(yellow(`Could not reliably determine position side for ${symbol} from info.side ('${info.side}') or ccxt side ('${activePositionRaw.side}'). Attempting best guess.`));
                     // If Bybit V5 info.side is None or empty string, it might imply flat.
                     if (!info.side || info.side === 'None') {
                         logger.info(`Position side for ${symbol} appears to be flat based on info.side.`);
                         return null;
                     }
                     // If still undetermined, maybe log an error or return null
                     logger.error(red(`CRITICAL: Unable to determine position side for ${symbol}. Raw info: ${JSON.stringify(info)}`));
                     return null;
                }

                if (side !== 'long' && side !== 'short') {
                     logger.error(red(`Determined invalid side '${side}' for active position ${symbol}.`));
                     return null;
                }

                // Determine entry price
                const entryPrice = safeMarketDecimal(info.avgPrice ?? activePositionRaw.entryPrice, 'entryPrice', false);
                if (!entryPrice) {
                    logger.error(red(`Could not parse entry price for active position ${symbol}. Raw: ${JSON.stringify(info)}`));
                    return null;
                }

                // Extract protection levels (these are strings from the API)
                const getProtectionField = (fieldName) => info[fieldName] !== undefined && String(info[fieldName]).trim() !== '' && String(info[fieldName]).trim() !== '0' ? String(info[fieldName]).trim() : null;
                const slPriceStr = getProtectionField('stopLoss');
                const tpPriceStr = getProtectionField('takeProfit');
                const tslDistStr = getProtectionField('trailingStop'); // Bybit uses 'trailingStop' for the distance/value
                const tslActPriceStr = getProtectionField('activePrice'); // Bybit uses 'activePrice' for the activation price

                // Construct the standardized PositionInfo object
                const positionResult = {
                    id: info.positionIdx ?? activePositionRaw.id, // Use Bybit positionIdx if available
                    symbol: symbol,
                    side: side,
                    size_decimal: sizeDecimal,
                    entryPrice: entryPrice,
                    stopLossPrice: slPriceStr,
                    takeProfitPrice: tpPriceStr,
                    trailingStopLoss: tslDistStr,
                    tslActivationPrice: tslActPriceStr,
                    info: info,
                    // Reset bot state flags each time. Persistent state needs external storage.
                    be_activated: false, // Reset: Has BE been applied *this cycle*?
                    tsl_activated: !!(tslDistStr && tslDistStr !== '0'), // Reset: Is TSL currently set according to API?
                };

                logger.info(green(bold(`Active ${side.toUpperCase()} Position Found (${symbol}): `)) +
                            `Size=${sizeDecimal.toFixed()}, Entry=${entryPrice.toFixed()}, SL=${slPriceStr ?? 'N/A'}, TP=${tpPriceStr ?? 'N/A'}, TSL=${tslDistStr ?? 'N/A'}`);
                logger.debug(`Full position details (${symbol}): ${JSON.stringify(positionResult)}`);
                return positionResult;

            } else {
                logger.info(`No active position found for ${symbol} (checked ${relevantPositions.length} entries).`);
                return null;
            }

        } catch (e) {
            const bybitCode = extractBybitErrorCode(e);
            logger.warn(yellow(`Get position info attempt ${attempt + 1} failed: ${e.message} ${bybitCode ? `(Code: ${bybitCode})` : ''}`));
            if (bybitCode === '110021') { logger.info(`Order/position not found (Code: 110021), likely no open position for ${symbol}.`); return null; }
            if (e instanceof ccxt.AuthenticationError) { logger.error(red("Auth error fetching positions.")); return null; }
            if (e instanceof ccxt.RateLimitExceeded) { const wait = RETRY_DELAY_SECONDS * 2 * 1000; logger.warn(yellow(`Rate limit hit. Waiting ${wait / 1000}s...`)); await delay(wait); continue; }
            if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeError) {
                if (attempt >= MAX_API_RETRIES) { logger.error(red(`Max retries exceeded fetching position info for ${symbol}.`)); break; }
                await delay(RETRY_DELAY_SECONDS * (attempt + 1) * 1000);
            } else {
                logger.error(red(`Non-retryable error fetching positions: ${e.message}`), { error: e });
                return null; // Non-retryable error
            }
        }
    }
    logger.error(red(`Failed to get position info for ${symbol} after all retries.`));
    return null;
}


/** Sets leverage for a derivatives symbol using Bybit V5 parameters with retries. */
async function setLeverageCcxt(exchange, symbol, leverage, marketInfo, logger) {
    if (!marketInfo.is_contract) {
        logger.info(`Leverage setting skipped for ${symbol}: Not a contract market.`);
        return true; // Consider success as no action needed
    }
    if (!Number.isInteger(leverage) || leverage <= 0) {
        logger.warn(yellow(`Leverage setting skipped for ${symbol}: Invalid leverage value ${leverage}. Must be a positive integer.`));
        return false;
    }
    // CCXT's unified setLeverage often requires specific params for Bybit V5
    if (!exchange.has['setLeverage']) {
        logger.error(red(`Exchange ${exchange.id} does not support setLeverage via ccxt unified method.`));
        // Potentially implement direct privatePost call here if needed
        return false;
    }

    const marketId = marketInfo.id;
    const category = marketInfo.contract_type_str.toLowerCase();
    if (category === 'spot' || category === 'unknown') {
        logger.warn(yellow(`Leverage setting skipped: Invalid category '${category}' for ${symbol}.`));
        return false;
    }

    logger.info(`Attempting to set leverage for ${marketId} (${category}) to ${leverage}x...`);
    for (let attempt = 0; attempt <= MAX_API_RETRIES; attempt++) {
        try {
            // Bybit V5 requires category, symbol, buyLeverage, sellLeverage
            const params = {
                category: category,
                symbol: marketId,
                buyLeverage: String(leverage), // Must be strings
                sellLeverage: String(leverage)
            };
            logger.debug(`Calling exchange.setLeverage with leverage=${leverage}, symbol=${marketId}, params=${JSON.stringify(params)} (Attempt ${attempt + 1})`);

            // Note: ccxt's setLeverage might internally call /v5/position/set-leverage
            const response = await exchange.setLeverage(leverage, marketId, params);

            // Response structure for Bybit V5 via ccxt might be nested under 'info'
            logger.debug(`Raw setLeverage response (${symbol}): ${JSON.stringify(response)}`);
            const info = response?.info;
            const retCode = info?.retCode ?? response?.retCode; // Check info first, then top level
            const retMsg = info?.retMsg ?? response?.retMsg ?? 'Unknown Bybit message';
            const retCodeStr = retCode != null ? String(retCode) : null;

            // Check Bybit success codes
            if (retCodeStr === '0') {
                logger.info(green(`Leverage successfully set for ${marketId} to ${leverage}x (Code: 0).`));
                return true;
            }
            // Bybit code meaning "Leverage not modified" - treat as success
            if (retCodeStr === '110045' || retMsg.includes("Leverage not modified")) {
                logger.info(yellow(`Leverage already set to ${leverage}x for ${marketId} (Code: ${retCodeStr}). Considered success.`));
                return true;
            }

            // If we got here, it wasn't a known success code
            throw new ccxt.ExchangeError(`Bybit API error setting leverage (${symbol}): ${retMsg} (Code: ${retCodeStr ?? 'N/A'})`);

        } catch (e) {
            const bybitCode = extractBybitErrorCode(e);
            logger.warn(yellow(`Set leverage attempt ${attempt + 1} failed: ${e.message} ${bybitCode ? `(Code: ${bybitCode})` : ''}`));

            if (e instanceof ccxt.AuthenticationError) { logger.error(red("Authentication error setting leverage. Check API keys.")); return false; }

            // Check for non-retryable Bybit codes (add more as identified)
            // 10001: Parameter error (likely invalid leverage value for risk limit)
            // 110009: Position is in reduce-only status
            // 110013: Margin mode related error
            // 110028: Cross/isolated margin mode error
            // 110043: Risk limit related error (leverage too high for current tier)
            // 110044: Leverage cannot be lower than current risk limit requirement
            // 110055: Cannot modify leverage with active order(s)
            // 3400045: Invalid leverage value (e.g., exceeds max allowed)
            // 3400084: Cannot change leverage with open position (though Bybit sometimes allows this?)
            const fatalCodes = ['10001', '110009', '110013', '110028', '110043', '110044', '110055', '3400045', '3400084'];
            const fatalMessages = [/margin mode/i, /position exists/i, /risk limit/i, /parameter error/i, /insufficient available balance/i, /invalid leverage/i, /active order/i];

            if ((bybitCode && fatalCodes.includes(bybitCode)) || fatalMessages.some(rx => rx.test(e.message))) {
                logger.error(red(`>> Hint: NON-RETRYABLE leverage error (${symbol}). Code: ${bybitCode}. Message: ${e.message}`));
                logger.error(red(`>> Action: Manual intervention may be required (check risk limits, open orders, position status, margin mode).`));
                return false; // Abort on these errors
            }

            if (e instanceof ccxt.RateLimitExceeded) { const wait = RETRY_DELAY_SECONDS * 3 * 1000; logger.warn(yellow(`Rate limit hit. Waiting ${wait / 1000}s...`)); await delay(wait); continue; }

            // Retry generic network/exchange errors
            if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeError) {
                if (attempt >= MAX_API_RETRIES) { logger.error(red(`Max retries exceeded setting leverage for ${symbol}.`)); break; } // Break loop
                await delay(RETRY_DELAY_SECONDS * (attempt + 1) * 1000);
            } else {
                // Unexpected error type
                logger.error(red(`Unexpected non-retryable error setting leverage: ${e.message}`), { error: e });
                return false;
            }
        }
    } // End retry loop
    logger.error(red(`Failed to set leverage for ${marketId} to ${leverage}x after all attempts.`));
    return false;
}


/** Calculates position size based on risk, SL, entry, balance, and market constraints using Decimal.js. */
async function calculatePositionSize(balance, riskPerTrade, slPrice, entryPrice, marketInfo, exchange, logger) {
    logger.info(bold(`--- Position Sizing Calculation (${marketInfo.symbol}) ---`));
    try {
        // --- Input Validation ---
        if (!balance || !(balance instanceof Decimal) || balance.lte(0)) {
            throw new Error(`Invalid or non-positive balance provided: ${balance?.toFixed()}`);
        }
        const riskDecimal = safeMarketDecimal(riskPerTrade, 'risk_per_trade', false); // Cannot be zero
        if (!riskDecimal || riskDecimal.lte(0) || riskDecimal.gt(1)) {
            throw new Error(`Invalid risk_per_trade: ${riskPerTrade}. Must be > 0 and <= 1.`);
        }
        if (!slPrice || !(slPrice instanceof Decimal) || slPrice.lte(0)) {
            throw new Error(`Invalid or non-positive SL price: ${slPrice?.toFixed()}`);
        }
        if (!entryPrice || !(entryPrice instanceof Decimal) || entryPrice.lte(0)) {
            throw new Error(`Invalid or non-positive entry price: ${entryPrice?.toFixed()}`);
        }
        if (slPrice.eq(entryPrice)) {
            throw new Error("SL price cannot be equal to entry price.");
        }

        // --- Market Constraints (ensure they are Decimals) ---
        const amountStep = marketInfo.amount_precision_step_decimal;
        const priceStep = marketInfo.price_precision_step_decimal; // Needed for cost checks later
        const minAmount = marketInfo.min_amount_decimal ?? new Decimal(0);
        const maxAmount = marketInfo.max_amount_decimal; // Can be null/Infinity
        const minCost = marketInfo.min_cost_decimal ?? new Decimal(0);
        // const maxCost = marketInfo.max_cost_decimal; // Max cost check is less common, implement if needed
        const contractSize = marketInfo.contract_size_decimal;

        if (!amountStep || amountStep.lte(0)) throw new Error("Invalid market amount step size.");
        if (!priceStep || priceStep.lte(0)) throw new Error("Invalid market price step size.");
        if (!contractSize || contractSize.lte(0)) throw new Error("Invalid market contract size.");

        logger.debug(`  Balance: ${balance.toFixed()} ${marketInfo.quote}`);
        logger.debug(`  Risk Per Trade: ${riskDecimal.toSignificantDigits(4)} (${riskDecimal.mul(100).toFixed(2)}%)`);
        logger.debug(`  Entry Price: ${entryPrice.toFixed()}`);
        logger.debug(`  Stop Loss Price: ${slPrice.toFixed()}`);
        logger.debug(`  Market: ${marketInfo.symbol}, Type: ${marketInfo.contract_type_str}`);
        logger.debug(`  Constraints: AmtStep=${amountStep.toFixed()}, MinAmt=${minAmount.toFixed()}, MaxAmt=${maxAmount?.toFixed() ?? 'None'}, ContrSize=${contractSize.toFixed()}, MinCost=${minCost.toFixed()}`);

        // --- Risk Amount in Quote Currency ---
        const riskAmountQuote = balance.mul(riskDecimal);
        logger.info(`  Max Risk Amount (Quote): ${riskAmountQuote.toFixed(8)} ${marketInfo.quote}`);

        // --- Stop Loss Distance ---
        const stopLossDistance = entryPrice.sub(slPrice).abs();
        if (stopLossDistance.lte(0)) {
            throw new Error("Stop loss distance is zero or negative (SL == Entry).");
        }
        logger.debug(`  Stop Loss Distance (Price): ${stopLossDistance.toFixed()}`);

        // --- Calculate Size based on Contract Type ---
        let calculatedSizeBase; // Size in contracts (for derivatives) or base currency (for spot)

        if (marketInfo.is_contract) { // Linear or Inverse Futures/Swaps
            if (marketInfo.is_linear) {
                // Risk per contract = SL distance * contract size (value change per contract)
                const valueChangePerContract = stopLossDistance.mul(contractSize);
                if (valueChangePerContract.abs().lt('1e-18')) throw new Error("Linear contract value change per contract is near zero.");
                calculatedSizeBase = riskAmountQuote.div(valueChangePerContract);
                logger.debug(`  Linear Calc: RiskQuote ${riskAmountQuote.toFixed(8)} / ValChangePerContract ${valueChangePerContract.toFixed()} = ${calculatedSizeBase.toFixed()} Contracts`);
            } else { // Inverse
                // Need entry and SL prices > 0 for inverse calc
                if (entryPrice.lte(0) || slPrice.lte(0)) throw new Error("Inverse calculation requires positive entry/SL prices.");
                // Risk per contract = contract size * |(1/SL) - (1/Entry)|
                const inverseFactor = (new Decimal(1).div(slPrice)).sub(new Decimal(1).div(entryPrice)).abs();
                if (inverseFactor.abs().lt('1e-18')) throw new Error("Inverse factor |1/SL - 1/Entry| is near zero.");
                const riskPerContract = contractSize.mul(inverseFactor);
                if (riskPerContract.abs().lt('1e-18')) throw new Error("Inverse risk per contract is near zero.");
                calculatedSizeBase = riskAmountQuote.div(riskPerContract);
                logger.debug(`  Inverse Calc: RiskQuote ${riskAmountQuote.toFixed(8)} / RiskPerContract ${riskPerContract.toFixed()} = ${calculatedSizeBase.toFixed()} Contracts`);
            }
        } else { // Spot
             // Risk per unit of base currency = SL distance
             // Size (Base) = Risk Amount (Quote) / SL Distance (Quote/Base)
            calculatedSizeBase = riskAmountQuote.div(stopLossDistance);
            logger.debug(`  Spot Calc: RiskQuote ${riskAmountQuote.toFixed(8)} / SL_Distance ${stopLossDistance.toFixed()} = ${calculatedSizeBase.toFixed()} ${marketInfo.base}`);
        }

        if (calculatedSizeBase.lte(0) || !calculatedSizeBase.isFinite()) {
            throw new Error(`Initial calculated size is invalid: ${calculatedSizeBase.toFixed()}`);
        }
        logger.info(`  Initial Calculated Size = ${calculatedSizeBase.toFixed()} ${marketInfo.is_contract ? 'Contracts' : marketInfo.base}`);

        // --- Apply Market Limits and Precision ---
        let adjustedSize = calculatedSizeBase;

        // 1. Min/Max Amount Limits
        if (adjustedSize.lt(minAmount)) {
            logger.warn(yellow(`Calculated size ${adjustedSize.toFixed()} is below minimum ${minAmount.toFixed()}. Adjusting UP to minimum.`));
            adjustedSize = minAmount;
        }
        if (maxAmount && adjustedSize.gt(maxAmount)) {
            logger.warn(yellow(`Calculated size ${adjustedSize.toFixed()} exceeds maximum ${maxAmount.toFixed()}. Adjusting DOWN to maximum.`));
            adjustedSize = maxAmount;
        }
        logger.debug(`  Size after Amount Limits: ${adjustedSize.toFixed()}`);

        // 2. Amount Precision (Rounding DOWN to be conservative and ensure it's placeable)
        // size = floor(size / step) * step
        const finalSize = adjustedSize.div(amountStep).floor().mul(amountStep);
        if (!finalSize.eq(adjustedSize)) {
            logger.info(`Applied amount precision step ${amountStep.toFixed()} (Rounded DOWN): ${adjustedSize.toFixed()} -> ${finalSize.toFixed()}`);
        } else {
             logger.debug(`Size already meets amount precision step ${amountStep.toFixed()}.`);
        }

        // 3. Final Validation after Precision
        if (finalSize.lte(0)) {
            throw new Error(`Final size is zero or negative after applying precision: ${finalSize.toFixed()}. Cannot place order.`);
        }
        // Re-check min amount AFTER precision rounding (it might round down below min)
        if (finalSize.lt(minAmount)) {
            // This can happen if minAmount itself is not a multiple of amountStep, or if rounding down pushed it below.
            // Option 1: Fail the sizing. Safer.
             throw new Error(`Final size ${finalSize.toFixed()} after precision is below minimum ${minAmount.toFixed()}. Cannot place order.`);
            // Option 2: Try rounding UP to the nearest step >= minAmount. Riskier, might slightly exceed original risk %.
            // const sizeRoundedUp = adjustedSize.div(amountStep).ceil().mul(amountStep);
            // if (sizeRoundedUp.gte(minAmount) && (!maxAmount || sizeRoundedUp.lte(maxAmount))) {
            //     logger.warn(yellow(`Final size rounded down below min. Trying to round UP to nearest step >= min: ${sizeRoundedUp.toFixed()}`));
            //     finalSize = sizeRoundedUp;
            // } else {
            //     throw new Error(`Final size ${finalSize.toFixed()} after precision is below minimum ${minAmount.toFixed()}, and rounding up failed or exceeded max.`);
            // }
        }
         // Re-check max amount AFTER precision rounding (should be fine if rounded down)
        if (maxAmount && finalSize.gt(maxAmount)) {
            // This shouldn't happen if we rounded down from a value that was already capped at max.
            throw new Error(`Final size ${finalSize.toFixed()} after precision exceeds maximum ${maxAmount.toFixed()}. Logic error?`);
        }

        // 4. Min Cost Check (Estimate cost and compare to minCost)
        // Estimated Cost = finalSize * entryPrice (for linear/spot)
        // Estimated Cost = finalSize * contractSize / entryPrice (for inverse) - approximation
        let estimatedCost = new Decimal(0);
        if (marketInfo.is_contract) {
            if (marketInfo.is_linear) estimatedCost = finalSize.mul(entryPrice).mul(contractSize); // Cost is size * price * contract_size? No, usually size * price
            else estimatedCost = finalSize.mul(contractSize).div(entryPrice); // Inverse: Cost = (Contracts * ContractSize) / Price
        } else { // Spot
            estimatedCost = finalSize.mul(entryPrice); // Cost = Amount (Base) * Price (Quote/Base)
        }

        if (estimatedCost.lt(minCost)) {
             throw new Error(`Estimated cost ${estimatedCost.toFixed(8)} ${marketInfo.quote} for size ${finalSize.toFixed()} is below minimum cost ${minCost.toFixed()} ${marketInfo.quote}. Increase risk or balance.`);
        }
        logger.debug(`Estimated Cost: ~${estimatedCost.toFixed(8)} ${marketInfo.quote} (Min Cost: ${minCost.toFixed()}) - OK`);


        logger.info(green(bold(`>>> Final Calculated Position Size: ${finalSize.toFixed()} ${marketInfo.is_contract ? 'Contracts' : marketInfo.base} <<<`)));
        logger.info(bold(`--- End Position Sizing (${marketInfo.symbol}) ---`));
        return finalSize; // Return the final, validated, Decimal size

    } catch (e) {
        logger.error(red(`Position sizing failed for ${marketInfo.symbol}: ${e.message}`), { error: e });
        return null; // Return null on any failure
    }
}


/** Places a market order with retries, error handling, and Bybit V5 parameterization. */
async function placeTrade(exchange, symbol, signal, size, marketInfo, logger, reduceOnly = false, params = null) {
    const sideMap = { "BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell" };
    const side = sideMap[signal];

    // --- Input Validation ---
    if (!side) {
        logger.error(red(`Invalid trade signal '${signal}' provided for ${symbol}. Cannot determine side.`));
        return null;
    }
    if (!size || !(size instanceof Decimal) || size.lte(0)) {
        logger.error(red(`Invalid position size '${size?.toFixed()}' provided for ${symbol}. Must be a positive Decimal.`));
        return null;
    }
    if (!marketInfo) {
        logger.error(red(`Market info missing for ${symbol}. Cannot place trade.`));
        return null;
    }

    const orderType = 'market';
    const actionDesc = reduceOnly ? "Close/Reduce" : "Open/Increase";
    const marketId = marketInfo.id; // Use exchange-specific ID
    const amountStr = size.toFixed(); // Convert final Decimal size to string for API

    logger.info(bold(`===> Attempting ${actionDesc} | ${side.toUpperCase()} ${orderType.toUpperCase()} | ${symbol} | Size: ${amountStr} <===`));

    if (!CONFIG.enable_trading) {
        logger.warn(yellow(bold(`!!! TRADING DISABLED !!!`)));
        logger.warn(yellow(`Simulated ${side} ${orderType} order for ${symbol}, size ${amountStr}.`));
        // Return a simulated order structure
        return {
            id: `sim_${Date.now()}`,
            timestamp: Date.now(),
            datetime: new Date().toISOString(),
            symbol: symbol,
            type: orderType,
            side: side,
            amount: size.toNumber(), // Simulate amount
            filled: size.toNumber(), // Simulate fully filled for market
            remaining: 0,
            price: null, // Market order has no specific price
            average: null, // Average fill price unknown in sim
            status: 'closed', // Simulate closed status for market
            fee: null,
            cost: null,
            reduceOnly: reduceOnly,
            info: { simulated: true, retCode: 0, retMsg: 'OK' } // Simulate Bybit OK response
        };
    }

    for (let attempt = 0; attempt <= MAX_API_RETRIES; attempt++) {
        try {
            let orderParams = { ...(params || {}) }; // Start with any explicitly passed params

            // Add Bybit V5 specific parameters if it's a contract
            if (marketInfo.is_contract) {
                const category = marketInfo.contract_type_str.toLowerCase();
                if (category !== 'spot') { // Should always be linear/inverse here
                    orderParams.category = category;
                }
                // For Bybit V5, positionIdx=0 is for one-way mode (hedged mode uses 1/2)
                orderParams.positionIdx = 0;

                if (reduceOnly) {
                    orderParams.reduceOnly = true;
                    // For Bybit V5, market reduce orders might need timeInForce = IOC or FOK
                    orderParams.timeInForce = 'IOC'; // ImmediateOrCancel is usually safer for market reduce
                    logger.debug(`Added reduceOnly=true and timeInForce=IOC for ${symbol}.`);
                }
            }
            // Add a client order ID for better tracking/debugging?
            // orderParams.clientOrderId = `pyrm_${symbol}_${Date.now()}`;

            logger.debug(`Calling exchange.createOrder (Attempt ${attempt + 1})`);
            logger.debug(`  Symbol: ${marketId}, Type: ${orderType}, Side: ${side}, Amount: ${amountStr}`);
            logger.debug(`  Params: ${JSON.stringify(orderParams)}`);

            // Use the string amount for the amount parameter
            const orderResult = await exchange.createOrder(marketId, orderType, side, parseFloat(amountStr), undefined, orderParams); // CCXT often expects float amount

            // --- Process Result ---
            const orderId = orderResult.id || 'N/A';
            const status = orderResult.status || 'N/A';
            // Parse numbers safely back to Decimal if needed, though raw result is usually fine
            const avgPrice = safeMarketDecimal(orderResult.average, 'order.average', true);
            const filled = safeMarketDecimal(orderResult.filled, 'order.filled', true);

            logger.info(green(`${actionDesc} Order Placed Successfully! `) +
                        `ID: ${orderId}, Status: ${status}` +
                        `${avgPrice ? `, AvgFillPrice: ~${avgPrice.toFixed()}` : ''}` +
                        `${filled ? `, FilledQty: ${filled.toFixed()}` : ''}`);
            logger.debug(`Full order result (${symbol}): ${JSON.stringify(orderResult)}`);

            // Basic check: If market order didn't fill immediately (unlikely but possible), log warning.
            if (status !== 'closed' && status !== 'filled') { // 'filled' might be used by some exchanges
                 logger.warn(yellow(`Market order ${orderId} status is '${status}' (not closed/filled). Monitor closely.`));
            }
            // Check if partially filled (also unlikely for market)
            if (filled && filled.lt(size)) {
                logger.warn(yellow(`Market order ${orderId} filled quantity ${filled.toFixed()} is less than requested ${size.toFixed()}.`));
            }

            return orderResult; // Return the successful order result

        } catch (e) {
            const bybitCode = extractBybitErrorCode(e);
            const errorMessage = e.message;
            logger.warn(yellow(`Order attempt ${attempt + 1} for ${symbol} failed: ${errorMessage} ${bybitCode ? `(Code: ${bybitCode})` : ''}`));

            // --- Specific Error Handling ---
            if (e instanceof ccxt.InsufficientFunds) {
                logger.error(red(`Order Failed (${symbol} ${actionDesc}): Insufficient funds. Code: ${bybitCode}. Check balance and margin requirements.`));
                return null; // Non-retryable
            }
            if (e instanceof ccxt.InvalidOrder) {
                 logger.error(red(`Order Failed (${symbol} ${actionDesc}): Invalid order parameters. Code: ${bybitCode}. Message: ${errorMessage}`));
                 // Add hints based on common Bybit codes / messages for InvalidOrder
                 const errLower = errorMessage.toLowerCase();
                 if (bybitCode === '10001' || /parameter error/i.test(errLower)) logger.error(red(`  >> Hint: Check order parameters (size, price limits, etc.) against API docs.`));
                 else if (bybitCode === '110007' || /invalid qty|too small|lower than limit/.test(errLower)) logger.error(red(`  >> Hint: Order size ${amountStr} might be below the minimum required quantity for ${symbol}. Check Market Info.`));
                 else if (bybitCode === '110014' || /precision/i.test(errLower)) logger.error(red(`  >> Hint: Order size ${amountStr} might not meet the step size precision for ${symbol}. Check Market Info.`));
                 else if (bybitCode === '110017' || /exceed.*maximum/i.test(errLower)) logger.error(red(`  >> Hint: Order size ${amountStr} might exceed the maximum allowed quantity for ${symbol}. Check Market Info.`));
                 else if (bybitCode === '110040' || /order cost/i.test(errLower)) logger.error(red(`  >> Hint: Order cost might be below the minimum required cost for ${symbol}. Check Market Info.`));
                 else if (bybitCode === '30086' || /reduce only/i.test(errLower)) logger.error(red(`  >> Hint: Reduce-only order failed. Ensure sufficient open position size exists in the correct direction.`));
                 else if (bybitCode === '3303001' || /lot size/i.test(errLower)) logger.error(red(`  >> Hint: Order size ${amountStr} related to lot size issue (precision/min/max). Check Market Info.`));
                 // Add more specific Bybit codes as needed
                 return null; // Generally non-retryable
            }
            // Check for other potentially fatal Bybit codes that aren't InsufficientFunds or InvalidOrder
            // 110006: Slippage tolerance related (less common for market?)
            // 110013: Margin mode related
            // 110025: Position status prevents order (e.g., liquidation)
            // 110031: Risk limit related
            // 110043: Risk limit related
            // 3400060: Symbol related error (e.g., trading suspended)
            // 3400088: Instrument status prevents order
            const fatalCodes = ['110006', '110013', '110025', '110031', '110043', '3400060', '3400088'];
            if (bybitCode && fatalCodes.includes(bybitCode)) {
                logger.error(red(`>> Hint: POTENTIALLY NON-RETRYABLE order error (${symbol}). Code: ${bybitCode}. Message: ${errorMessage}`));
                return null;
            }

            if (e instanceof ccxt.AuthenticationError) { logger.error(red("Authentication error placing order. Check API keys.")); return null; }
            if (e instanceof ccxt.RateLimitExceeded) { const wait = RETRY_DELAY_SECONDS * 3 * 1000; logger.warn(yellow(`Rate limit hit. Waiting ${wait / 1000}s...`)); await delay(wait); continue; } // Retry rate limits

            // Retry generic network/exchange errors
            if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeError) {
                if (attempt >= MAX_API_RETRIES) { logger.error(red(`Max retries exceeded placing order for ${symbol}.`)); break; } // Break loop
                await delay(RETRY_DELAY_SECONDS * (attempt + 1) * 1000);
            } else {
                // Unexpected error type
                logger.error(red(`Unexpected non-retryable error placing order: ${errorMessage}`), { error: e });
                return null;
            }
        }
    } // End retry loop

    logger.error(red(`Failed to place ${actionDesc} order for ${symbol} after all retries.`));
    return null;
}


/** Cancels an open order by ID with retries and Bybit V5 parameterization. */
async function cancelOrder(exchange, orderId, symbol, logger) {
    if (!orderId) {
        logger.debug(`Cancel order skipped: No order ID provided for ${symbol}.`);
        return true; // No action needed, considered success.
    }

    logger.info(`Attempting to cancel order ID: ${orderId} for ${symbol}...`);
    const marketInfo = await getMarketInfo(exchange, symbol, logger); // Need market info for params
    if (!marketInfo) {
        logger.error(red(`Cannot cancel order ${orderId} for ${symbol}: Failed to get market info.`));
        return false; // Cannot proceed without market info for params
    }

    const marketId = marketInfo.id; // Use exchange-specific ID

    for (let attempt = 0; attempt <= MAX_API_RETRIES; attempt++) {
        try {
            logger.debug(`Cancel order attempt ${attempt + 1} for ID ${orderId} (${symbol})...`);
            let params = {};
            // Bybit V5 cancelOrder often requires category and symbol
            if (marketInfo.is_contract) {
                params.category = marketInfo.contract_type_str.toLowerCase();
                params.symbol = marketId; // Provide symbol context
            } else if (marketInfo.type === 'spot') {
                 params.category = 'spot'; // Explicitly set for spot if needed
                 params.symbol = marketId;
            }

            logger.debug(`Calling exchange.cancelOrder with ID=${orderId}, Symbol=${symbol}, Params=${JSON.stringify(params)}`);
            const response = await exchange.cancelOrder(orderId, symbol, params); // Pass symbol hint too
            logger.debug(`Raw cancelOrder response (${symbol}, ID: ${orderId}): ${JSON.stringify(response)}`);

            // Check response - Bybit V5 usually returns the cancelled order ID on success
            // CCXT might standardize this. A successful return without error is usually enough.
            // Example Bybit V5 success: { retCode: 0, retMsg: 'OK', result: { orderId: '...', orderLinkId: '...' }, ... }
             const info = response?.info;
             const retCode = info?.retCode ?? response?.retCode;
             const retMsg = info?.retMsg ?? response?.retMsg ?? 'Unknown';

            if (retCode === 0) {
                 logger.info(green(`Successfully cancelled order ID: ${orderId} for ${symbol} (Code: 0).`));
                 return true;
            } else if (retCode === 110021 || retMsg.includes("Order does not exist")) {
                 // Order not found - could be already filled, cancelled, or wrong ID
                 logger.warn(yellow(`Order ID ${orderId} (${symbol}) not found during cancellation (Code: ${retCode}). Already filled/cancelled? Treating as success.`));
                 return true; // Treat OrderNotFound as success for cancellation purpose
            } else {
                 // Unexpected error code from Bybit
                 throw new ccxt.ExchangeError(`Bybit API error cancelling order (${symbol}, ID: ${orderId}): ${retMsg} (Code: ${retCode ?? 'N/A'})`);
            }

        } catch (e) {
            const bybitCode = extractBybitErrorCode(e);
            logger.warn(yellow(`Cancel order attempt ${attempt + 1} (ID: ${orderId}, Symbol: ${symbol}) failed: ${e.message} ${bybitCode ? `(Code: ${bybitCode})` : ''}`));

            // Handle OrderNotFound specifically - treat as success for cancellation
            if (e instanceof ccxt.OrderNotFound || bybitCode === '110021') {
                logger.warn(yellow(`Order ID ${orderId} (${symbol}) not found (ccxt/Bybit). Already filled/cancelled? Treating as success.`));
                return true;
            }
            if (e instanceof ccxt.AuthenticationError) { logger.error(red("Authentication error cancelling order.")); return false; }
            if (e instanceof ccxt.RateLimitExceeded) { const wait = RETRY_DELAY_SECONDS * 2 * 1000; logger.warn(yellow(`Rate limit hit. Waiting ${wait / 1000}s...`)); await delay(wait); continue; }

            // Retry generic network/exchange errors
            if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeError) {
                if (attempt >= MAX_API_RETRIES) { logger.error(red(`Max retries exceeded cancelling order ${orderId} (${symbol}).`)); break; }
                await delay(RETRY_DELAY_SECONDS * (attempt + 1) * 1000);
            } else {
                logger.error(red(`Non-retryable error cancelling order: ${e.message}`), { error: e });
                return false; // Non-retryable error
            }
        }
    } // End retry loop
    logger.error(red(`Failed to cancel order ID ${orderId} (${symbol}) after all retries.`));
    return false;
}


/**
 * Sets position protection (SL/TP/TSL) using Bybit's V5 private API endpoint `/v5/position/set-trading-stop`.
 * Handles formatting prices, parameter validation, and API call retries.
 * Note: Setting TSL (`tslDistance` > 0) will typically override/replace any fixed `slPrice` on Bybit V5.
 * Setting `slPrice` or `tpPrice` to 0 or null/undefined clears that specific protection.
 * Setting `tslDistance` to 0 clears the trailing stop.
 */
async function _set_position_protection(
    exchange,
    symbol,
    marketInfo,
    positionInfo, // Pass current position info to validate against
    logger,
    slPrice,      // Decimal | null | 0 (0 or null to clear)
    tpPrice,      // Decimal | null | 0 (0 or null to clear)
    tslDistance,  // Decimal | null | 0 (0 or null to clear)
    tslActivation // Decimal | null (Required if tslDistance > 0)
) {
    if (!marketInfo || !marketInfo.is_contract) {
        logger.warn(yellow(`Protection setting skipped for ${symbol}: Not a contract market.`));
        return false;
    }
    if (!positionInfo) {
        logger.error(red(`Protection setting failed for ${symbol}: Missing current position information.`));
        return false;
    }
    // Ensure we have necessary details from position info
    const posSide = positionInfo.side;
    const entryPrice = positionInfo.entryPrice; // Should already be Decimal
    if (!posSide || !entryPrice) {
        logger.error(red(`Protection setting failed for ${symbol}: Invalid position side ('${posSide}') or entry price ('${entryPrice?.toFixed()}') from position info.`));
        return false;
    }
    // Price precision is crucial for formatting
    const priceTick = marketInfo.price_precision_step_decimal;
    if (!priceTick || priceTick.lte(0)) {
        logger.error(red(`Protection setting failed for ${symbol}: Invalid market price precision step.`));
        return false;
    }

    const endpoint = '/v5/position/set-trading-stop';
    let params_to_set = {
        symbol: marketInfo.id, // Use exchange-specific ID
        category: marketInfo.contract_type_str.toLowerCase(), // 'linear' or 'inverse'
        positionIdx: 0 // One-way mode
    };
    let logParts = [bold(`Attempting protection update for ${symbol} (${posSide.toUpperCase()} @ ${entryPrice.toFixed()}):`)];
    let requestedChanges = false; // Track if any valid changes are requested

    try {
        // --- Format and Validate Stop Loss ---
        let fmtSl = '0'; // Default to '0' (clear SL) if not provided or invalid
        if (slPrice instanceof Decimal && slPrice.gt(0)) {
            const isValidSl = (posSide === 'long' && slPrice.lt(entryPrice)) || (posSide === 'short' && slPrice.gt(entryPrice));
            if (!isValidSl) {
                logger.error(red(`  - Invalid SL Price: ${slPrice.toFixed()} is not valid for a ${posSide} position with entry ${entryPrice.toFixed()}. SL not included.`));
                // Keep fmtSl = '0' or decide if this should be a hard failure
            } else {
                const formatted = formatPrice(exchange, symbol, slPrice);
                if (formatted) {
                    fmtSl = formatted;
                    logParts.push(`  - Setting SL: ${fmtSl}`);
                    requestedChanges = true;
                } else {
                    logger.error(red(`  - Failed to format valid SL price ${slPrice.toFixed()}. SL not included.`));
                }
            }
        } else if (slPrice === null || slPrice === undefined || (slPrice instanceof Decimal && slPrice.isZero())) {
             logParts.push(`  - Clearing SL (requested or default)`);
             requestedChanges = true; // Requesting to clear is a change
        } else if (slPrice) { // Provided but not valid Decimal > 0
            logger.warn(yellow(`  - Invalid SL input type or value: ${slPrice}. SL not included.`));
        }
        params_to_set.stopLoss = fmtSl; // Always set, '0' clears it

        // --- Format and Validate Take Profit ---
        let fmtTp = '0'; // Default to '0' (clear TP)
        if (tpPrice instanceof Decimal && tpPrice.gt(0)) {
            const isValidTp = (posSide === 'long' && tpPrice.gt(entryPrice)) || (posSide === 'short' && tpPrice.lt(entryPrice));
            if (!isValidTp) {
                logger.error(red(`  - Invalid TP Price: ${tpPrice.toFixed()} is not valid for a ${posSide} position with entry ${entryPrice.toFixed()}. TP not included.`));
            } else {
                const formatted = formatPrice(exchange, symbol, tpPrice);
                if (formatted) {
                    fmtTp = formatted;
                    logParts.push(`  - Setting TP: ${fmtTp}`);
                    requestedChanges = true;
                } else {
                    logger.error(red(`  - Failed to format valid TP price ${tpPrice.toFixed()}. TP not included.`));
                }
            }
        } else if (tpPrice === null || tpPrice === undefined || (tpPrice instanceof Decimal && tpPrice.isZero())) {
            logParts.push(`  - Clearing TP (requested or default)`);
            requestedChanges = true;
        } else if (tpPrice) {
            logger.warn(yellow(`  - Invalid TP input type or value: ${tpPrice}. TP not included.`));
        }
        params_to_set.takeProfit = fmtTp; // Always set, '0' clears it


        // --- Format and Validate Trailing Stop ---
        // Note: Setting TSL distance > 0 on Bybit V5 usually disables/replaces the fixed SL.
        let fmtTslDist = '0'; // Default to '0' (clear TSL)
        let fmtTslAct = '';   // Default to empty string (required when clearing TSL distance)

        if (tslDistance instanceof Decimal && tslDistance.gt(0)) { // Setting an active TSL
            // Activation price is required when setting distance > 0
            if (!(tslActivation instanceof Decimal) || tslActivation.lte(0)) {
                 logger.error(red(`  - Invalid TSL Activation Price: Required and must be positive Decimal when setting TSL distance. TSL not included.`));
            } else {
                // Validate activation price direction relative to entry
                 const isValidAct = (posSide === 'long' && tslActivation.gt(entryPrice)) || (posSide === 'short' && tslActivation.lt(entryPrice));
                 if (!isValidAct) {
                     logger.error(red(`  - Invalid TSL Activation Price: ${tslActivation.toFixed()} is wrong direction for ${posSide} entry ${entryPrice.toFixed()}. TSL not included.`));
                 } else {
                     const formattedDist = formatPrice(exchange, symbol, tslDistance); // TSL distance is a price difference, format like price
                     const formattedAct = formatPrice(exchange, symbol, tslActivation);
                     if (formattedDist && formattedAct) {
                         fmtTslDist = formattedDist;
                         fmtTslAct = formattedAct;
                         // Bybit V5 uses 'trailingStop' for distance and 'activePrice' for activation
                         params_to_set.trailingStop = fmtTslDist;
                         params_to_set.activePrice = fmtTslAct;
                         // Explicitly remove fixed SL param if setting active TSL, as TSL takes precedence
                         delete params_to_set.stopLoss;
                         logParts.push(green(`  - Setting TSL: Distance=${fmtTslDist}, Activation=${fmtTslAct} (Overrides fixed SL)`));
                         requestedChanges = true;
                     } else {
                          if (!formattedDist) logger.error(red(`  - Failed to format TSL distance ${tslDistance.toFixed()}. TSL not included.`));
                          if (!formattedAct) logger.error(red(`  - Failed to format TSL activation ${tslActivation.toFixed()}. TSL not included.`));
                     }
                 }
            }
        } else if (tslDistance === null || tslDistance === undefined || (tslDistance instanceof Decimal && tslDistance.isZero())) {
             // Explicitly clearing TSL
             params_to_set.trailingStop = '0';
             params_to_set.activePrice = ''; // Bybit requires empty string for activation when clearing TSL
             logParts.push(`  - Clearing TSL (requested or default)`);
             requestedChanges = true;
             // Ensure fixed SL (if set earlier) is included when TSL is cleared
             if (!params_to_set.stopLoss) params_to_set.stopLoss = fmtSl; // Restore formatted SL
        } else if (tslDistance) { // Invalid input type/value
             logger.warn(yellow(`  - Invalid TSL distance input type or value: ${tslDistance}. TSL not included.`));
        }
        // If TSL was NOT set (fmtTslDist is '0'), ensure activePrice is not sent or is empty
        if (params_to_set.trailingStop === '0') {
             params_to_set.activePrice = '';
        }


    } catch (e) {
        logger.error(red(`Error during protection parameter preparation for ${symbol}: ${e.message}`), { error: e });
        return false; // Hard failure during prep
    }

    // --- Check if any protection needs setting/clearing ---
    if (!requestedChanges) {
        logger.debug(`No valid protection changes requested for ${symbol}. Skipping API call.`);
        return true; // No changes needed, considered success.
    }

    // Clean up params: remove fields that are '0' or empty if API prefers absence?
    // Bybit V5 seems to require '0' to explicitly clear SL/TP/TSL. Keep them.
    // Remove activePrice if trailingStop is '0'.
    if (params_to_set.trailingStop === '0') delete params_to_set.activePrice;

    // Log the final parameters being sent
    logger.info(logParts.join('\n'));
    logger.debug(`  Final API params for ${endpoint} (${symbol}): ${JSON.stringify(params_to_set)}`);

    // --- Execute API Call ---
    if (!CONFIG.enable_trading) {
        logger.warn(yellow(`Trading disabled: Simulated protection set/update for ${symbol}. Parameters:`));
        logger.warn(yellow(JSON.stringify(params_to_set)));
        return true; // Simulate success
    }

    for (let attempt = 0; attempt <= MAX_API_RETRIES; attempt++) {
        try {
            logger.debug(`Executing privatePost ${endpoint} (${symbol}, Attempt ${attempt + 1})...`);
            // Use privatePost as this is not a standard ccxt unified method
            const response = await exchange.privatePost(endpoint, params_to_set);
            logger.debug(`Raw response from ${endpoint} (${symbol}): ${JSON.stringify(response)}`);

            // Check Bybit V5 response structure
            const retCode = response?.retCode;
            const retMsg = response?.retMsg ?? 'Unknown Bybit message';

            if (retCode === 0) {
                logger.info(green(`Protection set/updated successfully for ${symbol} (Code: 0).`));
                return true; // Success
            }
            // Specific non-error codes? e.g., "3400048: Modification failed as SL/TP setup is identical" - treat as success?
            if (retCode === 3400048) {
                 logger.info(yellow(`Protection not modified as request was identical to current settings (${symbol}, Code: 3400048). Considered success.`));
                 return true;
            }
            // Other specific codes indicating success despite non-zero code? Check Bybit docs.

            // If not a known success code, throw an error
            throw new ccxt.ExchangeError(`Bybit API error setting protection (${symbol}): ${retMsg} (Code: ${retCode ?? 'N/A'})`);

        } catch (e) {
            const bybitCode = extractBybitErrorCode(e);
            logger.warn(yellow(`Protection set attempt ${attempt + 1} failed: ${e.message} ${bybitCode ? `(Code: ${bybitCode})` : ''}`));

            // Check for non-retryable Bybit codes related to set-trading-stop
            // 10001: Parameter error (invalid price format, SL/TP direction wrong, etc.)
            // 110013: Margin mode conflict
            // 110025: Position status conflict (e.g., liquidation)
            // 110043: Risk limit issue
            // 3400048: Already identical (handled above as success)
            // 3400051: SL/TP cannot be the same price
            // 3400052: Trigger price error (e.g., TSL activation invalid direction)
            // 3400070: SL price invalid direction vs entry
            // 3400071: TP price invalid direction vs entry
            // 3400072: TSL activation price invalid direction vs entry
            // 3400073: TSL distance invalid (e.g., zero/negative when activation is set)
            const fatalCodes = ['10001', '110013', '110025', '110043', '3400051', '3400052', '3400070', '3400071', '3400072', '3400073'];
            const fatalMsgs = [/parameter error/i, /invalid price/i, /position status/i, /cannot be the same/i, /trigger price/i, /activation price/i, /distance invalid/i, /sl.*tp.*same/i, /invalid sl.*direction/i, /invalid tp.*direction/i];

            if ((bybitCode && fatalCodes.includes(bybitCode)) || fatalMsgs.some(rx => rx.test(e.message))) {
                logger.error(red(`>> Hint: NON-RETRYABLE protection error (${symbol}). Code: ${bybitCode}. Message: ${e.message}`));
                logger.error(red(`>> Action: Check parameters (SL/TP/TSL values & directions vs entry price), position status.`));
                return false; // Abort on these errors
            }

            if (e instanceof ccxt.AuthenticationError) { logger.error(red("Authentication error setting protection.")); return false; }
            if (e instanceof ccxt.RateLimitExceeded) { const wait = RETRY_DELAY_SECONDS * 3 * 1000; logger.warn(yellow(`Rate limit hit. Waiting ${wait / 1000}s...`)); await delay(wait); continue; }

            // Retry generic network/exchange errors
            if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeError) {
                if (attempt >= MAX_API_RETRIES) { logger.error(red(`Max retries exceeded setting protection for ${symbol}.`)); break; } // Break loop
                await delay(RETRY_DELAY_SECONDS * (attempt + 1) * 1000);
            } else {
                logger.error(red(`Unexpected non-retryable error setting protection: ${e.message}`), { error: e });
                return false; // Non-retryable error
            }
        }
    } // End retry loop
    logger.error(red(`Failed to set protection for ${symbol} after all retries.`));
    return false;
}


// --- Strategy Implementation ---

/**
 * Finds Pivot Highs (PH) and Pivot Lows (PL) using simple lookback logic.
 * This is a basic JavaScript implementation mimicking common pivot definitions.
 * Note: Danfo.js doesn't have a direct rolling window function suitable for this
 * specific non-causal (looks left and right) pivot logic easily. Iteration is clearer here.
 *
 * @param {DataFrame} df Danfo DataFrame with 'high', 'low', 'open', 'close' columns.
 * @param {number} phLeft Lookback period to the left for highs.
 * @param {number} phRight Lookforward period to the right for highs.
 * @param {number} plLeft Lookback period to the left for lows.
 * @param {number} plRight Lookforward period to the right for lows.
 * @param {boolean} useWicks If true, use 'high'/'low'. If false, use max/min of 'open'/'close'.
 * @returns {{isPivotHigh: boolean[], isPivotLow: boolean[]}} Arrays indicating pivot points.
 */
function findPivotsJS(df, phLeft, phRight, plLeft, plRight, useWicks) {
    const n = df.shape[0];
    const isPivotHigh = Array(n).fill(false);
    const isPivotLow = Array(n).fill(false);

    // Determine which columns to use based on useWicks
    const highPrice = useWicks ? df['high'].values : df.apply((row) => Math.max(row['open'], row['close']), 1).values;
    const lowPrice = useWicks ? df['low'].values : df.apply((row) => Math.min(row['open'], row['close']), 1).values;

    // Iterate through candles where a pivot *could* be identified (need enough bars left and right)
    // Max lookback = max(phLeft, plLeft), Max lookforward = max(phRight, plRight)
    const startIdx = Math.max(phLeft, plLeft);
    const endIdx = n - Math.max(phRight, plRight);

    for (let i = startIdx; i < endIdx; i++) {
        const currentHigh = highPrice[i];
        const currentLow = lowPrice[i];
        let isPh = true;
        let isPl = true;

        // Check Pivot High Requirements
        // Left side: current high must be strictly greater than left highs
        for (let j = 1; j <= phLeft; j++) {
            if (highPrice[i - j] >= currentHigh) {
                isPh = false;
                break;
            }
        }
        if (!isPh) continue; // No need to check right side if left fails

        // Right side: current high must be greater than or equal to right highs
        // (Using >= on the right helps select the first bar of a flat top as the pivot)
        for (let j = 1; j <= phRight; j++) {
            if (highPrice[i + j] > currentHigh) { // Stricter: Use > if only unique peak desired
                isPh = false;
                break;
            }
        }
        isPivotHigh[i] = isPh; // Mark if both sides passed


        // Check Pivot Low Requirements (independent check, could optimize slightly)
        // Left side: current low must be strictly less than left lows
        for (let j = 1; j <= plLeft; j++) {
            if (lowPrice[i - j] <= currentLow) {
                isPl = false;
                break;
            }
        }
        if (!isPl) continue; // No need to check right side if left fails

        // Right side: current low must be less than or equal to right lows
        // (Using <= on the right helps select the first bar of a flat bottom as the pivot)
         for (let j = 1; j <= plRight; j++) {
            if (lowPrice[i + j] < currentLow) { // Stricter: Use < if only unique trough desired
                isPl = false;
                break;
            }
        }
        isPivotLow[i] = isPl; // Mark if both sides passed
    }

    return { isPivotHigh, isPivotLow };
}


/**
 * Calculates strategy indicators (VT, ATR, VolNorm), identifies Pivot-based Order Blocks (OBs),
 * checks for OB violations, and generates trading signals based on trend and OB interactions.
 * Uses Decimal.js for price comparisons within OB logic.
 *
 * @param {DataFrame} df Input Danfo DataFrame with OHLCV data.
 * @param {object} config Symbol-specific configuration containing strategy_params.
 * @param {winston.Logger} logger Logger instance.
 * @param {Exchange} exchange CCXT exchange instance (needed for position check in signal logic).
 * @param {string} symbol Symbol being analyzed (needed for position check).
 * @returns {Promise<StrategyAnalysisResults>} Object containing results and the final signal.
 */
async function calculateStrategySignals(df, config, logger, exchange, symbol) {
    const functionStart = performance.now();
    // --- Initialize Default Results ---
    let results = {
        dataframe: null, // Will hold the df with calculated indicators
        last_close: null,
        current_trend_up: null,
        trend_just_changed: false,
        active_bull_boxes: [],
        active_bear_boxes: [],
        vol_norm_int: null,
        atr: null,
        upper_band: null,
        lower_band: null,
        signal: 'NONE'
    };

    // --- Input Validation ---
    const minRequiredCandles = Math.max(
        config.strategy_params?.vt_length ?? DEFAULT_VT_LENGTH,
        config.strategy_params?.vt_atr_period ?? DEFAULT_VT_ATR_PERIOD,
        config.strategy_params?.vt_vol_ema_length ?? DEFAULT_VT_VOL_EMA_LENGTH,
        (config.strategy_params?.ph_left ?? DEFAULT_PH_LEFT) + (config.strategy_params?.ph_right ?? DEFAULT_PH_RIGHT) + 1,
        (config.strategy_params?.pl_left ?? DEFAULT_PL_LEFT) + (config.strategy_params?.pl_right ?? DEFAULT_PL_RIGHT) + 1,
        50 // Absolute minimum
    );
    if (!df || df.shape[0] < minRequiredCandles) {
        logger.warn(yellow(`Not enough kline data (${df?.shape[0] ?? 0}) for strategy calculation (requires ~${minRequiredCandles}). Skipping analysis for ${symbol}.`));
        return results; // Return default results
    }
    logger.debug(`Calculating strategy signals for ${symbol} using ${df.shape[0]} candles...`);

    // --- Extract Strategy Parameters ---
    const sp = { // Merge symbol config with defaults
        vt_length: config.strategy_params?.vt_length ?? DEFAULT_VT_LENGTH,
        vt_atr_period: config.strategy_params?.vt_atr_period ?? DEFAULT_VT_ATR_PERIOD,
        vt_vol_ema_length: config.strategy_params?.vt_vol_ema_length ?? DEFAULT_VT_VOL_EMA_LENGTH,
        vt_atr_multiplier: config.strategy_params?.vt_atr_multiplier ?? DEFAULT_VT_ATR_MULTIPLIER,
        ob_source: config.strategy_params?.ob_source ?? DEFAULT_OB_SOURCE,
        ph_left: config.strategy_params?.ph_left ?? DEFAULT_PH_LEFT,
        ph_right: config.strategy_params?.ph_right ?? DEFAULT_PH_RIGHT,
        pl_left: config.strategy_params?.pl_left ?? DEFAULT_PL_LEFT,
        pl_right: config.strategy_params?.pl_right ?? DEFAULT_PL_RIGHT,
        ob_extend: config.strategy_params?.ob_extend ?? DEFAULT_OB_EXTEND,
        ob_max_boxes: config.strategy_params?.ob_max_boxes ?? DEFAULT_OB_MAX_BOXES,
        ob_entry_proximity_factor: config.strategy_params?.ob_entry_proximity_factor ?? DEFAULT_OB_ENTRY_PROXIMITY_FACTOR,
        ob_exit_proximity_factor: config.strategy_params?.ob_exit_proximity_factor ?? DEFAULT_OB_EXIT_PROXIMITY_FACTOR,
    };
    const useWicksForPivots = sp.ob_source.toLowerCase() === "wicks";
    logger.debug(`Strategy Params (${symbol}): ${JSON.stringify(sp)}`);

    let dfCalc = df.copy(); // Work on a copy

    try {
        // === Indicator Calculations (using technicalindicators library) ===
        // Ensure data is in the format expected by the library (arrays of numbers)
        const close = dfCalc['close'].values; // Array<number>
        const high = dfCalc['high'].values;   // Array<number>
        const low = dfCalc['low'].values;    // Array<number>
        const volume = dfCalc['volume'].values; // Array<number>

        // 1. EMA (for trend direction baseline)
        const emaResult = EMA.calculate({ period: sp.vt_length, values: close });
        const emaPadded = Array(close.length - emaResult.length).fill(NaN).concat(emaResult);
        dfCalc.addColumn(`EMA_${sp.vt_length}`, emaPadded, { inplace: true });

        // 2. ATR (for Volumatic Trend Bands & potentially SL/TP)
        const atrResult = ATR.calculate({ high, low, close, period: sp.vt_atr_period });
        const atrPadded = Array(close.length - atrResult.length).fill(NaN).concat(atrResult);
        dfCalc.addColumn('ATR', atrPadded, { inplace: true });
        // Convert last ATR to Decimal for results
        results.atr = safeMarketDecimal(dfCalc['ATR'].iloc(dfCalc.shape[0] - 1), 'atr', false);

        // 3. Volumatic Trend Bands (Upper/Lower)
        const atrMultNum = parseFloat(sp.vt_atr_multiplier); // Use float for Danfo ops
        // Need to handle NaNs from EMA/ATR padding before calculating bands
        dfCalc[`EMA_${sp.vt_length}`].fillNa({ method: 'auto', inplace: true }); // Forward fill EMA
        dfCalc['ATR'].fillNa({ method: 'auto', inplace: true }); // Forward fill ATR
        dfCalc.addColumn('VT_UpperBand', dfCalc[`EMA_${sp.vt_length}`].add(dfCalc['ATR'].mul(atrMultNum)), { inplace: true });
        dfCalc.addColumn('VT_LowerBand', dfCalc[`EMA_${sp.vt_length}`].sub(dfCalc['ATR'].mul(atrMultNum)), { inplace: true });
        // Convert last band values to Decimal for results
        results.upper_band = safeMarketDecimal(dfCalc['VT_UpperBand'].iloc(dfCalc.shape[0] - 1), 'upper_band', false);
        results.lower_band = safeMarketDecimal(dfCalc['VT_LowerBand'].iloc(dfCalc.shape[0] - 1), 'lower_band', false);


        // 4. Volume Normalization (Example - adapt Python logic)
        // This requires careful translation of the Python rolling std dev and EMA logic.
        // Using a simplified placeholder for now. Replace with actual logic.
        let volNormIntPadded = Array(close.length).fill(0); // Placeholder
        if (volume && volume.length === close.length && volume.some(v => v > 0)) {
            // Example: Calculate rolling EMA of volume
            // const volEmaResult = EMA.calculate({ period: sp.vt_vol_ema_length, values: volume });
            // const volEmaPadded = Array(close.length - volEmaResult.length).fill(NaN).concat(volEmaResult);
            // // Normalize current volume against its EMA (simple example)
            // volNormIntPadded = volume.map((v, i) => {
            //     const ema = volEmaPadded[i];
            //     if (isNaN(v) || isNaN(ema) || ema === 0) return 0;
            //     return Math.round((v / ema) * 100); // Example: % of EMA, rounded
            // });
             logger.debug("Volume normalization logic needs full implementation."); // Reminder
        }
        dfCalc.addColumn('VolNormInt', volNormIntPadded, { inplace: true });
        results.vol_norm_int = dfCalc['VolNormInt'].iloc(dfCalc.shape[0] - 1);


        // 5. Trend Direction & Change
        // Using Close > EMA as the trend definition for now (matches Python reference basis)
        // Could use VT Bands crossover logic if preferred (Close > LowerBand for Up, Close < UpperBand for Down)
        dfCalc.addColumn('TrendUp', dfCalc['close'].gt(dfCalc[`EMA_${sp.vt_length}`]), { inplace: true });
        dfCalc['TrendUp'] = dfCalc['TrendUp'].fillNa({ method: 'auto' }).asType('boolean'); // Handle potential NaNs from EMA start
        const trendUpVals = dfCalc['TrendUp'].values; // Array<boolean>
        const trendChanged = trendUpVals.map((up, i) => {
            if (i === 0) return false; // Cannot change on the first bar
            // Check if current and previous values are valid booleans and different
            return typeof up === 'boolean' && typeof trendUpVals[i-1] === 'boolean' && up !== trendUpVals[i-1];
        });
        dfCalc.addColumn('TrendChanged', trendChanged, { inplace: true });
        // Store last trend values in results
        results.current_trend_up = typeof trendUpVals[trendUpVals.length - 1] === 'boolean' ? trendUpVals[trendUpVals.length - 1] : null;
        results.trend_just_changed = trendChanged[trendChanged.length - 1] ?? false;


        // === Pivot and Order Block Identification ===
        logger.debug("Identifying Pivots...");
        const { isPivotHigh, isPivotLow } = findPivotsJS(dfCalc, sp.ph_left, sp.ph_right, sp.pl_left, sp.pl_right, useWicksForPivots);
        dfCalc.addColumn('PivotHigh', isPivotHigh, { inplace: true });
        dfCalc.addColumn('PivotLow', isPivotLow, { inplace: true });
        logger.debug(`Found ${isPivotHigh.filter(Boolean).length} potential PH, ${isPivotLow.filter(Boolean).length} potential PL.`);

        // --- Create and Manage Order Blocks (Requires Decimal precision) ---
        logger.debug("Processing Order Blocks...");
        let currentBullBoxes = []; let currentBearBoxes = [];
        const dfIndices = dfCalc.index; // Array of timestamps (numbers)
        // Get price columns as Decimals for precise comparison
        const dfHighDec = dfCalc['high'].values.map(v => safeMarketDecimal(v, 'h', false));
        const dfLowDec = dfCalc['low'].values.map(v => safeMarketDecimal(v, 'l', false));
        const dfOpenDec = dfCalc['open'].values.map(v => safeMarketDecimal(v, 'o', false));
        const dfCloseDec = dfCalc['close'].values.map(v => safeMarketDecimal(v, 'c', false));
        results.last_close = dfCloseDec[dfCloseDec.length - 1]; // Store last close (Decimal)

        // Iterate through the DataFrame to identify OBs from pivots
        for (let i = 0; i < dfCalc.shape[0]; i++) {
            const ts = dfIndices[i]; // Timestamp of the potential pivot candle

            // --- Create Bearish OB from Pivot High ---
            if (isPivotHigh[i] && dfHighDec[i] && dfLowDec[i] && dfOpenDec[i] && dfCloseDec[i]) {
                const obTop = dfHighDec[i]; // Top of Bear OB is the Pivot High price
                const obBottom = useWicksForPivots
                    ? dfLowDec[i] // Bottom is the low of the pivot candle
                    : Decimal.min(dfOpenDec[i], dfCloseDec[i]); // Bottom is the body low

                if (obTop.gt(obBottom)) { // Ensure valid range
                    currentBearBoxes.push({
                        id: `B_${ts}`, type: 'bear', timestamp: ts,
                        top: obTop, bottom: obBottom,
                        active: true, violated: false, violation_ts: null, extended_to_ts: ts
                    });
                }
            }

            // --- Create Bullish OB from Pivot Low ---
            if (isPivotLow[i] && dfHighDec[i] && dfLowDec[i] && dfOpenDec[i] && dfCloseDec[i]) {
                const obBottom = dfLowDec[i]; // Bottom of Bull OB is the Pivot Low price
                const obTop = useWicksForPivots
                    ? dfHighDec[i] // Top is the high of the pivot candle
                    : Decimal.max(dfOpenDec[i], dfCloseDec[i]); // Top is the body high

                if (obTop.gt(obBottom)) { // Ensure valid range
                    currentBullBoxes.push({
                        id: `L_${ts}`, type: 'bull', timestamp: ts,
                        top: obTop, bottom: obBottom,
                        active: true, violated: false, violation_ts: null, extended_to_ts: ts
                    });
                }
            }
        } // End of OB creation loop

        // Sort OBs newest first, limit count
        currentBearBoxes.sort((a, b) => b.timestamp - a.timestamp);
        currentBearBoxes = currentBearBoxes.slice(0, sp.ob_max_boxes);
        currentBullBoxes.sort((a, b) => b.timestamp - a.timestamp);
        currentBullBoxes = currentBullBoxes.slice(0, sp.ob_max_boxes);
        logger.debug(`Created ${currentBearBoxes.length} initial Bear OBs, ${currentBullBoxes.length} initial Bull OBs.`);

        // --- Check Violations & Extend OBs ---
        // Iterate through candles *after* each OB was formed
        for (let i = 0; i < dfCalc.shape[0]; i++) {
            const candleTs = dfIndices[i];
            const candleClose = dfCloseDec[i]; // Use Decimal close price
            if (!candleClose) continue; // Skip if close price is invalid

            // Check Bearish OBs
            currentBearBoxes.forEach(ob => {
                if (ob.active && candleTs > ob.timestamp) { // Only check candles after OB formation
                    // Violation check: Candle closed decisively above the OB top
                    if (candleClose.gt(ob.top)) {
                        ob.active = false;
                        ob.violated = true;
                        ob.violation_ts = candleTs;
                        // logger.debug(`Bear OB ${ob.id} violated at ${candleTs} by close ${candleClose.toFixed()}`);
                    } else if (sp.ob_extend) {
                        // Extend the OB's validity forward in time if not violated
                        ob.extended_to_ts = candleTs;
                    }
                }
            });

            // Check Bullish OBs
            currentBullBoxes.forEach(ob => {
                if (ob.active && candleTs > ob.timestamp) {
                    // Violation check: Candle closed decisively below the OB bottom
                    if (candleClose.lt(ob.bottom)) {
                        ob.active = false;
                        ob.violated = true;
                        ob.violation_ts = candleTs;
                         // logger.debug(`Bull OB ${ob.id} violated at ${candleTs} by close ${candleClose.toFixed()}`);
                    } else if (sp.ob_extend) {
                        ob.extended_to_ts = candleTs;
                    }
                }
            });
        } // End of violation check loop

        // Filter final active OBs for results
        results.active_bear_boxes = currentBearBoxes.filter(ob => ob.active);
        results.active_bull_boxes = currentBullBoxes.filter(ob => ob.active);
        logger.debug(`OB Analysis Complete: ${results.active_bull_boxes.length} final active Bull OBs, ${results.active_bear_boxes.length} final active Bear OBs.`);
        // logger.debug(`Active Bull OBs: ${JSON.stringify(results.active_bull_boxes.map(ob=>({id:ob.id, t:ob.top.toFixed(), b:ob.bottom.toFixed()})))})`);
        // logger.debug(`Active Bear OBs: ${JSON.stringify(results.active_bear_boxes.map(ob=>({id:ob.id, t:ob.top.toFixed(), b:ob.bottom.toFixed()})))})`);


        // === Signal Generation ===
        let signal = "NONE";
        // Ensure we have valid last close price and trend direction
        if (results.last_close && results.current_trend_up !== null) {
            const currentPrice = results.last_close;
            const isUpTrend = results.current_trend_up;

            // Check current position status (needed for exit logic)
            // Avoid calling getOpenPosition repeatedly if possible, maybe pass it in?
            // For now, call it here as needed for signal generation context.
            const currentPosition = await getOpenPosition(exchange, symbol, logger);

            // --- Entry Logic (Only if FLAT) ---
            if (!currentPosition) {
                logger.debug(`Signal Logic: No open position found for ${symbol}. Evaluating entry.`);
                const entryProxFactor = new Decimal(sp.ob_entry_proximity_factor); // e.g., 1.005

                if (isUpTrend) { // Uptrend: Look for BUY near an active Bullish OB
                    for (const ob of results.active_bull_boxes) {
                        // Price needs to be *near* or *within* the OB
                        // Condition: Price <= OB Top * Factor AND Price >= OB Bottom
                        const proximityTop = ob.top.mul(entryProxFactor);
                        if (currentPrice.lte(proximityTop) && currentPrice.gte(ob.bottom)) {
                            signal = "BUY";
                            logger.info(green(`Signal (${symbol}): BUY - Uptrend, Price ${currentPrice.toFixed()} near Bull OB ${ob.id} [${ob.bottom.toFixed()}-${ob.top.toFixed()}] (ProxTop: ${proximityTop.toFixed()})`));
                            break; // Take first valid entry signal
                        }
                    }
                } else { // Downtrend: Look for SELL near an active Bearish OB
                    for (const ob of results.active_bear_boxes) {
                        // Price needs to be *near* or *within* the OB
                        // Condition: Price >= OB Bottom / Factor AND Price <= OB Top
                        // Note: Dividing by factor for Bear OB bottom check (e.g., price > 99.5 if bottom=100, factor=1.005)
                        const proximityBottom = ob.bottom.div(entryProxFactor);
                        if (currentPrice.gte(proximityBottom) && currentPrice.lte(ob.top)) {
                            signal = "SELL";
                            logger.info(red(`Signal (${symbol}): SELL - Downtrend, Price ${currentPrice.toFixed()} near Bear OB ${ob.id} [${ob.bottom.toFixed()}-${ob.top.toFixed()}] (ProxBot: ${proximityBottom.toFixed()})`));
                            break; // Take first valid entry signal
                        }
                    }
                }
            }
            // --- Exit Logic (Only if IN a position) ---
            else {
                 logger.debug(`Signal Logic: Found open ${currentPosition.side} position for ${symbol}. Evaluating exit.`);
                 const exitProxFactor = new Decimal(sp.ob_exit_proximity_factor); // e.g., 1.001

                 // 1. Exit on Trend Reversal? (Check if enabled in config?)
                 // Let's assume exit on trend change is always checked first.
                 if (results.trend_just_changed) {
                     if (currentPosition.side === 'long' && !isUpTrend) {
                         signal = "EXIT_LONG";
                         logger.warn(yellow(`Signal (${symbol}): EXIT_LONG - Trend reversed to DOWN.`));
                     } else if (currentPosition.side === 'short' && isUpTrend) {
                         signal = "EXIT_SHORT";
                         logger.warn(yellow(`Signal (${symbol}): EXIT_SHORT - Trend reversed to UP.`));
                     }
                 }

                 // 2. Exit on OB Violation (if no trend reversal exit yet)
                 // Exit if price closes beyond the *opposing* active OB (indicating failure of the structure)
                 // Or closes beyond the *supporting* OB? Let's use opposing OB violation for exit signal.
                 if (signal === "NONE") {
                     if (currentPosition.side === 'long') {
                         // Check if price closed above a nearby Bearish OB (signalling potential reversal/stall)
                         for (const ob of results.active_bear_boxes) {
                             const proximityTop = ob.top.div(exitProxFactor); // Price needs to close clearly above top
                             if (currentPrice.gt(proximityTop)) {
                                 signal = "EXIT_LONG";
                                 logger.warn(yellow(`Signal (${symbol}): EXIT_LONG - Price ${currentPrice.toFixed()} closed above Bear OB ${ob.id} [${ob.bottom.toFixed()}-${ob.top.toFixed()}] (ProxTop: ${proximityTop.toFixed()})`));
                                 break;
                             }
                         }
                     } else { // side === 'short'
                         // Check if price closed below a nearby Bullish OB
                         for (const ob of results.active_bull_boxes) {
                              const proximityBottom = ob.bottom.mul(exitProxFactor); // Price needs to close clearly below bottom
                              if (currentPrice.lt(proximityBottom)) {
                                 signal = "EXIT_SHORT";
                                 logger.warn(yellow(`Signal (${symbol}): EXIT_SHORT - Price ${currentPrice.toFixed()} closed below Bull OB ${ob.id} [${ob.bottom.toFixed()}-${ob.top.toFixed()}] (ProxBot: ${proximityBottom.toFixed()})`));
                                 break;
                             }
                         }
                     }
                 }
            } // End exit logic
        } else {
            logger.warn(yellow(`Signal generation skipped for ${symbol}: Missing last close price or trend direction.`));
        }

        results.signal = signal;
        results.dataframe = dfCalc; // Store the DataFrame with indicators

        const functionEnd = performance.now();
        logger.debug(`Strategy calculation for ${symbol} took ${(functionEnd - functionStart).toFixed(2)} ms. Final Signal: ${results.signal}`);
        return results;

    } catch (e) {
        logger.error(red(`Error during strategy calculation for ${symbol}: ${e.message}`), { stack: e.stack });
        results.dataframe = dfCalc; // Return potentially partial DataFrame
        return results; // Return default/partial results on error
    }
}


/** Main analysis and trading logic loop for a single symbol config. */
async function analyzeAndTradeSymbol(exchange, symbolConfig, globalConfig, logger) {
    const symbol = symbolConfig.name;
    logger.info(magenta(bold(`\n===== Cycle Start: ${symbol} =====`)));
    const startTime = performance.now();
    let currentPosition = null; // Store position info retrieved during the cycle

    try {
        // === 1. Get Market Info ===
        const marketInfo = await getMarketInfo(exchange, symbol, logger);
        if (!marketInfo) throw new Error(`Failed to get market info for ${symbol}. Cannot proceed.`);
        if (!marketInfo.active) { logger.warn(yellow(`Market ${symbol} is inactive. Skipping cycle.`)); return; }
        const priceTick = marketInfo.price_precision_step_decimal;
        if (!priceTick || priceTick.lte(0)) throw new Error(`Invalid price tick size for ${symbol}.`);

        // === 2. Fetch Kline Data ===
        const timeframeKey = symbolConfig.interval || "5"; // Default to 5m if not specified
        if (!CCXT_INTERVAL_MAP[timeframeKey]) throw new Error(`Invalid interval key '${timeframeKey}' for ${symbol}.`);
        const fetchLimit = symbolConfig.fetch_limit || globalConfig.fetch_limit || DEFAULT_FETCH_LIMIT;
        const dfRaw = await fetchKlinesCcxt(exchange, symbol, timeframeKey, fetchLimit, logger);
        if (!dfRaw) throw new Error(`Failed to fetch klines for ${symbol}. Cannot proceed.`);

        // === 3. Calculate Strategy Signals ===
        // Pass symbol-specific config merged with potential global defaults if needed
        const strategyResults = await calculateStrategySignals(dfRaw, symbolConfig, logger, exchange, symbol);
        if (!strategyResults || !strategyResults.last_close) {
            throw new Error(`Strategy calculation failed or returned invalid results for ${symbol}.`);
        }
        const { last_close, current_trend_up, atr, signal: trade_signal, active_bull_boxes, active_bear_boxes } = strategyResults;
        logger.info(`Strategy Analysis (${symbol}): LastClose=${last_close.toFixed()}, TrendUp=${current_trend_up}, ATR=${atr?.toFixed(5) ?? 'N/A'}, Signal='${trade_signal}'`);
        logger.debug(` Active Bull OBs: ${active_bull_boxes.length}, Active Bear OBs: ${active_bear_boxes.length}`);

        // === 4. Check Current Position ===
        currentPosition = await getOpenPosition(exchange, symbol, logger);

        // === 5. Manage Existing Position ===
        if (currentPosition) {
            const posSide = currentPosition.side;
            const posSize = currentPosition.size_decimal; // Absolute size
            const entryPrice = currentPosition.entryPrice; // Decimal
            // Get current protection from position info (API state)
            const currentSlStr = currentPosition.stopLossPrice;
            const currentTpStr = currentPosition.takeProfitPrice;
            const currentTslStr = currentPosition.trailingStopLoss; // Distance string
            const currentTslActStr = currentPosition.tslActivationPrice; // Activation price string

            logger.info(cyan(bold(`# Managing Existing ${posSide.toUpperCase()} Position (${symbol}):`)) +
                        ` Size=${posSize.toFixed()}, Entry=${entryPrice.toFixed()}, ` +
                        `API SL=${currentSlStr ?? 'N/A'}, TP=${currentTpStr ?? 'N/A'}, TSL=${currentTslStr ?? 'N/A'}`);

            // --- 5a. Check for Strategy Exit Signal ---
            const shouldExit = (posSide === 'long' && trade_signal === "EXIT_LONG") || (posSide === 'short' && trade_signal === "EXIT_SHORT");
            if (shouldExit) {
                logger.warn(bold(yellow(`>>> Strategy Exit Signal '${trade_signal}' triggered for ${posSide} position on ${symbol} <<<`)));
                if (globalConfig.enable_trading) {
                    // Optional: Cancel existing SL/TP orders if placed separately (less common with set-trading-stop)
                    // logger.info("Attempting market close order to exit position...");
                    const closeSize = posSize; // Use the absolute size fetched
                    const orderResult = await placeTrade(exchange, symbol, trade_signal, closeSize, marketInfo, logger, true); // reduceOnly=true
                    if (orderResult) {
                        logger.info(green(`Position exit order placed successfully for ${symbol}.`));
                        // Maybe wait briefly and re-check position is flat?
                    } else {
                        logger.error(red(`CRITICAL: Failed to place position exit order for ${symbol}! Manual intervention may be required.`));
                        // Potentially try setting SL to entry as emergency stop?
                    }
                } else {
                    logger.warn(yellow(`Trading disabled: Would place ${posSide} exit order for ${symbol}.`));
                }
                return; // Exit the cycle after placing/simulating close order
            }

            // --- 5b. Manage Position Protection (BE, TSL) ---
            const protConfig = symbolConfig.protection || {}; // Get symbol-specific protection settings
            const enableBE = protConfig.enable_break_even;
            const enableTSL = protConfig.enable_trailing_stop;
            const currentPrice = last_close; // Use last close as current price for checks

            // Can only manage protection if ATR is available
            if ((enableBE || enableTSL) && atr && atr.gt(0)) {
                 logger.debug(`Managing protection for ${symbol}. BE Enabled: ${enableBE}, TSL Enabled: ${enableTSL}. Current Price: ${currentPrice.toFixed()}, ATR: ${atr.toFixed()}`);
                 const atrDec = atr; // Already Decimal

                 // Check if TSL is already active according to the API
                 const isTslActiveApi = !!(currentTslStr && currentTslStr !== '0');
                 logger.debug(`TSL Active (API): ${isTslActiveApi}`);

                 let slToSet = safeMarketDecimal(currentSlStr, 'current_sl', false); // Start with current SL
                 let tpToSet = safeMarketDecimal(currentTpStr, 'current_tp', false); // Start with current TP
                 let tslDistToSet = safeMarketDecimal(currentTslStr, 'current_tsl_dist', true); // Start with current TSL dist (can be 0)
                 let tslActToSet = safeMarketDecimal(currentTslActStr, 'current_tsl_act', false); // Start with current TSL act
                 let protectionNeedsUpdate = false; // Flag to track if API call is needed

                 // --- Break-Even Logic ---
                 // Apply BE only if enabled AND TSL is NOT currently active on the exchange
                 if (enableBE && !isTslActiveApi) {
                     const beTriggerMult = new Decimal(protConfig.break_even_trigger_atr_multiple || 1.0);
                     const beOffsetTicks = new Decimal(protConfig.break_even_offset_ticks || 2);
                     let beTargetPrice = null;
                     let potentialBeStopPrice = null;

                     if (posSide === 'long') {
                         beTargetPrice = entryPrice.add(atrDec.mul(beTriggerMult));
                         if (currentPrice.gte(beTargetPrice)) {
                             potentialBeStopPrice = entryPrice.add(priceTick.mul(beOffsetTicks)); // Move SL slightly above entry
                         }
                     } else { // short
                         beTargetPrice = entryPrice.sub(atrDec.mul(beTriggerMult));
                         if (currentPrice.lte(beTargetPrice)) {
                             potentialBeStopPrice = entryPrice.sub(priceTick.mul(beOffsetTicks)); // Move SL slightly below entry
                         }
                     }

                     if (potentialBeStopPrice && potentialBeStopPrice.gt(0)) {
                         logger.info(`BE Trigger Met (${posSide}, ${symbol}): Current=${currentPrice.toFixed()}, Target=${beTargetPrice.toFixed()}, Potential BE SL=${potentialBeStopPrice.toFixed()}`);
                         // Check if current SL is already at or better than the BE price
                         let currentSlIsWorse = true;
                         if (slToSet) { // If a current SL exists
                             if (posSide === 'long' && slToSet.gte(potentialBeStopPrice)) currentSlIsWorse = false;
                             if (posSide === 'short' && slToSet.lte(potentialBeStopPrice)) currentSlIsWorse = false;
                         }

                         if (currentSlIsWorse) {
                             logger.warn(bold(magenta(`>>> Applying Break-Even SL for ${symbol} at ${potentialBeStopPrice.toFixed()} <<<`)));
                             slToSet = potentialBeStopPrice; // Update the SL to be set
                             // Ensure TSL remains off if applying BE
                             tslDistToSet = new Decimal(0);
                             tslActToSet = null;
                             protectionNeedsUpdate = true;
                         } else {
                              logger.info(`BE (${symbol}): Current SL (${slToSet?.toFixed() ?? 'N/A'}) is already at/better than potential BE SL (${potentialBeStopPrice.toFixed()}). No BE update needed.`);
                         }
                     } else if (beTargetPrice) {
                         logger.debug(`BE not triggered (${symbol}): Price ${currentPrice.toFixed()} hasn't reached target ${beTargetPrice.toFixed()}.`);
                     }
                 } // End BE Logic

                 // --- Trailing Stop Activation Logic ---
                 // Apply TSL only if enabled AND TSL is NOT already active on the exchange
                 // TSL activation takes precedence over BE if both trigger conditions are met in the same cycle? Or BE first? Let's check BE first.
                 // Only check TSL activation IF BE wasn't just applied (i.e., protectionNeedsUpdate is still false or was only for TP)
                 // AND TSL is not already active from API
                 if (enableTSL && !isTslActiveApi && !protectionNeedsUpdate) { // Check TSL only if BE didn't trigger an update
                     const tslActPerc = new Decimal(protConfig.trailing_stop_activation_percentage || 0.003); // e.g., 0.3% move from entry
                     const tslCallbackRate = new Decimal(protConfig.trailing_stop_callback_rate || 0.005); // e.g., 0.5% trail distance
                     let tslTriggerPrice = null;
                     let potentialTslDist = null;
                     let potentialTslAct = null;

                     if (tslActPerc.gt(0) && tslCallbackRate.gt(0)) {
                         if (posSide === 'long') {
                             tslTriggerPrice = entryPrice.mul(Decimal.add(1, tslActPerc));
                             if (currentPrice.gte(tslTriggerPrice)) {
                                 // Activation price is typically the price at which TSL becomes active (e.g., the trigger price itself, or current price)
                                 // Bybit V5 uses 'activePrice'. Let's use the trigger price.
                                 potentialTslAct = tslTriggerPrice;
                                 // TSL distance is often calculated based on activation price or entry price. Let's use activation price * callback.
                                 potentialTslDist = potentialTslAct.mul(tslCallbackRate);
                             }
                         } else { // short
                             tslTriggerPrice = entryPrice.mul(Decimal.sub(1, tslActPerc));
                             if (currentPrice.lte(tslTriggerPrice)) {
                                 potentialTslAct = tslTriggerPrice;
                                 potentialTslDist = potentialTslAct.mul(tslCallbackRate);
                             }
                         }

                         if (potentialTslDist && potentialTslDist.gt(0) && potentialTslAct) {
                             logger.warn(bold(magenta(`>>> Activating Trailing Stop Loss for ${symbol} <<<`)));
                             logger.warn(magenta(`  Trigger: ${tslTriggerPrice.toFixed()}, Activation Price: ${potentialTslAct.toFixed()}, Distance: ${potentialTslDist.toFixed()}`));
                             // Set TSL parameters, explicitly clear fixed SL as TSL overrides it
                             slToSet = null; // Clear fixed SL
                             tslDistToSet = potentialTslDist;
                             tslActToSet = potentialTslAct;
                             protectionNeedsUpdate = true;
                         } else if (tslTriggerPrice) {
                              logger.debug(`TSL not activated (${symbol}): Price ${currentPrice.toFixed()} hasn't reached activation trigger ${tslTriggerPrice.toFixed()}.`);
                         }
                     } else {
                          logger.warn(yellow(`TSL skipped (${symbol}): Invalid activation percentage (${tslActPerc.toFixed()}) or callback rate (${tslCallbackRate.toFixed()}). Must be > 0.`));
                     }
                 } // End TSL Activation Logic

                 // --- Call Protection Update if Needed ---
                 if (protectionNeedsUpdate) {
                     logger.info(`Updating position protection for ${symbol}...`);
                     const protectSuccess = await _set_position_protection(
                         exchange, symbol, marketInfo, currentPosition, logger,
                         slToSet,    // Pass the potentially updated SL (null if TSL activated)
                         tpToSet,    // Pass the existing/unchanged TP
                         tslDistToSet, // Pass the potentially updated TSL distance (0 if BE applied)
                         tslActToSet   // Pass the potentially updated TSL activation (null if BE applied or TSL off)
                     );
                     if (protectSuccess) logger.info(green(`Protection updated successfully for ${symbol}.`));
                     else logger.error(red(`Failed to update protection for ${symbol}!`));
                 } else {
                     logger.debug(`No protection updates required for ${symbol} this cycle.`);
                 }

            } else {
                 logger.debug(`Skipping BE/TSL checks for ${symbol}: Disabled in config, or ATR unavailable.`);
            }

        // === 6. Enter New Position ===
        } else if (trade_signal === "BUY" || trade_signal === "SELL") {
            logger.info(cyan(bold(`# Strategy Entry Signal '${trade_signal}' triggered for ${symbol}. Evaluating entry...`)));

            // --- 6a. Pre-Entry Checks ---
            if (!globalConfig.enable_trading) {
                logger.warn(yellow(`Trading disabled: Would evaluate ${trade_signal} entry for ${symbol}.`));
                return; // Stop cycle if trading disabled
            }
            if (!atr || atr.lte(0)) {
                logger.error(red(`Cannot enter ${symbol}: Invalid ATR (${atr?.toFixed()}).`));
                return; // Stop cycle if ATR is invalid
            }

            // --- 6b. Calculate Initial SL and TP ---
            const protConfig = symbolConfig.protection || {};
            const slAtrMult = new Decimal(protConfig.initial_stop_loss_atr_multiple || 1.8);
            const tpAtrMult = new Decimal(protConfig.initial_take_profit_atr_multiple || 0.7); // Set to 0 or null in config to disable TP
            let initialSlPrice = null;
            let initialTpPrice = null; // Use null to indicate "don't set TP" vs 0 which means "clear TP"

            if (trade_signal === "BUY") {
                initialSlPrice = last_close.sub(atr.mul(slAtrMult));
                if (tpAtrMult.gt(0)) initialTpPrice = last_close.add(atr.mul(tpAtrMult));
            } else { // SELL
                initialSlPrice = last_close.add(atr.mul(slAtrMult));
                if (tpAtrMult.gt(0)) initialTpPrice = last_close.sub(atr.mul(tpAtrMult));
            }

            // Validate calculated SL/TP
            if (!initialSlPrice || initialSlPrice.lte(0)) {
                logger.error(red(`Cannot enter ${symbol}: Invalid initial SL price calculated (${initialSlPrice?.toFixed()}).`));
                return;
            }
            if (initialTpPrice && initialTpPrice.lte(0)) {
                logger.warn(yellow(`Initial TP price calculated negative or zero (${initialTpPrice.toFixed()}). Disabling initial TP for ${symbol}.`));
                initialTpPrice = null; // Don't set invalid TP
            }
            logger.info(`Calculated Initial Protection (${symbol}): SL=${initialSlPrice.toFixed()}, TP=${initialTpPrice?.toFixed() ?? 'Disabled'}`);

            // --- 6c. Calculate Position Size ---
            const balance = await fetchBalance(exchange, QUOTE_CURRENCY, logger);
            if (!balance || balance.lte(0)) {
                logger.error(red(`Cannot enter ${symbol}: Invalid or zero balance (${balance?.toFixed()}) for ${QUOTE_CURRENCY}.`));
                return;
            }
            const riskPerTrade = symbolConfig.risk_per_trade || globalConfig.risk_per_trade;
            if (!riskPerTrade || riskPerTrade <= 0 || riskPerTrade > 1) {
                 logger.error(red(`Cannot enter ${symbol}: Invalid risk_per_trade setting (${riskPerTrade}).`));
                 return;
            }
            const positionSize = await calculatePositionSize(balance, riskPerTrade, initialSlPrice, last_close, marketInfo, exchange, logger);
            if (!positionSize || positionSize.lte(0)) {
                logger.error(red(`Cannot enter ${symbol}: Position sizing failed or resulted in zero size.`));
                return;
            }

            // --- 6d. Set Leverage ---
            const leverage = symbolConfig.leverage || 0; // Get from symbol config
            if (marketInfo.is_contract && leverage > 0) {
                logger.info(`Setting leverage to ${leverage}x for ${symbol}...`);
                const leverageSet = await setLeverageCcxt(exchange, symbol, leverage, marketInfo, logger);
                if (!leverageSet) {
                    logger.error(red(`Cannot enter ${symbol}: Failed to set leverage to ${leverage}x.`));
                    return; // Stop entry if leverage fails
                }
            } else if (marketInfo.is_contract) {
                 logger.warn(yellow(`Leverage setting skipped for ${symbol}: Leverage is 0 or not specified in config.`));
            }

            // --- 6e. Place Entry Order ---
            logger.warn(bold(magenta(`>>> Initiating ${trade_signal} Market Entry for ${symbol} | Size: ${positionSize.toFixed()} <<<`)));
            const orderResult = await placeTrade(exchange, symbol, trade_signal, positionSize, marketInfo, logger, false); // reduceOnly=false
            if (!orderResult || (orderResult.status !== 'closed' && orderResult.status !== 'filled' && !orderResult.info?.simulated)) {
                // If simulated, status might be different, check info.simulated
                 logger.error(red(`Entry order failed or did not fill immediately for ${symbol}. Order Result: ${JSON.stringify(orderResult)}`));
                 // Attempt to cancel if partially filled? Complex. For now, just report failure.
                 return; // Stop cycle if entry order fails
            }
             if (orderResult.info?.simulated) {
                 logger.warn(yellow(`Entry order was simulated for ${symbol}. Skipping protection setting.`));
                 return; // Don't try to set protection on a simulated order
             }

            // --- 6f. Confirm Position and Set Initial Protection ---
            const confirmDelayMs = (symbolConfig.position_confirm_delay_seconds ?? globalConfig.position_confirm_delay_seconds ?? POSITION_CONFIRM_DELAY_SECONDS) * 1000;
            logger.info(`Waiting ${confirmDelayMs / 1000}s to confirm position opening and set initial protection for ${symbol}...`);
            await delay(confirmDelayMs);

            const confirmedPosition = await getOpenPosition(exchange, symbol, logger); // Re-check position after delay
            if (confirmedPosition && confirmedPosition.side === side.toLowerCase()) {
                logger.info(green(`Position ${confirmedPosition.side.toUpperCase()} confirmed for ${symbol}. Size: ${confirmedPosition.size_decimal.toFixed()}. Setting initial protection...`));
                const protectSuccess = await _set_position_protection(
                    exchange, symbol, marketInfo, confirmedPosition, logger,
                    initialSlPrice, // Calculated SL
                    initialTpPrice, // Calculated TP (or null if disabled)
                    null,           // No initial TSL distance
                    null            // No initial TSL activation
                );
                if (protectSuccess) {
                    logger.info(green(`Initial SL/TP set successfully for new ${symbol} position.`));
                } else {
                    logger.error(red(`CRITICAL: Failed to set initial SL/TP for ${symbol} after entry! Position is currently unprotected. Manual intervention required.`));
                    // Consider emergency measures? e.g., market close? Very risky.
                }
            } else {
                // This is a critical failure state
                logger.error(red(bold(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)));
                logger.error(red(bold(`CRITICAL ERROR: Entry order placed for ${symbol} (Order ID: ${orderResult?.id ?? 'N/A'}),`)));
                logger.error(red(bold(`BUT position confirmation failed or side mismatch!`)));
                logger.error(red(bold(`  Expected Side: ${side.toLowerCase()}, Confirmed Position: ${JSON.stringify(confirmedPosition)}`)));
                logger.error(red(bold(`  MANUAL INTERVENTION REQUIRED IMMEDIATELY on exchange for ${symbol}!`)));
                logger.error(red(bold(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)));
                // Definitely do NOT proceed. Maybe trigger an alert.
                // Consider stopping the bot entirely in this state?
                shutdownRequested = true; // Request shutdown on critical error
            }

        // === 7. No Action Needed ===
        } else {
            // Condition: No open position AND no entry signal
            if (!currentPosition && trade_signal === 'NONE') {
                logger.info(`No open position and no entry signal for ${symbol}. Holding pattern.`);
            }
            // Condition: Open position but no exit signal and no protection updates needed (handled in section 5)
            else if (currentPosition && trade_signal === 'NONE') {
                 logger.info(`Holding existing ${currentPosition.side} position for ${symbol}. No exit signal or protection updates needed.`);
            }
        }

    } catch (e) {
        logger.error(red(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`));
        logger.error(red(`!! Unhandled Error in Main Cycle for ${symbol}: ${e.message} !!`));
        logger.error(red(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`), { stack: e.stack });
        // Log current position state if available during error
        if (currentPosition) logger.error(red(`Position state at time of error: ${JSON.stringify(currentPosition)}`));
        // Consider if certain errors should trigger shutdown
        // e.g., if (e instanceof ccxt.AuthenticationError) shutdownRequested = true;
    } finally {
        const duration = (performance.now() - startTime) / 1000;
        logger.info(magenta(bold(`===== Cycle End: ${symbol} | Duration: ${duration.toFixed(2)}s =====\n`)));
    }
}


// --- Main Execution Loop & Shutdown Handler ---

/** Handles graceful shutdown signals. */
function signalHandler(signal) {
    initLogger.warn(yellow(`\nReceived ${signal}. Requesting graceful shutdown...`));
    shutdownRequested = true;
    // Optional: Add a timeout to force exit if shutdown takes too long
    setTimeout(() => {
        initLogger.error(red("Graceful shutdown timed out. Forcing exit."));
        process.exit(1);
    }, 30000); // Force exit after 30 seconds
}

process.on('SIGINT', signalHandler); // Ctrl+C
process.on('SIGTERM', signalHandler); // Kill/system shutdown

/** Main bot function */
async function main() {
    initLogger.info(bold("--- Pyrmethus Bot Starting ---"));
    const config = await loadConfig();
    const exchange = await initializeExchange();

    // Create loggers for each symbol
    const symbolLoggers = {};
    for (const symbolConfig of config.symbols) {
        symbolLoggers[symbolConfig.name] = setupLogger(symbolConfig.name);
    }

    const loopDelayMs = (config.loop_delay_seconds || LOOP_DELAY_SECONDS) * 1000;

    initLogger.info(cyan(`Starting main trading loop. Delay: ${loopDelayMs / 1000}s. Trading Enabled: ${config.enable_trading ? green('YES') : red('NO')}`));

    // --- Main Loop ---
    while (!shutdownRequested) {
        const loopStart = performance.now();
        initLogger.debug("--- New Loop Iteration ---");

        // Process symbols sequentially (can be parallelized if careful about shared state/rate limits)
        for (const symbolConfig of config.symbols) {
            if (shutdownRequested) break; // Check before processing each symbol
            const symbol = symbolConfig.name;
            const logger = symbolLoggers[symbol];
            try {
                 // Merge global config aspects (like enable_trading) with symbol specifics
                 const effectiveConfig = { ...config, ...symbolConfig };
                 await analyzeAndTradeSymbol(exchange, symbolConfig, config, logger);
            } catch (e) {
                 logger.error(red(`Unhandled exception during analyzeAndTradeSymbol for ${symbol}: ${e.message}`), { stack: e.stack });
                 // Decide if this should halt the bot or just skip the symbol for this loop
            }
        } // End of symbol loop

        const loopEnd = performance.now();
        const loopDuration = loopEnd - loopStart;
        const waitTime = Math.max(0, loopDelayMs - loopDuration);

        if (shutdownRequested) break; // Check if shutdown requested during the loop processing

        if (waitTime > 0) {
            initLogger.debug(`Loop finished in ${loopDuration.toFixed(0)}ms. Waiting ${waitTime.toFixed(0)}ms...`);
            await delay(waitTime);
        } else {
             initLogger.warn(yellow(`Loop processing time (${loopDuration.toFixed(0)}ms) exceeded target delay (${loopDelayMs}ms). Running next loop immediately.`));
        }
    } // End of while loop

    // --- Shutdown ---
    initLogger.info(bold("--- Shutdown sequence initiated ---"));
    // Add any cleanup tasks here (e.g., attempting to close open orders if configured)
    // Example: Check for open positions and try to close them (optional, risky)
    // if (config.close_positions_on_exit && config.enable_trading) {
    //    initLogger.warn(yellow("Attempting to close open positions before exiting..."));
    //    // Add logic to fetch all positions and place closing orders
    // }

    initLogger.info(bold("Pyrmethus Bot has completed its final cycle. Exiting."));
    // Flush logs before exiting
    await new Promise(resolve => winston.loggers.closeAll(resolve)); // Ensure logs are written
    process.exit(0);
}

// --- Start the Bot ---
main().catch(error => {
    initLogger.error(red(bold(`FATAL UNHANDLED ERROR in main execution: ${error.message}`)), { stack: error.stack });
    // Try to flush logs even on fatal error
    winston.loggers.closeAll(() => {
        process.exit(1);
    });
});
