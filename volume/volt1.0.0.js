// pyrmethus_volumatic_bot.js
// Enhanced trading bot (JavaScript/Node.js) incorporating Volumatic Trend + Pivot OB strategy
// with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
// Version 1.3.1-js: Upgraded from 1.3.0-js with full implementations, optimizations, and robust error handling.

/**
 * Pyrmethus Volumatic Bot: A Node.js Trading Bot for Bybit V5 (v1.3.1-js)
 *
 * Implements Volumatic Trend + Pivot Order Block strategy with advanced management.
 *
 * Key Features:
 * - Bybit V5 API via ccxt
 * - Volumatic Trend (EMA, ATR Bands, Volume Norm)
 * - Pivot High/Low Detection with Order Blocks
 * - Signal Generation (Trend + OB)
 * - Risk-based Position Sizing with Decimal.js
 * - Leverage, Market Orders, SL/TP, TSL, BE
 * - Robust API Interaction with Retries
 * - Danfo.js DataFrames for Data Processing
 * - Winston Logging with Nanocolors
 * - Luxon for Timezone Handling
 * - Graceful Shutdown
 */

// --- Core Libraries ---
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
Decimal.set({ precision: 28 });
dotenv.config();

// --- Constants ---
const BOT_VERSION = '1.3.1-js';
const API_KEY = process.env.BYBIT_API_KEY;
const API_SECRET = process.env.BYBIT_API_SECRET;
const CONFIG_FILE = 'config.json';
const LOG_DIRECTORY = 'bot_logs';
const DEFAULT_TIMEZONE = 'America/Chicago';
const TIMEZONE = process.env.TIMEZONE || DEFAULT_TIMEZONE;
const MAX_API_RETRIES = 3;
const RETRY_DELAY_SECONDS = 5;
const POSITION_CONFIRM_DELAY_SECONDS = 8;
const LOOP_DELAY_SECONDS = 15;
const BYBIT_API_KLINE_LIMIT = 1000;
const DEFAULT_FETCH_LIMIT = 750;
const MAX_DF_LEN = 2000;
const DEFAULT_VT_LENGTH = 40;
const DEFAULT_VT_ATR_PERIOD = 200;
const DEFAULT_VT_VOL_EMA_LENGTH = 950;
const DEFAULT_VT_ATR_MULTIPLIER = 3.0;
const DEFAULT_OB_SOURCE = 'Wicks';
const DEFAULT_PH_LEFT = 10;
const DEFAULT_PH_RIGHT = 10;
const DEFAULT_PL_LEFT = 10;
const DEFAULT_PL_RIGHT = 10;
const DEFAULT_OB_EXTEND = true;
const DEFAULT_OB_MAX_BOXES = 50;

// Global State
let QUOTE_CURRENCY = 'USDT';
let shutdownRequested = false;
let marketInfoCache = new Map();

// Timeframe Mapping
const VALID_INTERVALS = ['1', '3', '5', '15', '30', '60', '120', '240', 'D', 'W', 'M'];
const CCXT_INTERVAL_MAP = {
    '1': '1m', '3': '3m', '5': '5m', '15': '15m', '30': '30m',
    '60': '1h', '120': '2h', '240': '4h', 'D': '1d', 'W': '1w', 'M': '1M'
};

// Nanocolors Shortcuts
const { green, cyan, magenta, yellow, red, blue, gray, bold, dim } = nc;

// Ensure Log Directory
(async () => {
    try {
        await fs.mkdir(LOG_DIRECTORY, { recursive: true });
    } catch (e) {
        console.error(red(`FATAL: Could not create log directory '${LOG_DIRECTORY}': ${e}`));
        process.exit(1);
    }
})();

// --- Type Definitions (JSDoc) ---
/**
 * @typedef {object} OrderBlock
 * @property {string} id
 * @property {'bull' | 'bear'} type
 * @property {number} timestamp
 * @property {Decimal} top
 * @property {Decimal} bottom
 * @property {boolean} active
 * @property {boolean} violated
 * @property {number | null} violation_ts
 * @property {number | null} extended_to_ts
 */

/**
 * @typedef {object} StrategyAnalysisResults
 * @property {dfd.DataFrame | null} dataframe
 * @property {Decimal | null} last_close
 * @property {boolean | null} current_trend_up
 * @property {boolean} trend_just_changed
 * @property {OrderBlock[]} active_bull_boxes
 * @property {OrderBlock[]} active_bear_boxes
 * @property {number | null} vol_norm_int
 * @property {Decimal | null} atr
 * @property {Decimal | null} upper_band
 * @property {Decimal | null} lower_band
 * @property {'BUY' | 'SELL' | 'EXIT_LONG' | 'EXIT_SHORT' | 'NONE'} signal
 */

/**
 * @typedef {object} MarketInfo
 * @property {string} id
 * @property {string} symbol
 * @property {string} base
 * @property {string} quote
 * @property {string} type
 * @property {boolean} contract
 * @property {boolean} linear
 * @property {boolean} inverse
 * @property {string} contract_type_str
 * @property {Decimal} amount_precision_step_decimal
 * @property {Decimal} price_precision_step_decimal
 * @property {Decimal} contract_size_decimal
 * @property {Decimal | null} min_amount_decimal
 * @property {Decimal | null} max_amount_decimal
 * @property {object} info
 */

/**
 * @typedef {object} PositionInfo
 * @property {string | null} id
 * @property {string} symbol
 * @property {'long' | 'short' | null} side
 * @property {Decimal} size_decimal
 * @property {Decimal} entryPrice
 * @property {string | null} stopLossPrice
 * @property {string | null} takeProfitPrice
 * @property {string | null} trailingStopLoss
 * @property {string | null} tslActivationPrice
 * @property {object} info
 * @property {boolean} be_activated
 * @property {boolean} tsl_activated
 */

// --- Utility Functions ---
/**
 * Delays execution for the specified time.
 * @param {number} ms - Delay in milliseconds.
 * @returns {Promise<void>}
 */
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Converts a value to Decimal safely.
 * @param {any} value - Value to convert.
 * @param {string} fieldName - Field name for logging.
 * @param {boolean} [allowZero=true] - Allow zero values.
 * @returns {Decimal | null}
 */
function safeMarketDecimal(value, fieldName = 'value', allowZero = true) {
    if (value == null) return null;
    try {
        const dVal = new Decimal(String(value).trim());
        if (!allowZero && dVal.isZero()) return null;
        if (dVal.isNegative()) return null;
        return dVal;
    } catch (e) {
        return null;
    }
}

/**
 * Formats a price to exchange precision.
 * @param {ccxt.Exchange} exchange
 * @param {string} symbol
 * @param {Decimal | number | string} price
 * @returns {string | null}
 */
function formatPrice(exchange, symbol, price) {
    try {
        const priceDecimal = new Decimal(String(price));
        if (priceDecimal.isNaN() || priceDecimal.isNegative()) return null;
        const formattedStr = exchange.priceToPrecision(symbol, priceDecimal.toNumber());
        return new Decimal(formattedStr).gt(0) || priceDecimal.isZero() ? formattedStr : null;
    } catch (e) {
        return null;
    }
}

// --- Logging Setup ---
const loggers = {};

/**
 * Sets up a Winston logger.
 * @param {string} name - Logger name.
 * @returns {winston.Logger}
 */
function setupLogger(name) {
    const safeName = name.replace(/[/:]/g, '_');
    const loggerName = `pyrmethus_${safeName}`;
    if (loggers[loggerName]) return loggers[loggerName];

    const logFilename = path.join(LOG_DIRECTORY, `${loggerName}.log`);

    const consoleFormat = winston.format.printf(({ level, message, timestamp, label, ...meta }) => {
        const ts = DateTime.fromJSDate(new Date(timestamp)).setZone(TIMEZONE).toFormat('HH:mm:ss');
        let color = blue;
        if (level === 'error') color = red;
        else if (level === 'warn') color = yellow;
        else if (level === 'info') color = cyan;
        else if (level === 'debug') color = gray;

        let levelString = level.toUpperCase().padEnd(8);
        if (level === 'error') levelString = bold(levelString);
        if (level === 'debug') levelString = dim(levelString);

        let redactedMessage = String(message).replace(API_KEY || '', '***API_KEY***').replace(API_SECRET || '', '***API_SECRET***');
        const metaString = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : '';

        return `${blue(ts)} - ${color(levelString)} - ${magenta(`[${label}]`)} - ${redactedMessage}${metaString}`;
    });

    const fileFormat = winston.format.printf(({ level, message, timestamp, label, ...meta }) => {
        let redactedMessage = String(message).replace(API_KEY || '', '***API_KEY***').replace(API_SECRET || '', '***API_SECRET***');
        const metaString = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : '';
        return `${timestamp} ${level.toUpperCase().padEnd(8)} [${label}] ${redactedMessage}${metaString}`;
    });

    const logger = winston.createLogger({
        level: 'debug',
        format: winston.format.combine(
            winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
            winston.format.errors({ stack: true }),
            winston.format.splat(),
            winston.format.label({ label: name })
        ),
        transports: [
            new winston.transports.Console({
                level: process.env.CONSOLE_LOG_LEVEL || 'info',
                format: winston.format.combine(consoleFormat),
                handleExceptions: true
            }),
            new winston.transports.File({
                filename: logFilename,
                level: 'debug',
                format: winston.format.combine(fileFormat),
                maxsize: 10 * 1024 * 1024,
                maxFiles: 5,
                tailable: true,
                zippedArchive: true,
                handleExceptions: true
            })
        ],
        exitOnError: false
    });

    loggers[loggerName] = logger;
    return logger;
}

const initLogger = setupLogger('init');
initLogger.info(magenta(`Pyrmethus Volumatic Bot v${BOT_VERSION} awakening...`));

// --- Configuration ---
/**
 * Ensures config keys exist with defaults.
 * @param {object} config
 * @param {object} defaultConfig
 * @param {string} [parentKey='']
 * @returns {{ updatedConfig: object, changed: boolean }}
 */
function ensureConfigKeys(config, defaultConfig, parentKey = '') {
    let updatedConfig = { ...config };
    let changed = false;
    for (const key in defaultConfig) {
        const fullKeyPath = parentKey ? `${parentKey}.${key}` : key;
        if (!(key in updatedConfig)) {
            updatedConfig[key] = defaultConfig[key];
            changed = true;
            initLogger.info(yellow(`Config: Added '${fullKeyPath}' with default: ${JSON.stringify(defaultConfig[key])}`));
        } else if (typeof defaultConfig[key] === 'object' && !Array.isArray(defaultConfig[key])) {
            const nestedResult = ensureConfigKeys(updatedConfig[key], defaultConfig[key], fullKeyPath);
            if (nestedResult.changed) {
                updatedConfig[key] = nestedResult.updatedConfig;
                changed = true;
            }
        }
    }
    return { updatedConfig, changed };
}

/**
 * Loads and validates configuration.
 * @param {string} filepath
 * @returns {Promise<object>}
 */
async function loadConfig(filepath) {
    initLogger.info(cyan(`Conjuring configuration from '${filepath}'...`));
    const defaultConfig = {
        trading_pairs: ['BTC/USDT'],
        interval: '5',
        retry_delay: RETRY_DELAY_SECONDS,
        fetch_limit: DEFAULT_FETCH_LIMIT,
        orderbook_limit: 25,
        enable_trading: false,
        use_sandbox: true,
        risk_per_trade: 0.01,
        leverage: 20,
        max_concurrent_positions: 1,
        quote_currency: 'USDT',
        loop_delay_seconds: LOOP_DELAY_SECONDS,
        position_confirm_delay_seconds: POSITION_CONFIRM_DELAY_SECONDS,
        strategy_params: {
            vt_length: DEFAULT_VT_LENGTH,
            vt_atr_period: DEFAULT_VT_ATR_PERIOD,
            vt_vol_ema_length: DEFAULT_VT_VOL_EMA_LENGTH,
            vt_atr_multiplier: DEFAULT_VT_ATR_MULTIPLIER,
            ob_source: DEFAULT_OB_SOURCE,
            ph_left: DEFAULT_PH_LEFT,
            ph_right: DEFAULT_PH_RIGHT,
            pl_left: DEFAULT_PL_LEFT,
            pl_right: DEFAULT_PL_RIGHT,
            ob_extend: DEFAULT_OB_EXTEND,
            ob_max_boxes: DEFAULT_OB_MAX_BOXES,
            ob_entry_proximity_factor: 1.005,
            ob_exit_proximity_factor: 1.001
        },
        protection: {
            enable_trailing_stop: true,
            trailing_stop_callback_rate: 0.005,
            trailing_stop_activation_percentage: 0.003,
            enable_break_even: true,
            break_even_trigger_atr_multiple: 1.0,
            break_even_offset_ticks: 2,
            initial_stop_loss_atr_multiple: 1.8,
            initial_take_profit_atr_multiple: 0.7
        }
    };

    let configNeedsSaving = false;
    let loadedConfig = {};

    try {
        const fileContent = await fs.readFile(filepath, 'utf-8');
        loadedConfig = JSON.parse(fileContent);
        if (typeof loadedConfig !== 'object' || loadedConfig === null) throw new TypeError('Invalid JSON object');
    } catch (error) {
        if (error.code === 'ENOENT') {
            initLogger.warn(yellow(`Config '${filepath}' not found. Crafting default.`));
            await fs.writeFile(filepath, JSON.stringify(defaultConfig, null, 4));
            QUOTE_CURRENCY = defaultConfig.quote_currency;
            return defaultConfig;
        }
        initLogger.error(red(`Error loading config: ${error}. Using defaults.`));
        QUOTE_CURRENCY = defaultConfig.quote_currency;
        return defaultConfig;
    }

    const ensureResult = ensureConfigKeys(loadedConfig, defaultConfig);
    let updatedConfig = ensureResult.updatedConfig;
    if (ensureResult.changed) configNeedsSaving = true;

    const validateNumeric = (cfg, keyPath, minVal, maxVal, isStrictMin = false, isInt = false, allowZero = false) => {
        const keys = keyPath.split('.');
        let current = cfg;
        let def = defaultConfig;
        for (let i = 0; i < keys.length - 1; i++) {
            current = current[keys[i]];
            def = def[keys[i]];
        }
        const leafKey = keys[keys.length - 1];
        let value = current[leafKey];
        const defaultVal = def[leafKey];

        try {
            const numVal = new Decimal(String(value));
            const minDec = new Decimal(minVal);
            const maxDec = new Decimal(maxVal);
            const valid = (isStrictMin ? numVal.gt(minDec) : numVal.gte(minDec)) && numVal.lte(maxDec) || (allowZero && numVal.isZero());
            if (!valid) throw new Error(`Value out of range [${minVal}, ${maxVal}]${allowZero ? ' or 0' : ''}`);
            if (isInt && !numVal.isInteger()) {
                value = numVal.toInteger();
                current[leafKey] = value;
                configNeedsSaving = true;
                initLogger.info(yellow(`Corrected '${keyPath}' to integer: ${value}`));
            }
        } catch (e) {
            initLogger.warn(yellow(`Invalid '${keyPath}': ${value}. Using default: ${defaultVal}`));
            current[leafKey] = defaultVal;
            configNeedsSaving = true;
        }
    };

    const validateBoolean = (cfg, keyPath) => {
        const keys = keyPath.split('.');
        let current = cfg;
        let def = defaultConfig;
        for (let i = 0; i < keys.length - 1; i++) {
            current = current[keys[i]];
            def = def[keys[i]];
        }
        const leafKey = keys[keys.length - 1];
        if (typeof current[leafKey] !== 'boolean') {
            initLogger.warn(yellow(`Invalid '${keyPath}': Must be boolean. Using default: ${def[leafKey]}`));
            current[leafKey] = def[leafKey];
            configNeedsSaving = true;
        }
    };

    initLogger.debug('Validating configuration...');
    if (!Array.isArray(updatedConfig.trading_pairs) || !updatedConfig.trading_pairs.every(s => typeof s === 'string')) {
        updatedConfig.trading_pairs = defaultConfig.trading_pairs;
        configNeedsSaving = true;
    }
    if (!VALID_INTERVALS.includes(updatedConfig.interval)) {
        updatedConfig.interval = defaultConfig.interval;
        configNeedsSaving = true;
    }
    validateNumeric(updatedConfig, 'risk_per_trade', 0, 1, true);
    validateNumeric(updatedConfig, 'leverage', 0, 200, false, true, true);
    validateBoolean(updatedConfig, 'enable_trading');
    validateBoolean(updatedConfig, 'use_sandbox');
    validateNumeric(updatedConfig, 'strategy_params.vt_length', 1, 500, false, true);
    validateNumeric(updatedConfig, 'strategy_params.vt_atr_multiplier', 0.1, 20.0);
    validateBoolean(updatedConfig, 'protection.enable_trailing_stop');
    validateNumeric(updatedConfig, 'protection.initial_stop_loss_atr_multiple', 0.1, 100.0, true);

    if (configNeedsSaving) {
        try {
            await fs.writeFile(filepath, JSON.stringify(updatedConfig, null, 4));
            initLogger.info(green(`Updated config saved: ${filepath}`));
        } catch (e) {
            initLogger.error(red(`Failed to save config: ${e}`));
        }
    }

    QUOTE_CURRENCY = updatedConfig.quote_currency || 'USDT';
    initLogger.info(`Quote currency: ${yellow(QUOTE_CURRENCY)}`);
    return updatedConfig;
}

let CONFIG = {};

// --- Exchange Setup ---
/**
 * Initializes Bybit exchange via CCXT.
 * @param {winston.Logger} logger
 * @returns {Promise<ccxt.Exchange | null>}
 */
async function initializeExchange(logger) {
    logger.info(cyan(`Binding to Bybit exchange...`));
    try {
        const exchange = new ccxt.bybit({
            apiKey: API_KEY,
            secret: API_SECRET,
            enableRateLimit: true,
            options: {
                defaultType: 'linear',
                adjustForTimeDifference: true,
                fetchTickerTimeout: 15000,
                fetchBalanceTimeout: 20000,
                createOrderTimeout: 30000,
                fetchPositionsTimeout: 20000
            }
        });

        exchange.setSandboxMode(CONFIG.use_sandbox);
        logger.warn(CONFIG.use_sandbox ? yellow(`SANDBOX MODE`) : red(`LIVE MODE`));

        for (let attempt = 0; attempt < MAX_API_RETRIES; attempt++) {
            try {
                await exchange.loadMarkets(attempt > 0);
                if (Object.keys(exchange.markets).length) break;
                throw new Error('Empty markets');
            } catch (e) {
                if (attempt < MAX_API_RETRIES - 1) {
                    await delay(RETRY_DELAY_SECONDS * 1000 * (attempt + 1));
                } else {
                    logger.error(red(`Failed to load markets: ${e.message}`));
                    return null;
                }
            }
        }

        const balance = await fetchBalance(exchange, QUOTE_CURRENCY, logger);
        if (balance) {
            logger.info(green(`Initial balance: ${balance.toFixed()} ${QUOTE_CURRENCY}`));
            return exchange;
        }
        if (CONFIG.enable_trading) {
            logger.error(red(`Balance fetch failed. Cannot trade.`));
            return null;
        }
        logger.warn(yellow(`Balance fetch failed. Proceeding without balance.`));
        return exchange;
    } catch (e) {
        logger.error(red(`Exchange initialization failed: ${e.message}`));
        return null;
    }
}

/**
 * Fetches account balance.
 * @param {ccxt.Exchange} exchange
 * @param {string} currency
 * @param {winston.Logger} logger
 * @returns {Promise<Decimal | null>}
 */
async function fetchBalance(exchange, currency, logger) {
    for (let attempt = 0; attempt < MAX_API_RETRIES; attempt++) {
        try {
            const balance = await exchange.fetchBalance();
            const free = safeMarketDecimal(balance?.[currency]?.free, `balance_${currency}`, true);
            if (free) {
                logger.debug(`Balance: ${free.toFixed()} ${currency}`);
                return free;
            }
            throw new Error(`No free balance for ${currency}`);
        } catch (e) {
            logger.warn(`Balance fetch attempt ${attempt + 1} failed: ${e.message}`);
            if (attempt < MAX_API_RETRIES - 1) await delay(RETRY_DELAY_SECONDS * 1000 * (attempt + 1));
            else return null;
        }
    }
    logger.error(`Failed to fetch balance for ${currency}`);
    return null;
}

/**
 * Retrieves market information.
 * @param {ccxt.Exchange} exchange
 * @param {string} symbol
 * @param {winston.Logger} logger
 * @returns {Promise<MarketInfo | null>}
 */
async function getMarketInfo(exchange, symbol, logger) {
    if (marketInfoCache.has(symbol)) return marketInfoCache.get(symbol);

    try {
        if (!exchange.markets[symbol]) await exchange.loadMarkets(true);
        const market = exchange.market(symbol);
        if (!market) throw new Error(`Market ${symbol} not found`);

        const info = {
            id: market.id,
            symbol: market.symbol,
            base: market.base,
            quote: market.quote,
            type: market.type,
            contract: market.contract ?? false,
            linear: market.linear ?? false,
            inverse: market.inverse ?? false,
            contract_type_str: market.linear ? 'Linear' : market.inverse ? 'Inverse' : 'Spot',
            amount_precision_step_decimal: safeMarketDecimal(market.precision?.amount, 'amount_precision', false),
            price_precision_step_decimal: safeMarketDecimal(market.precision?.price, 'price_precision', false),
            contract_size_decimal: safeMarketDecimal(market.contractSize, 'contractSize', true) || new Decimal(1),
            min_amount_decimal: safeMarketDecimal(market.limits?.amount?.min, 'min_amount', true),
            max_amount_decimal: safeMarketDecimal(market.limits?.amount?.max, 'max_amount', true),
            info: market.info
        };

        if (!info.amount_precision_step_decimal || !info.price_precision_step_decimal) {
            logger.error(red(`Invalid precision for ${symbol}`));
            return null;
        }

        marketInfoCache.set(symbol, info);
        return info;
    } catch (e) {
        logger.error(`Failed to get market info for ${symbol}: ${e.message}`);
        return null;
    }
}

/**
 * Fetches OHLCV data as a DataFrame.
 * @param {ccxt.Exchange} exchange
 * @param {string} symbol
 * @param {string} timeframe
 * @param {number} limit
 * @param {winston.Logger} logger
 * @returns {Promise<dfd.DataFrame | null>}
 */
async function fetchKlinesCcxt(exchange, symbol, timeframe, limit, logger) {
    logger.info(cyan(`Gathering klines for ${symbol} | TF: ${timeframe} | Limit: ${limit}`));
    let allOhlcv = [];
    let since = undefined;
    const targetLimit = Math.min(limit, BYBIT_API_KLINE_LIMIT);

    for (let attempt = 0; attempt < MAX_API_RETRIES; attempt++) {
        try {
            while (allOhlcv.length < limit) {
                const ohlcv = await exchange.fetchOHLCV(symbol, timeframe, since, targetLimit);
                if (!ohlcv.length) break;
                allOhlcv.push(...ohlcv);
                since = ohlcv[ohlcv.length - 1][0] + 1;
                if (ohlcv.length < targetLimit) break;
            }
            break;
        } catch (e) {
            logger.warn(`Kline fetch attempt ${attempt + 1} failed: ${e.message}`);
            if (attempt < MAX_API_RETRIES - 1) await delay(RETRY_DELAY_SECONDS * 1000 * (attempt + 1));
            else {
                logger.error(`Failed to fetch klines for ${symbol}`);
                return null;
            }
        }
    }

    allOhlcv = allOhlcv.slice(-limit);
    if (!allOhlcv.length) {
        logger.warn(`No kline data for ${symbol}`);
        return null;
    }

    const data = allOhlcv.map(k => ({
        timestamp: k[0],
        open: safeMarketDecimal(k[1], 'open', false)?.toNumber() || 0,
        high: safeMarketDecimal(k[2], 'high', false)?.toNumber() || 0,
        low: safeMarketDecimal(k[3], 'low', false)?.toNumber() || 0,
        close: safeMarketDecimal(k[4], 'close', false)?.toNumber() || 0,
        volume: safeMarketDecimal(k[5], 'volume', true)?.toNumber() || 0
    }));

    let df = new dfd.DataFrame(data);
    df.setIndex({ column: 'timestamp', inplace: true });
    df.dropna({ axis: 0, inplace: true });

    if (df.shape[0] > MAX_DF_LEN) {
        df = df.iloc({ start: df.shape[0] - MAX_DF_LEN });
    }

    logger.info(green(`Processed ${df.shape[0]} klines for ${symbol}`));
    return df;
}

/**
 * Retrieves open position.
 * @param {ccxt.Exchange} exchange
 * @param {string} symbol
 * @param {winston.Logger} logger
 * @returns {Promise<PositionInfo | null>}
 */
async function getOpenPosition(exchange, symbol, logger) {
    for (let attempt = 0; attempt < MAX_API_RETRIES; attempt++) {
        try {
            const positions = await exchange.fetchPositions([symbol]);
            for (const pos of positions) {
                if (pos.contracts > 0) {
                    return {
                        id: pos.id || null,
                        symbol,
                        side: pos.side?.toLowerCase() || null,
                        size_decimal: safeMarketDecimal(pos.contracts, 'position_size', false),
                        entryPrice: safeMarketDecimal(pos.entryPrice, 'entry_price', false),
                        stopLossPrice: pos.info?.stopLoss || null,
                        takeProfitPrice: pos.info?.takeProfit || null,
                        trailingStopLoss: pos.info?.trailingStop || null,
                        tslActivationPrice: pos.info?.trailingStopActivationPrice || null,
                        info: pos.info,
                        be_activated: false,
                        tsl_activated: false
                    };
                }
            }
            return null;
        } catch (e) {
            logger.warn(`Position fetch attempt ${attempt + 1} failed: ${e.message}`);
            if (attempt < MAX_API_RETRIES - 1) await delay(RETRY_DELAY_SECONDS * 1000 * (attempt + 1));
            else {
                logger.error(`Failed to fetch position for ${symbol}`);
                return null;
            }
        }
    }
    return null;
}

/**
 * Calculates position size based on risk.
 * @param {Decimal} balance
 * @param {number} riskPerTrade
 * @param {Decimal} slPrice
 * @param {Decimal} entryPrice
 * @param {MarketInfo} marketInfo
 * @param {ccxt.Exchange} exchange
 * @param {winston.Logger} logger
 * @returns {Promise<Decimal | null>}
 */
async function calculatePositionSize(balance, riskPerTrade, slPrice, entryPrice, marketInfo, exchange, logger) {
    try {
        if (balance.lte(0) || entryPrice.lte(0) || slPrice.lte(0)) {
            logger.error(`Invalid inputs: balance=${balance}, entry=${entryPrice}, sl=${slPrice}`);
            return null;
        }
        const riskAmount = balance.mul(riskPerTrade);
        const priceDiff = entryPrice.sub(slPrice).abs();
        let size = riskAmount.div(priceDiff);
        const step = marketInfo.amount_precision_step_decimal;
        if (!step) {
            logger.error(`Invalid amount precision for ${marketInfo.symbol}`);
            return null;
        }
        size = size.div(step).floor().mul(step);
        const minSize = marketInfo.min_amount_decimal || new Decimal(0);
        if (size.lt(minSize)) {
            logger.warn(`Size ${size} below minimum ${minSize}. Using minimum.`);
            size = minSize;
        }
        const formattedSize = new Decimal(exchange.amountToPrecision(marketInfo.symbol, size.toNumber()));
        logger.info(green(`Position size: ${formattedSize.toFixed()} for ${marketInfo.symbol}`));
        return formattedSize;
    } catch (e) {
        logger.error(`Position sizing failed: ${e.message}`);
        return null;
    }
}

/**
 * Places a market order.
 * @param {ccxt.Exchange} exchange
 * @param {string} symbol
 * @param {'BUY' | 'SELL' | 'EXIT_LONG' | 'EXIT_SHORT'} signal
 * @param {Decimal} size
 * @param {MarketInfo} marketInfo
 * @param {winston.Logger} logger
 * @param {boolean} reduceOnly
 * @param {object} [params]
 * @returns {Promise<object | null>}
 */
async function placeTrade(exchange, symbol, signal, size, marketInfo, logger, reduceOnly = false, params = null) {
    const side = (signal === 'BUY' || signal === 'EXIT_SHORT') ? 'buy' : 'sell';
    const action = reduceOnly ? 'Close/Reduce' : 'Open/Increase';
    logger.info(bold(`===> ${action} | ${side.toUpperCase()} MARKET | ${symbol} | Size: ${size.toFixed()} <===`));

    if (!CONFIG.enable_trading) {
        logger.warn(`Trading disabled: Simulated ${side} order`);
        return { id: `sim_${Date.now()}`, status: 'simulated', filled: size.toNumber() };
    }

    for (let attempt = 0; attempt < MAX_API_RETRIES; attempt++) {
        try {
            const orderParams = { reduceOnly, ...(params || {}) };
            const order = await exchange.createOrder(symbol, 'market', side, size.toNumber(), undefined, orderParams);
            logger.info(green(`Order placed: ${side} ${size.toFixed()} @ market, ID: ${order.id}`));
            return order;
        } catch (e) {
            logger.warn(`Order attempt ${attempt + 1} failed: ${e.message}`);
            if (e instanceof ccxt.InsufficientFunds) {
                logger.error(red(`Insufficient funds for ${symbol}`));
                return null;
            }
            if (attempt < MAX_API_RETRIES - 1) await delay(RETRY_DELAY_SECONDS * 1000 * (attempt + 1));
            else {
                logger.error(red(`Failed to place order for ${symbol}`));
                return null;
            }
        }
    }
    return null;
}

/**
 * Sets position protection (SL/TP/TSL).
 * @param {ccxt.Exchange} exchange
 * @param {string} symbol
 * @param {MarketInfo} marketInfo
 * @param {PositionInfo} positionInfo
 * @param {winston.Logger} logger
 * @param {Decimal | null} slPrice
 * @param {Decimal | null} tpPrice
 * @param {Decimal | null} [tslDistance]
 * @param {Decimal | null} [tslActivation]
 * @returns {Promise<boolean>}
 */
async function _set_position_protection(exchange, symbol, marketInfo, positionInfo, logger, slPrice, tpPrice, tslDistance = null, tslActivation = null) {
    logger.info(`Setting protection for ${symbol}: SL=${slPrice?.toFixed() || 'None'}, TP=${tpPrice?.toFixed() || 'None'}`);
    if (!CONFIG.enable_trading) {
        logger.warn(`Trading disabled: Simulated protection`);
        return true;
    }

    try {
        const params = {
            symbol: marketInfo.id,
            positionIdx: 0, // Unified account
            stopLoss: slPrice ? formatPrice(exchange, symbol, slPrice) : '0',
            takeProfit: tpPrice && !tpPrice.isZero() ? formatPrice(exchange, symbol, tpPrice) : '0',
            trailingStop: tslDistance ? formatPrice(exchange, symbol, tslDistance) : '',
            triggerPrice: tslActivation ? formatPrice(exchange, symbol, tslActivation) : ''
        };
        await exchange.privatePostPositionTradingStop(params);
        logger.info(green(`Protection set: SL=${params.stopLoss}, TP=${params.takeProfit}`));
        return true;
    } catch (e) {
        logger.error(red(`Failed to set protection: ${e.message}`));
        return false;
    }
}

/**
 * Calculates strategy signals.
 * @param {dfd.DataFrame} df
 * @param {object} config
 * @param {winston.Logger} logger
 * @param {ccxt.Exchange} exchange
 * @param {string} symbol
 * @returns {Promise<StrategyAnalysisResults>}
 */
async function calculateStrategySignals(df, config, logger, exchange, symbol) {
    if (!df || df.shape[0] < 50) {
        logger.error(`Invalid DataFrame: ${df?.shape[0] || 0} rows`);
        return {
            dataframe: null,
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
    }

    const sp = config.strategy_params;
    const vtLen = sp.vt_length;
    const atrPeriod = sp.vt_atr_period;
    const volEmaLen = sp.vt_vol_ema_length;
    const atrMult = new Decimal(sp.vt_atr_multiplier);
    const phLeft = sp.ph_left;
    const phRight = sp.ph_right;
    const plLeft = sp.pl_left;
    const plRight = sp.pl_right;
    const obSource = sp.ob_source.toLowerCase();
    const obMaxBoxes = sp.ob_max_boxes;
    const entryProx = new Decimal(sp.ob_entry_proximity_factor);
    const exitProx = new Decimal(sp.ob_exit_proximity_factor);

    let dfCalc = df.copy();
    const close = dfCalc['close'].values;
    const high = dfCalc['high'].values;
    const low = dfCalc['low'].values;
    const volume = dfCalc['volume'].values;

    // Indicators
    const emaResult = EMA.calculate({ period: vtLen, values: close });
    const atrResult = ATR.calculate({ high, low, close, period: atrPeriod });
    dfCalc.addColumn(`EMA_${vtLen}`, Array(close.length - emaResult.length).fill(NaN).concat(emaResult), { inplace: true });
    dfCalc.addColumn('ATR', Array(close.length - atrResult.length).fill(NaN).concat(atrResult), { inplace: true });

    // Volume Normalization
    const volEmaResult = volume ? EMA.calculate({ period: volEmaLen, values: volume }) : [];
    const volEmaPadded = volume ? Array(close.length - volEmaResult.length).fill(NaN).concat(volEmaResult) : Array(close.length).fill(0);
    const volNorm = volume ? volume.map((v, i) => volEmaPadded[i] > 0 ? (v / volEmaPadded[i]) * 100 : 0) : Array(close.length).fill(0);
    dfCalc.addColumn('VolNormInt', volNorm.map(v => Math.min(Math.max(0, Math.round(v)), 500)), { inplace: true });

    // Bands
    const emaSeries = dfCalc[`EMA_${vtLen}`];
    const atrSeries = dfCalc['ATR'];
    dfCalc.addColumn('VT_UpperBand', emaSeries.add(atrSeries.mul(atrMult.toNumber())), { inplace: true });
    dfCalc.addColumn('VT_LowerBand', emaSeries.sub(atrSeries.mul(atrMult.toNumber())), { inplace: true });

    // Trend
    dfCalc.addColumn('TrendUp', dfCalc['close'].gt(dfCalc['VT_UpperBand']), { inplace: true });
    dfCalc.addColumn('TrendDown', dfCalc['close'].lt(dfCalc['VT_LowerBand']), { inplace: true });
    const trendUp = dfCalc['TrendUp'].values;
    const trendChanged = trendUp.map((up, i) => i > 0 && up !== trendUp[i - 1]);
    dfCalc.addColumn('TrendChanged', trendChanged, { inplace: true });

    // Pivots
    const source = obSource === 'wicks' ? high : close; // Use high for pivot high detection
    const pivotHigh = Array(close.length).fill(null);
    const pivotLow = Array(close.length).fill(null);
    for (let i = phLeft; i < close.length - phRight; i++) {
        const highWindow = source.slice(i - phLeft, i + phRight + 1);
        const lowWindow = low.slice(i - plLeft, i + plRight + 1);
        if (source[i] === Math.max(...highWindow)) pivotHigh[i] = high[i];
        if (low[i] === Math.min(...lowWindow)) pivotLow[i] = low[i];
    }
    dfCalc.addColumn('PivotHigh', pivotHigh, { inplace: true });
    dfCalc.addColumn('PivotLow', pivotLow, { inplace: true });

    // Order Blocks
    let bullBoxes = [];
    let bearBoxes = [];
    dfCalc['PivotHigh'].values.forEach((ph, i) => {
        if (ph) {
            bearBoxes.push({
                id: `Bear_${dfCalc.index[i]}`,
                type: 'bear',
                timestamp: dfCalc.index[i],
                top: new Decimal(high[i]),
                bottom: new Decimal(obSource === 'wicks' ? low[i] : close[i]),
                active: true,
                violated: false,
                violation_ts: null,
                extended_to_ts: dfCalc.index[dfCalc.index.length - 1]
            });
        }
    });
    dfCalc['PivotLow'].values.forEach((pl, i) => {
        if (pl) {
            bullBoxes.push({
                id: `Bull_${dfCalc.index[i]}`,
                type: 'bull',
                timestamp: dfCalc.index[i],
                top: new Decimal(obSource === 'wicks' ? high[i] : close[i]),
                bottom: new Decimal(low[i]),
                active: true,
                violated: false,
                violation_ts: null,
                extended_to_ts: dfCalc.index[dfCalc.index.length - 1]
            });
        }
    });

    // Filter Order Blocks
    const lastClose = new Decimal(close[close.length - 1]);
    bullBoxes = bullBoxes.filter(b => {
        if (b.active && lastClose.lt(b.bottom)) {
            b.active = false;
            b.violated = true;
            b.violation_ts = dfCalc.index[dfCalc.index.length - 1];
        }
        return b.active;
    }).slice(0, obMaxBoxes);
    bearBoxes = bearBoxes.filter(b => {
        if (b.active && lastClose.gt(b.top)) {
            b.active = false;
            b.violated = true;
            b.violation_ts = dfCalc.index[dfCalc.index.length - 1];
        }
        return b.active;
    }).slice(0, obMaxBoxes);

    // Signal Generation
    let signal = 'NONE';
    const position = await getOpenPosition(exchange, symbol, logger);
    const currentTrendUp = dfCalc['TrendUp'].iloc(-1);
    const trendJustChanged = dfCalc['TrendChanged'].iloc(-1);

    if (position) {
        if (position.side === 'long' && !currentTrendUp && trendJustChanged) signal = 'EXIT_LONG';
        else if (position.side === 'short' && currentTrendUp && trendJustChanged) signal = 'EXIT_SHORT';
    } else {
        if (currentTrendUp && trendJustChanged && bullBoxes.length) {
            for (const ob of bullBoxes) {
                if (lastClose.gte(ob.bottom) && lastClose.lte(ob.top.mul(entryProx))) {
                    signal = 'BUY';
                    break;
                }
            }
        } else if (!currentTrendUp && trendJustChanged && bearBoxes.length) {
            for (const ob of bearBoxes) {
                if (lastClose.lte(ob.top) && lastClose.gte(ob.bottom.div(entryProx))) {
                    signal = 'SELL';
                    break;
                }
            }
        }
    }

    return {
        dataframe: dfCalc,
        last_close: lastClose,
        current_trend_up: currentTrendUp,
        trend_just_changed: trendJustChanged,
        active_bull_boxes: bullBoxes,
        active_bear_boxes: bearBoxes,
        vol_norm_int: dfCalc['VolNormInt'].iloc(-1),
        atr: safeMarketDecimal(dfCalc['ATR'].iloc(-1), 'atr', false),
        upper_band: safeMarketDecimal(dfCalc['VT_UpperBand'].iloc(-1), 'upper_band', false),
        lower_band: safeMarketDecimal(dfCalc['VT_LowerBand'].iloc(-1), 'lower_band', false),
        signal
    };
}

/**
 * Analyzes and trades for a symbol.
 * @param {ccxt.Exchange} exchange
 * @param {string} symbol
 * @param {object} config
 * @param {winston.Logger} logger
 */
async function analyzeAndTradeSymbol(exchange, symbol, config, logger) {
    logger.info(magenta(`=== Analysis Cycle for ${symbol} ===`));
    const startTime = performance.now();

    try {
        // Market Info
        const marketInfo = await getMarketInfo(exchange, symbol, logger);
        if (!marketInfo) throw new Error(`Invalid market info for ${symbol}`);

        // Klines
        const timeframe = CCXT_INTERVAL_MAP[config.interval] || '5m';
        const dfRaw = await fetchKlinesCcxt(exchange, symbol, timeframe, config.fetch_limit || DEFAULT_FETCH_LIMIT, logger);
        if (!dfRaw) throw new Error(`Failed to fetch klines for ${symbol}`);

        // Strategy Signals
        const strategyResults = await calculateStrategySignals(dfRaw, config, logger, exchange, symbol);
        if (!strategyResults.last_close) throw new Error(`Invalid strategy results for ${symbol}`);

        const { last_close, current_trend_up, atr, signal } = strategyResults;
        logger.info(`Results: Close=${last_close.toFixed()}, TrendUp=${current_trend_up}, ATR=${atr?.toFixed() || 'N/A'}, Signal=${signal}`);

        // Position
        let positionInfo = await getOpenPosition(exchange, symbol, logger);

        if (positionInfo) {
            const posSide = positionInfo.side;
            const posSize = positionInfo.size_decimal;
            const entryPrice = positionInfo.entryPrice;
            let beActivated = positionInfo.be_activated;
            let tslActivated = positionInfo.tsl_activated;

            if (!entryPrice) throw new Error(`Invalid entry price for ${symbol}`);

            logger.info(cyan(`Managing ${posSide} position: Size=${posSize.toFixed()}, Entry=${entryPrice.toFixed()}`));

            // Exit Check
            const shouldExit = (posSide === 'long' && signal === 'EXIT_LONG') || (posSide === 'short' && signal === 'EXIT_SHORT');
            if (shouldExit) {
                logger.warn(bold(`>>> Exit Signal '${signal}' for ${posSide} position <<<`));
                const closeSize = posSize.abs();
                const orderResult = await placeTrade(exchange, symbol, signal, closeSize, marketInfo, logger, true);
                if (orderResult) logger.info(green(`Exit order placed: ID=${orderResult.id}`));
                else logger.error(red(`Failed to exit position for ${symbol}`));
                return;
            }

            // Protection Management
            const prot = config.protection;
            if ((prot.enable_break_even || prot.enable_trailing_stop) && atr && atr.gt(0)) {
                const currentPrice = last_close;
                const atrDec = atr;
                let slPrice = safeMarketDecimal(positionInfo.stopLossPrice, 'sl_price', false);
                let tpPrice = safeMarketDecimal(positionInfo.takeProfitPrice, 'tp_price', false) || new Decimal(0);

                // Break-Even
                if (prot.enable_break_even && !beActivated) {
                    const beTrigger = new Decimal(prot.break_even_trigger_atr_multiple);
                    const offsetTicks = new Decimal(prot.break_even_offset_ticks);
                    const priceTick = marketInfo.price_precision_step_decimal;
                    let bePrice = null;

                    if (posSide === 'long' && currentPrice.gte(entryPrice.add(atrDec.mul(beTrigger)))) {
                        bePrice = entryPrice.add(priceTick.mul(offsetTicks));
                    } else if (posSide === 'short' && currentPrice.lte(entryPrice.sub(atrDec.mul(beTrigger)))) {
                        bePrice = entryPrice.sub(priceTick.mul(offsetTicks));
                    }

                    if (bePrice && bePrice.gt(0)) {
                        const success = await _set_position_protection(exchange, symbol, marketInfo, positionInfo, logger, bePrice, tpPrice);
                        if (success) {
                            positionInfo.be_activated = true;
                            logger.info(green(`Break-even set: SL=${bePrice.toFixed()}`));
                        }
                    }
                }

                // Trailing Stop-Loss
                if (prot.enable_trailing_stop && !tslActivated && beActivated) {
                    const tslCallback = new Decimal(prot.trailing_stop_callback_rate);
                    const tslActivationPerc = new Decimal(prot.trailing_stop_activation_percentage);
                    let tslDistance = null;
                    let tslActivationPrice = null;

                    if (posSide === 'long' && currentPrice.gte(entryPrice.mul(1 + tslActivationPerc.toNumber()))) {
                        tslDistance = atrDec.mul(tslCallback);
                        tslActivationPrice = currentPrice;
                    } else if (posSide === 'short' && currentPrice.lte(entryPrice.mul(1 - tslActivationPerc.toNumber()))) {
                        tslDistance = atrDec.mul(tslCallback);
                        tslActivationPrice = currentPrice;
                    }

                    if (tslDistance && tslActivationPrice) {
                        const success = await _set_position_protection(exchange, symbol, marketInfo, positionInfo, logger, slPrice, tpPrice, tslDistance, tslActivationPrice);
                        if (success) {
                            positionInfo.tsl_activated = true;
                            logger.info(green(`Trailing stop set: Distance=${tslDistance.toFixed()}`));
                        }
                    }
                }
            }
        } else if (signal === 'BUY' || signal === 'SELL') {
            logger.info(cyan(`Evaluating ${signal} entry...`));
            if (!config.enable_trading) {
                logger.warn(`Trading disabled: Would enter ${signal}`);
                return;
            }
            if (!atr || atr.lte(0)) {
                logger.error(`Invalid ATR for ${symbol}`);
                return;
            }

            // SL/TP
            const prot = config.protection;
            const slAtrMult = new Decimal(prot.initial_stop_loss_atr_multiple);
            const tpAtrMult = new Decimal(prot.initial_take_profit_atr_multiple);
            let slPrice = null;
            let tpPrice = new Decimal(0);

            if (signal === 'BUY') {
                slPrice = last_close.sub(atr.mul(slAtrMult));
                if (tpAtrMult.gt(0)) tpPrice = last_close.add(atr.mul(tpAtrMult));
            } else {
                slPrice = last_close.add(atr.mul(slAtrMult));
                if (tpAtrMult.gt(0)) tpPrice = last_close.sub(atr.mul(tpAtrMult));
            }

            if (!slPrice || slPrice.lte(0)) {
                logger.error(`Invalid SL price: ${slPrice?.toFixed()}`);
                return;
            }
            logger.info(`Protections: SL=${slPrice.toFixed()}, TP=${tpPrice.isZero() ? 'Disabled' : tpPrice.toFixed()}`);

            // Size
            const balance = await fetchBalance(exchange, QUOTE_CURRENCY, logger);
            if (!balance || balance.lte(0)) {
                logger.error(`Invalid balance for ${symbol}`);
                return;
            }
            const positionSize = await calculatePositionSize(balance, config.risk_per_trade, slPrice, last_close, marketInfo, exchange, logger);
            if (!positionSize || positionSize.lte(0)) {
                logger.error(`Position sizing failed for ${symbol}`);
                return;
            }

            // Leverage
            if (marketInfo.contract && config.leverage > 0) {
                try {
                    await exchange.privatePostPositionSetLeverage({
                        symbol: marketInfo.id,
                        buyLeverage: config.leverage,
                        sellLeverage: config.leverage
                    });
                    logger.info(green(`Leverage set to ${config.leverage}x for ${symbol}`));
                } catch (e) {
                    logger.error(red(`Failed to set leverage: ${e.message}`));
                    return;
                }
            }

            // Trade
            logger.warn(bold(`>>> Initiating ${signal} entry: Size=${positionSize.toFixed()} <<<`));
            const orderResult = await placeTrade(exchange, symbol, signal, positionSize, marketInfo, logger, false);
            if (!orderResult) {
                logger.error(red(`Entry order failed for ${symbol}`));
                return;
            }

            // Confirm and Protect
            await delay(config.position_confirm_delay_seconds * 1000);
            const confirmedPosition = await getOpenPosition(exchange, symbol, logger);
            if (confirmedPosition) {
                const success = await _set_position_protection(exchange, symbol, marketInfo, confirmedPosition, logger, slPrice, tpPrice);
                if (success) logger.info(green(`Initial SL/TP set for ${symbol}`));
                else logger.error(red(`Failed to set SL/TP for ${symbol}`));
            } else {
                logger.error(red(`Order placed (ID: ${orderResult.id}) but position not confirmed`));
            }
        } else {
            logger.info(`No position or signal for ${symbol}`);
        }
    } catch (e) {
        logger.error(red(`Error in cycle for ${symbol}: ${e.message}`), { stack: e.stack });
    } finally {
        const duration = (performance.now() - startTime) / 1000;
        logger.info(magenta(`=== Cycle completed in ${duration.toFixed(2)}s ===`));
    }
}

/**
 * Main execution loop.
 */
async function main() {
    const mainLogger = setupLogger('main');
    mainLogger.info(magenta(bold(`--- Pyrmethus Volumatic Bot v${BOT_VERSION} Starting ---`)));

    process.on('SIGINT', () => signalHandler('SIGINT', mainLogger));
    process.on('SIGTERM', () => signalHandler('SIGTERM', mainLogger));

    CONFIG = await loadConfig(CONFIG_FILE);
    if (!API_KEY || !API_SECRET) {
        mainLogger.error(red('FATAL: Missing API credentials'));
        process.exit(1);
    }

    const exchange = await initializeExchange(mainLogger);
    if (!exchange) {
        mainLogger.error(red('Exchange initialization failed'));
        process.exit(1);
    }

    // Validate Pairs
    const validPairs = [];
    for (const symbol of CONFIG.trading_pairs) {
        const marketInfo = await getMarketInfo(exchange, symbol, mainLogger);
        if (marketInfo && marketInfo.info.active) {
            validPairs.push(symbol);
            mainLogger.info(green(`Valid pair: ${symbol}`));
        } else {
            mainLogger.error(red(`Invalid/inactive pair: ${symbol}`));
        }
    }
    if (!validPairs.length) {
        mainLogger.error(red('No valid trading pairs'));
        process.exit(1);
    }

    mainLogger.info(cyan(`Entering trading loop...`));
    while (!shutdownRequested) {
        const cycleStart = performance.now();
        mainLogger.info(yellow(`--- New Trading Cycle ---`));

        try {
            CONFIG = await loadConfig(CONFIG_FILE);
        } catch (e) {
            mainLogger.warn(`Failed to reload config: ${e.message}`);
        }

        for (const symbol of validPairs) {
            if (shutdownRequested) break;
            const symbolLogger = setupLogger(symbol);
            await analyzeAndTradeSymbol(exchange, symbol, CONFIG, symbolLogger);
        }

        if (shutdownRequested) break;

        const cycleDuration = (performance.now() - cycleStart) / 1000;
        const waitTime = Math.max(0, CONFIG.loop_delay_seconds - cycleDuration);
        mainLogger.info(`Cycle duration: ${cycleDuration.toFixed(2)}s. Waiting ${waitTime.toFixed(2)}s...`);

        const sleepStart = performance.now();
        while (performance.now() - sleepStart < waitTime * 1000) {
            if (shutdownRequested) break;
            await delay(100);
        }
    }

    mainLogger.info(magenta(bold(`--- Shutting Down ---`)));
    process.exit(0);
}

/**
 * Handles shutdown signals.
 * @param {string} sig
 * @param {winston.Logger} logger
 */
function signalHandler(sig, logger) {
    if (!shutdownRequested) {
        logger.info(yellow(bold(`Shutdown signal (${sig}) received. Exiting...`)));
        shutdownRequested = true;
    } else {
        logger.warn(red(`Forced exit on second ${sig}`));
        process.exit(1);
    }
}

// Start
main().catch(error => {
    initLogger.error(red(`CRITICAL ERROR: ${error.message}`), { stack: error.stack });
    process.exit(1);
});
