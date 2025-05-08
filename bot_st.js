#!/usr/bin/env node
// -*- coding: utf-8 -*-

/**
 * Enhanced Trading Bot using CCXT for Bybit Futures/Swaps (Linear Contracts) - Node.js Version.
 *
 * Strategy: Dual SuperTrend confirmation, Volume Spike filter, Order Book Pressure filter.
 * Features:
 * - Risk Management (Risk per Trade, Equity Calculation)
 * - ATR-based Stop Loss (SL), Take Profit (TP), and Trailing Stop Loss (TSL)
 * - Configuration via .env file
 * - Termux SMS Notifications (Optional)
 * - API Call Retries with Exponential Backoff (via helper)
 * - Persistent State Management (for active SL order ID) via atomic file writes
 * - Dry Run Mode for simulation (Default: Enabled)
 * - Colorized Logging with different levels (using nanocolors) and file output
 * - Graceful Shutdown (attempts to close positions)
 * - Specific error handling for common CCXT exceptions
 * - Data Caching for efficiency
 * - Bybit V5 API parameter considerations (Unified/Contract Accounts)
 * - Enhanced validation and error handling throughout
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
    SYMBOL: "BTC/USDT:USDT",        // Default trading pair (Bybit linear swap format) - Changed default to BTC for wider availability
    TIMEFRAME: "1m",                // Default candlestick timeframe
    LEVERAGE: 10.0,                 // Default leverage (Reduced default)
    RISK_PER_TRADE: 0.01,           // Default risk percentage (1%)
    SL_ATR_MULT: 1.5,               // Default Stop Loss ATR multiplier
    TP_ATR_MULT: 2.0,               // Default Take Profit ATR multiplier
    TRAILING_STOP_MULT: 1.5,        // Default Trailing Stop ATR multiplier
    SHORT_ST_PERIOD: 7,             // Default short SuperTrend period
    LONG_ST_PERIOD: 14,            // Default long SuperTrend period (also used for main ATR)
    ST_MULTIPLIER: 2.0,             // Default SuperTrend ATR multiplier
    VOLUME_SPIKE_THRESHOLD: 1.5,    // Default volume ratio threshold
    OB_PRESSURE_THRESHOLD: 0.6,     // Default order book buy pressure threshold (0.0 to 1.0)
    LOGGING_LEVEL: "INFO",          // Default logging level (DEBUG, INFO, WARN, ERROR)
    MAX_RETRIES: 3,                 // Default max retries for API calls
    RETRY_DELAY: 5,                 // Default delay (seconds) between retries
    CURRENCY: "USDT",               // Default quote currency for balance/risk
    EXCHANGE_TYPE: "swap",          // Default market type (swap/future/spot)
    ORDER_TRIGGER_PRICE_TYPE: "LastPrice", // Default trigger price for SL/TP (check Bybit docs: LastPrice, MarkPrice, IndexPrice)
    TIME_IN_FORCE: "GoodTillCancel",// Default Time-in-Force for orders
    VOLUME_SHORT_PERIOD: 5,         // Default short MA period for volume
    VOLUME_LONG_PERIOD: 20,        // Default long MA period for volume
    ORDER_BOOK_DEPTH: 10,           // Default depth for order book fetching
    CACHE_TTL: 30,                  // Default cache Time-To-Live (seconds) for API data
    STATE_FILE: "trading_bot_state.json", // Default filename for persistent state
    LOG_FILE_ENABLED: "true",       // Default: enable file logging (as string for consistent parsing)
    LOG_DIR: "logs",                // Default directory for log files
    DRY_RUN: "true",                // Default: dry run ENABLED (safer default)
    SMS_ENABLED: "false",           // Default: SMS disabled
});

// List of valid timeframes supported by CCXT/Bybit (adjust if needed)
const VALID_TIMEFRAMES = Object.freeze(["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"]); // Added weekly/monthly

// --- Global Variables ---
let botInstance = null; // Holds the main TradingBot instance for access by shutdown handlers

// --- Logging Setup ---
// Simple logger using nanocolors for console and basic async file appending.
const logLevels = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
let currentLogLevel = logLevels.INFO; // Default level, updated by Config
let logFilePath = null; // Path to the current log file, null if disabled
let logFileHandle = null; // File handle for efficient appending

// Initializes logging based on configuration (sets level and file path)
async function setupLogging(config) {
    const levelName = config.logging_level.toUpperCase();
    currentLogLevel = logLevels[levelName] ?? logLevels.INFO; // Default to INFO if invalid
    if (logLevels[levelName] === undefined) {
        console.warn(c.yellow(`[WARN] Invalid LOGGING_LEVEL '${config.logging_level}'. Defaulting to INFO.`));
    }

    if (config.log_file_enabled) {
        try {
            // Ensure log directory exists
            await fs.mkdir(config.log_dir, { recursive: true });
            // Create a unique log file name with timestamp
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            logFilePath = path.join(config.log_dir, `trading_bot_${timestamp}.log`);
            // Open file handle for writing
            logFileHandle = await fs.open(logFilePath, 'a'); // Open in append mode
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
            // Format message for file (without colors)
            const fileMessage = `${timestamp} [${level}] ${args.map(arg =>
                typeof arg === 'object' ? inspect(arg, { depth: 4, colors: false }) : String(arg) // Deeper inspection for file logs, no colors
            ).join(' ')}\n`;

            // Asynchronously write to the file using the handle
            logFileHandle.write(fileMessage).catch(err => {
                console.error(c.red(`[ERROR] Failed to write to log file '${logFilePath}': ${err.message}`));
                // Optional: Disable file logging after repeated errors?
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
    // Close the log file handle during shutdown
    async closeLogFile() {
        if (logFileHandle) {
            console.log(c.yellow("Closing log file handle..."));
            try {
                await logFileHandle.close();
                logFileHandle = null;
                console.log(c.green("Log file handle closed."));
            } catch (err) {
                console.error(c.red(`Error closing log file handle: ${err.message}`));
            }
        }
    }
};

// --- Utility Functions ---

// Simple asynchronous sleep function
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Wraps an async function call with retry logic for specific exceptions.
 * @param {Function} func - The async function to execute.
 * @param {number} maxRetries - Maximum number of retry attempts (0 means one attempt, no retries).
 * @param {number} delaySeconds - Delay between retries in seconds.
 * @param {Array<Error>} [allowedExceptions] - Array of CCXT/Error classes that trigger a retry. Defaults to common network/availability errors.
 * @param {string} [funcName] - Name of the function being called (for logging). Defaults to function's name or 'anonymous function'.
 * @returns {Promise<any>} - The result of the function if successful.
 * @throws {Error} - The last exception if all retries fail or a non-retryable error occurs.
 */
async function retryOnException(
    func,
    maxRetries,
    delaySeconds,
    allowedExceptions = [ // Default retryable CCXT network/availability errors
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
        // Consider adding ccxt.OperationFailed based on specific needs/contexts, e.g., Bybit 'busy' errors
        // ccxt.ExchangeError, // Be cautious retrying generic ExchangeError
    ],
    funcName = func.name || 'anonymous function'
) {
    let attempts = 0;
    while (attempts <= maxRetries) {
        attempts++;
        try {
            return await func(); // Attempt the function call
        } catch (e) {
            const isRetryable = allowedExceptions.some(excType => e instanceof excType);

            if (isRetryable && attempts <= maxRetries) {
                // Log retry attempt
                const delay = delaySeconds * 1000; // Convert to ms for sleep
                logger.warn(`[Retry] Attempt ${attempts}/${maxRetries + 1} for ${c.yellow(funcName)} failed: ${c.red(e.constructor.name)} - ${e.message}. Retrying in ${delaySeconds}s...`);
                await sleep(delay); // Wait before next retry
            } else {
                // If it's not a retryable error OR max retries are exhausted
                if (!isRetryable) {
                    logger.error(`[Error] Non-retryable error in ${c.yellow(funcName)}: ${c.red(e.constructor.name)} - ${e.message}`);
                    // Log stack trace for non-retryable errors to aid debugging
                    if (e instanceof Error) logger.debug("Non-retryable error stack trace:", e.stack);
                } else {
                    logger.error(`[Error] ${c.yellow(funcName)} failed after ${attempts} attempts due to ${c.red(e.constructor.name)}: ${e.message}`);
                    // Log stack trace on final failure
                    if (e instanceof Error) logger.debug("Final attempt error stack trace:", e.stack);
                }
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
// Loads, validates, and provides access to bot configuration settings.
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
        this.volume_short_period = parseInt(process.env.VOLUME_SHORT_PERIOD || DEFAULTS.VOLUME_SHORT_PERIOD, 10);
        this.volume_long_period = parseInt(process.env.VOLUME_LONG_PERIOD || DEFAULTS.VOLUME_LONG_PERIOD, 10);
        this.volume_spike_threshold = parseFloat(process.env.VOLUME_SPIKE_THRESHOLD || DEFAULTS.VOLUME_SPIKE_THRESHOLD);
        this.order_book_depth = parseInt(process.env.ORDER_BOOK_DEPTH || DEFAULTS.ORDER_BOOK_DEPTH, 10);
        this.ob_pressure_threshold = parseFloat(process.env.OB_PRESSURE_THRESHOLD || DEFAULTS.OB_PRESSURE_THRESHOLD);

        // --- Load Bot Behavior Parameters ---
        // Safer default: Assume dry run unless explicitly set to 'false'
        this.dry_run = (process.env.DRY_RUN || DEFAULTS.DRY_RUN).toLowerCase() !== 'false';
        this.logging_level = (process.env.LOGGING_LEVEL || DEFAULTS.LOGGING_LEVEL).toUpperCase();
        this.log_file_enabled = (process.env.LOG_FILE_ENABLED || DEFAULTS.LOG_FILE_ENABLED).toLowerCase() === 'true';
        this.log_dir = process.env.LOG_DIR || DEFAULTS.LOG_DIR;
        this.max_retries = parseInt(process.env.MAX_RETRIES || DEFAULTS.MAX_RETRIES, 10);
        this.retry_delay = parseInt(process.env.RETRY_DELAY || DEFAULTS.RETRY_DELAY, 10);
        this.cache_ttl = parseInt(process.env.CACHE_TTL || DEFAULTS.CACHE_TTL, 10);
        this.state_file = process.env.STATE_FILE || DEFAULTS.STATE_FILE;

        // --- Load Exchange/Order Parameters ---
        this.currency = process.env.CURRENCY || DEFAULTS.CURRENCY;
        this.exchange_type = process.env.EXCHANGE_TYPE || DEFAULTS.EXCHANGE_TYPE;
        this.order_trigger_price_type = process.env.ORDER_TRIGGER_PRICE_TYPE || DEFAULTS.ORDER_TRIGGER_PRICE_TYPE;
        this.time_in_force = process.env.TIME_IN_FORCE || DEFAULTS.TIME_IN_FORCE;

        // --- Load Notification Parameters ---
        this.sms_enabled = (process.env.SMS_ENABLED || DEFAULTS.SMS_ENABLED).toLowerCase() === 'true';
        this.sms_recipient_number = process.env.SMS_RECIPIENT_NUMBER || null;

        // --- Internal State ---
        this.termux_sms_available = false; // Determined during validation

        // --- Perform Validation and Checks ---
        this._validate(); // Validate loaded parameters
        this._checkTermuxSms(); // Check Termux SMS capability after validation

        // --- Finalize Logging Setup ---
        // Call setupLogging *after* validation ensures LOGGING_LEVEL is valid
        // setupLogging is async, but constructor cannot be. Call it synchronously here.
        // Workaround: Call async setupLogging and let it run, logging might be delayed slightly.
        // Proper setup (file handle opening) will complete before the main loop starts.
        setupLogging(this).catch(err => console.error("Error during async logging setup:", err));

        // Log the final configuration summary (masking the secret)
        const configSummary = { ...this };
        configSummary.bybit_api_secret = configSummary.bybit_api_secret ? '******' : null;
        // Use logger here, assuming setupLogging has set the level correctly by now,
        // even if file handle isn't open yet. Console logging will work.
        logger.info(c.green(`Configuration loaded successfully. Dry Run: ${c.bold(this.dry_run)}`));
        logger.debug("Full Config (Secret Masked):", configSummary);
    }

    // Internal method to validate configuration parameters
    _validate() {
        logger.debug("Validating configuration parameters...");
        const errors = [];

        // Check API keys only if not in dry run mode
        if (!this.dry_run && (!this.bybit_api_key || !this.bybit_api_secret)) {
            errors.push("BYBIT_API_KEY and BYBIT_API_SECRET environment variables are required when DRY_RUN is false (or not set to 'true').");
        }

        // Validate numerical parameters
        if (isNaN(this.leverage) || this.leverage <= 0) errors.push("LEVERAGE must be a positive number.");
        if (isNaN(this.risk_per_trade) || !(this.risk_per_trade > 0 && this.risk_per_trade <= 1.0)) { // Allow up to 100% risk but warn heavily
            errors.push("RISK_PER_TRADE must be a positive number (e.g., 0.01 for 1%).");
        }
        // Add warnings for high risk settings
        if (this.risk_per_trade > 0.05) {
            logger.warn(c.yellow(`High Risk Setting: RISK_PER_TRADE (${(this.risk_per_trade * 100).toFixed(1)}%) is > 5%. Ensure this is intended.`));
        }
         if (this.risk_per_trade > 0.1) {
            logger.warn(c.red(`EXTREME Risk Setting: RISK_PER_TRADE (${(this.risk_per_trade * 100).toFixed(1)}%) is > 10%. High chance of liquidation!`));
        }
        if (!VALID_TIMEFRAMES.includes(this.timeframe)) {
            errors.push(`Invalid TIMEFRAME: '${this.timeframe}'. Must be one of: ${VALID_TIMEFRAMES.join(', ')}`);
        }

        // Helper functions for common numerical validations
        const checkPositiveNumber = (key) => { if (isNaN(this[key]) || this[key] <= 0) errors.push(`${key.toUpperCase()} must be a positive number. Got: ${this[key]}`); };
        const checkPositiveInteger = (key) => { if (!Number.isInteger(this[key]) || this[key] <= 0) errors.push(`${key.toUpperCase()} must be a positive integer. Got: ${this[key]}`); };
        const checkNonNegativeInteger = (key) => { if (!Number.isInteger(this[key]) || this[key] < 0) errors.push(`${key.toUpperCase()} must be a non-negative integer. Got: ${this[key]}`); };

        checkPositiveNumber('sl_atr_mult');
        checkPositiveNumber('tp_atr_mult');
        checkPositiveNumber('trailing_stop_mult');
        checkPositiveInteger('short_st_period');
        checkPositiveInteger('long_st_period');
        checkPositiveNumber('st_multiplier');
        checkPositiveNumber('volume_spike_threshold');
        checkPositiveInteger('volume_short_period');
        checkPositiveInteger('volume_long_period');
        checkPositiveInteger('order_book_depth');
        checkNonNegativeInteger('max_retries'); // Can be 0
        checkPositiveInteger('retry_delay');
        checkPositiveInteger('cache_ttl');

        if (this.volume_short_period >= this.volume_long_period) errors.push("VOLUME_SHORT_PERIOD must be less than VOLUME_LONG_PERIOD.");
        if (isNaN(this.ob_pressure_threshold) || !(this.ob_pressure_threshold >= 0 && this.ob_pressure_threshold <= 1)) errors.push("OB_PRESSURE_THRESHOLD must be a number between 0 and 1 (inclusive).");

        // Validate SMS settings
        if (this.sms_enabled && (!this.sms_recipient_number || !this.sms_recipient_number.trim())) {
            logger.warn(c.yellow("SMS_ENABLED is true, but SMS_RECIPIENT_NUMBER is not set or empty. Disabling SMS notifications."));
            this.sms_enabled = false; // Automatically disable if number is missing
        }

        // Validate exchange and order parameters
        if (!['swap', 'future', 'spot'].includes(this.exchange_type)) {
            errors.push(`Invalid EXCHANGE_TYPE: '${this.exchange_type}'. Must be 'swap', 'future', or 'spot'. Ensure it matches the SYMBOL type.`);
        }
        // Common trigger price types (case-insensitive check)
        const valid_trigger_types = ['lastprice', 'markprice', 'indexprice'];
        if (!valid_trigger_types.includes(this.order_trigger_price_type.toLowerCase())) {
            errors.push(`Invalid ORDER_TRIGGER_PRICE_TYPE: '${this.order_trigger_price_type}'. Common options: LastPrice, MarkPrice, IndexPrice. Check Bybit documentation.`);
        }
        // Common Time-in-Force types (case-insensitive check, remove spaces)
        const valid_tif = ['goodtillcancel', 'immediateorcancel', 'fillorkill', 'postonly'];
        const tifLower = this.time_in_force.toLowerCase().replace(/\s/g, "");
        if (!valid_tif.includes(tifLower)) {
            errors.push(`Invalid TIME_IN_FORCE: '${this.time_in_force}'. Common options: GoodTillCancel (GTC), ImmediateOrCancel (IOC), FillOrKill (FOK), PostOnly.`);
        }

        // Validate logging level (already handled by setupLogging default, but check here too)
        if (!logLevels.hasOwnProperty(this.logging_level)) {
             // Warning already logged by setupLogging
             this.logging_level = "INFO"; // Ensure default is set if invalid
        }

        // If any errors were found, log them and throw a critical error
        if (errors.length > 0) {
            const errorMessage = "Configuration validation failed:\n" + errors.map(e => `- ${e}`).join('\n');
            logger.error(c.red(errorMessage));
            throw new Error(errorMessage); // Halt initialization
        }
        logger.debug(c.green("Configuration validation successful."));
    }

    // Checks if Termux SMS sending is likely possible
    _checkTermuxSms() {
        if (!this.sms_enabled) {
            this.termux_sms_available = false;
            logger.debug("SMS is disabled in config.");
            return;
        }

        // Basic check for Termux environment (presence of specific env var or directory)
        const isLikelyTermux = process.env.TERMUX_VERSION || (fs.existsSync && fs.existsSync('/data/data/com.termux'));
        if (!isLikelyTermux) {
             logger.info("Not running in a recognizable Termux environment. Disabling SMS feature.");
             this.sms_enabled = false;
             this.termux_sms_available = false;
             return;
        }

        // Check if the 'termux-sms-send' command exists and is executable using 'which'
        try {
             // execSync throws an error if the command is not found or not executable
             execSync('which termux-sms-send', { stdio: 'ignore' }); // Don't print 'which' output
             this.termux_sms_available = true;
             logger.info(c.green("Termux SMS command 'termux-sms-send' found. SMS enabled."));
        } catch (error) {
             logger.warn(c.yellow("Termux environment detected, but 'termux-sms-send' command not found or not executable in PATH. Disabling SMS. Ensure Termux:API app is installed and 'pkg install termux-api' was run."));
             this.sms_enabled = false;
             this.termux_sms_available = false;
        }
    }
}


// --- Notification Service ---
// Handles sending notifications, currently only Termux SMS.
class NotificationService {
    /**
     * Sends an SMS message using the Termux API command.
     * @param {string} message - The message content.
     * @param {Config} config - The bot's configuration object.
     */
    sendSms(message, config) {
        // Check if SMS should be sent based on config and availability
        if (!config.sms_enabled || !config.termux_sms_available || !config.sms_recipient_number) {
            logger.debug(c.gray(`SMS sending skipped (check enabled/available/recipient): ${message.substring(0, 80)}...`));
            return;
        }

        try {
            // Basic sanitization to prevent simple shell injection issues.
            let sanitizedMessage = message
                .replace(/"/g, "'")    // Replace double quotes with single quotes
                .replace(/`/g, "'")    // Replace backticks with single quotes
                .replace(/\$/g, "")   // Remove dollar signs (prevents variable expansion)
                .replace(/\\/g, "")   // Remove backslashes (prevents escape sequences)
                .replace(/;/g, ".")    // Replace semicolons with periods (prevents command chaining)
                .replace(/&/g, "and") // Replace ampersands (prevents background processes/chaining)
                .replace(/\|/g, "-");  // Replace pipes (prevents piping)
                // Consider adding more characters like (, ), {, }, <, > if needed

            // Enforce SMS length limit (standard GSM SMS is 160 chars)
            const maxSmsLength = 160;
            if (sanitizedMessage.length > maxSmsLength) {
                logger.warn(c.yellow(`Truncating long SMS message (${sanitizedMessage.length} chars) to ${maxSmsLength}.`));
                sanitizedMessage = sanitizedMessage.substring(0, maxSmsLength - 3) + "...";
            }

            // Escape single quotes within the message itself for the shell command
            const shellEscapedMessage = sanitizedMessage.replace(/'/g, "'\\''");

            // Construct the command string safely, quoting arguments
            const command = `termux-sms-send -n '${config.sms_recipient_number}' '${shellEscapedMessage}'`;

            logger.debug(`Executing SMS command: termux-sms-send -n '${config.sms_recipient_number}' '...'`); // Avoid logging full message at debug

            // Execute the command asynchronously with a timeout
            exec(command, { timeout: 30000 }, (error, stdout, stderr) => { // 30 second timeout
                if (error) {
                    // Log detailed error information if the command fails
                    logger.error(c.red(`SMS command failed: ${error.message}. Code: ${error.code}, Signal: ${error.signal}`));
                    if (stderr) logger.error(c.red(`SMS stderr: ${stderr.trim()}`));
                    if (stdout) logger.error(c.red(`SMS stdout (error context): ${stdout.trim()}`));
                    // Consider adding logic to disable SMS temporarily after repeated failures?
                    return;
                }
                // Log success
                logger.info(c.green(`SMS sent successfully to ${config.sms_recipient_number}: "${message.substring(0, 80)}..."`));
                // Log stderr/stdout even on success if they contain output (sometimes informational)
                if (stderr && stderr.trim()) logger.debug(`termux-sms-send stderr: ${stderr.trim()}`);
                if (stdout && stdout.trim()) logger.debug(`termux-sms-send stdout: ${stdout.trim()}`);
            });
        } catch (e) {
            // Catch synchronous errors during command preparation (less likely here)
            logger.error(c.red(`SMS sending failed with unexpected synchronous error: ${e.message}`), e.stack);
        }
    }
}

// --- Exchange Manager ---
// Handles CCXT initialization, market loading, data fetching, and caching.
class ExchangeManager {
    constructor(config) {
        this.config = config;
        this.exchange = null; // CCXT exchange instance, initialized asynchronously
        // Simple in-memory cache for API responses to reduce redundant calls
        this._caches = {
            ohlcv: { key: null, data: null, timestamp: 0, ttl: this.config.cache_ttl },
            order_book: { key: null, data: null, timestamp: 0, ttl: Math.max(1, Math.min(this.config.cache_ttl, 10)) }, // Shorter TTL for OB
            ticker: { key: null, data: null, timestamp: 0, ttl: Math.max(1, Math.min(this.config.cache_ttl, 5)) },     // Short TTL for Ticker
            balance: { key: null, data: null, timestamp: 0, ttl: this.config.cache_ttl },
            position: { key: null, data: null, timestamp: 0, ttl: Math.max(5, Math.min(this.config.cache_ttl, 15)) }, // Moderate TTL for Position
        };
    }

    // Asynchronous initialization method
    async initialize() {
        this.exchange = this._setupExchange(); // Create CCXT instance
        await this._loadMarketsAndValidate(); // Load markets and validate the configured symbol
        // Set leverage only if not in dry run and for futures/swaps
        if (!this.config.dry_run && ['swap', 'future'].includes(this.config.exchange_type)) {
            await this._setLeverage();
        }
        logger.info(c.green(`Exchange Manager initialized for ${c.bold(this.exchange.id)}, Symbol: ${c.bold(this.config.symbol)}, Type: ${c.bold(this.config.exchange_type)}`));
    }

    // Retrieves data from a specific cache if valid and not expired
    _getCache(cacheName, key) {
        const cache = this._caches[cacheName];
        if (!cache) return null; // Unknown cache name

        // Check if key matches and data exists
        if (cache.key === key && cache.data !== null) {
            const now = Date.now() / 1000;
            const age = now - cache.timestamp;
            if (age < cache.ttl) {
                // Cache hit and valid
                logger.debug(c.gray(`CACHE HIT: Using cached data for '${cacheName}' (Key: ${key}, Age: ${age.toFixed(1)}s < TTL: ${cache.ttl}s)`));
                // Return direct reference for performance. Caller should not modify.
                return cache.data;
            } else {
                // Cache hit but expired
                logger.debug(c.gray(`CACHE EXPIRED: Data for '${cacheName}' (Key: ${key}, Age: ${age.toFixed(1)}s >= TTL: ${cache.ttl}s)`));
            }
        }
        return null; // Cache miss or invalid
    }

    // Stores data in a specific cache
    _setCache(cacheName, key, data) {
        if (this._caches[cacheName]) {
            this._caches[cacheName].key = key;
            this._caches[cacheName].data = data;
            this._caches[cacheName].timestamp = Date.now() / 1000;
            logger.debug(c.gray(`CACHE SET: Updated cache for '${cacheName}' (Key: ${key})`));
        } else {
            logger.warn(c.yellow(`Attempted to set unknown cache: ${cacheName}`));
        }
    }

    // Creates and configures the CCXT exchange instance
    _setupExchange() {
        logger.info(c.blue(`Initializing Bybit ${this.config.exchange_type} exchange connection...`));
        let apiKey, apiSecret;

        if (this.config.dry_run) {
            // Use dummy keys for dry run - CCXT might still need valid-looking strings
            logger.info(c.magenta("Dry Run mode enabled. Using dummy API keys."));
            apiKey = "DRY_RUN_API_KEY";
            apiSecret = "DRY_RUN_API_SECRET";
        } else {
            // Use real keys from config for live trading
            if (!this.config.bybit_api_key || !this.config.bybit_api_secret) {
                 // This should have been caught by Config validation, but double-check
                 throw new Error("CRITICAL: API Key and Secret are required for live trading (DRY_RUN=false).");
            }
            apiKey = this.config.bybit_api_key;
            apiSecret = this.config.bybit_api_secret;
        }

        try {
            // CCXT exchange options
            const exchangeOptions = {
                apiKey: apiKey,
                secret: apiSecret,
                enableRateLimit: true, // Enable built-in rate limiting
                options: {
                    defaultType: this.config.exchange_type, // Specify market type (swap, future, spot)
                    adjustForTimeDifference: true, // Automatically sync time with server
                    recvWindow: 15000, // Increase request timeout window (default is often 5000ms)
                    // Bybit V5 often requires category, setting it here might help, but depends on account type
                    // It's often safer to set 'category' in params per API call (see fetch/order methods).
                    // 'brokerId': 'YOUR_BROKER_ID', // If using a broker ID
                }
            };

            // Check if 'bybit' class exists in the installed CCXT version
            if (!ccxt.hasOwnProperty('bybit')) {
                throw new Error("CCXT 'bybit' exchange class not found. Ensure CCXT is installed correctly (`npm install ccxt`).");
            }

            // Instantiate the Bybit exchange object
            const exchange = new ccxt.bybit(exchangeOptions);

            // Enable testnet/sandbox if needed (consult CCXT/Bybit documentation for current method)
            // if (this.config.use_sandbox) { // Assuming a config flag 'use_sandbox'
            //    exchange.setSandboxMode(true);
            //    // or sometimes: exchange.urls['api'] = exchange.urls['test'];
            //    logger.info(c.yellow("Using Bybit Testnet/Sandbox Mode."));
            // }

            logger.info(`CCXT ${c.bold(exchange.id)} instance created (Version: ${c.dim(exchange.version || 'N/A')}).`);
            return exchange;

        } catch (e) {
            // Catch potential errors during CCXT instantiation
            logger.error(c.red(`FATAL: Unexpected error during exchange setup: ${e.message}`), e.stack);
            // Wrap in a more generic error for the initialization phase
            throw new Error(`Exchange setup failed: ${e.message}`);
        }
    }

    // Loads market data from the exchange and validates the configured symbol
    async _loadMarketsAndValidate() {
        if (!this.exchange) throw new Error("Exchange not initialized before loading markets.");

        try {
            logger.info(`Loading exchange markets for ${this.exchange.id} (this may take a moment)...`);
            // Fetch and cache market data from the exchange
            // Use retry logic for loading markets as it's a critical network operation
            const loadMarketsFunc = async () => await this.exchange.loadMarkets();
            await retryOnException(
                loadMarketsFunc,
                this.config.max_retries,
                this.config.retry_delay,
                [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection], // Retry on connection/availability issues
                'loadMarkets'
            );
            logger.info(c.green(`Successfully loaded ${Object.keys(this.exchange.markets).length} markets from ${this.exchange.id}.`));

            // --- Symbol Validation ---
            const symbol = this.config.symbol;
            const exchangeType = this.config.exchange_type;
            logger.info(`Validating symbol '${c.bold(symbol)}' and type '${c.bold(exchangeType)}'...`);

            let market;
            try {
                 // Retrieve market data for the specific symbol
                 market = this.exchange.market(symbol);
                 // CCXT's market() might return undefined/null or throw BadSymbol
                 if (!market) throw new ccxt.BadSymbol(`Market data not found for symbol ${symbol}`);
            } catch (e) {
                 if (e instanceof ccxt.BadSymbol) {
                     // Provide helpful error message if symbol is not found
                     const availableSymbolsSample = Object.keys(this.exchange.markets)
                         .filter(s => this.exchange.markets[s]?.type === exchangeType) // Filter by configured type
                         .slice(0, 15); // Show a sample
                     logger.error(c.red(`Symbol '${symbol}' not found or invalid on ${this.exchange.id} for market type '${exchangeType}'.`), e);
                     logger.error(`Available symbols sample for type '${exchangeType}': ${availableSymbolsSample.join(', ')}...`);
                     throw new Error(`Symbol '${symbol}' not found or invalid for the configured exchange type.`);
                 } else {
                      // Handle other potential errors during market retrieval
                      logger.error(c.red(`Unexpected error retrieving market data for ${symbol}: ${e.message}`), e.stack);
                      throw new Error(`Failed to get market data for validation: ${e.message}`);
                 }
            }

            // --- Type Validation ---
            // Check if the market's type matches the configured exchange type
            const marketType = market.type; // e.g., 'swap', 'future', 'spot'
            if (marketType !== exchangeType) {
                throw new Error(`Symbol ${symbol}'s market type ('${marketType}') does not match configured EXCHANGE_TYPE ('${exchangeType}'). Check your SYMBOL and EXCHANGE_TYPE settings.`);
            }

            // --- Contract Type Validation (for Swaps/Futures) ---
            // Ensure the bot only works with LINEAR contracts (settled in quote currency, e.g., USDT)
            if (exchangeType === 'swap' || exchangeType === 'future') {
                const isLinear = market.linear === true;
                const isInverse = market.inverse === true;
                const settleCurrency = market.settle; // e.g., 'USDT' for linear, 'BTC' for inverse

                logger.debug(`Market details for ${symbol}: Linear=${isLinear}, Inverse=${isInverse}, Settle=${settleCurrency}`);

                // Explicitly reject inverse contracts
                if (isInverse) {
                    throw new Error(`Symbol ${symbol} is an INVERSE contract (settles in ${settleCurrency}). This bot currently supports LINEAR contracts (settled in ${this.config.currency}) only.`);
                }
                // Check linear flag and settle currency consistency
                // Allow if explicitly linear OR if settle currency matches config currency (handles cases where linear flag might be missing but settle is correct)
                if (!isLinear && (!settleCurrency || settleCurrency.toUpperCase() !== this.config.currency.toUpperCase())) {
                    throw new Error(`Symbol ${symbol} is not marked as linear and settles in ${settleCurrency || 'unknown'} (expected ${this.config.currency}). Bot requires LINEAR contracts.`);
                }
                // If flags are missing/ambiguous, use settle currency as a strong indicator
                if (!isLinear && !isInverse && settleCurrency && settleCurrency.toUpperCase() !== this.config.currency.toUpperCase()) {
                    logger.warn(c.yellow(`Market data for ${symbol} lacks clear linear/inverse flags, but settles in ${settleCurrency}. Assuming NOT LINEAR based on config currency ${this.config.currency}.`));
                    throw new Error(`Symbol ${symbol} settlement currency mismatch. Requires LINEAR contract settling in ${this.config.currency}.`);
                }
                // If flags are missing but settle currency *does* match, log a warning but proceed cautiously
                if (!isLinear && !isInverse && settleCurrency && settleCurrency.toUpperCase() === this.config.currency.toUpperCase()) {
                    logger.warn(c.yellow(`Market data for ${symbol} lacks clear linear/inverse flags. Assuming LINEAR based on matching settle currency '${settleCurrency}'. Verify manually.`));
                }
            }

            // --- Standardize Symbol ---
            // Use the symbol format returned by CCXT's market data for consistency
            const standardizedSymbol = market.symbol;
            if (standardizedSymbol !== symbol) {
                logger.info(`Standardizing symbol format from '${symbol}' to '${standardizedSymbol}' based on exchange market data.`);
                this.config.symbol = standardizedSymbol; // Update config in place
            }

            logger.info(c.green(`Symbol '${c.bold(this.config.symbol)}' validated successfully (Type: ${market.type}, Linear: ${market.linear ?? 'N/A'}, Settle: ${market.settle ?? 'N/A'}).`));

        } catch (e) {
            // Catch specific CCXT errors during the market loading/validation process
            if (e instanceof ccxt.AuthenticationError) {
                 logger.error(c.red(`Exchange Authentication Failed during market load: ${e.message}. Check API keys and permissions.`), e);
                 throw new Error(`Authentication Failed: ${e.message}`); // Propagate as critical init failure
            } else if (e instanceof ccxt.NetworkError || e instanceof ccxt.ExchangeNotAvailable || e instanceof ccxt.RequestTimeout || e instanceof ccxt.DDoSProtection) {
                 logger.error(c.red(`Failed to connect to exchange or load markets after retries: ${e.message}`), e);
                 throw new Error(`Exchange connection/market load failed: ${e.message}`); // Propagate
            } else if (e instanceof Error && (e.message.includes('Symbol') || e.message.includes('market type') || e.message.includes('LINEAR')) && (e.message.includes('validated') || e.message.includes('found') || e.message.includes('match') || e.message.includes('mismatch'))) {
                 // Catch validation errors thrown within this method
                 logger.error(c.red(`Symbol validation failed: ${e.message}`));
                 throw e; // Re-throw specific validation errors
            } else {
                 // Catch any other unexpected errors
                 logger.error(c.red(`Unexpected error during market loading or symbol validation: ${e.message}`), e.stack);
                 throw new Error(`Market loading/validation failed unexpectedly: ${e.message}`); // Propagate
            }
        }
    }

    // Sets the leverage for the configured symbol
    async _setLeverage() {
        const symbol = this.config.symbol;
        const leverage = this.config.leverage;

        if (!this.exchange) {
             logger.warn(c.yellow("Cannot set leverage: Exchange not initialized."));
             return;
        }
        // Check if the exchange instance supports setting leverage
        if (!this.exchange.has['setLeverage']) {
            logger.warn(c.yellow(`Exchange ${this.exchange.id} does not support setting leverage via setLeverage() method. Skipping.`));
            return;
        }

        logger.info(`Attempting to set leverage for ${c.bold(symbol)} to ${c.bold(leverage)}x...`);
        try {
            // Bybit V5 might require specific parameters
            const params = {};
            // Assume linear based on previous validation
            if (this.exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                 params.category = 'linear';
                 // For Unified Trading Account (UTA), leverage might be set per symbol without side specification
                 // For Contract accounts, sometimes buy/sell leverage is needed
                 params.buyLeverage = leverage;
                 params.sellLeverage = leverage;
                 logger.debug(`Adding Bybit V5 params for setLeverage: ${JSON.stringify(params)}`);
            }

            // Wrap the call in our retry helper
            const setLeverageFunc = async () => await this.exchange.setLeverage(leverage, symbol, params);
            const response = await retryOnException(
                setLeverageFunc,
                this.config.max_retries,
                this.config.retry_delay,
                [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable], // Retry only connection issues
                'setLeverage'
            );

            logger.info(c.green(`Leverage set command executed successfully for ${symbol}. Exchange response:`), response || "(No detailed response)");

        } catch (e) {
            // Handle specific errors after retries or for non-retryable ones
            if (e instanceof ccxt.ExchangeError) {
                 const errorMsgLower = e.message.toLowerCase();
                 // Handle common responses indicating success or non-critical issues
                 if (errorMsgLower.includes("leverage not modified") || errorMsgLower.includes("same leverage")) {
                      logger.info(`Leverage for ${symbol} was already set to ${leverage}x.`);
                 } else if (errorMsgLower.includes("position exists") || errorMsgLower.includes("open position")) {
                      logger.warn(c.yellow(`Cannot modify leverage for ${symbol} as an open position exists. Using current leverage.`));
                 }
                 // Handle critical errors
                 else if (errorMsgLower.includes("insufficient margin")) {
                      logger.error(c.red(`Failed to set leverage for ${symbol} due to insufficient margin: ${e.message}`));
                      // Consider if this should halt the bot, depending on strategy needs.
                 } else if (errorMsgLower.includes("invalid leverage") || errorMsgLower.includes("leverage limit")) {
                      logger.error(c.red(`Failed to set leverage: Invalid value (${leverage}) or exceeds limits for ${symbol}. Check exchange rules: ${e.message}`));
                      throw new Error(`Invalid leverage configuration: ${e.message}`); // Throw as it's a config issue
                 }
                 // Handle other exchange-specific errors
                 else {
                      logger.error(c.red(`Failed to set leverage for ${symbol}: ${e.constructor.name} - ${e.message}`), e.stack);
                      // Decide if this is critical. Continue cautiously for now.
                 }
            } else if (e instanceof ccxt.AuthenticationError) {
                 logger.error(c.red(`Authentication failed while trying to set leverage: ${e.message}. Check API permissions.`));
                 throw new Error(`Authentication failed setting leverage: ${e.message}`); // Critical init failure
            } else if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout) {
                 // Should have been handled by retry, but log if it somehow gets here
                 logger.warn(c.yellow(`Could not set leverage due to persistent connection issue after retries: ${e.message}. Assuming current leverage is acceptable.`));
            } else {
                // Catch-all for unexpected errors
                logger.error(c.red(`An unexpected error occurred while setting leverage: ${e.constructor.name} - ${e.message}`), e.stack);
                throw new Error(`Unexpected error setting leverage: ${e.message}`); // Treat as critical init failure
            }
        }
    }

    // --- Data Fetching Methods (with Cache and Retries) ---

    /**
     * Fetches OHLCV (candlestick) data for the configured symbol and timeframe.
     * @param {number} limit - The maximum number of candles to fetch.
     * @returns {Promise<Array<Array<number>> | null>} - Array of OHLCV candles or null on failure.
     */
    async fetchOhlcv(limit = 100) {
        const cacheName = "ohlcv";
        const cacheKey = `${this.config.symbol}_${this.config.timeframe}_${limit}`;
        const cachedData = this._getCache(cacheName, cacheKey);
        if (cachedData) return cachedData; // Return cached data if valid

        logger.debug(`Fetching ${limit} OHLCV candles for ${this.config.symbol} (${this.config.timeframe})...`);

        // Define the async function to be passed to the retry helper
        const fetchFunc = async () => {
             if (!this.exchange || !this.exchange.has['fetchOHLCV']) {
                 throw new ccxt.NotSupported("Exchange not initialized or does not support fetchOHLCV.");
             }
             // Fetch OHLCV data: symbol, timeframe, since (undefined), limit, params
             const params = {};
             if (this.exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                 params.category = 'linear'; // Assuming linear
             }
             return await this.exchange.fetchOHLCV(this.config.symbol, this.config.timeframe, undefined, limit, params);
        };

        try {
            // Execute the fetch function with retry logic
            const ohlcvData = await retryOnException(
                fetchFunc,
                this.config.max_retries,
                this.config.retry_delay,
                undefined, // Use default allowedExceptions
                'fetchOhlcv'
            );

            // Basic validation of the returned data structure
            if (!Array.isArray(ohlcvData)) {
                 logger.warn(c.yellow(`fetchOHLCV returned non-array data type: ${typeof ohlcvData}.`), ohlcvData);
                 return null; // Don't cache invalid type
            }
            if (ohlcvData.length > 0 && (!Array.isArray(ohlcvData[0]) || ohlcvData[0].length < OHLCV_SCHEMA.length)) { // Need at least the defined schema elements
                 logger.warn(c.yellow(`Received malformed OHLCV data structure. Expected array of arrays with >= ${OHLCV_SCHEMA.length} elements. First item:`), ohlcvData[0]);
                 return null; // Don't cache malformed structure
            }

            // Store valid data in cache
            this._setCache(cacheName, cacheKey, ohlcvData);

            if (ohlcvData.length > 0) {
                 const lastDt = new Date(ohlcvData[ohlcvData.length - 1][OHLCV_INDEX.TIMESTAMP]).toISOString();
                 logger.debug(`Fetched ${ohlcvData.length} OHLCV candles successfully. Last timestamp: ${lastDt}`);
            } else {
                 logger.debug("Fetched 0 OHLCV candles.");
            }

            // Return the fetched data (direct reference for performance)
            return ohlcvData;

        } catch (e) {
            // Catch errors *after* retry logic has failed or for non-retryable ones
            logger.error(c.red(`Failed to fetch OHLCV data for ${this.config.symbol} after retries or due to non-retryable error: ${e.constructor.name} - ${e.message}`), e.stack);
            return null; // Return null to indicate failure
        }
    }

    /**
     * Fetches the order book for the configured symbol.
     * @returns {Promise<object | null>} - Order book object { bids: [], asks: [], ... } or null on failure.
     */
    async fetchOrderBook() {
        const cacheName = "order_book";
        const depth = this.config.order_book_depth;
        const cacheKey = `${this.config.symbol}_${depth}`;
        const cachedData = this._getCache(cacheName, cacheKey);
        if (cachedData) return cachedData;

        logger.debug(`Fetching order book for ${this.config.symbol} (depth: ${depth})...`);

        const fetchFunc = async () => {
            if (!this.exchange || !this.exchange.has['fetchOrderBook']) {
                 throw new ccxt.NotSupported("Exchange not initialized or does not support fetchOrderBook.");
             }
            // Fetch order book: symbol, limit (depth), params
            const params = {};
            if (this.exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                 params.category = 'linear'; // Assuming linear
             }
            return await this.exchange.fetchOrderBook(this.config.symbol, depth, params);
        }

        try {
            const orderBookData = await retryOnException(
                fetchFunc,
                this.config.max_retries,
                this.config.retry_delay,
                undefined, // Use default retryable errors
                'fetchOrderBook'
            );

            // Validate the structure of the order book data
            if (!orderBookData || typeof orderBookData !== 'object' || !Array.isArray(orderBookData.bids) || !Array.isArray(orderBookData.asks)) {
                logger.warn(c.yellow("fetchOrderBook returned invalid or incomplete data structure."), orderBookData);
                return null;
            }

            this._setCache(cacheName, cacheKey, orderBookData);
            logger.debug(`Fetched order book successfully: ${orderBookData.bids.length} bids, ${orderBookData.asks.length} asks.`);
            return orderBookData; // Return direct reference

        } catch (e) {
            logger.error(c.red(`Failed to fetch order book for ${this.config.symbol} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
            return null;
        }
    }

     /**
      * Fetches the current position for the configured symbol.
      * Handles potential variations in CCXT responses and hedge mode.
      * Returns an aggregated position if multiple entries exist (e.g., hedge mode).
      * @returns {Promise<{side: PositionSide, size: number, entryPrice: number}>} - Position details.
      */
     async getPosition() {
        const cacheName = "position";
        const cacheKey = this.config.symbol;
        const cachedData = this._getCache(cacheName, cacheKey);

        // Validate cached structure before returning
        if (cachedData && typeof cachedData === 'object' &&
            cachedData.side !== undefined && cachedData.size !== undefined && cachedData.entryPrice !== undefined) {
             // Return direct reference. Caller should not modify.
             return cachedData;
        } else if (cachedData) {
             logger.warn(c.yellow("Invalid position data found in cache. Clearing and fetching fresh data."));
             this._setCache(cacheName, cacheKey, null); // Clear invalid cache entry
        }

        // Default return value if no position or error
        const defaultReturn = { side: PositionSide.NONE, size: 0.0, entryPrice: 0.0 };

        // --- Dry Run Simulation ---
        if (this.config.dry_run) {
            logger.debug(c.magenta("DRY RUN: Simulating no open position (for getPosition)."));
            // Cache the default state for dry run consistency within a cycle
            this._setCache(cacheName, cacheKey, defaultReturn);
            return defaultReturn;
        }

        logger.debug(`Fetching position for ${this.config.symbol}...`);

        // Define the async function for fetching positions
        const fetchFunc = async () => {
            if (!this.exchange) throw new Error("Exchange not initialized.");

            // Prefer fetchPositions for Bybit V5
            let positions = [];
            const symbolsToFetch = [this.config.symbol]; // Fetch only the relevant symbol
            const params = {};
             if (this.exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                 params.category = 'linear';
                 logger.debug(`Using Bybit V5 params for fetchPositions: ${JSON.stringify(params)}`);
            }

            // Bybit V5 primarily uses fetchPositions
            if (this.exchange.has['fetchPositions']) {
                 positions = await this.exchange.fetchPositions(symbolsToFetch, params);
            } else if (this.exchange.has['fetchPosition']) {
                 // Fallback to fetchPosition if fetchPositions isn't supported (less likely for Bybit V5)
                 logger.debug("fetchPositions not supported, attempting fetchPosition.");
                 try {
                     const positionData = await this.exchange.fetchPosition(this.config.symbol, params);
                     if (positionData && typeof positionData === 'object' && positionData.symbol === this.config.symbol) {
                          positions = [positionData];
                     } else {
                          positions = [];
                     }
                 } catch (e) {
                     if (e instanceof ccxt.NotSupported) {
                          throw new ccxt.NotSupported("Neither fetchPositions nor fetchPosition is supported by this exchange configuration.");
                     } else {
                          throw e; // Re-throw other errors
                     }
                 }
            } else {
                 throw new ccxt.NotSupported("Position fetching (fetchPositions/fetchPosition) not supported by this exchange configuration.");
            }

            // Filter results rigorously, as fetchPositions might return unrelated data
            return positions.filter(p => p && p.symbol === this.config.symbol);
        }

        try {
            // Execute fetch with retry logic
            const relevantPositions = await retryOnException(
                fetchFunc,
                this.config.max_retries,
                this.config.retry_delay,
                undefined, // Use default retryable errors
                'getPosition'
            );

            // If no relevant positions found after filtering
            if (!Array.isArray(relevantPositions) || relevantPositions.length === 0) {
                logger.debug(`No open positions found for symbol ${this.config.symbol}.`);
                this._setCache(cacheName, cacheKey, defaultReturn);
                return defaultReturn;
            }

            // --- Aggregate Position Info (Handles Hedge Mode / Multiple Entries) ---
            let netSize = 0.0;
            let totalValue = 0.0;    // Sum of (absolute_size * entry_price)
            let totalAbsSize = 0.0; // Sum of absolute_size

            const market = this.exchange.market(this.config.symbol);
            // Determine a small tolerance based on minimum order size for near-zero checks
            const minAmount = market?.limits?.amount?.min ?? 1e-9; // Default to small value if limits missing
            const sizeTolerance = Math.max(1e-9, minAmount / 100); // Use 1/100th of min amount or 1e-9

            for (const pos of relevantPositions) {
                 // Safely extract position details, providing fallbacks
                 // Bybit V5 fields in `info`: 'size', 'avgPrice', 'positionValue', 'side' (Long/Short/None)
                 const info = pos.info || {}; // Access raw info field safely
                 const sizeStr = info.size ?? pos.contracts ?? pos.contractSize ?? pos.size ?? '0';
                 const side = (info.side || pos.side || '').toLowerCase(); // Prefer 'info.side' (Long/Short/None)
                 const entryPriceStr = info.avgPrice ?? pos.entryPrice ?? pos.markPrice ?? '0'; // Use avgPrice or markPrice as fallback

                 let sizeNum = 0.0;
                 let entryPriceNum = 0.0;
                 try {
                     sizeNum = parseFloat(sizeStr);
                     entryPriceNum = parseFloat(entryPriceStr);
                     if (isNaN(sizeNum) || isNaN(entryPriceNum)) throw new Error("Parsed value is NaN");
                 } catch (parseError) {
                     logger.warn(c.yellow(`Could not parse size ('${sizeStr}') or entry price ('${entryPriceStr}') for a position entry. Skipping entry. Raw info:`), info);
                     continue; // Skip this entry if parsing fails
                 }

                 const absSize = Math.abs(sizeNum);

                 // Ignore entries with negligible size or where Bybit V5 side is 'None'
                 if (absSize < sizeTolerance || side === 'none') {
                     logger.debug(`Ignoring position entry: Size=${sizeNum.toFixed(8)}, Side='${side}'. Raw info:`, info);
                     continue;
                 }

                 logger.debug("Processing position entry:", { symbol: pos.symbol, side, size: sizeNum, entryPrice: entryPriceNum, info: info });

                 let positionDirection = 0; // +1 for long, -1 for short
                 if (side === 'long' || side === 'buy') { // V5 uses 'Buy'/'Sell' in some contexts, CCXT usually 'long'/'short'
                     positionDirection = 1;
                 } else if (side === 'short' || side === 'sell') {
                     positionDirection = -1;
                 } else {
                     // Ambiguous entry
                     logger.warn(c.yellow(`Position entry found with ambiguous/missing side ('${side}') and non-zero size (${sizeNum}). Skipping.`));
                     continue;
                 }

                 // Accumulate net size and weighted value
                 netSize += positionDirection * absSize;
                 totalValue += absSize * entryPriceNum;
                 totalAbsSize += absSize;
            } // End loop through position entries

            // --- Determine Final Position State ---
            // Check if the total absolute size is negligible after summing up entries
            if (totalAbsSize < sizeTolerance) {
                 logger.debug("Net position size is effectively zero after processing all entries.");
                 this._setCache(cacheName, cacheKey, defaultReturn);
                 return defaultReturn;
            }

            // Calculate the weighted average entry price
            const avgEntryPrice = totalValue / totalAbsSize;
            let finalPositionSide = PositionSide.NONE;
            let finalAbsSize = 0.0;

            // Determine final side based on the sign of the net size
            if (netSize > sizeTolerance) { // Net long position
                 finalPositionSide = PositionSide.LONG;
                 finalAbsSize = netSize;
                 logger.info(c.green(`Detected NET LONG position: Size=${finalAbsSize.toFixed(8)}, Avg Entry=${avgEntryPrice.toFixed(4)}`));
            } else if (netSize < -sizeTolerance) { // Net short position
                 finalPositionSide = PositionSide.SHORT;
                 finalAbsSize = Math.abs(netSize); // Store size as positive number
                 logger.info(c.red(`Detected NET SHORT position: Size=${finalAbsSize.toFixed(8)}, Avg Entry=${avgEntryPrice.toFixed(4)}`));
            } else {
                 // Net size is negligible after aggregation
                 logger.debug("Net position size negligible after aggregation. Treating as no position.");
                 // Keep default return values (NONE, 0, 0)
            }

            // Prepare and cache the final result
            const result = { side: finalPositionSide, size: finalAbsSize, entryPrice: avgEntryPrice };
            this._setCache(cacheName, cacheKey, result);
            return result;

        } catch (e) {
             // Handle errors during position fetching or processing
             if (e instanceof ccxt.NotSupported) {
                 logger.error(c.red(`Fetching positions not supported by exchange/config: ${e.message}. Assuming no position.`));
             } else if (e instanceof ccxt.AuthenticationError) {
                  logger.error(c.red(`Authentication failed while fetching position: ${e.message}.`));
                  // This might be critical, could throw to stop bot
             } else {
                 // General failure after retries
                 logger.error(c.red(`Failed to fetch or process position for ${this.config.symbol} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
             }
             // Cache the default return on error to avoid repeated failed attempts within cache TTL
             this._setCache(cacheName, cacheKey, defaultReturn);
             return defaultReturn; // Return default (NONE) conservatively
        }
    }

    /**
     * Fetches the account balance for a specific currency.
     * Handles variations in CCXT balance structures, especially for Bybit V5 Unified/Contract accounts.
     * Returns the EQUITY for derivative accounts, which includes unrealized PnL.
     * @param {string | null} currency - The currency code (e.g., 'USDT'). Defaults to config.currency.
     * @returns {Promise<number | null>} - The total available equity/balance, or null on failure. Returns null instead of 0.0 on failure.
     */
    async getBalance(currency = null) {
        const targetCurrency = (currency || this.config.currency).toUpperCase();
        const cacheName = "balance";
        // Include exchange type in key as balance might differ for swap/spot accounts
        const cacheKey = `${targetCurrency}_${this.config.exchange_type}`;
        const cachedData = this._getCache(cacheName, cacheKey);

        // Validate cached data type (must be a number, can be zero or negative)
        if (cachedData !== null && typeof cachedData === 'number' && !isNaN(cachedData)) {
             return cachedData; // Return valid cached balance
        } else if (cachedData !== null) {
             logger.warn(c.yellow(`Invalid balance data type (${typeof cachedData}) found in cache. Fetching fresh data.`));
             this._setCache(cacheName, cacheKey, null); // Clear invalid cache
        }

        const defaultReturn = null; // Return null on failure

        // --- Dry Run Simulation ---
        if (this.config.dry_run) {
            // Simulate a starting balance for calculations
            const simulatedBalance = 10000.0;
            logger.debug(c.magenta(`DRY RUN: Returning simulated balance of ${simulatedBalance.toFixed(2)} ${targetCurrency}`));
            this._setCache(cacheName, cacheKey, simulatedBalance);
            return simulatedBalance;
        }

        logger.debug(`Fetching balance/equity for ${targetCurrency} (Account Type: ${this.config.exchange_type})...`);

        // Define the async fetch function
        const fetchFunc = async () => {
             if (!this.exchange || !this.exchange.has['fetchBalance']) {
                  throw new ccxt.NotSupported("Exchange not initialized or does not support fetchBalance.");
             }
             // Bybit V5 API requires specific parameters depending on the account type
             const params = {};
             if (this.exchange.id === 'bybit') {
                  // Determine the correct account type to query based on the exchange type configured
                  if (['swap', 'future'].includes(this.config.exchange_type)) {
                       params.accountType = 'UNIFIED'; // Or 'CONTRACT' depending on account structure
                       logger.debug(`Fetching balance for Bybit derivatives account (attempting accountType=${params.accountType}).`);
                  } else if (this.config.exchange_type === 'spot') {
                       params.accountType = 'SPOT'; // Or 'FUND'
                       logger.debug(`Fetching balance for Bybit spot account (attempting accountType=${params.accountType}).`);
                  }
             }

             logger.debug(`Calling fetchBalance with params: ${JSON.stringify(params)}`);
             return await this.exchange.fetchBalance(params);
        }

        try {
            // Fetch balance with retry logic
            const balanceInfo = await retryOnException(
                fetchFunc,
                this.config.max_retries,
                this.config.retry_delay,
                undefined, // Use default retryable errors
                'getBalance'
            );

            if (!balanceInfo) {
                 logger.warn(c.yellow("fetchBalance returned null or empty data. Assuming null balance."));
                 this._setCache(cacheName, cacheKey, defaultReturn);
                 return defaultReturn;
            }

            let finalBalance = null; // Initialize to null
            let foundBalance = false;

            // --- Balance Parsing Logic (Multiple Strategies) ---
            logger.debug("Attempting to parse balance/equity from response:", balanceInfo);

            // Strategy 1: Exchange-specific 'info' field (Primary for Bybit V5)
            if (!foundBalance && balanceInfo.info) {
                 const info = balanceInfo.info;
                 logger.debug("Attempting to parse balance from exchange-specific 'info' field:", info);

                 // Bybit V5 structure often involves info.result.list array
                 if (info.result && Array.isArray(info.result.list) && info.result.list.length > 0) {
                      // Determine the target account type(s) based on config
                      let targetAccountTypes = [];
                      if (['swap', 'future'].includes(this.config.exchange_type)) {
                          targetAccountTypes = ['UNIFIED', 'CONTRACT'];
                      } else if (this.config.exchange_type === 'spot') {
                          targetAccountTypes = ['SPOT', 'FUND'];
                      }

                      // Find the entry matching the target currency AND one of the target account types
                      let accountData = null;
                      for (const accType of targetAccountTypes) {
                           accountData = info.result.list.find(item => item.coin === targetCurrency && item.accountType === accType);
                           if (accountData) {
                               logger.debug(`Found entry for currency ${targetCurrency} with matching accountType '${accType}'.`);
                               break;
                           }
                      }

                      // If not found by specific account type, try finding *any* entry for the currency
                      if (!accountData) {
                          const matchingEntries = info.result.list.filter(item => item.coin === targetCurrency);
                          if (matchingEntries.length > 0) {
                               accountData = matchingEntries[0]; // Use the first one found
                               logger.warn(c.yellow(`Could not find balance entry with preferred account types (${targetAccountTypes.join('/')}). Using first entry found for ${targetCurrency} (accountType: ${accountData.accountType || 'N/A'}). Result might be inaccurate.`));
                          }
                      }

                      if (accountData) {
                          logger.debug("Selected account data entry:", accountData);
                          // List of potential keys for EQUITY (Balance + UPL) in Bybit V5 responses
                          const equityKeys = [
                              'equity', 'accountEquity', 'totalEquity', // Prioritize equity keys
                              'walletBalance', // Fallback to wallet balance
                          ];

                          for (const key of equityKeys) {
                               if (accountData[key] !== undefined && accountData[key] !== null && accountData[key] !== '') {
                                    const balStr = String(accountData[key]);
                                    const bal = parseFloat(balStr);
                                    if (!isNaN(bal)) {
                                         finalBalance = bal;
                                         foundBalance = true;
                                         logger.debug(`Found balance/equity using Bybit V5 specific key '${key}' in 'info.result.list'. Value: ${balStr}`);
                                         if (key === 'walletBalance' && ['swap', 'future'].includes(this.config.exchange_type)) {
                                             logger.warn(c.yellow(`Using 'walletBalance' for derivatives. This excludes Unrealized PnL. Risk calculations might be affected.`));
                                         }
                                         break;
                                    } else {
                                         logger.debug(`Key '${key}' found but value ('${balStr}') is not a valid number.`);
                                    }
                               }
                          }
                      } else {
                           logger.debug(`Could not find a suitable account data object for currency ${targetCurrency} within 'info.result.list'.`);
                      }
                 } else {
                      logger.debug("'info.result.list' not found or empty in balance response 'info' field.");
                 }
            } // End Strategy 1 (Bybit V5 Info)


            // Strategy 2: Standard CCXT structure (balanceInfo[currency].total)
            if (!foundBalance && balanceInfo[targetCurrency] && balanceInfo[targetCurrency].total !== undefined && balanceInfo[targetCurrency].total !== null) {
                const balStr = String(balanceInfo[targetCurrency].total);
                const bal = parseFloat(balStr);
                if (!isNaN(bal)) {
                    finalBalance = bal;
                    foundBalance = true;
                    logger.debug(`Found balance using standard CCXT structure: balanceInfo[${targetCurrency}].total. Value: ${balStr}`);
                    if (['swap', 'future'].includes(this.config.exchange_type)) {
                         logger.warn(c.yellow(`Using balanceInfo[${targetCurrency}].total for derivatives. Verify if this includes Unrealized PnL (Equity).`));
                    }
                } else {
                    logger.debug(`Standard CCXT balanceInfo[${targetCurrency}].total found but value ('${balStr}') is not valid number.`);
                }
            }

            // Strategy 3: Standard CCXT free + used (Wallet balance fallback)
            if (!foundBalance && balanceInfo[targetCurrency] && balanceInfo[targetCurrency].free !== undefined && balanceInfo[targetCurrency].used !== undefined) {
                const freeStr = String(balanceInfo[targetCurrency].free);
                const usedStr = String(balanceInfo[targetCurrency].used);
                const free = parseFloat(freeStr);
                const used = parseFloat(usedStr);
                if (!isNaN(free) && !isNaN(used)) {
                    finalBalance = free + used;
                    foundBalance = true;
                    logger.warn(c.yellow(`Used fallback balance calculation (free + used) for ${targetCurrency}. This represents WALLET BALANCE, not EQUITY.`));
                    logger.debug(`Fallback balance: free=${freeStr}, used=${usedStr}, total=${finalBalance}`);
                }
            }

            // Strategy 4: Top-level 'total' dictionary (Less common fallback)
            if (!foundBalance && balanceInfo.total && balanceInfo.total[targetCurrency] !== undefined) {
                 const balStr = String(balanceInfo.total[targetCurrency]);
                 const bal = parseFloat(balStr);
                 if (!isNaN(bal)) {
                     finalBalance = bal;
                     foundBalance = true;
                     logger.debug(`Found balance using top-level 'total' dictionary: balanceInfo.total[${targetCurrency}]. Value: ${balStr}`);
                     if (['swap', 'future'].includes(this.config.exchange_type)) {
                         logger.warn(c.yellow(`Using balanceInfo.total[${targetCurrency}] for derivatives. Verify if this includes Unrealized PnL (Equity).`));
                     }
                 }
            }

            // --- Final Check and Return ---
            if (!foundBalance) {
                logger.warn(c.yellow(`Could not reliably determine balance/equity for currency '${targetCurrency}' from the response structure. Assuming null balance. Please inspect debug logs.`));
                logger.debug("Full raw balance response:", balanceInfo);
                this._setCache(cacheName, cacheKey, defaultReturn);
                return defaultReturn;
            }

            logger.info(c.green(`Fetched balance/equity for ${targetCurrency}: ${finalBalance !== null ? finalBalance.toFixed(4) : 'N/A'}`));
            this._setCache(cacheName, cacheKey, finalBalance);
            return finalBalance;

        } catch (e) {
            logger.error(c.red(`Failed to fetch balance for ${targetCurrency} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
            this._setCache(cacheName, cacheKey, defaultReturn); // Cache failure state
            return defaultReturn;
        }
    }

    /**
     * Fetches the current ticker price for the configured symbol.
     * @returns {Promise<number | null>} - The last price or null on failure.
     */
    async getCurrentPrice() {
        const cacheName = "ticker";
        const cacheKey = this.config.symbol;
        const cachedData = this._getCache(cacheName, cacheKey);

        if (cachedData !== null && typeof cachedData === 'number' && !isNaN(cachedData) && cachedData > 0) {
             return cachedData; // Return valid cached price
        } else if (cachedData !== null) {
             logger.warn(c.yellow(`Invalid ticker price data type (${typeof cachedData}) or value (${cachedData}) found in cache. Fetching fresh data.`));
             this._setCache(cacheName, cacheKey, null); // Clear invalid cache
        }

        logger.debug(`Fetching current ticker price for ${this.config.symbol}...`);
        const defaultReturn = null; // Return null if price cannot be fetched

        const fetchFunc = async () => {
            if (!this.exchange || !this.exchange.has['fetchTicker']) {
                 throw new ccxt.NotSupported("Exchange not initialized or does not support fetchTicker.");
            }
            // Fetch ticker: symbol, params
             const params = {};
             if (this.exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                 params.category = 'linear'; // Assuming linear
             }
            return await this.exchange.fetchTicker(this.config.symbol, params);
        }

        try {
            const ticker = await retryOnException(
                fetchFunc,
                this.config.max_retries,
                this.config.retry_delay,
                undefined, // Use default retryable errors
                'getCurrentPrice' // Function name for logging
            );

            if (!ticker) {
                 logger.warn(c.yellow("fetchTicker returned null or empty data. Cannot determine current price."));
                 this._setCache(cacheName, cacheKey, defaultReturn);
                 return defaultReturn;
            }

            // Extract the price - prioritize 'last', fallback to 'close', 'mark' as last resort?
            let lastPrice = null;
            let priceSource = null;
            if (ticker.last !== undefined && ticker.last !== null && parseFloat(ticker.last) > 0) {
                 lastPrice = parseFloat(ticker.last);
                 priceSource = 'last';
            } else if (ticker.close !== undefined && ticker.close !== null && parseFloat(ticker.close) > 0) {
                 lastPrice = parseFloat(ticker.close);
                 priceSource = 'close';
                 logger.debug("Using 'close' price from ticker as 'last' price was unavailable or invalid.");
            } else if (ticker.mark !== undefined && ticker.mark !== null && parseFloat(ticker.mark) > 0) {
                 lastPrice = parseFloat(ticker.mark);
                 priceSource = 'mark';
                 logger.debug("Using 'mark' price from ticker as 'last' and 'close' were unavailable or invalid.");
            } else {
                 logger.warn(c.yellow("Could not find valid 'last', 'close', or 'mark' price in the ticker response."), ticker);
            }

            // Validate the extracted price
            if (lastPrice === null || isNaN(lastPrice) || lastPrice <= 0) {
                 logger.warn(c.yellow(`Fetched ticker price is invalid (NaN or non-positive). Source: ${priceSource || 'N/A'}, Ticker: `), ticker);
                 this._setCache(cacheName, cacheKey, defaultReturn);
                 return defaultReturn;
            }

            logger.debug(`Current price fetched successfully (Source: ${priceSource}): ${lastPrice.toFixed(4)}`);
            this._setCache(cacheName, cacheKey, lastPrice);
            return lastPrice;

        } catch (e) {
            logger.error(c.red(`Failed to fetch ticker price for ${this.config.symbol} after retries: ${e.constructor.name} - ${e.message}`), e.stack);
            this._setCache(cacheName, cacheKey, defaultReturn); // Cache failure
            return defaultReturn;
        }
    }
}


// --- Indicator Calculations ---
// Contains static methods for calculating technical indicators.
class Indicators {
    /**
     * Calculates the Average True Range (ATR).
     * @param {Array<Array<number>>} ohlcv - Array of OHLCV data [[ts, o, h, l, c, v], ...].
     * @param {number} period - The ATR period.
     * @returns {number | null} - The calculated ATR value or null if calculation fails or result is non-positive.
     */
    static calculateAtr(ohlcv, period) {
        // Validate period input
        if (!Number.isInteger(period) || period <= 0) {
            logger.warn(c.yellow(`Invalid period (${period}) provided for ATR calculation.`));
            return null;
        }
        // Ensure sufficient data (need 'period' intervals, so 'period + 1' candles)
        const requiredCandles = period + 1;
        if (!Array.isArray(ohlcv) || ohlcv.length < requiredCandles) {
            logger.debug(`Insufficient data for ATR(${period}): need ${requiredCandles} candles, got ${ohlcv.length}`);
            return null;
        }

        // Get indices from schema
        const highIdx = OHLCV_INDEX.HIGH;
        const lowIdx = OHLCV_INDEX.LOW;
        const closeIdx = OHLCV_INDEX.CLOSE;
        const trueRanges = [];

        try {
            // Calculate True Range (TR) for the last 'period' candles
            // Loop starts from 'ohlcv.length - period' up to the last candle
            for (let i = ohlcv.length - period; i < ohlcv.length; i++) {
                 if (i === 0) continue; // Should not happen due to length check, but safe guard.

                 const high = parseFloat(ohlcv[i][highIdx]);
                 const low = parseFloat(ohlcv[i][lowIdx]);
                 const prevClose = parseFloat(ohlcv[i - 1][closeIdx]);

                 // Check for invalid numbers
                 if (isNaN(high) || isNaN(low) || isNaN(prevClose)) {
                     throw new Error(`NaN value encountered in OHLCV data at index ${i} or ${i-1} during ATR calculation.`);
                 }

                 // Calculate True Range
                 const tr = Math.max(
                     high - low,              // High - Low
                     Math.abs(high - prevClose), // Abs(High - Previous Close)
                     Math.abs(low - prevClose)   // Abs(Low - Previous Close)
                 );
                 trueRanges.push(tr);
            }

             // Ensure we calculated the correct number of TRs
             if (trueRanges.length !== period) {
                logger.error(c.red(`ATR calculation failed: Expected ${period} True Range values, but collected ${trueRanges.length}. Check OHLCV data quality.`));
                return null;
            }

            // Calculate the Average True Range (SMA of TRs)
            if (period === 0) return null; // Avoid division by zero
            const atr = trueRanges.reduce((sum, val) => sum + val, 0) / period;

            // Check for non-positive ATR which can break logic
            if (isNaN(atr) || atr <= 1e-12) { // Use tolerance for near-zero check
                 logger.warn(c.yellow(`Calculated ATR(${period}) is non-positive or negligible (${atr}). Returning null.`));
                 return null;
            }

            // logger.debug(`Calculated ATR(${period}) = ${atr.toFixed(8)}`); // Log with more precision
            return atr;

        } catch (e) {
            logger.error(c.red(`Error during ATR calculation: ${e.message}`), e.stack);
            return null;
        }
    }

    /**
     * Calculates the SuperTrend indicator (Simplified Stateless Version).
     * ============================= WARNING ==================================
     * This is a SIMPLIFIED, STATELESS calculation. It approximates the previous
     * SuperTrend state based on recent candles. IT WILL NOT MATCH a stateful
     * implementation (like TradingView's) perfectly, especially near trend flips
     * or during volatile periods. Use with caution and understand its limitations.
     * For accurate SuperTrend, a stateful calculation across candles is required.
     * ========================================================================
     * @param {Array<Array<number>>} ohlcv - Array of OHLCV data [[ts, o, h, l, c, v], ...].
     * @param {number} period - The SuperTrend period (used for ATR).
     * @param {number} multiplier - The ATR multiplier.
     * @returns {{value: number | null, isUptrend: boolean | null}} - Approximated SuperTrend value and trend direction for the latest candle.
     */
    static calculateSupertrend(ohlcv, period, multiplier) {
         // Require at least period + 3 candles for safety margin in approximations.
         const requiredCandles = period + 3;
         if (!Array.isArray(ohlcv) || ohlcv.length < requiredCandles) {
             logger.debug(`Insufficient data for simplified Supertrend(${period}): need ${requiredCandles} candles, got ${ohlcv.length}`);
             return { value: null, isUptrend: null };
         }
         // Validate inputs
         if (!Number.isInteger(period) || period <= 0 || isNaN(multiplier) || multiplier <= 0) {
             logger.warn(c.yellow(`Invalid Supertrend parameters: period=${period}, multiplier=${multiplier}`));
             return { value: null, isUptrend: null };
         }

         const highIdx = OHLCV_INDEX.HIGH;
         const lowIdx = OHLCV_INDEX.LOW;
         const closeIdx = OHLCV_INDEX.CLOSE;

         try {
             // --- Calculate ATR ending at the second-to-last candle (index -2) ---
             const atrSlice = ohlcv.slice(0, -1); // Data up to candle at index -2
             const atr = Indicators.calculateAtr(atrSlice, period);
             if (atr === null) { // ATR validation already checks for <= 0
                 logger.warn(c.yellow(`Failed to calculate valid ATR for Supertrend (using data up to second-to-last candle).`));
                 return { value: null, isUptrend: null };
             }

             // --- Get data for the latest candle (index -1) ---
             const latestCandle = ohlcv[ohlcv.length - 1];
             const high = parseFloat(latestCandle[highIdx]);
             const low = parseFloat(latestCandle[lowIdx]);
             const close = parseFloat(latestCandle[closeIdx]);

             // --- Get data for the previous candle (index -2) ---
             const prevCandle = ohlcv[ohlcv.length - 2];
             const prevHigh = parseFloat(prevCandle[highIdx]);
             const prevLow = parseFloat(prevCandle[lowIdx]);
             const prevClose = parseFloat(prevCandle[closeIdx]);

             // --- Get data for the candle before previous (index -3) ---
             const prevPrevCandle = ohlcv[ohlcv.length - 3];
             const prevPrevClose = parseFloat(prevPrevCandle[closeIdx]);

             // Check for NaN values in required prices
             if ([high, low, close, prevHigh, prevLow, prevClose, prevPrevClose].some(isNaN)) {
                 throw new Error("NaN value encountered in required OHLCV data for Supertrend calculation.");
             }

             // --- Calculate Basic Upper/Lower Bands for the LATEST candle (-1) ---
             const hl2 = (high + low) / 2;
             const basicUpper = hl2 + multiplier * atr;
             const basicLower = hl2 - multiplier * atr;

             // --- Approximate Previous Supertrend State (using candle at -2) ---
             let prevFinalUpperApprox = null; // Estimated upper band after candle -2 closed
             let prevFinalLowerApprox = null; // Estimated lower band after candle -2 closed
             let prevTrendUpApprox = null;    // Estimated trend after candle -2 closed

             // Calculate ATR ending at candle -3 to approximate bands relevant to candle -2
             const prevAtrSlice = ohlcv.slice(0, -2); // Data up to candle at index -3
             const prevAtr = Indicators.calculateAtr(prevAtrSlice, period);

             if (prevAtr !== null) {
                 // Calculate basic bands for the PREVIOUS candle (-2) using prevAtr
                 const prevHl2 = (prevHigh + prevLow) / 2;
                 const prevBasicUpperApprox = prevHl2 + multiplier * prevAtr;
                 const prevBasicLowerApprox = prevHl2 - multiplier * prevAtr;

                 // Estimate trend *before* candle -2 closed (based on candle -3 close vs its bands)
                 let trendBeforePrevApprox = null;
                 if (prevPrevClose > prevBasicLowerApprox) trendBeforePrevApprox = true;
                 else if (prevPrevClose < prevBasicUpperApprox) trendBeforePrevApprox = false;

                 // Estimate the final state *after* candle -2 closed
                 if (trendBeforePrevApprox === true) { // Approx trend was UP before -2 closed
                     prevFinalLowerApprox = prevBasicLowerApprox; // Simplified start
                     if (prevClose < prevFinalLowerApprox) { // Flip DOWN
                         prevTrendUpApprox = false;
                         prevFinalUpperApprox = prevBasicUpperApprox; // Upper band resets
                     } else { // Stayed UP
                         prevTrendUpApprox = true;
                         // Approx update lower band (max of current basic and prev approx band)
                         prevFinalLowerApprox = Math.max(prevFinalLowerApprox, prevBasicLowerApprox);
                     }
                 } else if (trendBeforePrevApprox === false) { // Approx trend was DOWN before -2 closed
                     prevFinalUpperApprox = prevBasicUpperApprox; // Simplified start
                     if (prevClose > prevFinalUpperApprox) { // Flip UP
                         prevTrendUpApprox = true;
                         prevFinalLowerApprox = prevBasicLowerApprox; // Lower band resets
                     } else { // Stayed DOWN
                         prevTrendUpApprox = false;
                         // Approx update upper band (min of current basic and prev approx band)
                         prevFinalUpperApprox = Math.min(prevFinalUpperApprox, prevBasicUpperApprox);
                     }
                 } else {
                     // Fallback: Determine trend based on prevClose vs its *own* basic bands
                     if (prevClose > prevBasicLowerApprox) prevTrendUpApprox = true;
                     else if (prevClose < prevBasicUpperApprox) prevTrendUpApprox = false;
                     // If fallback worked, set the approx bands
                     if(prevTrendUpApprox !== null) {
                        prevFinalLowerApprox = prevBasicLowerApprox;
                        prevFinalUpperApprox = prevBasicUpperApprox;
                     } else {
                         logger.debug("Could not approximate Supertrend trend before previous candle (-2) even with fallback.");
                     }
                 }
             } else {
                 logger.debug("Could not calculate previous ATR (ending at -3) for Supertrend previous state approximation. Using simpler fallback.");
                 // Fallback: Determine previous trend based on prevClose (-2) vs *current* basic bands (very crude)
                 if (prevClose > basicLower) prevTrendUpApprox = true;
                 else if (prevClose < basicUpper) prevTrendUpApprox = false;
                 // If fallback worked, set the approx bands based on current basic bands
                 if(prevTrendUpApprox !== null) {
                     prevFinalLowerApprox = basicLower;
                     prevFinalUpperApprox = basicUpper;
                 } else {
                      logger.debug("Could not approximate Supertrend trend before previous candle (-2) using crude fallback.");
                 }
             }
             logger.debug(`Supertrend Prev State Approx: TrendUp=${prevTrendUpApprox}, Lower=${prevFinalLowerApprox?.toFixed(4)}, Upper=${prevFinalUpperApprox?.toFixed(4)}`);


             // --- Determine Current SuperTrend Value and Trend (for candle -1) ---
             let currentStValue = null;
             let currentTrendUp = null;

             if (prevTrendUpApprox === null) {
                 // If previous trend approximation failed completely, fallback to current close vs current bands
                 logger.debug("Supertrend previous state approximation failed. Determining current trend based on current close vs current basic bands.");
                 if (close > basicLower) {
                     currentTrendUp = true;
                     currentStValue = basicLower;
                 } else if (close < basicUpper) {
                     currentTrendUp = false;
                     currentStValue = basicUpper;
                 } else {
                     logger.debug("Supertrend state indeterminate (prev state unknown, current close between basic bands).");
                     return { value: null, isUptrend: null };
                 }
             } else {
                 // Use the approximated previous state to determine the current state
                 if (prevTrendUpApprox === true) { // Previous trend was UP
                     const finalLower = Math.max(basicLower, prevFinalLowerApprox ?? -Infinity);
                     if (close < finalLower) { // Flip DOWN
                         currentTrendUp = false;
                         currentStValue = basicUpper;
                     } else { // Stay UP
                         currentTrendUp = true;
                         currentStValue = finalLower;
                     }
                 } else { // Previous trend was DOWN
                     const finalUpper = Math.min(basicUpper, prevFinalUpperApprox ?? Infinity);
                     if (close > finalUpper) { // Flip UP
                         currentTrendUp = true;
                         currentStValue = basicLower;
                     } else { // Stay DOWN
                         currentTrendUp = false;
                         currentStValue = finalUpper;
                     }
                 }
             }

             // Final validation of calculated ST value
             if (currentStValue === null || isNaN(currentStValue) || currentStValue <= 0) {
                  logger.warn(c.yellow(`Calculated Supertrend value (${currentStValue}) is invalid or non-positive. Returning null.`));
                  return { value: null, isUptrend: null };
             }

             return { value: currentStValue, isUptrend: currentTrendUp };

         } catch (e) {
             logger.error(c.red(`Error during simplified Supertrend calculation: ${e.message}`), e.stack);
             return { value: null, isUptrend: null }; // Return null on error
         }
    }

    /**
     * Calculates the ratio of short-term average volume to long-term average volume.
     * @param {Array<Array<number>>} ohlcv - Array of OHLCV data [[ts, o, h, l, c, v], ...].
     * @param {number} shortPeriod - The short moving average period for volume.
     * @param {number} longPeriod - The long moving average period for volume.
     * @returns {number | null} - The volume ratio or null if calculation fails or result is invalid.
     */
    static calculateVolumeRatio(ohlcv, shortPeriod, longPeriod) {
        // Validate periods
        if (!Number.isInteger(shortPeriod) || !Number.isInteger(longPeriod) || shortPeriod <= 0 || longPeriod <= 0 || shortPeriod >= longPeriod) {
            logger.warn(c.yellow(`Invalid periods for Volume Ratio calculation: short=${shortPeriod}, long=${longPeriod}`));
            return null;
        }
        // Ensure sufficient data
        if (!Array.isArray(ohlcv) || ohlcv.length < longPeriod) {
            logger.debug(`Insufficient data for Volume Ratio(${shortPeriod}/${longPeriod}): need ${longPeriod} candles, got ${ohlcv.length}`);
            return null;
        }

        const volumeIdx = OHLCV_INDEX.VOLUME;
        try {
            // Extract volumes for the long period
            const volumes = ohlcv.slice(-longPeriod).map(candle => {
                const vol = parseFloat(candle[volumeIdx]);
                if (isNaN(vol) || vol < 0) throw new Error(`Invalid (NaN or negative) volume encountered at timestamp ${candle[OHLCV_INDEX.TIMESTAMP]}.`);
                return vol;
            });

            // Check if we got the expected number of volumes
            if (volumes.length !== longPeriod) {
                throw new Error(`Volume extraction mismatch: expected ${longPeriod}, got ${volumes.length}.`);
            }

            // Calculate short-term average volume
            const shortVolumes = volumes.slice(-shortPeriod);
            if (shortPeriod === 0) return null; // Avoid division by zero
            const shortAvgVol = shortVolumes.reduce((sum, v) => sum + v, 0) / shortPeriod;

            // Calculate long-term average volume
            if (longPeriod === 0) return null; // Avoid division by zero
            const longAvgVol = volumes.reduce((sum, v) => sum + v, 0) / longPeriod;

            // Handle division by zero or negligible long-term volume
            if (longAvgVol <= 1e-12) { // Use a small tolerance
                logger.debug("Long term average volume is zero or negligible. Cannot calculate volume ratio.");
                return null;
            }

            // Calculate the ratio
            const volumeRatio = shortAvgVol / longAvgVol;

            // Validate the result
            if (isNaN(volumeRatio) || volumeRatio < 0) {
                 logger.warn(c.yellow(`Calculated volume ratio is invalid (NaN or negative): ${volumeRatio}. ShortAvg=${shortAvgVol}, LongAvg=${longAvgVol}`));
                 return null;
            }
            // logger.debug(`Volume Ratio(${shortPeriod}/${longPeriod}): ${volumeRatio.toFixed(2)} (ShortAvg=${shortAvgVol.toFixed(2)}, LongAvg=${longAvgVol.toFixed(2)})`);
            return volumeRatio;

        } catch (e) {
            logger.error(c.red(`Error processing volume data for Volume Ratio: ${e.message}`), e.stack);
            return null;
        }
    }

    /**
     * Calculates the order book pressure ratio (Total Bid Volume / Total Volume) in the top N levels.
     * @param {object | null} orderBook - The order book object from CCXT { bids: [[price, amount], ...], asks: [[price, amount], ...] } or null.
     * @param {number} depth - The number of order book levels (depth) to consider.
     * @returns {number | null} - The buy pressure ratio (0 to 1) or null if calculation fails or input is invalid.
     */
    static calculateOrderBookPressure(orderBook, depth) {
        // Validate depth
        if (!Number.isInteger(depth) || depth <= 0) {
             logger.warn(c.yellow(`Invalid depth (${depth}) specified for order book pressure calculation.`));
             return null;
        }
        // Validate order book structure
        if (!orderBook || !Array.isArray(orderBook.bids) || !Array.isArray(orderBook.asks)) {
            logger.debug("Invalid or incomplete order book data received for pressure calculation.");
            return null;
        }

        try {
            // Extract top 'depth' bids and asks, ensuring they are valid [price, amount] pairs
            // Filter out levels with non-numeric, NaN, or non-positive amounts
            const filterValidLevel = level => Array.isArray(level) && level.length >= 2 && typeof level[1] === 'number' && !isNaN(level[1]) && level[1] > 0;
            const topBids = orderBook.bids.slice(0, depth).filter(filterValidLevel);
            const topAsks = orderBook.asks.slice(0, depth).filter(filterValidLevel);

            // Sum the amounts (volumes) from the top levels
            const bidVolume = topBids.reduce((sum, level) => sum + level[1], 0);
            const askVolume = topAsks.reduce((sum, level) => sum + level[1], 0);

            // Check for NaN results after summing (unlikely with filter, but safe)
            if (isNaN(bidVolume) || isNaN(askVolume)) {
                throw new Error("NaN value encountered after summing order book volumes.");
            }

            const totalVolume = bidVolume + askVolume;

            // Handle case where total volume is zero or negligible
            if (totalVolume <= 1e-12) { // Use tolerance
                logger.debug(`Total volume in order book top ${depth} levels is zero or negligible. Cannot calculate pressure ratio.`);
                return null; // Return null as ratio is undefined/unreliable
            }

            // Calculate the buy pressure ratio
            const buyPressureRatio = bidVolume / totalVolume;
            // logger.debug(`Order Book Pressure (Depth ${depth}): ${buyPressureRatio.toFixed(3)} (BidVol=${bidVolume.toFixed(2)}, AskVol=${askVolume.toFixed(2)})`);
            return buyPressureRatio;

        } catch (e) {
             logger.error(c.red(`Error calculating order book pressure: ${e.message}`), e.stack);
             return null;
        }
    }
}


// --- Order Manager ---
// Handles placing, canceling, and managing orders, including state persistence for SL.
class OrderManager {
    constructor(exchangeMgr, config, notificationSvc) {
        this.exchangeMgr = exchangeMgr; // Instance of ExchangeManager
        this.config = config;           // Instance of Config
        this.notifier = notificationSvc; // Instance of NotificationService
        this.active_sl_order_id = null; // Stores the ID of the currently active stop-loss order (placed manually by TSL)
    }

    // Asynchronous initialization (e.g., loading state from file)
    async initialize() {
         await this._loadState(); // Load persistent state on startup
    }

    // Loads the active SL order ID from the state file
    async _loadState() {
        const stateFile = this.config.state_file;
        try {
            // Check if the state file exists using fs.promises.stat (throws error if not found)
            await fs.stat(stateFile);
            logger.info(`Loading state from ${stateFile}...`);
            const data = await fs.readFile(stateFile, 'utf-8');
            const state = JSON.parse(data); // Parse the JSON content

            // Validate the loaded state structure and value
            const loadedId = state?.active_sl_order_id;
            if (loadedId && typeof loadedId === 'string' && loadedId.trim()) {
                this.active_sl_order_id = loadedId.trim();
                logger.info(`Loaded active SL order ID from state: ${c.bold(this.active_sl_order_id)}. Status will be verified if needed.`);
            } else {
                logger.info("No valid active SL order ID found in state file or state file empty.");
                this.active_sl_order_id = null;
            }
        } catch (err) {
             if (err.code === 'ENOENT') {
                  // File does not exist - this is normal on first run
                  logger.info(`State file '${stateFile}' not found. Initializing with no active SL.`);
                  this.active_sl_order_id = null;
             } else if (err instanceof SyntaxError) {
                  // JSON parsing error
                  logger.error(c.red(`Failed to parse JSON from state file ${stateFile}: ${err.message}. Resetting state.`));
                  this.active_sl_order_id = null;
                  // Consider renaming the corrupt file?
                  // await fs.rename(stateFile, `${stateFile}.corrupt.${Date.now()}`).catch(renameErr => logger.error(`Failed to rename corrupt state file: ${renameErr.message}`));
             } else {
                  // Other file system errors (permissions, etc.)
                  logger.error(c.red(`Failed to read or process state file ${stateFile}: ${err.message}. Resetting state.`), err);
                  this.active_sl_order_id = null;
             }
        }
    }

    // Saves the current active SL order ID to the state file atomically
    async _saveState() {
        const stateFile = this.config.state_file;
        // Ensure we save null if the ID is not a valid string
        const valueToSave = (this.active_sl_order_id && typeof this.active_sl_order_id === 'string' && this.active_sl_order_id.trim())
                            ? this.active_sl_order_id.trim()
                            : null;

        logger.debug(`Saving state (Active SL ID: ${valueToSave || 'None'}) to ${stateFile}...`);

        const stateData = { active_sl_order_id: valueToSave };
        const tempStateFile = stateFile + ".tmp." + process.pid; // Temporary file path with PID for robustness

        try {
            // 1. Write the new state to a temporary file
            await fs.writeFile(tempStateFile, JSON.stringify(stateData, null, 4), 'utf-8'); // Pretty-print JSON

            // 2. Atomically rename the temporary file to the actual state file
            await fs.rename(tempStateFile, stateFile);

            logger.debug("State saved successfully.");
        } catch (err) {
            logger.error(c.red(`Failed to write state to ${stateFile}: ${err.message}`), err);
            // Attempt to clean up the temporary file if it exists after a failure
            try {
                 await fs.unlink(tempStateFile);
                 logger.debug(`Removed temporary state file ${tempStateFile} after write error.`);
            } catch (rmErr) {
                 // Ignore errors during cleanup, but log them
                 if (rmErr.code !== 'ENOENT') { // Don't log if file doesn't exist
                    logger.error(c.red(`Failed to remove temporary state file ${tempStateFile} after write error: ${rmErr.message}`));
                 }
            }
        }
    }

     /**
      * Internal helper to execute exchange requests (place/cancel orders, fetch order) with centralized
      * retry logic, error handling, and logging.
      * @param {Function} exchangeApiCall - An async function that takes the ccxt exchange instance and performs the API call.
      * @param {string} description - A description of the action for logging purposes.
      * @param {Array<Error>} [allowedRetryExceptions] - Exceptions that should trigger a retry. Defaults to network/availability errors.
      * @param {boolean} [isOrderNotFoundOk=false] - If true, ccxt.OrderNotFound will be treated as a non-error (logged as WARN, returns null).
      * @returns {Promise<object | null>} - The result from the exchange API (e.g., order object) or null on failure or if OrderNotFound is ok.
      */
     async _executeExchangeRequest(
         exchangeApiCall,
         description,
         allowedRetryExceptions = [ // Default retryable errors for order operations
             ccxt.NetworkError,
             ccxt.RequestTimeout,
             ccxt.ExchangeNotAvailable,
             ccxt.DDoSProtection,
         ],
         isOrderNotFoundOk = false
     ) {
         // Wrap the provided function to pass the exchange instance
         const wrappedFunc = async () => {
             if (!this.exchangeMgr || !this.exchangeMgr.exchange) {
                  throw new Error("Exchange manager or CCXT exchange object is not initialized.");
             }
             return await exchangeApiCall(this.exchangeMgr.exchange); // Pass the exchange instance to the function
         };

         try {
             logger.info(`Attempting exchange request: ${c.yellow(description)}`);
             // Use the retryOnException utility
             const result = await retryOnException(
                 wrappedFunc,
                 this.config.max_retries,
                 this.config.retry_delay,
                 allowedRetryExceptions,
                 description // Use description as function name for retry logging
             );

             // --- Success Handling & Logging ---
             const isPlaceOrder = description.toLowerCase().includes('place') || description.toLowerCase().includes('create');
             const isCancelOrder = description.toLowerCase().includes('cancel');
             const isFetchOrder = description.toLowerCase().includes('fetch');

             if (isPlaceOrder && result && typeof result === 'object' && result.id) {
                 // Log detailed info for successful order placements
                 let logMsg = `${c.green("Order placement request successful:")} ${description} -> ID=${c.bold(result.id)}, Status=${result.status || 'N/A'}, Symbol=${result.symbol || 'N/A'}, Side=${result.side || 'N/A'}, Type=${result.type || 'N/A'}, Amount=${result.amount || 'N/A'}`;
                 if (result.price) logMsg += `, Price=${result.price}`;
                 if (result.average) logMsg += `, AvgFillPrice=${result.average}`;
                 if (result.stopPrice) logMsg += `, StopPrice=${result.stopPrice}`;
                 if (result.cost) logMsg += `, Cost=${result.cost}`;
                 logger.info(logMsg);
                 logger.debug("Full order response info:", result.info || result);
                 return result;
             } else if (isCancelOrder || isFetchOrder) {
                 logger.info(c.green(`Exchange request '${description}' completed successfully.`));
                 logger.debug(`Response for '${description}':`, result);
                 return result; // Return result (might be order object for fetch, confirmation for cancel)
             } else if (isPlaceOrder) {
                 // Handle cases where placement succeeded according to API but response lacks ID or expected structure
                 logger.error(c.red(`Order placement function for '${description}' completed but returned unexpected/invalid data (missing ID?):`), result);
                 this.notifier.sendSms(`ALERT: ${this.config.symbol} Order '${description}' placed but response invalid. Check exchange!`, this.config);
                 return null; // Treat as failure in our logic
             } else {
                 // Generic success for other types of operations
                  logger.info(c.green(`Exchange request '${description}' completed successfully.`));
                  logger.debug(`Response for '${description}':`, result);
                  return result;
             }

         } catch (e) {
             // --- Error Handling AFTER Retries or for Non-Retryable Errors ---
             logger.error(c.red(`Exchange request '${description}' FAILED: ${e.constructor.name} - ${e.message}`));

             // Handle specific CCXT errors with tailored logging and notifications
             if (e instanceof ccxt.InsufficientFunds) {
                  logger.error(c.red(`Reason: Insufficient funds. Check account balance and order size/cost.`));
                  this.notifier.sendSms(`ALERT: ${this.config.symbol} Order '${description.substring(0, 50)}...' failed: Insufficient Funds`, this.config);
             } else if (e instanceof ccxt.InvalidOrder) {
                  logger.error(c.red(`Reason: Invalid order parameters. Check amount, price, symbol limits, precision, etc.`), e);
                  this.notifier.sendSms(`ERROR: ${this.config.symbol} Order '${description.substring(0, 50)}...' failed (Invalid Params): ${e.message.substring(0, 80)}`, this.config);
             } else if (e instanceof ccxt.OrderNotFound) {
                  // If OrderNotFound is expected (e.g., cancelling already filled/cancelled order, or fetching a non-existent one)
                  if (isOrderNotFoundOk) {
                       logger.warn(c.yellow(`Reason: Order not found on exchange (Treated as OK for '${description}').`));
                       return null; // Return null as per function contract for "Ok not found"
                  } else {
                       logger.error(c.red(`Reason: Order not found on exchange (Unexpected for '${description}').`), e);
                       // Send notification only if it was unexpected
                       this.notifier.sendSms(`ERROR: ${this.config.symbol} Order not found unexpectedly during '${description.substring(0, 50)}...'.`, this.config);
                  }
             } else if (e instanceof ccxt.AuthenticationError) {
                  logger.error(c.red(`Reason: Authentication error! Check API Key, Secret, and Permissions. Bot may need restart.`), e);
                  this.notifier.sendSms("CRITICAL: Bot Authentication Error! Trading halted. Check API keys/permissions.", this.config);
                  // Consider signaling bot stop? throw e; // Re-throw to potentially stop the bot loop
             } else if (e instanceof ccxt.PermissionDenied) {
                 logger.error(c.red(`Reason: Permission denied. Check API key permissions for trading/withdrawal etc.`), e);
                 this.notifier.sendSms(`ERROR: ${this.config.symbol} Request '${description.substring(0, 50)}...' failed: Permission Denied. Check API permissions.`, this.config);
             } else if (e instanceof ccxt.ExchangeError) {
                  // Catch other specific exchange errors not explicitly handled above
                  logger.error(c.red(`Reason: A specific exchange error occurred. See details below.`), e);
                  this.notifier.sendSms(`ERROR: ${this.config.symbol} Request '${description.substring(0, 50)}...' failed (Exchange Error): ${e.message.substring(0, 80)}`, this.config);
             } else if (allowedRetryExceptions.some(excType => e instanceof excType)) {
                 // If a retryable error gets through (e.g., max retries exceeded)
                 logger.error(c.red(`Reason: Request failed after maximum retries due to persistent ${e.constructor.name}.`));
                 this.notifier.sendSms(`WARN: ${this.config.symbol} Request '${description.substring(0, 50)}...' failed after retries (${e.constructor.name}).`, this.config);
             } else {
                  // Catch-all for unexpected errors (programming errors, etc.)
                  logger.error(c.red(`Reason: An unexpected error occurred during the request. See details below.`), e.stack);
                  this.notifier.sendSms(`CRITICAL: ${this.config.symbol} Unexpected error during request '${description.substring(0, 50)}...': ${e.constructor.name}. Check logs.`, this.config);
             }
             return null; // Indicate failure to the caller
         }
     }


    /**
     * Places a market order with optional Stop Loss and Take Profit.
     * Uses Bybit V5 parameters for attaching SL/TP directly to the market order if possible.
     * @param {Side} side - 'buy' or 'sell'.
     * @param {number} amount - The amount/quantity to trade in base currency.
     * @param {number} priceForSignals - The price used for signal generation (e.g., latest close), needed for SL/TP validation.
     * @param {number | null} [slPrice=null] - Stop loss trigger price.
     * @param {number | null} [tpPrice=null] - Take profit trigger price.
     * @returns {Promise<object | null>} - The resulting order object from CCXT or null on failure.
     */
    async placeMarketOrder(side, amount, priceForSignals, slPrice = null, tpPrice = null) {
        const symbol = this.config.symbol;
        const exchange = this.exchangeMgr.exchange;

        // Pre-checks
        if (!exchange) { logger.error(c.red("Cannot place order: Exchange not initialized.")); return null; }
        if (amount <= 0) { logger.error(c.red(`Cannot place ${side} order: Amount must be positive, got ${amount}.`)); return null; }
        if (![Side.BUY, Side.SELL].includes(side)) { logger.error(c.red(`Invalid side specified: ${side}`)); return null; }
        if (isNaN(priceForSignals) || priceForSignals <= 0) {
            logger.error(c.red(`Invalid priceForSignals (${priceForSignals}) provided for market order placement and validation.`));
            return null;
        }

        try {
            // --- Prepare Order Parameters ---
            const market = exchange.market(symbol);
            if (!market) throw new Error(`Market data for ${symbol} not loaded.`);

            const amountStr = exchange.amountToPrecision(symbol, amount);
            const slPriceStr = slPrice !== null ? exchange.priceToPrecision(symbol, slPrice) : null;
            const tpPriceStr = tpPrice !== null ? exchange.priceToPrecision(symbol, tpPrice) : null;

            const amountPrecise = parseFloat(amountStr);
            const slPricePrecise = slPriceStr !== null ? parseFloat(slPriceStr) : null;
            const tpPricePrecise = tpPriceStr !== null ? parseFloat(tpPriceStr) : null;

            // Validate amount against market limits (minimum amount)
            const minAmount = market?.limits?.amount?.min;
            const sizeTolerance = Math.max(1e-9, (minAmount ?? 1e-9) / 100); // Use 1/100th of min amount or 1e-9
            if (amountPrecise < sizeTolerance) {
                 logger.error(c.red(`Order amount ${amount} (${amountPrecise} after precision) is zero or negligible (tolerance ${sizeTolerance}). Cannot place order.`));
                 return null;
            }
            if (minAmount !== undefined && amountPrecise < minAmount) {
                 logger.error(c.red(`Order amount ${amount} (${amountPrecise} after precision) is below the minimum required (${minAmount}) for ${symbol}.`));
                 this.notifier.sendSms(`ERROR: ${symbol} Order failed. Amount ${amountPrecise} below minimum ${minAmount}.`, this.config);
                 return null;
            }

            // --- Bybit V5 Specific Parameters for Market Order with SL/TP ---
            const params = {
                'timeInForce': this.config.time_in_force, // Check Bybit docs for market orders with SL/TP
                'reduceOnly': false, // This is an entry order
                'positionIdx': 0, // 0 for One-Way mode. 1 for Buy Hedge, 2 for Sell Hedge. Assume One-Way.
            };

            // Add category if needed (based on exchange type)
            if (exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                 params.category = 'linear';
            }

            // Add SL/TP parameters if provided and validate them against the signal price
            if (slPriceStr) {
                 if ((side === Side.BUY && slPricePrecise >= priceForSignals) || (side === Side.SELL && slPricePrecise <= priceForSignals)) {
                      logger.error(c.red(`Invalid SL price ${slPriceStr} for ${side} order relative to current price ${priceForSignals}. SL would trigger immediately.`));
                      this.notifier.sendSms(`ERROR: ${symbol} Order failed. Invalid SL price ${slPriceStr} vs current ${priceForSignals}.`, this.config);
                      return null;
                 }
                params.stopLoss = slPriceStr;
                params.slTriggerBy = this.config.order_trigger_price_type;
            }
            if (tpPriceStr) {
                if ((side === Side.BUY && tpPricePrecise <= priceForSignals) || (side === Side.SELL && tpPricePrecise >= priceForSignals)) {
                     logger.error(c.red(`Invalid TP price ${tpPriceStr} for ${side} order relative to current price ${priceForSignals}. TP would trigger immediately.`));
                     this.notifier.sendSms(`ERROR: ${symbol} Order failed. Invalid TP price ${tpPriceStr} vs current ${priceForSignals}.`, this.config);
                     return null;
                }
                params.takeProfit = tpPriceStr;
                params.tpTriggerBy = this.config.order_trigger_price_type;
            }
             // If SL or TP is set, Bybit V5 might need tpslMode
             if (slPriceStr || tpPriceStr) {
                  params.tpslMode = "Full"; // Apply to the entire position triggered by this order. Use "Partial" if only for this order's size.
             }

            const orderDescription = `Place Market ${side.toUpperCase()} ${amountStr} ${symbol} | SL: ${slPriceStr || 'N/A'} | TP: ${tpPriceStr || 'N/A'}`;
            logger.info(`Prepared Order: ${orderDescription}`);
            logger.debug(`Order Params: ${JSON.stringify(params)}`);

            // --- Dry Run Simulation ---
            if (this.config.dry_run) {
                logger.info(c.magenta(`DRY RUN: Simulating ${orderDescription}`));
                // Estimate fill price (use the provided signal price)
                const simulatedAvgPrice = priceForSignals;
                if (!simulatedAvgPrice || simulatedAvgPrice <= 0) {
                     logger.error(c.red("DRY RUN Error: Invalid priceForSignals for simulation. Aborting placement."));
                     return null;
                }

                // Create a simulated order object mimicking CCXT structure
                const simulatedOrder = {
                    id: `dry_${Date.now()}`,
                    clientOrderId: `dry_${Date.now()}_cli`,
                    symbol: symbol, side: side, type: "market", amount: amountPrecise,
                    filled: amountPrecise, // Assume market order fills completely
                    price: null, // Market orders don't have a specified price
                    average: simulatedAvgPrice, // Simulated fill price
                    cost: amountPrecise * simulatedAvgPrice, // Estimated cost
                    status: "closed", // Assume filled immediately
                    timestamp: Date.now(),
                    datetime: new Date().toISOString(),
                    fee: { currency: this.config.currency, cost: (amountPrecise * simulatedAvgPrice * 0.0006), rate: 0.0006 }, // Simulate fee (e.g., 0.06%)
                    info: { simulated: true, sl: slPriceStr, tp: tpPriceStr, params: params }
                };
                logger.info(c.magenta(`DRY RUN: Market order ${simulatedOrder.id} simulated as filled at ~${simulatedAvgPrice.toFixed(4)}.`));
                // In dry run, clear any previously tracked SL state as a new position is 'opened'
                if (this.active_sl_order_id) {
                     logger.debug("DRY RUN: Clearing previously tracked active SL order ID due to new simulated entry.");
                     this.active_sl_order_id = null;
                     await this._saveState();
                }
                this.notifier.sendSms(`DRY RUN: Placed ${side.toUpperCase()} ${amountStr} ${symbol}`, this.config);
                return simulatedOrder;
            }

            // --- Live Order Placement ---
            const placeFunc = async (exch) => {
                 // Using generic createOrder is often more reliable for passing complex params like SL/TP
                 return await exch.createOrder(symbol, 'market', side, amountPrecise, undefined, params);
            };

            // Execute using the helper
            const order = await this._executeExchangeRequest(
                placeFunc,
                orderDescription,
                // Retry standard network/availability errors, but maybe not ExchangeError for placements
                [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection]
            );

            // --- Post-Placement Handling ---
            if (order && order.id) {
                logger.info(c.green(`Market order ${order.id} placed successfully.`));
                // If order placed successfully, assume any previous MANUALLY tracked SL (from TSL) is now irrelevant.
                // Clear the tracked SL ID and save state.
                // Note: If SL/TP was attached to the market order, Bybit handles it server-side. We don't track these.
                if (this.active_sl_order_id) {
                     logger.info(`Clearing tracked active SL order ID ${this.active_sl_order_id} as new market order ${order.id} was placed (server-side SL/TP may apply).`);
                     this.active_sl_order_id = null;
                     await this._saveState();
                }
                this.notifier.sendSms(`${symbol}: ${side.toUpperCase()} ${amountStr} order placed. ID: ${order.id || 'N/A'}`, this.config);
                return order;
            } else {
                // Failure logged by _executeExchangeRequest
                logger.error(c.red(`Failed to place market order ${orderDescription}.`));
                // Notification potentially sent by helper
                return null;
            }

        } catch (e) {
             // Catch errors during parameter preparation or initial checks (before API call)
             logger.error(c.red(`Error preparing or placing market order for ${symbol}: ${e.message}`), e.stack);
             this.notifier.sendSms(`ERROR: Failed to prepare ${side} order for ${symbol}: ${e.message.substring(0,100)}`, this.config);
             return null;
        }
    }

    /**
     * Closes the current open position using a market order.
     * Crucially, attempts to cancel any tracked active SL order *before* closing.
     * @param {PositionSide} positionSide - The side of the position to close ('long' or 'short').
     * @param {number} amount - The amount/size of the position to close.
     * @returns {Promise<object | null>} - The closing order object from CCXT or null on failure.
     */
    async closePosition(positionSide, amount) {
        // Validate inputs
        if (positionSide === PositionSide.NONE || amount <= 0) {
            logger.warn(c.yellow(`Attempted to close position with invalid side (${positionSide}) or amount (${amount}). Skipping.`));
            return null;
        }

        const symbol = this.config.symbol;
        const exchange = this.exchangeMgr.exchange;
        if (!exchange) { logger.error(c.red("Cannot close position: Exchange not initialized.")); return null; }

        // Determine the side needed to close the position
        const closeSide = (positionSide === PositionSide.LONG) ? Side.SELL : Side.BUY;
        const orderDescription = `Close ${positionSide.toUpperCase()} (Market ${closeSide.toUpperCase()} ${amount.toFixed(8)} ${symbol})`;
        logger.info(c.bold(`Attempting to close position: ${orderDescription}`));

        // --- CRITICAL STEP: Cancel Active SL Order First ---
        // Ensure any manually tracked SL order (from TSL logic) is cancelled before sending the closing order.
        const cancelReason = `Closing ${positionSide} position`;
        const cancelSuccess = await this._cancelActiveSlOrder(cancelReason); // This handles state saving if successful

        if (!cancelSuccess) {
             // If SL cancellation failed, abort closing the position to avoid potential double orders or unexpected state.
             logger.error(c.red(`CRITICAL: Failed to cancel active SL order ${this.active_sl_order_id || 'N/A'} before closing ${positionSide} position. ABORTING CLOSE ORDER for safety. Manual intervention may be required.`));
             this.notifier.sendSms(`CRITICAL: CLOSE ABORTED for ${symbol}. Failed to cancel SL ${this.active_sl_order_id || 'N/A'}. Check Exchange!`, this.config);
             return null; // Abort the close operation
        }
        // If cancellation was successful (or no SL was tracked), active_sl_order_id is now null and state saved.

        try {
             // --- Prepare Close Order Parameters ---
             const market = exchange.market(symbol);
             if (!market) throw new Error(`Market data for ${symbol} not loaded.`);
             const amountStr = exchange.amountToPrecision(symbol, amount);
             const amountPrecise = parseFloat(amountStr);

             // Validate closing amount (redundant check, but safe)
             const minAmount = market?.limits?.amount?.min;
             const sizeTolerance = Math.max(1e-9, (minAmount ?? 1e-9) / 100);
             if (amountPrecise < sizeTolerance) {
                  logger.error(c.red(`Close order amount ${amount} became invalid (${amountPrecise} < tolerance ${sizeTolerance}) after precision. Cannot close.`));
                  this.notifier.sendSms(`CRITICAL: ${symbol} SL cancelled but CLOSE FAILED (amount ${amountPrecise} invalid). Manual check required!`, this.config);
                  return null;
             }
             if (minAmount !== undefined && amountPrecise < minAmount) {
                 logger.error(c.red(`Close order amount ${amountPrecise} is below minimum ${minAmount}. Cannot close.`));
                  this.notifier.sendSms(`CRITICAL: ${symbol} SL cancelled but CLOSE FAILED (amount ${amountPrecise} < min ${minAmount}). Manual check required!`, this.config);
                  return null;
             }

            // --- Dry Run Simulation ---
            if (this.config.dry_run) {
                logger.info(c.magenta(`DRY RUN: Simulating ${orderDescription}`));
                // Fetch current price for simulation (use cache)
                const currentPrice = await this.exchangeMgr.getCurrentPrice();
                 const simulatedAvgPrice = currentPrice; // Use ticker price for close simulation
                 if (!simulatedAvgPrice || simulatedAvgPrice <= 0) {
                      logger.error(c.red("DRY RUN Error: Could not get valid price for close simulation. Aborting placement."));
                      return null;
                 }
                const simulatedOrder = {
                    id: `dry_close_${Date.now()}`, symbol: symbol, side: closeSide, type: "market",
                    amount: amountPrecise, filled: amountPrecise, price: null, average: simulatedAvgPrice,
                    cost: amountPrecise * simulatedAvgPrice, status: "closed", reduceOnly: true,
                    timestamp: Date.now(), datetime: new Date().toISOString(),
                    info: { simulated: true, reduceOnly: true }
                };
                // SL ID already cleared by _cancelActiveSlOrder simulation
                logger.info(c.magenta(`DRY RUN: Close order ${simulatedOrder.id} simulated as filled.`));
                this.notifier.sendSms(`DRY RUN: Closed ${positionSide} ${amountStr} ${symbol}`, this.config);
                return simulatedOrder;
            }

            // --- Live Position Closing ---
            // Parameters for a closing market order
            const params = {
                'reduceOnly': true, // Ensure this order only reduces the position
                 'positionIdx': 0 // Assume One-Way mode
            };
            if (exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                 params.category = 'linear';
            }

            // Define the function for the execution helper
            const closeFunc = async (exch) => {
                 // Use generic createOrder for reliability with reduceOnly param
                  return await exch.createOrder(symbol, 'market', closeSide, amountPrecise, undefined, params);
            };

            // Execute using the helper
            const order = await this._executeExchangeRequest(
                closeFunc,
                orderDescription,
                // Retry standard network errors for close orders
                [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection]
            );

            // --- Post-Close Handling ---
            if (order && order.id) {
                logger.info(c.green(`Position close order request successful. Order ID: ${order.id}`));
                // State (active_sl_order_id) was already updated by _cancelActiveSlOrder
                this.notifier.sendSms(`${symbol}: Close order placed for ${positionSide} position.`, this.config);
                return order;
            } else {
                // Failure logged by _executeExchangeRequest
                logger.error(c.red(`CRITICAL: Failed to place position close order ${orderDescription}. Manual intervention REQUIRED.`));
                this.notifier.sendSms(`CRITICAL: FAILED to CLOSE ${positionSide} position for ${symbol}! Check Exchange Manually!`, this.config);
                return null;
            }

        } catch (e) {
             // Catch errors during close order preparation (after SL cancel)
             logger.error(c.red(`Error during close position process (after SL cancel): ${e.message}`), e.stack);
             this.notifier.sendSms(`CRITICAL: Error closing ${symbol} position AFTER SL cancel: ${e.message.substring(0,100)}. Manual check needed!`, this.config);
             return null;
        }
    }


    /**
     * Cancels the currently tracked active stop-loss order, if one exists.
     * Updates the internal state (active_sl_order_id) and saves it.
     * @param {string} reason - Reason for cancellation (for logging).
     * @returns {Promise<boolean>} - True if cancellation succeeded or no SL was active/found, False if cancellation failed.
     */
    async _cancelActiveSlOrder(reason = "Unknown reason") {
        // Check if we are actually tracking an SL order ID
        if (!this.active_sl_order_id) {
            logger.debug("No active SL order ID tracked, nothing to cancel.");
            return true; // Considered success as there's nothing to do
        }

        const slOrderIdToCancel = this.active_sl_order_id; // Store locally in case state changes during async ops
        logger.info(`Attempting to cancel active SL order ${c.bold(slOrderIdToCancel)} due to: ${c.dim(reason)}`);

        // --- Dry Run Simulation ---
        if (this.config.dry_run) {
             logger.info(c.magenta(`DRY RUN: Simulating cancellation of SL order ${slOrderIdToCancel}.`));
             this.active_sl_order_id = null; // Clear tracked ID
             await this._saveState(); // Save the cleared state
             return true; // Simulate success
        }

        // --- Live Cancellation ---
        // Define the cancellation function for the helper
        const cancelFunc = async (exch) => {
             // Bybit V5 might need specific params for cancelling stop orders
             const params = {
                   // 'orderFilter': 'StopOrder', // Example V5 param if cancelling conditional orders
             };
             if (exch.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                 params.category = 'linear';
             }

             logger.debug(`Calling cancelOrder: ID=${slOrderIdToCancel}, Symbol=${this.config.symbol}, Params=${JSON.stringify(params)}`);
             // Arguments: id, symbol, params
             return await exch.cancelOrder(slOrderIdToCancel, this.config.symbol, params);
        };

        // Execute using the helper. Treat OrderNotFound as OK for cancellation.
        const cancelResult = await this._executeExchangeRequest(
            cancelFunc,
            `Cancel SL Order ${slOrderIdToCancel}`,
            // Retry standard network errors
            [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection],
            true // isOrderNotFoundOk = true
        );

        // --- Handle Cancellation Result ---
        // If the helper returned non-null, the cancellation API call was successful.
        // If the helper returned null, it means either OrderNotFound occurred (which is OK here) or a real error occurred.
        if (cancelResult !== null) {
             // API call succeeded (didn't throw critical error, wasn't OrderNotFound)
             logger.info(c.green(`Successfully requested cancellation for SL order ${slOrderIdToCancel}.`));
             this.active_sl_order_id = null; // Clear the tracked ID
             await this._saveState(); // Persist the change
             return true; // Indicate success
        } else {
            // Helper returned null. This means either:
            // 1. OrderNotFound occurred (logged as WARN by helper, treated as OK by us).
            // 2. Another error occurred after retries (logged as ERROR by helper).
            // We need to confirm the order is actually gone before clearing our state.

            logger.warn(c.yellow(`Cancel request for SL ${slOrderIdToCancel} returned null (might be OrderNotFound or failure). Re-checking order status...`));
            try {
                 // Define fetch function for helper
                 const fetchFunc = async (exch) => {
                      const params = {};
                       if (exch.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                           params.category = 'linear';
                       }
                      return await exch.fetchOrder(slOrderIdToCancel, this.config.symbol, params);
                 };
                 // Use helper to fetch, treat OrderNotFound as OK here too
                 const orderStatus = await this._executeExchangeRequest(
                     fetchFunc,
                     `Fetch SL Order ${slOrderIdToCancel} Status After Cancel Attempt`,
                     undefined, // Default retries
                     true // isOrderNotFoundOk = true
                 );

                 if (orderStatus && ['open', 'untriggered'].includes(orderStatus.status?.toLowerCase())) {
                     // Order still exists and is open/pending trigger - Cancellation FAILED
                     logger.error(c.red(`Cancellation of SL order ${slOrderIdToCancel} FAILED. Order status is still '${orderStatus.status}'.`));
                     // Do NOT clear the ID, cancellation failed.
                     return false; // Indicate failure
                 } else {
                     // Order is closed, canceled, rejected, expired, or not found (fetch returned null due to OrderNotFound)
                     logger.info(c.green(`SL order ${slOrderIdToCancel} confirmed closed/canceled/gone after cancellation attempt (Status: ${orderStatus?.status || 'Not Found'}). Clearing state.`));
                     this.active_sl_order_id = null; // Clear the ID
                     await this._saveState();
                     return true; // Indicate effective success
                 }
            } catch (fetchError) {
                 // This catch block is less likely if _executeExchangeRequest handles errors properly
                 // But capture unexpected errors during the fetch process itself
                 logger.error(c.red(`Unexpected error re-checking status of SL order ${slOrderIdToCancel}: ${fetchError.message}. Assuming cancellation failed.`), fetchError.stack);
                 // Do NOT clear the ID, state is uncertain.
                 return false; // Indicate failure
            }
        }
    }

    /**
     * Calculates the appropriate position size based on equity, risk percentage, and ATR-based stop loss distance.
     * Adjusts for exchange minimum/maximum order size and precision.
     * @param {number} entryPrice - The estimated entry price for the trade.
     * @param {number} atr - The current Average True Range value.
     * @param {number | null} equity - The current account equity in the quote currency (can be null if fetch failed).
     * @returns {Promise<number | null>} - The calculated position size in base currency, or null if calculation fails.
     */
    async calculatePositionSize(entryPrice, atr, equity) {
        const exchange = this.exchangeMgr.exchange;
        if (!exchange) { logger.error(c.red("Cannot calculate position size: Exchange not initialized.")); return null; }

        // --- Input Validation ---
        if (atr === null || isNaN(atr) || atr <= 0) { // ATR validation in indicator ensures > 0 if not null
            logger.warn(c.yellow(`Invalid ATR value (${atr}) provided for position sizing.`));
            return null;
        }
        if (equity === null) {
             logger.error(c.red("Equity is null (failed fetch?). Cannot calculate position size.")); return null;
        }
        if (isNaN(equity)) {
             logger.error(c.red("Equity is NaN. Cannot calculate position size.")); return null;
        }
        if (equity <= 0 && !this.config.dry_run) { // Only block if live and equity <= 0
            logger.warn(c.yellow(`Equity (${equity.toFixed(2)}) is zero or negative. Cannot calculate position size for live trading.`));
            return null;
        } else if (equity <= 0 && this.config.dry_run) {
            logger.warn(c.magenta(`DRY RUN: Equity (${equity.toFixed(2)}) is zero or negative. Proceeding with calculation based on risk amount (which will be <= 0).`));
        }
        if (isNaN(entryPrice) || entryPrice <= 0) {
             logger.warn(c.yellow(`Invalid entry price (${entryPrice?.toFixed(4)}) provided for position sizing.`));
             return null;
        }
        // Validate risk parameters from config
        if (isNaN(this.config.risk_per_trade) || !(this.config.risk_per_trade > 0)) {
             logger.error(c.red(`Invalid risk_per_trade config (${this.config.risk_per_trade}). Must be positive.`));
             return null;
        }
        if (isNaN(this.config.sl_atr_mult) || this.config.sl_atr_mult <= 0) {
             logger.error(c.red(`Invalid sl_atr_mult config (${this.config.sl_atr_mult}). Must be positive.`));
             return null;
        }

        try {
            // --- Risk Calculation ---
            const riskAmountQuote = equity * this.config.risk_per_trade;
            const slDistancePrice = atr * this.config.sl_atr_mult;

            // Ensure stop distance is meaningful
            if (slDistancePrice <= 1e-12) { // Use tolerance
                logger.warn(c.yellow(`Calculated SL distance (${slDistancePrice.toFixed(8)}) is zero or negligible based on ATR (${atr}) and multiplier (${this.config.sl_atr_mult}). Cannot size position.`));
                return null;
            }
             // Ensure risk amount is positive (can be zero/negative in dry run if equity is <= 0)
            if (riskAmountQuote <= 0 && !this.config.dry_run) {
                logger.warn(c.yellow(`Calculated risk amount (${riskAmountQuote.toFixed(2)}) is zero or negative. Cannot size position for live trading.`));
                return null;
            }

            // --- Initial Size Calculation ---
            // Size = (Amount to Risk) / (Stop Distance per Unit)
            const initialPositionSizeBase = riskAmountQuote / slDistancePrice;

            logger.debug(`Position Sizing Inputs: Equity=${equity.toFixed(2)}, Risk=${(this.config.risk_per_trade*100).toFixed(2)}%, Entry=${entryPrice.toFixed(4)}, ATR=${atr.toFixed(4)}, SL Mult=${this.config.sl_atr_mult}`);
            logger.debug(`Calculated Risk Amount: ${riskAmountQuote.toFixed(2)} ${this.config.currency}`);
            logger.debug(`Calculated SL Distance (Price): ${slDistancePrice.toFixed(8)}`);
            logger.debug(`Initial Calculated Size (Base): ${initialPositionSizeBase.toFixed(8)}`);

             // If initial size is zero or negative (can happen if riskAmount is zero/negative in dry run)
            if (initialPositionSizeBase <= 0) {
                logger.warn(c.yellow(`Initial calculated size (${initialPositionSizeBase.toFixed(8)}) is zero or negative. Cannot place trade.`));
                return null;
            }

            // --- Market Limits and Precision ---
            const market = exchange.market(this.config.symbol);
            if (!market) throw new Error(`Market data not found for ${this.config.symbol} during sizing.`);

            const limits = market.limits || {};
            const amountLimits = limits.amount || {}; // Limits on order size (base currency)
            const costLimits = limits.cost || {};     // Limits on order cost (quote currency: size * price)
            const precision = market.precision || {}; // Rules for formatting amount/price

            // Extract limit values, using undefined if not present
            const minAmount = amountLimits.min;
            const maxAmount = amountLimits.max;
            const minCost = costLimits.min;
            const maxCost = costLimits.max;

            // Define tolerance for checking near-zero size after adjustments
            const sizeTolerance = Math.max(1e-9, (minAmount ?? 1e-9) / 100);

            let adjustedSize = initialPositionSizeBase; // Start with the calculated size

            // --- Apply Cost Limits (Approximate Check) ---
            let currentCost = adjustedSize * entryPrice;
            logger.debug(`Estimated Cost (Before Amount Limits/Precision): ${currentCost.toFixed(2)}`);

            if (minCost !== undefined && currentCost < minCost) {
                // If cost is too low based on risk target, fail the trade.
                 logger.error(c.red(`Trade aborted: Calculated position size (${adjustedSize.toFixed(8)}) results in cost (${currentCost.toFixed(2)}) below minimum limit (${minCost}). Risk=${(this.config.risk_per_trade*100).toFixed(1)}% may be too low for this asset/price.`));
                 this.notifier.sendSms(`WARN: ${this.config.symbol} Trade skipped. Size needed for risk target below min cost limit (${minCost}).`, this.config);
                 return null;
            }
            if (maxCost !== undefined && currentCost > maxCost) {
                // If cost is too high, reduce size to meet maxCost.
                logger.warn(c.yellow(`Calculated cost (${currentCost.toFixed(2)}) exceeds maximum cost limit (${maxCost}). Reducing size.`));
                adjustedSize = maxCost / entryPrice;
                currentCost = adjustedSize * entryPrice; // Update cost
                logger.info(`Size reduced to ${adjustedSize.toFixed(8)} due to max cost limit. New Cost: ${currentCost.toFixed(2)}`);
            }

            // --- Apply Amount Limits ---
            logger.debug(`Size after Cost Limits: ${adjustedSize.toFixed(8)}`);
            if (minAmount !== undefined && adjustedSize < minAmount) {
                 // If size is below min amount after cost adjustments (or initially)
                 logger.error(c.red(`Calculated/Adjusted size ${adjustedSize.toFixed(8)} is below minimum amount limit (${minAmount}). Cannot place trade.`));
                 this.notifier.sendSms(`WARN: ${this.config.symbol} Trade skipped. Calculated size ${adjustedSize.toFixed(8)} below min amount limit ${minAmount}.`, this.config);
                 return null;
            }
            if (maxAmount !== undefined && adjustedSize > maxAmount) {
                 // If size exceeds max amount
                 logger.warn(c.yellow(`Calculated/Adjusted size ${adjustedSize.toFixed(8)} exceeds maximum amount limit (${maxAmount}). Clamping size to max amount.`));
                 adjustedSize = maxAmount;
            }

            // --- Apply Precision ---
            logger.debug(`Size before Precision Formatting: ${adjustedSize.toFixed(8)}`);
            const preciseSizeStr = exchange.amountToPrecision(this.config.symbol, adjustedSize);
            const finalSize = parseFloat(preciseSizeStr);
            logger.debug(`Size AFTER Precision Formatting: ${finalSize.toFixed(8)} (${preciseSizeStr})`);


            // --- Final Validation AFTER Precision ---
            // Ensure the final size after precision is still valid and meets limits
            if (finalSize < sizeTolerance) {
                 logger.error(c.red(`Final size ${finalSize.toFixed(8)} became negligible after applying precision rules.`));
                 return null;
            }
            if (minAmount !== undefined && finalSize < minAmount) {
                 // This can happen if precision rounding pushes the value below the minimum
                 logger.error(c.red(`Final size ${finalSize.toFixed(8)} (after precision) is below minimum amount limit (${minAmount}).`));
                 this.notifier.sendSms(`WARN: ${this.config.symbol} Trade skipped. Final size ${finalSize.toFixed(8)} (post-precision) < min amount ${minAmount}.`, this.config);
                 return null;
            }
             // Re-check cost with the final precise size
            const finalCost = finalSize * entryPrice;
            logger.debug(`Final Estimated Cost: ${finalCost.toFixed(2)}`);
            if (minCost !== undefined && finalCost < minCost) {
                 // This might happen if rounding down due to precision makes cost too low
                 logger.error(c.red(`Final cost ${finalCost.toFixed(2)} (using precise size ${finalSize}) is below minimum cost limit (${minCost}).`));
                 this.notifier.sendSms(`WARN: ${this.config.symbol} Trade skipped. Final cost ${finalCost.toFixed(2)} (post-precision) < min cost ${minCost}.`, this.config);
                 return null;
            }
             // Max cost check (less likely to fail after precision if checked before, but good practice)
             if (maxCost !== undefined && finalCost > maxCost) {
                 logger.error(c.red(`Final cost ${finalCost.toFixed(2)} (using precise size ${finalSize}) exceeds maximum cost limit (${maxCost}). Logic error?`));
                 return null; // Should have been caught earlier
             }

            // --- Success ---
            logger.info(c.green(`Calculated final position size: ${c.bold(finalSize.toFixed(8))} ${market.base || ''}`));
            return finalSize;

        } catch (e) {
             // Catch errors during calculation or limit checking
             logger.error(c.red(`Error during position size calculation or adjustment: ${e.message}`), e.stack);
             return null;
        }
    }


    /**
     * Manages the Trailing Stop Loss (TSL).
     * Checks if the current SL needs adjustment based on price movement and ATR.
     * If conditions met, cancels the old SL (if any) and places a new one.
     * IMPORTANT: This manages SL orders placed *manually by this function*. It does not manage
     * SL orders attached directly to market/limit entries via exchange parameters (server-side SL/TP).
     * @param {PositionSide} positionSide - Current position side ('long', 'short').
     * @param {number} positionAmount - Current position size.
     * @param {number} entryPrice - Position entry price.
     * @param {number} currentPrice - Current market price (ticker recommended).
     * @param {number} currentAtr - Current ATR value.
     */
    async updateTrailingStop(positionSide, positionAmount, entryPrice, currentPrice, currentAtr) {
        const exchange = this.exchangeMgr.exchange;
        if (!exchange) { logger.error(c.red("Cannot update TSL: Exchange not initialized.")); return; }

        // --- Pre-conditions Check ---
        if (positionSide === PositionSide.NONE || positionAmount <= 0) {
            // If no position, ensure any lingering tracked SL ID is cleared
            if (this.active_sl_order_id) {
                logger.warn(c.yellow(`No active position, but an SL order ID (${this.active_sl_order_id}) is still tracked. Attempting to cancel potentially orphaned SL.`));
                await this._cancelActiveSlOrder("Orphaned SL cleanup - no active position");
            }
            return; // No position to trail stop for
        }
        // Validate necessary inputs for calculation
        if (currentAtr === null || isNaN(currentAtr) || currentAtr <= 0) { // ATR validation done in indicator func
             logger.warn(c.yellow(`Cannot update TSL: Invalid ATR value (${currentAtr}).`));
             return;
        }
        if (isNaN(this.config.trailing_stop_mult) || this.config.trailing_stop_mult <= 0) {
             logger.warn(c.yellow(`Cannot update TSL: Invalid TRAILING_STOP_MULT config (${this.config.trailing_stop_mult}).`));
             return;
        }
        if (isNaN(entryPrice) || entryPrice <= 0) {
             logger.warn(c.yellow(`Cannot update TSL: Invalid entry price (${entryPrice}).`));
             return;
        }
        if (isNaN(currentPrice) || currentPrice <= 0) {
             logger.warn(c.yellow(`Cannot update TSL: Invalid current price (${currentPrice}).`));
             return;
        }

        // --- Calculate Potential New SL Price based on TSL rules ---
        const trailingDistance = currentAtr * this.config.trailing_stop_mult;
        if (trailingDistance <= 1e-12) { // Use tolerance
             logger.warn(c.yellow(`Calculated trailing distance (${trailingDistance.toFixed(8)}) is negligible. Skipping TSL update.`));
             return;
        }

        let potentialNewSlPrice = null;
        let slOrderSide = null; // The side of the STOP order needed to close the position
        if (positionSide === PositionSide.LONG) {
            potentialNewSlPrice = currentPrice - trailingDistance;
            slOrderSide = Side.SELL;
        } else { // SHORT position
            potentialNewSlPrice = currentPrice + trailingDistance;
            slOrderSide = Side.BUY;
        }

        logger.debug(`TSL Calculation: PosSide=${positionSide}, CurPrice=${currentPrice.toFixed(4)}, ATR=${currentAtr.toFixed(4)}, ` +
                     `TSL_Mult=${this.config.trailing_stop_mult}, TrailDist=${trailingDistance.toFixed(4)}, PotentialNewSL=${potentialNewSlPrice.toFixed(4)}, Entry=${entryPrice.toFixed(4)}`);

        // Ensure potential SL price is valid (positive)
        if (potentialNewSlPrice <= 0) {
             logger.warn(c.yellow(`Calculated potential new SL price (${potentialNewSlPrice.toFixed(4)}) is zero or negative. Skipping TSL update.`));
             return;
        }

        // --- Get Current Active SL Order Price (if one is tracked) ---
        let currentActiveSlPrice = null; // The trigger price of the currently tracked SL order
        if (this.active_sl_order_id) {
             logger.debug(`Checking status of tracked active SL order: ${this.active_sl_order_id}`);
             try {
                  // Define the fetch function for the helper
                  const fetchSlFunc = async (exch) => {
                       const params = {};
                       if (exch.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                           params.category = 'linear';
                       }
                       return await exch.fetchOrder(this.active_sl_order_id, this.config.symbol, params);
                  }

                  // Fetch order status using the helper (handles retries, treats OrderNotFound as OK)
                   const slOrderInfo = await this._executeExchangeRequest(
                       fetchSlFunc,
                       `Fetch Tracked SL Order ${this.active_sl_order_id} Status`,
                       undefined, // Default retries
                       true // isOrderNotFoundOk = true
                   );

                  // Process the fetched order info
                  if (slOrderInfo && (slOrderInfo.status === 'open' || slOrderInfo.status === 'untriggered')) {
                       // Order exists and is active
                       if (slOrderInfo.stopPrice !== undefined && slOrderInfo.stopPrice !== null) {
                           const parsedPrice = parseFloat(slOrderInfo.stopPrice);
                           if (!isNaN(parsedPrice) && parsedPrice > 0) {
                               currentActiveSlPrice = parsedPrice;
                               logger.debug(`Found active open/untriggered SL order ${this.active_sl_order_id}. Current trigger price: ${currentActiveSlPrice.toFixed(4)}`);
                           } else {
                                logger.warn(c.yellow(`Could not parse valid stopPrice ('${slOrderInfo.stopPrice}') for active SL order ${this.active_sl_order_id}. Treating SL price as unknown.`));
                           }
                       } else {
                           logger.warn(c.yellow(`Active SL order ${this.active_sl_order_id} found (status: '${slOrderInfo.status}'), but missing 'stopPrice'. Treating SL price as unknown.`));
                       }
                  } else if (slOrderInfo) {
                       // Order found but not in an active state
                       logger.info(`Tracked SL order ${this.active_sl_order_id} found but status is '${slOrderInfo.status}'. Clearing tracked state.`);
                       this.active_sl_order_id = null;
                       await this._saveState(); // Update state file
                  } else {
                       // Helper returned null (OrderNotFound or error)
                       logger.warn(c.yellow(`Tracked SL order ${this.active_sl_order_id} could not be fetched or was not found. Clearing tracked state.`));
                       this.active_sl_order_id = null;
                       await this._saveState(); // Update state file
                  }
             } catch(e) {
                 // Catch unexpected errors during the fetch *process* itself (less likely with helper)
                 logger.error(c.red(`Unexpected error fetching SL order ${this.active_sl_order_id} status: ${e.message}. Skipping TSL update this cycle.`), e.stack);
                 return; // Skip TSL update if status is uncertain due to error
             }
        } else {
            logger.debug("No active SL order is currently tracked by the bot.");
        }

        // --- Logic: Decide if SL Needs Update ---
        let shouldUpdate = false;
        // Use a tolerance based on price precision
        const market = exchange.market(this.config.symbol);
        const priceTickSize = market?.precision?.price ?? 1e-8;
        const priceTolerance = priceTickSize * 2; // e.g., 2 ticks tolerance

        // Format potential new SL price to exchange precision for accurate comparison
        let potentialNewSlPriceFormatted = null;
        try {
            potentialNewSlPriceFormatted = parseFloat(exchange.priceToPrecision(this.config.symbol, potentialNewSlPrice));
            if (isNaN(potentialNewSlPriceFormatted) || potentialNewSlPriceFormatted <= 0) {
                 logger.warn(c.yellow(`Potential new SL price ${potentialNewSlPrice.toFixed(4)} became invalid after precision formatting (${potentialNewSlPriceFormatted}). Skipping TSL.`));
                 return;
            }
        } catch (formatError) {
             logger.error(c.red(`Error formatting potential SL price ${potentialNewSlPrice}: ${formatError.message}. Skipping TSL.`), formatError.stack);
             return;
        }


        if (currentActiveSlPrice === null) {
             // --- Case 1: No Active SL Tracked ---
             // Place initial TSL only if it's already profitable (better than entry price by tolerance)
             if (positionSide === PositionSide.LONG && (potentialNewSlPriceFormatted > entryPrice + priceTolerance)) {
                 logger.info(c.cyan(`Condition Met: No active SL tracked. Potential new TSL ${potentialNewSlPriceFormatted.toFixed(4)} is profitable ( > Entry ${entryPrice.toFixed(4)}). Placing initial TSL.`));
                 shouldUpdate = true;
             } else if (positionSide === PositionSide.SHORT && (potentialNewSlPriceFormatted < entryPrice - priceTolerance)) {
                 logger.info(c.cyan(`Condition Met: No active SL tracked. Potential new TSL ${potentialNewSlPriceFormatted.toFixed(4)} is profitable ( < Entry ${entryPrice.toFixed(4)}). Placing initial TSL.`));
                 shouldUpdate = true;
             } else {
                 logger.debug(`Condition NOT Met: No active SL tracked, potential new SL ${potentialNewSlPriceFormatted.toFixed(4)} not yet profitable vs entry ${entryPrice.toFixed(4)}. Waiting.`);
             }
        } else {
             // --- Case 2: Active SL Exists ---
             // Update only if the new potential SL is *strictly better* (further in profit by tolerance) than the current active SL.
             if (positionSide === PositionSide.LONG) {
                 // For LONG, new SL must be HIGHER than current SL
                 if (potentialNewSlPriceFormatted > currentActiveSlPrice + priceTolerance) {
                     logger.info(c.cyan(`Condition Met: Trailing SL for LONG. New SL ${potentialNewSlPriceFormatted.toFixed(4)} is better than Current SL ${currentActiveSlPrice.toFixed(4)}.`));
                     shouldUpdate = true;
                 } else {
                     logger.debug(`Condition NOT Met: Potential new TSL ${potentialNewSlPriceFormatted.toFixed(4)} for LONG is not better than current active SL ${currentActiveSlPrice.toFixed(4)} (Tolerance: ${priceTolerance.toFixed(8)}). Holding current SL.`);
                 }
             } else { // SHORT position
                 // For SHORT, new SL must be LOWER than current SL
                 if (potentialNewSlPriceFormatted < currentActiveSlPrice - priceTolerance) {
                     logger.info(c.cyan(`Condition Met: Trailing SL for SHORT. New SL ${potentialNewSlPriceFormatted.toFixed(4)} is better than Current SL ${currentActiveSlPrice.toFixed(4)}.`));
                     shouldUpdate = true;
                 } else {
                     logger.debug(`Condition NOT Met: Potential new TSL ${potentialNewSlPriceFormatted.toFixed(4)} for SHORT is not better than current active SL ${currentActiveSlPrice.toFixed(4)} (Tolerance: ${priceTolerance.toFixed(8)}). Holding current SL.`);
                 }
             }
        }

        // --- Execute Update if Needed ---
        if (shouldUpdate) {
            logger.info(c.bold(`ACTION: Updating Trailing Stop Loss for ${positionSide} position.`));
            try {
                // Format amount and the final new SL price string
                const newSlPriceStr = exchange.priceToPrecision(this.config.symbol, potentialNewSlPriceFormatted); // Use the already formatted price
                const amountStr = exchange.amountToPrecision(this.config.symbol, positionAmount);
                const amountPrecise = parseFloat(amountStr);

                // Final check on formatted values before placing order
                const minAmount = market?.limits?.amount?.min;
                const sizeTolerance = Math.max(1e-9, (minAmount ?? 1e-9) / 100);
                if (amountPrecise < sizeTolerance || parseFloat(newSlPriceStr) <= 0) {
                     logger.error(c.red(`TSL update aborted: Invalid amount (${amountPrecise}) or SL price (${newSlPriceStr}) after final precision formatting.`));
                     return;
                }

                 // --- Step 1: Cancel Existing SL Order (if one was tracked and active) ---
                 // Re-check if ID exists, as it might have been cleared if found inactive above
                 if (this.active_sl_order_id) {
                     logger.info(`Cancelling previous SL order ${this.active_sl_order_id} before placing new TSL at ${newSlPriceStr}.`);
                     const cancelOk = await this._cancelActiveSlOrder(`Trailing stop update to ${newSlPriceStr}`);
                     if (!cancelOk) {
                          // If cancellation failed, ABORT the TSL update to avoid potential issues.
                          logger.error(c.red("CRITICAL: Failed to cancel previous SL order. Aborting TSL update for safety. Position might have outdated or no SL. Manual check advised."));
                          this.notifier.sendSms(`CRITICAL: TSL update ABORTED for ${this.config.symbol}. Failed to cancel old SL ${this.active_sl_order_id}. Check position!`, this.config);
                          return; // Stop the update process
                     }
                     // If cancelOk is true, active_sl_order_id is now null and state saved.
                 } else {
                     logger.debug("No active SL order was tracked, proceeding directly to place new TSL.");
                 }


                 // --- Step 2: Place the New Trailing Stop Order (as Stop Market) ---
                 const stopPriceNum = parseFloat(newSlPriceStr);
                 if (isNaN(stopPriceNum)) throw new Error("Invalid stopPrice number for TSL order.");

                 logger.info(`Placing new TSL order: Side=${slOrderSide}, Amount=${amountStr}, TriggerPrice=${newSlPriceStr}`);

                 // Parameters for STOP MARKET order (Bybit V5)
                 const slParams = {
                     'triggerDirection': (positionSide === PositionSide.LONG) ? 2 : 1, // 1: Rise (for short SL), 2: Fall (for long SL)
                     'triggerBy': this.config.order_trigger_price_type,
                     'reduceOnly': true,
                     'closeOnTrigger': true, // Ensures it closes the position (Bybit specific)
                     'orderType': "Market", // Execute as Market after trigger (Bybit specific in params)
                     'basePrice': exchange.priceToPrecision(this.config.symbol, currentPrice), // Bybit V5 requires basePrice for stop orders
                     'positionIdx': 0, // Assume One-Way mode
                 };
                 if (exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) {
                     slParams.category = 'linear';
                 }
                 // Ensure basePrice is valid
                 const basePriceNum = parseFloat(slParams.basePrice);
                 if (isNaN(basePriceNum) || basePriceNum <= 0) {
                     logger.error(c.red(`Invalid basePrice (${slParams.basePrice} from currentPrice ${currentPrice}) for TSL params. Aborting TSL placement.`));
                     return;
                 }

                 const orderDescription = `Place Trailing Stop ${slOrderSide.toUpperCase()} ${amountStr} @ ${newSlPriceStr}`;
                 logger.debug(`TSL Order Params: ${JSON.stringify(slParams)}`);

                 // Define the function to place the stop order using createOrder
                 const placeTslFunc = async (exch) => {
                      // CCXT createOrder handles mapping 'Stop' type and params for Bybit V5
                      return await exch.createOrder(
                          this.config.symbol,
                          'Stop', // Use 'Stop' type for conditional orders that execute as market/limit
                          slOrderSide,
                          amountPrecise,
                          undefined, // Price is undefined for market execution after trigger
                          {
                              ...slParams, // Contains orderType: 'Market'
                              'stopPrice': stopPriceNum, // Pass the trigger price here
                          }
                      );
                 };

                 // Execute placement using the helper
                 const newSlOrder = await this._executeExchangeRequest(
                     placeTslFunc,
                     orderDescription,
                     // Retry standard network errors
                     [ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection]
                 );

                 // --- Step 3: Update State if Successful ---
                 if (newSlOrder && newSlOrder.id) {
                     // Successfully placed the new SL order
                     this.active_sl_order_id = newSlOrder.id; // Track the new order ID
                     await this._saveState(); // Persist the new ID
                     logger.info(c.green(`Trailing SL successfully updated/placed. New active SL Order ID: ${c.bold(this.active_sl_order_id)}, Trigger: ${newSlPriceStr}`));
                     this.notifier.sendSms(`${this.config.symbol} TSL updated: Trigger @ ${newSlPriceStr}`, this.config);
                 } else {
                     // Failed to place the new SL order (helper logged the error)
                     logger.error(c.red("CRITICAL: Failed to place the new trailing stop loss order after cancelling the old one (if any). Position may be UNPROTECTED."));
                     // Ensure state reflects no active SL if placement failed
                     this.active_sl_order_id = null;
                     await this._saveState();
                     this.notifier.sendSms(`CRITICAL: Failed to place new TSL for ${this.config.symbol} after update attempt! Position UNPROTECTED. Check exchange!`, this.config);
                 }

            } catch (e) {
                // Catch unexpected errors during the TSL update execution phase
                logger.error(c.red(`Unexpected error during TSL update execution: ${e.message}`), e.stack);
                 // Ensure state is cleared if update failed mid-way
                 if (shouldUpdate && !this.active_sl_order_id) {
                     logger.error(c.red("Clearing SL state due to error during TSL placement after previous SL cancel."));
                     this.active_sl_order_id = null; // Ensure state is cleared
                     await this._saveState();
                 }
                 this.notifier.sendSms(`ERROR: Unexpected error updating TSL for ${this.config.symbol}. Check logs.`, this.config);
            }
        } // end if shouldUpdate
    }
}


// --- Trading Bot Class ---
// Orchestrates the overall bot logic, manages cycles, and integrates other components.
class TradingBot {
    constructor() {
        // Initialize components to null, will be set up in async initialize method
        this.config = null;
        this.notifier = null;
        this.exchangeMgr = null;
        this.orderMgr = null;
        this.indicators = Indicators; // Access static indicator methods via this.indicators
        this.last_candle_ts = null;   // Timestamp (ms) of the last fully processed candle
        this.cycle_count = 0;         // Counter for trading cycles
        this.start_time = Date.now() / 1000; // Bot start time in seconds
        this._stop_requested = false; // Flag to signal graceful shutdown
        this._isRunning = false;      // Flag indicating if the main loop is active
    }

    // Asynchronous initialization sequence
    async initialize() {
        logger.info(c.blue("----- Initializing Trading Bot -----"));
        try {
            // 1. Load Configuration (includes initial logging setup call)
            this.config = new Config();
            // Async logging setup (file handle opening) continues in background

            // 2. Initialize Services and Managers
            this.notifier = new NotificationService();
            this.exchangeMgr = new ExchangeManager(this.config);
            await this.exchangeMgr.initialize(); // Connect to exchange, load markets, set leverage

            this.orderMgr = new OrderManager(this.exchangeMgr, this.config, this.notifier);
            await this.orderMgr.initialize(); // Load persistent state (SL order ID)

            // Initialization successful
            logger.info(c.green(c.bold("Trading Bot initialization complete.")));
            logger.info(`Trading Pair: ${c.bold(this.config.symbol)}, Timeframe: ${c.bold(this.config.timeframe)}`);
            if (this.config.dry_run) {
                 logger.info(c.magenta(c.bold("Dry Run Mode: ENABLED. No real trades will be executed.")));
            } else {
                 logger.warn(c.red(c.bold("--- LIVE TRADING IS ACTIVE ---")));
                 logger.warn(c.yellow("Ensure configuration, risk parameters, and strategy logic are thoroughly tested and understood."));
            }
        } catch (e) {
             // Catch critical errors during initialization
             logger.error(c.red(`CRITICAL ERROR during bot initialization: ${e.message}`), e.stack);
             // Attempt to send an emergency notification if possible
             this._attemptEmergencyNotification(`CRITICAL: Bot startup FAILED: ${e.message.substring(0, 100)}`);
             throw e; // Re-throw the error to prevent the bot from starting
        }
    }

    // Helper to send emergency notifications during startup/shutdown failures
    _attemptEmergencyNotification(message) {
        try {
            // Use initialized components if available, otherwise try creating defaults
            const cfg = this.config || DEFAULTS; // Use defaults if config failed
            const ntf = this.notifier || new NotificationService(); // Minimal fallback notifier
            // Re-check conditions needed for SMS sending
            const canSendSms = cfg.sms_enabled &&
                               cfg.sms_recipient_number &&
                               (cfg.termux_sms_available !== undefined ? cfg.termux_sms_available : false); // Check flag if config exists

            if (canSendSms) {
                 ntf.sendSms(message, cfg);
            } else {
                 // Log to console if SMS isn't possible
                 logger.error(c.red(`Emergency Notification (SMS Disabled/Unavailable/Config Error): ${message}`));
            }
        } catch (notifyErr) {
            // Catch errors within the notification attempt itself
            logger.error(c.red(`Failed to send critical failure notification: ${notifyErr.message}`), notifyErr.stack);
        }
    }

    // Calculates the required number of candles based on indicator periods
    _getRequiredOhlcvLimit() {
        // Determine max period needed by indicators + buffer
        const stNeed = Math.max(this.config.short_st_period, this.config.long_st_period) + 3; // Supertrend needs lookback + history for approx
        const volNeed = this.config.volume_long_period; // Volume ratio needs long period
        const atrNeed = this.config.long_st_period + 1; // ATR needs period + 1
        const buffer = 5; // Add a small buffer
        const calculatedNeed = Math.max(stNeed, volNeed, atrNeed) + buffer;
        logger.debug(`Calculated OHLCV candle requirement based on indicator periods: ${calculatedNeed}`);

        // Consider exchange limits on fetching OHLCV data
        const maxExchangeLimit = this.exchangeMgr?.exchange?.limits?.fetchOHLCV?.max ?? 1000; // Default to 1000 (Bybit often allows 1000/1500)
        const finalLimit = Math.min(calculatedNeed, maxExchangeLimit);

        if (finalLimit < calculatedNeed) {
             logger.warn(c.yellow(`Calculated required candle limit (${calculatedNeed}) exceeds exchange's reported max fetch limit (${maxExchangeLimit}). Using ${finalLimit}. Indicator accuracy might be slightly affected on initial candles.`));
        }
        return finalLimit;
    }

     /**
      * Calculates the time (in milliseconds) to sleep until the start of the next
      * candle interval plus a small buffer.
      * @returns {number} Milliseconds to sleep.
      */
     _calculateSleepTimeMs() {
        try {
            // Get timeframe duration in seconds from CCXT
            const timeframeSec = this.exchangeMgr?.exchange?.parseTimeframe?.(this.config.timeframe);
            if (!timeframeSec || timeframeSec <= 0) {
                logger.error(c.red(`Failed to parse timeframe duration ('${this.config.timeframe}') to seconds. Result: ${timeframeSec}. Defaulting sleep to 60000ms.`));
                return 60000; // Default to 1 minute
            }

            const timeframeMs = timeframeSec * 1000;
            const nowMs = Date.now();

            // Calculate the start time of the *next* candle interval
            const currentIntervalStartMs = Math.floor(nowMs / timeframeMs) * timeframeMs;
            const nextIntervalStartMs = currentIntervalStartMs + timeframeMs;

            // Add a small buffer (e.g., 2-5 seconds) to ensure the candle data is available when we wake up
            const bufferMs = 3000; // 3 seconds buffer
            const targetWakeTimeMs = nextIntervalStartMs + bufferMs;

            // Calculate total sleep time needed
            let sleepNeededMs = targetWakeTimeMs - nowMs;

            // If we are already past the target wake time (e.g., cycle took too long), calculate sleep until the *next* target
            if (sleepNeededMs <= 0) {
                 const nextTargetWakeTimeMs = nextIntervalStartMs + timeframeMs + bufferMs;
                 sleepNeededMs = nextTargetWakeTimeMs - nowMs;
                 logger.warn(c.yellow(`Current cycle exceeded timeframe! Now=${new Date(nowMs).toISOString()}, Original Target=${new Date(targetWakeTimeMs).toISOString()}. Sleeping until next target: ${new Date(nextTargetWakeTimeMs).toISOString()}`));
            }

            // Ensure a minimum sleep time to prevent overly tight loops if calculation is off
            const minSleepMs = 100;
            const finalSleepMs = Math.max(minSleepMs, sleepNeededMs);

             const nextStartDt = new Date(nextIntervalStartMs).toISOString();
             const wakeDt = new Date(targetWakeTimeMs).toISOString();
             logger.debug(`Timing: Timeframe=${this.config.timeframe}(${timeframeSec}s). Next Interval Starts=${nextStartDt}. Target Wake=${wakeDt}. Calculated Sleep=${finalSleepMs.toFixed(0)}ms.`);
             return finalSleepMs;

        } catch (e) {
             // Catch errors during timeframe parsing or calculation
             logger.error(c.red(`Error calculating sleep time for timeframe '${this.config.timeframe}': ${e.message}. Defaulting sleep to 60000ms.`), e.stack);
             return 60000; // Default to 1 minute on error
        }
    }

    /**
     * Executes a single trading logic cycle.
     * Fetches data, calculates indicators, generates signals, and executes actions.
     */
    async tradeLogic() {
         // Ensure all required components are initialized before proceeding
         if (!this.config || !this.notifier || !this.exchangeMgr || !this.orderMgr || !this.indicators) {
              logger.error(c.red("Critical Error: Bot components not initialized in tradeLogic. Stopping bot."));
              this._stop_requested = true; // Signal shutdown immediately
              return;
         }

         this.cycle_count++;
         const cycleStartDt = new Date().toISOString();
         logger.info(c.blue(`\n===== Cycle ${this.cycle_count} Start: ${cycleStartDt} =====`));

         // --- 1. Fetch Required Market Data ---
         const ohlcvLimit = this._getRequiredOhlcvLimit();
         logger.debug(`Fetching OHLCV data (limit=${ohlcvLimit})...`);
         const ohlcv = await this.exchangeMgr.fetchOhlcv(ohlcvLimit);

         // Validate OHLCV data
         if (!ohlcv || ohlcv.length === 0) {
             logger.warn(c.yellow("Failed to fetch valid OHLCV data or received empty array. Skipping cycle."));
             return; // Skip cycle if data is missing
         }

         // Check minimum length requirement for indicators
         const minCandlesNeeded = Math.max(
             this.config.long_st_period + 3, // Supertrend needs lookback + history for approx
             this.config.volume_long_period,  // Volume ratio needs long period
             this.config.long_st_period + 1    // ATR needs period + 1
         );
         if (ohlcv.length < minCandlesNeeded) {
             logger.warn(c.yellow(`Insufficient OHLCV data received (Got ${ohlcv.length}, Need >= ${minCandlesNeeded} for indicators). Skipping cycle.`));
             return; // Skip if not enough data for calculations
         }

         // --- Check for New Candle ---
         const currentCandleTs = ohlcv[ohlcv.length - 1][OHLCV_INDEX.TIMESTAMP];
         if (this.last_candle_ts !== null && currentCandleTs <= this.last_candle_ts) {
             logger.info(c.gray(`No new candle detected (Current TS: ${currentCandleTs} <= Last Processed TS: ${this.last_candle_ts}). Waiting for next cycle.`));
             return; // Skip cycle if candle hasn't updated
         }
         const currentCandleDt = new Date(currentCandleTs).toISOString();
         logger.info(`New candle detected. Processing candle for timestamp: ${c.bold(currentCandleDt)} (${currentCandleTs})`);


         // --- Fetch Other Data Concurrently ---
         logger.debug("Fetching position, order book, current price, and balance (if needed)...");
         let positionInfo, orderBook, currentPriceTicker, equity;
         try {
              // Fetch position first (uses cache, determines if balance fetch is needed)
              positionInfo = await this.exchangeMgr.getPosition();
              const { side: positionSide } = positionInfo; // Get side immediately

              // Fetch other data, including balance only if not in position or if balance is needed for sizing
              [orderBook, currentPriceTicker, equity] = await Promise.all([
                 this.exchangeMgr.fetchOrderBook(),
                 this.exchangeMgr.getCurrentPrice(),
                 // Fetch balance every cycle (uses cache) - needed for risk calc / TSL checks potentially
                 this.exchangeMgr.getBalance()
              ]);
         } catch (fetchError) {
              logger.error(c.red(`Error fetching required data batch: ${fetchError.message}. Skipping cycle.`), fetchError.stack);
              return;
         }
         // Destructure position info fully now
         const { side: positionSide, size: positionAmount, entryPrice } = positionInfo;


         // --- Determine Reference Prices ---
         // Use the close of the latest candle for signal generation
         const priceForSignals = parseFloat(ohlcv[ohlcv.length - 1][OHLCV_INDEX.CLOSE]);
         // Use the more real-time ticker price for TSL updates if available, fallback to close
         const priceForTsl = currentPriceTicker ?? priceForSignals;

         // Validate prices used in logic
         if (isNaN(priceForSignals) || priceForSignals <= 0) {
              logger.error(c.red(`Invalid reference price for signals (latest close: ${priceForSignals}). Skipping cycle.`));
              return;
         }
         // TSL price is validated within the TSL function itself

         logger.debug(`Reference Prices: Signal Price (Candle Close) = ${priceForSignals.toFixed(4)}, TSL Price (Ticker/Close) = ${priceForTsl?.toFixed(4) ?? 'N/A'}`);

         // --- 2. Calculate Indicators ---
         logger.debug("Calculating indicators...");
         let atr = null, shortSt = { value: null, isUptrend: null }, longSt = { value: null, isUptrend: null };
         let volumeRatio = null, obPressure = null;
         let indicatorsOk = false; // Flag to check if all necessary indicators calculated successfully

         try {
             // Calculate all indicators based on fetched data
             atr = this.indicators.calculateAtr(ohlcv, this.config.long_st_period); // Use long period for main ATR
             shortSt = this.indicators.calculateSupertrend(ohlcv, this.config.short_st_period, this.config.st_multiplier);
             longSt = this.indicators.calculateSupertrend(ohlcv, this.config.long_st_period, this.config.st_multiplier);
             volumeRatio = this.indicators.calculateVolumeRatio(ohlcv, this.config.volume_short_period, this.config.volume_long_period);
             obPressure = this.indicators.calculateOrderBookPressure(orderBook, this.config.order_book_depth);

             // Log calculated indicator values for monitoring
             const formatIndicator = (val, decimals = 4) => val !== null ? val.toFixed(decimals) : c.gray('N/A');
             const formatTrend = (trend) => trend === true ? c.green('UP') : trend === false ? c.red('DOWN') : c.gray('N/A');
             logger.info(`Indicators: Price=${c.bold(priceForSignals.toFixed(4))}, ATR=${c.dim(formatIndicator(atr, 6))}, ` + // More precision for ATR
                         `ST(${this.config.short_st_period})=${formatTrend(shortSt.isUptrend)} (${c.dim(formatIndicator(shortSt.value))}), ` +
                         `ST(${this.config.long_st_period})=${formatTrend(longSt.isUptrend)} (${c.dim(formatIndicator(longSt.value))}), ` +
                         `VolRatio=${c.dim(formatIndicator(volumeRatio, 2))}, OBPressure=${c.dim(formatIndicator(obPressure, 3))}`);

             // Check if all required indicators are valid for generating signals
             indicatorsOk = atr !== null &&
                            shortSt.isUptrend !== null && // Value check done in indicator
                            longSt.isUptrend !== null && // Value check done in indicator
                            volumeRatio !== null &&
                            obPressure !== null;

             if (!indicatorsOk) {
                 logger.warn(c.yellow("One or more required indicators could not be calculated. Signal generation skipped."));
             }

         } catch (e) {
              // Catch errors during the calculation process itself
              logger.error(c.red(`Error during indicator calculation phase: ${e.message}. Skipping rest of cycle.`), e.stack);
              return; // Stop processing this cycle if indicators fail critically
         }

         // --- 3. Generate Trading Signals ---
         let longEntrySignal = false;
         let shortEntrySignal = false;
         let longExitSignal = false;
         let shortExitSignal = false;

         if (indicatorsOk) {
             logger.debug("Generating trading signals based on calculated indicators...");

             // --- Long Entry Condition ---
             longEntrySignal = (
                 shortSt.isUptrend === true &&
                 longSt.isUptrend === true &&
                 volumeRatio > this.config.volume_spike_threshold &&
                 obPressure > this.config.ob_pressure_threshold
             );

             // --- Short Entry Condition ---
             shortEntrySignal = (
                 shortSt.isUptrend === false &&
                 longSt.isUptrend === false &&
                 volumeRatio > this.config.volume_spike_threshold &&
                 obPressure < (1.0 - this.config.ob_pressure_threshold) // Low buy pressure = high sell pressure
             );

             // --- Exit Conditions ---
             // Exit based on the *shorter* SuperTrend flipping against the position
             longExitSignal = (positionSide === PositionSide.LONG && shortSt.isUptrend === false);
             shortExitSignal = (positionSide === PositionSide.SHORT && shortSt.isUptrend === true);

             logger.info(`Signals Generated: Long Entry=${longEntrySignal ? c.green('TRUE') : 'false'}, Short Entry=${shortEntrySignal ? c.red('TRUE') : 'false'}, ` +
                         `Long Exit=${longExitSignal ? c.red('TRUE') : 'false'}, Short Exit=${shortExitSignal ? c.green('TRUE') : 'false'}`);
         }
         // If indicators were not OK, signals remain false.

         // --- 4. Log Current State (Position & Balance) ---
         const posSideColored = positionSide === PositionSide.LONG ? c.green(positionSide) : positionSide === PositionSide.SHORT ? c.red(positionSide) : c.gray(positionSide);
         logger.info(`Current State: Position=${posSideColored}, Size=${c.bold(positionAmount.toFixed(8))}, AvgEntry=${c.dim(entryPrice.toFixed(4))}`);
         if (equity !== null) {
             logger.info(`Current Equity: ${c.bold(equity.toFixed(2))} ${this.config.currency}`);
             // Add a check for insufficient equity in live mode when no position
             if (positionSide === PositionSide.NONE && equity <= 0 && !this.config.dry_run) {
                  logger.warn(c.yellow("Account equity is zero or negative. Cannot open new trades."));
             }
         } else {
             logger.error(c.red("Equity could not be determined (fetch/parse failed). Balance check skipped."));
         }


         // --- 5. Execute Actions Based on Signals and State ---
         let actionTakenThisCycle = false; // Flag to prevent multiple actions (e.g., exit and TSL update) in one cycle

         // --- Handle Exits First ---
         if (longExitSignal) {
              logger.info(c.bold(c.red("ACTION: Long exit signal triggered. Closing LONG position.")));
              const closeOrder = await this.orderMgr.closePosition(positionSide, positionAmount);
              if (closeOrder) actionTakenThisCycle = true;
         } else if (shortExitSignal) {
               logger.info(c.bold(c.green("ACTION: Short exit signal triggered. Closing SHORT position.")));
               const closeOrder = await this.orderMgr.closePosition(positionSide, positionAmount);
              if (closeOrder) actionTakenThisCycle = true;
         }

         // --- Handle Entries (only if not in a position and no exit occurred this cycle) ---
         if (!actionTakenThisCycle && positionSide === PositionSide.NONE) {
             // Ensure equity is valid before attempting entry
             if (equity === null) {
                 logger.error(c.red("Cannot attempt entry: Equity could not be determined."));
             } else if (equity <= 0 && !this.config.dry_run) {
                 logger.warn(c.yellow("Cannot attempt entry: Equity is zero or negative."));
             }
             // Proceed if equity is valid or in dry run
             else if (longEntrySignal) {
                  logger.info(c.bold(c.green("ACTION: Long entry signal triggered.")));
                  // Calculate size, SL, TP (ensure ATR is valid)
                  if (atr !== null) {
                      const amount = await this.orderMgr.calculatePositionSize(priceForSignals, atr, equity);
                      if (amount && amount > 0) {
                           const slPrice = priceForSignals - atr * this.config.sl_atr_mult;
                           const tpPrice = priceForSignals + atr * this.config.tp_atr_mult;
                           logger.info(`Placing LONG order: Amount=${amount.toFixed(8)}, Entry~=${priceForSignals.toFixed(4)}, SL=${slPrice.toFixed(4)}, TP=${tpPrice.toFixed(4)}`);
                           // Pass priceForSignals for validation within placeMarketOrder
                           const entryOrder = await this.orderMgr.placeMarketOrder(Side.BUY, amount, priceForSignals, slPrice, tpPrice);
                           if (entryOrder) actionTakenThisCycle = true;
                      } else {
                          logger.warn(c.yellow("Long entry signal detected, but position size calculation failed or resulted in zero/null amount. Order not placed."));
                      }
                  } else {
                       logger.warn(c.yellow("Long entry signal detected, but ATR is invalid. Cannot calculate size/SL/TP. Order not placed."));
                  }
             } else if (shortEntrySignal) {
                  logger.info(c.bold(c.red("ACTION: Short entry signal triggered.")));
                   // Calculate size, SL, TP (ensure ATR is valid)
                   if (atr !== null) {
                       const amount = await this.orderMgr.calculatePositionSize(priceForSignals, atr, equity);
                      if (amount && amount > 0) {
                           const slPrice = priceForSignals + atr * this.config.sl_atr_mult;
                           const tpPrice = priceForSignals - atr * this.config.tp_atr_mult;
                           logger.info(`Placing SHORT order: Amount=${amount.toFixed(8)}, Entry~=${priceForSignals.toFixed(4)}, SL=${slPrice.toFixed(4)}, TP=${tpPrice.toFixed(4)}`);
                           // Pass priceForSignals for validation
                           const entryOrder = await this.orderMgr.placeMarketOrder(Side.SELL, amount, priceForSignals, slPrice, tpPrice);
                          if (entryOrder) actionTakenThisCycle = true;
                      } else {
                          logger.warn(c.yellow("Short entry signal detected, but position size calculation failed or resulted in zero/null amount. Order not placed."));
                      }
                   } else {
                       logger.warn(c.yellow("Short entry signal detected, but ATR is invalid. Cannot calculate size/SL/TP. Order not placed."));
                   }
             }
         }

         // --- Manage Trailing Stop (only if in position AND no entry/exit action taken this cycle) ---
         if (!actionTakenThisCycle && positionSide !== PositionSide.NONE) {
              // Ensure we have valid price and ATR for TSL calculation
              if (!isNaN(priceForTsl) && priceForTsl > 0 && atr !== null) { // ATR validation done in indicator func
                   logger.info(c.cyan(`ACTION: Holding ${positionSide} position. Checking/updating Trailing Stop Loss...`));
                   // Call the TSL update logic
                   await this.orderMgr.updateTrailingStop(
                       positionSide,
                       positionAmount,
                       entryPrice,
                       priceForTsl, // Use Ticker price or fallback Close price
                       atr
                   );
                   // TSL update is background management
              } else {
                   logger.warn(c.yellow(`Skipping TSL update due to invalid inputs: price_for_tsl (${priceForTsl}), atr (${atr}).`));
              }
         } else if (!actionTakenThisCycle && positionSide === PositionSide.NONE) {
              // If no position and no entry signal, just log holding cash
              logger.info(c.gray("ACTION: Hold (No position, no entry signal)."));
         }

         // --- Mark Candle as Processed ---
         // Update last_candle_ts only after all logic for the current candle is complete
         this.last_candle_ts = currentCandleTs;
         logger.debug(`Updated last processed candle timestamp to: ${this.last_candle_ts}`);

         logger.info(c.blue(`===== Cycle ${this.cycle_count} End =====`));
    }


    /**
     * Starts the main trading loop of the bot.
     */
    async run() {
        // Prevent starting if already running or not initialized
        if (this._isRunning) {
             logger.warn("Bot is already running. Ignoring start request.");
             return;
        }
        if (!this.config || !this.notifier || !this.exchangeMgr || !this.orderMgr) {
             logger.error(c.red("Cannot start main loop: Bot components not initialized properly. Run initialize() first."));
             return;
        }

        this._isRunning = true;
        this._stop_requested = false; // Ensure stop flag is reset on start

        logger.info(c.bold(c.blue("----- Starting Trading Bot Main Loop -----")));
        // Send startup notification
        let startupMessage = `Trading Bot started: ${this.config.symbol} ${this.config.timeframe}`;
        if (this.config.dry_run) {
            startupMessage += " (DRY RUN)";
            this.notifier.sendSms(startupMessage, this.config);
        } else {
            startupMessage += " (LIVE TRADING)";
            this.notifier.sendSms(`ALERT: ${startupMessage}. Monitor closely!`, this.config); // Add alert prefix for live
        }


        // --- Main Loop ---
        while (!this._stop_requested) {
            const cycleStartTime = process.hrtime.bigint(); // High-resolution timer start
            try {
                // Execute one cycle of the trading logic
                await this.tradeLogic();

                const cycleEndTime = process.hrtime.bigint();
                const cycleDurationMs = Number(cycleEndTime - cycleStartTime) / 1e6; // Duration in milliseconds
                logger.debug(`Cycle execution time: ${cycleDurationMs.toFixed(1)} ms.`);

                // --- Log Periodic Health Status ---
                const healthCheckIntervalCycles = 60; // e.g., hourly on 1m timeframe
                 if (this.cycle_count > 0 && this.cycle_count % healthCheckIntervalCycles === 0) {
                     const uptimeSeconds = (Date.now() / 1000) - this.start_time;
                     const uptimeStr = new Date(uptimeSeconds * 1000).toISOString().slice(11, 19); // HH:MM:SS
                     const avgCycleTimeMs = (uptimeSeconds * 1000) / this.cycle_count;

                     // Fetch current state (uses cache mostly) for logging
                     try {
                          // Use cached values where possible to reduce API load during health check
                          const hcPosInfo = await this.exchangeMgr.getPosition(); // Uses cache
                          const hcEquity = await this.exchangeMgr.getBalance(); // Uses cache
                          const { side: hcPosSide, size: hcPosAmount, entryPrice: hcEntryPrice } = hcPosInfo;
                          const hcPosSideColored = hcPosSide === PositionSide.LONG ? c.green(hcPosSide) : hcPosSide === PositionSide.SHORT ? c.red(hcPosSide) : c.gray(hcPosSide);

                          logger.info(
                              c.blue(`--- Health Check --- Uptime=${uptimeStr}, Cycles=${this.cycle_count}, `) +
                              c.dim(`AvgCycleTime=${avgCycleTimeMs.toFixed(0)}ms | `) +
                              c.blue(`Position=${hcPosSideColored}(${hcPosAmount.toFixed(8)}@${hcEntryPrice.toFixed(4)}), Equity=${hcEquity !== null ? hcEquity.toFixed(2) : 'N/A'} | `) +
                              c.dim(`ActiveSL_ID=${this.orderMgr.active_sl_order_id || 'None'} ---`)
                          );
                     } catch (healthErr) {
                          logger.error(c.red(`Error during periodic health check data fetch: ${healthErr.message}`), healthErr.stack);
                     }
                 }


                // --- Wait Until Next Candle Interval ---
                const sleepDurationMs = this._calculateSleepTimeMs();
                // Adjust sleep time by subtracting the time taken for the current cycle execution, ensure minimum sleep
                const actualSleepMs = Math.max(50, sleepDurationMs - cycleDurationMs);
                if (actualSleepMs < 1000) { // Log if sleep is very short
                     logger.debug(`Short sleep duration calculated: ${actualSleepMs.toFixed(0)}ms.`);
                }

                await sleep(actualSleepMs); // Pause execution

            } catch (e) {
                 // --- Handle Critical Errors within the Main Loop ---
                  logger.error(c.red(`CRITICAL ERROR encountered in main loop (Cycle ${this.cycle_count}): ${e.constructor.name} - ${e.message}`), e.stack);

                  if (e instanceof ccxt.AuthenticationError || e instanceof ccxt.PermissionDenied) {
                      // Authentication/Permission errors are usually fatal - stop the bot
                      const reason = e instanceof ccxt.AuthenticationError ? "Authentication" : "Permission";
                      logger.error(c.red(`${reason} failed during execution. Shutting down bot. Check API keys/permissions.`));
                      this.notifier.sendSms(`CRITICAL: Bot ${reason} FAILED mid-run! Shutting down.`, this.config);
                      this._stop_requested = true; // Signal stop
                  } else if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeNotAvailable || e instanceof ccxt.DDoSProtection) {
                      // For persistent connection issues not handled by retries inside logic
                      logger.error(c.red(`Persistent connection/timeout error in main loop: ${e.constructor.name}. Pausing before next cycle attempt...`));
                      this.notifier.sendSms(`WARN: Bot experiencing persistent connection issues (${e.constructor.name}). Pausing.`, this.config);
                      // Wait longer before retrying the cycle
                      await sleep(Math.max(this.config.retry_delay * 5000, 30000)); // e.g., 30 seconds minimum
                  } else {
                      // Catch-all for other unexpected errors (e.g., programming errors in tradeLogic)
                      logger.error(c.red(`An unexpected critical error occurred. Bot will pause and attempt to continue.`));
                      this.notifier.sendSms(`CRITICAL BOT ERROR: ${e.constructor.name} - ${e.message.substring(0,100)}. Bot paused. Check logs!`, this.config);
                      // Pause significantly to allow for investigation or recovery
                      const pauseDuration = Math.max(this.config.retry_delay * 10000, 60000); // Min 60s pause
                      logger.info(`Pausing for ${pauseDuration / 1000} seconds due to unexpected critical error...`);
                      await sleep(pauseDuration);
                 }
                 // Loop will continue unless _stop_requested is set true
            }
        } // --- End of main while loop ---

        // --- Shutdown Sequence ---
        logger.info(c.blue("Stop request received or critical error occurred. Exiting main loop."));
        await this.attemptSafeShutdown(); // Attempt to close positions etc.
        await logger.closeLogFile(); // Close the log file handle
        this._isRunning = false; // Mark bot as not running
        logger.info(c.bold(c.blue("----- Trading Bot Stopped -----")));
    }

    /**
     * Signals the bot to stop gracefully after the current cycle.
     */
    stop() {
        if (!this._stop_requested) {
             logger.info(c.yellow("Graceful stop requested. Bot will shut down after completing the current cycle (if running)."));
             this._stop_requested = true; // Set the flag to exit the main loop
        } else {
             logger.info("Stop already requested.");
        }
    }

    /**
     * Attempts a safe shutdown: checks for open positions and tries to close them (in live mode).
     */
    async attemptSafeShutdown() {
        logger.info(c.yellow("--- Initiating Safe Shutdown Sequence ---"));

        // Check if components needed for shutdown are available
        if (!this.config || !this.notifier || !this.exchangeMgr || !this.orderMgr) {
             logger.error(c.red("Cannot perform safe shutdown: Bot components not fully initialized. Manual check required."));
             this._attemptEmergencyNotification("ERROR: Bot shutdown failed (components missing). Manual check required!");
             return;
        }

        // Skip position closing in dry run mode
        if (this.config.dry_run) {
            logger.info(c.magenta("Dry run mode: No real positions to close during shutdown."));
            this.notifier.sendSms(`Bot shutdown initiated (Dry Run) for ${this.config.symbol}.`, this.config);
            return;
        }

        // --- Live Mode Shutdown ---
        try {
            // Fetch the current position one last time, bypassing cache
            logger.info("Checking for open positions to close (Live Mode)...");
            this.exchangeMgr._setCache('position', this.config.symbol, null); // Clear cache for fresh data
            const { side: posSide, size: posAmount } = await this.exchangeMgr.getPosition();

            if (posSide !== PositionSide.NONE && posAmount > 0) {
                // Position found, attempt to close
                logger.warn(c.yellow(`Open ${posSide.toUpperCase()} position detected (Size: ${posAmount.toFixed(8)}). Attempting to close market...`));
                this.notifier.sendSms(`ALERT: Bot shutting down. Closing ${posSide} position (${posAmount.toFixed(8)}) for ${this.config.symbol}.`, this.config);

                // Try closing the position (includes SL cancellation)
                let closedOrder = null;
                const closeAttempts = 2; // Try closing twice if the first attempt fails
                for (let attempt = 1; attempt <= closeAttempts; attempt++) {
                     logger.info(`Attempting to place close order (Attempt ${attempt}/${closeAttempts})...`);
                     // closePosition handles SL cancel and placing the market close order
                     closedOrder = await this.orderMgr.closePosition(posSide, posAmount);

                     if (closedOrder && closedOrder.id) {
                          logger.info(c.green(`Position close order placed successfully during shutdown (Attempt ${attempt}). Order ID: ${closedOrder.id}`));
                          // Optional: Wait briefly and re-verify position is closed
                          logger.info("Waiting 5 seconds to allow order processing before final check...");
                          await sleep(5000);
                          this.exchangeMgr._setCache('position', this.config.symbol, null); // Clear cache again
                          const { side: finalPosSide } = await this.exchangeMgr.getPosition();
                          if (finalPosSide === PositionSide.NONE) {
                               logger.info(c.green("Position confirmed CLOSED after waiting."));
                               this.notifier.sendSms(`${this.config.symbol} position successfully closed during shutdown.`, this.config);
                          } else {
                               logger.warn(c.yellow(`Position still detected as ${finalPosSide} after placing close order. Status might be delayed or close failed. Manual verification strongly advised!`));
                               this.notifier.sendSms(`WARN: ${this.config.symbol} position may NOT be fully closed after shutdown order. Check Exchange Manually!`, this.config);
                          }
                          break; // Exit loop on successful order placement
                     } else {
                          // Close order placement failed (helper logged details)
                          logger.error(c.red(`Position close order placement FAILED during shutdown (Attempt ${attempt}).`));
                          if (attempt < closeAttempts) {
                              logger.info(`Waiting ${this.config.retry_delay}s before retry...`);
                              await sleep(this.config.retry_delay * 1000);
                          }
                     }
                } // End close attempt loop

                // If closing failed after all attempts
                if (!closedOrder) {
                     logger.error(c.red(c.bold("CRITICAL: FAILED to place position close order during shutdown after multiple attempts. Manual intervention REQUIRED on the exchange.")));
                     this.notifier.sendSms(`CRITICAL: FAILED to close ${this.config.symbol} position during shutdown! Check Exchange Manually NOW!`, this.config);
                }

            } else {
                // No open position found
                logger.info("No open position found. No closing action needed.");
                 this.notifier.sendSms(`Bot shutdown initiated (No position) for ${this.config.symbol}.`, this.config);
            }

        } catch (e) {
            // Catch errors during the shutdown check/close process itself
            logger.error(c.red(`Error during safe shutdown position check/close: ${e.message}`), e.stack);
            this.notifier.sendSms(`ERROR during bot shutdown process for ${this.config.symbol}: ${e.message.substring(0,100)}. Manual check advised!`, this.config);
        }
        logger.info(c.yellow("--- Safe Shutdown Sequence Finished ---"));
    }
}


// --- Main Execution Function ---
async function main() {
    // Assign to global instance for access by shutdown handlers
    botInstance = new TradingBot();

    try {
        // Perform all asynchronous initialization steps
        await botInstance.initialize();

        // Start the main trading loop if initialization succeeded
        await botInstance.run();

    } catch (e) {
        // Catch initialization errors that prevent the bot from starting
        // Logging and notification attempts are handled within initialize() or its called methods
        // Log again here for final confirmation
        logger.error(c.red(c.bold(`Bot failed to initialize and start: ${e.message}`)));
        // Ensure log file is closed if opened partially
        await logger.closeLogFile();
        // Exit with an error code if initialization fails
        process.exitCode = 1; // Set exit code instead of calling process.exit directly
    }
}

// --- Graceful Shutdown Handler ---
// Listens for termination signals (Ctrl+C, kill command)
let isShuttingDown = false; // Prevent multiple shutdown attempts
const shutdown = async (signal) => {
    // Use console.log directly as logger might be involved in shutdown
    console.log(`\n${c.yellow(c.bold(`\nReceived ${signal} signal.`))}`); // Add newline for clarity

    if (isShuttingDown) {
        console.log(c.yellow("Shutdown already in progress... Press Ctrl+C again to force exit (unsafe)."));
        return;
    }
    isShuttingDown = true;

    if (!botInstance) {
        console.log(c.yellow("Bot instance not available. Exiting immediately."));
        await logger.closeLogFile(); // Attempt to close logs even if bot failed early
        process.exitCode = 0;
        return;
    }

    if (botInstance._isRunning && !botInstance._stop_requested) {
         // If bot is running and stop hasn't been requested yet
         console.log(c.yellow("Initiating graceful shutdown... Signaling main loop to stop."));
         botInstance.stop(); // Signal the main loop to stop
         // The main loop's `run` method will handle the shutdown sequence and log closing.
         // Add a timeout to force exit if shutdown hangs
         setTimeout(() => {
            console.error(c.red("Shutdown timed out after 30 seconds. Forcing exit. Manual check of exchange required!"));
            process.exit(1); // Force exit with error code
         }, 30000).unref(); // 30 seconds timeout, unref allows node to exit if shutdown completes sooner

    } else if (botInstance._isRunning && botInstance._stop_requested) {
         // If stop was already requested (e.g., multiple Ctrl+C)
         console.log(c.yellow("Shutdown already requested and likely in progress..."));
         // Allow the existing shutdown process or timeout to complete.
    } else {
         // If bot instance exists but the main loop isn't running
         console.log(c.yellow("Bot loop not running. Attempting cleanup and exiting."));
         await botInstance.attemptSafeShutdown(); // Attempt cleanup just in case
         await logger.closeLogFile(); // Ensure logs are closed
         process.exitCode = 0;
    }
};

// Register signal handlers
process.on('SIGINT', () => shutdown('SIGINT'));   // Ctrl+C from terminal
process.on('SIGTERM', () => shutdown('SIGTERM')); // kill command (default)

// --- Start the Bot ---
main().catch(e => {
    // Catch any unhandled promise rejections specifically from the main() call itself
    console.error(c.red(`Unhandled critical error during bot startup sequence: ${e.message}`), e.stack);
    logger.closeLogFile().finally(() => process.exit(1)); // Attempt log close, then exit
});

// --- Global Error Handlers (Catch-all Safety Net) ---

// Catch unhandled promise rejections globally
process.on('unhandledRejection', async (reason, promise) => {
  console.error(c.red(c.bold('\n--- UNHANDLED PROMISE REJECTION ---')));
  console.error(c.red('Reason:'), reason); // Reason can be Error object or other type
  // Log stack if reason is an Error
  if (reason instanceof Error) {
      console.error(c.red('Stack:'), reason.stack);
  }
  // console.error(c.red('Promise:'), promise); // Promise object can be large

  if (!isShuttingDown) {
      isShuttingDown = true; // Prevent recursive shutdown attempts
      // Attempt emergency notification/shutdown if possible
      if (botInstance && botInstance._attemptEmergencyNotification) {
          const reasonStr = (reason instanceof Error) ? reason.message : String(reason);
          botInstance._attemptEmergencyNotification(`CRITICAL: Unhandled Rejection! Reason: ${reasonStr}`.substring(0, 150));
      }
      // Attempt safe shutdown if bot instance exists
      if (botInstance && botInstance.attemptSafeShutdown) {
           console.error(c.red("Attempting safe shutdown due to unhandled rejection..."));
           await botInstance.attemptSafeShutdown();
      }
      await logger.closeLogFile(); // Attempt to close log file
      console.error(c.red("Exiting due to unhandled promise rejection."));
      process.exitCode = 1; // Set exit code
      // Force exit after a short delay if shutdown doesn't complete
      setTimeout(() => process.exit(1), 5000).unref();
  }
});

// Catch uncaught synchronous exceptions globally
process.on('uncaughtException', async (error) => {
  console.error(c.red(c.bold('\n--- UNCAUGHT EXCEPTION ---')));
  console.error(c.red('Error:'), error);
  console.error(c.red('Stack:'), error.stack);

  if (!isShuttingDown) {
      isShuttingDown = true;
      // Attempt emergency notification/shutdown
      if (botInstance && botInstance._attemptEmergencyNotification) {
           botInstance._attemptEmergencyNotification(`CRITICAL: Uncaught Exception! Error: ${error.message}`.substring(0, 150));
      }
      // Attempt safe shutdown if bot instance exists (could be risky if shutdown logic also throws)
      if (botInstance && botInstance.attemptSafeShutdown) {
           console.error(c.red("Attempting safe shutdown due to uncaught exception..."));
           await botInstance.attemptSafeShutdown();
      }
      await logger.closeLogFile(); // Attempt to close log file
      console.error(c.red("Exiting due to uncaught exception."));
      process.exitCode = 1; // Set exit code
      // Force exit after a short delay
      setTimeout(() => process.exit(1), 5000).unref();
  }
});

