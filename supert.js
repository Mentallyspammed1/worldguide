#!/usr/bin/env node
// -*- coding: utf-8 -*-

// ███████╗██╗   ██╗███████╗ ██████╗ ███████╗███████╗██╗   ██╗███████╗
// ██╔════╝██║   ██║██╔════╝██╔════╝ ██╔════╝██╔════╝██║   ██║██╔════╝
// ███████╗██║   ██║███████╗██║  ███╗███████╗███████╗██║   ██║███████╗
// ╚════██║██║   ██║╚════██║██║   ██║╚════██║╚════██║██║   ██║╚════██║
// ███████║╚██████╔╝███████║╚██████╔╝███████║███████║╚██████╔╝███████║
// ╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚══════╝
// Pyrmethus Roofed/Fisher/SuperTrend/ADX Bot - Enhanced Invocation v1.3

/**
 * @fileoverview Enhanced Trading Bot using CCXT for Bybit Linear Contracts.
 * Strategy: Combines Ehlers Roofing Filter, Inverse Fisher Transform, SuperTrend, and ADX.
 * Features:
 * - Robust indicator calculations with input validation.
 * - Asynchronous file logging with rotation and levels.
 * - Enhanced state management with validation and atomic saving.
 * - Reliable order placement and management using CCXT unified methods.
 * - Bybit V5 API parameter compatibility.
 * - Improved error handling with specific CCXT error recognition and retries.
 * - Consistent use of market precision.
 * - Refined Dry Run simulation.
 * - ATR-based Trailing Stop Loss (TSL) using Cancel & Replace.
 * - Graceful shutdown handling signals.
 * - Termux SMS notifications (optional).
 */

// --- Core Node.js Modules & Libraries ---
import ccxt from 'ccxt';
import dotenv from 'dotenv';
import { exec } from 'child_process';
import fs from 'fs'; // Use synchronous fs only for initial log directory check
import fsPromises from 'fs/promises'; // Use async promises for most file operations
import path from 'path';
import { fileURLToPath } from 'url';
import { inspect } from 'util'; // For detailed object logging
import {
    cyan, green, yellow, red, magenta, gray, bold, dim, reset
} from 'nanocolors'; // Using nanocolors alias 'c' is less common in shared code
import { EMA, ATR, ADX, InverseFisherTransform } from 'technicalindicators';

dotenv.config(); // Load environment variables from .env file

// --- Constants ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const LOG_LEVELS = Object.freeze({ DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3, CRITICAL: 4 });
const PositionSide = Object.freeze({ LONG: 'long', SHORT: 'short', NONE: null });
const DEFAULT_PERSISTENCE_FILE = 'trading_state_pyrmethus_v1_3.json';
const DEFAULT_LOG_FILE = 'pyrmethus_bot.log';
const DEFAULT_LOG_LEVEL = 'INFO';
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_DELAY_MS = 5000;
const DEFAULT_POLL_INTERVAL_MS = 30000;
const MAX_LOG_FILE_SIZE = 10 * 1024 * 1024; // 10 MB limit for log rotation

// --- Configuration Loading & Validation ---
/**
 * Loads and validates configuration from environment variables.
 * @returns {object} The validated configuration object.
 * @throws {Error} If critical configuration is missing or invalid.
 */
function loadAndValidateConfig() {
    const config = {
        BYBIT_API_KEY: process.env.BYBIT_API_KEY || null,
        BYBIT_API_SECRET: process.env.BYBIT_API_SECRET || null,
        SYMBOL: process.env.SYMBOL,
        LEVERAGE: parseInt(process.env.LEVERAGE, 10),
        TIMEFRAME: process.env.TIMEFRAME,
        ORDER_AMOUNT_USD: parseFloat(process.env.ORDER_AMOUNT_USD),
        ROOF_FAST_EMA: parseInt(process.env.ROOF_FAST_EMA, 10),
        ROOF_SLOW_EMA: parseInt(process.env.ROOF_SLOW_EMA, 10),
        ST_EMA_PERIOD: parseInt(process.env.ST_EMA_PERIOD, 10),
        ST_ATR_PERIOD: parseInt(process.env.ST_ATR_PERIOD, 10),
        ST_MULTIPLIER: parseFloat(process.env.ST_MULTIPLIER),
        FISHER_PERIOD: parseInt(process.env.FISHER_PERIOD, 10),
        ADX_PERIOD: parseInt(process.env.ADX_PERIOD, 10),
        MIN_ADX_LEVEL: parseFloat(process.env.MIN_ADX_LEVEL),
        MAX_ADX_LEVEL: parseFloat(process.env.MAX_ADX_LEVEL),
        RANGE_ATR_PERIOD: parseInt(process.env.RANGE_ATR_PERIOD, 10),
        MIN_ATR_PERCENTAGE: parseFloat(process.env.MIN_ATR_PERCENTAGE),
        TSL_ATR_MULTIPLIER: parseFloat(process.env.TSL_ATR_MULTIPLIER),
        INITIAL_SL_ATR_MULTIPLIER: parseFloat(process.env.INITIAL_SL_ATR_MULTIPLIER),
        POLL_INTERVAL_MS: parseInt(process.env.POLL_INTERVAL_MS || DEFAULT_POLL_INTERVAL_MS, 10),
        PERSISTENCE_FILE: process.env.PERSISTENCE_FILE || DEFAULT_PERSISTENCE_FILE,
        LOG_LEVEL: (process.env.LOG_LEVEL || DEFAULT_LOG_LEVEL).toUpperCase(),
        LOG_TO_FILE: process.env.LOG_TO_FILE === 'true',
        LOG_FILE_PATH: process.env.LOG_FILE_PATH || DEFAULT_LOG_FILE,
        DRY_RUN: process.env.DRY_RUN === 'true',
        TERMUX_NOTIFY: process.env.TERMUX_NOTIFY === 'true',
        NOTIFICATION_PHONE_NUMBER: process.env.NOTIFICATION_PHONE_NUMBER || null,
        CLOSE_ON_SHUTDOWN: process.env.CLOSE_ON_SHUTDOWN !== 'false', // Default true
        MAX_RETRIES: parseInt(process.env.MAX_RETRIES || DEFAULT_MAX_RETRIES, 10),
        RETRY_DELAY_MS: parseInt(process.env.RETRY_DELAY_MS || DEFAULT_RETRY_DELAY_MS, 10),
    };

    const errors = [];
    const requiredKeys = ['SYMBOL', 'TIMEFRAME', 'ORDER_AMOUNT_USD', 'LEVERAGE'];
    const numericKeysPositive = [
        'LEVERAGE', 'ORDER_AMOUNT_USD', 'ROOF_FAST_EMA', 'ROOF_SLOW_EMA',
        'ST_EMA_PERIOD', 'ST_ATR_PERIOD', 'ST_MULTIPLIER', 'FISHER_PERIOD',
        'ADX_PERIOD', 'RANGE_ATR_PERIOD', 'TSL_ATR_MULTIPLIER', 'INITIAL_SL_ATR_MULTIPLIER',
    ];
    const numericKeysNonNegative = [
         'MIN_ADX_LEVEL', 'MAX_ADX_LEVEL', 'MIN_ATR_PERCENTAGE', 'MAX_RETRIES', 'RETRY_DELAY_MS', 'POLL_INTERVAL_MS'
    ];

    // Check required keys (only if not in dry run for API keys)
    for (const key of requiredKeys) {
        if (!config[key]) {
            errors.push(`Missing required environment variable: ${key}`);
        }
    }
    if (!config.DRY_RUN && (!config.BYBIT_API_KEY || !config.BYBIT_API_SECRET)) {
        errors.push("Missing BYBIT_API_KEY or BYBIT_API_SECRET for live trading.");
    }

    // Check positive numeric keys
    for (const key of numericKeysPositive) {
        if (isNaN(config[key]) || config[key] <= 0) {
            errors.push(`Environment variable ${key} must be a positive number. Found: ${config[key]}`);
        }
    }
    // Check non-negative numeric keys
     for (const key of numericKeysNonNegative) {
        if (isNaN(config[key]) || config[key] < 0) {
            errors.push(`Environment variable ${key} must be a non-negative number. Found: ${config[key]}`);
        }
    }

    // Check specific logic constraints
    if (config.ROOF_FAST_EMA >= config.ROOF_SLOW_EMA) errors.push("ROOF_FAST_EMA must be less than ROOF_SLOW_EMA.");
    if (config.MIN_ADX_LEVEL >= config.MAX_ADX_LEVEL) errors.push("MIN_ADX_LEVEL must be less than MAX_ADX_LEVEL.");
    if (config.POLL_INTERVAL_MS < 1000) errors.push("POLL_INTERVAL_MS should be at least 1000ms.");
    if (config.RETRY_DELAY_MS < 500) errors.push("RETRY_DELAY_MS should be at least 500ms.");

    // Validate Termux config
    if (config.TERMUX_NOTIFY && !config.NOTIFICATION_PHONE_NUMBER) errors.push("TERMUX_NOTIFY is true, but NOTIFICATION_PHONE_NUMBER is missing.");

    // Validate Log Level
    if (!LOG_LEVELS.hasOwnProperty(config.LOG_LEVEL)) {
        errors.push(`Invalid LOG_LEVEL: ${config.LOG_LEVEL}. Use DEBUG, INFO, WARN, ERROR, or CRITICAL.`);
        config.LOG_LEVEL = DEFAULT_LOG_LEVEL; // Use default if invalid
    }

    // Validate file logging path if enabled
    if (config.LOG_TO_FILE && !config.LOG_FILE_PATH) {
         errors.push("LOG_TO_FILE is true, but LOG_FILE_PATH is missing.");
         config.LOG_FILE_PATH = DEFAULT_LOG_FILE; // Use default if invalid
    }

    if (errors.length > 0) {
        // Use console.error directly as logger might not be fully initialized
        console.error(red(bold("Configuration Errors Found:")));
        errors.forEach(err => console.error(red(`- ${err}`)));
        console.error(red(bold("Please check your .env file. Halting bot.")));
        process.exit(1);
    }

    return config;
}
const CONFIG = loadAndValidateConfig();

// --- Global State & Control ---
let logFileStream = null;
let logFileRotationCheckScheduled = false;
let isShuttingDown = false;
let mainLoopTimeoutId = null;
const persistenceFilePath = path.join(__dirname, CONFIG.PERSISTENCE_FILE);
const logFilePath = path.resolve(__dirname, CONFIG.LOG_FILE_PATH); // Use absolute path for clarity

// --- Enhanced Logger ---
/**
 * Rotates the log file if it exceeds the size limit.
 * @returns {Promise<void>}
 */
async function rotateLogFileIfNeeded() {
    if (!CONFIG.LOG_TO_FILE || !logFileStream) return;

    try {
        const stats = await fsPromises.stat(logFilePath);
        if (stats.size > MAX_LOG_FILE_SIZE) {
            logger.info(`Log file size (${(stats.size / 1024 / 1024).toFixed(1)}MB) exceeds limit (${MAX_LOG_FILE_SIZE / 1024 / 1024}MB). Rotating...`);
            await closeLogFileStream(); // Close current stream

            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const rotatedPath = `${logFilePath}.${timestamp}`;
            await fsPromises.rename(logFilePath, rotatedPath);
            logger.info(`Log file rotated to: ${rotatedPath}`);

            await setupLogFileStream(); // Re-open the stream
        }
    } catch (error) {
        if (error.code !== 'ENOENT') { // Ignore if file doesn't exist yet
            console.error(red(bold(`[ERROR] Failed during log rotation check for ${logFilePath}: ${error.message}`)));
        }
    } finally {
        // Schedule the next check (e.g., every hour)
        logFileRotationCheckScheduled = false; // Reset flag
        if (!isShuttingDown) {
            setTimeout(() => {
                 logFileRotationCheckScheduled = true;
                 rotateLogFileIfNeeded();
             }, 3600 * 1000); // Check every hour
        }
    }
}

/**
 * Opens the log file stream for appending.
 * @returns {Promise<void>}
 */
async function setupLogFileStream() {
    if (!CONFIG.LOG_TO_FILE) return;
    try {
        // Ensure directory exists (sync check is acceptable at startup)
        const logDir = path.dirname(logFilePath);
        if (!fs.existsSync(logDir)) {
            fs.mkdirSync(logDir, { recursive: true });
        }
        // Use 'a' flag to append
        logFileStream = fs.createWriteStream(logFilePath, { flags: 'a' });
        logFileStream.on('error', (err) => {
            console.error(red(bold(`[ERROR] Log file stream error: ${err.message}`)));
            logFileStream = null; // Stop trying to write
        });
        // Initial log rotation check
        if (!logFileRotationCheckScheduled) {
             logFileRotationCheckScheduled = true;
             rotateLogFileIfNeeded(); // Check size on startup
        }
        console.log(cyan(`[INFO] Logging to file enabled: ${logFilePath}`));
    } catch (err) {
        console.error(red(bold(`[ERROR] Failed to open log file ${logFilePath}: ${err.message}. File logging disabled.`)));
        logFileStream = null;
    }
}

/**
 * Closes the log file stream gracefully.
 * @returns {Promise<void>}
 */
async function closeLogFileStream() {
    return new Promise((resolve) => {
        if (logFileStream) {
            console.log(yellow("[INFO] Closing log file stream..."));
            const streamToClose = logFileStream;
            logFileStream = null; // Prevent further writes immediately
            streamToClose.end(() => {
                console.log(green("[INFO] Log file stream closed."));
                resolve();
            });
            // Handle potential errors on close
            streamToClose.on('error', (err) => {
                 console.error(red(bold(`[ERROR] Error closing log file stream: ${err.message}`)));
                 resolve(); // Resolve even on error to not block shutdown
            });
        } else {
            resolve();
        }
    });
}

const logger = {
    _log: (level, levelName, colorFn, messageParts) => {
        const currentLevel = LOG_LEVELS[CONFIG.LOG_LEVEL];
        if (level < currentLevel) return;
        // Reduce log noise during shutdown, only log INFO and above
        if (isShuttingDown && level < LOG_LEVELS.INFO) return;

        const timestamp = new Date().toISOString();
        // Format message using util.inspect for objects
        const message = messageParts.map(part =>
            typeof part === 'object' && part !== null ? inspect(part, { depth: 3, colors: false }) : String(part)
        ).join(' ');

        // Console logging with color
        const consoleMessage = colorFn(`[${levelName.padEnd(5)}] ${timestamp} ${message}`);
        const consoleMethod = level >= LOG_LEVELS.ERROR ? console.error : level === LOG_LEVELS.WARN ? console.warn : console.log;
        consoleMethod(consoleMessage);

        // File logging (append, handle stream errors)
        if (logFileStream && logFileStream.writable) {
            const fileMessage = `[${levelName}] ${timestamp} ${message}\n`;
            logFileStream.write(fileMessage, 'utf8', (err) => {
                if (err) {
                    console.error(red(bold(`[ERROR] Failed to write to log file: ${err.message}`)));
                    // Consider closing stream after repeated errors?
                    // closeLogFileStream().then(() => logFileStream = null);
                }
            });
        }
    },
    debug: (...args) => logger._log(LOG_LEVELS.DEBUG, 'DEBUG', dim(gray), args),
    info: (...args) => logger._log(LOG_LEVELS.INFO, 'INFO', cyan, args),
    warn: (...args) => logger._log(LOG_LEVELS.WARN, 'WARN', yellow, args),
    error: (...args) => logger._log(LOG_LEVELS.ERROR, 'ERROR', red, args),
    critical: (...args) => logger._log(LOG_LEVELS.CRITICAL, 'CRIT', red(bold), args),
    trade: (...args) => logger._log(LOG_LEVELS.INFO, 'TRADE', green(bold), args),
    dryRun: (...args) => CONFIG.DRY_RUN && logger._log(LOG_LEVELS.INFO, 'DRYRUN', magenta, args),
};

// --- Bot State Management ---
let state = {
    positionSide: PositionSide.NONE, // 'long', 'short', or null
    entryPrice: null,               // number | null
    positionAmount: null,           // number | null
    currentTSL: null,               // { price: number, orderId: string | null } | null
    lastSignal: null,               // 'long' or 'short' to prevent immediate flip-flop entries
    cycleCount: 0                   // number
};

/**
 * Loads the bot's state from the persistence file.
 * Validates loaded data structure and consistency.
 * Initializes state if the file doesn't exist or is invalid.
 * @returns {Promise<void>}
 */
async function loadState() {
    try {
        logger.info(`Loading state from: ${persistenceFilePath}`);
        const data = await fsPromises.readFile(persistenceFilePath, 'utf8');
        const loadedState = JSON.parse(data);

        // Validate loaded object structure
        if (typeof loadedState !== 'object' || loadedState === null) throw new Error("Invalid state file format: not an object.");
        if (typeof loadedState.state !== 'object' || loadedState.state === null) throw new Error("Invalid state file format: missing 'state' object.");
        if (typeof loadedState.currentPosition !== 'object' || loadedState.currentPosition === null) throw new Error("Invalid state file format: missing 'currentPosition' object.");
        if (typeof loadedState.activeOrders !== 'object' || loadedState.activeOrders === null) throw new Error("Invalid state file format: missing 'activeOrders' object.");

        // --- Restore Core State ---
        state.cycleCount = (typeof loadedState.state.cycleCount === 'number' && loadedState.state.cycleCount >= 0) ? loadedState.state.cycleCount : 0;
        state.lastSignal = ['long', 'short'].includes(loadedState.state.lastSignal) ? loadedState.state.lastSignal : null;

        // --- Restore Position State ---
        const loadedPosSide = loadedState.currentPosition.side;
        const loadedPosAmount = loadedState.currentPosition.size ?? loadedState.currentPosition.positionAmount; // Allow old key
        const loadedEntryPrice = loadedState.currentPosition.entryPrice;

        state.positionSide = Object.values(PositionSide).includes(loadedPosSide) ? loadedPosSide : PositionSide.NONE;
        state.positionAmount = (typeof loadedPosAmount === 'number' && Number.isFinite(loadedPosAmount) && loadedPosAmount >= 0) ? loadedPosAmount : null;
        state.entryPrice = (typeof loadedEntryPrice === 'number' && Number.isFinite(loadedEntryPrice) && loadedEntryPrice >= 0) ? loadedEntryPrice : null;

        // --- Restore TSL State ---
        const loadedTSL = loadedState.activeOrders?.stopLoss ?? loadedState.currentTSL; // Allow old key
        if (loadedTSL && typeof loadedTSL.price === 'number' && Number.isFinite(loadedTSL.price) && loadedTSL.price > 0 && typeof loadedTSL.id === 'string') {
            state.currentTSL = { price: loadedTSL.price, orderId: loadedTSL.id };
        } else {
            state.currentTSL = null;
        }

        // --- Ensure Consistency ---
        if (state.positionSide === PositionSide.NONE) {
            if (state.positionAmount !== null || state.entryPrice !== null || state.currentTSL !== null) {
                logger.warn("State inconsistency detected: Position side is NONE but amount/entry/TSL exist. Resetting to consistent FLAT state.");
                state.positionAmount = null;
                state.entryPrice = null;
                state.currentTSL = null;
            }
        } else { // If position exists
            if (state.positionAmount === null || state.positionAmount <= 0 || state.entryPrice === null || state.entryPrice <= 0) {
                logger.error(red(`CRITICAL State inconsistency: Position side is ${state.positionSide} but amount/entry is invalid. Resetting state to FLAT for safety.`));
                state.positionSide = PositionSide.NONE;
                state.positionAmount = null;
                state.entryPrice = null;
                state.currentTSL = null; // Also clear TSL if position state is corrupt
            }
        }

        logger.info(`State loaded successfully. Cycle: ${state.cycleCount}, Position: ${state.positionSide || 'NONE'}`);
        logger.debug("Loaded State Details:", state);

    } catch (error) {
        if (error.code === 'ENOENT') {
            logger.warn(`State file not found: ${persistenceFilePath}. Initializing fresh state.`);
        } else if (error instanceof SyntaxError) {
             logger.error(`Error parsing state file JSON: ${error.message}. Initializing fresh state.`);
        } else {
            logger.error(`Error loading state file: ${error.message}. Initializing fresh state.`, error.stack);
        }
        // Initialize with defaults if loading fails
        state = { positionSide: PositionSide.NONE, entryPrice: null, positionAmount: null, currentTSL: null, lastSignal: null, cycleCount: 0 };
        await saveState(); // Attempt to save the initial fresh state
    }
}

/**
 * Saves the current bot state to the persistence file atomically.
 * @returns {Promise<void>}
 */
async function saveState() {
    if (isShuttingDown) {
        logger.debug("Shutdown in progress, skipping state save.");
        return;
    }
    const tempFilePath = `${persistenceFilePath}.tmp_${process.pid}`; // Use PID for temp file uniqueness
    try {
        // Consolidate state to save
        const stateToSave = {
            state: {
                cycleCount: state.cycleCount,
                lastSignal: state.lastSignal,
            },
            currentPosition: {
                side: state.positionSide,
                size: state.positionAmount, // Use 'size' key for consistency
                entryPrice: state.entryPrice,
            },
            activeOrders: { // Standardize saved structure
                stopLoss: state.currentTSL ? { id: state.currentTSL.orderId, price: state.currentTSL.price } : null,
                takeProfit: null // Add TP tracking here if implemented
            },
            timestamp: new Date().toISOString()
        };

        const stateString = JSON.stringify(stateToSave, null, 2); // Pretty print JSON
        await fsPromises.writeFile(tempFilePath, stateString, 'utf8');
        await fsPromises.rename(tempFilePath, persistenceFilePath);
        logger.debug(`State saved successfully to ${persistenceFilePath}`);
    } catch (error) {
        logger.error(`Error saving state to ${persistenceFilePath}: ${error.message}`);
        // Attempt to clean up temp file
        try { await fsPromises.unlink(tempFilePath); } catch (cleanupError) { /* ignore cleanup error */ }
    }
}

// --- Termux SMS Dispatch ---
/**
 * Sends an SMS notification using Termux API if enabled.
 * @param {string} message The message content to send.
 */
function sendTermuxSms(message) {
    if (!CONFIG.TERMUX_NOTIFY || !CONFIG.NOTIFICATION_PHONE_NUMBER) {
        logger.debug('Termux SMS dormant.');
        return;
    }
    // Basic sanitization (remove potentially harmful shell chars)
    const sanitizedMessage = String(message).replace(/[`$!\\;&|<>*?()#~"]/g, "").substring(0, 160); // Max 160 chars for SMS
    const command = `termux-sms-send -n "${CONFIG.NOTIFICATION_PHONE_NUMBER}" "${sanitizedMessage}"`;

    logger.info(`Dispatching Termux SMS: "${sanitizedMessage}"`);
    exec(command, { timeout: 15000 }, (error, stdout, stderr) => {
        if (error) {
            logger.error(`Termux SMS dispatch failed: ${error.message}`);
            if (stderr) logger.error(`SMS stderr: ${stderr.trim()}`);
            return;
        }
        if (stderr) {
            logger.warn(`Termux SMS stderr whispers: ${stderr.trim()}`);
        }
        logger.info(`Termux SMS dispatch command executed.`);
        if (stdout.trim()) { // Log stdout only if it contains something
            logger.debug(`Termux SMS stdout echoes: ${stdout.trim()}`);
        }
    });
}

// --- Utility Charms ---
/**
 * Pauses execution for a specified duration.
 * @param {number} ms - Milliseconds to sleep.
 * @returns {Promise<void>}
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Wraps an async function call with retry logic using exponential backoff with jitter.
 * @param {Function} func - The async function to execute.
 * @param {string} funcName - Name of the function being called (for logging).
 * @returns {Promise<any>} - The result of the function if successful.
 * @throws {Error} - The last exception if all retries fail or a non-retryable error occurs.
 */
async function retryOnException(func, funcName) {
    let attempts = 0;
    const maxRetries = CONFIG.MAX_RETRIES;
    const initialDelay = CONFIG.RETRY_DELAY_MS;
    // Define retryable CCXT errors
    const retryableExceptions = [
        ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection, ccxt.RateLimitExceeded
    ];

    while (attempts <= maxRetries) {
        attempts++;
        try {
            return await func();
        } catch (e) {
            // Check if error is explicitly retryable OR an ExchangeError with specific retryable messages/codes
            const isRetryable = retryableExceptions.some(excType => e instanceof excType) ||
                                (e instanceof ccxt.ExchangeError && (
                                    e.message.includes('busy') ||
                                    e.message.includes('try again later') ||
                                    e.message.includes('operation timed out') ||
                                    e.message.includes('ret_code=10006') || // Bybit: request frequency too high
                                    e.message.includes('ret_code=10016') || // Bybit: service unavailable / internal error
                                    (e.httpStatus === 503) || // HTTP Service Unavailable
                                    (e.httpStatus === 504)    // HTTP Gateway Timeout
                                ));

            if (isRetryable && attempts <= maxRetries) {
                // Exponential backoff with jitter
                const delay = initialDelay * Math.pow(1.5, attempts - 1) * (1 + (Math.random() * 0.2 - 0.1)); // +/- 10% jitter
                const delaySec = (delay / 1000).toFixed(1);
                logger.warn(`[Retry] Attempt ${attempts}/${maxRetries + 1} for ${funcName} failed: ${e.constructor.name} - ${e.message}. Retrying in ${delaySec}s...`);
                await sleep(delay);
            } else {
                // Non-retryable error or max retries exceeded
                const reason = isRetryable ? `failed after ${attempts} attempts` : 'non-retryable error';
                logger.error(`[Error] ${funcName} ${reason}: ${e.constructor.name} - ${e.message}`);
                if (e instanceof Error && LOG_LEVELS[CONFIG.LOG_LEVEL] <= LOG_LEVELS.DEBUG) {
                     logger.debug(`${funcName} Error Stack:`, e.stack);
                }
                throw e; // Re-throw the error
            }
        }
    }
    // Should not be reached if maxRetries >= 0
    throw new Error(`${funcName} retry loop completed unexpectedly after ${attempts} attempts.`);
}


// --- Exchange Citadel Connection ---
const exchange = new ccxt.bybit({
    apiKey: CONFIG.BYBIT_API_KEY,
    secret: CONFIG.BYBIT_API_SECRET,
    options: {
        defaultType: 'linear', // Ensure linear contracts
        adjustForTimeDifference: true,
        recvWindow: 10000, // Request timeout window (milliseconds)
        // Bybit V5 specific options might be needed depending on CCXT version and account type
        // e.g., fetchPositionsRequiresSymbol: true
    },
    enableRateLimit: true, // Use CCXT's built-in rate limiter
});

if (CONFIG.DRY_RUN) {
    logger.warn(bold(magenta('DRY RUN MODE ACTIVE: Operating in simulation. No real orders will be placed.')));
    // Note: Setting sandbox mode might require specific testnet API keys.
    // exchange.setSandboxMode(true); // Uncomment if using Bybit testnet keys
}

// --- Market Divination (Indicator Calculation) ---

/**
 * Calculates all necessary technical indicators based on OHLCV data.
 * @param {Array<Array<number>>} candles - OHLCV candle data [[timestamp, open, high, low, close, volume], ...]
 * @returns {Promise<object|null>} An object containing the latest indicator values, or null if calculation fails.
 */
async function calculateIndicators(candles) {
    // Determine minimum candles needed based on longest indicator period + buffer
    const requiredCandlesMin = Math.max(
        CONFIG.ROOF_SLOW_EMA, // Ehlers Roofing Filter
        CONFIG.ST_EMA_PERIOD + CONFIG.ST_ATR_PERIOD, // SuperTrend (EMA basis + ATR lookback)
        CONFIG.FISHER_PERIOD, // Fisher Transform
        CONFIG.ADX_PERIOD, // ADX
        CONFIG.RANGE_ATR_PERIOD // ATR for range filter
    ) + 10; // Increased buffer for calculation stability

    if (!Array.isArray(candles) || candles.length < requiredCandlesMin) {
        logger.warn(`Insufficient candle data (${candles?.length || 0}) for indicator calculation (need at least ${requiredCandlesMin}).`);
        return null;
    }

    // Prepare input arrays defensively, filtering non-finite values
    const high = candles.map(c => c?.[OHLCV_INDEX.high]).filter(Number.isFinite);
    const low = candles.map(c => c?.[OHLCV_INDEX.low]).filter(Number.isFinite);
    const close = candles.map(c => c?.[OHLCV_INDEX.close]).filter(Number.isFinite);
    const hlc3 = candles.map(c => (c?.[OHLCV_INDEX.high] + c?.[OHLCV_INDEX.low] + c?.[OHLCV_INDEX.close]) / 3).filter(Number.isFinite);

    // Check if filtering removed too much data (reducing below required length)
    const minLengthAfterFilter = requiredCandlesMin - 20; // Allow some filtered candles
     if (high.length < minLengthAfterFilter || low.length < minLengthAfterFilter || close.length < minLengthAfterFilter || hlc3.length < minLengthAfterFilter) {
         logger.warn(`Insufficient finite candle data after filtering (High: ${high.length}, Low: ${low.length}, Close: ${close.length}, HLC3: ${hlc3.length}). Need approx ${minLengthAfterFilter}. Skipping indicators.`);
         return null;
     }

    try {
        // --- Ehlers Roofing Filter (on HLC3) ---
        logger.debug(`Calculating Roofing Filter (Fast EMA ${CONFIG.ROOF_FAST_EMA}, Slow EMA ${CONFIG.ROOF_SLOW_EMA}) on HLC3...`);
        const emaFastRoof = EMA.calculate({ period: CONFIG.ROOF_FAST_EMA, values: hlc3 });
        const emaSlowRoof = EMA.calculate({ period: CONFIG.ROOF_SLOW_EMA, values: hlc3 });
        if (!emaFastRoof || !emaSlowRoof || emaFastRoof.length === 0 || emaSlowRoof.length === 0) {
            logger.warn("Roofing Filter EMA calculation failed or yielded zero length."); return null;
        }
        // Align arrays and calculate the filter value (difference)
        const roofLength = Math.min(emaFastRoof.length, emaSlowRoof.length);
        const roofFilteredPrice = emaFastRoof.slice(-roofLength).map((fast, i) => fast - emaSlowRoof.slice(-roofLength)[i]);
        logger.debug(`Roofing Filter applied. Output length: ${roofFilteredPrice.length}`);

        // --- SuperTrend (Basis uses Roofed Price) ---
        logger.debug(`Calculating EMA(${CONFIG.ST_EMA_PERIOD}) on Roofed Price for ST Basis...`);
        const stBasisSmoothed = EMA.calculate({ period: CONFIG.ST_EMA_PERIOD, values: roofFilteredPrice });
        logger.debug(`Calculating ATR(${CONFIG.ST_ATR_PERIOD}) for SuperTrend (using regular HLC)...`);
        const atrSuperTrendArr = ATR.calculate({ period: CONFIG.ST_ATR_PERIOD, high, low, close });
        if (!stBasisSmoothed || !atrSuperTrendArr || stBasisSmoothed.length === 0 || atrSuperTrendArr.length === 0) {
             logger.warn('Failed to calculate Smoothed Basis or ATR for SuperTrend.'); return null;
        }
        // Align all inputs for custom SuperTrend calculation
        const stInputLength = Math.min(stBasisSmoothed.length, atrSuperTrendArr.length, high.length, low.length, close.length);
        if (stInputLength === 0) { logger.warn('SuperTrend input alignment resulted in zero length.'); return null; }
        const superTrendInput = {
            high: high.slice(-stInputLength),
            low: low.slice(-stInputLength),
            close: close.slice(-stInputLength),
            atr: atrSuperTrendArr.slice(-stInputLength),
            basis: stBasisSmoothed.slice(-stInputLength),
            multiplier: CONFIG.ST_MULTIPLIER,
        };
        logger.debug('Invoking custom SuperTrend calculation with roofed basis...');
        const stResult = calculateCustomSuperTrend(superTrendInput); // Use the custom function
        if (!stResult || stResult.length === 0) { logger.warn('Custom SuperTrend calculation failed or yielded zero length.'); return null; }

        // --- Ehlers Inverse Fisher Transform (on High/Low) ---
        logger.debug(`Calculating Fisher Transform(${CONFIG.FISHER_PERIOD})...`);
        const fisherInput = { high, low, period: CONFIG.FISHER_PERIOD };
        const fisherResultArr = InverseFisherTransform.calculate(fisherInput); // Returns [{fisher: val, signal: val}]
        if (!fisherResultArr || fisherResultArr.length === 0) { logger.warn('Fisher Transform calculation failed or yielded zero length.'); return null; }
        // Extract fisher and signal lines separately
        const fisherValues = fisherResultArr.map(f => f.fisher);
        const fisherSignalValues = fisherResultArr.map(f => f.signal);

        // --- ADX (Average Directional Index) ---
        logger.debug(`Calculating ADX(${CONFIG.ADX_PERIOD})...`);
        const adxInput = { high, low, close, period: CONFIG.ADX_PERIOD };
        const adxResultArr = ADX.calculate(adxInput); // Returns [{adx: val, pdi: val, mdi: val}]
         if (!adxResultArr || adxResultArr.length === 0) { logger.warn('ADX calculation failed or yielded zero length.'); return null; }

        // --- ATR for Range Filter ---
        logger.debug(`Calculating ATR(${CONFIG.RANGE_ATR_PERIOD}) for Range Filter...`);
        const atrRangeResultArr = ATR.calculate({ period: CONFIG.RANGE_ATR_PERIOD, high, low, close });
         if (!atrRangeResultArr || atrRangeResultArr.length === 0) { logger.warn('ATR Range calculation failed or yielded zero length.'); return null; }

        // --- Extract Latest Valid Values ---
        // Helper to find the last finite number or valid object in an array
        const findLastValid = (arr, checkFn = (v) => Number.isFinite(v)) => {
            for (let i = arr.length - 1; i >= 0; i--) {
                if (arr[i] !== undefined && arr[i] !== null && checkFn(arr[i])) {
                    return { value: arr[i], index: i };
                }
            }
            return { value: null, index: -1 };
        };

        const { value: latestST, index: stIdx } = findLastValid(stResult, v => typeof v === 'object' && Number.isFinite(v.value));
        const { value: latestFisher, index: fisherIdx } = findLastValid(fisherValues);
        const { value: latestFisherSignal, index: fisherSigIdx } = findLastValid(fisherSignalValues);
        const { value: latestAdx, index: adxIdx } = findLastValid(adxResultArr, v => typeof v === 'object' && Number.isFinite(v.adx));
        const { value: latestAtrRange, index: atrRangeIdx } = findLastValid(atrRangeResultArr);
        const { value: latestAtrST, index: atrStIdx } = findLastValid(atrSuperTrendArr);
        const { value: latestClosePrice, index: closeIdx } = findLastValid(close); // Use original close array

        // Check if all latest values could be found
        if (stIdx < 0 || fisherIdx < 0 || fisherSigIdx < 0 || adxIdx < 0 || atrRangeIdx < 0 || atrStIdx < 0 || closeIdx < 0) {
             logger.warn("Could not extract one or more latest valid indicator values.");
             logger.debug(`Extraction Indices: ST=${stIdx}, Fisher=${fisherIdx}, FisherSig=${fisherSigIdx}, ADX=${adxIdx}, ATR_Range=${atrRangeIdx}, ATR_ST=${atrStIdx}, Close=${closeIdx}`);
             return null;
        }

        // --- Check for Index Alignment ---
        // Check if indexes are reasonably close (allow some lag due to different calculation warmups)
        const allIndices = [stIdx, fisherIdx, fisherSigIdx, adxIdx, atrRangeIdx, atrStIdx, closeIdx];
        const maxIdx = Math.max(...allIndices);
        const minIdx = Math.min(...allIndices);
        const indexSpread = maxIdx - minIdx;
        if (indexSpread > 5) { // Allow a lag of up to 5 periods between indicators
            logger.warn(`Indicator calculation resulted in significant index divergence (Min: ${minIdx}, Max: ${maxIdx}, Spread: ${indexSpread}). Results may be misaligned.`);
            // Optional: return null if strict alignment is critical for the strategy
            // return null;
        } else {
            logger.debug(`Indicator indices aligned within tolerance (Spread: ${indexSpread}).`);
        }

        // --- Get Previous Fisher Values Safely ---
        // Use the found indices to access previous values, checking bounds
        const prevFisher = (fisherIdx > 0 && fisherIdx < fisherValues.length) ? fisherValues[fisherIdx - 1] : NaN;
        const prevFisherSignal = (fisherSigIdx > 0 && fisherSigIdx < fisherSignalValues.length) ? fisherSignalValues[fisherSigIdx - 1] : NaN;

        // --- Final Validation of Extracted Values ---
        if (!latestST || !Number.isFinite(latestST.value) ||
            !latestAdx || !Number.isFinite(latestAdx.adx) || !Number.isFinite(latestAdx.pdi) || !Number.isFinite(latestAdx.mdi) ||
            !Number.isFinite(latestFisher) || !Number.isFinite(latestFisherSignal) ||
            !Number.isFinite(prevFisher) || !Number.isFinite(prevFisherSignal) || // Check previous values too
            !Number.isFinite(latestAtrRange) || !Number.isFinite(latestAtrST) ||
            !Number.isFinite(latestClosePrice))
        {
            logger.warn("Could not divine all latest indicator values (some are NaN or invalid after extraction).");
            logger.debug(`Latest Values: ST=${inspect(latestST)}, Fisher=${latestFisher}, FisherSig=${latestFisherSignal}, ADX=${inspect(latestAdx)}, ATR_Range=${latestAtrRange}, ATR_ST=${latestAtrST}, Close=${latestClosePrice}`);
            return null;
        }

        logger.debug(`Latest Indicators: Price=${latestClosePrice.toFixed(4)}, ST=${latestST.value.toFixed(4)} (${latestST.direction}), Fisher=${latestFisher.toFixed(3)}, Sig=${latestFisherSignal.toFixed(3)}, ADX=${latestAdx.adx.toFixed(2)}, ATR_Range=${latestAtrRange.toFixed(4)}, ATR_ST=${latestAtrST.toFixed(4)}`);

        // Return the structured indicator data
        return {
            superTrendValue: latestST.value,
            superTrendDirection: latestST.direction,
            fisherValue: latestFisher,
            fisherSignalValue: latestFisherSignal,
            prevFisherValue: prevFisher,
            prevFisherSignalValue: prevFisherSignal,
            adx: latestAdx.adx,
            pdi: latestAdx.pdi,
            mdi: latestAdx.mdi,
            atrRange: latestAtrRange,
            atrSuperTrend: latestAtrST,
            price: latestClosePrice
        };

    } catch (error) {
        logger.error(`Error during indicator calculation: ${error.message}`);
        console.error(error); // Log stack trace for debugging
        return null;
    }
}

// --- Custom SuperTrend Calculation (using provided basis) ---
/**
 * Calculates SuperTrend based on provided basis, ATR, and multiplier.
 * This version is stateful and iterates through the provided data.
 * @param {object} input - Input data for SuperTrend calculation.
 * @param {number[]} input.high - Array of high prices.
 * @param {number[]} input.low - Array of low prices.
 * @param {number[]} input.close - Array of close prices.
 * @param {number[]} input.atr - Array of ATR values.
 * @param {number[]} input.basis - Array of basis values (e.g., smoothed roofed price).
 * @param {number} input.multiplier - ATR multiplier.
 * @returns {Array<{value: number, direction: 'up' | 'down'}>} Array of SuperTrend objects with value and direction. Returns empty array on critical failure.
 */
function calculateCustomSuperTrend(input) {
    const { high, low, close, atr, basis, multiplier } = input;

    // Validate inputs
    const n = basis?.length;
    if (!n || n !== high?.length || n !== low?.length || n !== close?.length || n !== atr?.length) {
        logger.error("CustomSuperTrend Error: Input arrays have mismatched lengths or are empty.");
        return [];
    }
    if (typeof multiplier !== 'number' || multiplier <= 0) {
         logger.error(`CustomSuperTrend Error: Invalid multiplier ${multiplier}.`);
         return [];
    }

    const trend = new Array(n);
    let currentDirection = 'up'; // Initial assumption (will be corrected)
    let currentStValue = NaN;

    for (let i = 0; i < n; i++) {
        // Validate data for the current index
        if (!Number.isFinite(basis[i]) || !Number.isFinite(atr[i]) || !Number.isFinite(close[i]) || atr[i] <= 0) {
            logger.warn(`CustomSuperTrend Warning: Invalid data at index ${i} (Basis=${basis[i]}, ATR=${atr[i]}, Close=${close[i]}). Carrying over previous state.`);
            // Carry over the previous state if data is invalid
            const prevTrend = i > 0 ? trend[i - 1] : { value: NaN, direction: 'up' }; // Use initial if first bar
            trend[i] = { value: prevTrend.value, direction: prevTrend.direction };
            continue;
        }

        // Calculate basic upper/lower bands for this index
        const upperBandBasic = basis[i] + multiplier * atr[i];
        const lowerBandBasic = basis[i] - multiplier * atr[i];

        // Get previous state (trend direction and ST value)
        let prevDirection = (i > 0 && trend[i - 1]) ? trend[i - 1].direction : 'up'; // Default to 'up' if no previous
        let prevStValue = (i > 0 && trend[i - 1] && Number.isFinite(trend[i - 1].value)) ? trend[i - 1].value : NaN;

        // Determine current ST value and direction based on previous state and bands
        let nextStValue = NaN;
        let nextDirection = prevDirection;

        if (prevDirection === 'up') {
            // Trail stop up using the lower band, ensuring it doesn't decrease
            nextStValue = Math.max(lowerBandBasic, isNaN(prevStValue) ? -Infinity : prevStValue);
            if (close[i] < nextStValue) { // Price crossed below trailing stop
                nextDirection = 'down';
                nextStValue = upperBandBasic; // Start new down-trend stop at upper band
            }
        } else { // prevDirection === 'down'
            // Trail stop down using the upper band, ensuring it doesn't increase
            nextStValue = Math.min(upperBandBasic, isNaN(prevStValue) ? Infinity : prevStValue);
            if (close[i] > nextStValue) { // Price crossed above trailing stop
                nextDirection = 'up';
                nextStValue = lowerBandBasic; // Start new up-trend stop at lower band
            }
        }

        // Final validation and assignment
        if (!Number.isFinite(nextStValue)) {
             logger.warn(`CustomSuperTrend Warning: Calculated ST value at index ${i} is non-finite. Setting to NaN.`);
             nextStValue = NaN;
             // If ST value is NaN, should direction reset? Let's keep the calculated direction.
        }
        trend[i] = { value: nextStValue, direction: nextDirection };
    }
    return trend;
}


// --- The Art of Trading (Core Logic) ---

/**
 * Checks entry/exit conditions based on indicators and manages orders.
 * @param {ccxt.Market} market - The market object from ccxt.
 * @param {object} indicators - The calculated indicator values.
 * @returns {Promise<void>}
 */
async function checkAndPlaceOrder(market, indicators) {
    const {
        superTrendValue, superTrendDirection,
        fisherValue, fisherSignalValue, prevFisherValue, prevFisherSignalValue,
        adx, atrRange, atrSuperTrend, price
    } = indicators;

    // --- Filters ---
    const adxFilterPassed = adx >= CONFIG.MIN_ADX_LEVEL && adx <= CONFIG.MAX_ADX_LEVEL;
    const atrValuePercent = price > 0 ? (atrRange / price) * 100 : 0;
    const atrFilterPassed = atrValuePercent >= CONFIG.MIN_ATR_PERCENTAGE;

    let filterReason = '';
    if (!adxFilterPassed) filterReason += `ADX (${adx.toFixed(2)}) outside [${CONFIG.MIN_ADX_LEVEL}-${CONFIG.MAX_ADX_LEVEL}]`;
    if (!atrFilterPassed) filterReason += `${filterReason ? '; ' : ''}ATR (${atrValuePercent.toFixed(3)}%) < ${CONFIG.MIN_ATR_PERCENTAGE}%`;

    if (!adxFilterPassed || !atrFilterPassed) {
        logger.info(`Filters block action: ${filterReason}.`);
        if (state.positionSide === PositionSide.NONE && state.lastSignal) {
             logger.debug(`Clearing lastSignal (${state.lastSignal}) due to filter block.`);
             state.lastSignal = null;
             await saveState(); // Save cleared signal state
        }
        return; // Stop processing if filters fail
    }
    logger.info(green(`Filters Passed: ADX=${adx.toFixed(2)}, ATR%=${atrValuePercent.toFixed(3)}%`));

    // --- Signal Conditions ---
    // Fisher Cross: Current value crosses its signal line
    const fisherBullishCross = fisherValue > fisherSignalValue && prevFisherValue <= prevFisherSignalValue;
    const fisherBearishCross = fisherValue < fisherSignalValue && prevFisherValue >= prevFisherSignalValue;

    // Entry Conditions: SuperTrend direction aligns with Fisher cross
    const longCondition = superTrendDirection === 'up' && fisherBullishCross;
    const shortCondition = superTrendDirection === 'down' && fisherBearishCross;

    // Exit Conditions: SuperTrend flips direction (primary exit trigger)
    const closeLongCondition = superTrendDirection === 'down';
    const closeShortCondition = superTrendDirection === 'up';

    // Log signal components
    logger.info(`Signals: Px=${magenta(price.toFixed(market.precision.price))}, ST=${magenta(superTrendValue.toFixed(market.precision.price))} ${superTrendDirection === 'up' ? green('Up') : red('Down')}`);
    logger.info(`Fisher: Val=${magenta(fisherValue.toFixed(3))}, Sig=${magenta(fisherSignalValue.toFixed(3))} | Cross: ${fisherBullishCross ? green('BULL') : fisherBearishCross ? red('BEAR') : gray('NONE')}`);

    const currentSide = state.positionSide;
    let orderAmount = 0;

    // --- Calculate Order Amount ---
    if (price > 0) {
        try {
            // Calculate amount based on fixed USD order size
            const rawAmount = CONFIG.ORDER_AMOUNT_USD / price;
            orderAmount = parseFloat(exchange.amountToPrecision(CONFIG.SYMBOL, rawAmount));
        } catch (e) {
             logger.error(`Error calculating order amount precision: ${e.message}`);
             return; // Cannot proceed without valid amount
        }
    } else {
        logger.error("Cannot calculate order amount: Invalid current price (0 or less).");
        return;
    }

     // --- Validate Order Amount against Market Limits ---
     const minAmount = market.limits?.amount?.min;
     if (minAmount !== undefined && orderAmount < minAmount) {
         logger.error(`Calculated order amount ${orderAmount} is less than minimum ${minAmount}. Cannot place order. Increase ORDER_AMOUNT_USD or check symbol.`);
         return;
     }
     // Optional: Check maxAmount if needed


    // --- Position Management ---

    // 1. Check for Exits based on SuperTrend flip first
    if (currentSide === PositionSide.LONG && closeLongCondition) {
        logger.info(yellow('Exit Signal: SuperTrend flipped DOWN. Closing LONG.'));
        await closePosition(market, `SuperTrend flipped DOWN`);
        // Exit function after closing action
        return;
    } else if (currentSide === PositionSide.SHORT && closeShortCondition) {
        logger.info(yellow('Exit Signal: SuperTrend flipped UP. Closing SHORT.'));
        await closePosition(market, `SuperTrend flipped UP`);
        // Exit function after closing action
        return;
    }

    // 2. Check for Entries if flat (and no exit occurred)
    if (currentSide === PositionSide.NONE) {
        if (longCondition && state.lastSignal !== 'long') {
            // Prevent immediate re-entry if last signal was also long but filtered/failed
            await openPosition(market, PositionSide.LONG, orderAmount, price, atrSuperTrend);
            state.lastSignal = 'long'; // Record the signal that led to the attempt
            await saveState();
        } else if (shortCondition && state.lastSignal !== 'short') {
            await openPosition(market, PositionSide.SHORT, orderAmount, price, atrSuperTrend);
            state.lastSignal = 'short';
            await saveState();
        } else {
             // No entry condition met, or trying to re-enter immediately in same direction
             if (state.lastSignal) {
                 if ((longCondition && state.lastSignal === 'long') || (shortCondition && state.lastSignal === 'short')) {
                     logger.debug(`Entry condition met (${longCondition ? 'long' : 'short'}), but lastSignal was the same. Preventing immediate re-entry. Clearing lastSignal.`);
                 } else {
                     logger.debug(`No entry condition met. Clearing lastSignal (${state.lastSignal}).`);
                 }
                 state.lastSignal = null;
                 await saveState(); // Save cleared signal state
             } else {
                  logger.debug("No entry conditions met.");
             }
        }
    } else { // Already in a position and no exit signal from SuperTrend
        if (state.lastSignal) {
            // Clear lastSignal if we are already in a position
            logger.debug(`In position (${state.positionSide}). Clearing previous lastSignal (${state.lastSignal}).`);
            state.lastSignal = null;
            await saveState();
        }
        // TSL update happens in the main runBot loop *after* this check
        logger.debug(`Holding ${currentSide} position. TSL will be managed.`);
    }
}


// --- Opening a New Position ---

/**
 * Opens a new long or short position with an initial Stop Loss attached.
 * @param {ccxt.Market} market - The market object from ccxt.
 * @param {PositionSide.LONG | PositionSide.SHORT} positionSide - The side of the order ('long' or 'short').
 * @param {number} amount - The amount of the asset to trade (already precision adjusted).
 * @param {number} entryPrice - The approximate current price for logging and SL calculation.
 * @param {number} atrValue - The current ATR value (from SuperTrend calc) for initial SL calculation.
 * @returns {Promise<void>}
 */
async function openPosition(market, positionSide, amount, entryPrice, atrValue) {
    logger.trade(`${bold(positionSide.toUpperCase())} entry signal confirmed. Attempting entry...`);
    const orderSide = positionSide === PositionSide.LONG ? 'buy' : 'sell';

    // --- Calculate Initial Stop Loss ---
    const slMultiplier = CONFIG.INITIAL_SL_ATR_MULTIPLIER;
    if (isNaN(slMultiplier) || slMultiplier <= 0) {
        logger.error(red("INITIAL_SL_ATR_MULTIPLIER is invalid or missing. Cannot open position without initial SL."));
        return;
    }
    if (isNaN(atrValue) || atrValue <= 0) {
        logger.error(red(`Invalid ATR value (${atrValue}) for initial SL calculation. Cannot open position.`));
        return;
    }

    let initialSlPrice;
    if (positionSide === PositionSide.LONG) {
        initialSlPrice = entryPrice - slMultiplier * atrValue;
    } else { // SHORT
        initialSlPrice = entryPrice + slMultiplier * atrValue;
    }

    // Apply precision and validate SL price
    let slPriceFormatted;
    try {
        slPriceFormatted = parseFloat(exchange.priceToPrecision(CONFIG.SYMBOL, initialSlPrice));
        if (isNaN(slPriceFormatted) || slPriceFormatted <= 0) throw new Error("Calculated SL price is invalid (NaN or <= 0).");
        // Validate SL logic: SL should be below entry for long, above for short
        if ((positionSide === PositionSide.LONG && slPriceFormatted >= entryPrice) ||
            (positionSide === PositionSide.SHORT && slPriceFormatted <= entryPrice)) {
             throw new Error(`Calculated SL price ${slPriceFormatted} is illogical relative to entry ${entryPrice}. Check ATR/Multiplier.`);
        }
    } catch (e) {
         logger.error(red(`Error calculating or validating initial SL price: ${e.message}. Raw SL=${initialSlPrice}. Cannot open position.`));
         return;
    }

    // --- Prepare Order Parameters with Stop Loss ---
    // Using Bybit V5 parameters for attaching SL to market order
    const params = {
        'category': 'linear',
        'positionIdx': 0, // 0 for one-way mode
        'stopLoss': slPriceFormatted.toString(), // SL trigger price (string format often preferred)
        'slTriggerBy': 'LastPrice', // Or MarkPrice, IndexPrice - make configurable? Use LastPrice for now.
        'tpslMode': 'Full' // Apply SL to the entire position triggered by this order
        // Optional: Add TP params here too if desired: 'takeProfit': tpPriceFormatted, 'tpTriggerBy': 'LastPrice'
    };

    logger.info(`Placing ${positionSide.toUpperCase()} market order: Amount=${amount.toFixed(market.precision.amount)}, EntryEst~=${entryPrice.toFixed(market.precision.price)}, Initial SL=${slPriceFormatted} (ATR_ST=${atrValue.toFixed(4)})`);
    logger.debug("Order Params:", params);

    // --- Execute Order ---
    if (!CONFIG.DRY_RUN) {
        try {
            // Place the market order with attached SL
            const orderResponse = await retryOnException(
                async () => await exchange.createMarketOrder(CONFIG.SYMBOL, orderSide, amount, undefined, params),
                'createMarketOrder (Entry with SL)'
            );

            logger.info(`Market order sent: ID ${orderResponse.id}, Side: ${orderResponse.side}, Amount: ${orderResponse.amount}, Avg Price: ${orderResponse.average ?? 'N/A'}`);
            logger.info("Waiting briefly for position confirmation...");
            await sleep(5000); // Wait 5 seconds for exchange state update

            // --- Verify Position Opened Correctly ---
            const livePosition = await fetchCurrentPosition(market.id, true); // Force fetch latest state
            const sizeTolerance = market.limits.amount.min / 10 || 1e-9; // Tolerance for size comparison

            if (livePosition && livePosition.side === positionSide && Math.abs(parseFloat(livePosition.contracts) - amount) <= sizeTolerance) {
                // --- Update State with Actual Position Details ---
                state.positionSide = positionSide;
                state.entryPrice = parseFloat(livePosition.entryPrice);
                state.positionAmount = parseFloat(livePosition.contracts); // Use actual contracts amount
                state.currentTSL = null; // Clear any old TSL state initially
                state.lastSignal = null; // Clear signal after successful entry

                logger.trade(green(`POSITION CONFIRMED: ${state.positionSide.toUpperCase()} ${state.positionAmount} ${CONFIG.SYMBOL} @ ${state.entryPrice.toFixed(market.precision.price)}`));

                // --- Initialize TSL Based on Current Conditions ---
                // Fetch fresh ATR for initial TSL placement (if TSL multiplier > 0)
                if (CONFIG.TSL_ATR_MULTIPLIER > 0) {
                    logger.info("Initializing Trailing Stop Loss ward...");
                    const freshCandlesForTsl = await safeFetchOHLCV(CONFIG.TIMEFRAME, Math.max(CONFIG.ST_ATR_PERIOD, 50) + 10);
                    if (freshCandlesForTsl) {
                        const atrValues = ATR.calculate({ period: CONFIG.ST_ATR_PERIOD, high: freshCandlesForTsl.map(c => c[OHLCV_INDEX.high]), low: freshCandlesForTsl.map(c => c[OHLCV_INDEX.low]), close: freshCandlesForTsl.map(c => c[OHLCV_INDEX.close]) });
                        const latestAtrForTsl = atrValues?.length > 0 ? atrValues[atrValues.length - 1] : NaN;
                        if (!isNaN(latestAtrForTsl) && latestAtrForTsl > 0) {
                            await updateTrailingStopLoss(market, latestAtrForTsl); // Place the initial TSL order (will cancel the initial one)
                        } else {
                            logger.warn("Could not calculate fresh ATR for initial TSL placement. Initial SL remains, TSL may be delayed.");
                            // Store the initial SL as the current TSL state for now
                            state.currentTSL = { price: slPriceFormatted, orderId: null }; // No ID for server-side SL
                        }
                    } else {
                        logger.warn("Could not fetch fresh candles for initial TSL. Initial SL remains, TSL may be delayed.");
                        state.currentTSL = { price: slPriceFormatted, orderId: null };
                    }
                } else {
                     logger.info("Trailing Stop Loss is disabled (TSL_ATR_MULTIPLIER <= 0). Initial SL remains.");
                     state.currentTSL = { price: slPriceFormatted, orderId: null }; // Store initial SL price
                }

            } else {
                logger.error(red(`POSITION CONFIRMATION FAILED for ${positionSide} ${CONFIG.SYMBOL} entry.`));
                logger.error(` -> Expected: Side=${positionSide}, Amount~=${amount}`);
                logger.error(` -> Found: Side=${livePosition?.side || 'N/A'}, Amount=${livePosition?.contracts || 'N/A'}`);
                logger.error(` -> Original Order ID: ${orderResponse.id}, Status: ${orderResponse.status}`);
                sendTermuxSms(`BOT ALERT: Failed confirmation ${positionSide} ${CONFIG.SYMBOL} entry. Order ${orderResponse.id}. Check Exchange!`);
                // Attempt to cancel the SL order that might have been placed (difficult without ID)
                // Reset state as entry likely failed or is inconsistent.
                state.positionSide = PositionSide.NONE;
                state.entryPrice = null;
                state.positionAmount = null;
                state.currentTSL = null;
                state.lastSignal = null; // Clear signal on failure
            }
        } catch (e) {
            logger.error(`Error placing ${positionSide} order: ${e.message}`);
            if (e instanceof ccxt.InsufficientFunds) {
                logger.error(red("Insufficient funds to place order. Check account balance."));
                sendTermuxSms(`BOT ERROR: Insufficient funds for ${positionSide} order ${CONFIG.SYMBOL}.`);
            } else if (e instanceof ccxt.InvalidOrder) {
                logger.error(red(`Invalid order parameters: ${e.message}. Check limits, precision, or SL/TP placement.`));
                sendTermuxSms(`BOT ERROR: Invalid order params for ${positionSide} order ${CONFIG.SYMBOL}. ${e.message.substring(0, 80)}`);
            } else {
                logger.error(red(`Exchange error placing order: ${e.constructor.name} - ${e.message}`));
                console.error(e); // Log full stack trace for unexpected errors
                sendTermuxSms(`BOT ERROR: Failed ${positionSide} order ${CONFIG.SYMBOL}. ${e.message.substring(0, 80)}`);
            }
            // Ensure state remains flat on failure
            state.positionSide = PositionSide.NONE;
            state.entryPrice = null;
            state.positionAmount = null;
            state.currentTSL = null;
            state.lastSignal = null; // Clear signal on failure
            // Do not proceed after failed entry attempt
        }
    } else { // Dry Run
        // Simulate successful entry
        state.positionSide = positionSide;
        state.entryPrice = entryPrice; // Use estimated entry price
        state.positionAmount = amount;
        // Simulate the initial SL being set (store price, no real ID)
        state.currentTSL = { price: slPriceFormatted, orderId: `dry_sl_${Date.now()}` };
        state.lastSignal = null; // Clear signal after simulated entry

        logger.dryRun(`Simulated ${positionSide.toUpperCase()} entry: Amt=${amount.toFixed(market.precision.amount)}, Entry=${entryPrice.toFixed(market.precision.price)}, Initial SL=${slPriceFormatted}`);
        sendTermuxSms(`BOT DRY RUN: Simulated ${positionSide} entry ${CONFIG.SYMBOL} @ ${entryPrice.toFixed(market.precision.price)}`);
    }
    // Save state after successful live entry or dry run simulation
    await saveState();
}


// --- Closing the Current Position ---

/**
 * Closes the currently open position using a market order.
 * Cancels any tracked TSL order first.
 * @param {ccxt.Market} market - The market object from ccxt.
 * @param {string} reason - The reason for closing the position (for logging).
 * @returns {Promise<void>}
 */
async function closePosition(market, reason) {
    const positionSide = state.positionSide;
    const positionAmount = state.positionAmount;

    if (positionSide === PositionSide.NONE || !positionAmount || positionAmount <= 0) {
        logger.warn(`Close attempt (${reason}), but no position found in memory or amount is invalid.`);
        // Ensure state is consistent if called erroneously
        if (state.positionSide !== PositionSide.NONE) {
            logger.warn("Resetting inconsistent state to FLAT.");
            state.positionSide = PositionSide.NONE;
            state.entryPrice = null;
            state.positionAmount = null;
            await cancelAllSLTPOrders("Clearing orders on inconsistent close call"); // Use the correct helper name
            await saveState();
        }
        return;
    }

    const closeSide = positionSide === PositionSide.LONG ? 'sell' : 'buy';
    const amount = positionAmount; // Use the stored position amount

    logger.trade(`Closing ${positionSide.toUpperCase()} position. Reason: ${reason}`);
    logger.info(`Placing CLOSE order: Side=${closeSide}, Amount=${amount}`);

    // --- CRITICAL: Cancel existing TSL order before closing position ---
    await cancelAllSLTPOrders(`Closing position (${reason})`); // Use helper function

    if (!CONFIG.DRY_RUN) {
        try {
            // Place a market order to close the position
            const params = {
                reduceOnly: true, // CRITICAL: Ensure it only reduces/closes the position
                'category': 'linear', // Required for Bybit V5
                'positionIdx': 0, // 0 for one-way mode
             };
            const orderResponse = await retryOnException(
                async () => await exchange.createMarketOrder(CONFIG.SYMBOL, closeSide, amount, undefined, params),
                'createMarketOrder (Close)'
            );
            logger.info(`Close order sent: ID ${orderResponse.id}, Side: ${orderResponse.side}, Amount: ${orderResponse.amount}, Avg Price: ${orderResponse.average ?? 'N/A'}`);
            sendTermuxSms(`BOT TRADE: Closed ${positionSide} ${CONFIG.SYMBOL}. Reason: ${reason}`);

            // Optimistically reset state, verification happens next cycle
            state.positionSide = PositionSide.NONE;
            state.entryPrice = null;
            state.positionAmount = null;
            state.currentTSL = null; // Ensure TSL state is cleared
            state.lastSignal = null; // Clear last signal after closing

        } catch (e) {
            logger.error(`Error closing ${positionSide} position: ${e.message}`);
            // Handle cases where the position might already be closed
            if (e instanceof ccxt.ExchangeError && (e.message.includes("reduce-only") || e.message.includes("position size is zero") || e.message.includes("ret_code=110025") || e.message.includes("ret_code=30017"))) {
                 logger.warn(`Position might have been already closed or reduceOnly failed. Verifying...`);
                 await sleep(1000); // Wait a bit before verifying
                 const livePosition = await fetchCurrentPosition(market.id, true); // Force fetch
                 if (livePosition && parseFloat(livePosition.contracts) > 0) {
                      logger.error(red(`POSITION STILL EXISTS after close error! Manual check required! Position: ${inspect(livePosition)}`));
                      sendTermuxSms(`BOT CRITICAL: Failed closing ${positionSide} ${CONFIG.SYMBOL} but position persists! Check Exchange!`);
                      // Do NOT reset state if position still exists
                      return; // Stop state reset
                 } else {
                      logger.info("Position confirmed closed after error.");
                      // Reset state as position is confirmed gone
                      state.positionSide = PositionSide.NONE;
                      state.entryPrice = null;
                      state.positionAmount = null;
                      state.currentTSL = null;
                      state.lastSignal = null;
                 }
            } else {
                console.error(e); // Log full stack trace for other errors
                sendTermuxSms(`BOT ERROR: Failed closing ${positionSide} ${CONFIG.SYMBOL}. ${e.message.substring(0,80)}`);
                // Don't reset state yet, verification needed in the next cycle's reconciliation
                return; // Stop state reset
            }
        }
    } else { // Dry Run
        logger.dryRun(`Simulated CLOSE ${positionSide.toUpperCase()}: Side=${closeSide}, Amt=${amount}. Reason: ${reason}`);
        sendTermuxSms(`BOT DRY RUN: Simulated closing ${positionSide} ${CONFIG.SYMBOL}. Reason: ${reason}`);
        // Reset state in dry run
        state.positionSide = PositionSide.NONE;
        state.entryPrice = null;
        state.positionAmount = null;
        state.currentTSL = null;
        state.lastSignal = null;
    }

    // Save state only after successful close (or simulated close, or confirmed closure after error)
    await saveState();
}


// --- The Guardian Ward (Trailing Stop Loss) ---

/**
 * Updates the Trailing Stop Loss (TSL) order.
 * Calculates potential new SL based on current price and ATR.
 * If the new SL is better than the existing one, cancels the old SL and places a new one.
 * Manages the `state.currentTSL` object { price: number, orderId: string | null }.
 * @param {ccxt.Market} market - The market object from ccxt.
 * @param {number} atrValue - The current ATR value (from SuperTrend calc) for TSL calculation.
 * @returns {Promise<void>}
 */
async function updateTrailingStopLoss(market, atrValue) {
    // --- Pre-checks ---
    if (state.positionSide === PositionSide.NONE || !state.entryPrice || !state.positionAmount) {
        logger.debug("TSL ward sleeps (no position).");
        // Ensure TSL state is clear if erroneously present
        if (state.currentTSL) {
             logger.warn("Clearing inconsistent TSL state while position is NONE.");
             state.currentTSL = null;
             await saveState();
        }
        return;
    }
    if (CONFIG.TSL_ATR_MULTIPLIER <= 0) {
        logger.debug("TSL disabled (TSL_ATR_MULTIPLIER <= 0).");
        return;
    }
    if (isNaN(atrValue) || atrValue <= 0) {
        logger.warn(`Invalid ATR (${atrValue}) provided for TSL calculation. Skipping TSL update.`);
        return;
    }

    // --- Fetch Current Price ---
    let currentPrice;
    try {
        const ticker = await retryOnException(async () => await exchange.fetchTicker(CONFIG.SYMBOL, { 'category': 'linear' }), 'fetchTicker (TSL)');
        if (!ticker || !ticker.last || ticker.last <= 0) {
            logger.warn("Could not fetch valid current price from ticker for TSL update.");
            return;
        }
        currentPrice = ticker.last;
    } catch (e) {
         logger.warn(`Failed to fetch ticker for TSL update: ${e.message}`);
         return;
    }

    // --- Calculate Potential New TSL Price ---
    const multiplier = CONFIG.TSL_ATR_MULTIPLIER;
    let potentialNewTslPrice;
    if (state.positionSide === PositionSide.LONG) {
        potentialNewTslPrice = currentPrice - atrValue * multiplier;
    } else { // SHORT
        potentialNewTslPrice = currentPrice + atrValue * multiplier;
    }

    // Apply precision and validate
    let newTslPriceFormatted;
    try {
        newTslPriceFormatted = parseFloat(exchange.priceToPrecision(CONFIG.SYMBOL, potentialNewTslPrice));
        if (isNaN(newTslPriceFormatted) || newTslPriceFormatted <= 0) throw new Error("Calculated TSL price is invalid.");
    } catch (e) {
         logger.error(`Error applying TSL price precision: ${e.message}. Raw=${potentialNewTslPrice}`);
         return;
    }

    // --- Determine if Update is Needed ---
    const currentSlPrice = state.currentTSL?.price; // Price from our state
    let shouldUpdate = false;
    const priceTolerance = market.info?.tickSize * 2 || atrValue * 0.05; // Use 2 ticks or 5% ATR as buffer

    if (currentSlPrice === null || currentSlPrice === undefined) {
        // No active SL tracked: Place initial TSL only if profitable vs entry
         if ((state.positionSide === PositionSide.LONG && newTslPriceFormatted > state.entryPrice + priceTolerance) ||
             (state.positionSide === PositionSide.SHORT && newTslPriceFormatted < state.entryPrice - priceTolerance)) {
             logger.info(cyan(`TSL: Placing initial profitable TSL at ${newTslPriceFormatted.toFixed(market.precision.price)}.`));
             shouldUpdate = true;
         } else {
             logger.debug(`TSL: Potential initial SL ${newTslPriceFormatted.toFixed(market.precision.price)} not yet profitable vs entry ${state.entryPrice.toFixed(market.precision.price)}. Holding.`);
         }
    } else {
        // Active SL exists: Update only if new SL is strictly better (higher for long, lower for short)
        if (state.positionSide === PositionSide.LONG && newTslPriceFormatted > currentSlPrice + priceTolerance) {
            logger.info(cyan(`TSL Update (Long): New SL ${newTslPriceFormatted.toFixed(market.precision.price)} > Current SL ${currentSlPrice.toFixed(market.precision.price)}. Updating.`));
            shouldUpdate = true;
        } else if (state.positionSide === PositionSide.SHORT && newTslPriceFormatted < currentSlPrice - priceTolerance) {
            logger.info(cyan(`TSL Update (Short): New SL ${newTslPriceFormatted.toFixed(market.precision.price)} < Current SL ${currentSlPrice.toFixed(market.precision.price)}. Updating.`));
            shouldUpdate = true;
        } else {
             logger.debug(`TSL: Potential new SL ${newTslPriceFormatted.toFixed(market.precision.price)} not better than current ${currentSlPrice.toFixed(market.precision.price)}. Holding.`);
        }
    }

    // Prevent setting SL too close to current price (can cause immediate trigger on spread/slippage)
    if (shouldUpdate) {
        const minDistanceBuffer = Math.max(atrValue * 0.25, market.info?.tickSize * 5 || atrValue * 0.1); // Min 5 ticks or 25% ATR
        if (state.positionSide === PositionSide.LONG && (currentPrice - newTslPriceFormatted) < minDistanceBuffer) {
            logger.debug(`New TSL ${newTslPriceFormatted} too close to current price ${currentPrice} (Dist: ${(currentPrice - newTslPriceFormatted).toFixed(4)}, Min Buff: ${minDistanceBuffer.toFixed(4)}). Holding TSL.`);
            shouldUpdate = false;
        }
        if (state.positionSide === PositionSide.SHORT && (newTslPriceFormatted - currentPrice) < minDistanceBuffer) {
            logger.debug(`New TSL ${newTslPriceFormatted} too close to current price ${currentPrice} (Dist: ${(newTslPriceFormatted - currentPrice).toFixed(4)}, Min Buff: ${minDistanceBuffer.toFixed(4)}). Holding TSL.`);
            shouldUpdate = false;
        }
    }


    // --- Execute Update (Cancel & Replace) ---
    if (shouldUpdate) {
        logger.info(bold(`ACTION: Updating Trailing Stop Loss via Cancel & Replace to ${newTslPriceFormatted.toFixed(market.precision.price)}...`));
        const slOrderSide = state.positionSide === PositionSide.LONG ? 'sell' : 'buy';

        // --- 1. Cancel Existing SL Order (if tracked) ---
        const cancelSuccess = await cancelAllSLTPOrders(`TSL Update to ${newTslPriceFormatted}`); // Use helper
        if (!cancelSuccess) {
            logger.error(red(`CRITICAL: Failed to cancel existing SL order during TSL update. Aborting update.`));
            // State might be inconsistent, re-verify next cycle.
            return;
        }
        // cancelAllSLTPOrders should have cleared state.currentTSL

        // --- 2. Place New SL Order ---
        try {
            const slParams = {
                'category': 'linear',
                'positionIdx': 0,
                'reduceOnly': true,
                'closeOnTrigger': true, // Recommended for Bybit V5 SL/TP
                'triggerBy': 'LastPrice', // Make configurable? Defaulting to LastPrice
                'triggerDirection': state.positionSide === PositionSide.LONG ? 2 : 1, // 1=Rise, 2=Fall
                'orderType': 'Market', // Execute as Market on trigger
                'basePrice': exchange.priceToPrecision(CONFIG.SYMBOL, currentPrice), // Required for Bybit V5 stop orders
                'stopPrice': newTslPriceFormatted, // The trigger price
            };
            // Validate basePrice
             if (isNaN(parseFloat(slParams.basePrice)) || parseFloat(slParams.basePrice) <= 0) {
                 throw new Error(`Invalid basePrice (${slParams.basePrice}) derived from currentPrice (${currentPrice}).`);
             }

            const placeTslFunc = async () => await exchange.createOrder(
                CONFIG.SYMBOL, 'Stop', slOrderSide, state.positionAmount, undefined, slParams
            );

            const newSlOrder = await retryOnException(placeTslFunc, `Place New TSL Order @ ${newTslPriceFormatted}`);

            if (newSlOrder && newSlOrder.id) {
                 // --- 3. Update State ---
                 state.currentTSL = { price: newTslPriceFormatted, orderId: newSlOrder.id };
                 await saveState();
                 logger.info(green(`Trailing SL successfully updated. New SL ID: ${state.currentTSL.orderId}, Trigger: ${state.currentTSL.price.toFixed(market.precision.price)}`));
                 sendTermuxSms(`${CONFIG.SYMBOL} TSL updated: Trigger @ ${newTslPriceFormatted.toFixed(market.precision.price)}`);
            } else {
                 logger.error(red(`CRITICAL: Failed to place new TSL order at ${newTslPriceFormatted.toFixed(market.precision.price)} after cancelling previous. Position might be UNPROTECTED!`));
                 state.currentTSL = null; // Ensure state reflects no active SL
                 await saveState();
                 sendTermuxSms(`CRITICAL: Failed place new TSL for ${CONFIG.SYMBOL}. Pos UNPROTECTED! Check Exchange!`);
            }
        } catch (placeError) {
            logger.error(red(`CRITICAL: Error placing new TSL order: ${placeError.message}. Position might be UNPROTECTED!`), placeError.stack);
            state.currentTSL = null; // Ensure state reflects no active SL
            await saveState();
            sendTermuxSms(`CRITICAL: Error placing TSL for ${CONFIG.SYMBOL}. Pos UNPROTECTED! Check Logs!`);
        }
    }
}


// --- Banishing Orders (Cancellation Helpers) ---

/**
 * Cancels an order by its ID. Includes retry logic.
 * @param {string} orderId - The ID of the order to cancel.
 * @param {string} orderTypeLabel - A label for the order type (e.g., 'TSL') for logging.
 * @returns {Promise<boolean>} True if cancellation was successful or order was already gone, false otherwise.
 */
async function cancelOrder(orderId, orderTypeLabel) {
    if (!orderId || orderId.startsWith('dry_')) {
        logger.debug(`Skipping cancel ${orderTypeLabel}: No real ID or dry run ID (${orderId})`);
        return true; // Consider simulation successful
    }

    logger.info(`Attempting banishment of ${orderTypeLabel} order: ${orderId}`);
    if (CONFIG.DRY_RUN) {
        logger.dryRun(`Simulated banishment of ${orderTypeLabel} order: ${orderId}`);
        return true; // Simulation successful
    }

    try {
        const params = { 'category': 'linear' }; // Bybit V5 param
        await retryOnException(
            async () => await exchange.cancelOrder(orderId, CONFIG.SYMBOL, params),
            `cancelOrder (${orderTypeLabel} ${orderId})`
        );
        logger.info(green(`${orderTypeLabel} order ${orderId} banished.`));
        return true;
    } catch (e) {
        // Handle cases where the order is already gone as success
        if (e instanceof ccxt.OrderNotFound || (e instanceof ccxt.ExchangeError && (
            e.message.includes("Order does not exist") ||
            e.message.includes("already closed") ||
            e.message.includes("has been filled") ||
            e.message.includes("canceled") ||
            e.message.includes("ret_code=30034") || // Bybit: Order has finished
            e.message.includes("ret_code=10001") || // Bybit: Parameter error (often implies order gone)
            e.message.includes("ret_code=110001")   // Bybit: Order not found or finished
        ))) {
             logger.warn(`${orderTypeLabel} order ${orderId} already vanished or finished.`);
             return true; // Order is gone, which is the goal
        } else if (e instanceof ccxt.InvalidOrder) {
             logger.warn(`${orderTypeLabel} order ${orderId} likely already closed/filled (InvalidOrder): ${e.message}`);
             return true; // Order is effectively gone
        } else {
            // Log other errors as failures
            logger.error(red(`Error banishing ${orderTypeLabel} ${orderId}: ${e.constructor.name} - ${e.message}`));
            console.error(e);
            sendTermuxSms(`BOT ALERT: Failed cancelling ${orderTypeLabel} ${orderId} ${CONFIG.SYMBOL}. ${e.message.substring(0,80)}`);
            return false; // Cancellation failed
        }
    }
}

/**
 * Cancels all active bot-managed SL/TP orders based on state.
 * @param {string} reason - Reason for cancellation.
 * @returns {Promise<boolean>} True if all tracked orders were successfully cancelled or confirmed gone, false otherwise.
 */
async function cancelAllSLTPOrders(reason) {
    logger.debug(`Cancelling all active SL/TP orders. Reason: ${reason}`);
    let allCancelled = true;
    let changed = false;

    // Cancel Stop Loss (TSL) if tracked
    if (state.currentTSL?.orderId) {
        const slId = state.currentTSL.orderId;
        const success = await cancelOrder(slId, 'TSL');
        if (success) {
            logger.info(`Cleared TSL state for order ${slId}.`);
            state.currentTSL = null;
            changed = true;
        } else {
             logger.error(red(`Failed to cancel TSL order ${slId}. State remains unchanged, manual check advised!`));
             allCancelled = false; // Mark failure
        }
    }

    // Add similar logic here if separate TP orders are tracked in state
    // if (state.currentTP?.orderId) { ... }

    if (changed) {
        await saveState(); // Save state if any order was cleared
    }
    return allCancelled; // Return overall success/failure
}


// --- Data Fetching Helpers ---

/**
 * Safely fetches OHLCV data with retry logic and validation.
 * @param {string} timeframe - The timeframe string (e.g., '1h', '15m').
 * @param {number} limit - The number of candles to fetch.
 * @returns {Promise<Array<Array<number>>|null>} OHLCV data or null on failure.
 */
async function safeFetchOHLCV(timeframe, limit) {
    try {
        const params = { 'category': 'linear' }; // Bybit V5 param
        const candles = await retryOnException(
            async () => await exchange.fetchOHLCV(CONFIG.SYMBOL, timeframe, undefined, limit, params),
            `fetchOHLCV (${CONFIG.SYMBOL} ${timeframe})`
        );

        // Validate response structure
        if (!Array.isArray(candles)) {
            logger.warn(`fetchOHLCV returned non-array: ${typeof candles}`);
            return null;
        }
        if (candles.length > 0) {
             // Basic structure validation of the last candle
            const lastCandle = candles[candles.length - 1];
            if (!Array.isArray(lastCandle) || lastCandle.length < OHLCV_SCHEMA.length || lastCandle.slice(0, OHLCV_SCHEMA.length).some(v => typeof v !== 'number')) {
                logger.warn(`Fetched candles have invalid structure or non-numeric values: ${inspect(lastCandle)}`);
                return null;
            }
        } else {
            logger.debug(`fetchOHLCV returned 0 candles for ${CONFIG.SYMBOL} ${timeframe}.`);
        }

        logger.debug(`Successfully fetched ${candles.length} candles for ${CONFIG.SYMBOL} ${timeframe}.`);
        return candles;

    } catch (e) {
        // Error already logged by retryOnException if retries failed
        logger.error(`Failed to fetch OHLCV for ${CONFIG.SYMBOL} ${timeframe} after retries.`);
        return null;
    }
}

/**
 * Fetches the current position for the configured symbol with retry logic.
 * Handles dry run simulation based on internal state. Standardizes the output.
 * @param {string} symbolId - The exchange-specific symbol ID (usually same as CONFIG.SYMBOL).
 * @param {boolean} [forceFetch=false] - If true, bypasses cache (not implemented in helper, handled by caller).
 * @returns {Promise<object|null>} Standardized position object { side, contracts, entryPrice, symbol, ... } or null.
 */
async function fetchCurrentPosition(symbolId, forceFetch = false) {
    // Handle Dry Run
    if (CONFIG.DRY_RUN) {
        if (state.positionSide !== PositionSide.NONE) {
            logger.debug(`DRY RUN: Returning simulated position: ${state.positionSide} ${state.positionAmount}`);
            return {
                symbol: symbolId,
                side: state.positionSide,
                contracts: state.positionAmount,
                entryPrice: state.entryPrice,
                leverage: CONFIG.LEVERAGE,
                // Add other common fields needed by reconciliation logic, default to 0/null
                unrealizedPnl: 0, initialMargin: 0, maintMargin: 0, liquidationPrice: null,
                info: { simulated: true } // Add simulation flag
            };
        }
        logger.debug("DRY RUN: No simulated position.");
        return null;
    }

    // Live mode: Fetch actual position
    try {
        const params = { 'category': 'linear' }; // Bybit V5 param
        // Bybit V5 requires the symbol for fetchPositions
        const positions = await retryOnException(
            async () => await exchange.fetchPositions([symbolId], params),
            `fetchPositions (${symbolId})`
        );

        // Find the position matching the symbol that has a non-zero size
        const position = positions.find(p => {
            if (!p || p.symbol !== symbolId) return false;
            // Check standard 'contracts' field and Bybit 'size' field in info
            const contractsStr = p.contracts ?? p.info?.size ?? '0';
            const size = parseFloat(contractsStr);
            // Also check entry price validity
            const entryPriceStr = p.entryPrice ?? p.info?.avgPrice ?? null;
            return !isNaN(size) && size !== 0 && entryPriceStr !== null && parseFloat(entryPriceStr) > 0;
        });

        if (position) {
             // --- Standardize the position object ---
             let standardizedSide = position.side?.toLowerCase(); // 'long', 'short', or potentially null/undefined
             const infoSide = position.info?.side?.toLowerCase(); // Bybit V5 uses 'Buy'/'Sell'

             // Infer side if missing or inconsistent
             if (!standardizedSide && infoSide === 'buy') standardizedSide = 'long';
             if (!standardizedSide && infoSide === 'sell') standardizedSide = 'short';

             // Convert side to enum value
             let finalSide = PositionSide.NONE;
             if (standardizedSide === 'long') finalSide = PositionSide.LONG;
             if (standardizedSide === 'short') finalSide = PositionSide.SHORT;

             // Standardize contracts/size
             const contracts = parseFloat(position.contracts ?? position.info?.size ?? '0');
             // Standardize entry price
             const entryPrice = parseFloat(position.entryPrice ?? position.info?.avgPrice ?? '0');

             // Create a standardized return object
             const standardizedPosition = {
                 ...position, // Include all original fields from CCXT
                 side: finalSide, // Use the enum value
                 contracts: contracts, // Ensure it's a number
                 entryPrice: entryPrice, // Ensure it's a number
             };

             logger.debug(`Fetched position ${symbolId}: ${finalSide} ${contracts} @ ${entryPrice}`);
             return standardizedPosition;
        } else {
             logger.debug(`Fetched positions for ${symbolId}, but no active position found.`);
             return null;
        }

    } catch (e) {
        // Error already logged by retryOnException
        logger.error(`Error fetching position for ${symbolId} after retries.`);
        return null; // Return null on failure
    }
}


/**
 * Sets the leverage for the symbol with retry logic.
 * @param {ccxt.Market} market - The market object from ccxt.
 * @returns {Promise<void>}
 * @throws {Error} If leverage setting fails critically.
 */
async function setLeverage(market) {
     const leverageValue = CONFIG.LEVERAGE; // Already parsed as int
     if (isNaN(leverageValue) || leverageValue <= 0) {
         logger.warn("Leverage configuration is invalid or missing. Skipping leverage setting.");
         return;
     }

     if (CONFIG.DRY_RUN) {
         logger.dryRun(`Simulated setting leverage to ${leverageValue}x for ${CONFIG.SYMBOL}`);
         return;
     }

     if (!exchange.has['setLeverage']) {
         logger.warn(`Exchange ${exchange.id} does not support setLeverage via CCXT. Skipping.`);
         return;
     }

     try {
         logger.info(`Attempting to set leverage for ${CONFIG.SYMBOL} to ${leverageValue}x`);
         // Bybit V5 requires specific params
         const params = {
             'category': 'linear',
             'buyLeverage': leverageValue.toString(), // Bybit expects strings for leverage
             'sellLeverage': leverageValue.toString(),
         };
         await retryOnException(
             async () => await exchange.setLeverage(leverageValue, CONFIG.SYMBOL, params),
             `setLeverage (${CONFIG.SYMBOL} ${leverageValue}x)`
         );
         logger.info(`Leverage for ${CONFIG.SYMBOL} set command executed.`);
         // Note: Verification might require fetching position data later.
     } catch (e) {
         // Handle specific errors after retries
         if (e instanceof ccxt.ExchangeError && (e.message.includes('leverage not modified') || e.message.includes('Leverage is not changed') || e.message.includes("ret_code=30036"))) {
             logger.warn(`Leverage for ${CONFIG.SYMBOL} already set to ${leverageValue}x or not modified.`);
         } else if (e instanceof ccxt.ExchangeError && (e.message.includes('position exists') || e.message.includes('open order') || e.message.includes("ret_code=30018") || e.message.includes("ret_code=30019"))) {
             logger.warn(`Cannot modify leverage for ${CONFIG.SYMBOL}: Existing position or orders found.`);
         } else {
             logger.error(`Failed to set leverage for ${CONFIG.SYMBOL} to ${leverageValue}x: ${e.message}`);
             if (LOG_LEVELS[CONFIG.LOG_LEVEL] <= LOG_LEVELS.DEBUG) console.error(e);
             // This might be critical, consider halting if leverage can't be set.
             throw new Error(`Could not set leverage for ${CONFIG.SYMBOL}. Halting initialization.`);
         }
     }
}


// --- The Grand Ritual Loop ---

/**
 * The main execution loop of the trading bot.
 */
async function runBot() {
    if (isShuttingDown) {
        logger.info("Shutdown requested, skipping ritual cycle.");
        return;
    }

    state.cycleCount++;
    logger.info(cyan(bold(`\n----- Ritual Cycle ${state.cycleCount} Started (${new Date().toISOString()}) -----`)));

    try {
        // --- Load Market Data ---
        // Load markets periodically or rely on CCXT cache? Load once at init, refresh if errors occur?
        // For simplicity, assume markets are loaded at init. Get specific market info.
        const market = exchange.market(CONFIG.SYMBOL);
        if (!market || !market.linear) {
            logger.critical(red(bold(`Market ${CONFIG.SYMBOL} is invalid, not found, or not a Linear contract!`)));
            logger.critical(red(bold("Halting bot. Please check SYMBOL configuration and exchange connection.")));
            await shutdown("Invalid Market"); // Trigger shutdown
            return; // Stop cycle
        }
        logger.debug(`Using Market: ${market.id}, Type: ${market.type}, Linear: ${market.linear}, Precision: P=${market.precision.price}, A=${market.precision.amount}`);

        // --- State Synchronization ---
        await reconcileStateWithExchange(market);
        if (isShuttingDown) return; // Re-check after potential critical error during reconciliation

        // --- Gather Ingredients & Perform Divination ---
        const candleLimit = Math.max(
             CONFIG.ROOF_SLOW_EMA, CONFIG.ST_EMA_PERIOD + CONFIG.ST_ATR_PERIOD,
             CONFIG.FISHER_PERIOD, CONFIG.ADX_PERIOD, CONFIG.RANGE_ATR_PERIOD
        ) + 100; // Generous buffer
        logger.debug(`Fetching ${candleLimit} candles for ${CONFIG.TIMEFRAME}...`);
        const candles = await safeFetchOHLCV(CONFIG.TIMEFRAME, candleLimit); // Uses retry
        if (!candles) {
             logger.warn(`Failed to fetch sufficient candle data. Skipping strategy execution this cycle.`);
             scheduleNextCycle(); // Schedule next attempt
             return;
        }
        if (candles.length < candleLimit * 0.8) { // Check if fetch returned significantly less than requested
            logger.warn(`Insufficient candle data fetched (${candles?.length} / ${candleLimit}). Skipping strategy execution this cycle.`);
            scheduleNextCycle();
            return;
        }

        logger.debug("Calculating indicators...");
        const indicators = await calculateIndicators(candles);
        if (!indicators) {
            logger.warn("Indicator calculation failed. Skipping strategy execution this cycle.");
            scheduleNextCycle();
            return;
        }

        // --- Log Current State & Indicators ---
        const positionStatus = state.positionSide !== PositionSide.NONE
            ? `${state.positionSide === PositionSide.LONG ? green(state.positionSide.toUpperCase()) : red(state.positionSide.toUpperCase())} (Entry: ${magenta(state.entryPrice?.toFixed(market.precision.price))}, Amt: ${state.positionAmount?.toFixed(market.precision.amount)})`
            : bold('FLAT');
        logger.info(`Position: ${positionStatus}`);
        logger.info(`Indicators: Px=${magenta(indicators.price.toFixed(market.precision.price))}, ST=${magenta(indicators.superTrendValue.toFixed(market.precision.price))} ${indicators.superTrendDirection === 'up' ? green('Up') : red('Down')}, ADX=${magenta(indicators.adx.toFixed(2))}`);
        logger.info(`Fisher: Val=${magenta(indicators.fisherValue.toFixed(3))}, Sig=${magenta(indicators.fisherSignalValue.toFixed(3))}, ATR_ST=${magenta(indicators.atrSuperTrend.toFixed(4))}`);
        if (state.currentTSL) {
            logger.info(`Active TSL: Trigger=${magenta(state.currentTSL.price.toFixed(market.precision.price))}, ID=${gray(state.currentTSL.orderId || 'Server-Side')}`);
        }

        // --- Enact Strategy ---
        // 1. Manage Existing Position: TSL and Exit Signal Check
        if (state.positionSide !== PositionSide.NONE) {
            logger.debug("Position active. Updating TSL & checking for strategy exit signal...");
            await updateTrailingStopLoss(market, indicators.atrSuperTrend); // Update TSL first
            // checkAndPlaceOrder handles exit logic based on SuperTrend flip
            await checkAndPlaceOrder(market, indicators);
        }
        // 2. Look for Entries if Flat
        else {
            logger.debug("Position flat. Scanning for entry signals...");
            await checkAndPlaceOrder(market, indicators); // Check for entry
        }

        // Final state save is handled within sub-functions like open/close/updateTSL/checkAndPlaceOrder

        logger.info(cyan(bold(`----- Ritual Cycle ${state.cycleCount} Completed -----`)));

    } catch (e) {
        // --- Global Error Handling for the Cycle ---
        logger.critical(red(bold(`Unhandled Exception during Cycle ${state.cycleCount}:`)));
        logger.critical(red(e.message));
        console.error(e); // Log the full stack trace

        // Handle specific CCXT errors or general errors that might halt the bot
        if (e instanceof ccxt.AuthenticationError) {
            logger.critical(red(bold("CRITICAL: Authentication failed! Check API Keys. Halting bot.")));
            sendTermuxSms(`BOT CRITICAL: Auth Error ${CONFIG.SYMBOL}. Check API Keys! Bot halted.`);
            await shutdown("Authentication Error"); // Initiate shutdown
            return; // Stop loop
        } else if (e instanceof ccxt.InvalidNonce) {
             logger.error(red(`Invalid Nonce error: ${e.message}. Might require API key regeneration or time sync.`));
             // Consider halting or just warning? Halting might be safer.
             await shutdown("Invalid Nonce");
             return;
        } else if (e instanceof ccxt.InsufficientFunds) {
             // This might occur during TSL adjustment if margin is low, not necessarily critical to halt.
             logger.error(red(`Insufficient Funds detected during cycle: ${e.message}. Check account balance.`));
             sendTermuxSms(`BOT ERROR: Insufficient Funds ${CONFIG.SYMBOL} Cycle ${state.cycleCount}.`);
        } else if (e instanceof ccxt.InvalidOrder) {
             logger.error(`Invalid Order parameters detected during cycle: ${e.message}. Check config or market limits.`);
        } else if (e instanceof ccxt.ExchangeError || e instanceof ccxt.NetworkError) {
            // These were likely already handled by retryOnException, but catch here just in case
            logger.error(`A potentially recovered Exchange/Network Error bubbled up: ${e.message}.`);
            sendTermuxSms(`BOT WARNING: Exchange/Network Err ${CONFIG.SYMBOL} Cycle ${state.cycleCount}. ${e.message.substring(0, 80)}`);
        } else {
            // General unexpected errors
            logger.critical("An unexpected critical error occurred in the main loop.");
            sendTermuxSms(`BOT CRITICAL FAILURE Cycle ${state.cycleCount}. Check logs! ${e.message.substring(0, 80)}`);
            // Consider halting for unknown errors to prevent erratic behavior
            await shutdown("Unexpected Critical Error");
            return;
        }
        // Attempt to save state even after non-halting errors
        await saveState();
    } finally {
        // --- Schedule Next Cycle ---
        scheduleNextCycle();
    }
}

/**
 * Schedules the next execution of the runBot function.
 */
function scheduleNextCycle() {
     if (!isShuttingDown) {
            const interval = CONFIG.POLL_INTERVAL_MS;
            logger.debug(`Awaiting ${interval / 1000}s until next cycle...`);
            // Clear previous timeout just in case
            if (mainLoopTimeoutId) clearTimeout(mainLoopTimeoutId);
            mainLoopTimeoutId = setTimeout(runBot, interval);
        } else {
             logger.info("Shutdown in progress, not scheduling next cycle.");
        }
}

/**
 * Reconciles the bot's internal state with the actual state on the exchange.
 * Fetches live position and corrects internal state if mismatches are found.
 * @param {ccxt.Market} market - The market object.
 * @returns {Promise<void>}
 */
async function reconcileStateWithExchange(market) {
    logger.debug("Reconciling internal state with exchange...");

    if (CONFIG.DRY_RUN) {
        logger.debug("DRY RUN: Skipping exchange state reconciliation.");
        return;
    }

    let stateCorrected = false;
    try {
        // Fetch live position (uses retry logic internally)
        const livePositionData = await fetchCurrentPosition(market.id, true); // Force fetch

        // --- Standardize Live Position Data ---
        const livePosition = {
             side: livePositionData?.side || PositionSide.NONE, // Use the enum value
             size: livePositionData?.contracts ? parseFloat(livePositionData.contracts) : 0.0,
             entryPrice: livePositionData?.entryPrice ? parseFloat(livePositionData.entryPrice) : 0.0,
        };

        // --- Compare with Internal State ---
        const statePosition = {
             side: state.positionSide,
             size: state.positionAmount || 0.0,
             entryPrice: state.entryPrice || 0.0,
        };
        const sizeTolerance = market.limits?.amount?.min / 10 || 1e-9; // Tolerance for size comparison

        // Scenario 1: State has position, exchange flat.
        if (statePosition.side !== PositionSide.NONE && livePosition.side === PositionSide.NONE) {
            logger.warn(yellow(`State/Reality Mismatch: Memory had ${statePosition.side} position, but exchange reports FLAT.`));
            logger.warn(yellow(`Possible reasons: Manual closure, SL/TP hit, Liquidation.`));
            sendTermuxSms(`BOT ALERT: Position mismatch ${CONFIG.SYMBOL}. Bot thought ${statePosition.side}, exchange FLAT. State reset.`);
            await cancelAllSLTPOrders("Clearing orders due to position mismatch (state->live=flat)");
            state.positionSide = PositionSide.NONE; state.entryPrice = null; state.positionAmount = null; state.currentTSL = null; state.lastSignal = null;
            stateCorrected = true;
        }
        // Scenario 2: State flat, exchange has position. CRITICAL.
        else if (statePosition.side === PositionSide.NONE && livePosition.side !== PositionSide.NONE) {
            logger.critical(red(bold(`CRITICAL STATE MISMATCH: Bot memory FLAT, but exchange reports OPEN ${livePosition.side} ${CONFIG.SYMBOL}!`)));
            logger.critical(red(bold(` -> Exchange Position: Size=${livePosition.size.toFixed(market.precision.amount)}, Entry=${livePosition.entryPrice.toFixed(market.precision.price)}`)));
            logger.critical(red(bold(" -> Manual intervention REQUIRED. Halting bot cycles.")));
            sendTermuxSms(`BOT CRITICAL: State mismatch ${CONFIG.SYMBOL}. Exchange OPEN ${livePosition.side}, bot FLAT. Manual check REQUIRED! Bot halted.`);
            await shutdown("Critical state mismatch"); // Initiate shutdown
            return; // Stop reconciliation
        }
        // Scenario 3: State and exchange have positions. Check consistency.
        else if (statePosition.side !== PositionSide.NONE && livePosition.side !== PositionSide.NONE) {
             if (statePosition.side !== livePosition.side || Math.abs(statePosition.size - livePosition.size) > sizeTolerance) {
                  logger.warn(yellow(`State/Reality Drift Detected: Updating bot memory from live data.`));
                  logger.warn(yellow(` -> Memory: ${statePosition.side} ${statePosition.size.toFixed(market.precision.amount)} @ ${statePosition.entryPrice?.toFixed(market.precision.price)}`));
                  logger.warn(yellow(` -> Reality: ${livePosition.side} ${livePosition.size.toFixed(market.precision.amount)} @ ${livePosition.entryPrice?.toFixed(market.precision.price)}`));

                  state.positionSide = livePosition.side;
                  state.entryPrice = livePosition.entryPrice;
                  state.positionAmount = livePosition.size;
                  stateCorrected = true;

                  // If position state was corrected, existing SL might be wrong. Re-evaluate TSL.
                  logger.info('Re-evaluating TSL due to state reconciliation.');
                  await cancelAllSLTPOrders("Clearing orders for TSL recalculation after state reconciliation");
                  // Fetch fresh ATR to update TSL correctly
                  const freshCandlesForTsl = await safeFetchOHLCV(CONFIG.TIMEFRAME, Math.max(CONFIG.ST_ATR_PERIOD, 50) + 10);
                  if (freshCandlesForTsl) {
                      const atrValues = ATR.calculate({ period: CONFIG.ST_ATR_PERIOD, high: freshCandlesForTsl.map(c=>c[OHLCV_INDEX.high]), low: freshCandlesForTsl.map(c=>c[OHLCV_INDEX.low]), close: freshCandlesForTsl.map(c=>c[OHLCV_INDEX.close]) });
                      const latestAtrForTsl = atrValues?.length > 0 ? atrValues[atrValues.length - 1] : NaN;
                      if (!isNaN(latestAtrForTsl) && latestAtrForTsl > 0) {
                          await updateTrailingStopLoss(market, latestAtrForTsl); // Place new TSL
                      } else { logger.warn("Could not calculate fresh ATR for TSL reconciliation."); }
                  } else { logger.warn("Could not fetch fresh candles for TSL reconciliation."); }
             } else {
                 logger.debug("Memory/reality position match verified.");
             }
        }
        // Scenario 4: Both flat.
        else { // statePosition.side === PositionSide.NONE && livePosition.side === PositionSide.NONE
            logger.debug("Memory/reality align: Bot is FLAT and exchange confirms no position.");
            // Ensure TSL state is clear if flat
            if (state.currentTSL) {
                 logger.warn("Found TSL state while flat. Clearing potentially dangling TSL state.");
                 await cancelAllSLTPOrders("Clearing dangling TSL state while flat");
                 stateCorrected = true; // State was corrected by clearing TSL
            }
        }

        if (stateCorrected) {
             await saveState(); // Save reconciled state
        }
        logger.debug("State reconciliation complete.");

    } catch (error) {
        logger.error(red(`Error during state reconciliation: ${error.message}`), error.stack);
        // If reconciliation fails, proceed with caution, state might be inaccurate.
        // Consider halting if reconciliation repeatedly fails?
    }
}

// --- The Awakening Ritual ---

/**
 * Initializes the bot, validates configuration, sets up logging and exchange connection,
 * loads state, and starts the main loop.
 */
async function initialize() {
    console.log(cyan(bold("\nInitializing Pyrmethus Roofed/Fisher/ST/ADX Bot v1.3...")));
    console.log(cyan(bold("=======================================================")));

    try {
        // 1. Config is already loaded and validated globally

        // 2. Setup Logging (including file if enabled)
        await setupLogFileStream(); // Ensure file stream is ready

        // 3. Log Configuration Summary (now that logger is ready)
        logger.info(bold("--- Configuration Summary ---"));
        logger.info(` Exchange: ${cyan('Bybit (Linear)')}`);
        logger.info(` Symbol: ${cyan(CONFIG.SYMBOL)} | Timeframe: ${cyan(CONFIG.TIMEFRAME)}`);
        logger.info(` Cycle Interval: ${cyan(CONFIG.POLL_INTERVAL_MS / 1000)}s | Log Level: ${cyan(CONFIG.LOG_LEVEL)} | Log File: ${CONFIG.LOG_TO_FILE ? green('Enabled') : gray('Disabled')} (${CONFIG.LOG_FILE_PATH})`);
        logger.info(` Order Size (USD): ${cyan(CONFIG.ORDER_AMOUNT_USD)} | Leverage: ${cyan(CONFIG.LEVERAGE)}x`);
        logger.info(bold("--- Strategy Parameters ---"));
        logger.info(` Roofing Filter EMAs: ${cyan(CONFIG.ROOF_FAST_EMA)} / ${cyan(CONFIG.ROOF_SLOW_EMA)} (on HLC3)`);
        logger.info(` SuperTrend: Basis EMA(${cyan(CONFIG.ST_EMA_PERIOD)}) of Roofed Price, ATR(${cyan(CONFIG.ST_ATR_PERIOD)}), Multiplier(${cyan(CONFIG.ST_MULTIPLIER)})`);
        logger.info(` Confirmation: Fisher Transform (${cyan(CONFIG.FISHER_PERIOD)}) Crossover (on High/Low)`);
        logger.info(` Filters: ADX(${cyan(CONFIG.ADX_PERIOD)}) Range [${cyan(CONFIG.MIN_ADX_LEVEL)}-${cyan(CONFIG.MAX_ADX_LEVEL)}], ATR(${cyan(CONFIG.RANGE_ATR_PERIOD)}) Volatility >= ${cyan(CONFIG.MIN_ATR_PERCENTAGE)}%`);
        logger.info(` Risk Management: Initial SL (${cyan(CONFIG.INITIAL_SL_ATR_MULTIPLIER)}x ATR_ST), Trailing SL (${cyan(CONFIG.TSL_ATR_MULTIPLIER)}x ATR_ST)`);
        logger.info(bold("--- Settings ---"));
        if (CONFIG.DRY_RUN) logger.warn(magenta(bold(" DRY RUN MODE ACTIVE - No real trades will be executed.")));
        else logger.info(green(bold(" LIVE TRADING MODE ACTIVE")));
        if (CONFIG.TERMUX_NOTIFY) logger.info(` Termux SMS Notifications: ${green('Enabled')} for ${cyan(CONFIG.NOTIFICATION_PHONE_NUMBER || 'N/A')}`);
        else logger.info(" Termux SMS Notifications: Disabled");
        logger.info(` State Persistence File: ${cyan(CONFIG.PERSISTENCE_FILE)}`);
        logger.info(` API Retries: Max ${cyan(CONFIG.MAX_RETRIES)} | Initial Delay ${cyan(CONFIG.RETRY_DELAY_MS)}ms`);
        logger.info(` Close Position on Shutdown: ${CONFIG.CLOSE_ON_SHUTDOWN ? green('Enabled') : gray('Disabled')}`);
        console.log(cyan(bold("=======================================================")));

        // 4. Load Initial State & Reconcile
        // loadState includes validation and reconciliation via verifyStateWithExchange
        await loadState();

        // 5. Connect to Exchange & Set Leverage (already done implicitly by ccxt instance creation)
        logger.info("Connecting to exchange and loading market definitions...");
        await exchange.loadMarkets(true); // Force reload markets on init
        const market = exchange.market(CONFIG.SYMBOL);
        if (!market) {
            throw new Error(`Market ${CONFIG.SYMBOL} not found on exchange after loading markets.`);
        }
        await setLeverage(market); // Set leverage (includes dry run check & retry)

        // 6. Initial State Verification (already done within loadState -> verifyState)
        // Could potentially run reconcileStateWithExchange again here for extra safety, but likely redundant.
        // await reconcileStateWithExchange(market);


    } catch (e) {
        // Use console.error as logger might have failed during setup
        console.error(red(bold(`\n--- BOT FAILED TO INITIALIZE ---`)));
        console.error(red(`Fatal Initialization Error: ${e.message}`), e.stack);
        await closeLogFileStream(); // Attempt to close log file before exiting
        process.exit(1); // Stop the bot if initialization fails critically
    }

    // 7. Start the Main Loop
    logger.info(green(bold("Initialization complete. Starting the main ritual loop...")));
    runBot(); // Start the first cycle
}

/**
 * Performs graceful shutdown procedures.
 * @param {string} reason - Reason for shutdown.
 * @returns {Promise<void>}
 */
async function shutdown(reason = "Signal received") {
    if (isShuttingDown) return; // Prevent multiple shutdowns
    isShuttingDown = true;
    logger.warn(yellow(bold(`\n--- Initiating Graceful Shutdown (${reason}) ---`)));

    // Clear main loop timeout
    if (mainLoopTimeoutId) {
        clearTimeout(mainLoopTimeoutId);
        mainLoopTimeoutId = null;
        logger.info("Cleared pending ritual cycle timeout.");
    }

    // Cancel open bot-managed orders (TSL)
    logger.info("Cancelling any active bot-managed TSL orders...");
    await cancelAllSLTPOrders(`Bot Shutdown (${reason})`);

    // Optionally close position
    if (CONFIG.CLOSE_ON_SHUTDOWN && !CONFIG.DRY_RUN) {
         logger.info("Attempting to close open position due to shutdown setting...");
         // Fetch latest state first, bypassing cache
         const position = await fetchCurrentPosition(CONFIG.SYMBOL, true); // Force fetch
         if (position && parseFloat(position.contracts) > 0) {
              const market = exchange.market(CONFIG.SYMBOL);
              await closePosition(market, `Graceful shutdown (${reason})`);
         } else {
              logger.info("No open position found to close.");
         }
    } else if (CONFIG.CLOSE_ON_SHUTDOWN && CONFIG.DRY_RUN) {
         logger.dryRun("DRY RUN: Skipping position closure on shutdown.");
    } else {
         logger.info("CLOSE_ON_SHUTDOWN is disabled. Position (if any) will remain open.");
    }

    // Final save state attempt (might capture closed position)
    logger.info("Attempting final state save...");
    await saveState();

    // Close log file stream
    await closeLogFileStream();

    logger.info(yellow(bold("--- Shutdown Complete ---")));
    process.exit(0); // Exit gracefully
}

// --- Signal Handlers for Graceful Shutdown ---
process.on('SIGINT', () => shutdown('SIGINT')); // Ctrl+C
process.on('SIGTERM', () => shutdown('SIGTERM')); // kill/systemd stop
process.on('uncaughtException', async (error, origin) => {
  console.error(red(bold('\n--- !!! UNCAUGHT EXCEPTION !!! ---')));
  console.error(red(`Origin: ${origin}`));
  console.error(red('Error:'), error);
  if (logger && typeof logger.critical === 'function') { // Check if logger is available
      logger.critical('--- !!! UNCAUGHT EXCEPTION !!! ---');
      logger.critical(`Origin: ${origin}`);
      logger.critical(error); // Log the error object itself
  }
  // Attempt a minimal shutdown
  await closeLogFileStream(); // Try to close log file
  process.exit(1); // Exit immediately, state might be corrupt
});
process.on('unhandledRejection', async (reason, promise) => {
  console.error(red(bold('\n--- !!! UNHANDLED REJECTION !!! ---')));
  console.error(red(`Reason:`), reason);
  if (logger && typeof logger.critical === 'function') {
      logger.critical('--- !!! UNHANDLED REJECTION !!! ---');
      logger.critical(`Reason: ${reason instanceof Error ? reason.message : reason}`);
      // logger.critical('Promise:', promise); // Promise object can be large
  }
   // Attempt a minimal shutdown
  await closeLogFileStream();
  process.exit(1);
});

// Begin the enchantment...
initialize();
