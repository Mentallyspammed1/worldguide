Okay, this is the **full, enhanced setup script** (`setup_bybit_bot.sh` v2.0) that will create the necessary files and embed the **complete, advanced Bybit Trading Bot JavaScript code** you provided.

This setup script includes:

*   Prerequisite checks (Node/npm).
*   Project directory setup with overwrite protection.
*   Generation of a comprehensive `.env` file with all relevant parameters for the advanced bot.
*   Creation of the `bybit_trading_bot.js` file containing your full advanced bot logic.
*   Optional interactive installation of the correct dependencies (`ccxt`, `dotenv`, `nanocolors`).
*   Clear final instructions tailored to the advanced bot.

```bash
#!/usr/bin/env bash

# === Awesome Bybit Advanced Trading Bot Setup Script v2.0 ===
# Embeds the full dual-SuperTrend bot with risk management, TSL, etc.

# --- Configuration ---
PROJECT_DIR="bybit_advanced_bot" # Updated name
BOT_SCRIPT="bybit_trading_bot.js"
ENV_FILE=".env"
# Dependencies for the advanced bot (no prompts/technicalindicators needed based on provided JS)
NPM_PACKAGES="ccxt dotenv nanocolors"

# --- Colors ---
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_INFO='\033[0;34m'    # Blue
C_SUCCESS='\033[0;32m' # Green
C_WARN='\033[0;33m'    # Yellow
C_ERR='\033[0;31m'     # Red
C_HIGHLIGHT='\033[0;35m' # Magenta
C_DIM='\033[2m'        # Dim

# --- Helper Functions ---
info() { echo -e "${C_BOLD}${C_INFO}ℹ INFO:${C_RESET} $1"; }
success() { echo -e "${C_BOLD}${C_SUCCESS}✔ SUCCESS:${C_RESET} $1"; }
warn() { echo -e "${C_BOLD}${C_WARN}⚠ WARNING:${C_RESET} $1"; }
error() { echo -e "${C_BOLD}${C_ERR}✖ ERROR:${C_RESET} $1"; }
highlight() { echo -e "${C_HIGHLIGHT}$1${C_RESET}"; }
step() { echo -e "\n${C_BOLD}${C_INFO}➡️  Step $1: $2${C_RESET}";}

# --- Script Start ---
echo -e "\n${C_BOLD}${C_HIGHLIGHT}🚀 Launching Advanced Bybit Trading Bot Setup! 🚀${C_RESET}"
echo "--------------------------------------------------"

# Step 1: Check Node.js and npm
step 1 "Checking Node.js & npm prerequisites..."
if ! command -v node >/dev/null 2>&1; then
    error "Node.js is not installed or not in PATH. Please install Node.js (which includes npm) first."
    echo -e "${C_DIM}   Visit https://nodejs.org/ or use a package manager (e.g., 'sudo apt install nodejs npm', 'brew install node').${C_RESET}"
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    error "npm (Node Package Manager) not found. It usually comes with Node.js."
    echo -e "${C_DIM}   Try reinstalling Node.js or check your system's PATH variable.${C_RESET}"
    exit 1
fi
node_version=$(node -v)
npm_version=$(npm -v)
success "Node.js ($node_version) and npm ($npm_version) found."

# Step 2: Check/Create Project Directory
step 2 "Setting up Project Directory..."
if [ -d "$PROJECT_DIR" ]; then
    warn "Project directory '${C_BOLD}${PROJECT_DIR}${C_WARN}' already exists."
    read -p "$(echo -e ${C_WARN}"❓ Overwrite? This will ${C_ERR}${C_BOLD}PERMANENTLY DELETE ALL EXISTING FILES${C_WARN} in '${PROJECT_DIR}'! (y/N): "${C_RESET})" -n 1 -r REPLY
    REPLY=${REPLY:-N} # Default to No
    echo # Newline
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        info "Removing existing directory: $PROJECT_DIR"
        rm -rf "$PROJECT_DIR" || { error "Failed removing '$PROJECT_DIR'. Check permissions." && exit 1; }
        success "Existing directory removed."
    else
        info "${C_DIM}Keeping existing directory. Be aware files might be overwritten.${C_RESET}"
    fi
fi

info "Ensuring project directory exists: ${C_BOLD}${PROJECT_DIR}${C_RESET}"
mkdir -p "$PROJECT_DIR" || { error "Failed creating '$PROJECT_DIR'. Check permissions." && exit 1; }
success "Directory '$PROJECT_DIR' created."

# Change into the project directory
cd "$PROJECT_DIR" || { error "Failed changing to '$PROJECT_DIR'." && exit 1; }
info "${C_DIM}Current directory: $(pwd)${C_RESET}"

# Step 3: Create .env file with comprehensive options
step 3 "Creating Comprehensive Environment File (${C_BOLD}${ENV_FILE}${C_RESET})..."
touch "$ENV_FILE"
cat << 'EOF' > "$ENV_FILE"
# --- Environment Variables for Advanced Bybit Trading Bot ---
# MANDATORY: Edit API Key/Secret and review ALL other settings.

# --- MANDATORY: Bybit API Credentials ---
# Permissions needed: Read/Write for "Trade" (Contracts/Derivatives).
# Ensure "Unified Trading" or "Derivatives" toggle is enabled for keys if applicable.
# Disable "Withdraw". Restrict IP strongly recommended.
EXCHANGE_API_KEY=YOUR_BYBIT_API_KEY
EXCHANGE_API_SECRET=YOUR_BYBIT_API_SECRET

# --- MANDATORY: Exchange Selection (Should be 'bybit') ---
EXCHANGE_ID=bybit

# --- Trading Parameters ---
SYMBOL="BTC/USDT:USDT"        # Trading pair (e.g., BTC/USDT:USDT for Linear Swaps)
TIMEFRAME="1m"                # Candlestick timeframe (1m, 5m, 15m, 1h, 4h, 1d etc.)
LEVERAGE=10.0                 # Leverage (e.g., 10 for 10x) - Ensure supported by Bybit for the symbol
RISK_PER_TRADE=0.01           # Risk as a percentage of equity (e.g., 0.01 for 1%)
CURRENCY="USDT"               # Quote currency for balance/risk calculations (usually USDT for linear swaps)
EXCHANGE_TYPE="swap"          # Market type ('swap', 'future', 'spot' - MUST match SYMBOL type)

# --- Strategy Parameters ---
SHORT_ST_PERIOD=7             # Short SuperTrend period (e.g., for exits)
LONG_ST_PERIOD=14            # Long SuperTrend period (e.g., for entries and main ATR)
ST_MULTIPLIER=2.0             # SuperTrend ATR multiplier (for both short & long ST)
VOLUME_SHORT_PERIOD=5         # Short MA period for volume ratio
VOLUME_LONG_PERIOD=20        # Long MA period for volume ratio
VOLUME_SPIKE_THRESHOLD=1.5    # Volume ratio required for entry (e.g., 1.5 means 50% above long average)
ORDER_BOOK_DEPTH=10           # Depth for order book pressure calculation
OB_PRESSURE_THRESHOLD=0.6     # Order book buy pressure threshold (0.0 to 1.0) - Entry if BUY pressure > threshold OR SELL pressure > (1-threshold)

# --- Risk Management (ATR Multipliers) ---
SL_ATR_MULT=1.5               # Stop Loss ATR multiplier (placed on entry)
TP_ATR_MULT=2.0               # Take Profit ATR multiplier (placed on entry)
TRAILING_STOP_MULT=1.5        # Trailing Stop ATR multiplier (updates existing separate SL order)

# --- Order Parameters ---
ORDER_TRIGGER_PRICE_TYPE="LastPrice" # Trigger for SL/TP (Bybit: LastPrice, MarkPrice, IndexPrice)
TIME_IN_FORCE="GoodTillCancel" # Time-in-Force for orders (GTC, IOC, FOK, PostOnly)

# --- Bot Behavior ---
DRY_RUN="true"                # Set to "false" to enable LIVE trading. Default is TRUE (Safe).
LOGGING_LEVEL="INFO"          # Logging verbosity (DEBUG, INFO, WARN, ERROR)
LOG_FILE_ENABLED="true"       # Log to file ('true' or 'false')
LOG_DIR="logs"                # Directory for log files
MAX_RETRIES=3                 # Max retries for failed API calls
RETRY_DELAY=5                 # Delay (seconds) between retries
CACHE_TTL=30                  # API data cache duration (seconds)
STATE_FILE="trading_bot_state.json" # File to store persistent state (e.g., active SL ID)

# --- Optional: SMS Notifications (Requires Termux:API) ---
SMS_ENABLED="false"           # Enable SMS alerts ('true' or 'false')
SMS_RECIPIENT_NUMBER=         # Full phone number (+country code) if SMS_ENABLED is true

EOF
if [ $? -ne 0 ]; then error "Failed writing to '$ENV_FILE'." && exit 1; fi
success "'$ENV_FILE' created. ${C_BOLD}${C_WARN}<<< EDIT ME NOW! Check ALL settings, especially API keys & DRY_RUN! >>>${C_RESET}"

# Step 4: Create the JavaScript bot file
step 4 "Creating Advanced Bybit Bot Script (${C_BOLD}${BOT_SCRIPT}${C_RESET})..."
touch "$BOT_SCRIPT"
# --- Embed the Full Advanced JavaScript Code Provided by User ---
cat << 'EOF' > "$BOT_SCRIPT"
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
            await fs.mkdir(config.log_dir, { recursive: true });
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            logFilePath = path.join(config.log_dir, `trading_bot_${timestamp}.log`);
            logFileHandle = await fs.open(logFilePath, 'a'); // Open in append mode
            console.log(c.cyan(`Logging ${levelName} and higher level messages to file: ${logFilePath}`));
        } catch (err) {
            console.error(c.red(`[ERROR] Failed to create log directory or open log file: ${err.message}. File logging disabled.`));
            logFilePath = null; logFileHandle = null;
        }
    } else {
        console.log(c.gray("File logging is disabled in configuration."));
        logFilePath = null; logFileHandle = null;
    }
}

// Logger object with methods for different levels
const logger = {
    log(level, ...args) {
        if (logLevels[level] < currentLogLevel) return;
        const timestamp = new Date().toISOString();
        const levelColor = { DEBUG: c.gray, INFO: c.cyan, WARN: c.yellow, ERROR: c.red, }[level] || c.white;
        const consoleMessage = `${c.dim(timestamp)} [${levelColor(c.bold(level))}] ${args.map(arg => typeof arg === 'object' ? inspect(arg, { depth: 3, colors: true }) : String(arg)).join(' ')}`;
        const consoleMethod = level === 'WARN' ? console.warn : level === 'ERROR' ? console.error : console.log;
        consoleMethod(consoleMessage);

        if (logFileHandle) {
            const fileMessage = `${timestamp} [${level}] ${args.map(arg => typeof arg === 'object' ? inspect(arg, { depth: 4, colors: false }) : String(arg)).join(' ')}\n`;
            logFileHandle.write(fileMessage).catch(err => {
                console.error(c.red(`[ERROR] Failed to write to log file '${logFilePath}': ${err.message}`));
            });
        }
    },
    debug(...args) { this.log('DEBUG', ...args); },
    info(...args) { this.log('INFO', ...args); },
    warn(...args) { this.log('WARN', ...args); },
    error(...args) { this.log('ERROR', ...args); },
    async closeLogFile() {
        if (logFileHandle) {
            console.log(c.yellow("Closing log file handle..."));
            try { await logFileHandle.close(); logFileHandle = null; console.log(c.green("Log file handle closed.")); }
            catch (err) { console.error(c.red(`Error closing log file handle: ${err.message}`)); }
        }
    }
};

// --- Utility Functions ---
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Wraps an async function call with retry logic for specific exceptions.
 */
async function retryOnException(
    func, maxRetries, delaySeconds,
    allowedExceptions = [ ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection ],
    funcName = func.name || 'anonymous function'
) {
    let attempts = 0;
    while (attempts <= maxRetries) {
        attempts++;
        try { return await func(); }
        catch (e) {
            const isRetryable = allowedExceptions.some(excType => e instanceof excType);
            if (isRetryable && attempts <= maxRetries) {
                const delay = delaySeconds * 1000 * Math.pow(1.5, attempts - 1); // Optional: exponential backoff
                logger.warn(`[Retry] Attempt ${attempts}/${maxRetries + 1} for ${c.yellow(funcName)} failed: ${c.red(e.constructor.name)} - ${e.message}. Retrying in ${(delay/1000).toFixed(1)}s...`);
                await sleep(delay);
            } else {
                if (!isRetryable) { logger.error(`[Error] Non-retryable error in ${c.yellow(funcName)}: ${c.red(e.constructor.name)} - ${e.message}`, (e instanceof Error ? e.stack : '')); }
                else { logger.error(`[Error] ${c.yellow(funcName)} failed after ${attempts} attempts due to ${c.red(e.constructor.name)}: ${e.message}`, (e instanceof Error ? e.stack : '')); }
                throw e;
            }
        }
    }
    const unexpectedError = new Error(`${funcName} retry loop completed unexpectedly.`); logger.error(c.red(unexpectedError.message)); throw unexpectedError;
}


// --- Configuration Class ---
class Config {
    constructor() {
        console.log(c.blue("Loading configuration from .env file..."));
        dotenv.config();
        this.bybit_api_key = process.env.BYBIT_API_KEY || null;
        this.bybit_api_secret = process.env.BYBIT_API_SECRET || null;
        this.symbol = process.env.SYMBOL || DEFAULTS.SYMBOL;
        this.leverage = parseFloat(process.env.LEVERAGE || DEFAULTS.LEVERAGE);
        this.risk_per_trade = parseFloat(process.env.RISK_PER_TRADE || DEFAULTS.RISK_PER_TRADE);
        this.sl_atr_mult = parseFloat(process.env.SL_ATR_MULT || DEFAULTS.SL_ATR_MULT);
        this.tp_atr_mult = parseFloat(process.env.TP_ATR_MULT || DEFAULTS.TP_ATR_MULT);
        this.trailing_stop_mult = parseFloat(process.env.TRAILING_STOP_MULT || DEFAULTS.TRAILING_STOP_MULT);
        this.timeframe = process.env.TIMEFRAME || DEFAULTS.TIMEFRAME;
        this.short_st_period = parseInt(process.env.SHORT_ST_PERIOD || DEFAULTS.SHORT_ST_PERIOD, 10);
        this.long_st_period = parseInt(process.env.LONG_ST_PERIOD || DEFAULTS.LONG_ST_PERIOD, 10);
        this.st_multiplier = parseFloat(process.env.ST_MULTIPLIER || DEFAULTS.ST_MULTIPLIER);
        this.volume_short_period = parseInt(process.env.VOLUME_SHORT_PERIOD || DEFAULTS.VOLUME_SHORT_PERIOD, 10);
        this.volume_long_period = parseInt(process.env.VOLUME_LONG_PERIOD || DEFAULTS.VOLUME_LONG_PERIOD, 10);
        this.volume_spike_threshold = parseFloat(process.env.VOLUME_SPIKE_THRESHOLD || DEFAULTS.VOLUME_SPIKE_THRESHOLD);
        this.order_book_depth = parseInt(process.env.ORDER_BOOK_DEPTH || DEFAULTS.ORDER_BOOK_DEPTH, 10);
        this.ob_pressure_threshold = parseFloat(process.env.OB_PRESSURE_THRESHOLD || DEFAULTS.OB_PRESSURE_THRESHOLD);
        this.dry_run = (process.env.DRY_RUN || DEFAULTS.DRY_RUN).toLowerCase() !== 'false';
        this.logging_level = (process.env.LOGGING_LEVEL || DEFAULTS.LOGGING_LEVEL).toUpperCase();
        this.log_file_enabled = (process.env.LOG_FILE_ENABLED || DEFAULTS.LOG_FILE_ENABLED).toLowerCase() === 'true';
        this.log_dir = process.env.LOG_DIR || DEFAULTS.LOG_DIR;
        this.max_retries = parseInt(process.env.MAX_RETRIES || DEFAULTS.MAX_RETRIES, 10);
        this.retry_delay = parseInt(process.env.RETRY_DELAY || DEFAULTS.RETRY_DELAY, 10);
        this.cache_ttl = parseInt(process.env.CACHE_TTL || DEFAULTS.CACHE_TTL, 10);
        this.state_file = process.env.STATE_FILE || DEFAULTS.STATE_FILE;
        this.currency = process.env.CURRENCY || DEFAULTS.CURRENCY;
        this.exchange_type = process.env.EXCHANGE_TYPE || DEFAULTS.EXCHANGE_TYPE;
        this.order_trigger_price_type = process.env.ORDER_TRIGGER_PRICE_TYPE || DEFAULTS.ORDER_TRIGGER_PRICE_TYPE;
        this.time_in_force = process.env.TIME_IN_FORCE || DEFAULTS.TIME_IN_FORCE;
        this.sms_enabled = (process.env.SMS_ENABLED || DEFAULTS.SMS_ENABLED).toLowerCase() === 'true';
        this.sms_recipient_number = process.env.SMS_RECIPIENT_NUMBER || null;
        this.termux_sms_available = false;

        this._validate();
        this._checkTermuxSms();
        // Call setupLogging immediately - it handles its async part internally
        setupLogging(this).catch(err => console.error("Async logging setup error:", err));

        const configSummary = { ...this }; configSummary.bybit_api_secret = configSummary.bybit_api_secret ? '******' : null;
        logger.info(c.green(`Config loaded. Dry Run: ${c.bold(this.dry_run)}`)); logger.debug("Config (Secret Masked):", configSummary);
    }

    _validate() {
        logger.debug("Validating config..."); const errors = [];
        if (!this.dry_run && (!this.bybit_api_key || !this.bybit_api_secret)) errors.push("BYBIT_API_KEY/SECRET required if not dry run.");
        if (isNaN(this.leverage) || this.leverage <= 0) errors.push("LEVERAGE must be > 0.");
        if (isNaN(this.risk_per_trade) || !(this.risk_per_trade > 0 && this.risk_per_trade <= 1.0)) errors.push("RISK_PER_TRADE must be > 0 and <= 1.");
        if (this.risk_per_trade > 0.05) logger.warn(c.yellow(`High Risk: RISK_PER_TRADE > 5% (${(this.risk_per_trade * 100).toFixed(1)}%)`));
        if (this.risk_per_trade > 0.1) logger.warn(c.red(`EXTREME Risk: RISK_PER_TRADE > 10% (${(this.risk_per_trade * 100).toFixed(1)}%)!`));
        if (!VALID_TIMEFRAMES.includes(this.timeframe)) errors.push(`Invalid TIMEFRAME: '${this.timeframe}'. Valid: ${VALID_TIMEFRAMES.join()}`);
        const checkPosNum = (k) => { if (isNaN(this[k]) || this[k] <= 0) errors.push(`${k.toUpperCase()} must be > 0. Got: ${this[k]}`); };
        const checkPosInt = (k) => { if (!Number.isInteger(this[k]) || this[k] <= 0) errors.push(`${k.toUpperCase()} must be positive integer. Got: ${this[k]}`); };
        const checkNNInt = (k) => { if (!Number.isInteger(this[k]) || this[k] < 0) errors.push(`${k.toUpperCase()} must be non-negative integer. Got: ${this[k]}`); };
        ['sl_atr_mult', 'tp_atr_mult', 'trailing_stop_mult', 'st_multiplier', 'volume_spike_threshold'].forEach(checkPosNum);
        ['short_st_period', 'long_st_period', 'volume_short_period', 'volume_long_period', 'order_book_depth', 'retry_delay', 'cache_ttl'].forEach(checkPosInt);
        checkNNInt('max_retries');
        if (this.volume_short_period >= this.volume_long_period) errors.push("VOLUME_SHORT_PERIOD must be < VOLUME_LONG_PERIOD.");
        if (isNaN(this.ob_pressure_threshold) || !(this.ob_pressure_threshold >= 0 && this.ob_pressure_threshold <= 1)) errors.push("OB_PRESSURE_THRESHOLD must be 0-1.");
        if (this.sms_enabled && (!this.sms_recipient_number || !this.sms_recipient_number.trim())) { logger.warn("SMS_ENABLED=true but SMS_RECIPIENT_NUMBER missing. Disabling SMS."); this.sms_enabled = false; }
        if (!['swap', 'future', 'spot'].includes(this.exchange_type)) errors.push(`Invalid EXCHANGE_TYPE: '${this.exchange_type}'. Use swap, future, or spot.`);
        const valid_triggers = ['lastprice', 'markprice', 'indexprice']; if (!valid_triggers.includes(this.order_trigger_price_type.toLowerCase())) errors.push(`Invalid ORDER_TRIGGER_PRICE_TYPE: '${this.order_trigger_price_type}'. Use LastPrice, MarkPrice, IndexPrice.`);
        const valid_tif = ['goodtillcancel', 'immediateorcancel', 'fillorkill', 'postonly']; if (!valid_tif.includes(this.time_in_force.toLowerCase().replace(/\s/g, ""))) errors.push(`Invalid TIME_IN_FORCE: '${this.time_in_force}'. Use GTC, IOC, FOK, PostOnly.`);
        if (!logLevels.hasOwnProperty(this.logging_level)) { this.logging_level = "INFO"; }
        if (errors.length > 0) { const msg = "Config validation failed:\n" + errors.map(e => `- ${e}`).join('\n'); logger.error(c.red(msg)); throw new Error(msg); }
        logger.debug(c.green("Config validated successfully."));
    }

    _checkTermuxSms() {
        if (!this.sms_enabled) { logger.debug("SMS disabled."); return; }
        const isTermux = process.env.TERMUX_VERSION || (fs.existsSync && fs.existsSync('/data/data/com.termux'));
        if (!isTermux) { logger.warn("Not in Termux env. Disabling SMS."); this.sms_enabled = false; return; }
        try { execSync('which termux-sms-send', { stdio: 'ignore' }); this.termux_sms_available = true; logger.info(c.green("Termux SMS command found. SMS enabled.")); }
        catch (error) { logger.warn(c.yellow("Termux detected, but 'termux-sms-send' not found/executable. Disable SMS. Install Termux:API and run 'pkg install termux-api'?")); this.sms_enabled = false; }
    }
}

// --- Notification Service ---
class NotificationService {
    sendSms(message, config) {
        if (!config.sms_enabled || !config.termux_sms_available || !config.sms_recipient_number) { logger.debug(c.gray(`SMS skip: ${message.substring(0, 50)}...`)); return; }
        try {
            let sanitized = message.replace(/["`$\;]/g, '.').replace(/[\\|&<>(){}]/g, ''); // Basic sanitize
            const maxLen = 160; if (sanitized.length > maxLen) { sanitized = sanitized.substring(0, maxLen - 3) + "..."; }
            const shellEscaped = sanitized.replace(/'/g, "'\\''");
            const command = `termux-sms-send -n '${config.sms_recipient_number}' '${shellEscaped}'`;
            logger.debug(`Exec SMS cmd: termux-sms-send -n '${config.sms_recipient_number}' '...'`);
            exec(command, { timeout: 30000 }, (error, stdout, stderr) => {
                if (error) { logger.error(c.red(`SMS cmd fail: ${error.message}. Code:${error.code}, Sig:${error.signal}`)); if(stderr) logger.error(c.red(`Stder: ${stderr.trim()}`)); return; }
                logger.info(c.green(`SMS sent OK to ${config.sms_recipient_number}: "${message.substring(0, 50)}..."`));
                if (stderr?.trim()) logger.debug(`termux-sms stderr: ${stderr.trim()}`);
                if (stdout?.trim()) logger.debug(`termux-sms stdout: ${stdout.trim()}`);
            });
        } catch (e) { logger.error(c.red(`SMS sync err: ${e.message}`), e.stack); }
    }
}

// --- Exchange Manager ---
class ExchangeManager {
    constructor(config) { this.config = config; this.exchange = null; this._caches = { ohlcv: { k: null, d: null, t: 0, ttl: config.cache_ttl }, order_book: { k: null, d: null, t: 0, ttl: Math.max(1, Math.min(config.cache_ttl, 10)) }, ticker: { k: null, d: null, t: 0, ttl: Math.max(1, Math.min(config.cache_ttl, 5)) }, balance: { k: null, d: null, t: 0, ttl: config.cache_ttl }, position: { k: null, d: null, t: 0, ttl: Math.max(5, Math.min(config.cache_ttl, 15)) }, }; }
    async initialize() { this.exchange = this._setupExchange(); await this._loadMarketsAndValidate(); if (!this.config.dry_run && ['swap', 'future'].includes(this.config.exchange_type)) { await this._setLeverage(); } logger.info(c.green(`ExchMgr init OK: ${c.bold(this.exchange.id)} ${c.bold(this.config.symbol)} ${c.bold(this.config.exchange_type)}`)); }
    _getCache(n, k) { const c = this._caches[n]; if (!c) return null; if (c.k === k && c.d !== null) { const age = Date.now() / 1000 - c.t; if (age < c.ttl) { logger.debug(c.gray(`CACHE HIT ${n} (K:${k}, Age:${age.toFixed(1)}<${c.ttl})`)); return c.d; } else { logger.debug(c.gray(`CACHE EXP ${n} (K:${k}, Age:${age.toFixed(1)}>=${c.ttl})`)); } } return null; }
    _setCache(n, k, d) { if (this._caches[n]) { this._caches[n].k = k; this._caches[n].d = d; this._caches[n].t = Date.now() / 1000; logger.debug(c.gray(`CACHE SET ${n} (K:${k})`)); } else { logger.warn(c.yellow(`Unknown cache: ${n}`)); } }
    _setupExchange() { logger.info(c.blue(`Init Bybit ${this.config.exchange_type} connection...`)); let apiKey, apiSecret; if (this.config.dry_run) { logger.info(c.magenta("Dry Run: Using dummy API keys.")); apiKey="DRY"; apiSecret="DRY"; } else { if (!this.config.bybit_api_key || !this.config.bybit_api_secret) throw new Error("CRIT: Real API Key/Secret required for live."); apiKey=this.config.bybit_api_key; apiSecret=this.config.bybit_api_secret; } try { const opts = { apiKey, secret: apiSecret, enableRateLimit: true, options: { defaultType: this.config.exchange_type, adjustForTimeDifference: true, recvWindow: 15000, } }; if (!ccxt.hasOwnProperty('bybit')) throw new Error("CCXT 'bybit' missing."); const ex = new ccxt.bybit(opts); logger.info(`CCXT ${c.bold(ex.id)} instance OK (v${c.dim(ex.version||'N/A')}).`); return ex; } catch (e) { throw new Error(`Exch setup fail: ${e.message}`); } }
    async _loadMarketsAndValidate() { if (!this.exchange) throw new Error("Exch not init."); try { logger.info(`Loading markets for ${this.exchange.id}...`); const loadFn = async () => await this.exchange.loadMarkets(true); await retryOnException(loadFn, this.config.max_retries, this.config.retry_delay, [ccxt.NetworkError, ccxt.RequestTimeout], 'loadMarkets'); logger.info(c.green(`Loaded ${Object.keys(this.exchange.markets).length} markets.`)); logger.info(`Validating symbol '${c.bold(this.config.symbol)}' type '${c.bold(this.config.exchange_type)}'...`); let market; try { market = this.exchange.market(this.config.symbol); if (!market) throw new ccxt.BadSymbol(`Market data NF for ${this.config.symbol}`); } catch (e) { if (e instanceof ccxt.BadSymbol) { const samples = Object.keys(this.exchange.markets).filter(s => this.exchange.markets[s]?.type === this.config.exchange_type).slice(0, 10); logger.error(c.red(`Symbol '${this.config.symbol}' NF for type '${this.config.exchange_type}'. Sample Avail: ${samples.join()||'None found'}...`), e); throw new Error(`Symbol '${this.config.symbol}' invalid for type.`); } else { throw e; } } if (market.type !== this.config.exchange_type) throw new Error(`Symbol ${this.config.symbol} type ('${market.type}') != config type ('${this.config.exchange_type}').`); if (['swap', 'future'].includes(this.config.exchange_type)) { const isLinear = market.linear === true; const isInverse = market.inverse === true; const settle = market.settle; logger.debug(`Market: Lin=${isLinear},Inv=${isInverse},Set=${settle}`); if (isInverse) throw new Error(`Symbol ${this.config.symbol} is INVERSE. Bot needs LINEAR (${this.config.currency}).`); if (!isLinear && (!settle || settle.toUpperCase() !== this.config.currency.toUpperCase())) throw new Error(`Symbol ${this.config.symbol} not linear & settle != ${this.config.currency}. Requires LINEAR.`); if (!isLinear && !isInverse && settle && settle.toUpperCase() === this.config.currency.toUpperCase()) logger.warn(c.yellow(`Market ${this.config.symbol} flags missing, assume LINEAR from settle=${settle}. Verify!`)); } const stdSymbol = market.symbol; if (stdSymbol !== this.config.symbol) { logger.info(`Standardizing symbol: '${this.config.symbol}' -> '${stdSymbol}'`); this.config.symbol = stdSymbol; } logger.info(c.green(`Symbol '${c.bold(this.config.symbol)}' OK (Type:${market.type}, Lin:${market.linear??'N/A'}, Set:${market.settle??'N/A'}).`)); } catch (e) { if (e instanceof Error && (e.message.includes('Symbol') || e.message.includes('market type') || e.message.includes('LINEAR')) && (e.message.includes('valid') || e.message.includes('found') || e.message.includes('match'))) { logger.error(c.red(`Symbol Validation Fail: ${e.message}`)); throw e; } else { logger.error(c.red(`Market/Symbol Validation Unexpected Err: ${e.message}`), e.stack); throw new Error(`Market validation fail: ${e.message}`); } } }
    async _setLeverage() { const { symbol, leverage } = this.config; if (!this.exchange?.has?.['setLeverage']) { logger.warn(c.yellow(`Exch does not support setLeverage(). Skip.`)); return; } logger.info(`Trying set leverage ${c.bold(symbol)} -> ${c.bold(leverage)}x...`); try { const params = {}; if (this.exchange.id === 'bybit' && ['swap', 'future'].includes(this.config.exchange_type)) { params.category = 'linear'; params.buyLeverage = leverage; params.sellLeverage = leverage; logger.debug(`Bybit V5 setLeverage params: ${JSON.stringify(params)}`); } const setLevFn = async () => await this.exchange.setLeverage(leverage, symbol, params); const resp = await retryOnException(setLevFn, this.config.max_retries, this.config.retry_delay, [ccxt.NetworkError], 'setLeverage'); logger.info(c.green(`Leverage set OK for ${symbol}. Resp:`), resp || "(No detail)"); } catch (e) { if (e instanceof ccxt.ExchangeError) { const msgL = e.message.toLowerCase(); if (msgL.includes("not modified") || msgL.includes("same leverage")) logger.info(`Leverage already ${leverage}x.`); else if (msgL.includes("position exists") || msgL.includes("open position")) logger.warn(c.yellow(`Cannot change leverage ${symbol} with open pos.`)); else if (msgL.includes("insufficient margin")) logger.error(c.red(`Failed setLev ${symbol}: insuf margin: ${e.message}`)); else if (msgL.includes("invalid leverage") || msgL.includes("limit")) { logger.error(c.red(`Failed setLev: Invalid val (${leverage}) or exceeds limits ${symbol}: ${e.message}`)); throw new Error(`Invalid leverage config: ${e.message}`); } else logger.error(c.red(`Failed setLev ${symbol}: ${e.constructor.name} - ${e.message}`), e.stack); } else if (e instanceof ccxt.AuthenticationError) { logger.error(c.red(`Auth fail setting lev: ${e.message}. Check API perms.`)); throw new Error(`Auth fail setting lev: ${e.message}`); } else if (e instanceof ccxt.NetworkError) logger.warn(c.yellow(`Could not set lev due to conn issue: ${e.message}. Using current.`)); else { logger.error(c.red(`Unexpected err setLev: ${e.constructor.name} - ${e.message}`), e.stack); throw new Error(`Unexpected err setting lev: ${e.message}`); } } }
    async fetchOhlcv(limit=100) { const cacheName = "ohlcv"; const k=`${this.config.symbol}_${this.config.timeframe}_${limit}`; const cached = this._getCache(cacheName, k); if (cached) return cached; logger.debug(`Fetch ${limit} OHLCV ${this.config.symbol} ${this.config.timeframe}...`); const fetchFn = async () => { if (!this.exchange?.has?.['fetchOHLCV']) throw new ccxt.NotSupported("fetchOHLCV NA."); const params = {}; if (this.exchange.id === 'bybit' && ['swap','future'].includes(this.config.exchange_type)) params.category = 'linear'; return await this.exchange.fetchOHLCV(this.config.symbol, this.config.timeframe, undefined, limit, params); }; try { const d = await retryOnException(fetchFn, this.config.max_retries, this.config.retry_delay, undefined, 'fetchOhlcv'); if (!Array.isArray(d)) { logger.warn(`fetchOHLCV got non-array: ${typeof d}.`); return null; } if (d.length > 0 && (!Array.isArray(d[0]) || d[0].length < OHLCV_SCHEMA.length)) { logger.warn(`Malformed OHLCV struct. Need >=${OHLCV_SCHEMA.length} elems. Got:`, d[0]); return null; } const cleanedData = d.map(c => [c[0], c[1], c[2], c[3], c[4], c[5] ? Number(c[5]) : 0]); this._setCache(cacheName, k, cleanedData); if (cleanedData.length > 0) logger.debug(`Fetched ${cleanedData.length} OK. Last: ${new Date(cleanedData[cleanedData.length-1][OHLCV_INDEX.TIMESTAMP]).toISOString()}`); else logger.debug("Fetched 0 candles."); return cleanedData; } catch (e) { logger.error(c.red(`Fetch OHLCV FAIL ${this.config.symbol}: ${e.constructor.name} - ${e.message}`), e.stack); return null; } }
    async fetchOrderBook() { const cacheName = "order_book"; const d = this.config.order_book_depth; const k=`${this.config.symbol}_${d}`; const cached = this._getCache(cacheName, k); if (cached) return cached; logger.debug(`Fetch OB ${this.config.symbol} (d:${d})...`); const fetchFn = async () => { if (!this.exchange?.has?.['fetchOrderBook']) throw new ccxt.NotSupported("fetchOrderBook NA."); const params = {}; if (this.exchange.id === 'bybit' && ['swap','future'].includes(this.config.exchange_type)) params.category = 'linear'; return await this.exchange.fetchOrderBook(this.config.symbol, d, params); }; try { const ob = await retryOnException(fetchFn, this.config.max_retries, this.config.retry_delay, undefined, 'fetchOrderBook'); if (!ob || !Array.isArray(ob.bids) || !Array.isArray(ob.asks)) { logger.warn(c.yellow("fetchOrderBook got invalid struct."), ob); return null; } this._setCache(cacheName, k, ob); logger.debug(`Fetched OB OK: ${ob.bids.length} bids, ${ob.asks.length} asks.`); return ob; } catch (e) { logger.error(c.red(`Fetch OB FAIL ${this.config.symbol}: ${e.constructor.name} - ${e.message}`), e.stack); return null; } }
    async getPosition() { const cacheName="position"; const k=this.config.symbol; const cached=_getCache(cacheName,k); if (cached?.side!==undefined&&cached?.size!==undefined&&cached?.entryPrice!==undefined) return cached; if (cached) { logger.warn("Invalid pos cache. Fetching."); _setCache(cacheName,k,null); } const defRet={side:PositionSide.NONE,size:0.0,entryPrice:0.0}; if (this.config.dry_run) { logger.debug(c.magenta("DRY RUN: Simulating no open pos.")); _setCache(cacheName,k,defRet); return defRet; } logger.debug(`Fetching position ${k}...`); const fetchFn=async()=>{ if (!this.exchange) throw new Error("Exch not init."); let positions=[]; const syms=[k]; const params={}; if (this.exchange.id==='bybit' && ['swap','future'].includes(this.config.exchange_type)) { params.category='linear'; logger.debug(`Bybit V5 fetchPositions params: ${JSON.stringify(params)}`); } if (this.exchange.has['fetchPositions']) positions = await this.exchange.fetchPositions(syms, params); else if (this.exchange.has['fetchPosition']) { logger.debug("Attempt fetchPosition fallback."); try { const p=await this.exchange.fetchPosition(k,params); if (p?.symbol===k) positions=[p]; } catch(e) { if (!(e instanceof ccxt.NotSupported)) throw e; else throw new ccxt.NotSupported("fetchPositions/fetchPosition NA."); } } else throw new ccxt.NotSupported("Position fetching NA."); return positions.filter(p => p?.symbol === k); }; try { const relPos = await retryOnException(fetchFn,this.config.max_retries,this.config.retry_delay,undefined,'getPosition'); if (!Array.isArray(relPos) || relPos.length === 0) { logger.debug(`No open pos for ${k}.`); this._setCache(cacheName,k,defRet); return defRet; } let netSize=0.0; let totalVal=0.0; let totalAbsSize=0.0; const market = this.exchange.market(k); const minAmt=market?.limits?.amount?.min ?? 1e-9; const szTol=Math.max(1e-9, minAmt/100); for (const pos of relPos) { const info=pos.info||{}; const szStr=info.size??pos.contracts??pos.size??'0'; const side=(info.side||pos.side||'').toLowerCase(); const ePStr=info.avgPrice??pos.entryPrice??info.markPrice??'0'; let szNum=NaN, ePriceNum=NaN; try { szNum=parseFloat(szStr); ePriceNum=parseFloat(ePStr); if (isNaN(szNum)||isNaN(ePriceNum)) throw new Error("NaN parse"); } catch(pE) { logger.warn(c.yellow(`Cannot parse size/eP for pos entry (${szStr}/${ePStr}). Skip. Info:`), info); continue; } const absSz=Math.abs(szNum); if (absSz<szTol||side==='none') continue; let pDir=0; if (side==='long'||side==='buy') pDir=1; else if (side==='short'||side==='sell') pDir=-1; else continue; netSize+=pDir*absSz; totalVal+=absSz*ePriceNum; totalAbsSize+=absSz; logger.debug("Proc pos entry:", { sym:pos.symbol,side,size:szNum,eP:ePriceNum}); } if (totalAbsSize<szTol) { logger.debug("Net pos size negligible."); this._setCache(cacheName,k,defRet); return defRet; } const avgEP=totalVal/totalAbsSize; let finSide=PositionSide.NONE; let finSize=0.0; if (netSize>szTol) { finSide=PositionSide.LONG; finSize=netSize; logger.info(c.green(`NET LONG pos: Sz=${finSize.toFixed(8)}, AvgE=${avgEP.toFixed(4)}`)); } else if (netSize<-szTol) { finSide=PositionSide.SHORT; finSize=Math.abs(netSize); logger.info(c.red(`NET SHORT pos: Sz=${finSize.toFixed(8)}, AvgE=${avgEP.toFixed(4)}`)); } else logger.debug("Net size neglig. No pos."); const result={side:finSide,size:finSize,entryPrice:avgEP}; this._setCache(cacheName,k,result); return result; } catch (e) { if (e instanceof ccxt.AuthenticationError) logger.error(c.red(`Auth fail fetching pos: ${e.message}.`)); else logger.error(c.red(`Fetch/Proc pos FAIL ${k}: ${e.constructor.name} - ${e.message}`), e.stack); this._setCache(cacheName,k,defRet); return defRet; } }
    async getBalance(currency=null) { const targetCcy=(currency||this.config.currency).toUpperCase(); const cacheName="balance"; const k=`${targetCcy}_${this.config.exchange_type}`; const cached=this._getCache(cacheName,k); if (cached!==null&&typeof cached==='number'&&!isNaN(cached)) return cached; if (cached!==null) { logger.warn(`Invalid balance cache type (${typeof cached}). Fetching.`); this._setCache(cacheName,k,null); } const defRet=null; if (this.config.dry_run) { const simBal=10000.0; logger.debug(c.magenta(`DRY: Sim balance ${simBal.toFixed(2)} ${targetCcy}`)); this._setCache(cacheName,k,simBal); return simBal; } logger.debug(`Fetch bal/eq ${targetCcy} (Type:${this.config.exchange_type})...`); const fetchFn=async()=>{ if (!this.exchange?.has?.['fetchBalance']) throw new ccxt.NotSupported("fetchBalance NA."); const params={}; if (this.exchange.id==='bybit') { if (['swap','future'].includes(this.config.exchange_type)) params.accountType='UNIFIED'; else if (this.config.exchange_type==='spot') params.accountType='SPOT'; } return await this.exchange.fetchBalance(params); }; try { const balInfo=await retryOnException(fetchFn,this.config.max_retries,this.config.retry_delay,undefined,'getBalance'); if (!balInfo) { logger.warn("fetchBalance got null/empty. Assume null balance."); this._setCache(cacheName, k, defRet); return defRet; } let finalBal=null; let found=false; logger.debug("Attempt parse balance/equity:", inspect(balInfo,{depth:2,colors:true})); if (!found && balInfo.info?.result?.list?.length > 0) { logger.debug("Try Bybit V5 info.result.list..."); let targetTypes=[]; if (['swap','future'].includes(this.config.exchange_type)) targetTypes=['UNIFIED','CONTRACT']; else if (this.config.exchange_type==='spot') targetTypes=['SPOT','FUND']; let accData=null; for (const type of targetTypes) { accData = balInfo.info.result.list.find(i=>i.coin===targetCcy && i.accountType===type); if(accData) break;} if (!accData) accData=balInfo.info.result.list.find(i=>i.coin===targetCcy); const eqKeys=['equity','accountEquity','totalEquity','walletBalance']; if (accData) { for (const key of eqKeys) { if (accData[key]!==undefined&&accData[key]!==null&&accData[key]!=='') { const bal=parseFloat(String(accData[key])); if (!isNaN(bal)) { finalBal=bal; found=true; logger.debug(`Found via info.result.list key '${key}': ${bal}`); if (key==='walletBalance' && ['swap','future'].includes(this.config.exchange_type)) logger.warn("Using walletBalance for derivs (no UPL)."); break; } } } } else { logger.debug(`No entry for ${targetCcy} in info.result.list.`); } } if (!found && balInfo[targetCcy]?.total !== undefined) { const bal=parseFloat(String(balInfo[targetCcy].total)); if (!isNaN(bal)) { finalBal=bal; found=true; logger.debug(`Found via std CCXT total: ${bal}`); if (['swap','future'].includes(this.config.exchange_type)) logger.warn(`Using std total for derivs. Verify UPL inclusion.`); } } if (!found && balInfo[targetCcy]?.free !== undefined && balInfo[targetCcy]?.used !== undefined) { const free=parseFloat(String(balInfo[targetCcy].free)); const used=parseFloat(String(balInfo[targetCcy].used)); if (!isNaN(free) && !isNaN(used)) { finalBal=free+used; found=true; logger.warn(`Used fallback balance (free+used) for ${targetCcy}: ${finalBal}. WALLET BAL only.`); } } if (!found && balInfo.total?.[targetCcy] !== undefined) { const bal=parseFloat(String(balInfo.total[targetCcy])); if (!isNaN(bal)) { finalBal=bal; found=true; logger.debug(`Found via top-level total: ${bal}`); if (['swap','future'].includes(this.config.exchange_type)) logger.warn(`Using top-level total for derivs. Verify UPL.`); } } if (!found) { logger.warn(c.yellow(`Cannot determine balance/equity for ${targetCcy}. Assume null.`)); logger.debug("Raw balance resp:", balInfo); this._setCache(cacheName, k, defRet); return defRet; } logger.info(c.green(`Fetched bal/eq ${targetCcy}: ${finalBal !== null ? finalBal.toFixed(4) : 'N/A'}`)); this._setCache(cacheName, k, finalBal); return finalBal; } catch (e) { logger.error(c.red(`Fetch balance ${targetCcy} FAIL: ${e.constructor.name} - ${e.message}`), e.stack); this._setCache(cacheName, k, defRet); return defRet; } }
    async getCurrentPrice() { const cacheName="ticker"; const k=this.config.symbol; const cached = this._getCache(cacheName,k); if (cached!==null&&typeof cached==='number'&&!isNaN(cached)&&cached>0) return cached; if (cached!==null) { logger.warn(`Invalid ticker cache (${typeof cached}, ${cached}). Fetching.`); this._setCache(cacheName,k,null); } logger.debug(`Fetch ticker ${k}...`); const defRet=null; const fetchFn=async()=>{ if (!this.exchange?.has?.['fetchTicker']) throw new ccxt.NotSupported("fetchTicker NA."); const params={}; if (this.exchange.id==='bybit' && ['swap','future'].includes(this.config.exchange_type)) params.category='linear'; return await this.exchange.fetchTicker(k, params); }; try { const ticker = await retryOnException(fetchFn,this.config.max_retries,this.config.retry_delay,undefined,'getCurrentPrice'); if (!ticker) { logger.warn("fetchTicker got null/empty."); this._setCache(cacheName,k,defRet); return defRet; } let lastPrice = null; let src=null; if (ticker.last!==undefined&&ticker.last!==null&&parseFloat(ticker.last)>0) { lastPrice=parseFloat(ticker.last); src='last'; } else if (ticker.close!==undefined&&ticker.close!==null&&parseFloat(ticker.close)>0) { lastPrice=parseFloat(ticker.close); src='close'; logger.debug("Use 'close' as 'last' unavailable."); } else if (ticker.mark!==undefined&&ticker.mark!==null&&parseFloat(ticker.mark)>0) { lastPrice=parseFloat(ticker.mark); src='mark'; logger.debug("Use 'mark' as 'last/close' unavailable."); } if (lastPrice===null||isNaN(lastPrice)||lastPrice<=0) { logger.warn(c.yellow(`Invalid price from ticker (src:${src||'N/A'}): ${lastPrice}. Ticker: `), ticker); this._setCache(cacheName,k,defRet); return defRet; } logger.debug(`Current price (src:${src}): ${lastPrice.toFixed(4)}`); this._setCache(cacheName, k, lastPrice); return lastPrice; } catch(e) { logger.error(c.red(`Fetch ticker ${k} FAIL: ${e.constructor.name} - ${e.message}`), e.stack); this._setCache(cacheName,k,defRet); return defRet; } }
}

// --- Indicator Calculations ---
class Indicators {
    static calculateAtr(ohlcv, period) { if (!Number.isInteger(period) || period <= 0) { logger.warn(`Invalid ATR period ${period}`); return null; } const req = period + 1; if (!Array.isArray(ohlcv) || ohlcv.length < req) { logger.debug(`Need ${req} candles for ATR(${period}), got ${ohlcv?.length}`); return null; } const [hi,li,ci]=[OHLCV_INDEX.HIGH,OHLCV_INDEX.LOW,OHLCV_INDEX.CLOSE]; const trs=[]; try { for (let i=ohlcv.length-period; i<ohlcv.length; i++) { const h=parseFloat(ohlcv[i][hi]); const l=parseFloat(ohlcv[i][li]); const pc=parseFloat(ohlcv[i-1][ci]); if (isNaN(h)||isNaN(l)||isNaN(pc)) throw new Error(`NaN OHLCV at idx ${i}`); const tr=Math.max(h-l,Math.abs(h-pc),Math.abs(l-pc)); trs.push(tr); } if (trs.length!==period) throw new Error(`TR length mismatch: ${trs.length} vs ${period}`); const atr = trs.reduce((s,v)=>s+v,0) / period; if (isNaN(atr)||atr<=1e-12) { logger.warn(`Invalid ATR result ${atr}`); return null; } return atr; } catch(e) { logger.error(c.red(`ATR Calc Error: ${e.message}`), e.stack); return null; } }
    static calculateSupertrend(ohlcv, period, multiplier) { const req = period + 3; if (!Array.isArray(ohlcv)||ohlcv.length<req) { logger.debug(`Need ${req} for simp ST(${period}), got ${ohlcv?.length}`); return {value:null,isUptrend:null}; } if (!Number.isInteger(period)||period<=0||isNaN(multiplier)||multiplier<=0) { logger.warn(`Invalid ST params: p=${period}, m=${multiplier}`); return {value:null,isUptrend:null}; } const [hi,li,ci] = [...[OHLCV_INDEX.HIGH,OHLCV_INDEX.LOW,OHLCV_INDEX.CLOSE]]; try { const atrSlice = ohlcv.slice(0,-1); const atr = Indicators.calculateAtr(atrSlice, period); if (atr === null) { logger.warn(`ATR fail for ST calc.`); return {value:null,isUptrend:null}; } const latest=ohlcv[ohlcv.length-1]; const prev=ohlcv[ohlcv.length-2]; const prevPrev=ohlcv[ohlcv.length-3]; const [h,l,c]=[latest[hi],latest[li],latest[ci]].map(Number); const [ph,pl,pc]=[prev[hi],prev[li],prev[ci]].map(Number); const ppc=Number(prevPrev[ci]); if ([h,l,c,ph,pl,pc,ppc].some(isNaN)) throw new Error("NaN in OHLCV for ST"); const hl2=(h+l)/2.0; const bUpper=hl2+multiplier*atr; const bLower=hl2-multiplier*atr; let prevTrendUpApprox=null; const prevAtrSlice=ohlcv.slice(0,-2); const prevAtr=Indicators.calculateAtr(prevAtrSlice, period); let pFinalUpperApprox=null, pFinalLowerApprox=null; if (prevAtr!==null) { const pHL2=(ph+pl)/2.0; const pBUpper=pHL2+multiplier*prevAtr; const pBLower=pHL2-multiplier*prevAtr; let trendBeforePrev=null; if (ppc>pBLower) trendBeforePrev=true; else if (ppc<pBUpper) trendBeforePrev=false; if (trendBeforePrev===true) { pFinalLowerApprox=pBLower; if (pc<pFinalLowerApprox) { prevTrendUpApprox=false; pFinalUpperApprox=pBUpper; } else { prevTrendUpApprox=true; pFinalLowerApprox=Math.max(pFinalLowerApprox, pBLower); } } else if (trendBeforePrev===false) { pFinalUpperApprox=pBUpper; if (pc>pFinalUpperApprox) { prevTrendUpApprox=true; pFinalLowerApprox=pBLower; } else { prevTrendUpApprox=false; pFinalUpperApprox=Math.min(pFinalUpperApprox, pBUpper); } } else { if (pc>pBLower) prevTrendUpApprox=true; else if (pc<pBUpper) prevTrendUpApprox=false; if(prevTrendUpApprox!==null) {pFinalLowerApprox=pBLower;pFinalUpperApprox=pBUpper;} } } if (prevTrendUpApprox === null) { if (pc > bLower) prevTrendUpApprox = true; else if (pc < bUpper) prevTrendUpApprox = false; if(prevTrendUpApprox!==null) {pFinalLowerApprox=bLower;pFinalUpperApprox=bUpper;} else logger.debug("Could not approx prev ST trend."); } logger.debug(`ST Prev Approx: TrendUp=${prevTrendUpApprox}, Lwr=${pFinalLowerApprox?.toFixed(4)}, Upr=${pFinalUpperApprox?.toFixed(4)}`); let currentStValue=null; let currentTrendUp=null; if(prevTrendUpApprox===null) { if(c>bLower){currentTrendUp=true;currentStValue=bLower;} else if(c<bUpper){currentTrendUp=false;currentStValue=bUpper;} else return {value:null,isUptrend:null}; } else { if(prevTrendUpApprox===true) {const fL=Math.max(bLower, pFinalLowerApprox??-Infinity); if(c<fL){currentTrendUp=false;currentStValue=bUpper;} else {currentTrendUp=true;currentStValue=fL;} } else {const fU=Math.min(bUpper, pFinalUpperApprox??Infinity); if(c>fU){currentTrendUp=true;currentStValue=bLower;} else {currentTrendUp=false;currentStValue=fU;} } } if(currentStValue===null||isNaN(currentStValue)||currentStValue<=0) { logger.warn(`Invalid ST val ${currentStValue}. Null.`); return {value:null,isUptrend:null}; } return {value:currentStValue, isUptrend:currentTrendUp}; } catch(e) { logger.error(c.red(`Simp ST Calc Err: ${e.message}`), e.stack); return {value:null,isUptrend:null}; } }
    static calculateVolumeRatio(ohlcv, shortP, longP) { if (!Number.isInteger(shortP)||!Number.isInteger(longP)||shortP<=0||longP<=0||shortP>=longP) { logger.warn(`Invalid Vol Ratio periods: S=${shortP}, L=${longP}`); return null; } if (!Array.isArray(ohlcv)||ohlcv.length<longP) { logger.debug(`Need ${longP} for Vol Ratio(${shortP}/${longP}), got ${ohlcv?.length}`); return null; } const vi=OHLCV_INDEX.VOLUME; try { const vols = ohlcv.slice(-longP).map(c=>{const v=parseFloat(c[vi]); if(isNaN(v)||v<0) throw new Error(`Invalid vol ${c[vi]}`); return v;}); if (vols.length!==longP) throw new Error(`Vol extract mismatch`); const shortVols = vols.slice(-shortP); if (shortP===0) return null; const shortAvg = shortVols.reduce((s,v)=>s+v,0)/shortP; if (longP===0) return null; const longAvg = vols.reduce((s,v)=>s+v,0)/longP; if (longAvg<=1e-12) { logger.debug("Long Vol Avg negligible. Ratio NA."); return null; } const ratio = shortAvg/longAvg; if (isNaN(ratio)||ratio<0) { logger.warn(`Invalid vol ratio ${ratio}`); return null; } return ratio; } catch(e) { logger.error(c.red(`Vol Ratio Calc Err: ${e.message}`), e.stack); return null; } }
    static calculateOrderBookPressure(ob, depth) { if (!Number.isInteger(depth)||depth<=0) { logger.warn(`Invalid OB depth ${depth}`); return null; } if (!ob?.bids?.length || !ob?.asks?.length) { logger.debug("Invalid OB data for pressure."); return null; } try { const isValid=(l)=>Array.isArray(l)&&l.length>=2&&typeof l[1]==='number'&&!isNaN(l[1])&&l[1]>0; const bVols = ob.bids.slice(0,depth).filter(isValid).reduce((s,l)=>s+l[1],0); const aVols = ob.asks.slice(0,depth).filter(isValid).reduce((s,l)=>s+l[1],0); const totalVol = bVols+aVols; if (totalVol<=1e-12) { logger.debug(`OB top ${depth} vol negligible.`); return null; } const pressure = bVols/totalVol; return pressure; } catch(e) { logger.error(c.red(`OB Pressure Calc Err: ${e.message}`), e.stack); return null; } }
}

// --- Order Manager ---
class OrderManager {
    constructor(exchangeMgr, config, notificationSvc) { this.exchangeMgr = exchangeMgr; this.config = config; this.notifier = notificationSvc; this.active_sl_order_id = null; }
    async initialize() { await this._loadState(); }
    async _loadState() { const sf=this.config.state_file; try { await fs.stat(sf); logger.info(`Loading state from ${sf}...`); const data=await fs.readFile(sf,'utf-8'); const state=JSON.parse(data); const loadedId=state?.active_sl_order_id; if(loadedId&&typeof loadedId==='string'&&loadedId.trim()){this.active_sl_order_id=loadedId.trim();logger.info(`Loaded active SL ID: ${c.bold(this.active_sl_order_id)}.`);} else {logger.info("No valid SL ID in state.");this.active_sl_order_id=null;} } catch (err) { if (err.code==='ENOENT'){logger.info(`State file '${sf}' not found. Init empty.`);this.active_sl_order_id=null;} else if (err instanceof SyntaxError) {logger.error(c.red(`Parse JSON state fail ${sf}: ${err.message}. Reset state.`));this.active_sl_order_id=null;} else { logger.error(c.red(`Read state fail ${sf}: ${err.message}. Reset state.`),err);this.active_sl_order_id=null;} } }
    async _saveState() { const sf=this.config.state_file; const valToSave=(this.active_sl_order_id&&typeof this.active_sl_order_id==='string'&&this.active_sl_order_id.trim()) ? this.active_sl_order_id.trim() : null; logger.debug(`Saving state (Active SL ID: ${valToSave || 'None'}) to ${sf}...`); const stateData={active_sl_order_id:valToSave}; const tempFile=sf + ".tmp."+process.pid; try { await fs.writeFile(tempFile,JSON.stringify(stateData,null,4),'utf-8'); await fs.rename(tempFile,sf); logger.debug("State saved OK."); } catch (err) { logger.error(c.red(`Write state FAIL ${sf}: ${err.message}`),err); try { await fs.unlink(tempFile); } catch(rmErr){if(rmErr.code!=='ENOENT'){logger.error(c.red(`Del temp state fail ${tempFile}: ${rmErr.message}`));}}} }
    async _executeExchangeRequest( exchangeApiCall, description, allowedRetryEx=[ccxt.NetworkError,ccxt.RequestTimeout,ccxt.ExchangeNotAvailable,ccxt.DDoSProtection], isONFOk=false ) { const wrapped=async()=>{if(!this.exchangeMgr?.exchange) throw new Error("ExchMgr/CCXT not init."); return await exchangeApiCall(this.exchangeMgr.exchange);}; try { logger.info(`Attempt Exch Req: ${c.yellow(description)}`); const result = await retryOnException(wrapped, this.config.max_retries, this.config.retry_delay, allowedRetryEx, description); const isPlace=description.toLowerCase().includes('place')||description.toLowerCase().includes('create'); const isCancel=description.toLowerCase().includes('cancel'); const isFetch=description.toLowerCase().includes('fetch'); if (isPlace && result?.id) { let logMsg = `${c.green("Order Place OK:")} ${description} -> ID=${c.bold(result.id)},St=${result.status||'N/A'},Sym=${result.symbol||'N/A'},Sd=${result.side||'N/A'},Ty=${result.type||'N/A'},Amt=${result.amount||'N/A'}`; if(result.price)logMsg+=`,P=${result.price}`; if(result.average)logMsg+=`,Avg=${result.average}`; if(result.stopPrice)logMsg+=`,SP=${result.stopPrice}`; logger.info(logMsg); logger.debug("Full order resp info:", result.info || result); return result; } else if (isCancel||isFetch) { logger.info(c.green(`Exch Req '${description}' OK.`)); logger.debug(`Resp '${description}':`, result); return result; } else if (isPlace) { logger.error(c.red(`Order place '${description}' OK but invalid resp (no ID?):`), result); this.notifier.sendSms(`ALERT: ${this.config.symbol} Order '${description}' placed but resp invalid? Check exch!`, this.config); return null; } else { logger.info(c.green(`Exch Req '${description}' OK.`)); logger.debug(`Resp '${description}':`, result); return result; } } catch (e) { logger.error(c.red(`Exch Req '${description}' FAILED: ${e.constructor.name} - ${e.message}`)); if (e instanceof ccxt.InsufficientFunds) { logger.error(c.red(`Reason: Insuf Funds.`)); this.notifier.sendSms(`ALERT: ${this.config.symbol} Order '${description.substring(0,50)}...' fail: Insuf Funds`, this.config); } else if (e instanceof ccxt.InvalidOrder) { logger.error(c.red(`Reason: Invalid order params. Check limits/precision.`), e); this.notifier.sendSms(`ERROR: ${this.config.symbol} Ord '${description.substring(0,50)}...' fail (Inv Params): ${e.message.substring(0,80)}`, this.config); } else if (e instanceof ccxt.OrderNotFound) { if(isONFOk){logger.warn(c.yellow(`Reason: Ord not found (Treated OK for '${description}').`));return null;} else {logger.error(c.red(`Reason: Ord not found (Unexpected for '${description}').`), e); this.notifier.sendSms(`ERROR: ${this.config.symbol} Ord NF unexpected (${description.substring(0,50)}...).`, this.config); } } else if (e instanceof ccxt.AuthenticationError) { logger.error(c.red(`Reason: Auth error! Check API Keys/Perms.`),e); this.notifier.sendSms("CRIT: Bot Auth Error! Trading HALTED.", this.config); throw e; // Halt bot on auth error } else if (e instanceof ccxt.PermissionDenied) { logger.error(c.red(`Reason: Permission Denied. Check API key perms.`),e); this.notifier.sendSms(`ERROR: ${this.config.symbol} Req '${description.substring(0,50)}...' fail: Perm Denied. Check API perms.`, this.config); } else if (e instanceof ccxt.ExchangeError) { logger.error(c.red(`Reason: Exchange Error. Details below.`), e); this.notifier.sendSms(`ERROR: ${this.config.symbol} Req '${description.substring(0,50)}...' fail (Exch Err): ${e.message.substring(0,80)}`, this.config); } else if (allowedRetryEx.some(exc=>e instanceof exc)) { logger.error(c.red(`Reason: Req failed after retries (${e.constructor.name}).`)); this.notifier.sendSms(`WARN: ${this.config.symbol} Req '${description.substring(0,50)}...' fail after retries (${e.constructor.name}).`, this.config); } else { logger.error(c.red(`Reason: Unexpected error. Details below.`), e.stack); this.notifier.sendSms(`CRIT: ${this.config.symbol} Unexpected err req '${description.substring(0,50)}...': ${e.constructor.name}. Check logs.`, this.config); } return null; } }
    async placeMarketOrder(side, amount, priceForSignals, slPrice=null, tpPrice=null) { const sym=this.config.symbol; const ex=this.exchangeMgr.exchange; if (!ex) { logger.error(c.red("No exch init.")); return null; } if (amount<=0) { logger.error(c.red(`Order Amt <= 0 (${amount}).`)); return null; } if (![Side.BUY,Side.SELL].includes(side)) { logger.error(c.red(`Invalid side ${side}.`)); return null; } if (isNaN(priceForSignals)||priceForSignals<=0) { logger.error(c.red(`Invalid priceForSig ${priceForSignals}.`)); return null; } try { const market = ex.market(sym); if(!market) throw new Error(`Market ${sym} NF.`); const amountStr = ex.amountToPrecision(sym, amount); const slStr = slPrice!==null ? ex.priceToPrecision(sym, slPrice) : null; const tpStr = tpPrice!==null ? ex.priceToPrecision(sym, tpPrice) : null; const amountNum = parseFloat(amountStr); const slNum = slStr!==null ? parseFloat(slStr) : null; const tpNum = tpStr!==null ? parseFloat(tpStr) : null; const minAmt = market?.limits?.amount?.min; const szTol = Math.max(1e-9, (minAmt??1e-9)/100); if (amountNum < szTol) { logger.error(c.red(`Order Amt ${amount} (${amountNum}) negligible.`)); return null; } if (minAmt!==undefined && amountNum < minAmt) { logger.error(c.red(`Order Amt ${amountNum} < min ${minAmt}.`)); this.notifier.sendSms(`ERR: ${sym} Ord fail. Amt ${amountNum} < min ${minAmt}.`, this.config); return null; } const params = {'timeInForce':this.config.time_in_force,'reduceOnly':false,'positionIdx':0}; if (ex.id==='bybit'&&['swap','future'].includes(this.config.exchange_type)) params.category='linear'; if (slStr) { if ((side===Side.BUY&&slNum>=priceForSignals)||(side===Side.SELL&&slNum<=priceForSignals)) { logger.error(c.red(`Invalid SL ${slStr} vs price ${priceForSignals}.`)); this.notifier.sendSms(`ERR: ${sym} Ord fail. Invalid SL ${slStr} vs ${priceForSignals}.`, this.config); return null; } params.stopLoss=slStr; params.slTriggerBy=this.config.order_trigger_price_type; } if (tpStr) { if ((side===Side.BUY&&tpNum<=priceForSignals)||(side===Side.SELL&&tpNum>=priceForSignals)) { logger.error(c.red(`Invalid TP ${tpStr} vs price ${priceForSignals}.`)); this.notifier.sendSms(`ERR: ${sym} Ord fail. Invalid TP ${tpStr} vs ${priceForSignals}.`, this.config); return null; } params.takeProfit=tpStr; params.tpTriggerBy=this.config.order_trigger_price_type; } if (slStr||tpStr) params.tpslMode="Full"; const desc=`Place Mkt ${side.toUpperCase()} ${amountStr} ${sym}|SL:${slStr||'N'}|TP:${tpStr||'N'}`; logger.info(`Prep Ord: ${desc}`); logger.debug(`Params: ${JSON.stringify(params)}`); if(this.config.dry_run){logger.info(c.magenta(`DRY RUN: Sim ${desc}`)); const simAvg=priceForSignals; if(!simAvg||simAvg<=0) {logger.error(c.red("DRY Err: Inv priceForSig")); return null;} const simOrd={id:`dry_${Date.now()}`,symbol:sym,side:side,type:"market",amount:amountNum,filled:amountNum,price:null,average:simAvg,cost:amountNum*simAvg,status:"closed",timestamp:Date.now(),datetime:new Date().toISOString(),info:{simulated:true,sl:slStr,tp:tpStr,params:params}}; logger.info(c.magenta(`DRY: Mkt ord ${simOrd.id} sim filled @ ~${simAvg.toFixed(4)}.`)); if(this.active_sl_order_id){logger.debug("DRY: Clear old SL ID.");this.active_sl_order_id=null;await this._saveState();} this.notifier.sendSms(`DRY: Placed ${side.toUpperCase()} ${amountStr} ${sym}`, this.config); return simOrd;} const placeFn=async(x)=>await x.createOrder(sym,'market',side,amountNum,undefined,params); const order=await this._executeExchangeRequest(placeFn,desc,[ccxt.NetworkError,ccxt.RequestTimeout]); if(order?.id){logger.info(c.green(`Mkt ord ${order.id} placed OK.`)); if(this.active_sl_order_id){logger.info(`Clear tracked SL ID ${this.active_sl_order_id} as new entry placed.`);this.active_sl_order_id=null;await this._saveState();} this.notifier.sendSms(`${sym}: ${side.toUpperCase()} ${amountStr} ord placed. ID:${order.id||'N/A'}`,this.config); return order;} else {logger.error(c.red(`Failed place mkt ord ${desc}.`));return null;} } catch (e) { logger.error(c.red(`Err prep/place mkt ord ${sym}: ${e.message}`), e.stack); this.notifier.sendSms(`ERR: Prep ${side} ord fail ${sym}: ${e.message.substring(0,100)}`, this.config); return null; } }
    async closePosition(posSide, amount) { if(posSide===PositionSide.NONE||amount<=0) {logger.warn(`Invalid close call: Sd=${posSide}, Amt=${amount}.`);return null;} const sym=this.config.symbol; const ex=this.exchangeMgr.exchange; if(!ex){logger.error("No exch for close.");return null;} const closeSD = (posSide===PositionSide.LONG) ? Side.SELL : Side.BUY; const desc=`Close ${posSide.toUpperCase()} (Mkt ${closeSD.toUpperCase()} ${amount.toFixed(8)} ${sym})`; logger.info(c.bold(`Attempt close: ${desc}`)); const cancelOk = await this._cancelActiveSlOrder(`Closing ${posSide} pos`); if(!cancelOk) { logger.error(c.red(`CRIT: Fail cancel SL ${this.active_sl_order_id || 'N/A'} before close ${posSide}. ABORT CLOSE. Manual check needed!`)); this.notifier.sendSms(`CRIT: CLOSE ABORT ${sym}. Fail cancel SL ${this.active_sl_order_id||'N/A'}. Check Exch!`, this.config); return null;} try { const market = ex.market(sym); if(!market) throw new Error(`Market ${sym} NF.`); const amtStr=ex.amountToPrecision(sym, amount); const amtNum=parseFloat(amtStr); const minAmt=market?.limits?.amount?.min; const szTol=Math.max(1e-9,(minAmt??1e-9)/100); if(amtNum<szTol){logger.error(c.red(`Close amt ${amount} invalid after prec (${amtNum}).`)); this.notifier.sendSms(`CRIT: ${sym} SL cancelled but CLOSE FAIL (amt ${amtNum} invalid). Manual check!`, this.config); return null;} if(minAmt!==undefined&&amtNum<minAmt){logger.error(c.red(`Close amt ${amtNum} < min ${minAmt}.`)); this.notifier.sendSms(`CRIT: ${sym} SL cancelled but CLOSE FAIL (amt ${amtNum} < min ${minAmt}). Manual check!`, this.config); return null;} if(this.config.dry_run){logger.info(c.magenta(`DRY RUN: Sim ${desc}`)); const curP=await this.exchangeMgr.getCurrentPrice(); const simAvg=curP; if(!simAvg||simAvg<=0){logger.error("DRY Err: No price for close sim."); return null;} const simOrd={id:`dry_close_${Date.now()}`,symbol:sym,side:closeSD,type:"market",amount:amtNum,filled:amtNum,price:null,average:simAvg,cost:amtNum*simAvg,status:"closed",reduceOnly:true,timestamp:Date.now(),datetime:new Date().toISOString(),info:{simulated:true,reduceOnly:true}}; logger.info(c.magenta(`DRY: Close ord ${simOrd.id} sim filled.`)); this.notifier.sendSms(`DRY: Closed ${posSide} ${amtStr} ${sym}`, this.config); return simOrd;} const params={'reduceOnly':true,'positionIdx':0}; if(ex.id==='bybit'&&['swap','future'].includes(this.config.exchange_type)) params.category='linear'; const closeFn=async(x)=>await x.createOrder(sym,'market',closeSD,amtNum,undefined,params); const order=await this._executeExchangeRequest(closeFn,desc,[ccxt.NetworkError,ccxt.RequestTimeout]); if (order?.id) { logger.info(c.green(`Pos close ord OK. ID: ${order.id}`)); this.notifier.sendSms(`${sym}: Close ord placed for ${posSide}.`, this.config); return order; } else { logger.error(c.red(`CRIT: FAIL place pos close ord ${desc}. Manual check REQ.`)); this.notifier.sendSms(`CRIT: FAILED to CLOSE ${posSide} pos ${sym}! Check Exch Manually NOW!`, this.config); return null; } } catch (e) { logger.error(c.red(`Err during close proc (post SL cancel): ${e.message}`), e.stack); this.notifier.sendSms(`CRIT: Err closing ${sym} post SL cancel: ${e.message.substring(0,100)}. Manual check!`, this.config); return null; } }
    async _cancelActiveSlOrder(reason="Unknown") { if(!this.active_sl_order_id){logger.debug("No active SL ID tracked.");return true;} const slID=this.active_sl_order_id; logger.info(`Attempt cancel active SL ${c.bold(slID)} Rsn: ${c.dim(reason)}`); if(this.config.dry_run){logger.info(c.magenta(`DRY: Sim cancel SL ${slIDA}.`));this.active_sl_order_id=null;await this._saveState();return true;} const cancelFn=async(x)=>{const params={};if(x.id==='bybit'&&['swap','future'].includes(this.config.exchange_type))params.category='linear';logger.debug(`Call cancelOrder: ID=${slID}, Sym=${this.config.symbol}, Params=${JSON.stringify(params)}`);return await x.cancelOrder(slID,this.config.symbol,params);}; const cancelRes=await this._executeExchangeRequest(cancelFn,`Cancel SL Order ${slID}`,undefined,true); if(cancelRes!==null){logger.info(c.green(`OK req cancel SL ${slID}.`));this.active_sl_order_id=null;await this._saveState();return true;} else {logger.warn(c.yellow(`Cancel req ${slID} null (Maybe Ok-NF or Fail). Re-check status...`)); try { const fetchFn=async(x)=>{const params={};if(x.id==='bybit'&&['swap','future'].includes(this.config.exchange_type))params.category='linear';return await x.fetchOrder(slID,this.config.symbol,params);}; const ordStat=await this._executeExchangeRequest(fetchFn,`Fetch SL ${slID} Status Post Cancel`,undefined,true); if (ordStat && ['open','untriggered'].includes(ordStat.status?.toLowerCase())) { logger.error(c.red(`Cancel SL ${slID} FAIL. Status still '${ordStat.status}'.`)); return false; } else { logger.info(c.green(`SL ${slID} confirm closed/gone (Stat:${ordStat?.status||'NF'}). Clear state.`)); this.active_sl_order_id=null; await this._saveState(); return true; } } catch(fetchErr) { logger.error(c.red(`Unexpected err re-check SL ${slID} status: ${fetchErr.message}. Assume fail.`)); return false; } } }
    async calculatePositionSize(entryPrice, atr, equity) { const ex=this.exchangeMgr.exchange; if(!ex){logger.error("No exch for size calc."); return null;} if(atr===null||isNaN(atr)||atr<=0){logger.warn(`Invalid ATR ${atr} for size.`); return null;} if(equity===null){logger.error("Equity null. Cannot size.");return null;} if(isNaN(equity)){logger.error("Equity NaN. Cannot size.");return null;} if(equity<=0&&!this.config.dry_run){logger.warn(`Eq <=0 (${equity.toFixed(2)}). Cannot size live.`); return null;} if(equity<=0&&this.config.dry_run){logger.warn(`DRY: Eq <=0. Calc proceed.`);} if(isNaN(entryPrice)||entryPrice<=0){logger.warn(`Invalid entry price ${entryPrice?.toFixed(4)} for size.`); return null;} if(isNaN(this.config.risk_per_trade)||!(this.config.risk_per_trade>0)){logger.error(`Invalid risk config ${this.config.risk_per_trade}.`);return null;} if(isNaN(this.config.sl_atr_mult)||this.config.sl_atr_mult<=0){logger.error(`Invalid SL mult config ${this.config.sl_atr_mult}.`);return null;} try { const riskAmtQ=equity*this.config.risk_per_trade; const slDistP=atr*this.config.sl_atr_mult; if(slDistP<=1e-12){logger.warn(`SL dist ${slDistP.toFixed(8)} negligible.`); return null;} if(riskAmtQ<=0&&!this.config.dry_run){logger.warn(`Risk amt <= 0 (${riskAmtQ.toFixed(2)}). Cannot size live.`); return null;} const initSizeB=riskAmtQ/slDistP; logger.debug(`Size Inputs: Eq=${equity.toFixed(2)}, R%=${(this.config.risk_per_trade*100).toFixed(2)}, E=${entryPrice.toFixed(4)}, ATR=${atr.toFixed(4)}, SLM=${this.config.sl_atr_mult}`); logger.debug(`Calc RiskAmtQ: ${riskAmtQ.toFixed(2)}, SLDistP: ${slDistP.toFixed(8)}, InitSizeB: ${initSizeB.toFixed(8)}`); if(initSizeB<=0){logger.warn(`Init size <=0 (${initSizeB.toFixed(8)}).`); return null;} const market=ex.market(this.config.symbol); if(!market) throw new Error(`Market ${this.config.symbol} NF.`); const limits=market.limits||{}; const amtL=limits.amount||{}; const costL=limits.cost||{}; const minAmt=amtL.min; const maxAmt=amtL.max; const minCost=costL.min; const maxCost=costL.max; const szTol=Math.max(1e-9,(minAmt??1e-9)/100); let adjSize=initSizeB; let curCost=adjSize*entryPrice; logger.debug(`Est Cost B4 Limits: ${curCost.toFixed(2)}`); if(minCost!==undefined&&curCost<minCost){logger.error(c.red(`Abort: Calc cost ${curCost.toFixed(2)} < min ${minCost}. Risk% too low?`)); this.notifier.sendSms(`WARN: ${this.config.symbol} Skip. Size < min cost limit ${minCost}.`, this.config); return null;} if(maxCost!==undefined&&curCost>maxCost){logger.warn(`Cost ${curCost.toFixed(2)} > max ${maxCost}. Reduce size.`); adjSize=maxCost/entryPrice; curCost=adjSize*entryPrice; logger.info(`Size reduced to ${adjSize.toFixed(8)} by max cost. New Cost: ${curCost.toFixed(2)}`);} logger.debug(`Size post CostL: ${adjSize.toFixed(8)}`); if(minAmt!==undefined&&adjSize<minAmt){logger.error(c.red(`Size ${adjSize.toFixed(8)} < min amt ${minAmt}.`)); this.notifier.sendSms(`WARN: ${this.config.symbol} Skip. Calc size ${adjSize.toFixed(8)} < min ${minAmt}.`, this.config); return null;} if(maxAmt!==undefined&&adjSize>maxAmt){logger.warn(`Size ${adjSize.toFixed(8)} > max amt ${maxAmt}. Clamp size.`); adjSize=maxAmt;} logger.debug(`Size B4 Prec: ${adjSize.toFixed(8)}`); const preciseSizeStr=ex.amountToPrecision(this.config.symbol,adjSize); const finalSize=parseFloat(preciseSizeStr); logger.debug(`Size AFTER Prec: ${finalSize.toFixed(8)} (${preciseSizeStr})`); if(finalSize<szTol){logger.error(`Final size ${finalSize.toFixed(8)} negligible post precision.`);return null;} if(minAmt!==undefined&&finalSize<minAmt){logger.error(c.red(`Final size ${finalSize.toFixed(8)} post prec < min amt ${minAmt}.`)); this.notifier.sendSms(`WARN: ${this.config.symbol} Skip. Final size ${finalSize.toFixed(8)} < min ${minAmt}.`, this.config); return null;} const finalCost=finalSize*entryPrice; logger.debug(`Final Est Cost: ${finalCost.toFixed(2)}`); if(minCost!==undefined&&finalCost<minCost){logger.error(c.red(`Final cost ${finalCost.toFixed(2)} (sz ${finalSize}) < min cost ${minCost}.`)); this.notifier.sendSms(`WARN: ${this.config.symbol} Skip. Final cost ${finalCost.toFixed(2)} < min ${minCost}.`, this.config); return null;} if(maxCost!==undefined&&finalCost>maxCost){logger.error(c.red(`Final cost ${finalCost.toFixed(2)} (sz ${finalSize}) > max cost ${maxCost}. Logic err?`)); return null;} logger.info(c.green(`Calculated Final Size: ${c.bold(finalSize.toFixed(8))} ${market.base||''}`)); return finalSize; } catch(e) { logger.error(c.red(`Err during size calc: ${e.message}`), e.stack); return null;} }
    async updateTrailingStop(posSide, posAmt, entryP, curP, curAtr) { const ex=this.exchangeMgr.exchange; if(!ex){logger.error("No exch for TSL."); return;} if(posSide===PositionSide.NONE||posAmt<=0){ if(this.active_sl_order_id){logger.warn(`No pos, but SL ID ${this.active_sl_order_id} tracked. Cancel orphan.`); await this._cancelActiveSlOrder("Orphan SL cleanup");} return;} if(curAtr===null||isNaN(curAtr)||curAtr<=0){logger.warn(`Invalid ATR ${curAtr} for TSL.`); return;} if(isNaN(this.config.trailing_stop_mult)||this.config.trailing_stop_mult<=0){logger.warn(`Invalid TSL mult ${this.config.trailing_stop_mult}.`); return;} if(isNaN(entryP)||entryP<=0){logger.warn(`Invalid entryP ${entryP} for TSL.`); return;} if(isNaN(curP)||curP<=0){logger.warn(`Invalid curP ${curP} for TSL.`); return;} const trailDist=curAtr*this.config.trailing_stop_mult; if(trailDist<=1e-12){logger.warn(`Trail dist ${trailDist.toFixed(8)} too small.`); return;} let potentialNewSL=null; let slSide=null; if(posSide===PositionSide.LONG){potentialNewSL=curP-trailDist;slSide=Side.SELL;} else {potentialNewSL=curP+trailDist;slSide=Side.BUY;} logger.debug(`TSL Calc: Pos=${posSide},CurP=${curP.toFixed(4)},ATR=${curAtr.toFixed(4)},TMult=${this.config.trailing_stop_mult},TDist=${trailDist.toFixed(4)},PotSL=${potentialNewSL.toFixed(4)},Entry=${entryP.toFixed(4)}`); if(potentialNewSL<=0){logger.warn(`Pot SL <=0 (${potentialNewSL.toFixed(4)}). Skip TSL.`); return;} let curActiveSLPrice=null; if(this.active_sl_order_id){logger.debug(`Check tracked SL ${this.active_sl_order_id}...`); try { const fetchSlFn=async(x)=>{const p={};if(x.id==='bybit'&&['swap','future'].includes(this.config.exchange_type))p.category='linear';return await x.fetchOrder(this.active_sl_order_id,this.config.symbol,p);}; const slInfo=await this._executeExchangeRequest(fetchSlFn,`Fetch Tracked SL ${this.active_sl_order_id} Status`,undefined,true); if(slInfo&&['open','untriggered'].includes(slInfo.status?.toLowerCase())){ if(slInfo.stopPrice!==undefined&&slInfo.stopPrice!==null){const parsedP=parseFloat(slInfo.stopPrice);if(!isNaN(parsedP)&&parsedP>0){curActiveSLPrice=parsedP;logger.debug(`Found active SL ${this.active_sl_order_id}. Trig:${curActiveSLPrice.toFixed(4)}`);} else logger.warn(`Cannot parse stopP ${slInfo.stopPrice} for SL ${this.active_sl_order_id}.`);} else logger.warn(`Active SL ${this.active_sl_order_id} no stopP.`); } else if (slInfo) { logger.info(`Tracked SL ${this.active_sl_order_id} status '${slInfo.status}'. Clear track.`); this.active_sl_order_id=null; await this._saveState(); } else { logger.warn(c.yellow(`Tracked SL ${this.active_sl_order_id} NF/gone. Clear track.`)); this.active_sl_order_id=null; await this._saveState(); } } catch(e) { logger.error(c.red(`Unexpected err fetch SL ${this.active_sl_order_id}: ${e.message}. Skip TSL cycle.`), e.stack); return; } } else logger.debug("No active SL tracked."); let shouldUpdate=false; const market=ex.market(this.config.symbol); const pTick=market?.precision?.price??1e-8; const pTol=pTick*2; let potNewSLFmt=null; try { potNewSLFmt=parseFloat(ex.priceToPrecision(this.config.symbol,potentialNewSL)); if(isNaN(potNewSLFmt)||potNewSLFmt<=0){logger.warn(`Pot SL ${potentialNewSL.toFixed(4)} invalid post fmt (${potNewSLFmt}). Skip TSL.`); return;} } catch(fmtErr){logger.error(c.red(`Err format SL price ${potentialNewSL}: ${fmtErr.message}. Skip TSL.`),fmtErr.stack);return;} if(curActiveSLPrice===null){ if(posSide===PositionSide.LONG&&(potNewSLFmt>entryP+pTol)){logger.info(c.cyan(`Cond Met: No active SL. New TSL ${potNewSLFmt.toFixed(4)} profitable > Entry ${entryP.toFixed(4)}. Place initial TSL.`)); shouldUpdate=true;} else if(posSide===PositionSide.SHORT&&(potNewSLFmt<entryP-pTol)){logger.info(c.cyan(`Cond Met: No active SL. New TSL ${potNewSLFmt.toFixed(4)} profitable < Entry ${entryP.toFixed(4)}. Place initial TSL.`)); shouldUpdate=true;} else logger.debug(`Cond NOT Met: No active SL, pot new SL ${potNewSLFmt.toFixed(4)} not profitable vs entry ${entryP.toFixed(4)}.`); } else { if(posSide===PositionSide.LONG){if(potNewSLFmt>curActiveSLPrice+pTol){logger.info(c.cyan(`Cond Met: Trail LONG. New SL ${potNewSLFmt.toFixed(4)} > Cur SL ${curActiveSLPrice.toFixed(4)}.`)); shouldUpdate=true;} else logger.debug(`Cond NOT Met: Pot TSL ${potNewSLFmt.toFixed(4)} LONG not > cur SL ${curActiveSLPrice.toFixed(4)}.`);} else {if(potNewSLFmt<curActiveSLPrice-pTol){logger.info(c.cyan(`Cond Met: Trail SHORT. New SL ${potNewSLFmt.toFixed(4)} < Cur SL ${curActiveSLPrice.toFixed(4)}.`)); shouldUpdate=true;} else logger.debug(`Cond NOT Met: Pot TSL ${potNewSLFmt.toFixed(4)} SHORT not < cur SL ${curActiveSLPrice.toFixed(4)}.`);} } if(shouldUpdate){logger.info(c.bold(`ACTION: Update TSL for ${posSide} pos.`)); try { const newSLStr=ex.priceToPrecision(this.config.symbol,potNewSLFmt); const amtStr=ex.amountToPrecision(this.config.symbol,posAmt); const amtNum=parseFloat(amtStr); const minAmt=market?.limits?.amount?.min; const szTol=Math.max(1e-9,(minAmt??1e-9)/100); const newSLNum=parseFloat(newSLStr); if(amtNum<szTol||newSLNum<=0){logger.error(`TSL abort: Invalid amt ${amtNum} or SL ${newSLStr}.`); return;} if(this.active_sl_order_id){logger.info(`Cancel prev SL ${this.active_sl_order_id} b4 new TSL @ ${newSLStr}.`); const cancelOk=await this._cancelActiveSlOrder(`TSL update to ${newSLStr}`); if(!cancelOk){logger.error(c.red("CRIT: Fail cancel prev SL. Abort TSL update. Pos may have old/no SL. Manual check!")); this.notifier.sendSms(`CRIT: TSL update ABORT ${this.config.symbol}. Fail cancel old SL ${this.active_sl_order_id}. Check pos!`, this.config); return;} } else {logger.debug("No active SL tracked, place new TSL.");} logger.info(`Placing new TSL: Side=${slSide}, Amt=${amtStr}, TrigP=${newSLStr}`); const slParams={'triggerDirection':(posSide===PositionSide.LONG)?2:1,'triggerBy':this.config.order_trigger_price_type,'reduceOnly':true,'closeOnTrigger':true,'orderType':"Market",'basePrice':ex.priceToPrecision(this.config.symbol,curP),'positionIdx':0}; if(ex.id==='bybit'&&['swap','future'].includes(this.config.exchange_type)) slParams.category='linear'; const basePNum=parseFloat(slParams.basePrice); if(isNaN(basePNum)||basePNum<=0){logger.error(c.red(`Invalid baseP ${slParams.basePrice} for TSL. Abort TSL.`));return;} const tslDesc=`Place TSL ${slSide.toUpperCase()} ${amtStr} @ ${newSLStr}`; logger.debug(`TSL Params: ${JSON.stringify(slParams)}`); const placeTslFn=async(x)=>await x.createOrder(this.config.symbol,'Stop',slSide,amtNum,undefined,{...slParams,'stopPrice':newSLNum}); const newSLOrd=await this._executeExchangeRequest(placeTslFn,tslDesc,[ccxt.NetworkError,ccxt.RequestTimeout]); if(newSLOrd?.id){this.active_sl_order_id=newSLOrd.id;await this._saveState();logger.info(c.green(`TSL OK. New active SL ID: ${c.bold(this.active_sl_order_id)}, Trig: ${newSLStr}`)); this.notifier.sendSms(`${this.config.symbol} TSL updated: Trig @ ${newSLStr}`, this.config);} else {logger.error(c.red("CRIT: FAIL place new TSL ord after cancel (if any). Pos UNPROTECTED.")); this.active_sl_order_id=null; await this._saveState(); this.notifier.sendSms(`CRIT: Failed place new TSL ${this.config.symbol} post update! Pos UNPROTECTED. Check Exch!`, this.config);} } catch(e) { logger.error(c.red(`Unexpected err TSL execute: ${e.message}`), e.stack); if(shouldUpdate&&!this.active_sl_order_id){logger.error(c.red("Clear SL state due to TSL place err after cancel."));this.active_sl_order_id=null;await this._saveState();} this.notifier.sendSms(`ERROR: Unexpected err update TSL ${this.config.symbol}. Check logs.`, this.config);} } // end if shouldUpdate
    }
}

// --- Trading Bot Class ---
class TradingBot {
    constructor() { this.config=null; this.notifier=null; this.exchangeMgr=null; this.orderMgr=null; this.indicators=Indicators; this.last_candle_ts=null; this.cycle_count=0; this.start_time=Date.now()/1000; this._stop_requested=false; this._isRunning=false; }
    async initialize() { logger.info(c.blue("----- Init Trading Bot -----")); try { this.config = new Config(); await setupLogging(this.config); this.notifier = new NotificationService(); this.exchangeMgr = new ExchangeManager(this.config); await this.exchangeMgr.initialize(); this.orderMgr = new OrderManager(this.exchangeMgr, this.config, this.notifier); await this.orderMgr.initialize(); logger.info(c.green(c.bold("Bot init complete."))); logger.info(`Pair: ${c.bold(this.config.symbol)}, TF: ${c.bold(this.config.timeframe)}`); if (this.config.dry_run) logger.info(c.magenta(c.bold("Dry Run Mode: ENABLED."))); else { logger.warn(c.red(c.bold("--- LIVE TRADING ACTIVE ---"))); logger.warn(c.yellow("Ensure config/risk tested!")); } } catch (e) { logger.error(c.red(`CRIT INIT ERROR: ${e.message}`), e.stack); this._attemptEmergencyNotification(`CRIT: Bot startup FAILED: ${e.message.substring(0,100)}`); throw e; } }
    _attemptEmergencyNotification(msg) { try { const cfg=this.config || DEFAULTS; const ntf=this.notifier || new NotificationService(); const canSms = cfg.sms_enabled && cfg.sms_recipient_number && (cfg.termux_sms_available!==undefined?cfg.termux_sms_available:false); if(canSms) ntf.sendSms(msg, cfg); else logger.error(c.red(`Emergency Notify Fail (SMS disabled/NA/cfg err): ${msg}`)); } catch (notifyErr) { logger.error(c.red(`Fail send crit notify: ${notifyErr.message}`), notifyErr.stack); } }
    _getRequiredOhlcvLimit() { const stNeed = Math.max(this.config.short_st_period, this.config.long_st_period)+3; const volNeed = this.config.volume_long_period; const atrNeed = this.config.long_st_period+1; const buffer=5; const calcNeed = Math.max(stNeed, volNeed, atrNeed) + buffer; const maxExch = this.exchangeMgr?.exchange?.limits?.fetchOHLCV?.max ?? 1000; const finalLimit = Math.min(calcNeed, maxExch); if(finalLimit < calcNeed) logger.warn(c.yellow(`Need ${calcNeed} candles > exch max ${maxExch}. Using ${finalLimit}. Init accuracy slight reduce.`)); return finalLimit; }
    _calculateSleepTimeMs() { try { const tfSec = this.exchangeMgr?.exchange?.parseTimeframe?.(this.config.timeframe); if (!tfSec || tfSec <= 0) { logger.error(c.red(`Parse TF fail '${this.config.timeframe}'. Default sleep 60s.`)); return 60000; } const tfMs = tfSec * 1000; const nowMs = Date.now(); const curIntStartMs = Math.floor(nowMs / tfMs) * tfMs; const nextIntStartMs = curIntStartMs + tfMs; const bufferMs = 3000; const targetWakeMs = nextIntStartMs + bufferMs; let sleepNeededMs = targetWakeMs - nowMs; if (sleepNeededMs <= 0) { const nextTargetWakeMs = nextIntStartMs + tfMs + bufferMs; sleepNeededMs = nextTargetWakeMs - nowMs; logger.warn(c.yellow(`Cycle exceeded TF! New Target=${new Date(nextTargetWakeMs).toISOString()}`)); } const finalSleepMs = Math.max(100, sleepNeededMs); logger.debug(`Timing: TF=${this.config.timeframe}(${tfSec}s). NextStart=${new Date(nextIntStartMs).toISOString()}. TargetWake=${new Date(targetWakeMs).toISOString()}. Sleep=${finalSleepMs.toFixed(0)}ms.`); return finalSleepMs; } catch (e) { logger.error(c.red(`Err calc sleep time: ${e.message}. Default sleep 60s.`), e.stack); return 60000; } }
    async tradeLogic() { if (!this.config || !this.notifier || !this.exchangeMgr || !this.orderMgr || !this.indicators) { logger.error("Crit Err: Components not init in tradeLogic. Stop."); this._stop_requested = true; return; } this.cycle_count++; logger.info(c.blue(`\n===== Cycle ${this.cycle_count} Start: ${new Date().toISOString()} =====`)); const ohlcvLimit=this._getRequiredOhlcvLimit(); logger.debug(`Fetch OHLCV (lim=${ohlcvLimit})...`); const ohlcv = await this.exchangeMgr.fetchOhlcv(ohlcvLimit); if (!ohlcv || ohlcv.length === 0) { logger.warn(c.yellow("No valid OHLCV fetched. Skip cycle.")); return; } const minNeed = Math.max(this.config.long_st_period+3, this.config.volume_long_period, this.config.long_st_period+1); if (ohlcv.length < minNeed) { logger.warn(c.yellow(`Insuff OHLCV (Got ${ohlcv.length}, Need >= ${minNeed}). Skip cycle.`)); return; } const curTs = ohlcv[ohlcv.length-1][OHLCV_INDEX.TIMESTAMP]; if (this.last_candle_ts !== null && curTs <= this.last_candle_ts) { logger.info(c.gray(`No new candle (${curTs} <= ${this.last_candle_ts}). Wait next cycle.`)); return; } logger.info(`New candle: ${c.bold(new Date(curTs).toISOString())} (${curTs})`); let posInfo, ob, tickerP, equity; try { posInfo = await this.exchangeMgr.getPosition(); const { side: posSideNow } = posInfo; [ob, tickerP, equity] = await Promise.all([ this.exchangeMgr.fetchOrderBook(), this.exchangeMgr.getCurrentPrice(), this.exchangeMgr.getBalance() ]); } catch (fetchErr) { logger.error(c.red(`Err fetch data batch: ${fetchErr.message}. Skip cycle.`), fetchErr.stack); return; } const { side: posSide, size: posAmt, entryPrice: posEntry } = posInfo; const priceSig = parseFloat(ohlcv[ohlcv.length-1][OHLCV_INDEX.CLOSE]); const priceTsl = tickerP ?? priceSig; if (isNaN(priceSig) || priceSig <= 0) { logger.error(c.red(`Invalid priceSig ${priceSig}. Skip cycle.`)); return; } logger.debug(`Ref Prices: SigPrice=${priceSig.toFixed(4)}, TSLPrice=${priceTsl?.toFixed(4) ?? 'N/A'}`); let atr=null, shortSt={v:null,up:null}, longSt={v:null,up:null}, volR=null, obP=null; let indOk=false; try { atr=this.indicators.calculateAtr(ohlcv,this.config.long_st_period); shortSt=this.indicators.calculateSupertrend(ohlcv,this.config.short_st_period,this.config.st_multiplier); longSt=this.indicators.calculateSupertrend(ohlcv,this.config.long_st_period,this.config.st_multiplier); volR=this.indicators.calculateVolumeRatio(ohlcv,this.config.volume_short_period,this.config.volume_long_period); obP=this.indicators.calculateOrderBookPressure(ob,this.config.order_book_depth); const fmtInd = (v,d=4)=>v!==null?v.toFixed(d):c.gray('N/A'); const fmtTr = (t)=>t===true?c.green('UP'):t===false?c.red('DOWN'):c.gray('N/A'); logger.info(`Indics: P=${c.bold(priceSig.toFixed(4))}, ATR=${c.dim(fmtInd(atr,6))}, ` + `ST(${this.config.short_st_period})=${fmtTr(shortSt.isUptrend)} (${c.dim(fmtInd(shortSt.value))}), ` + `ST(${this.config.long_st_period})=${fmtTr(longSt.isUptrend)} (${c.dim(fmtInd(longSt.value))}), ` + `VolR=${c.dim(fmtInd(volR,2))}, OBP=${c.dim(fmtInd(obP,3))}`); indOk = atr!==null && shortSt.isUptrend !== null && longSt.isUptrend !== null && volR !== null && obP !== null; if (!indOk) logger.warn(c.yellow("One+ indicators failed. Signal gen skipped.")); } catch (e) { logger.error(c.red(`Indicator calc phase err: ${e.message}. Skip cycle.`), e.stack); return; } let longE=false, shortE=false, longX=false, shortX=false; if (indOk) { logger.debug("Gen signals..."); longE=(shortSt.isUptrend===true && longSt.isUptrend===true && volR > this.config.volume_spike_threshold && obP > this.config.ob_pressure_threshold); shortE=(shortSt.isUptrend===false && longSt.isUptrend===false && volR > this.config.volume_spike_threshold && obP < (1.0 - this.config.ob_pressure_threshold)); longX=(posSide === PositionSide.LONG && shortSt.isUptrend === false); shortX=(posSide === PositionSide.SHORT && shortSt.isUptrend === true); logger.info(`Signals: LongE=${longE?c.green('T'):'f'}, ShortE=${shortE?c.red('T'):'f'}, LongX=${longX?c.red('T'):'f'}, ShortX=${shortX?c.green('T'):'f'}`); } const posSideCol = posSide===PositionSide.LONG?c.green(posSide):posSide===PositionSide.SHORT?c.red(posSide):c.gray(posSide); logger.info(`State: Pos=${posSideCol}, Sz=${c.bold(posAmt.toFixed(8))}, AvgE=${c.dim(posEntry.toFixed(4))}`); if (equity !== null) { logger.info(` Equity: ${c.bold(equity.toFixed(2))} ${this.config.currency}`); if(posSide===PositionSide.NONE&&equity<=0&&!this.config.dry_run) logger.warn(c.yellow("Equity <=0. No new trades live.")); } else logger.error(c.red("Equity unknown (fetch fail).")); let actionTaken=false; if (longX) { logger.info(c.bold(c.red("ACT: Long exit signal. Close LONG."))); const order = await this.orderMgr.closePosition(posSide, posAmt); if (order) actionTaken=true; } else if (shortX) { logger.info(c.bold(c.green("ACT: Short exit signal. Close SHORT."))); const order = await this.orderMgr.closePosition(posSide, posAmt); if (order) actionTaken=true; } if (!actionTaken && posSide === PositionSide.NONE) { if (equity===null) logger.error(c.red("Cannot enter: Equity unknown.")); else if (equity<=0 && !this.config.dry_run) logger.warn(c.yellow("Cannot enter: Equity <= 0.")); else if (longE) { logger.info(c.bold(c.green("ACT: Long entry signal."))); if (atr !== null) { const amt = await this.orderMgr.calculatePositionSize(priceSig, atr, equity); if (amt && amt > 0) { const sl = priceSig - atr * this.config.sl_atr_mult; const tp = priceSig + atr * this.config.tp_atr_mult; logger.info(`Place LONG Amt=${amt.toFixed(8)}, E~=${priceSig.toFixed(4)}, SL=${sl.toFixed(4)}, TP=${tp.toFixed(4)}`); const order = await this.orderMgr.placeMarketOrder(Side.BUY, amt, priceSig, sl, tp); if (order) actionTaken=true; } else logger.warn(c.yellow("Long entry: Size calc fail/zero. No order.")); } else logger.warn(c.yellow("Long entry: ATR invalid. No order.")); } else if (shortE) { logger.info(c.bold(c.red("ACT: Short entry signal."))); if (atr !== null) { const amt = await this.orderMgr.calculatePositionSize(priceSig, atr, equity); if (amt && amt > 0) { const sl = priceSig + atr * this.config.sl_atr_mult; const tp = priceSig - atr * this.config.tp_atr_mult; logger.info(`Place SHORT Amt=${amt.toFixed(8)}, E~=${priceSig.toFixed(4)}, SL=${sl.toFixed(4)}, TP=${tp.toFixed(4)}`); const order = await this.orderMgr.placeMarketOrder(Side.SELL, amt, priceSig, sl, tp); if (order) actionTaken=true; } else logger.warn(c.yellow("Short entry: Size calc fail/zero. No order.")); } else logger.warn(c.yellow("Short entry: ATR invalid. No order.")); } } if (!actionTaken && posSide !== PositionSide.NONE) { if (!isNaN(priceTsl) && priceTsl > 0 && atr !== null) { logger.info(c.cyan(`ACT: Hold ${posSide}. Check/Update TSL...`)); await this.orderMgr.updateTrailingStop(posSide, posAmt, posEntry, priceTsl, atr); } else logger.warn(c.yellow(`Skip TSL update due to invalid inputs: priceTsl (${priceTsl}), atr (${atr}).`)); } else if (!actionTaken && posSide === PositionSide.NONE) { logger.info(c.gray("ACT: Hold (Flat, No Entry Signal).")); } this.last_candle_ts=curTs; logger.debug(`Updated last candle TS: ${this.last_candle_ts}`); logger.info(c.blue(`===== Cycle ${this.cycle_count} End =====`)); }

    async run() { if (this._isRunning) { logger.warn("Bot already running."); return; } if (!this.config || !this.notifier || !this.exchangeMgr || !this.orderMgr) { logger.error("Cannot start: Components not init."); return; } this._isRunning = true; this._stop_requested = false; logger.info(c.bold(c.blue("----- Start Trading Bot Main Loop -----"))); let startupMsg = `Bot start: ${this.config.symbol} ${this.config.timeframe}`; if (this.config.dry_run) { startupMsg += " (DRY RUN)"; this.notifier.sendSms(startupMsg, this.config); } else { startupMsg += " (LIVE TRADING)"; this.notifier.sendSms(`ALERT: ${startupMsg}. Monitor!`, this.config); } while (!this._stop_requested) { const cycleStart = process.hrtime.bigint(); try { await this.tradeLogic(); const cycleEnd = process.hrtime.bigint(); const cycleDurMs = Number(cycleEnd - cycleStart) / 1e6; logger.debug(`Cycle time: ${cycleDurMs.toFixed(1)} ms.`); const healthChkInt = 60; if (this.cycle_count > 0 && this.cycle_count % healthChkInt === 0) { const uptimeS = (Date.now()/1000) - this.start_time; const uptimeStr = new Date(uptimeS * 1000).toISOString().slice(11,19); const avgCycleMs = (uptimeS * 1000) / this.cycle_count; try { const hcPos = await this.exchangeMgr.getPosition(); const hcEq = await this.exchangeMgr.getBalance(); const { side: hcSd, size: hcAmt, entryPrice: hcEP } = hcPos; const hcSdCol = hcSd===PositionSide.LONG?c.green(hcSd):hcSd===PositionSide.SHORT?c.red(hcSd):c.gray(hcSd); logger.info( c.blue(`--- Health Check --- Up=${uptimeStr}, Cyc=${this.cycle_count}, `) + c.dim(`AvgT=${avgCycleMs.toFixed(0)}ms | `) + c.blue(`Pos=${hcSdCol}(${hcAmt.toFixed(8)}@${hcEP.toFixed(4)}), Eq=${hcEq!==null?hcEq.toFixed(2):'N/A'} | `) + c.dim(`ActSL=${this.orderMgr.active_sl_order_id||'N'} ---`) ); } catch (hErr) { logger.error(c.red(`Err health check data: ${hErr.message}`), hErr.stack); } } const sleepDurMs = this._calculateSleepTimeMs(); const actualSleepMs = Math.max(50, sleepDurMs - cycleDurMs); if (actualSleepMs < 1000) logger.debug(`Short sleep: ${actualSleepMs.toFixed(0)}ms.`); await sleep(actualSleepMs); } catch (e) { logger.error(c.red(`CRIT MAIN LOOP ERR (Cyc ${this.cycle_count}): ${e.constructor.name} - ${e.message}`), e.stack); if (e instanceof ccxt.AuthenticationError || e instanceof ccxt.PermissionDenied) { const rsn=e instanceof ccxt.AuthenticationError?"Auth":"Perm"; logger.error(c.red(`${rsn} fail mid-run. Shutdown. Check API.`)); this.notifier.sendSms(`CRIT: Bot ${rsn} FAILED mid-run! Shutdown.`, this.config); this._stop_requested=true; } else if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeNotAvailable || e instanceof ccxt.DDoSProtection) { logger.error(c.red(`Persistent conn/timeout err: ${e.constructor.name}. Pause before retry...`)); this.notifier.sendSms(`WARN: Bot persistent conn issues (${e.constructor.name}). Pause.`, this.config); await sleep(Math.max(this.config.retry_delay * 5000, 30000)); } else { logger.error(c.red(`Unexpected crit err. Pause attempt continue.`)); this.notifier.sendSms(`CRIT BOT ERROR: ${e.constructor.name} - ${e.message.substring(0,100)}. Bot paused. Check logs!`, this.config); const pauseDur = Math.max(this.config.retry_delay * 10000, 60000); logger.info(`Pause ${pauseDur/1000}s due to crit err...`); await sleep(pauseDur); } } } logger.info(c.blue("Exit main loop.")); await this.attemptSafeShutdown(); await logger.closeLogFile(); this._isRunning=false; logger.info(c.bold(c.blue("----- Trading Bot Stopped -----"))); }
    stop() { if (!this._stop_requested) { logger.info(c.yellow("Graceful stop requested. Will shut down post-cycle.")); this._stop_requested = true; } else logger.info("Stop already requested."); }
    async attemptSafeShutdown() { logger.info(c.yellow("--- Init Safe Shutdown ---")); if (!this.config || !this.notifier || !this.exchangeMgr || !this.orderMgr) { logger.error(c.red("Cannot safe shutdown: Components not init. Manual check req.")); this._attemptEmergencyNotification("ERR: Bot shutdown fail (comp miss). Manual check!"); return; } if (this.config.dry_run) { logger.info(c.magenta("Dry run: No real pos to close.")); this.notifier.sendSms(`Bot shutdown (Dry Run) ${this.config.symbol}.`, this.config); return; } try { logger.info("Check open pos (Live Mode)..."); this.exchangeMgr._setCache('position', this.config.symbol, null); const { side: posSide, size: posAmount } = await this.exchangeMgr.getPosition(); if (posSide !== PositionSide.NONE && posAmount > 0) { logger.warn(c.yellow(`Open ${posSide.toUpperCase()} pos (${posAmount.toFixed(8)}) found. Attempt close market...`)); this.notifier.sendSms(`ALERT: Bot shutdown. Closing ${posSide} pos ${posAmount.toFixed(8)} ${this.config.symbol}.`, this.config); let closed=null; const attempts=2; for (let i=1; i<=attempts; i++) { logger.info(`Attempt close ord (Try ${i}/${attempts})...`); closed = await this.orderMgr.closePosition(posSide, posAmount); if (closed?.id) { logger.info(c.green(`Close ord OK shutdown (Try ${i}). ID: ${closed.id}`)); logger.info("Wait 5s for final check..."); await sleep(5000); this.exchangeMgr._setCache('position', this.config.symbol, null); const { side: finalSide } = await this.exchangeMgr.getPosition(); if (finalSide === PositionSide.NONE) { logger.info(c.green("Pos confirm CLOSED.")); this.notifier.sendSms(`${this.config.symbol} pos close OK shutdown.`, this.config); } else { logger.warn(c.yellow(`Pos still ${finalSide} after close ord! Manual verify!`)); this.notifier.sendSms(`WARN: ${this.config.symbol} pos MAY NOT be closed post shutdown ord! Check Exch!`, this.config); } break; } else { logger.error(c.red(`Close ord FAIL shutdown (Try ${i}).`)); if (i<attempts) { logger.info(`Wait ${this.config.retry_delay}s before retry...`); await sleep(this.config.retry_delay*1000); } } } if (!closed) { logger.error(c.red(c.bold("CRIT: FAIL close pos shutdown after attempts. Manual intervention REQ!"))); this.notifier.sendSms(`CRIT: FAILED close ${this.config.symbol} pos shutdown! Check Exch NOW!`, this.config); } } else { logger.info("No open pos found."); this.notifier.sendSms(`Bot shutdown (No pos) ${this.config.symbol}.`, this.config); } } catch (e) { logger.error(c.red(`Err safe shutdown: ${e.message}`), e.stack); this.notifier.sendSms(`ERR bot shutdown proc ${this.config.symbol}: ${e.message.substring(0,100)}. Manual check!`, this.config); } logger.info(c.yellow("--- Safe Shutdown Finished ---")); }
}

// --- Main Execution ---
async function main() {
    botInstance = new TradingBot(); // Assign to global
    try {
        await botInstance.initialize();
        await botInstance.run();
    } catch (e) { /* Init errors already handled, log again if needed */ logger.error(c.red(c.bold(`Bot start fail: ${e.message}`))); await logger.closeLogFile(); process.exitCode = 1; }
}

// --- Graceful Shutdown Handler ---
let isShuttingDown = false;
const shutdown = async (signal) => {
    console.log(`\n${c.yellow(c.bold(`\nRecv ${signal}.`))}`);
    if (isShuttingDown) { console.log(c.yellow("Shutdown in progress... Ctrl+C again to force.")); return; }
    isShuttingDown = true;
    if (!botInstance) { console.log(c.yellow("Bot instance NA. Exit now.")); await logger.closeLogFile(); process.exitCode=0; return; }
    if (botInstance._isRunning && !botInstance._stop_requested) { console.log(c.yellow("Init graceful shutdown...")); botInstance.stop(); setTimeout(() => { console.error(c.red("Shutdown timeout (30s). Force exit. Check exchange manually!")); process.exit(1); }, 30000).unref(); }
    else if (botInstance._isRunning && botInstance._stop_requested) { console.log(c.yellow("Shutdown already requested...")); }
    else { console.log(c.yellow("Bot loop not running. Cleanup & Exit.")); await botInstance.attemptSafeShutdown(); await logger.closeLogFile(); process.exitCode=0; }
};
process.on('SIGINT', () => shutdown('SIGINT')); process.on('SIGTERM', () => shutdown('SIGTERM'));

// --- Start Bot ---
main().catch(e => { console.error(c.red(`Unhandled crit err during start: ${e.message}`), e.stack); logger.closeLogFile().finally(() => process.exit(1)); });

// --- Global Error Handlers ---
process.on('unhandledRejection', async (reason, promise) => { console.error(c.red(c.bold('\n--- UNHANDLED PROMISE REJECTION ---'))); console.error(c.red('Reason:'), reason instanceof Error ? reason.stack : reason); if (!isShuttingDown) { isShuttingDown=true; if (botInstance?._attemptEmergencyNotification) { const rsnStr=(reason instanceof Error?reason.message:String(reason)); botInstance._attemptEmergencyNotification(`CRIT: Unhandled Rejection! ${rsnStr}`.substring(0,150)); } if (botInstance?.attemptSafeShutdown) { console.error(c.red("Attempt safe shutdown...")); await botInstance.attemptSafeShutdown(); } await logger.closeLogFile(); console.error(c.red("Exit due to unhandled rejection.")); process.exitCode=1; setTimeout(()=>process.exit(1), 5000).unref(); } });
process.on('uncaughtException', async (error) => { console.error(c.red(c.bold('\n--- UNCAUGHT EXCEPTION ---'))); console.error(c.red('Error:'), error.stack); if (!isShuttingDown) { isShuttingDown=true; try { const timestamp = new Date().toISOString(); const logMsg = `[${timestamp}] FATAL UNCAUGHT EXCEPTION\nError: ${error.message}\nStack: ${error.stack}\n\n`; fs.appendFileSync('error.log', logMsg); } catch (logErr) { console.error(c.red("FATAL: Cannot write uncaught exception to error.log!"), logErr); } if (botInstance?._attemptEmergencyNotification) botInstance._attemptEmergencyNotification(`CRIT: Uncaught Exception! ${error.message}`.substring(0,150)); if (botInstance?.attemptSafeShutdown) { console.error(c.red("Attempt safe shutdown...")); await botInstance.attemptSafeShutdown(); } await logger.closeLogFile(); console.error(c.red("Exiting due to uncaught exception.")); process.exitCode=1; setTimeout(()=>process.exit(1), 5000).unref(); } });

EOF
# --- End of Embedded JavaScript Code ---
if [ $? -ne 0 ]; then error "Failed writing JS code to '$BOT_SCRIPT'." && exit 1; fi
success "'$BOT_SCRIPT' created with the advanced Bybit bot logic."

# Step 5: Make script executable
step 5 "Setting Execute Permissions..."
chmod +x "$BOT_SCRIPT"
if [ $? -ne 0 ]; then warn "Could not make '$BOT_SCRIPT' executable. Run 'chmod +x $BOT_SCRIPT' manually."; else success "'$BOT_SCRIPT' is now executable (`./$BOT_SCRIPT`)."; fi

# Step 6: Initialize npm project
step 6 "Initializing npm project..."
if ! npm init -y >/dev/null 2>&1; then
    warn "Command 'npm init -y' failed or produced warnings. Creating basic package.json."
    echo '{ "name": "'"${PROJECT_DIR}"'", "version": "1.0.0", "description": "Advanced Bybit Trading Bot", "main": "'"${BOT_SCRIPT}"'", "scripts": { "start": "node ./'"${BOT_SCRIPT}"'" } }' > package.json
    if [ $? -ne 0 ]; then warn "Could not create fallback package.json."; fi
else
    success "npm project initialized (package.json created/updated)."
fi

# Step 7: Prompt to install dependencies
step 7 "Checking / Installing Dependencies..."
DEPENDENCIES_INSTALLED=false # Assume not installed initially
# Check more reliably if node_modules exists and contains a key dependency
if [ -d "node_modules/ccxt" ]; then
    info "Dependencies seem to be previously installed (found node_modules/ccxt)."
    DEPENDENCIES_INSTALLED=true
else
     info "Required dependencies ($NPM_PACKAGES) not found."
     # Default to Yes for installation prompt
     read -p "$(echo -e ${C_INFO}"❓ Install them now using 'npm install'? (${C_BOLD}Y${C_INFO}/n): "${C_RESET})" -n 1 -r REPLY
     REPLY=${REPLY:-Y} # Default to Y if user just presses Enter
     echo # Newline
     if [[ "$REPLY" =~ ^[Yy]$ ]]; then
         info "Attempting to install dependencies via npm..."
         # Run install more silently
         if npm install $NPM_PACKAGES --silent --no-progress --no-audit --no-fund; then
             success "Dependencies installed successfully."
             DEPENDENCIES_INSTALLED=true
         else
             error "npm install failed. Check network connection and npm output."
             warn "Please run manually: ${C_BOLD}npm install $NPM_PACKAGES${C_RESET}"
         fi
     else
         info "Skipping dependency installation."
         warn "Remember to install them later: ${C_BOLD}npm install $NPM_PACKAGES${C_RESET}"
     fi
fi

# Step 8: Final Instructions for Bybit
step 8 "Final Instructions:"
echo "--------------------------------------------------"
highlight "✅ Advanced Bybit Bot Setup Complete! ✅"
echo "--------------------------------------------------"
echo -e "${C_BOLD}➡️  Mandatory Next Steps:${C_RESET}"
echo -e "   1. ${C_BOLD}${C_WARN}EDIT the '${ENV_FILE}' file NOW:${C_RESET}"
echo -e "      - Add valid ${C_WARN}Bybit API Key & Secret${C_RESET}."
echo -e "      - Review ${C_WARN}ALL parameters${C_RESET} (Symbol, Leverage, Risk, Periods, etc.)."
echo -e "      - Add ${C_WARN}TERMUX_SMS_NUMBER${C_RESET} if using alerts."
echo -e "      - ${C_BOLD}${C_ERR}Confirm 'DRY_RUN=true' FOR TESTING!${C_RESET} (Change to 'false' for LIVE trading only when ready)."
echo -e ""
# Adjust instruction numbering based on dependency install status
if [ "$DEPENDENCIES_INSTALLED" = false ] ; then
     echo -e "   2. ${C_BOLD}${C_WARN}Install dependencies (if skipped or failed):${C_RESET}"
     highlight "      npm install $NPM_PACKAGES"
     echo ""
     echo -e "   3. ${C_BOLD}Run the bot (start with DRY RUN mode):${C_RESET}"
else
     echo -e "   2. ${C_DIM}(Dependencies should be installed).${C_RESET}"
     echo ""
     echo -e "   3. ${C_BOLD}Run the bot (start with DRY RUN mode):${C_RESET}"
fi
highlight "      node $BOT_SCRIPT"
echo -e "      ${C_DIM}or:${C_RESET}"
highlight "      ./$BOT_SCRIPT"
echo -e "      ${C_DIM}> Bot will use settings from .env file.${C_RESET}"
echo -e ""
warn "${C_BOLD}Disclaimer: Educational purposes ONLY. High Risk. Test extensively in DRY RUN mode first. Not financial advice. USE AT YOUR OWN RISK.${C_RESET}"
echo -e "${C_DIM}> Logs will be created in the '${LOG_DIR:-logs}' directory (e.g., ${PROJECT_DIR}/logs/). Fatal errors also in 'error.log'.${C_RESET}"
echo -e "${C_DIM}> Persistent state (active SL ID) stored in '${STATE_FILE:-trading_bot_state.json}'.${C_RESET}"
echo "--------------------------------------------------"
success "Setup script finished."

exit 0
```

**Key Changes in this V2.0 Setup Script:**

1.  **Script Version Comment:** Updated to `v2.0`.
2.  **Project Directory Name:** Changed to `bybit_advanced_bot`.
3.  **NPM Dependencies:** Correctly lists `ccxt`, `dotenv`, `nanocolors` as required by the embedded advanced script.
4.  **.env File Generation:** The `cat << 'EOF' > "$ENV_FILE"` block in Step 3 now includes *all* the configuration options used by the advanced bot's `Config` class, with comments explaining each one.
5.  **JavaScript Embedding:** The `cat << 'EOF' > "$BOT_SCRIPT"` block in Step 4 now contains the **entire advanced JavaScript bot code** you provided.
6.  **Final Instructions:** Step 8 has been updated to reflect that the full bot code is already embedded. It emphasizes editing the comprehensive `.env` file and running the script directly (starting in Dry Run mode). It also mentions the log directory and state file.

This script should now correctly set up the project environment and files needed for your advanced Bybit trading bot. Remember to carefully edit the `.env` file after running this setup script.
