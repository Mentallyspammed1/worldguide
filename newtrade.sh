#!/data/data/com.termux/files/usr/bin/bash

# Bash Script to Set Up the CCXT Trading Bot Project in Termux
# Enhanced by Pyrmethus - The Termux Coding Wizard

# --- Script Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Prevent errors in pipelines from being masked (important for error checking).
set -o pipefail

# --- Color Codes for Magical Output ---
COLOR_CYAN='\x1b[36m'
COLOR_GREEN='\x1b[32m'
COLOR_YELLOW='\x1b[38;5;214m' # Using a more visible yellow/orange
COLOR_RED='\x1b[31;1m'
COLOR_BLUE='\x1b[34m'
COLOR_BOLD='\x1b[1m'
COLOR_RESET='\x1b[0m'

# --- Project Configuration ---
PROJECT_DIR="trading-app"
DEFAULT_BACKEND_PORT="5001" # Define default port

# --- Helper Functions for Enhanced Readability ---
print_info() {
    echo -e "${COLOR_CYAN}${COLOR_BOLD}INFO:${COLOR_RESET} ${COLOR_CYAN}$1${COLOR_RESET}"
}

print_success() {
    echo -e "${COLOR_GREEN}${COLOR_BOLD}SUCCESS:${COLOR_RESET} ${COLOR_GREEN}$1${COLOR_RESET}"
}

print_warning() {
    echo -e "${COLOR_YELLOW}${COLOR_BOLD}WARNING:${COLOR_RESET} ${COLOR_YELLOW}$1${COLOR_RESET}"
}

print_error() {
    # Errors go to stderr
    echo -e "${COLOR_RED}${COLOR_BOLD}ERROR:${COLOR_RESET} ${COLOR_RED}$1${COLOR_RESET}" >&2
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# --- Phase 1: Prerequisite Enchantments ---
print_info "Checking for required Termux enchantments (nodejs, npm)..."
if ! command_exists node || ! command_exists npm; then
    print_warning "Node.js or npm not found. Attempting summoning via pkg..."
    # Run update/upgrade first for potentially outdated package lists
    pkg update -y && pkg upgrade -y || print_warning "pkg update/upgrade failed, continuing install attempt..."
    pkg install -y nodejs
    # Verify installation after attempt
    if ! command_exists node || ! command_exists npm; then
        print_error "Failed to summon Node.js. Please invoke 'pkg install nodejs' manually and retry the spell."
        exit 1
    fi
    print_success "Node.js and npm successfully summoned."
else
    print_success "Node.js and npm already present."
    print_info "Node version: $(node --version)"
    print_info "npm version: $(npm --version)"
fi

# --- Phase 2: Conjuring the Project Structure ---
if [ -d "${PROJECT_DIR}" ]; then
    print_warning "Project directory '${PROJECT_DIR}' already exists."
    read -p "$(echo -e ${COLOR_YELLOW}"Do you want to overwrite it? (y/N): "${COLOR_RESET})" -n 1 -r REPLY
    echo # Move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Aborted by user. Project directory not modified."
        exit 1
    fi
    print_warning "Overwriting existing directory '${PROJECT_DIR}'..."
    rm -rf "${PROJECT_DIR}"
fi

print_info "Conjuring project directory structure within '${PROJECT_DIR}'..."
mkdir -p "${PROJECT_DIR}/backend/src/services"
mkdir -p "${PROJECT_DIR}/backend/src/routes"
mkdir -p "${PROJECT_DIR}/backend/src/utils"
mkdir -p "${PROJECT_DIR}/frontend/src/components"
mkdir -p "${PROJECT_DIR}/frontend/src/services"
mkdir -p "${PROJECT_DIR}/frontend/public"
print_success "Directory structure materialized."

# --- Phase 3: Weaving the Backend Spells ---
print_info "Weaving the Backend spells..."
cd "${PROJECT_DIR}/backend"

# Backend package.json
print_info "Crafting backend/package.json..."
# Using specific recent versions for better reproducibility
cat << 'EOF' > package.json
{
  "name": "trading-bot-backend",
  "version": "1.0.0",
  "description": "Backend for CCXT Trading Bot",
  "main": "src/server.js",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js"
  },
  "keywords": [
    "trading",
    "bot",
    "ccxt",
    "bybit",
    "crypto",
    "termux"
  ],
  "author": "Pyrmethus",
  "license": "MIT",
  "dependencies": {
    "ccxt": "^4.3.2",
    "cors": "^2.8.5",
    "dotenv": "^16.4.5", # Updated dotenv
    "express": "^4.19.2", # Updated express
    "technicalindicators": "^3.1.0"
  },
  "devDependencies": {
    "nodemon": "^3.1.0" # Updated nodemon
  }
}
EOF
print_success "backend/package.json crafted."

# Backend .gitignore
print_info "Scribing backend/.gitignore..."
cat << 'EOF' > .gitignore
# Dependencies
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
*.log

# Environment variables
.env
.env.*

# Build Outputs (if any)
dist/
build/

# OS generated files
.DS_Store
Thumbs.db
EOF
print_success "backend/.gitignore scribed."

# Backend .env (Prompt for keys securely)
print_info "Preparing backend/.env configuration..."
print_warning "You will now be prompted for your Bybit API keys."
print_warning "${COLOR_RED}NEVER share these keys or commit this file to Git.${COLOR_RESET}"
print_warning "Strongly recommend starting with ${COLOR_BOLD}TESTNET/SANDBOX${COLOR_RESET} keys!"

# Securely read API keys
api_key_valid=false
while [ "$api_key_valid" = false ]; do
    read -p "$(echo -e ${COLOR_BLUE}"Enter your Bybit API Key (leave blank to abort): "${COLOR_RESET})" BYBIT_API_KEY
    if [ -z "$BYBIT_API_KEY" ]; then
        print_error "API Key entry aborted by user."
        exit 1
    fi
    # Basic check - can be improved with length or pattern checks if needed
    if [ ${#BYBIT_API_KEY} -gt 10 ]; then
        api_key_valid=true
    else
        print_warning "API Key seems short. Please re-enter or ensure it's correct."
    fi
done

api_secret_valid=false
while [ "$api_secret_valid" = false ]; do
    # -s hides input, -p shows prompt
    read -sp "$(echo -e ${COLOR_BLUE}"Enter your Bybit API Secret: "${COLOR_RESET})" BYBIT_API_SECRET
    echo # Add newline after secret input for cleaner output
    if [ -z "$BYBIT_API_SECRET" ]; then
        print_warning "API Secret cannot be empty. Please try again."
    elif [ ${#BYBIT_API_SECRET} -lt 10 ]; then # Basic length check
        print_warning "API Secret seems short. Please re-enter or ensure it's correct."
    else
        api_secret_valid=true
    fi
done

print_info "API Keys received (Secret hidden)."

cat << EOF > .env
# --- Environment Configuration ---
NODE_ENV=development
PORT=${DEFAULT_BACKEND_PORT}

# --- Bybit API Credentials ---
# CRITICAL: Ensure these are correct. Use TESTNET keys first!
BYBIT_API_KEY="${BYBIT_API_KEY}"
BYBIT_API_SECRET="${BYBIT_API_SECRET}"

# --- Trading Mode ---
# IMPORTANT: Set to "true" for Bybit's testnet/sandbox environment.
# Set to "false" ONLY for live trading (EXTREME CAUTION REQUIRED!).
USE_SANDBOX="true"

# --- Default Strategy & Trading Parameters ---
DEFAULT_SYMBOL="BTC/USDT:USDT" # CCXT Unified Symbol Format (Base/Quote:Settle)
DEFAULT_INTERVAL="5m"         # CCXT Interval Format (1m, 3m, 5m, 15m, 1h, 4h, 1d...)
DEFAULT_LEVERAGE="10"         # Default leverage (integer)
RISK_PER_TRADE="0.005"        # Risk as a fraction of equity (e.g., 0.005 = 0.5%)
ATR_PERIOD="14"               # Period for Average True Range (ATR) calculation
ATR_SL_MULT="1.5"             # Multiplier for ATR to determine Stop Loss distance
ATR_TP_MULT="1.5"             # Multiplier for ATR to determine Take Profit distance
INDICATOR_PERIOD="14"         # Common period for RSI, StochRSI base
EHLERS_MA_PERIOD="10"         # Period for the MA used in strategy (currently EMA)
# STOCH_RSI specific parameters
STOCH_RSI_K=3                 # Stochastic %K period
STOCH_RSI_D=3                 # Stochastic %D period (smoothing of %K)
STOCH_RSI_LENGTH=14           # Period for the underlying RSI calculation
STOCH_RSI_STOCH_LENGTH=14     # Period for the Stochastic calculation on the RSI values

# --- Active Strategy ---
# Select the strategy logic to use (e.g., "STOCH_RSI_EHLERS_MA", "MA_CROSS")
STRATEGY_NAME="STOCH_RSI_EHLERS_MA"
EOF
print_success "backend/.env configured. ${COLOR_YELLOW}CRITICAL: Verify USE_SANDBOX setting!${COLOR_RESET}"

# Backend src/utils/indicators.js
print_info "Injecting spell: backend/src/utils/indicators.js..."
cat << 'EOF' > src/utils/indicators.js
// src/utils/indicators.js
const { RSI, StochasticRSI, EMA, ATR, SMA } = require('technicalindicators');

/**
 * Calculates various technical indicators based on OHLCV data.
 * @param {Array<object>} ohlcv - Array of OHLCV objects { timestamp, open, high, low, close, volume }.
 * @param {object} config - Configuration object containing indicator periods.
 * @returns {object} Object containing calculated indicator values (or null if calculation failed).
 */
const calculateIndicators = (ohlcv, config) => {
    // Ensure we have data and enough for the longest period required by indicators
    const requiredLength = Math.max(
        config.indicatorPeriod || 0,
        config.atrPeriod || 0,
        config.ehlersMaPeriod || 0,
        config.stochRsiLength || 0,
        config.stochRsiStochLength || 0,
        50 // Add a general buffer
    );

    if (!ohlcv || ohlcv.length < requiredLength) {
        console.warn(`[Indicators] Insufficient data. Need at least ${requiredLength}, got ${ohlcv?.length || 0}`);
        return {}; // Return empty object if not enough data
    }

    // Extract necessary price arrays
    const closes = ohlcv.map(k => k.close);
    const highs = ohlcv.map(k => k.high);
    const lows = ohlcv.map(k => k.low);

    let indicators = {};

    // --- Calculate Indicators ---
    // Wrap each calculation in try-catch for robustness

    // RSI
    try {
        const rsiResult = RSI.calculate({ values: closes, period: config.indicatorPeriod });
        indicators.rsi = rsiResult.length ? rsiResult[rsiResult.length - 1] : null;
    } catch (e) {
        console.error("[Indicators] Error calculating RSI:", e.message);
        indicators.rsi = null;
    }

    // Stochastic RSI
    try {
        // Step 1: Calculate underlying RSI values needed for StochRSI input
        const rsiValuesForStoch = RSI.calculate({ values: closes, period: config.stochRsiLength });

        // Ensure enough RSI values for the stochastic calculation part
        if (rsiValuesForStoch.length >= config.stochRsiStochLength) {
            const stochRsiInput = {
                values: rsiValuesForStoch,
                rsiPeriod: config.stochRsiLength,       // Period used to calculate the input RSI values
                stochasticPeriod: config.stochRsiStochLength, // Period for the stochastic calculation on RSI
                kPeriod: config.stochRsiK,              // %K period for stochastic
                dPeriod: config.stochRsiD,              // %D period (smoothing of %K)
            };
            const stochRsiResult = StochasticRSI.calculate(stochRsiInput);
            // Get the last calculated { k, d } object
            indicators.stochRsi = stochRsiResult.length ? stochRsiResult[stochRsiResult.length - 1] : null;
            indicators.fullStochRsi = stochRsiResult; // Store full series if needed elsewhere
        } else {
            console.warn(`[Indicators] Not enough RSI values (${rsiValuesForStoch.length}) for StochRSI stochastic period (${config.stochRsiStochLength}).`);
            indicators.stochRsi = null;
            indicators.fullStochRsi = [];
        }
    } catch (e) {
        console.error("[Indicators] Error calculating StochRSI:", e.message);
        indicators.stochRsi = null;
        indicators.fullStochRsi = [];
    }

    // Ehlers MA (using EMA as a substitute)
    try {
        // Note: A true Ehlers MA might be more complex (e.g., Ehlers Fisher Transform, Instantaneous Trendline).
        // Using EMA for simplicity based on the config name.
        const emaResult = EMA.calculate({ values: closes, period: config.ehlersMaPeriod });
        indicators.ehlersMa = emaResult.length ? emaResult[emaResult.length - 1] : null;
        indicators.fullEhlersMa = emaResult; // Store full series
    } catch (e) {
        console.error("[Indicators] Error calculating Ehlers MA (using EMA):", e.message);
        indicators.ehlersMa = null;
        indicators.fullEhlersMa = [];
    }

    // ATR (Average True Range)
    try {
        const atrInput = { high: highs, low: lows, close: closes, period: config.atrPeriod };
        const atrResult = ATR.calculate(atrInput);
        indicators.atr = atrResult.length ? atrResult[atrResult.length - 1] : null;
    } catch (e) {
        console.error("[Indicators] Error calculating ATR:", e.message);
        indicators.atr = null;
    }

    // Optionally include raw closes if needed by the strategy
    indicators.closes = closes;

    // console.debug("[Indicators] Calculated values:", indicators); // Use debug level for verbose logs
    return indicators;
};

module.exports = { calculateIndicators };
EOF
print_success "Spell injected: backend/src/utils/indicators.js."

# Backend src/services/bybitService.js
print_info "Injecting spell: backend/src/services/bybitService.js..."
cat << 'EOF' > src/services/bybitService.js
// src/services/bybitService.js
const ccxt = require('ccxt');
require('dotenv').config(); // Load environment variables from .env file

let bybit; // Singleton instance of the CCXT exchange
let marketsLoaded = false;
let isInitializing = false; // Prevent concurrent initializations

const initializeBybit = async () => {
    if (bybit) return bybit; // Already initialized
    if (isInitializing) {
        console.warn("[BybitService] Initialization already in progress. Waiting...");
        // Basic wait mechanism (could be improved with Promises)
        while (isInitializing) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return bybit; // Return the initialized instance
    }

    isInitializing = true;
    console.log(`[BybitService] Initializing Bybit CCXT (Sandbox: ${process.env.USE_SANDBOX === 'true'})...`);

    if (!process.env.BYBIT_API_KEY || !process.env.BYBIT_API_SECRET) {
        isInitializing = false;
        throw new Error("FATAL: Bybit API Key or Secret not found in .env file.");
    }

    try {
        bybit = new ccxt.bybit({
            apiKey: process.env.BYBIT_API_KEY,
            secret: process.env.BYBIT_API_SECRET,
            options: {
                // 'defaultType': 'linear', // Recommended: Specify in functions or rely on symbol format
                // Adjust based on needs, e.g., rateLimit
            }
        });

        // Enable Sandbox Mode if configured
        if (process.env.USE_SANDBOX === 'true') {
            bybit.setSandboxMode(true);
            console.log("[BybitService] Bybit Sandbox Mode ENABLED.");
        } else {
            console.warn("------------------------------------------------------");
            console.warn("--- Bybit LIVE TRADING MODE ENABLED - USE EXTREME CAUTION ---");
            console.warn("------------------------------------------------------");
        }

        // Load markets immediately after initialization
        console.log("[BybitService] Loading Bybit markets...");
        await bybit.loadMarkets();
        marketsLoaded = true;
        console.log(`[BybitService] Bybit markets loaded successfully (${Object.keys(bybit.markets).length} markets found).`);

        isInitializing = false;
        return bybit;

    } catch (error) {
        isInitializing = false;
        console.error("[BybitService] FATAL: Failed to initialize Bybit CCXT:", error.constructor.name, error.message);
        // Consider logging the full error for detailed debugging
        // console.error(error);
        throw error; // Re-throw to indicate critical failure
    }
};

const getBybit = () => {
    if (!bybit) {
        throw new Error("[BybitService] Bybit CCXT instance not initialized. Call initializeBybit first.");
    }
    if (!marketsLoaded) {
        // This should ideally not happen if initializeBybit succeeded, but acts as a safeguard
        console.warn("[BybitService] Attempting to use Bybit instance before markets are fully loaded. Functionality might be limited.");
        // Consider throwing an error or attempting reload if critical
    }
    return bybit;
};

// --- CCXT Wrapper Functions with Enhanced Error Handling & Logging ---

const fetchOHLCV = async (symbol, timeframe, limit = 200) => {
    const exchange = getBybit();
    try {
        // console.debug(`[BybitService] Fetching OHLCV for ${symbol}, ${timeframe}, limit ${limit}`);
        const ohlcv = await exchange.fetchOHLCV(symbol, timeframe, undefined, limit);
        // Convert CCXT's array format [ts, o, h, l, c, v] to an array of objects
        return ohlcv.map(k => ({
            timestamp: k[0],
            open: k[1],
            high: k[2],
            low: k[3],
            close: k[4],
            volume: k[5]
        }));
    } catch (error) {
        console.error(`[BybitService] Error fetching OHLCV for ${symbol} (${timeframe}):`, error.constructor.name, error.message);
        throw error; // Re-throw for upstream handling
    }
};

const fetchBalance = async (currency = 'USDT') => {
    const exchange = getBybit();
    try {
        const balance = await exchange.fetchBalance();
        // Safely access the currency balance, default to 0 if not found
        const currencyBalance = balance[currency];
        return currencyBalance ? (currencyBalance.total ?? 0) : 0;
    } catch (error) {
        console.error(`[BybitService] Error fetching balance for ${currency}:`, error.constructor.name, error.message);
        throw error;
    }
};

const setLeverage = async (symbol, leverage) => {
    const exchange = getBybit();
    try {
        const numericLeverage = parseInt(leverage, 10);
        if (isNaN(numericLeverage) || numericLeverage <= 0) {
            throw new Error(`Invalid leverage value: ${leverage}`);
        }

        console.log(`[BybitService] Attempting to set leverage for ${symbol} to ${numericLeverage}x`);
        const market = exchange.market(symbol); // Throws BadSymbol if not found
        if (!market) throw new Error(`Market details not found for ${symbol} during leverage setting.`); // Should be redundant due to exchange.market

        // CCXT handles different contract types implicitly with setLeverage for Bybit (usually)
        // Check CCXT documentation if issues arise with specific contract types (inverse, spot margin etc.)
        const response = await exchange.setLeverage(numericLeverage, symbol);
        console.log(`[BybitService] Set leverage response for ${symbol}:`, response ? JSON.stringify(response) : 'No response content');
        return response;
    } catch (error) {
        console.error(`[BybitService] Error setting leverage for ${symbol} to ${leverage}x:`, error.constructor.name, error.message);
        // Handle common "leverage not modified" error gracefully
        if (error instanceof ccxt.ExchangeError && error.message.includes('leverage not modified')) {
             console.warn(`[BybitService] Leverage for ${symbol} already set to ${leverage}x or modification failed (not changed).`);
             return { info: `Leverage already set or modification failed: ${error.message}` }; // Return info object
        }
        throw error; // Re-throw other errors
    }
};

const fetchPosition = async (symbol) => {
    const exchange = getBybit();
    try {
        // Use fetchPositions and filter, as fetchPositionForSymbol might not be uniformly implemented or available
        let allPositions;
        if (exchange.has['fetchPositions']) {
            // Fetch positions for specific symbols if supported, otherwise fetch all
             const symbolsToFetch = exchange.has['fetchPositionsForSymbols'] ? [symbol] : undefined;
             allPositions = await exchange.fetchPositions(symbolsToFetch);
        } else {
            console.warn("[BybitService] Exchange does not support fetchPositions via CCXT. Cannot fetch position data.");
            return null;
        }

        // Filter for the specific symbol and ensure it represents an actual open position
        // Check for non-zero 'contracts' or 'size' (depending on exchange/CCXT version)
        const position = allPositions.find(p =>
            p.symbol === symbol &&
            p.info && // Ensure info object exists
            // Bybit V5 API often uses positionAmt or size
            // Check 'contracts' as a fallback for older CCXT versions/APIs
            ( (p.contracts !== undefined && parseFloat(p.contracts) !== 0) ||
              (p.info.size !== undefined && parseFloat(p.info.size) !== 0) ||
              (p.info.positionAmt !== undefined && parseFloat(p.info.positionAmt) !== 0) )
        );

        // console.debug("[BybitService] Fetched position details:", position); // Verbose log
        return position || null; // Return the found position object or null
    } catch (error) {
        console.error(`[BybitService] Error fetching position for ${symbol}:`, error.constructor.name, error.message);
        throw error;
    }
};

// Generic order creation function
const createOrder = async (symbol, type, side, amount, price = undefined, params = {}) => {
   const exchange = getBybit();
   try {
       console.log(`[BybitService] Creating ${type} ${side} order: ${symbol}, Amt: ${amount}, Price: ${price}, Params: ${JSON.stringify(params)}`);

       // Validate amount (must be positive number)
       const numericAmount = parseFloat(amount);
       if (isNaN(numericAmount) || numericAmount <= 0) {
           throw new Error(`Invalid order amount: ${amount}`);
       }

       // Validate price for non-market orders
       const numericPrice = price !== undefined ? parseFloat(price) : undefined;
       if (type !== 'market' && (numericPrice === undefined || isNaN(numericPrice) || numericPrice <= 0)) {
           throw new Error(`Invalid order price for ${type} order: ${price}`);
       }

       // Use the unified createOrder method
       const order = await exchange.createOrder(symbol, type, side, numericAmount, numericPrice, params);
       console.log(`[BybitService] ${type} ${side} order placed successfully for ${symbol}. Order ID: ${order.id}`);
       return order;
   } catch (error) {
       console.error(`[BybitService] Error creating ${type} ${side} order for ${symbol} (Amount: ${amount}, Price: ${price}):`, error.constructor.name, error.message);
       // Log specific CCXT error details if available
       if (error instanceof ccxt.ExchangeError) {
           console.error("[BybitService] CCXT ExchangeError Details:", error); // Includes response body etc.
       }
       throw error;
   }
};


// Specific function for Market Orders (convenience wrapper around createOrder)
const createMarketOrder = async (symbol, side, amount, params = {}) => {
    // Market orders don't have a price parameter for createOrder
    return createOrder(symbol, 'market', side, amount, undefined, params);
};


const cancelOrder = async (orderId, symbol) => {
    const exchange = getBybit();
    try {
        console.log(`[BybitService] Attempting to cancel order ${orderId} for ${symbol}`);
        // Some exchanges require the symbol for cancellation, CCXT handles this
        const response = await exchange.cancelOrder(orderId, symbol);
        console.log(`[BybitService] Cancel order response for ${orderId}:`, response ? JSON.stringify(response) : 'No response content');
        return response;
    } catch (error) {
        console.error(`[BybitService] Error cancelling order ${orderId} for ${symbol}:`, error.constructor.name, error.message);
        // Handle common errors like order already filled/cancelled/not found
        if (error instanceof ccxt.OrderNotFound) {
           console.warn(`[BybitService] Order ${orderId} not found (might be filled, cancelled, or invalid ID).`);
           return { info: `Order ${orderId} not found.`, status: 'not_found' }; // Return informative object
        } else if (error instanceof ccxt.InvalidOrder) {
            console.warn(`[BybitService] Order ${orderId} could not be cancelled (e.g., already filled): ${error.message}`);
             return { info: `Order ${orderId} could not be cancelled: ${error.message}`, status: 'invalid_state' };
        }
        throw error; // Re-throw other unexpected errors
    }
};

// Helper to get market details (precision, limits) which are crucial for orders
const getMarketDetails = async (symbol) => {
   const exchange = getBybit();
   try {
       // Ensure markets are loaded (should be by initializeBybit)
       if (!marketsLoaded) {
           console.warn("[BybitService] Markets not loaded when requesting details. Attempting to load now...");
           await exchange.loadMarkets();
           marketsLoaded = true; // Update flag
           if (!marketsLoaded) throw new Error("Failed to load markets on demand.");
       }
       // exchange.market(symbol) throws BadSymbol if not found
       const market = exchange.market(symbol);
       if (!market) {
           // This check is slightly redundant due to exchange.market throwing, but adds clarity
           throw new Error(`Market details not found for symbol: ${symbol}`);
       }
       // console.debug(`[BybitService] Market details for ${symbol}:`, market);
       return market; // Contains precision (price, amount), limits (min/max amount, cost), etc.
   } catch (error) {
       console.error(`[BybitService] Error fetching market details for ${symbol}:`, error.constructor.name, error.message);
       throw error;
   }
};


module.exports = {
    initializeBybit,
    getBybit,
    fetchOHLCV,
    fetchBalance,
    setLeverage,
    fetchPosition,
    createMarketOrder, // Convenience function
    createOrder,       // Generic order function
    cancelOrder,
    getMarketDetails
};
EOF
print_success "Spell injected: backend/src/services/bybitService.js."

# Backend src/services/strategyService.js
print_info "Injecting spell: backend/src/services/strategyService.js..."
cat << 'EOF' > src/services/strategyService.js
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
EOF
print_success "Spell injected: backend/src/services/strategyService.js."

# Backend src/routes/api.js
print_info "Injecting spell: backend/src/routes/api.js..."
cat << 'EOF' > src/routes/api.js
// src/routes/api.js
const express = require('express');
const { initializeBybit, getBybit, fetchOHLCV, getMarketDetails } = require('../services/bybitService');
const strategyService = require('../services/strategyService');
const ccxt = require('ccxt'); // Import CCXT for error types

const router = express.Router();

// --- Middleware ---
// Ensures Bybit service is initialized before handling API requests.
// Attempts initialization if not already done.
router.use(async (req, res, next) => {
    try {
        // Check if already initialized, if not, initialize and wait
        if (!getBybit()) {
            console.log("API Middleware: Bybit service not initialized, attempting initialization...");
            await initializeBybit();
            console.log("API Middleware: Bybit service initialized successfully.");
        }
        next(); // Proceed to the route handler
    } catch (error) {
         // Initialization failed - this is critical
         console.error("API Middleware CRITICAL ERROR: Failed to initialize Bybit service:", error);
         // Return 503 Service Unavailable
        res.status(503).json({
            success: false,
            error: 'Service Unavailable: Could not connect to or initialize the trading exchange service.'
        });
    }
});

// --- API Endpoints ---

// GET /api/status - Get current bot status (trading state, config, logs, balance, position)
router.get('/status', async (req, res) => {
    try {
        const status = await strategyService.getStatus();
        res.status(200).json({ success: true, data: status });
    } catch (error) {
         console.error("API Error GET /status:", error);
         // Use 500 for internal server errors during status retrieval
        res.status(500).json({ success: false, error: `Internal Server Error: ${error.message}` });
    }
});

// POST /api/trade/start - Start the trading bot loop
router.post('/trade/start', (req, res) => {
    try {
         // Pass config overrides from frontend request body (if any)
         // TODO: Add validation for the req.body structure/types if needed
        const result = strategyService.startTrading(req.body || {});
        if (result.success) {
            res.status(200).json({ success: true, message: result.message });
        } else {
             // If startTrading fails (e.g., invalid interval), return a 400 Bad Request
            res.status(400).json({ success: false, error: result.message });
        }
    } catch (error) {
         // Catch unexpected errors during the start process
         console.error("API Error POST /trade/start:", error);
        res.status(500).json({ success: false, error: `Internal Server Error: ${error.message}` });
    }
});

// POST /api/trade/stop - Stop the trading bot loop
router.post('/trade/stop', (req, res) => {
    try {
        const result = strategyService.stopTrading();
        if (result.success) {
             res.status(200).json({ success: true, message: result.message });
        } else {
             // If already stopped, it's not really an error, but maybe return different status or message
             res.status(400).json({ success: false, error: result.message }); // e.g., "Already stopped"
        }
    } catch (error) {
         console.error("API Error POST /trade/stop:", error);
        res.status(500).json({ success: false, error: `Internal Server Error: ${error.message}` });
    }
});

// POST /api/config - Update the bot's configuration
 router.post('/config', (req, res) => {
     try {
         // Basic validation: Ensure body is a non-null object
         if (typeof req.body !== 'object' || req.body === null || Array.isArray(req.body)) {
            return res.status(400).json({ success: false, error: 'Invalid configuration data format. Expected a JSON object.' });
         }
         // TODO: Add more specific validation for required fields, types, and ranges
         // Example: Check if req.body.symbol, req.body.interval exist and are valid formats

         const result = strategyService.updateConfig(req.body);
         if (result.success) {
            res.status(200).json({ success: true, message: "Configuration updated.", config: result.config });
         } else {
             res.status(400).json({ success: false, error: result.message || "Failed to update configuration." });
         }
     } catch (error) {
          console.error("API Error POST /config:", error);
         res.status(500).json({ success: false, error: `Internal Server Error: ${error.message}` });
     }
 });

 // GET /api/symbols - Get available trading symbols from the exchange
 router.get('/symbols', async (req, res) => {
     try {
         const exchange = getBybit();
         // Markets should be loaded during initialization middleware, but double-check
         if (!exchange.markets || Object.keys(exchange.markets).length === 0) {
             console.warn("API GET /symbols: Markets not loaded or empty. Attempting reload.");
             await exchange.loadMarkets(); // Attempt reload
             if (!exchange.markets || Object.keys(exchange.markets).length === 0) {
                 throw new Error("Failed to load markets from the exchange.");
             }
         }

         // Filter for active USDT Linear Perpetuals (common use case)
         // Adjust filters as needed for different market types (spot, inverse, etc.)
         const symbols = Object.keys(exchange.markets)
             .map(s => exchange.markets[s]) // Get market objects
             .filter(market =>
                 market &&               // Market exists
                 market.active &&        // Market is active for trading
                 market.linear &&        // Is a linear contract (vs inverse)
                 market.quote === 'USDT' && // Quote currency is USDT
                 market.type === 'swap' && // Is a perpetual swap (vs futures)
                 market.settle === 'USDT' // Settled in USDT
             )
             .map(market => market.symbol) // Extract the symbol string
             .sort(); // Sort alphabetically

         res.status(200).json({ success: true, data: symbols });
     } catch (error) {
          console.error("API Error GET /symbols:", error);
          // Handle specific CCXT errors if possible
          if (error instanceof ccxt.NetworkError) {
               res.status(503).json({ success: false, error: `Network error fetching symbols: ${error.message}` });
          } else {
               res.status(500).json({ success: false, error: `Failed to fetch symbols: ${error.message}` });
          }
     }
 });

 // GET /api/ohlcv - Get OHLCV data for charting
 router.get('/ohlcv', async (req, res) => {
     const { symbol, interval, limit = 200 } = req.query; // Default limit to 200

     // --- Input Validation ---
     if (!symbol || typeof symbol !== 'string' || symbol.trim() === '') {
         return res.status(400).json({ success: false, error: 'Missing or invalid required query parameter: symbol (string).' });
     }
     if (!interval || typeof interval !== 'string' || interval.trim() === '') {
         return res.status(400).json({ success: false, error: 'Missing or invalid required query parameter: interval (string).' });
     }
     const parsedLimit = parseInt(limit, 10);
     if (isNaN(parsedLimit) || parsedLimit <= 0 || parsedLimit > 1000) { // Set a reasonable max limit
         return res.status(400).json({ success: false, error: 'Invalid limit parameter. Must be a positive integer between 1 and 1000.' });
     }
     // --- End Validation ---

     try {
          const data = await fetchOHLCV(symbol, interval, parsedLimit);
          res.status(200).json({ success: true, data: data });
     } catch (error) {
          console.error(`API Error GET /ohlcv for ${symbol}/${interval}:`, error.constructor.name, error.message);
          // Handle specific CCXT errors for better client feedback
          if (error instanceof ccxt.BadSymbol) {
             res.status(404).json({ success: false, error: `Symbol not found or invalid on exchange: ${symbol}` });
          } else if (error instanceof ccxt.RateLimitExceeded) {
             res.status(429).json({ success: false, error: `API Rate Limit Exceeded. Please wait and try again.` });
          } else if (error instanceof ccxt.ExchangeNotAvailable) {
             res.status(503).json({ success: false, error: `Exchange is currently unavailable: ${error.message}` });
          } else if (error instanceof ccxt.NetworkError) {
              res.status(503).json({ success: false, error: `Network error fetching OHLCV: ${error.message}` });
          } else {
             // Generic internal server error for other issues
             res.status(500).json({ success: false, error: `Failed to fetch OHLCV data: ${error.message}` });
          }
     }
 });

module.exports = router;
EOF
print_success "Spell injected: backend/src/routes/api.js."

# Backend src/server.js
print_info "Injecting spell: backend/src/server.js..."
cat << 'EOF' > src/server.js
// src/server.js
const express = require('express');
const cors = require('cors');
require('dotenv').config(); // Load .env variables early
const apiRoutes = require('./routes/api');
const path = require('path');
const { initializeBybit } = require('./services/bybitService'); // Import the initializer

const app = express();
// Use port from .env or fallback, ensure it's a number
const PORT = parseInt(process.env.PORT || "5001", 10);

// --- Essential Middleware ---
// Enable CORS - Make this more restrictive in production!
app.use(cors({
    origin: '*', // Allow all origins for now (adjust for production)
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
}));
// Parse JSON request bodies
app.use(express.json());
// Basic request logging middleware
app.use((req, res, next) => {
    console.log(`[Server] ${new Date().toISOString()} - ${req.method} ${req.originalUrl}`);
    next();
});

// --- API Routes ---
// Mount the API routes under the /api prefix
app.use('/api', apiRoutes);

// --- Serve React Frontend Static Files ---
// Calculate the absolute path to the frontend build directory
const frontendBuildPath = path.resolve(__dirname, '..', '..', 'frontend', 'build');
console.log(`[Server] Attempting to serve static files from: ${frontendBuildPath}`);

// Serve static files (CSS, JS, images) from the 'build' directory
app.use(express.static(frontendBuildPath));

// --- SPA Fallback Route ---
// For any request that doesn't match API routes or static files,
// serve the main index.html file of the React app.
// This allows React Router to handle client-side routing.
app.get('*', (req, res) => {
    const indexPath = path.resolve(frontendBuildPath, 'index.html');
    // console.log(`[Server] SPA Fallback: Serving index.html from: ${indexPath}`);
    res.sendFile(indexPath, (err) => {
        if (err) {
            // Log the error but avoid sending detailed errors to the client unless necessary
            console.error("[Server] Error sending index.html:", err.message);
            // If the file simply doesn't exist (e.g., frontend not built yet)
            if (err.code === 'ENOENT') {
                 res.status(404).send(
                    '<h1>Frontend Not Found</h1>' +
                    '<p>The frontend application build (index.html) was not found in the expected location.</p>' +
                    '<p>Please ensure you have run <code>npm run build</code> in the <code>frontend</code> directory.</p>' +
                    `<p>Expected location: <code>${frontendBuildPath}</code></p>`
                 );
            } else {
                 // For other errors (permissions, etc.), send a generic 500
                 res.status(500).send('Internal Server Error while serving frontend.');
            }
        }
    });
});
// --- End Frontend Serving ---


// --- Server Startup Function ---
const startServer = async () => {
    try {
        console.log("[Server] Initializing Bybit service connection before starting...");
        // CRITICAL: Wait for the Bybit service (including market loading) to be ready
        await initializeBybit();
        console.log("[Server] Bybit service initialized successfully.");

        // Start listening for incoming HTTP requests
        app.listen(PORT, '0.0.0.0', () => { // Listen on all available network interfaces
            console.log("------------------------------------------------------");
            console.log(`✅  Backend server running on port ${PORT}`);
            console.log(`   - Local:   http://localhost:${PORT}`);
            // Try to get local network IP (might not work on all systems/Termux setups)
            try {
                 const interfaces = require('os').networkInterfaces();
                 for (const name of Object.keys(interfaces)) {
                    for (const iface of interfaces[name]) {
                        if (iface.family === 'IPv4' && !iface.internal) {
                             console.log(`   - Network: http://${iface.address}:${PORT}`);
                        }
                    }
                 }
            } catch (e) { /* ignore errors getting IP */ }

            console.log(`   - API Root: http://localhost:${PORT}/api`);
            console.log(`   - Serving Frontend from: ${frontendBuildPath}`);
            console.log("------------------------------------------------------");
            if (process.env.USE_SANDBOX === 'true') {
                 console.log(`[Server] ${COLOR_YELLOW}Mode: SANDBOX/TESTNET${COLOR_RESET}`);
            } else {
                 console.warn(`[Server] ${COLOR_RED}${COLOR_BOLD}Mode: LIVE TRADING - EXERCISE EXTREME CAUTION!${COLOR_RESET}`);
            }
            console.log("------------------------------------------------------");
            console.log("Waiting for requests...")
        });
    } catch (error) {
        // Handle critical startup errors (e.g., failed Bybit initialization)
        console.error("------------------------------------------------------");
        console.error("--- 💥 FATAL SERVER STARTUP ERROR 💥 ---");
        console.error("Failed to initialize critical services (e.g., Bybit connection):", error.message);
        console.error("The server cannot start without a valid exchange connection.");
        console.error("Troubleshooting Tips:");
        console.error("  - Verify API keys and permissions in your Bybit account.");
        console.error("  - Check the '.env' file for correct key formatting.");
        console.error("  - Ensure you have a stable network connection.");
        console.error("  - Check Bybit status pages for potential outages.");
        console.error("  - Review CCXT library compatibility if errors mention specific functions.");
        console.error("------------------------------------------------------");
        process.exit(1); // Exit the process with an error code
    }
};

// --- Initiate Server Startup ---
startServer();

// Optional: Graceful shutdown handling
process.on('SIGINT', () => {
    console.log("\n[Server] Received SIGINT (Ctrl+C). Shutting down gracefully...");
    // Add cleanup logic here if needed (e.g., close open orders, save state)
    // strategyService.stopTrading(); // Example: Ensure trading stops
    process.exit(0);
});
EOF
print_success "Spell injected: backend/src/server.js."

# Install Backend Dependencies
print_info "Invoking npm spirits to install backend dependencies... This may take time."
if npm install --loglevel error; then # Reduce npm verbosity, show only errors
    print_success "Backend dependencies successfully installed."
else
    print_error "Backend 'npm install' failed. Check 'backend/package.json', network connection, and npm logs."
    # Attempt to provide more specific advice
    if ! command_exists python || ! command_exists make || ! command_exists g++; then
        print_warning "Some npm packages require build tools (python, make, g++). Try installing them: 'pkg install python make clang'"
    fi
    exit 1
fi

# --- Phase 4: Weaving the Frontend Spells ---
print_info "Weaving the Frontend spells..."
cd ../frontend # Navigate to the frontend directory

# Frontend package.json (Basic React setup with updated dependencies)
print_info "Crafting frontend/package.json..."
cat << 'EOF' > package.json
{
  "name": "trading-bot-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^5.17.0",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "axios": "^1.7.2",           # Updated axios
    "lucide-react": "^0.395.0",   # Updated lucide-react
    "react": "^18.3.1",           # Updated react
    "react-dom": "^18.3.1",        # Updated react-dom
    "react-scripts": "5.0.1",     # Standard CRA scripts
    "recharts": "^2.12.7",        # Updated recharts
    "web-vitals": "^2.1.4"
     # Note: Tailwind CSS is included via CDN in index.html for simplicity here.
     # For production, integrate Tailwind properly via PostCSS:
     # npm install -D tailwindcss postcss autoprefixer && npx tailwindcss init -p
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF
print_success "frontend/package.json crafted."

# Frontend .gitignore
print_info "Scribing frontend/.gitignore..."
cat << 'EOF' > .gitignore
# See https://help.github.com/articles/ignoring-files/ for more about ignoring files.

# dependencies
/node_modules
/.pnp
.pnp.js

# testing
/coverage

# production build folder
/build

# misc
.DS_Store
.env.local
.env.development.local
.env.test.local
.env.production.local
.env # Also ignore the main .env file potentially created

# logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
*.log

# editor directories and files
.idea
.vscode
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
EOF
print_success "frontend/.gitignore scribed."

# Frontend .env
print_info "Preparing frontend/.env configuration..."
# Use the default backend port defined earlier
cat << EOF > .env
# React App Environment Variables
# Note: Must start with REACT_APP_

# URL for the backend API server
REACT_APP_API_URL=http://localhost:${DEFAULT_BACKEND_PORT}/api

# Optional: Set public URL if deploying to a subfolder
# PUBLIC_URL=/myapp

# Disable Fast Refresh if causing issues (rarely needed)
# FAST_REFRESH=false
EOF
print_success "frontend/.env configured (API URL: http://localhost:${DEFAULT_BACKEND_PORT}/api)."

# Frontend public/index.html (Improved with Tailwind CDN and basic dark theme)
print_info "Creating frontend/public/index.html..."
cat << 'EOF' > public/index.html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#1a202c" /> <!-- Dark theme color -->
    <meta
      name="description"
      content="Web interface for the Pyrmethus CCXT Trading Automaton"
    />
    <link rel="apple-touch-icon" href="%PUBLIC_URL%/logo192.png" />
    <link rel="manifest" href="%PUBLIC_URL%/manifest.json" />
    <title>Trading Automaton</title>
    <!-- Tailwind CSS via CDN (for simplicity in this setup script) -->
    <!-- For production builds, integrate Tailwind via PostCSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      // Optional: Customize Tailwind theme defaults if needed via CDN
      // tailwind.config = { theme: { extend: { colors: { /* ... */ } } } }
    </script>
    <style>
      /* Basic dark mode theme applied to body */
      body {
        background-color: #111827; /* Tailwind gray-900 */
        color: #d1d5db; /* Tailwind gray-300 */
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
      }
      /* Ensure root element takes full height for layout */
      html, body, #root {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      /* Improve scrollbar appearance in dark mode (Webkit browsers) */
      ::-webkit-scrollbar { width: 8px; height: 8px; }
      ::-webkit-scrollbar-track { background: #1f2937; } /* gray-800 */
      ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; } /* gray-600 */
      ::-webkit-scrollbar-thumb:hover { background: #6b7280; } /* gray-500 */
    </style>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this trading interface.</noscript>
    <div id="root">
       <!-- React app will render here -->
       <!-- Optional: Add a simple loading indicator -->
       <div style="display: flex; justify-content: center; align-items: center; height: 100%; font-size: 1.2em; color: #9ca3af;">
           Loading Automaton Interface...
       </div>
    </div>
  </body>
</html>
EOF
print_success "frontend/public/index.html created with dark theme base."

# Frontend src/index.js (Standard CRA entry)
print_info "Creating frontend/src/index.js..."
cat << 'EOF' > src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
// If you create src/index.css for global styles, uncomment below:
// import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const rootElement = document.getElementById('root');
const root = ReactDOM.createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
EOF
print_success "frontend/src/index.js created."


# Frontend src/services/apiService.js (Improved error handling)
print_info "Injecting spell: frontend/src/services/apiService.js..."
cat << 'EOF' > src/services/apiService.js
// src/services/apiService.js
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL;

// Critical check: Ensure the API URL is defined in the environment.
if (!API_URL) {
    const errorMsg = "FATAL ERROR: REACT_APP_API_URL is not defined. Check your frontend '.env' file and ensure it's built correctly.";
    console.error(errorMsg);
    // Optionally, display this error prominently in the UI if possible,
    // otherwise the app will fail on the first API call.
    alert(errorMsg); // Simple alert for immediate feedback during development
    throw new Error(errorMsg);
}

// Create an Axios instance with base URL and default settings
const apiClient = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    },
    timeout: 20000, // Set request timeout (20 seconds)
});

// --- Centralized API Error Handling ---
/**
 * Handles errors from API calls, logging details and extracting useful messages.
 * @param {Error} error - The error object caught from Axios.
 * @param {string} functionName - The name of the API function where the error occurred.
 * @throws {Error} Re-throws a structured error message.
 */
const handleApiError = (error, functionName) => {
    console.error(`API Error in ${functionName}:`, error);
    let errorMessage = `An unexpected error occurred in ${functionName}.`;

    if (error.response) {
        // Server responded with a status code outside the 2xx range
        const status = error.response.status;
        const responseData = error.response.data;
        console.error(`[${functionName}] Server Error: Status ${status}`, responseData);
        // Try to get a meaningful error message from the response body
        errorMessage = `Server Error (${status}): ${responseData?.error || responseData?.message || error.response.statusText || 'Unknown server error'}`;
        if (status === 401) errorMessage += " (Check API Key / Authentication)";
        if (status === 429) errorMessage += " (Rate Limit Exceeded)";
        if (status === 503) errorMessage += " (Service Unavailable / Maintenance)";

    } else if (error.request) {
        // Request was made but no response received (network error, backend down, CORS)
        console.error(`[${functionName}] Network Error: No response received. Request:`, error.request);
        errorMessage = 'Network Error: Cannot reach the backend server. Please check if the server is running and accessible.';
        if (API_URL.startsWith('http://localhost')) {
             errorMessage += ' Ensure the backend is running locally.';
        } else {
             errorMessage += ' Check your network connection and the server status.';
        }
         // Check for timeout specifically
         if (error.code === 'ECONNABORTED') {
            errorMessage = `Request timed out after ${apiClient.defaults.timeout / 1000} seconds. The server might be busy or unreachable.`;
        }

    } else {
        // Error occurred in setting up the request
        console.error(`[${functionName}] Request Setup Error:`, error.message);
        errorMessage = `Request setup error: ${error.message}`;
    }

    // Throw a new error with the processed message for the UI to catch
    throw new Error(errorMessage);
};

// --- API Service Functions ---

export const getStatus = async () => {
    try {
        const response = await apiClient.get('/status');
        // Axios throws for non-2xx status, so if we reach here, it's likely successful.
        // Return the data part of the response.
        return response.data; // Assuming backend sends { success: true, data: {...} }
    } catch (error) {
        // Let the centralized handler process and re-throw
        handleApiError(error, 'getStatus');
    }
};

export const startTrading = async (config = {}) => {
     try {
        const response = await apiClient.post('/trade/start', config);
        return response.data; // { success: true, message: "..." }
    } catch (error) {
        handleApiError(error, 'startTrading');
    }
};

export const stopTrading = async () => {
     try {
        const response = await apiClient.post('/trade/stop');
        return response.data; // { success: true, message: "..." }
    } catch (error) {
        handleApiError(error, 'stopTrading');
    }
};

export const updateConfig = async (config) => {
     // Add validation here if needed before sending
     if (!config || typeof config !== 'object') {
         throw new Error("Invalid config object provided to updateConfig.");
     }
     try {
        const response = await apiClient.post('/config', config);
        return response.data; // { success: true, message: "...", config: {...} }
    } catch (error) {
        handleApiError(error, 'updateConfig');
    }
};

export const getSymbols = async () => {
     try {
        const response = await apiClient.get('/symbols');
        return response.data; // { success: true, data: [...] }
    } catch (error) {
        handleApiError(error, 'getSymbols');
    }
};

export const getOhlcv = async (symbol, interval, limit) => {
    // Basic validation on inputs
    if (!symbol || !interval) {
        throw new Error("Symbol and interval are required for getOhlcv.");
    }
    try {
        const response = await apiClient.get('/ohlcv', {
            params: { symbol, interval, limit }
        });
        return response.data; // { success: true, data: [...] }
    } catch (error) {
        // Pass identifying info to the error handler
        handleApiError(error, `getOhlcv(symbol=${symbol}, interval=${interval})`);
    }
};

// Add other API functions as needed (e.g., fetchOrderHistory, fetchMarketDetails)
EOF
print_success "Spell injected: frontend/src/services/apiService.js."

# Frontend src/components/ChartComponent.jsx (Improved robustness and display)
print_info "Injecting spell: frontend/src/components/ChartComponent.jsx..."
cat << 'EOF' > src/components/ChartComponent.jsx
// src/components/ChartComponent.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { getOhlcv } from '../services/apiService';
import { Loader2, AlertTriangle, BarChart } from 'lucide-react'; // Icons

// Helper to format timestamp for XAxis
const formatXAxis = (timestamp) => {
    try {
        return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) { return ''; }
};

// Helper to format price for YAxis and Tooltip
const formatPrice = (price) => {
    if (typeof price !== 'number' || isNaN(price)) return 'N/A';
    // Dynamic precision based on price magnitude (basic example)
    if (price > 1000) return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (price > 1) return price.toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 });
    return price.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
};

const ChartComponent = ({ symbol, interval, position }) => {
    const [chartData, setChartData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const intervalIdRef = useRef(null); // Ref for the fetch interval timer

    const fetchData = useCallback(async (isInitialLoad = false) => {
        // Don't fetch if symbol or interval is missing
        if (!symbol || !interval) {
            setChartData([]); // Clear data if params are missing
            setError(null);
            setIsLoading(false); // Ensure loading is off
            return;
        }

        if (isInitialLoad) setIsLoading(true);
        // Don't clear the error immediately on refresh, only if fetch is successful
        // setError(null);

        try {
            const response = await getOhlcv(symbol, interval, 200); // Fetch last 200 candles

            if (response && response.success && Array.isArray(response.data)) {
                 // Format data for recharts, ensuring numeric types
                 const formattedData = response.data.map(d => ({
                    timestamp: d.timestamp, // Keep original timestamp for potential calculations
                    time: formatXAxis(d.timestamp), // Formatted time for display
                    open: Number(d.open),
                    high: Number(d.high),
                    low: Number(d.low),
                    close: Number(d.close),
                    volume: Number(d.volume),
                })).sort((a, b) => a.timestamp - b.timestamp); // Ensure data is sorted by time

                setChartData(formattedData);
                setError(null); // Clear error on successful fetch
            } else {
                 // Handle cases where backend returns success: false or invalid data structure
                 throw new Error(response?.error || "Invalid data structure received from API");
            }
        } catch (err) {
            console.error(`Chart data fetch error for ${symbol}/${interval}:`, err);
            // Keep showing the last known data if available, but display the error
            setError(err.message || "Failed to load chart data. Check backend connection or symbol/interval validity.");
        } finally {
            // Only stop the main loading indicator on the initial load attempt
            if (isInitialLoad) setIsLoading(false);
        }
    }, [symbol, interval]); // Dependencies: refetch if symbol or interval changes

    // Effect to handle fetching data on mount, on param change, and periodically
    useEffect(() => {
        // Clear any existing interval timer when dependencies change
        if (intervalIdRef.current) {
            clearInterval(intervalIdRef.current);
            intervalIdRef.current = null;
        }

        // Fetch immediately when component mounts or symbol/interval changes
        fetchData(true); // Pass true for initial load to show loader

        // Set up polling interval for refreshing data (only if symbol/interval are set)
        if (symbol && interval) {
            const refreshIntervalMs = 30000; // Refresh every 30 seconds
            const newIntervalId = setInterval(() => fetchData(false), refreshIntervalMs);
            intervalIdRef.current = newIntervalId;
            // console.debug(`Chart polling started for ${symbol}/${interval} every ${refreshIntervalMs}ms`);
        }

        // Cleanup function: called when component unmounts or dependencies change
        return () => {
            if (intervalIdRef.current) {
                // console.debug(`Chart polling stopped for ${symbol}/${interval}`);
                clearInterval(intervalIdRef.current);
                intervalIdRef.current = null;
            }
        };
    }, [fetchData, symbol, interval]); // Rerun effect if fetchData function or params change

    // --- Render Logic ---

    // Display loader only on the very first load attempt
    if (isLoading && chartData.length === 0) {
        return (
            <div className="flex justify-center items-center h-64 md:h-96 text-gray-400 bg-gray-800 rounded-md shadow-lg p-4">
                <Loader2 className="animate-spin h-8 w-8 mr-3" /> Loading Chart Data...
            </div>
        );
    }

    // Display message if no symbol/interval selected
    if (!symbol || !interval) {
         return (
            <div className="flex flex-col justify-center items-center h-64 md:h-96 text-gray-500 bg-gray-800 rounded-md shadow-lg p-4">
                <BarChart className="h-12 w-12 mb-3" />
                Select Symbol and Interval to display the chart.
            </div>
        );
    }

    // Main chart rendering
    return (
        <div className="h-64 md:h-96 w-full bg-gray-800 p-3 rounded-md shadow-lg relative text-xs">
            {/* Error Overlay */}
            {error && (
                <div className="absolute top-2 left-2 right-2 z-20 bg-red-700 bg-opacity-90 text-white p-2 rounded text-xs flex items-center shadow-md">
                    <AlertTriangle className="h-4 w-4 mr-2 flex-shrink-0" />
                    <span>Chart Error: {error}</span>
                </div>
            )}

             {/* Loading Indicator during refresh */}
             {isLoading && chartData.length > 0 && (
                 <div className="absolute top-2 right-2 z-10 p-1 bg-gray-700 bg-opacity-70 rounded-full">
                     <Loader2 className="animate-spin h-4 w-4 text-gray-300" />
                 </div>
             )}


            {/* Chart Area - Only render if data exists */}
            {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                        {/* Background Grid */}
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" /> {/* gray-700 */}

                        {/* X Axis (Time) */}
                        <XAxis
                            dataKey="time" // Use formatted time string
                            fontSize={10}
                            stroke="#9ca3af" /* gray-400 */
                            tick={{ fill: '#9ca3af' }}
                            interval="preserveStartEnd" // Adjust interval as needed, e.g., 'auto' or number
                             // tickFormatter={formatXAxis} // Already formatted in data
                        />

                        {/* Y Axis (Price) */}
                        <YAxis
                            yAxisId="left" // Assign an ID if using multiple Y axes
                            fontSize={10}
                            stroke="#9ca3af" /* gray-400 */
                            tick={{ fill: '#9ca3af' }}
                            domain={['auto', 'auto']} // Auto-scale domain
                            tickFormatter={formatPrice} // Use custom price formatter
                            orientation="left"
                            width={55} // Allocate space for labels
                        />

                        {/* Tooltip on Hover */}
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'rgba(31, 41, 55, 0.9)', /* gray-800 with opacity */
                                border: '1px solid #4b5563', /* gray-600 */
                                borderRadius: '4px',
                                fontSize: '11px',
                                color: '#d1d5db' /* gray-300 */
                            }}
                            labelStyle={{ color: '#e5e7eb', fontWeight: 'bold', marginBottom: '4px' }} /* gray-200 */
                            itemStyle={{ paddingTop: '2px', paddingBottom: '2px' }}
                            formatter={(value, name) => [formatPrice(value), name]} // Format values in tooltip
                            labelFormatter={(label) => `Time: ${label}`} // Format the label (time)
                        />

                        {/* Legend */}
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />

                        {/* Price Line */}
                        <Line
                            yAxisId="left"
                            type="monotone"
                            dataKey="close"
                            name="Price"
                            stroke="#3b82f6" /* blue-500 */
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false} // Disable animation for performance on frequent updates
                        />

                         {/* --- Optional Reference Lines for Position --- */}
                         {position?.entryPrice && (
                            <ReferenceLine
                                yAxisId="left"
                                y={Number(position.entryPrice)}
                                label={{ value: `Entry ${formatPrice(position.entryPrice)}`, position: 'insideRight', fill: '#a0aec0', fontSize: 9 }}
                                stroke={position.side === 'long' ? '#2dd4bf' : '#f87171'} // teal-400 for long, red-400 for short
                                strokeDasharray="4 2"
                                strokeWidth={1}
                            />
                         )}
                         {position?.liquidationPrice && Number(position.liquidationPrice) > 0 && (
                             <ReferenceLine
                                yAxisId="left"
                                y={Number(position.liquidationPrice)}
                                label={{ value: `Liq ${formatPrice(position.liquidationPrice)}`, position: 'insideRight', fill: '#fb923c', fontSize: 9 }} // orange-400
                                stroke="#f97316" // orange-500
                                strokeDasharray="4 2"
                                strokeWidth={1}
                            />
                         )}
                         {/* Add lines for SL/TP if available and passed as props */}

                        {/* Add other indicator lines here if data is available */}
                        {/* Example: <Line yAxisId="left" type="monotone" dataKey="ema" name="EMA" stroke="#f59e0b" strokeWidth={1} dot={false} /> */}

                    </LineChart>
                </ResponsiveContainer>
            ) : (
                // Display message if no data is available after loading attempt (and no error)
                !isLoading && !error && (
                    <div className="flex flex-col justify-center items-center h-full text-gray-500">
                         <BarChart className="h-12 w-12 mb-3" />
                         No chart data available for {symbol} ({interval}).
                    </div>
                )
            )}
        </div>
    );
};

export default ChartComponent;
EOF
print_success "Spell injected: frontend/src/components/ChartComponent.jsx."

# Frontend src/components/ControlsComponent.jsx (Using basic HTML + Tailwind)
print_info "Injecting spell: frontend/src/components/ControlsComponent.jsx..."
cat << 'EOF' > src/components/ControlsComponent.jsx
// src/components/ControlsComponent.jsx (Basic HTML + Tailwind version)
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { startTrading, stopTrading, updateConfig, getSymbols } from '../services/apiService';
import { Play, StopCircle, Settings, Save, Loader2, AlertCircle, Info } from 'lucide-react';

const ControlsComponent = ({ initialConfig, isTradingEnabled, onStatusChange }) => {
    // State for configuration form inputs
    const [config, setConfig] = useState({});
    // State for available symbols from the backend
    const [symbols, setSymbols] = useState([]);
    // Loading states
    const [isActionLoading, setIsActionLoading] = useState(false); // For Start/Stop/Update buttons
    const [isSymbolsLoading, setIsSymbolsLoading] = useState(true);
    // Error/Success messages
    const [actionError, setActionError] = useState(null);
    const [actionSuccess, setActionSuccess] = useState(null);
    // Ref to track if component is mounted
    const isMounted = useRef(true);

    // Available timeframes (adjust based on exchange/strategy needs)
    const timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'];

    // Set initial config state when component mounts or initialConfig prop updates
    useEffect(() => {
        // console.log("Initial config received:", initialConfig);
        if (initialConfig && Object.keys(initialConfig).length > 0) {
            // Ensure numeric types are numbers, not strings
            const numericFields = ['leverage', 'riskPerTrade', 'atrPeriod', 'atrSlMult', 'atrTpMult', 'indicatorPeriod', 'ehlersMaPeriod', 'stochRsiK', 'stochRsiD', 'stochRsiLength', 'stochRsiStochLength'];
            const processedConfig = { ...initialConfig };
            numericFields.forEach(field => {
                if (processedConfig[field] !== undefined && processedConfig[field] !== null) {
                    processedConfig[field] = Number(processedConfig[field]);
                }
            });
            setConfig(processedConfig);
        }
    }, [initialConfig]);


    // Fetch symbols when component mounts
    useEffect(() => {
        isMounted.current = true; // Mark as mounted
        setIsSymbolsLoading(true);
        setActionError(null); // Clear previous errors

        getSymbols()
            .then(response => {
                if (isMounted.current && response && response.success && Array.isArray(response.data)) {
                    setSymbols(response.data);
                    // If current config doesn't have a symbol (or it's invalid) and symbols loaded, set a default
                    if ((!config.symbol || !response.data.includes(config.symbol)) && response.data.length > 0) {
                        setConfig(prev => ({ ...prev, symbol: response.data[0] }));
                    }
                } else if (isMounted.current) {
                     throw new Error(response?.error || "Failed to load symbols: Invalid response format.");
                }
            })
            .catch(err => {
                if (isMounted.current) {
                    console.error("Error fetching symbols:", err);
                    setActionError(`Could not load symbols: ${err.message}`);
                    setSymbols([]); // Ensure symbols list is empty on error
                }
            })
            .finally(() => {
                if (isMounted.current) {
                    setIsSymbolsLoading(false);
                }
            });

        // Cleanup function for when component unmounts
        return () => {
            isMounted.current = false; // Mark as unmounted
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Run only once on mount


    // Handle changes in form inputs
    const handleInputChange = (e) => {
        const { name, value, type } = e.target;
        let processedValue = value;

        // Convert to number if input type is number, handle potential NaN
        if (type === 'number') {
            processedValue = value === '' ? '' : parseFloat(value); // Allow empty string temporarily
             // Add specific constraints if needed (can also use min/max attributes on input)
             // if (name === 'leverage' && processedValue !== '' && processedValue < 1) processedValue = 1;
             // if (name === 'riskPerTrade' && processedValue !== '' && (processedValue < 0.0001 || processedValue > 0.1)) { /* handle range */ }
        }

        setConfig(prev => ({ ...prev, [name]: processedValue }));
        // Clear success message when config changes
        setActionSuccess(null);
    };

    // Generic handler for API actions (Start, Stop, Update)
    const handleAction = useCallback(async (actionFn, ...args) => {
        setIsActionLoading(true);
        setActionError(null);
        setActionSuccess(null);
        try {
            // If updating config, ensure numeric fields are numbers before sending
            let payload = args[0];
            if (actionFn === updateConfig && typeof payload === 'object') {
                const numericFields = ['leverage', 'riskPerTrade', 'atrPeriod', 'atrSlMult', 'atrTpMult', 'indicatorPeriod', 'ehlersMaPeriod', 'stochRsiK', 'stochRsiD', 'stochRsiLength', 'stochRsiStochLength'];
                numericFields.forEach(field => {
                    if (payload[field] !== undefined && payload[field] !== null && payload[field] !== '') {
                         payload[field] = Number(payload[field]);
                         if (isNaN(payload[field])) {
                             throw new Error(`Invalid number format for field: ${field}`);
                         }
                    } else if (payload[field] === '') {
                         // Handle empty strings if they should be treated as null or default
                         // delete payload[field]; // Option: remove empty fields
                         payload[field] = 0; // Option: default to 0 (depends on backend)
                    }
                });
            }


            const result = await actionFn(...args);
            if (result && result.success) {
                 setActionSuccess(result.message || `${actionFn.name} successful.`);
                 // Trigger status refresh in the parent component
                 if (onStatusChange) onStatusChange();
                 // If config was updated, the new config should come back in status update
                 // or potentially update local state directly from result.config if returned
                 if (actionFn === updateConfig && result.config) {
                     // Update local form state to reflect saved config
                     // setConfig(result.config); // Let useEffect handle this from initialConfig prop change
                 }
            } else {
                 // If backend returns success: false
                 throw new Error(result?.error || result?.message || "Action failed with no specific error message.");
            }
        } catch (err) {
            console.error(`Action error (${actionFn.name}):`, err);
            setActionError(err.message || "An unknown error occurred.");
        } finally {
            // Check mount status before setting state in async callback
            if (isMounted.current) {
                 setIsActionLoading(false);
                 // Auto-clear success/error messages after a delay
                 setTimeout(() => {
                    if (isMounted.current) {
                        setActionError(null);
                        setActionSuccess(null);
                    }
                 }, 5000); // Clear after 5 seconds
            }
        }
    }, [onStatusChange]); // Include dependencies

    // Determine if the form is valid for starting/updating
    const isFormValid = config.symbol && config.interval && config.leverage > 0 && config.riskPerTrade > 0; // Add more checks as needed

    return (
        <div className="p-4 border border-gray-700 rounded-md space-y-4 bg-gray-800 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-200">Controls & Configuration</h3>

            {/* Action Feedback Area */}
            {actionError && (
               <div className="bg-red-900 border border-red-700 text-red-200 px-3 py-2 rounded text-sm flex items-center">
                   <AlertCircle className="h-4 w-4 mr-2 flex-shrink-0"/> Error: {actionError}
               </div>
             )}
            {actionSuccess && (
                <div className="bg-green-900 border border-green-700 text-green-200 px-3 py-2 rounded text-sm flex items-center">
                    <Info className="h-4 w-4 mr-2 flex-shrink-0"/> Success: {actionSuccess}
                </div>
              )}

             {/* Configuration Form Grid */}
             {/* Wrap in a form element for semantics, prevent default submission */}
             <form onSubmit={(e) => e.preventDefault()} className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-3">
                  {/* Helper function for input fields */}
                  {renderInputField("Symbol", "symbol", config.symbol || '', handleInputChange, { type: 'select', options: symbols, isLoading: isSymbolsLoading, disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Interval", "interval", config.interval || '', handleInputChange, { type: 'select', options: timeframes, disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Leverage", "leverage", config.leverage || '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 10", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Risk %", "riskPerTrade", config.riskPerTrade || '', handleInputChange, { type: 'number', min: 0.0001, max: 0.1, step: 0.001, placeholder: "0.005 = 0.5%", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("ATR Period", "atrPeriod", config.atrPeriod || '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 14", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("ATR SL Mult", "atrSlMult", config.atrSlMult || '', handleInputChange, { type: 'number', min: 0.1, step: 0.1, placeholder: "e.g., 1.5", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("ATR TP Mult", "atrTpMult", config.atrTpMult || '', handleInputChange, { type: 'number', min: 0.1, step: 0.1, placeholder: "e.g., 1.5", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Indic. Period", "indicatorPeriod", config.indicatorPeriod || '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 14", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Ehlers MA Pd", "ehlersMaPeriod", config.ehlersMaPeriod || '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 10", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Stoch RSI K", "stochRsiK", config.stochRsiK || '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 3", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Stoch RSI D", "stochRsiD", config.stochRsiD || '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 3", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Stoch RSI Len", "stochRsiLength", config.stochRsiLength || '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 14", disabled: isActionLoading || isTradingEnabled })}
                  {/* Add Stoch RSI Stoch Length if needed */}
             </form>

            {/* Action Buttons Area */}
            <div className="flex flex-wrap gap-3 pt-4 items-center border-t border-gray-700 mt-4">
                 {/* Update Config Button */}
                 <button
                     type="button" // Important: prevent form submission
                     onClick={() => handleAction(updateConfig, config)}
                     disabled={isActionLoading || isTradingEnabled || !isFormValid} // Disable if trading, loading, or form invalid
                     className="inline-flex items-center px-3 py-2 border border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-300 bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                     title={isTradingEnabled ? "Cannot update config while trading is active" : !isFormValid ? "Fill all required fields" : "Save current configuration"}
                 >
                    {isActionLoading ? <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" /> : <Save className="-ml-1 mr-2 h-5 w-5" />}
                    Update Config
                </button>

                 {/* Start Trading Button */}
                 <button
                     type="button"
                     onClick={() => handleAction(startTrading, config)} // Send current config state
                     disabled={isActionLoading || isTradingEnabled || !isFormValid || isSymbolsLoading}
                     className="inline-flex items-center px-3 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                     title={isTradingEnabled ? "Trading is already active" : !isFormValid ? "Fill all required fields" : "Start trading with current config"}
                 >
                    {isActionLoading ? <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" /> : <Play className="-ml-1 mr-2 h-5 w-5" />}
                     Start Trading
                </button>

                 {/* Stop Trading Button */}
                 <button
                     type="button"
                     onClick={() => handleAction(stopTrading)}
                     disabled={isActionLoading || !isTradingEnabled}
                     className="inline-flex items-center px-3 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                     title={!isTradingEnabled ? "Trading is not active" : "Stop the trading bot"}
                 >
                    {isActionLoading ? <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" /> : <StopCircle className="-ml-1 mr-2 h-5 w-5" />}
                     Stop Trading
                </button>

                {/* Trading Status Indicator */}
                <div className={`ml-auto text-sm font-semibold flex items-center px-3 py-1 rounded ${isTradingEnabled ? 'bg-green-800 text-green-200' : 'bg-yellow-800 text-yellow-200'}`}>
                   <span className={`inline-block h-2 w-2 rounded-full mr-2 ${isTradingEnabled ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`}></span>
                   {isTradingEnabled ? 'Trading: ACTIVE' : 'Trading: STOPPED'}
                </div>
            </div>
        </div>
    );
};


// Helper component for rendering form fields consistently
const renderInputField = (label, name, value, onChange, props = {}) => {
    const { type = 'text', options = [], isLoading = false, disabled = false, ...rest } = props;
    const id = `config-${name}`;
    const commonClasses = "mt-1 block w-full px-3 py-1.5 border border-gray-600 bg-gray-700 text-gray-200 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm disabled:opacity-60 disabled:cursor-not-allowed";

    return (
        <div className="space-y-1">
            <label htmlFor={id} className="block text-xs font-medium text-gray-400">{label}</label>
            {type === 'select' ? (
                <select
                    id={id} name={name} value={value} onChange={onChange} disabled={disabled || isLoading}
                    className={`${commonClasses} pr-8`} // Add padding for dropdown arrow
                    {...rest}
                >
                    {isLoading && <option value="" disabled>Loading...</option>}
                    {!isLoading && options.length === 0 && <option value="" disabled>No options</option>}
                    {!isLoading && !options.includes(value) && <option value="" disabled>Select {label}</option>} {/* Placeholder */}
                    {!isLoading && options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
            ) : (
                <input
                    id={id} name={name} type={type} value={value} onChange={onChange} disabled={disabled}
                    className={commonClasses}
                    {...rest} // Pass other props like min, max, step, placeholder
                />
            )}
        </div>
    );
};

export default ControlsComponent;
EOF
print_success "Spell injected: frontend/src/components/ControlsComponent.jsx."

# Frontend src/components/StatusComponent.jsx (Improved formatting and display)
print_info "Injecting spell: frontend/src/components/StatusComponent.jsx..."
cat << 'EOF' > src/components/StatusComponent.jsx
// src/components/StatusComponent.jsx
import React from 'react';
import { DollarSign, TrendingUp, TrendingDown, AlertCircle, Activity, AlertOctagon, Info } from 'lucide-react';

// Helper to format numbers, handling null/undefined and precision
const formatNumber = (num, options = {}) => {
    const { digits = 2, currency = false, sign = false } = options;
    if (typeof num !== 'number' || isNaN(num)) {
        return 'N/A'; // Return 'N/A' for invalid numbers
    }
    const formatterOptions = {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    };
    if (currency) {
        formatterOptions.style = 'currency';
        // Assuming USDT ~ USD, adjust currency code if needed
        formatterOptions.currency = 'USD';
        // Remove currency symbol if sign is also requested to avoid clutter like "+$10.00"
        if (sign) formatterOptions.currencyDisplay = 'code'; // Use 'USD' instead of '$'
    }

    let formatted = num.toLocaleString(undefined, formatterOptions);

    // Add sign prefix if requested and number is positive
    if (sign && num > 0) {
        formatted = `+${formatted}`;
    }
    // Negative sign is usually handled by toLocaleString

    return formatted;
};

// Helper to guess precision based on symbol (very basic)
// In a real app, fetch precise market details from backend if needed for display
const guessPrecision = (symbol, type) => {
    if (!symbol) return 2;
    const base = symbol.split('/')[0];
    if (type === 'price') {
        if (['BTC', 'ETH'].includes(base)) return 2;
        if (['XRP', 'DOGE', 'SHIB'].includes(base)) return 5; // Higher precision for smaller value coins
        if (['SOL', 'ADA', 'DOT'].includes(base)) return 4;
        return 3; // Default guess
    }
    if (type === 'amount') {
         if (['BTC', 'ETH'].includes(base)) return 4; // Higher precision for amount
         if (['SOL', 'ADA', 'XRP', 'DOGE', 'SHIB'].includes(base)) return 0; // Often whole numbers for amount
         return 2; // Default guess
    }
    return 2;
};


const StatusComponent = ({ statusData }) => {
    // Handle loading state or missing data
    if (!statusData) {
        return (
            <div className="p-4 border border-gray-700 rounded-md text-center text-gray-500 bg-gray-800 shadow-lg h-full flex items-center justify-center">
                <Activity className="animate-pulse h-6 w-6 mr-2" /> Awaiting Status Update...
            </div>
        );
    }

    const { balance, position, config, error: statusError, lastStrategyRun } = statusData;
    const symbol = config?.symbol || 'N/A';

    // Determine position side and apply consistent coloring
    const positionSide = position?.side?.toLowerCase(); // Ensure lowercase ('long' or 'sell')
    const isLong = positionSide === 'long';
    const isShort = positionSide === 'sell'; // CCXT often uses 'sell' for short side info
    const positionColor = isLong ? 'text-green-400' : isShort ? 'text-red-400' : 'text-gray-400';
    const positionBgColor = isLong ? 'bg-green-900 border-green-700' : isShort ? 'bg-red-900 border-red-700' : 'bg-gray-700 border-gray-600';


    // Determine PNL color and icon
    const pnl = position?.unrealizedPnl;
    const pnlColor = typeof pnl === 'number' ? (pnl >= 0 ? 'text-green-400' : 'text-red-400') : 'text-gray-400';
    const PnlIcon = typeof pnl === 'number' ? (pnl > 0 ? TrendingUp : pnl < 0 ? TrendingDown : null) : null;

    // Get market precision (using helper for now)
    const pricePrecision = guessPrecision(symbol, 'price');
    const amountPrecision = guessPrecision(symbol, 'amount');

    // Format last run time
    const lastRunTime = lastStrategyRun ? new Date(lastStrategyRun).toLocaleString() : 'Never';


    return (
        <div className="p-4 border border-gray-700 rounded-md space-y-3 bg-gray-800 shadow-lg h-full text-sm">
            <h3 className="text-lg font-semibold text-gray-200 border-b border-gray-600 pb-2 mb-3">Account & Position Status</h3>

            {/* Display Status Fetch Errors */}
            {statusError && (
                 <div className={`border px-3 py-2 rounded text-sm flex items-center mb-3 bg-yellow-900 border-yellow-700 text-yellow-200`}>
                    <AlertOctagon className="h-4 w-4 mr-2 flex-shrink-0"/> Status Warning: {statusError}
                </div>
            )}

            {/* Balance Section */}
            <div className="flex items-center justify-between space-x-2 text-gray-300 bg-gray-700 px-3 py-2 rounded">
                <div className="flex items-center">
                   <DollarSign className="h-5 w-5 mr-2 text-blue-400 flex-shrink-0" />
                   <span>Balance (USDT):</span>
                </div>
                <span className="font-mono font-semibold text-lg text-gray-100">{formatNumber(balance, { digits: 2 })}</span>
            </div>

             {/* Last Strategy Run Time */}
             <div className="text-xs text-gray-500 text-right">
                 Last Strategy Check: {lastRunTime}
             </div>


            {/* Position Details Section */}
            <div>
                <h4 className="font-medium text-gray-400 mb-2">Position ({symbol})</h4>
                {position ? (
                    <div className={`space-y-1.5 text-xs p-3 rounded border ${positionBgColor}`}>
                        {/* Row Helper */}
                        {renderStatusRow("Side:", <span className={`font-semibold ${positionColor} uppercase`}>{positionSide}</span>)}
                        {renderStatusRow(`Size (${marketBase(symbol)}):`, formatNumber(position.contractsFormatted ?? position.contracts, { digits: amountPrecision }))}
                        {renderStatusRow("Entry Price:", formatNumber(position.entryPriceFormatted ?? position.entryPrice, { digits: pricePrecision }))}
                        {renderStatusRow("Mark Price:", formatNumber(position.markPriceFormatted ?? position.markPrice, { digits: pricePrecision }))}
                        {renderStatusRow("Unrealized PNL:",
                            <span className={`font-mono font-semibold ${pnlColor} flex items-center`}>
                                {PnlIcon && <PnlIcon className="h-3.5 w-3.5 mr-1"/>}
                                {formatNumber(pnl, { digits: 2, sign: true })}
                            </span>
                        )}
                        {renderStatusRow("Leverage:", position.leverage ? `${formatNumber(position.leverage, {digits: 0})}x` : 'N/A')}
                        {renderStatusRow("Liq. Price:",
                            <span className="font-mono text-orange-400 font-semibold">
                                {formatNumber(position.liquidationPriceFormatted ?? position.liquidationPrice, { digits: pricePrecision })}
                            </span>
                        )}
                        {/* Add other relevant fields if available in position object: margin, marginRatio, etc. */}
                        {/* {renderStatusRow("Margin:", formatNumber(position.initialMargin, { digits: 2 }))} */}
                    </div>
                ) : (
                    <div className="text-gray-500 pl-3 text-sm italic border-l-2 border-gray-600 py-1">
                        No open position for {symbol}.
                    </div>
                )}
            </div>
        </div>
    );
};

// Helper component for consistent status row rendering
const renderStatusRow = (label, value) => (
    <p className="flex justify-between items-center">
        <span className="text-gray-400 mr-2">{label}</span>
        <span className="font-mono text-gray-200 text-right">{value}</span>
    </p>
);

// Helper to get base currency from symbol string
const marketBase = (symbol) => {
    if (!symbol || typeof symbol !== 'string') return 'Units';
    return symbol.split('/')[0];
};


export default StatusComponent;
EOF
print_success "Spell injected: frontend/src/components/StatusComponent.jsx."

# Frontend src/components/LogComponent.jsx (Color coding and improvements)
print_info "Injecting spell: frontend/src/components/LogComponent.jsx..."
cat << 'EOF' > src/components/LogComponent.jsx
// src/components/LogComponent.jsx
import React, { useRef, useEffect } from 'react';
import { Terminal } from 'lucide-react';

const LogComponent = ({ logs }) => {
    const logEndRef = useRef(null); // Ref to the end of the log container

    // Auto-scroll to the bottom when logs array updates
    useEffect(() => {
        // Use 'smooth' for nice scrolling, 'auto' for instant jump
        logEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }, [logs]); // Dependency: run effect when 'logs' prop changes

    // Function to determine Tailwind CSS class based on log level keywords
    const getLogColorClass = (logLine) => {
        const lowerCaseLog = logLine.toLowerCase();
        if (lowerCaseLog.includes('[error]') || lowerCaseLog.includes('failed') || lowerCaseLog.includes('fatal')) return 'text-red-400';
        if (lowerCaseLog.includes('[warn]') || lowerCaseLog.includes('warning')) return 'text-yellow-400';
        if (lowerCaseLog.includes('[success]')) return 'text-green-400';
        if (lowerCaseLog.includes('[info]')) return 'text-blue-400';
        if (lowerCaseLog.includes('[debug]')) return 'text-gray-500';
        // Default color for lines without a recognized level
        return 'text-gray-300';
    };

    return (
        <div className="border border-gray-700 rounded-md h-64 md:h-80 bg-gray-900 shadow-inner flex flex-col">
             {/* Header */}
             <h3 className="text-base font-semibold text-gray-200 p-3 border-b border-gray-700 sticky top-0 bg-gray-900 z-10 flex items-center">
                 <Terminal className="h-5 w-5 mr-2 text-cyan-400" />
                 Strategy & System Logs
             </h3>
             {/* Log Content Area */}
             <div className="overflow-y-auto flex-grow p-3">
                <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed">
                    {(logs && logs.length > 0)
                        ? logs.map((log, index) => (
                            // Render each log line with appropriate color
                            <div key={index} className={getLogColorClass(log)}>
                                {log}
                            </div>
                          ))
                        : (
                            // Message when no logs are available
                            <span className="text-gray-500 italic flex items-center justify-center h-full">
                                Logs will appear here once trading starts or actions occur...
                            </span>
                        )
                    }
                    {/* Empty div at the end to act as a target for scrolling */}
                    <div ref={logEndRef} style={{ height: '1px' }} />
                </pre>
            </div>
        </div>
    );
};

export default LogComponent;
EOF
print_success "Spell injected: frontend/src/components/LogComponent.jsx."

# Frontend src/App.jsx (Main application structure, enhanced state handling)
print_info "Injecting spell: frontend/src/App.jsx..."
cat << 'EOF' > src/App.jsx
// src/App.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import ChartComponent from './components/ChartComponent';
import ControlsComponent from './components/ControlsComponent';
import StatusComponent from './components/StatusComponent';
import LogComponent from './components/LogComponent';
import { getStatus } from './services/apiService';
import { Loader2, AlertTriangle, WifiOff, Activity } from 'lucide-react'; // Icons

// Constants
const STATUS_POLL_INTERVAL_MS = 10000; // Poll every 10 seconds

function App() {
    // State to hold the entire status object from the backend
    const [statusData, setStatusData] = useState(null); // e.g., { isTradingEnabled, config, logs, balance, position, error, lastUpdate }
    // State for critical errors preventing status updates (e.g., network, backend down)
    const [globalError, setGlobalError] = useState(null);
    // State for initial loading phase
    const [isLoadingInitial, setIsLoadingInitial] = useState(true);
    // Ref for the status polling interval timer
    const statusIntervalRef = useRef(null);
    // Ref to track component mount status
    const isMounted = useRef(true);

    // Callback function to fetch status from the backend
    const fetchStatus = useCallback(async (isInitial = false) => {
         // console.debug(`Fetching status... (Initial: ${isInitial})`);
         // Don't show loading indicator for background refreshes unless it's the initial load
         if (!isInitial && !isLoadingInitial) {
             // Optionally add a subtle refresh indicator somewhere if desired
         }

        try {
            const response = await getStatus(); // Fetches { success: true, data: {...} } or throws error

            if (!isMounted.current) return; // Don't update state if component unmounted

            if (response && response.success && response.data) {
                setStatusData(response.data);
                // Clear global error if fetch succeeds after a failure
                if (globalError) setGlobalError(null);
            } else {
                 // This case might indicate an issue with apiService returning unexpected format
                 // or backend sending success: false without error being thrown by axios
                 throw new Error(response?.error || "Received invalid status response from backend.");
            }
        } catch (err) {
            console.error("Critical Error fetching status:", err);
            if (!isMounted.current) return; // Don't update state if component unmounted

            // Set global error message based on the type of error
            // Keep existing status data visible if possible, but show the error prominently
            setGlobalError(`Failed to fetch status: ${err.message}`);

            // Optionally, stop polling if the error seems persistent (e.g., repeated network errors)
            // if (err.message.includes('Network Error')) { /* Add logic here */ }
        } finally {
             if (isMounted.current && isInitial) {
                 setIsLoadingInitial(false); // Mark initial load as complete
             }
        }
    }, [globalError]); // Include globalError dependency to potentially clear it

    // Effect for initial load and setting up the polling interval
    useEffect(() => {
        isMounted.current = true; // Set mounted ref

        // Fetch status immediately when component mounts
        fetchStatus(true); // Pass true for initial load

        // Clear any previous interval timer if component re-renders unexpectedly
        if (statusIntervalRef.current) {
            clearInterval(statusIntervalRef.current);
        }

        // Set up polling interval to fetch status periodically
        // console.log(`Setting up status polling interval: ${STATUS_POLL_INTERVAL_MS}ms`);
        statusIntervalRef.current = setInterval(() => fetchStatus(false), STATUS_POLL_INTERVAL_MS);

        // Cleanup function: clear interval when component unmounts
        return () => {
            // console.log("Clearing status polling interval.");
            isMounted.current = false; // Set unmounted ref
            if (statusIntervalRef.current) {
                clearInterval(statusIntervalRef.current);
                statusIntervalRef.current = null;
            }
        };
    }, [fetchStatus]); // Rerun effect if fetchStatus function identity changes (it shouldn't with useCallback [])

    // --- Render Logic ---
    return (
        // Main container with padding and max-width for larger screens
        <div className="container mx-auto p-3 sm:p-4 md:p-6 space-y-6 max-w-7xl min-h-screen flex flex-col">
            {/* Header */}
            <header className="text-center my-4">
                 <h1 className="text-2xl sm:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
                     Pyrmethus Trading Automaton
                 </h1>
                 <p className="text-sm text-gray-500">CCXT Bot Interface for Termux</p>
            </header>

            <main className="flex-grow space-y-6">
                {/* Global Error Display Area */}
                 {globalError && (
                     <div className="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded text-center flex items-center justify-center shadow-lg">
                         <WifiOff className="h-5 w-5 mr-3 flex-shrink-0"/> Critical Error: {globalError}
                     </div>
                 )}

                 {/* Initial Loading State */}
                 {isLoadingInitial && (
                     <div className="flex flex-col justify-center items-center text-gray-400 py-16">
                         <Loader2 className="animate-spin h-12 w-12 mb-4 text-blue-500" />
                         <p className="text-lg">Initializing Automaton Interface...</p>
                         <p className="text-sm">Connecting to backend and fetching initial status...</p>
                     </div>
                 )}

                 {/* Main Content Area - Render only after initial load attempt */}
                 {!isLoadingInitial && (
                     <>
                        {/* Controls Component */}
                        {/* Pass status safely, providing defaults if statusData is null */}
                        <ControlsComponent
                             initialConfig={statusData?.config} // Pass config object or undefined
                             isTradingEnabled={statusData?.isTradingEnabled || false} // Default to false if status is null
                             // Pass fetchStatus callback so Controls can trigger an immediate refresh after actions
                             onStatusChange={() => fetchStatus(false)}
                         />

                        {/* Chart and Status Grid */}
                         <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                             {/* Chart takes more space on larger screens */}
                             <div className="lg:col-span-2">
                                 {/* Pass relevant parts of status safely */}
                                 <ChartComponent
                                     symbol={statusData?.config?.symbol}
                                     interval={statusData?.config?.interval}
                                     position={statusData?.position} // Pass position data for reference lines
                                 />
                             </div>
                             {/* Status component */}
                             <div className="lg:col-span-1">
                                  {/* Pass the whole status object or null */}
                                  <StatusComponent statusData={statusData} />
                             </div>
                         </div>

                         {/* Log Component */}
                         {/* Pass logs array safely, defaulting to empty array */}
                         <LogComponent logs={statusData?.logs || []} />
                    </>
                )}
            </main>

             {/* Footer */}
             <footer className="text-center text-xs text-gray-600 mt-8 pb-4">
                 <p>Disclaimer: Trading involves substantial risk. This tool is for educational and experimental purposes.</p>
                 <p>Always test thoroughly in Sandbox/Testnet mode before considering live trading.</p>
                 <p>&copy; {new Date().getFullYear()} Pyrmethus. Use at your own risk.</p>
             </footer>
        </div>
    );
}

export default App;
EOF
print_success "Spell injected: frontend/src/App.jsx."

# Install Frontend Dependencies
print_info "Invoking npm spirits to install frontend dependencies... This may also take time."
if npm install --loglevel error; then # Reduce verbosity
    print_success "Frontend dependencies successfully installed."
else
    print_error "Frontend 'npm install' failed. Check 'frontend/package.json', network connection, and npm logs."
    exit 1
fi

# --- Phase 5: Final Incantations and Guidance ---
cd ../.. # Return to the root directory where the script was executed

# Final Success Message
echo -e "\n${COLOR_GREEN}=====================================================${COLOR_RESET}"
print_success "${COLOR_BOLD}Project setup spell cast successfully! Your trading automaton framework awaits in '${PROJECT_DIR}'.${COLOR_RESET}"
echo -e "${COLOR_GREEN}=====================================================${COLOR_RESET}\n"

# Crucial Next Steps and Warnings
echo -e "${COLOR_YELLOW}${COLOR_BOLD}🔮 IMPORTANT NEXT STEPS & WARNINGS 🔮${COLOR_RESET}"
echo -e "${COLOR_YELLOW}-----------------------------------------------------${COLOR_RESET}"
echo -e "${COLOR_CYAN}1. ${COLOR_BOLD}Verify Backend Configuration:${COLOR_RESET}"
echo -e "   - Navigate: ${COLOR_BLUE}cd ${PROJECT_DIR}/backend${COLOR_RESET}"
echo -e "   - Edit:     ${COLOR_BLUE}nano .env${COLOR_RESET}"
echo -e "   - ${COLOR_RED}${COLOR_BOLD}CRITICAL: Confirm API Keys are correct.${COLOR_RESET}"
echo -e "   - ${COLOR_RED}${COLOR_BOLD}CRITICAL: ENSURE 'USE_SANDBOX=true' IS SET FOR TESTNET!${COLOR_RESET}"
echo -e "   - Adjust strategy parameters (Risk, Intervals, etc.) as needed."
echo -e "${COLOR_YELLOW}-----------------------------------------------------${COLOR_RESET}"
echo -e "${COLOR_CYAN}2. ${COLOR_BOLD}Build the Frontend Application:${COLOR_RESET}"
echo -e "   - Navigate: ${COLOR_BLUE}cd ${PROJECT_DIR}/frontend${COLOR_RESET}"
echo -e "   - Execute:  ${COLOR_BLUE}npm run build${COLOR_RESET}"
echo -e "   - ${COLOR_YELLOW}(This creates the optimized 'build' folder served by the backend)${COLOR_RESET}"
echo -e "${COLOR_YELLOW}-----------------------------------------------------${COLOR_RESET}"
echo -e "${COLOR_CYAN}3. ${COLOR_BOLD}Summon the Backend Server:${COLOR_RESET}"
echo -e "   - Navigate: ${COLOR_BLUE}cd ${PROJECT_DIR}/backend${COLOR_RESET}"
echo -e "   - Recommended (for development): ${COLOR_BLUE}npm run dev${COLOR_RESET} (uses nodemon for auto-restarts)"
echo -e "   - Standard Start:            ${COLOR_BLUE}node src/server.js${COLOR_RESET}"
echo -e "   - ${COLOR_YELLOW}(Keep this terminal session running for the server)${COLOR_RESET}"
echo -e "${COLOR_YELLOW}-----------------------------------------------------${COLOR_RESET}"
echo -e "${COLOR_CYAN}4. ${COLOR_BOLD}Access Your Automaton Interface:${COLOR_RESET}"
echo -e "   - Open a web browser on a device connected to the ${COLOR_BOLD}same network${COLOR_RESET} as your Termux device."
echo -e "   - Go to:    ${COLOR_BLUE}http://<TERMUX_DEVICE_IP>:${DEFAULT_BACKEND_PORT}${COLOR_RESET}"
echo -e "   - (Find your Termux IP using the 'ip addr' or 'ifconfig' command)"
echo -e "   - Or, if accessing directly on the device: ${COLOR_BLUE}http://localhost:${DEFAULT_BACKEND_PORT}${COLOR_RESET}"
echo -e "${COLOR_YELLOW}-----------------------------------------------------${COLOR_RESET}"
echo -e "${COLOR_RED}${COLOR_BOLD}⚠️⚠️ EXTREME CAUTION ADVISED ⚠️⚠️${COLOR_RESET}"
echo -e "${COLOR_RED}▶ ${COLOR_BOLD}TEST THOROUGHLY:${COLOR_RESET} ${COLOR_YELLOW}Use Sandbox/Testnet mode EXCLUSIVELY until you fully understand the bot's behavior and risks.${COLOR_RESET}"
echo -e "${COLOR_RED}▶ ${COLOR_BOLD}LIVE TRADING IS HIGHLY RISKY.${COLOR_RESET} ${COLOR_YELLOW}You are solely responsible for any financial losses incurred.${COLOR_RESET}"
echo -e "${COLOR_RED}▶ ${COLOR_BOLD}NEVER SHARE YOUR API KEYS.${COLOR_RESET} ${COLOR_YELLOW}Do not commit the `.env` file to public repositories.${COLOR_RESET}"
echo -e "${COLOR_RED}▶ ${COLOR_BOLD}MONITOR CLOSELY:${COLOR_RESET} ${COLOR_YELLOW}Automated systems require supervision. Check logs and performance regularly.${COLOR_RESET}"
echo -e "${COLOR_YELLOW}-----------------------------------------------------${COLOR_RESET}"
print_success "May your automation journey be prosperous (and cautious)!"
echo -e "${COLOR_GREEN}=====================================================${COLOR_RESET}"

exit 0 # Indicate successful completion of the script
