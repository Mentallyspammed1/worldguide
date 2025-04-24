#!/bin/bash

# Neon Colors (Bright ANSI)
RESET='\033[0m'
NEON_GREEN='\033[1;92m'
NEON_CYAN='\033[1;96m'
NEON_MAGENTA='\033[1;95m'
NEON_YELLOW='\033[1;93m'
BRIGHT_WHITE='\033[1;97m'
RED='\033[1;91m' # For emphasis
BOLD='\033[1m'

# Function for printing colored messages
cecho() {
  local color="$1"
  local message="$2"
  echo -e "${color}${message}${RESET}"
}

# --- Script Start ---
cecho "$NEON_CYAN" "${BOLD}🚀 Starting Full Trading Bot Setup (TRF + EIT + SL/TP + State + SMS + OB Filter)...${RESET}"
PROJECT_DIR="twin-range-bot"

# Check if directory exists
if [ -d "$PROJECT_DIR" ]; then
  cecho "$NEON_YELLOW" "⚠️ Directory '$PROJECT_DIR' already exists. Skipping creation."
else
  cecho "$NEON_GREEN" "📂 Creating project directory: $PROJECT_DIR"
  mkdir "$PROJECT_DIR"
  if [ $? -ne 0 ]; then
    cecho "$RED" "❌ Error creating directory. Aborting."
    exit 1
  fi
fi

cecho "$NEON_GREEN" "⤵️ Entering directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || exit 1 # Exit if cd fails

# --- Create .gitignore ---
cecho "$NEON_GREEN" "📄 Creating .gitignore file..."
cat <<'EOF' > .gitignore
# Node modules
node_modules

# Environment variables
.env

# Log files
*.log

# Bot state file
*.json

# OS generated files
.DS_Store
Thumbs.db
EOF
cecho "$NEON_MAGENTA" ".gitignore created."

# --- Create .env file ---
cecho "$NEON_GREEN" "🔑 Creating .env file (Add your API keys & phone number!)..."
cat <<EOF > .env
# .env file
# -----------------------------------------------------------------------------
# !! IMPORTANT: Replace placeholders with your actual Bybit API Key & Secret !!
# !! Ensure keys have Unified Trading Account TRADE permissions enabled      !!
# -----------------------------------------------------------------------------
BYBIT_API_KEY=YOUR_BYBIT_API_KEY_HERE
BYBIT_API_SECRET=YOUR_BYBIT_SECRET_HERE

# --- Trading Parameters ---
TIMEFRAME=5m           # Timeframe for candles (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
ORDER_SIZE_USD=10      # Amount in USD equivalent per trade
SYMBOL=                  # Optional: Pre-set symbol (e.g., BTC/USDT:USDT), otherwise bot will prompt

# --- Strategy Parameters ---
# Twin Range Filter (Default: 27, 1.6, 55, 2.0)
PER1=27
MULT1=1.6
PER2=55
MULT2=2.0
# Ehlers Instantaneous Trendline Confirmation (Alpha controls smoothness, smaller=smoother)
EIT_ALPHA_FAST=0.07
EIT_ALPHA_SLOW=0.05

# --- Risk Management ---
# !! REQUIRED > 0 FOR BOT TO OPEN NEW TRADES !!
STOP_LOSS_PERCENT=1.5  # Stop Loss percentage (e.g., 1.5 for 1.5%).
TAKE_PROFIT_PERCENT=3.0 # Take Profit percentage (e.g., 3.0 for 3.0%).

# --- Bot State ---
STATE_FILE=bot_state.json # File to save current position & entry price

# --- Termux SMS Alerts ---
# !! REQUIRES BOT TO RUN IN TERMUX ON ANDROID WITH TERMUX:API INSTALLED !!
SMS_ENABLE=false              # Set to true to enable SMS alerts
SMS_RECIPIENT_NUMBER=+1234567890 # !! REPLACE with YOUR phone number (include country code like +1) !!

# --- Order Book Filter ---
ORDERBOOK_FILTER_ENABLE=true       # Set to true to enable order book checks before entry
ORDERBOOK_MAX_SPREAD_PERCENT=0.1     # Max allowed % spread ((ask-bid)/mid)*100. Skip trade if wider. (e.g. 0.1 = 0.1%)
ORDERBOOK_IMBALANCE_THRESHOLD=0.6  # L1 Imbalance Ratio = BidSize / (BidSize+AskSize).
                                   # For LONG: Ratio must be >= threshold (e.g. 0.6 means 60% buy pressure needed)
                                   # For SHORT: Ratio must be <= (1 - threshold) (e.g. 0.4 means 60% sell pressure needed)
EOF
cecho "$NEON_MAGENTA" ".env created."
cecho "$RED" "${BOLD}👉 IMPORTANT: Edit '.env' now and add your API keys and SMS recipient phone number!${RESET}"
cecho "$NEON_YELLOW" "${BOLD}👉 ALSO: Set STOP_LOSS_PERCENT and TAKE_PROFIT_PERCENT > 0 to enable trading.${RESET}"


# --- Create package.json ---
cecho "$NEON_GREEN" "📦 Creating package.json file..."
cat <<'EOF' > package.json
{
  "name": "supercharged-twin-range-bot",
  "version": "1.1.0",
  "description": "Bybit UTA live trading bot: TRF + EIT + SL/TP + State + SMS + OB Filter",
  "main": "bot.js",
  "type": "module",
  "scripts": {
    "start": "node bot.js",
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [
    "ccxt",
    "bybit",
    "trading",
    "bot",
    "uta",
    "crypto",
    "termux",
    "ehlers"
  ],
  "author": "",
  "license": "MPL-2.0",
  "dependencies": {
    "ccxt": "^4.3.1",
    "dotenv": "^16.4.5",
    "readline": "^1.3.0"
  }
}
EOF
cecho "$NEON_MAGENTA" "package.json created."

# --- Create bot.js ---
cecho "$NEON_GREEN" "🤖 Creating bot.js file with all features..."
# Use 'cat <<'EOF'' to prevent shell variable/command expansion ($ ` \) inside the JS code
cat <<'EOF' > bot.js
// bot.js
import ccxt from 'ccxt';
import dotenv from 'dotenv';
import readline from 'readline';
import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process'; // For executing Termux commands

// --- Load Environment Variables ---
dotenv.config();

// --- Configuration ---
const exchangeId = 'bybit';
const apiKey = process.env.BYBIT_API_KEY;
const apiSecret = process.env.BYBIT_API_SECRET;

// Strategy & Trading Parameters
const defaultTimeframe = process.env.TIMEFRAME || '5m';
const orderSizeUSD = parseFloat(process.env.ORDER_SIZE_USD || '10');
const per1 = parseInt(process.env.PER1 || '27');
const mult1 = parseFloat(process.env.MULT1 || '1.6');
const per2 = parseInt(process.env.PER2 || '55');
const mult2 = parseFloat(process.env.MULT2 || '2.0');
const eitAlphaFast = parseFloat(process.env.EIT_ALPHA_FAST || '0.07');
const eitAlphaSlow = parseFloat(process.env.EIT_ALPHA_SLOW || '0.05');
const stopLossPercent = parseFloat(process.env.STOP_LOSS_PERCENT || '0');
const takeProfitPercent = parseFloat(process.env.TAKE_PROFIT_PERCENT || '0');
const stateFilePath = path.resolve(process.env.STATE_FILE || 'bot_state.json');
const presetSymbol = process.env.SYMBOL || '';

// Termux SMS Config
const smsEnable = process.env.SMS_ENABLE === 'true';
const smsRecipientNumber = process.env.SMS_RECIPIENT_NUMBER || '';

// Order Book Filter Config
const orderbookFilterEnable = process.env.ORDERBOOK_FILTER_ENABLE === 'true';
const orderbookMaxSpreadPercent = parseFloat(process.env.ORDERBOOK_MAX_SPREAD_PERCENT || '0.1');
const orderbookImbalanceThreshold = parseFloat(process.env.ORDERBOOK_IMBALANCE_THRESHOLD || '0.6'); // Threshold for LONG dominance

const source = 'close';
const requiredHistoryCandles = Math.max(per1, per2) * 3 + 2; // +2 for EIT lookback

// --- State Variables ---
let currentPosition = 'flat';
let entryPrice = null;
let ohlcvCache = []; // Stores [timestamp, open, high, low, close, volume]
let indicatorState = { // Stores previous values needed for calculations
    filt_1: null, upward_1: 0, downward_1: 0, FUB_1: null, FLB_1: null, TRF_1: null,
    ITrendFast_1: null, ITrendFast_2: null, ITrendSlow_1: null, ITrendSlow_2: null,
    src_1: null, src_2: null,
};
let selectedSymbol = '';
let isTermuxApiAvailable = false; // Flag to check if Termux API works

// --- Neon Colors for Console Output ---
const RESET = '\x1b[0m';
const NEON_GREEN = '\x1b[1;92m';
const NEON_CYAN = '\x1b[1;96m';
const NEON_MAGENTA = '\x1b[1;95m';
const NEON_YELLOW = '\x1b[1;93m';
const RED = '\x1b[1;91m';
const BOLD = '\x1b[1m';

// --- Helper Functions ---
const log = (color, ...args) => console.log(color, ...args, RESET);
const nz = (value, replacement = 0) => (value === null || isNaN(value) || typeof value === 'undefined') ? replacement : value;

// --- Termux SMS Alert Function ---
async function checkTermuxApi() {
    if (!smsEnable) return;
    return new Promise((resolve) => {
        // Use termux-toast which is usually less intrusive than sending a test SMS
        exec('termux-toast -g bottom "Bot API Check"', (error, stdout, stderr) => {
            if (error) {
                log(RED, `❌ Termux API check failed. SMS alerts disabled. Error: ${error.message}`);
                log(NEON_YELLOW, `   Ensure Termux:API app is installed via F-Droid/GitHub and run 'pkg install termux-api'.`);
                isTermuxApiAvailable = false;
            } else {
                log(NEON_GREEN, '✅ Termux API detected. SMS alerts enabled.');
                isTermuxApiAvailable = true;
            }
            resolve();
        });
    });
}

async function sendSmsAlert(message) {
    // Add symbol prefix automatically if selectedSymbol is available
    const prefix = selectedSymbol ? `[${selectedSymbol.split(':')[0]}] ` : '[BOT] '; // e.g. [BTC/USDT] or [BOT]
    const fullMessage = prefix + message;

    if (!smsEnable || !isTermuxApiAvailable) return;
    if (!smsRecipientNumber || !smsRecipientNumber.startsWith('+') || smsRecipientNumber.length < 10) { // Basic validation
        log(NEON_YELLOW, `⚠️ Cannot send SMS: SMS_RECIPIENT_NUMBER (${smsRecipientNumber}) is invalid or not set in .env. Needs country code.`);
        return;
    }
    // Sanitize message slightly for command line (remove chars that might break the command)
    const safeMessage = fullMessage.replace(/[`$&()|;"']/g, '');
    const command = `termux-sms-send -n "${smsRecipientNumber}" "${safeMessage}"`;

    // Execute the command - no need to await unless we need confirmation it sent
    exec(command, (error, stdout, stderr) => {
        if (error) {
            log(RED, `❌ Failed to send SMS: ${error.message}`);
            // Optional: Add logic to temporarily disable SMS after multiple failures
        }
        if (stderr && !stderr.includes('result: 0')) { // Sometimes stderr gives success messages too
            log(NEON_YELLOW, `SMS stderr: ${stderr}`);
        }
        // log(NEON_CYAN, `SMS alert sent: "${safeMessage}"`); // Uncomment for debugging SMS sends
    });
}

// --- State Persistence ---
async function saveState() {
    try {
        const state = { currentPosition, entryPrice, symbol: selectedSymbol }; // Also save symbol for context
        await fs.writeFile(stateFilePath, JSON.stringify(state, null, 2));
        // log(NEON_CYAN, `💾 State saved: ${JSON.stringify(state)}`);
    } catch (error) {
        log(RED, `❌ Error saving state:`, error.message);
        await sendSmsAlert(`ERR: Failed saving state file! ${error.message.substring(0,50)}`);
    }
}

async function loadState() {
    try {
        if (await fs.stat(stateFilePath).catch(() => false)) {
            const data = await fs.readFile(stateFilePath, 'utf8');
            const state = JSON.parse(data);
            // Only load state if symbol matches (or if no symbol saved yet)
            if (!state.symbol || state.symbol === selectedSymbol || !selectedSymbol) {
                currentPosition = state.currentPosition || 'flat';
                entryPrice = state.entryPrice || null;
                log(NEON_GREEN, `✅ State loaded: Pos=${currentPosition}, Entry=${entryPrice || 'N/A'}${state.symbol ? `, Symbol=${state.symbol}`:''}`);
            } else {
                 log(NEON_YELLOW, `🟡 State file symbol (${state.symbol}) doesn't match current (${selectedSymbol}). Starting fresh.`);
                 currentPosition = 'flat'; entryPrice = null;
            }
        } else {
            log(NEON_YELLOW, `🟡 State file not found. Starting fresh.`);
        }
    } catch (error) {
        log(RED, `❌ Error loading state:`, error.message);
        currentPosition = 'flat'; entryPrice = null;
    }
}


// --- CCXT Exchange Setup ---
const exchange = new ccxt[exchangeId]({
    apiKey: apiKey,
    secret: apiSecret,
    enableRateLimit: true,
    options: { defaultType: 'unified' },
});

// --- Indicator Calculations ---
// Ehlers Instantaneous Trendline Calculation (Simplified Filter)
function calculateEIT(src, src_1, src_2, alpha, prevTrend_1, prevTrend_2) {
    const alphaSq = alpha * alpha;
    const term1 = (alpha - alphaSq / 4) * src;
    const term2 = 0.5 * alphaSq * nz(src_1);
    const term3 = (alpha - 3 * alphaSq / 4) * nz(src_2);
    const term4 = 2 * (1 - alpha) * nz(prevTrend_1);
    const term5 = (1 - alpha) * (1 - alpha) * nz(prevTrend_2);
    return term1 + term2 - term3 + term4 - term5;
}
// PineScript rngfilt() function
function calculateRngfilt(x, r, prevFilt) {
    if (prevFilt === null) return x;
    let filt = x;
    if (x > prevFilt) { filt = (x - r < prevFilt) ? prevFilt : x - r; }
    else { filt = (x + r > prevFilt) ? prevFilt : x + r; }
    return filt;
}
// Main function to calculate all indicators for the latest candle
function calculateIndicatorsForLastCandle(currentCandle, prevState) {
    if (!currentCandle || ohlcvCache.length < 3) {
        log(RED, "calculateIndicators: Not enough candle data (need 3)."); return null;
    }
    const prevCandle1 = ohlcvCache[ohlcvCache.length - 2]; // Candle[1]
    const prevCandle2 = ohlcvCache[ohlcvCache.length - 3]; // Candle[2]
    const currentClose = currentCandle[4]; const prevClose1 = prevCandle1[4]; const prevClose2 = prevCandle2[4];

    const prevFilt = nz(prevState.filt_1); const prevUpward = nz(prevState.upward_1); const prevDownward = nz(prevState.downward_1);
    const prevFUB = prevState.FUB_1; const prevFLB = prevState.FLB_1; const prevTRF = prevState.TRF_1;
    const prevITrendFast1 = prevState.ITrendFast_1; const prevITrendFast2 = prevState.ITrendFast_2;
    const prevITrendSlow1 = prevState.ITrendSlow_1; const prevITrendSlow2 = prevState.ITrendSlow_2;
    const prevSrc1 = nz(prevState.src_1); const prevSrc2 = nz(prevState.src_2);

    // TRF Calculation (Simplified smooth range)
    const absDiff = Math.abs(currentClose - prevClose1);
    const avgRange1 = absDiff; const smrng1_val = avgRange1 * mult1;
    const avgRange2 = absDiff; const smrng2_val = avgRange2 * mult2;
    const smrng = (smrng1_val + smrng2_val) / 2;
    const filt = calculateRngfilt(currentClose, smrng, prevFilt);
    let upward = 0, downward = 0;
    if (filt > prevFilt) { upward = prevUpward + 1; downward = 0; } else if (filt < prevFilt) { upward = 0; downward = prevDownward + 1; } else { upward = prevUpward; downward = prevDownward; }
    const STR = filt + smrng; const STS = filt - smrng;
    let FUB = STR; if (prevFUB !== null) FUB = (STR < prevFUB || prevClose1 > prevFUB) ? STR : prevFUB;
    let FLB = STS; if (prevFLB !== null) FLB = (STS > prevFLB || prevClose1 < prevFLB) ? STS : prevFLB;
    let TRF = FUB;
    if (prevTRF !== null && prevFUB !== null && prevFLB !== null) { if (prevTRF === prevFUB) TRF = (currentClose <= FUB) ? FUB : FLB; else TRF = (currentClose >= FLB) ? FLB : FUB; }
    else if (prevFLB !== null) { TRF = (currentClose >= FLB) ? FLB : FUB; }

    // EIT Calculation
    const ITrendFast = calculateEIT(currentClose, prevClose1, prevClose2, eitAlphaFast, prevITrendFast1, prevITrendFast2);
    const ITrendSlow = calculateEIT(currentClose, prevClose1, prevClose2, eitAlphaSlow, prevITrendSlow1, prevITrendSlow2);

    // Signals & Confirmation
    const baseLong = prevClose1 <= nz(prevTRF) && currentClose > TRF;
    const baseShort = prevClose1 >= nz(prevTRF) && currentClose < TRF;
    const trendConfirmLong = currentClose > ITrendFast && currentClose > ITrendSlow && ITrendFast > ITrendSlow;
    const trendConfirmShort = currentClose < ITrendFast && currentClose < ITrendSlow && ITrendFast < ITrendSlow;
    const finalLong = baseLong && trendConfirmLong;
    const finalShort = baseShort && trendConfirmShort;

    // Next State Update
    const nextState = {
        filt_1: filt, upward_1: upward, downward_1: downward, FUB_1: FUB, FLB_1: FLB, TRF_1: TRF,
        ITrendFast_1: ITrendFast, ITrendFast_2: nz(prevITrendFast1),
        ITrendSlow_1: ITrendSlow, ITrendSlow_2: nz(prevITrendSlow1),
        src_1: currentClose, src_2: prevClose1,
    };

    return {
        timestamp: currentCandle[0], close: currentClose, TRF: TRF, ITrendFast: ITrendFast, ITrendSlow: ITrendSlow,
        longSignal: finalLong, shortSignal: finalShort, trendConfirmLong: trendConfirmLong, trendConfirmShort: trendConfirmShort,
        nextState: nextState,
    };
 }

// --- Order Book Analysis ---
async function analyzeOrderBookL1(symbol) {
    if (!orderbookFilterEnable) {
        return { proceed: true, reason: "Filter disabled" };
    }
    try {
        const orderbook = await exchange.fetchOrderBook(symbol, 1); // Fetch L1
        if (!orderbook?.bids?.length || !orderbook?.asks?.length) {
            log(NEON_YELLOW, "⚠️ Order book data unavailable/incomplete.");
            return { proceed: false, reason: "OB data unavailable" };
        }
        const bestBidPrice = orderbook.bids[0][0]; const bestBidSize = orderbook.bids[0][1];
        const bestAskPrice = orderbook.asks[0][0]; const bestAskSize = orderbook.asks[0][1];
        if (bestBidPrice <= 0 || bestAskPrice <= 0) return { proceed: false, reason: "Invalid OB prices" };

        const spread = bestAskPrice - bestBidPrice;
        const midPrice = (bestAskPrice + bestBidPrice) / 2;
        const spreadPercent = (spread / midPrice) * 100;
        const totalL1Size = bestBidSize + bestAskSize;
        const imbalanceRatio = totalL1Size > 0 ? bestBidSize / totalL1Size : 0.5; // Default to 0.5 if zero size

        const spreadOk = spreadPercent <= orderbookMaxSpreadPercent;
        // Imbalance check depends on signal direction (will be checked later)
        const imbalanceRequirementMetForLong = imbalanceRatio >= orderbookImbalanceThreshold;
        const imbalanceRequirementMetForShort = imbalanceRatio <= (1 - orderbookImbalanceThreshold);

        const details = `Spread=${spreadPercent.toFixed(3)}% (Max ${orderbookMaxSpreadPercent}%), Imb=${(imbalanceRatio * 100).toFixed(1)}% (Thresh ${orderbookImbalanceThreshold * 100}%)`;
        log(NEON_CYAN, `OrderBook L1: ${details}`);

        if (!spreadOk) {
            log(NEON_YELLOW, `🚦 OB Filter FAIL: Spread too wide.`);
            return { proceed: false, reason: `Spread wide (${spreadPercent.toFixed(3)}%)`, spreadOk, imbalanceRatio };
        }

        // Return details even if imbalance check might fail later
        return { proceed: true, reason: "Spread OK", spreadOk, imbalanceRatio, imbalanceRequirementMetForLong, imbalanceRequirementMetForShort };

    } catch (error) {
        log(RED, `❌ Error fetching/analyzing order book: ${error.message}`);
        await sendSmsAlert(`ERR: OrderBook fetch failed! ${error.message.substring(0,50)}`);
        return { proceed: false, reason: "OB fetch error" };
    }
}


// --- Trading Logic ---
async function fetchAndUpdateData(symbol, timeframe, limit) {
    try {
        // log(NEON_CYAN, `⏳ Fetching ${limit} ${timeframe} candles...`); // Less verbose fetching log
        const ohlcv = await exchange.fetchOHLCV(symbol, timeframe, undefined, limit);
        ohlcvCache = ohlcv;
        if (!ohlcv || ohlcv.length === 0) log(NEON_YELLOW, "⚠️ Fetched empty candle data.");
        // else log(NEON_GREEN, `📊 Fetched ${ohlcv.length}. Last: ${new Date(ohlcv[ohlcv.length - 1][0]).toISOString()}`); // Also verbose
        return ohlcv;
    } catch (error) {
        log(RED, `❌ Error fetching OHLCV: ${error.message}`);
        await sendSmsAlert(`ERR: Failed fetch OHLCV!`);
        return null;
    }
}
async function initializeIndicatorState(symbol, timeframe) {
     log(NEON_CYAN, `⏳ Initializing indicators for ${symbol}...`);
     await sendSmsAlert(`INFO: Initializing ${timeframe}...`);
     const initialData = await fetchAndUpdateData(symbol, timeframe, requiredHistoryCandles);
     if (!initialData || initialData.length < requiredHistoryCandles) {
         const msg = `FATAL: Not enough history for ${symbol} to initialize. Exiting.`; log(RED, msg);
         await sendSmsAlert(msg); process.exit(1);
     }
     let tempState = { /* initial empty state */ };
     for (let i = 2; i < initialData.length; i++) {
        const current = initialData[i]; const tempCache = initialData.slice(0, i + 1);
        const originalCache = ohlcvCache; ohlcvCache = tempCache;
        const result = calculateIndicatorsForLastCandle(current, tempState);
        ohlcvCache = originalCache; if (result) tempState = result.nextState;
     }
     indicatorState = tempState; log(NEON_GREEN, `✅ Indicator state initialized.`);
    //  await sendSmsAlert(`OK: Init complete.`); // Less verbose SMS
     ohlcvCache = ohlcvCache.slice(-3); // Keep only last 3 needed for next calculation
 }
// Function to place market order with SL/TP using Bybit UTA params
async function createMarketOrderWithSLTP(symbol, side, amount, price, slPercent, tpPercent) {
    const market = exchange.markets[symbol];
    if (!market) throw new Error(`Market ${symbol} not loaded`);
    const params = {}; let slPrice = null; let tpPrice = null;
    const pp = market.precision?.price || 4; // Price precision

    if (slPercent > 0) {
        slPrice = (side === 'buy') ? price * (1 - slPercent / 100) : price * (1 + slPercent / 100);
        params.stopLoss = exchange.priceToPrecision(symbol, slPrice);
    }
    if (tpPercent > 0) {
        tpPrice = (side === 'buy') ? price * (1 + tpPercent / 100) : price * (1 - tpPercent / 100);
        params.takeProfit = exchange.priceToPrecision(symbol, tpPrice);
    }

    const orderDesc = `${side.toUpperCase()} ${amount} @ ~${price.toFixed(pp)}`;
    const slTpDesc = `SL ${params.stopLoss || 'N/A'}, TP ${params.takeProfit || 'N/A'}`;
    log(NEON_CYAN, `🛒 Placing market ${orderDesc} | ${slTpDesc}`);
    await sendSmsAlert(`ORDER: ${orderDesc} | ${slTpDesc}`);

    try {
        const order = await exchange.createMarketOrder(symbol, side, amount, undefined, params);
        const avgPrice = order.average || order.price; // Use average if available
        log(NEON_GREEN, `✅ Order Placed: ID ${order.id}, AvgPx ${avgPrice ? avgPrice.toFixed(pp) : 'N/A'}`);
        await sendSmsAlert(`OK: ${side.toUpperCase()} ID ${order.id} Placed ${avgPrice ? ' AvgPx '+avgPrice.toFixed(pp) : ''}.`);
        return order;
    } catch (error) {
        log(RED, `❌ Error placing order: ${error.message}`);
        await sendSmsAlert(`ERR: Failed place ${side.toUpperCase()}! ${error.message.substring(0,60)}`);
        if (error instanceof ccxt.InsufficientFunds) log(RED, "Insufficient funds.");
        throw error; // Re-throw
    }
}

// Main check/trade loop
async function checkSignalAndTrade(symbol, timeframe) {
    let latestIndicators = null; let completedCandle = null;
    const pp = exchange.markets[symbol]?.precision?.price || 4; // Price precision

    // --- Fetch & Calculate ---
    try {
        const latestCandles = await exchange.fetchOHLCV(symbol, timeframe, undefined, 3);
        if (!latestCandles || latestCandles.length < 3) { log(NEON_YELLOW, "⚠️ No candle data."); return; }
        const lastCachedTs = ohlcvCache.length > 0 ? ohlcvCache[ohlcvCache.length - 1][0] : 0;
        const secondLatestTs = latestCandles[latestCandles.length - 2][0];

        if (secondLatestTs > lastCachedTs) { // New completed candle
            log(NEON_MAGENTA, `🕯️ New Candle: ${new Date(secondLatestTs).toISOString()}`);
            ohlcvCache = latestCandles.slice(); // Update cache
            completedCandle = ohlcvCache[ohlcvCache.length - 2];
            latestIndicators = calculateIndicatorsForLastCandle(completedCandle, indicatorState);
            if (latestIndicators) {
                indicatorState = latestIndicators.nextState; // Update state
                // Log main indicator values
                 log(NEON_CYAN, `Indicators: Close=${latestIndicators.close.toFixed(pp)}, TRF=${latestIndicators.TRF?.toFixed(pp)}, EITF=${latestIndicators.ITrendFast?.toFixed(pp)}, EITS=${latestIndicators.ITrendSlow?.toFixed(pp)}, Long=${latestIndicators.longSignal}, Short=${latestIndicators.shortSignal}`);
            } else { log(NEON_YELLOW, "⚠️ Indicator calc failed."); return; }
        } else { return; } // No new candle
    } catch (error) { log(RED, `❌ Error fetch/calc: ${error.message}`); return; }

    if (!latestIndicators) return; // Exit if calc failed

    // --- Position Sync ---
    try {
         const positionSymbol = symbol.replace('/', '');
         const positions = await exchange.fetchPositions([positionSymbol]);
         let currentPosSize = 0; let exchangeSide = 'flat'; let exchangeEntryPrice = null;
         if (positions?.length > 0) {
             const pos = positions.find(p => p?.info?.symbol === positionSymbol);
             if (pos?.contracts && Math.abs(parseFloat(pos.contracts)) > 0) {
                 currentPosSize = Math.abs(parseFloat(pos.contracts));
                 exchangeSide = (parseFloat(pos.contracts) > 0) ? 'long' : 'short';
                 exchangeEntryPrice = parseFloat(pos.entryPrice);
                 if (currentPosition !== exchangeSide || entryPrice !== exchangeEntryPrice) {
                     const prevPos = currentPosition; // Store previous position for message
                     log(NEON_YELLOW, `🔄 Syncing: Exchange has ${exchangeSide} ${currentPosSize} @ ${exchangeEntryPrice || 'N/A'}. Bot thought ${currentPosition}.`);
                     currentPosition = exchangeSide; entryPrice = exchangeEntryPrice; await saveState();
                     if (prevPos !== 'flat' && exchangeSide === 'flat') { // If bot thought it had a position, but now it's flat
                        await sendSmsAlert(`INFO: Position ${prevPos} closed (Sync/SL/TP?).`);
                     } else if (prevPos === 'flat' && exchangeSide !== 'flat') { // Position opened externally?
                        await sendSmsAlert(`WARN: Position ${exchangeSide} opened externally? Synced.`);
                     }
                 }
             } else if (currentPosition !== 'flat') { // No position on exchange, but bot thought there was
                  log(NEON_YELLOW, `🔄 Position closed on exchange. Syncing state to flat.`);
                  await sendSmsAlert(`INFO: Position ${currentPosition} closed (Sync/SL/TP?).`);
                  currentPosition = 'flat'; entryPrice = null; await saveState();
             }
         } else if (currentPosition !== 'flat') { // Error fetching or no position, sync to flat if needed
             log(NEON_YELLOW, "⚠️ Pos fetch failed/empty. Assuming flat.");
             currentPosition = 'flat'; entryPrice = null; await saveState();
         }

        // --- Order Book & Trade Execution ---
        const market = exchange.markets[symbol];
        const price = latestIndicators.close;
        const amount = orderSizeUSD / price;
        const preciseAmount = exchange.amountToPrecision(symbol, amount);
        const minAmount = market.limits?.amount?.min;

        if (parseFloat(preciseAmount) <= 0 || (minAmount && parseFloat(preciseAmount) < minAmount)) {
            log(RED, `❌ Amount ${preciseAmount} invalid/too small (Min: ${minAmount}).`); return;
        }
        const needSLTP = (latestIndicators.longSignal && currentPosition !== 'long') || (latestIndicators.shortSignal && currentPosition !== 'short');
        if (needSLTP && (stopLossPercent <= 0 || takeProfitPercent <= 0)) {
             log(RED, `❌ SL/TP percentages must be > 0 in .env to open NEW positions.`); return;
        }

        // --- OrderBook Check ---
        let orderBookCheck = { proceed: true, spreadOk: true, imbalanceRequirementMetForLong: true, imbalanceRequirementMetForShort: true }; // Assume pass if disabled
        if (orderbookFilterEnable && (latestIndicators.longSignal || latestIndicators.shortSignal)) {
            log(NEON_CYAN, `🧐 Analyzing Order Book...`);
            orderBookCheck = await analyzeOrderBookL1(symbol);
            if (!orderBookCheck.proceed || !orderBookCheck.spreadOk) { // Check generic proceed and spread first
                 log(NEON_YELLOW, `🚦 OB Filter REJECTED: ${orderBookCheck.reason}`);
                 await sendSmsAlert(`FILTER: Trade skipped. Reason: ${orderBookCheck.reason}`); return;
            }
        }

        // --- Execute Trades ---
        // LONG SIGNAL: Check base signal, not already long, and OB filter pass (spread + specific imbalance)
        if (latestIndicators.longSignal && currentPosition !== 'long' && orderBookCheck.spreadOk && orderBookCheck.imbalanceRequirementMetForLong) {
            const signalMsg = `SIGNAL: LONG @ ~${price.toFixed(pp)}`; log(NEON_GREEN, BOLD + signalMsg + RESET); await sendSmsAlert(signalMsg);
            if (currentPosition === 'short') { /* Close short logic... */
                log(NEON_YELLOW, `⏳ Closing short (${currentPosSize})...`);
                try { await exchange.createMarketOrder(symbol, 'buy', currentPosSize, undefined, { reduceOnly: true }); log(NEON_GREEN, `✅ Short closed.`); await sendSmsAlert(`INFO: Short closed for new LONG.`); currentPosition = 'flat'; entryPrice = null; await saveState(); await new Promise(r => setTimeout(r, 2000)); }
                catch (e) { log(RED, `❌ Err closing short: ${e.message}`); await sendSmsAlert(`ERR: Failed closing short!`); return; }
            }
            try { const order = await createMarketOrderWithSLTP(symbol, 'buy', preciseAmount, price, stopLossPercent, takeProfitPercent); currentPosition = 'long'; entryPrice = parseFloat(order.average || order.price || price); await saveState(); } catch (e) { /* Handled in order func */ }
        }
        // SHORT SIGNAL: Check base signal, not already short, and OB filter pass (spread + specific imbalance)
        else if (latestIndicators.shortSignal && currentPosition !== 'short' && orderBookCheck.spreadOk && orderBookCheck.imbalanceRequirementMetForShort) {
            const signalMsg = `SIGNAL: SHORT @ ~${price.toFixed(pp)}`; log(NEON_MAGENTA, BOLD + signalMsg + RESET); await sendSmsAlert(signalMsg);
             if (currentPosition === 'long') { /* Close long logic... */
                 log(NEON_YELLOW, `⏳ Closing long (${currentPosSize})...`);
                 try { await exchange.createMarketOrder(symbol, 'sell', currentPosSize, undefined, { reduceOnly: true }); log(NEON_GREEN, `✅ Long closed.`); await sendSmsAlert(`INFO: Long closed for new SHORT.`); currentPosition = 'flat'; entryPrice = null; await saveState(); await new Promise(r => setTimeout(r, 2000)); }
                 catch (e) { log(RED, `❌ Err closing long: ${e.message}`); await sendSmsAlert(`ERR: Failed closing long!`); return; }
             }
             try { const order = await createMarketOrderWithSLTP(symbol, 'sell', preciseAmount, price, stopLossPercent, takeProfitPercent); currentPosition = 'short'; entryPrice = parseFloat(order.average || order.price || price); await saveState(); } catch (e) { /* Handled in order func */ }
        }
        // OB Filter Failed Imbalance specifically (Spread was OK)
        else if (orderbookFilterEnable && latestIndicators.longSignal && currentPosition !== 'long' && orderBookCheck.spreadOk && !orderBookCheck.imbalanceRequirementMetForLong) {
             log(NEON_YELLOW, `🚦 OB Filter REJECTED LONG: Imbalance too low.`); await sendSmsAlert(`FILTER: LONG skipped. Low buy imbalance.`);
        }
        else if (orderbookFilterEnable && latestIndicators.shortSignal && currentPosition !== 'short' && orderBookCheck.spreadOk && !orderBookCheck.imbalanceRequirementMetForShort) {
             log(NEON_YELLOW, `🚦 OB Filter REJECTED SHORT: Imbalance too high.`); await sendSmsAlert(`FILTER: SHORT skipped. Low sell imbalance.`);
        }
        // Holding position
        else {
             if (currentPosition === 'long') log(NEON_CYAN, `Holding LONG.`);
             else if (currentPosition === 'short') log(NEON_CYAN, `Holding SHORT.`);
             // else log(NEON_CYAN, `No signal or condition met.`); // Too verbose
        }

    } catch (error) { // Catch errors in the main position sync / trade execution block
        log(RED, `❌ CRITICAL Error during trade/sync: ${error.message}`);
        await sendSmsAlert(`CRITICAL: Trade logic error! ${error.message.substring(0,100)}`);
        if (error instanceof ccxt.AuthenticationError) { log(RED, "AUTH ERROR!"); await sendSmsAlert("FATAL: AUTH ERROR! Check API keys!"); process.exit(1); }
    }
}

// --- Main Application Logic ---
async function run() {
    console.clear();
    log(NEON_CYAN, BOLD + "🚀 Starting Supercharged Twin Range Filter Bot..." + RESET);
    log(NEON_YELLOW, `   SMS Alerts: ${smsEnable}, OB Filter: ${orderbookFilterEnable}`);

    await checkTermuxApi(); // Check Termux early

    // API Key Check
    if (!apiKey || !apiSecret || apiKey === 'YOUR_BYBIT_API_KEY_HERE') {
        log(RED, "❌ API keys missing in .env!"); await sendSmsAlert("FATAL: API keys missing!"); process.exit(1);
    }

    // Load Markets
    try { await exchange.loadMarkets(); log(NEON_GREEN, "✅ Markets loaded."); }
    catch (error) { log(RED, "❌ Error loading markets:", error.message); await sendSmsAlert("FATAL: Cannot load markets."); process.exit(1); }

    // Select Symbol
    if (presetSymbol && exchange.markets[presetSymbol]?.contract) {
        selectedSymbol = presetSymbol; log(NEON_GREEN, `✅ Using preset symbol: ${selectedSymbol}`);
    } else {
        if (presetSymbol) log(NEON_YELLOW, `⚠️ Preset symbol "${presetSymbol}" invalid. Prompting...`);
        selectedSymbol = await askSymbol();
    }
    if (!selectedSymbol) { log(RED, "❌ No valid symbol selected. Exiting."); process.exit(1); }

    await loadState(); // Load state *after* symbol is known to check for match

    log(NEON_CYAN, `${BOLD}▶️ Trading: ${selectedSymbol} | TF: ${defaultTimeframe} | Size: ${orderSizeUSD} USD | SL: ${stopLossPercent}% | TP: ${takeProfitPercent}%${RESET}`);
    await sendSmsAlert(`START: ${selectedSymbol.split(':')[0]} ${defaultTimeframe}. Pos: ${currentPosition}. SL ${stopLossPercent}%, TP ${takeProfitPercent}%.`);

     // Check SL/TP config AFTER startup SMS sent
     if (stopLossPercent <= 0 || takeProfitPercent <= 0) {
         log(NEON_YELLOW, `⚠️ SL/TP not set > 0. Bot will NOT open NEW positions.`);
         await sendSmsAlert(`WARN: SL/TP not set > 0. No new trades will open.`);
     }


    // Initialize Indicators & Start Loop
    await initializeIndicatorState(selectedSymbol, defaultTimeframe);
    log(NEON_MAGENTA, `\n▶️ Starting main loop. Interval: ~${defaultTimeframe}...`);
    let intervalMillis = Math.max(15 * 1000, exchange.parseTimeframe(defaultTimeframe) * 1000);
    await checkSignalAndTrade(selectedSymbol, defaultTimeframe); // Initial check
    setInterval(() => checkSignalAndTrade(selectedSymbol, defaultTimeframe), intervalMillis);
}

// --- Symbol Prompt Function ---
function askSymbol() {
    return new Promise((resolve) => {
        const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
        const exampleSymbols = Object.keys(exchange.markets)
            .filter(s => exchange.markets[s].contract && (s.endsWith(':USDT') || s.endsWith('USDT')))
            .sort().slice(0, 15).join(', ');
        console.log(NEON_YELLOW + "\nAvailable contract market examples:" + RESET); console.log(NEON_CYAN + exampleSymbols + RESET);
        function promptUser() {
            rl.question(`\x1b[1;96mEnter Bybit market symbol (e.g., BTC/USDT:USDT): \x1b[0m`, (input) => {
                const S1 = input.toUpperCase().trim(); const S2 = S1.includes('/') ? S1 : `${S1.replace(/USDT$/, '')}/USDT`;
                const S3 = S2.endsWith(':USDT') ? S2 : `${S2}:USDT`; const S4 = S1.endsWith(':USDT') ? S1 : `${S1}:USDT`;
                const potential = [...new Set([S1, S2, S3, S4])]; const foundSymbol = potential.find(s => exchange.markets[s]?.contract);
                if (foundSymbol) { rl.close(); resolve(foundSymbol); }
                else { log(RED, `❌ Symbol '${input}' not found or invalid.`); promptUser(); }
            });
        } promptUser();
    });
 }

// --- Start the bot ---
run().catch(async error => { // Make catch async for final SMS
    log(RED, BOLD + "💥💥💥 UNHANDLED CRITICAL ERROR! 💥💥💥" + RESET);
    log(RED, error); // Log the full error object
    await sendSmsAlert(`FATAL CRASH: ${error?.message?.substring(0, 100) || 'Unknown error'}`);
    process.exit(1);
});

process.on('unhandledRejection', async (reason, promise) => {
  log(RED, BOLD+'Unhandled Rejection at:', promise, 'reason:', reason + RESET);
  await sendSmsAlert(`FATAL REJECTION: ${reason?.message?.substring(0,100) || reason}`);
  // Optionally exit, or let it potentially recover if it was minor (though usually indicates a bug)
  // process.exit(1);
});

process.on('uncaughtException', async (error, origin) => {
  log(RED, BOLD + `Uncaught Exception: ${error.message} at ${origin}` + RESET);
  log(RED, error.stack);
  await sendSmsAlert(`FATAL EXCEPTION: ${error?.message?.substring(0,100) || 'Unknown error'}`);
  process.exit(1); // Mandatory exit on uncaught exceptions
});
EOF
cecho "$NEON_MAGENTA" "bot.js created with all features."

# --- Install Dependencies ---
cecho "$NEON_GREEN" "💾 Installing Node.js dependencies (ccxt, dotenv, readline)..."
npm install
if [ $? -ne 0 ]; then
  cecho "$RED" "❌ Error installing dependencies with npm. Please check npm and network connection."
  cecho "$NEON_YELLOW" "Try running 'npm install' manually in the '$PROJECT_DIR' directory."
  exit 1
fi
cecho "$NEON_GREEN" "✅ Dependencies installed successfully."

# --- Final Instructions ---
cecho "$NEON_CYAN" "\n🎉 ${BOLD}Full Bot Setup Complete! 🎉${RESET}"
echo -e "${NEON_YELLOW}=======================================================================${RESET}"
echo -e "${BRIGHT_WHITE} ${BOLD}👉 Next Steps:${RESET}"
echo -e "${NEON_MAGENTA}   1. ${RED}${BOLD}EDIT '.env' FILE:${RESET}"
echo -e "${NEON_YELLOW}      - Add your actual Bybit API Key and Secret."
echo -e "${NEON_YELLOW}      - ${BOLD}Set your phone number in SMS_RECIPIENT_NUMBER (with country code like +1...).${RESET}"
echo -e "${NEON_YELLOW}      - ${BOLD}Set SMS_ENABLE=true if using Termux.${RESET}"
echo -e "${NEON_YELLOW}      - ${BOLD}Set STOP_LOSS_PERCENT and TAKE_PROFIT_PERCENT > 0 to enable trading.${RESET}"
echo -e "${NEON_YELLOW}      - Review other settings (Timeframe, Order Size, Strategy Params, OB Filter)."
echo ""
echo -e "${NEON_MAGENTA}   2. ${BOLD}TERMUX SETUP (If enabling SMS):${RESET}"
echo -e "${NEON_YELLOW}      - Ensure you are running this IN Termux on Android."
echo -e "${NEON_YELLOW}      - Install Termux:API app (from F-Droid/GitHub, NOT Play Store usually)."
echo -e "${NEON_YELLOW}      - Inside Termux, run: ${BOLD}pkg update && pkg install termux-api${RESET}"
echo -e "${NEON_YELLOW}      - Grant SMS permissions when the bot first tries the API (or manually via Android settings)."
echo ""
echo -e "${NEON_MAGENTA}   3. ${BOLD}RUN THE BOT:${RESET}"
echo -e "${NEON_GREEN}      cd $PROJECT_DIR && node bot.js${RESET}"
echo -e "${NEON_YELLOW}=======================================================================${RESET}"
cecho "$RED" "⚠️ ${BOLD}Disclaimer: Trading bots involve significant risk. TEST THOROUGHLY ON TESTNET FIRST. Use at your own risk. Ensure your SL/TP settings are appropriate.${RESET}"

exit 0
