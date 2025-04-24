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
