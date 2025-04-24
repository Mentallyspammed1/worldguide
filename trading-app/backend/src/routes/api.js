// src/routes/api.js
const express = require('express');
const { initializeBybit, getBybit, fetchOHLCV, getMarketDetails } = require('../services/bybitService');
const strategyService = require('../services/strategyService');
const ccxt = require('ccxt'); // Import CCXT for error types

const router = express.Router();

// --- Middleware ---
// Ensures Bybit service is ATTEMPTED to initialize before handling API requests.
// It doesn't block if initialization fails here; individual routes should handle service readiness.
router.use(async (req, res, next) => {
    try {
        // Check if the instance exists, if not, attempt initialization ONCE.
        if (!getBybit()) {
            console.log("API Middleware: Bybit service instance not found, attempting initialization...");
            await initializeBybit(); // This might throw if keys are wrong etc.
            console.log("API Middleware: Bybit service initialization attempt complete.");
        }
        next(); // Proceed to the route handler regardless of initialization success here
    } catch (error) {
         // Initialization failed critically (e.g., bad keys)
         console.error("API Middleware CRITICAL ERROR: Failed to initialize Bybit service:", error.message);
         // Let subsequent routes handle the lack of a ready service, but log it here.
         // You could return 503 here, but it might mask issues if only certain routes need the service.
         // For this setup, let routes fail if they depend on a non-initialized service.
         next();
    }
});

// Helper function to handle route errors consistently
const handleRouteError = (res, error, functionName) => {
    console.error(`API Error in ${functionName}:`, error.constructor.name, error.message);
    let statusCode = 500; // Default to Internal Server Error
    let errorMessage = `Internal Server Error: ${error.message}`;

    // Handle specific CCXT or known errors
    if (error instanceof ccxt.AuthenticationError) {
        statusCode = 401; // Unauthorized
        errorMessage = "Authentication Failed: Invalid API Key or Secret.";
    } else if (error instanceof ccxt.BadSymbol) {
        statusCode = 404; // Not Found
        errorMessage = `Symbol Not Found or Invalid: ${error.message}`;
    } else if (error instanceof ccxt.RateLimitExceeded) {
        statusCode = 429; // Too Many Requests
        errorMessage = "API Rate Limit Exceeded. Please wait and try again.";
    } else if (error instanceof ccxt.ExchangeNotAvailable || error instanceof ccxt.NetworkError) {
        statusCode = 503; // Service Unavailable
        errorMessage = `Exchange or Network Unavailable: ${error.message}`;
    } else if (error.message.includes("Bybit service is not initialized") || error.message.includes("Bybit markets are not loaded")) {
         statusCode = 503; // Service Unavailable (backend issue)
         errorMessage = `Service Unavailable: ${error.message}`;
    } else if (error.message.startsWith("Invalid ")) { // Catch validation errors etc.
         statusCode = 400; // Bad Request
         errorMessage = error.message;
    }
    // Add more specific error handling as needed

    res.status(statusCode).json({ success: false, error: errorMessage });
};


// --- API Endpoints ---

// GET /api/status - Get current bot status (trading state, config, logs, balance, position)
router.get('/status', async (req, res) => {
    try {
        // getStatus internally checks if the service is ready
        const status = await strategyService.getStatus();
        res.status(200).json({ success: true, data: status });
    } catch (error) {
         handleRouteError(res, error, 'GET /status');
    }
});

// POST /api/trade/start - Start the trading bot loop
router.post('/trade/start', (req, res) => {
    try {
         // Pass config overrides from frontend request body (if any)
         // Basic validation: ensure body is an object if present
         const configUpdate = req.body;
         if (configUpdate && (typeof configUpdate !== 'object' || Array.isArray(configUpdate))) {
             throw new Error("Invalid configuration data format. Expected a JSON object.");
         }

        const result = strategyService.startTrading(configUpdate || {});
        if (result.success) {
            res.status(200).json({ success: true, message: result.message });
        } else {
             // If startTrading fails (e.g., invalid interval), return a 400 Bad Request
            res.status(400).json({ success: false, error: result.message });
        }
    } catch (error) {
         handleRouteError(res, error, 'POST /trade/start');
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
             res.status(200).json({ success: true, message: result.message }); // Still return success, message indicates state
        }
    } catch (error) {
         handleRouteError(res, error, 'POST /trade/stop');
    }
});

// POST /api/config - Update the bot's configuration
 router.post('/config', (req, res) => {
     try {
         // Basic validation: Ensure body is a non-null object
         const newConfig = req.body;
         if (typeof newConfig !== 'object' || newConfig === null || Array.isArray(newConfig)) {
            return res.status(400).json({ success: false, error: 'Invalid configuration data format. Expected a JSON object.' });
         }
         // More specific validation could be added here for field types/ranges if needed

         const result = strategyService.updateConfig(newConfig);
         if (result.success) {
            res.status(200).json({ success: true, message: "Configuration updated.", config: result.config });
         } else {
             // Should not happen if updateConfig handles errors, but as fallback
             res.status(400).json({ success: false, error: result.message || "Failed to update configuration." });
         }
     } catch (error) {
          handleRouteError(res, error, 'POST /config');
     }
 });

 // GET /api/symbols - Get available trading symbols from the exchange
 router.get('/symbols', async (req, res) => {
     try {
         const exchange = getBybit(); // Get instance (middleware should have tried init)
         if (!exchange || !exchange.markets || Object.keys(exchange.markets).length === 0) {
              // Check if markets are loaded; if not, maybe initialization failed earlier
              if (!exchange) throw new Error("Bybit service is not initialized.");
              // Attempt reload if instance exists but markets are missing (should not happen often)
             console.warn("API GET /symbols: Markets not loaded or empty. Attempting reload.");
             await exchange.loadMarkets();
             if (!exchange.markets || Object.keys(exchange.markets).length === 0) {
                 throw new Error("Failed to load markets from the exchange after attempting reload.");
             }
         }

         // Filter for active USDT Linear Perpetuals (common use case)
         // Adjust filters as needed for different market types (spot, inverse, etc.)
         const symbols = Object.values(exchange.markets) // Get market objects directly
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
          handleRouteError(res, error, 'GET /symbols');
     }
 });

 // GET /api/ohlcv - Get OHLCV data for charting
 router.get('/ohlcv', async (req, res) => {
     const { symbol, interval, limit = 200 } = req.query; // Default limit to 200

     try {
         // --- Input Validation ---
         if (!symbol || typeof symbol !== 'string' || symbol.trim() === '') {
             throw new Error('Missing or invalid required query parameter: symbol (string).');
         }
         if (!interval || typeof interval !== 'string' || interval.trim() === '') {
              throw new Error('Missing or invalid required query parameter: interval (string).');
         }
         const parsedLimit = parseInt(limit, 10);
         if (isNaN(parsedLimit) || parsedLimit <= 0 || parsedLimit > 1000) { // Set a reasonable max limit
              throw new Error('Invalid limit parameter. Must be a positive integer between 1 and 1000.');
         }
         // --- End Validation ---

          const data = await fetchOHLCV(symbol, interval, parsedLimit);
          res.status(200).json({ success: true, data: data });

     } catch (error) {
          handleRouteError(res, error, `GET /ohlcv (${symbol}/${interval})`);
     }
 });

module.exports = router;
