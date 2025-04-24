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
