// src/server.js
const express = require('express');
const cors = require('cors');
require('dotenv').config(); // Load .env variables early
const apiRoutes = require('./routes/api');
const path = require('path');
const os = require('os'); // Required for network interface lookup
const { initializeBybit } = require('./services/bybitService'); // Import the initializer
const strategyService = require('./services/strategyService'); // Import for shutdown

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
    console.log(`[Server] ${new Date().toISOString()} - ${req.method} ${req.originalUrl} from ${req.ip}`);
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
    // Ignore requests that seem like API calls but weren't caught by apiRoutes
    if (req.path.startsWith('/api/')) {
        return res.status(404).send('API endpoint not found.');
    }

    const indexPath = path.resolve(frontendBuildPath, 'index.html');
    // console.log(`[Server] SPA Fallback: Serving index.html for path ${req.path}`);
    res.sendFile(indexPath, (err) => {
        if (err) {
            // Log the error but avoid sending detailed errors to the client unless necessary
            console.error("[Server] Error sending index.html:", err.message);
            // If the file simply doesn't exist (e.g., frontend not built yet)
            if (err.code === 'ENOENT') {
                 res.status(404).type('html').send( // Send HTML response for clarity
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
    let serverInstance; // To hold the server instance for graceful shutdown
    try {
        console.log("[Server] Initializing Bybit service connection before starting...");
        // CRITICAL: Wait for the Bybit service (including market loading) to be ready
        // initializeBybit() handles the singleton logic internally.
        await initializeBybit();
        console.log("[Server] Bybit service initialization attempt completed.");
        // Note: Initialization might have failed if keys are bad, but we proceed
        // letting the API routes handle the unavailable service.

        // Start listening for incoming HTTP requests
        serverInstance = app.listen(PORT, '0.0.0.0', () => { // Listen on all available network interfaces
            console.log("------------------------------------------------------");
            console.log(`✅  Backend server running on port ${PORT}`);
            console.log(`   - Local:   http://localhost:${PORT}`);
            // Try to get local network IP (might not work on all systems/Termux setups)
            try {
                 const interfaces = os.networkInterfaces();
                 for (const name of Object.keys(interfaces)) {
                    for (const iface of interfaces[name]) {
                        // Skip over internal (i.e. 127.0.0.1) and non-ipv4 addresses
                        if (iface.family === 'IPv4' && !iface.internal) {
                             console.log(`   - Network: http://${iface.address}:${PORT}`);
                             // Stop after finding the first external IPv4
                             break;
                        }
                    }
                 }
            } catch (e) { console.warn("[Server] Could not determine network IP:", e.message); }

            console.log(`   - API Root: http://localhost:${PORT}/api`);
            console.log(`   - Serving Frontend from: ${frontendBuildPath}`);
            console.log("------------------------------------------------------");
            if (process.env.USE_SANDBOX === 'true') {
                 // Use console.info or a specific color for sandbox mode
                 console.log(`[Server] \x1b[33mMode: SANDBOX/TESTNET\x1b[0m`);
            } else {
                 console.warn(`[Server] \x1b[31;1mMode: LIVE TRADING - EXERCISE EXTREME CAUTION!\x1b[0m`);
            }
            console.log("------------------------------------------------------");
            console.log("Waiting for requests...")
        });

         // Handle server listening errors (e.g., port already in use)
         serverInstance.on('error', (error) => {
             if (error.syscall !== 'listen') {
                 throw error;
             }
             const bind = typeof PORT === 'string' ? 'Pipe ' + PORT : 'Port ' + PORT;
             switch (error.code) {
                 case 'EACCES':
                     console.error(`[Server] ${bind} requires elevated privileges`);
                     process.exit(1);
                     break;
                 case 'EADDRINUSE':
                     console.error(`[Server] ${bind} is already in use`);
                     process.exit(1);
                     break;
                 default:
                     throw error;
             }
         });

    } catch (error) {
        // Handle critical startup errors (e.g., failed Bybit initialization due to fatal config issue)
        console.error("------------------------------------------------------");
        console.error("--- 💥 FATAL SERVER STARTUP ERROR 💥 ---");
        console.error("Failed to initialize critical services (e.g., Bybit connection):", error.message);
        console.error("The server cannot start without a valid exchange connection configuration.");
        console.error("Troubleshooting Tips:");
        console.error("  - Verify API keys and permissions in your Bybit account.");
        console.error("  - Check the '.env' file for correct key formatting.");
        console.error("  - Ensure you have a stable network connection.");
        console.error("  - Check Bybit status pages for potential outages.");
        console.error("  - Review CCXT library compatibility if errors mention specific functions.");
        console.error("------------------------------------------------------");
        process.exit(1); // Exit the process with an error code
    }

     // Optional: Graceful shutdown handling
     const shutdown = (signal) => {
         console.log(`\n[Server] Received ${signal}. Shutting down gracefully...`);
         // Stop the trading loop first
         strategyService.stopTrading();

         // Add other cleanup logic here (e.g., close database connections)

         // Close the HTTP server
         if (serverInstance) {
             serverInstance.close(() => {
                 console.log('[Server] HTTP server closed.');
                 // Exit the process after server is closed
                 process.exit(0);
             });
         } else {
              process.exit(0); // Exit if server wasn't successfully started
         }

         // Force exit if server doesn't close within a timeout
         setTimeout(() => {
             console.error('[Server] Could not close connections in time, forcing shutdown.');
             process.exit(1);
         }, 10000); // 10 seconds timeout
     };

     process.on('SIGINT', () => shutdown('SIGINT')); // Ctrl+C
     process.on('SIGTERM', () => shutdown('SIGTERM')); // kill command

};

// --- Initiate Server Startup ---
startServer();

// Catch unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('------------------------------------------------------');
  console.error('--- 💥 Unhandled Rejection at:', promise, 'reason:', reason, '💥 ---');
  console.error('------------------------------------------------------');
  // Consider exiting or implementing more robust error handling/logging
  // process.exit(1); // Optionally exit on unhandled rejections
});

// Catch uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('------------------------------------------------------');
  console.error('--- 💥 Uncaught Exception:', error, '💥 ---');
  console.error('------------------------------------------------------');
  // It's generally recommended to exit gracefully after an uncaught exception
  process.exit(1);
});
