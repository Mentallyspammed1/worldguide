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
