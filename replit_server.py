#!/usr/bin/env python
"""
Replit Server Entry Point

This script is specifically designed to run the Flask application in Replit's environment.
"""

import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("replit_server")

# Import after logging setup
from flask import Flask
from app import app

logger.info("Starting Flask application in Replit...")

# Make the app accessible to Replit
server = app

if __name__ == "__main__":
    # This shouldn't be called directly in Replit, but we include it just in case
    logger.info("Running application directly (not recommended in Replit)...")
    from app import socketio
    
    # Adding basic routes for health checks
    @app.route('/health')
    def health_check():
        return {"status": "ok", "service": "Pyrmethus Trading Bot"}, 200
    
    # Run the application
    try:
        socketio.run(app, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error running server: {e}")
        # Fall back to Flask's run method
        app.run(host='0.0.0.0', port=5000)