#!/usr/bin/env python
"""
Direct Server Runner for Replit

This script starts the Flask application with SocketIO support
in a way that's compatible with Replit's workflow system.
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("server_runner")

# Set environment variables
os.environ["FLASK_ENV"] = "development"

logger.info("Starting Flask application server...")

# Import necessary modules
try:
    # Import our app
    from app import app, socketio
    
    # Import dependencies to ensure they're registered
    import socket_handlers
    import models
except ImportError as e:
    logger.error(f"Failed to import application modules: {e}")
    sys.exit(1)

# Run the application
if __name__ == "__main__":
    try:
        # Run with SocketIO - this will work in Replit's environment
        logger.info("Starting application on port 5000...")
        # Try SocketIO's run method first
        try:
            socketio.run(
                app, 
                host="0.0.0.0", 
                port=5000,
                debug=False,
                allow_unsafe_werkzeug=True
            )
        except TypeError as e:
            # Fall back to Flask's run method if SocketIO parameters are incompatible
            logger.warning(f"SocketIO run error: {e}, falling back to Flask app.run()")
            app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)