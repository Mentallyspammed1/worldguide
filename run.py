#!/usr/bin/env python
"""
Custom Runner for Flask Application with SocketIO

This script provides a more reliable way to run the application in Replit
with WebSocket support.
"""

import logging
import os
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("runner")

logger.info("Starting Flask application runner...")

def run_app():
    """Run the Flask application with SocketIO support"""
    # Import necessary modules
    try:
        from app import app, socketio
        import socket_handlers  # Ensure socket handlers are registered
        import models  # Ensure models are registered
    except ImportError as e:
        logger.error(f"Error importing application modules: {e}")
        return 1
    
    # Run the application with SocketIO
    try:
        logger.info("Starting SocketIO server on port 5000")
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True  # Only use in development
        )
    except Exception as e:
        logger.error(f"Error running application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_app())