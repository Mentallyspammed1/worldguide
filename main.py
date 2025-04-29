"""
Main Module for Pyrmethus Trading Bot

This is the main entry point for the application.
"""

import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# Log app startup
logger.info("Pyrmethus Trading Bot starting up...")

# Import after logging setup
from app import app, socketio

# Create database tables
with app.app_context():
    from app import db
    import models
    
    try:
        db.create_all()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

# For direct execution
if __name__ == "__main__":
    try:
        socketio.run(
            app, 
            host="0.0.0.0", 
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    except TypeError as e:
        # Fall back to basic parameters if advanced options not supported
        logger.warning(f"SocketIO run error: {e}, falling back to basic parameters")
        socketio.run(app, host="0.0.0.0", port=5000)