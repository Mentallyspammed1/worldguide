"""
Main Module

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

# Import app here to avoid circular imports
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

# Run the app using SocketIO instead of Flask's built-in server
if __name__ == "__main__":  # This block won't run under Gunicorn
    socketio.run(app, host="0.0.0.0", port=5000)