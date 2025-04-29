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
from app import app

# Create database tables
with app.app_context():
    from app import db
    import models
    
    try:
        db.create_all()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")