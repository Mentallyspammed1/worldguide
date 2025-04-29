"""
Main Application Module

This is the entry point for the Pyrmethus trading bot application.
It imports the Flask app and sets up any additional configuration.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.getcwd(), 'bot_logs', 'app.log'), mode='a')
    ]
)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.getcwd(), 'bot_logs'), exist_ok=True)

# Import Flask app
try:
    from app import app
    logger = logging.getLogger("main")
    logger.info("Pyrmethus Trading Bot starting up...")
except Exception as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

if __name__ == "__main__":
    # Start the Flask development server
    # Note: In production, use a proper WSGI server like gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)