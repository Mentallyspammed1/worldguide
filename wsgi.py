"""
WSGI Entry Point

This file is used for Gunicorn to serve the application with WebSocket support.
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wsgi")

# Log app startup
logger.info("WSGI application starting...")

# Import SocketIO app
from app import app, socketio

# Create database tables if they don't exist
with app.app_context():
    from app import db
    import models
    
    try:
        db.create_all()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

# Import socket handlers
import socket_handlers

# Create application for Gunicorn with WebSocket support
# The variable name "application" is what Gunicorn looks for by default
# For Flask-SocketIO, the app itself can be used
application = app

# For direct execution
if __name__ == "__main__":
    logger.info("Running directly with SocketIO")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)