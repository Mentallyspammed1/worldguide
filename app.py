"""
App Configuration Module

This module configures the Flask application and database connection.
"""

import os
import logging

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logger = logging.getLogger("app")

class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)

# create the app
app = Flask(__name__)

# setup a secret key, required by sessions
app.secret_key = os.environ.get("SESSION_SECRET", "pyrmethus_dev_key")

# If behind a proxy, enable this for proper hostname/protocol detection
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Debug mode configuration
app.config["DEBUG"] = True if os.environ.get("FLASK_ENV") == "development" else False

# Configure application paths
app.config["CONFIG_PATH"] = os.environ.get("CONFIG_PATH", "config.json")
app.config["DATA_PATH"] = os.environ.get("DATA_PATH", "data")

# Initialize the app with extensions
db.init_app(app)

# Initialize SocketIO with async mode that's compatible with the environment
# Note: We don't specify async_mode, letting Flask-SocketIO choose the best available
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    logger=True,  # Enable logging
    engineio_logger=True  # Show detailed connection logs
)

# Register error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal server error: {e}")
    return render_template('errors/500.html'), 500

# Create all tables on startup
with app.app_context():
    # Import the models here to be included in create_all
    import models
    from models import User, TradeHistory

    # Try to create database tables
    try:
        db.create_all()
        logger.info("Database tables created or verified")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

# Import views after app initialization to avoid circular imports
from flask import render_template
import views

# Initialize WebSocket handlers
import socket_handlers
socket_handlers.init_socket_handlers()