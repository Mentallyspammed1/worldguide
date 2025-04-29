"""
App Configuration Module

This module configures the Flask application and database connection.
"""

import os
import logging
from datetime import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logger
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Define custom SQLAlchemy model class
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with model class
db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)

# Set up secret key from environment or use a default for development
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key_change_in_production")

# Configure ProxyFix for proper URL generation behind proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database connection
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,  # Recycle connections after 5 minutes
    "pool_pre_ping": True,  # Check connection validity before use
}

# Configure application
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["REMEMBER_COOKIE_SECURE"] = True
app.config["REMEMBER_COOKIE_HTTPONLY"] = True

# Path for storing configuration files
app.config["CONFIG_PATH"] = os.path.join(os.getcwd(), "config.json")
app.config["STATE_PATH"] = os.path.join(os.getcwd(), "bot_state.json")

# Initialize database with app
db.init_app(app)

# Initialize database tables within app context
with app.app_context():
    # Import models here to ensure they're registered with SQLAlchemy
    import models  # noqa: F401
    
    # Create all tables
    db.create_all()
    
    # Log successful initialization
    logger.info("Initialized database with default settings")

# Import views after app is configured to avoid circular imports
import views  # noqa: F401

# Initialize additional components here
logger.info("Application initialized successfully")