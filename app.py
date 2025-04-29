"""
Flask Application Configuration

This module initializes the Flask application and its extensions,
including SQLAlchemy for database ORM, Flask-Login for authentication,
and configuration loading.
"""

import os
import logging
from datetime import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set up SQLAlchemy base class
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with base class
db = SQLAlchemy(model_class=Base)

# Create Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "trading_bot_development_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # For proper URL generation behind proxy

# Load database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///trading_bot.db"
)
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
db.init_app(app)

# Import models (must be after db initialization)
with app.app_context():
    # Import models
    import models  # noqa: F401
    
    # Create tables if they don't exist
    db.create_all()
    
    # Initialize database with default settings if empty
    from models import User, Setting
    if not User.query.first() and not Setting.query.first():
        # Add default global settings
        default_settings = [
            {"key": "default_exchange", "value": "bybit", "value_type": "string"},
            {"key": "default_timeframe", "value": "15m", "value_type": "string"},
            {"key": "default_leverage", "value": "10.0", "value_type": "float"},
            {"key": "risk_percentage", "value": "1.0", "value_type": "float"},
            {"key": "max_positions", "value": "3", "value_type": "int"},
            {"key": "app_version", "value": "1.0.0", "value_type": "string"},
            {"key": "initialized_at", "value": datetime.utcnow().isoformat(), "value_type": "string"}
        ]
        
        for setting in default_settings:
            db.session.add(Setting(
                user_id=None,  # Global setting
                key=setting["key"],
                value=setting["value"],
                value_type=setting["value_type"]
            ))
        
        db.session.commit()
        app.logger.info("Initialized database with default settings")

# Import views after app is initialized
import views  # noqa: F401