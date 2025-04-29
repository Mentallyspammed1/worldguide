"""
Database Setup Module

This module handles database initialization and migrations.
"""

import logging
import os

from app import app, db

# Configure logger
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database with required tables"""
    with app.app_context():
        try:
            # Import all models to ensure they're registered with SQLAlchemy
            import models
            
            # Create tables
            db.create_all()
            logger.info("Initialized database with default settings")
            
            # Check if we need to create an admin user
            from models import User
            from werkzeug.security import generate_password_hash
            
            if User.query.filter_by(username='admin').first() is None:
                # Create default admin user if it doesn't exist
                admin = User(
                    username='admin',
                    email='admin@example.com',
                    password_hash=generate_password_hash('admin'),
                    is_admin=True
                )
                db.session.add(admin)
                db.session.commit()
                logger.info("Created default admin user")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise