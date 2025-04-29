"""
View functions for the trading bot web interface

This module contains the route handlers and view functions for the trading bot web interface,
including authentication, dashboard, configuration, and API endpoints.
"""

import json
import logging
import os
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Union

import ccxt
import pandas as pd
from flask import (
    Response, flash, jsonify, redirect, render_template, request, 
    session, url_for
)
from flask_login import (
    LoginManager, current_user, login_required, login_user, logout_user
)
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import generate_password_hash

from app import app, db
from models import ExchangeAccount, Position, Setting, Strategy, Trade, User
from trading_bot import TradingBot

# Configure logger
logger = logging.getLogger("views")

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    return User.query.get(int(user_id))


# -------------------------
# Auth routes
# -------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))
        else:
            flash("Invalid username or password", "danger")
    
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")
        
        # Validate input
        error = None
        if not username or not email or not password:
            error = "All fields are required"
        elif password != confirm:
            error = "Passwords must match"
        elif User.query.filter_by(username=username).first():
            error = f"User {username} already exists"
        elif User.query.filter_by(email=email).first():
            error = f"Email {email} already in use"
        
        if error:
            flash(error, "danger")
        else:
            # Create new user
            user = User(username=username, email=email)
            user.set_password(password)
            
            try:
                db.session.add(user)
                db.session.commit()
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for("login"))
            except SQLAlchemyError as e:
                db.session.rollback()
                logger.error(f"Database error during registration: {e}")
                flash("An error occurred during registration. Please try again.", "danger")
    
    return render_template("register.html")


# -------------------------
# Main routes
# -------------------------
@app.route("/")
def index():
    """Home page route"""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """Dashboard page route"""
    # Get user's exchange accounts
    exchange_accounts = ExchangeAccount.query.filter_by(user_id=current_user.id).all()
    
    # Get active positions
    positions = Position.query.filter_by(
        user_id=current_user.id, is_active=True
    ).order_by(Position.created_at.desc()).all()
    
    # Get recent trades
    trades = Trade.query.filter_by(
        user_id=current_user.id
    ).order_by(Trade.entry_time.desc()).limit(20).all()
    
    # Calculate performance metrics
    all_trades = Trade.query.filter_by(user_id=current_user.id).all()
    metrics = calculate_performance_metrics(all_trades)
    
    return render_template(
        "dashboard.html",
        exchange_accounts=exchange_accounts,
        positions=positions,
        trades=trades,
        metrics=metrics
    )


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    """User settings page route"""
    if request.method == "POST":
        # Process form submission
        pass
    
    # Get user settings
    user_settings = Setting.query.filter_by(user_id=current_user.id).all()
    global_settings = Setting.query.filter_by(user_id=None).all()
    
    return render_template(
        "settings.html",
        user_settings=user_settings,
        global_settings=global_settings
    )


@app.route("/accounts", methods=["GET", "POST"])
@login_required
def accounts():
    """Exchange account management page"""
    if request.method == "POST":
        # Process form submission
        action = request.form.get("action")
        
        if action == "add":
            # Add new exchange account
            name = request.form.get("name")
            exchange = request.form.get("exchange")
            api_key = request.form.get("api_key")
            api_secret = request.form.get("api_secret")
            is_testnet = "is_testnet" in request.form
            
            # Add validation here
            
            # Create new exchange account
            account = ExchangeAccount(
                user_id=current_user.id,
                name=name,
                exchange=exchange,
                api_key_encrypted=api_key,  # Should encrypt before storing
                api_secret_encrypted=api_secret,  # Should encrypt before storing
                is_testnet=is_testnet
            )
            
            try:
                db.session.add(account)
                db.session.commit()
                flash(f"Exchange account '{name}' added successfully", "success")
            except SQLAlchemyError as e:
                db.session.rollback()
                logger.error(f"Database error: {e}")
                flash("An error occurred. Please try again.", "danger")
        
        elif action == "delete":
            # Delete exchange account
            account_id = request.form.get("account_id")
            account = ExchangeAccount.query.filter_by(
                id=account_id, user_id=current_user.id
            ).first()
            
            if account:
                try:
                    db.session.delete(account)
                    db.session.commit()
                    flash(f"Exchange account '{account.name}' deleted", "success")
                except SQLAlchemyError as e:
                    db.session.rollback()
                    logger.error(f"Database error: {e}")
                    flash("An error occurred. Please try again.", "danger")
            else:
                flash("Exchange account not found", "danger")
    
    # Get user's exchange accounts
    exchange_accounts = ExchangeAccount.query.filter_by(user_id=current_user.id).all()
    
    return render_template(
        "accounts.html",
        exchange_accounts=exchange_accounts
    )


@app.route("/strategies", methods=["GET", "POST"])
@login_required
def strategies():
    """Strategy management page"""
    if request.method == "POST":
        # Process form submission
        pass
    
    # Get user's strategies
    user_strategies = Strategy.query.filter_by(user_id=current_user.id).all()
    
    return render_template(
        "strategies.html",
        strategies=user_strategies
    )


# -------------------------
# API routes
# -------------------------
@app.route("/api/performance", methods=["GET"])
@login_required
def api_performance():
    """API endpoint to get trading performance statistics"""
    # Get user's trades
    trades = Trade.query.filter_by(user_id=current_user.id).all()
    metrics = calculate_performance_metrics(trades)
    
    return jsonify(metrics)


@app.route("/api/market_data", methods=["GET"])
@login_required
def api_market_data():
    """API endpoint to get market data for charting"""
    symbol = request.args.get("symbol", "BTC/USDT:USDT")
    timeframe = request.args.get("timeframe", "15m")
    
    try:
        # Get exchange account
        account_id = request.args.get("account_id")
        if account_id:
            account = ExchangeAccount.query.filter_by(
                id=account_id, user_id=current_user.id
            ).first()
            
            if not account:
                return jsonify({"error": "Exchange account not found"}), 404
            
            # Use account credentials
            # In a real implementation, you'd decrypt the API credentials
            api_key = account.api_key_encrypted
            api_secret = account.api_secret_encrypted
            
            # Initialize trading bot with account credentials
            bot = TradingBot(
                exchange=account.exchange,
                api_key=api_key,
                api_secret=api_secret,
                is_testnet=account.is_testnet
            )
        else:
            # Use a temporary bot instance with defaults
            bot = TradingBot()
        
        # Fetch market data
        bot.set_symbol(symbol)
        bot.set_timeframe(timeframe)
        candles = bot.fetch_candles(limit=100)
        
        if candles is None or len(candles) == 0:
            return jsonify({"error": "No data available"}), 404
        
        # Format for charting
        chart_data = []
        for candle in candles:
            chart_data.append({
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            })
        
        # Fetch current ticker
        ticker = bot.fetch_ticker()
        
        return jsonify({
            "data": chart_data,
            "ticker": ticker,
            "last_update": datetime.utcnow().timestamp() * 1000
        })
    
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return jsonify({"error": str(e)}), 500


# -------------------------
# Helper functions
# -------------------------
def calculate_performance_metrics(trades: List[Trade]) -> Dict:
    """Calculate performance metrics from trade history"""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "average_win": 0,
            "average_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "monthly_returns": []
        }
    
    # Basic metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    # PnL metrics
    total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
    total_win_pnl = sum(t.pnl for t in winning_trades)
    total_loss_pnl = sum(t.pnl for t in losing_trades)
    
    profit_factor = (total_win_pnl / abs(total_loss_pnl)) if total_loss_pnl else float('inf')
    
    average_win = (total_win_pnl / win_count) if win_count > 0 else 0
    average_loss = (total_loss_pnl / loss_count) if loss_count > 0 else 0
    
    largest_win = max((t.pnl for t in winning_trades), default=0)
    largest_loss = min((t.pnl for t in losing_trades), default=0)
    
    # Calculate drawdown
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)
    cumulative_pnl = 0
    peak = 0
    drawdown = 0
    max_drawdown = 0
    
    for trade in sorted_trades:
        if trade.pnl is not None:
            cumulative_pnl += trade.pnl
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
    
    # Format and return metrics
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "average_win": round(average_win, 2),
        "average_loss": round(average_loss, 2),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_drawdown, 2)
    }