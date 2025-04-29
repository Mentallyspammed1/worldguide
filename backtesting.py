"""
Advanced Backtesting Module

This module provides sophisticated backtesting capabilities with:
- Multi-timeframe testing
- Transaction cost modeling
- Position sizing strategies
- Realistic slippage simulation
- Equity curve calculation
- Performance metrics
- Drawdown analysis
- Monte Carlo simulations
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from position_management import Position, PositionManager
from ehlers_supertrend_strategy import calculate_ehlers_supertrend_signal
from strategies import (
    calculate_momentum_divergence_strategy,
    calculate_multi_timeframe_trend_strategy,
    calculate_support_resistance_breakout_strategy
)

# Configure logger
logger = logging.getLogger("trading_bot.backtest")

class BacktestResult:
    """Class representing backtest results with analysis methods"""
    
    def __init__(self, 
                 trades: List[Dict], 
                 equity_curve: pd.DataFrame,
                 initial_balance: float,
                 settings: Dict,
                 positions: List[Position] = None,
                 metadata: Dict = None):
        """
        Initialize backtest results
        
        Args:
            trades: List of trade dictionaries
            equity_curve: Equity curve DataFrame
            initial_balance: Initial balance
            settings: Backtest settings
            positions: List of Position objects
            metadata: Additional backtest metadata
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_balance = initial_balance
        self.settings = settings
        self.positions = positions or []
        self.metadata = metadata or {}
        
        # Compute metrics
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        if not self.trades or len(self.trades) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "total_return": 0,
                "total_return_pct": 0,
                "annualized_return": 0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "profit_to_max_drawdown": 0,
                "avg_trade_duration": 0
            }
        
        # Prepare equity curve if it doesn't have required columns
        if "balance" not in self.equity_curve.columns:
            logger.warning("Equity curve missing 'balance' column, using trades to reconstruct")
            self._reconstruct_equity_curve()
        
        # Trade metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in self.trades if t.get("pnl", 0) < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        winning_amounts = [t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0]
        losing_amounts = [abs(t.get("pnl", 0)) for t in self.trades if t.get("pnl", 0) < 0]
        
        avg_win = sum(winning_amounts) / len(winning_amounts) if winning_amounts else 0
        avg_loss = sum(losing_amounts) / len(losing_amounts) if losing_amounts else 0
        
        gross_profit = sum(winning_amounts)
        gross_loss = sum(losing_amounts)
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0 if gross_profit == 0 else float('inf')
        
        # Return metrics
        final_balance = self.equity_curve['balance'].iloc[-1]
        total_return = final_balance - self.initial_balance
        total_return_pct = (final_balance / self.initial_balance - 1) * 100
        
        # Drawdown metrics
        drawdowns = self._calculate_drawdowns()
        max_drawdown = drawdowns["max_drawdown"]
        max_drawdown_pct = drawdowns["max_drawdown_pct"]
        
        # Consecutive win/loss streaks
        streaks = self._calculate_streaks()
        max_consecutive_wins = streaks["max_wins"]
        max_consecutive_losses = streaks["max_losses"]
        
        # Risk-adjusted metrics
        returns = self.equity_curve['return_pct'].dropna()
        
        sharpe_ratio = returns.mean() / returns.std() if not returns.empty and returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Calculate annualized return
        if len(self.equity_curve) > 1:
            start_date = self.equity_curve.index[0]
            end_date = self.equity_curve.index[-1]
            
            # Check if index is datetime
            if isinstance(start_date, (pd.Timestamp, datetime)):
                days = (end_date - start_date).days
                years = days / 365
                
                if years > 0:
                    annualized_return = (1 + total_return_pct/100) ** (1/years) - 1
                    annualized_return = annualized_return * 100  # Convert to percentage
                else:
                    annualized_return = 0
            else:
                annualized_return = 0
        else:
            annualized_return = 0
        
        # Calculate average trade duration
        durations = []
        for trade in self.trades:
            entry_time = trade.get("entry_timestamp")
            exit_time = trade.get("exit_timestamp")
            
            if entry_time and exit_time:
                duration_seconds = (exit_time - entry_time) / 1000  # Convert ms to seconds
                durations.append(duration_seconds)
        
        avg_trade_duration = sum(durations) / len(durations) if durations else 0
        
        # Additional metrics
        profit_to_max_drawdown = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate * 100,  # Convert to percentage
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "annualized_return": annualized_return,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "profit_to_max_drawdown": profit_to_max_drawdown,
            "avg_trade_duration": avg_trade_duration
        }
    
    def _calculate_drawdowns(self) -> Dict:
        """
        Calculate drawdown metrics
        
        Returns:
            Dict: Drawdown metrics
        """
        # Calculate running maximum
        if 'balance' not in self.equity_curve.columns:
            return {"max_drawdown": 0, "max_drawdown_pct": 0}
            
        equity_curve = self.equity_curve.copy()
        equity_curve['peak'] = equity_curve['balance'].cummax()
        
        # Calculate drawdown in currency and percentage
        equity_curve['drawdown'] = equity_curve['peak'] - equity_curve['balance']
        equity_curve['drawdown_pct'] = (equity_curve['drawdown'] / equity_curve['peak']) * 100
        
        # Get maximum drawdown
        max_drawdown = equity_curve['drawdown'].max()
        max_drawdown_pct = equity_curve['drawdown_pct'].max()
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        current_peak = None
        
        for idx, row in equity_curve.iterrows():
            if row['balance'] < row['peak']:
                # In drawdown
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = idx
                    current_peak = row['peak']
            elif in_drawdown:
                # End of drawdown
                in_drawdown = False
                drawdown_end = idx
                recovery = row['balance']
                
                drawdown_amount = current_peak - equity_curve.loc[drawdown_start:drawdown_end, 'balance'].min()
                drawdown_pct = (drawdown_amount / current_peak) * 100
                duration = (drawdown_end - drawdown_start).total_seconds() / (24 * 3600)  # days
                
                if drawdown_pct > 1.0:  # Only record significant drawdowns
                    drawdown_periods.append({
                        "start": drawdown_start,
                        "end": drawdown_end,
                        "amount": drawdown_amount,
                        "percentage": drawdown_pct,
                        "duration_days": duration,
                        "recovery": recovery
                    })
        
        # Check if still in drawdown at end of period
        if in_drawdown:
            drawdown_end = equity_curve.index[-1]
            drawdown_amount = current_peak - equity_curve.loc[drawdown_start:drawdown_end, 'balance'].min()
            drawdown_pct = (drawdown_amount / current_peak) * 100
            duration = (drawdown_end - drawdown_start).total_seconds() / (24 * 3600)  # days
            
            if drawdown_pct > 1.0:
                drawdown_periods.append({
                    "start": drawdown_start,
                    "end": drawdown_end,
                    "amount": drawdown_amount,
                    "percentage": drawdown_pct,
                    "duration_days": duration,
                    "recovery": "In progress"
                })
        
        # Sort drawdown periods by size
        drawdown_periods.sort(key=lambda x: x["percentage"], reverse=True)
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "drawdown_periods": drawdown_periods
        }
    
    def _calculate_streaks(self) -> Dict:
        """
        Calculate trade streaks
        
        Returns:
            Dict: Streak metrics
        """
        if not self.trades:
            return {"max_wins": 0, "max_losses": 0, "current_streak": 0}
        
        # Calculate streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in self.trades:
            pnl = trade.get("pnl", 0)
            
            if pnl > 0:
                # Winning trade
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                
                max_win_streak = max(max_win_streak, current_streak)
            elif pnl < 0:
                # Losing trade
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                
                max_loss_streak = max(max_loss_streak, abs(current_streak))
            # Flat trades (pnl = 0) don't affect streaks
        
        return {
            "max_wins": max_win_streak,
            "max_losses": max_loss_streak,
            "current_streak": current_streak
        }
    
    def _reconstruct_equity_curve(self) -> None:
        """Reconstruct equity curve from trades"""
        if len(self.trades) == 0:
            logger.warning("No trades available to reconstruct equity curve")
            return
        
        # Create timestamps list with all trade entries and exits
        timestamps = []
        for trade in self.trades:
            entry_time = trade.get("entry_timestamp")
            exit_time = trade.get("exit_timestamp")
            
            if entry_time:
                timestamps.append(entry_time)
            if exit_time:
                timestamps.append(exit_time)
        
        # Convert timestamps to datetime
        datetimes = [datetime.fromtimestamp(ts/1000) for ts in sorted(set(timestamps))]
        
        # Create empty DataFrame
        self.equity_curve = pd.DataFrame(index=datetimes)
        self.equity_curve['balance'] = self.initial_balance
        
        # Apply trades to calculate equity curve
        for trade in sorted(self.trades, key=lambda x: x.get("exit_timestamp", 0)):
            exit_time = trade.get("exit_timestamp")
            if exit_time:
                exit_dt = datetime.fromtimestamp(exit_time/1000)
                pnl = trade.get("pnl", 0)
                
                # Update balance for all points after this trade
                mask = self.equity_curve.index >= exit_dt
                self.equity_curve.loc[mask, 'balance'] += pnl
        
        # Calculate returns
        self.equity_curve['return'] = self.equity_curve['balance'].pct_change()
        self.equity_curve['return_pct'] = self.equity_curve['return'] * 100
    
    def plot_equity_curve(self, filepath: str = None) -> Optional[Figure]:
        """
        Plot equity curve
        
        Args:
            filepath: Path to save plot image (optional)
            
        Returns:
            Figure: Matplotlib figure (if filepath not provided)
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot equity curve
            self.equity_curve['balance'].plot(ax=ax, label='Equity')
            
            # Add equity peaks
            peaks = self.equity_curve['balance'].cummax()
            peaks.plot(ax=ax, linestyle='--', color='green', alpha=0.5, label='Equity Peak')
            
            # Shade drawdown periods
            ax.fill_between(self.equity_curve.index, 
                           self.equity_curve['balance'], 
                           peaks, 
                           where=peaks > self.equity_curve['balance'],
                           color='red', alpha=0.3, label='Drawdowns')
            
            # Add annotations for entry and exit points
            for trade in self.trades:
                entry_time = trade.get("entry_timestamp")
                exit_time = trade.get("exit_timestamp")
                pnl = trade.get("pnl", 0)
                
                if entry_time and exit_time:
                    entry_dt = datetime.fromtimestamp(entry_time/1000)
                    exit_dt = datetime.fromtimestamp(exit_time/1000)
                    
                    # Get balance at entry and exit
                    try:
                        entry_balance = self.equity_curve.loc[self.equity_curve.index <= entry_dt, 'balance'].iloc[-1]
                        exit_balance = entry_balance + pnl
                        
                        # Add markers for entry and exit
                        if pnl > 0:
                            ax.scatter(exit_dt, exit_balance, color='green', s=30, alpha=0.7)
                        elif pnl < 0:
                            ax.scatter(exit_dt, exit_balance, color='red', s=30, alpha=0.7)
                    except (IndexError, KeyError):
                        # Skip if we can't find the balance at this time
                        continue
            
            # Add key metrics as text
            metrics_text = (
                f"Total Return: {self.metrics['total_return_pct']:.2f}%\n"
                f"Win Rate: {self.metrics['win_rate']:.2f}%\n"
                f"Profit Factor: {self.metrics['profit_factor']:.2f}\n"
                f"Max Drawdown: {self.metrics['max_drawdown_pct']:.2f}%\n"
                f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}"
            )
            
            # Position text in the upper left with a semi-transparent background
            ax.text(0.02, 0.98, metrics_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add title and labels
            symbol = self.settings.get("symbol", "Unknown")
            timeframe = self.settings.get("timeframe", "Unknown")
            strategy = self.settings.get("strategy", "Unknown")
            
            ax.set_title(f"Equity Curve - {symbol} {timeframe} ({strategy})", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Equity", fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or return figure
            if filepath:
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved equity curve plot to {filepath}")
                return None
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            return None
    
    def plot_drawdowns(self, filepath: str = None) -> Optional[Figure]:
        """
        Plot drawdowns
        
        Args:
            filepath: Path to save plot image (optional)
            
        Returns:
            Figure: Matplotlib figure (if filepath not provided)
        """
        # Calculate drawdowns if not already done
        drawdowns = self._calculate_drawdowns()
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot drawdown percentage over time
            drawdown_curve = (self.equity_curve['balance'] / self.equity_curve['balance'].cummax() - 1) * 100
            drawdown_curve.plot(ax=ax, color='red', label='Drawdown %')
            
            # Add horizontal lines at key drawdown levels
            ax.axhline(y=-5, color='yellow', linestyle='--', alpha=0.7, label='-5%')
            ax.axhline(y=-10, color='orange', linestyle='--', alpha=0.7, label='-10%')
            ax.axhline(y=-20, color='red', linestyle='--', alpha=0.7, label='-20%')
            
            # Add annotations for major drawdown periods
            major_drawdowns = [d for d in drawdowns.get("drawdown_periods", []) 
                              if d["percentage"] > 10]
            
            for i, dd in enumerate(major_drawdowns[:5]):  # Limit to top 5
                start = dd["start"]
                end = dd["end"]
                pct = dd["percentage"]
                
                # Add text annotation at the bottom of the drawdown
                min_idx = drawdown_curve.loc[start:end].idxmin()
                min_value = drawdown_curve.loc[min_idx]
                
                ax.annotate(f"{pct:.1f}%", 
                           xy=(min_idx, min_value), 
                           xytext=(min_idx, min_value - 5),
                           arrowprops=dict(arrowstyle="->", color='black', alpha=0.7),
                           fontsize=9, ha='center')
            
            # Add title and labels
            symbol = self.settings.get("symbol", "Unknown")
            timeframe = self.settings.get("timeframe", "Unknown")
            strategy = self.settings.get("strategy", "Unknown")
            
            ax.set_title(f"Drawdown Analysis - {symbol} {timeframe} ({strategy})", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Drawdown %", fontsize=12)
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits
            min_dd = min(drawdown_curve.min(), -25)  # At least -25%
            ax.set_ylim(min_dd, 5)  # Top at +5%
            
            # Add key metrics as text
            metrics_text = (
                f"Max Drawdown: {drawdowns['max_drawdown_pct']:.2f}%\n"
                f"Longest Underwater Period: {max([d['duration_days'] for d in drawdowns.get('drawdown_periods', [{'duration_days': 0}])]):.1f} days\n"
                f"Profit/Max Drawdown: {self.metrics['profit_to_max_drawdown']:.2f}\n"
                f"# of 5%+ Drawdowns: {len([d for d in drawdowns.get('drawdown_periods', []) if d['percentage'] >= 5])}"
            )
            
            # Position text in the upper right with a semi-transparent background
            ax.text(0.98, 0.98, metrics_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or return figure
            if filepath:
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved drawdown plot to {filepath}")
                return None
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting drawdowns: {e}")
            return None
    
    def to_dict(self) -> Dict:
        """
        Convert backtest result to dictionary
        
        Returns:
            Dict: Results as dictionary
        """
        # Convert equity curve to serializable format
        equity_curve_dict = {
            "timestamps": [int(ts.timestamp() * 1000) for ts in self.equity_curve.index],
            "balance": self.equity_curve["balance"].tolist()
        }
        
        return {
            "metrics": self.metrics,
            "trades": self.trades,
            "equity_curve": equity_curve_dict,
            "initial_balance": self.initial_balance,
            "settings": self.settings,
            "metadata": self.metadata
        }
    
    def to_json(self, filepath: str) -> None:
        """
        Save backtest results to JSON file
        
        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved backtest results to {filepath}")


class Backtester:
    """Advanced backtester for trading strategies"""
    
    def __init__(self, config: Dict):
        """
        Initialize backtester
        
        Args:
            config: Backtester configuration
        """
        self.config = config
        self.symbol = config.get("symbol", "BTC/USDT")
        self.timeframe = config.get("timeframe", "1h")
        self.initial_balance = config.get("initial_balance", 10000)
        self.commission = config.get("commission", 0.1)  # In percentage
        self.slippage = config.get("slippage", 0.05)  # In percentage
        
        # Get strategy configuration
        self.strategy_config = config.get("strategy", {})
        self.strategy_name = config.get("strategy_name", "auto")
        
        # Initialize position manager for backtest
        self.position_manager = PositionManager("backtest_positions.json")
        
        # Initialize result collectors
        self.trades = []
        self.equity_curve = []
        
        # Debugging and metadata
        self.debug_info = []
        self.metadata = {}
    
    def run(self, data: pd.DataFrame, higher_tf_data: pd.DataFrame = None) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: Historical OHLCV data
            higher_tf_data: Higher timeframe data for multi-timeframe strategies
            
        Returns:
            BacktestResult: Backtest results
        """
        if data.empty:
            logger.error("Cannot run backtest on empty dataset")
            return BacktestResult([], pd.DataFrame(), self.initial_balance, self.config)
        
        # Reset state
        self.trades = []
        self.equity_curve = []
        self.debug_info = []
        self.metadata = {
            "start_date": data.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "duration_days": (data.index[-1] - data.index[0]).days,
            "n_bars": len(data),
            "timeframe": self.timeframe,
            "symbol": self.symbol
        }
        
        # Initialize account balance
        balance = self.initial_balance
        
        # Prepare equity curve dataframe
        equity_data = []
        
        # Track current position
        current_position = None
        position_side = None
        
        # Track consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        # Track ongoing trades
        open_trades = []  # List of trades that haven't been closed yet
        
        # Process bars one by one
        logger.info(f"Starting backtest with {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        for i in range(100, len(data)):  # Skip the first few bars to have enough data for indicators
            current_bar = data.iloc[:i+1]
            current_timestamp = current_bar.index[-1]
            
            # Initialize record for this bar
            equity_record = {
                "timestamp": current_timestamp,
                "balance": balance,
                "open_position": position_side is not None,
                "unrealized_pnl": 0
            }
            
            # Get current prices
            current_price = current_bar["close"].iloc[-1]
            
            # Update any open trades with current price
            for trade in open_trades:
                entry_price = trade["entry_price"]
                amount = trade["amount"]
                side = trade["side"]
                
                if side == "long":
                    unrealized_pnl = (current_price - entry_price) * amount
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * amount
                
                trade["current_price"] = current_price
                trade["unrealized_pnl"] = unrealized_pnl
                
                # Update equity record
                equity_record["unrealized_pnl"] += unrealized_pnl
                
                # Check for stop loss or take profit
                if self._check_exit_conditions(trade, current_bar, higher_tf_data):
                    # Exit the position
                    exit_slippage = self._calculate_slippage(current_price, side, False)
                    exit_price = current_price * (1 - exit_slippage if side == "long" else 1 + exit_slippage)
                    
                    # Calculate PnL
                    if side == "long":
                        pnl = (exit_price - entry_price) * amount
                        pnl_pct = (exit_price / entry_price - 1) * 100
                    else:  # short
                        pnl = (entry_price - exit_price) * amount
                        pnl_pct = (1 - exit_price / entry_price) * 100
                    
                    # Apply commission
                    commission_amount = (exit_price * amount) * (self.commission / 100)
                    pnl -= commission_amount
                    
                    # Update balance
                    balance += pnl
                    
                    # Record the trade
                    trade["exit_timestamp"] = int(current_timestamp.timestamp() * 1000)
                    trade["exit_price"] = exit_price
                    trade["pnl"] = pnl
                    trade["pnl_pct"] = pnl_pct
                    trade["commission"] = commission_amount
                    trade["status"] = "closed"
                    trade["exit_reason"] = "signal" if trade.get("exit_reason") is None else trade.get("exit_reason")
                    
                    # Track consecutive wins/losses
                    if pnl > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                    elif pnl < 0:
                        consecutive_wins = 0
                        consecutive_losses += 1
                    
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    # Add to completed trades
                    self.trades.append(trade.copy())
                    
                    # Mark for removal from open trades
                    trade["status"] = "closed"
                    
                    # Log the exit
                    logger.debug(f"Exited {side} position at {exit_price:.2f} with PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Remove closed trades from open_trades
            open_trades = [t for t in open_trades if t["status"] == "open"]
            
            # Update position side based on open trades
            if len(open_trades) > 0:
                position_side = open_trades[0]["side"]  # Assuming all open trades have same side
            else:
                position_side = None
            
            # Check for entry signal if no position
            if position_side is None:
                signal_strength, signal_side, params = self._calculate_signal(current_bar, higher_tf_data)
                
                if signal_strength > 0.5 and signal_side:  # Threshold for entry
                    # Calculate position size
                    position_size = self._calculate_position_size(balance, current_price, params)
                    
                    # Apply slippage to entry
                    entry_slippage = self._calculate_slippage(current_price, signal_side, True)
                    entry_price = current_price * (1 + entry_slippage if signal_side == "long" else 1 - entry_slippage)
                    
                    # Calculate amount after commission
                    amount = position_size / entry_price
                    commission_amount = (entry_price * amount) * (self.commission / 100)
                    
                    # Initialize trade record
                    trade = {
                        "symbol": self.symbol,
                        "side": signal_side,
                        "entry_timestamp": int(current_timestamp.timestamp() * 1000),
                        "entry_price": entry_price,
                        "amount": amount,
                        "size_usd": position_size,
                        "commission": commission_amount,
                        "status": "open",
                        "params": params,
                        "signal_strength": signal_strength,
                        "strategy": self.strategy_name
                    }
                    
                    # Add to open trades
                    open_trades.append(trade)
                    
                    # Update position side
                    position_side = signal_side
                    
                    # Log the entry
                    logger.debug(f"Entered {signal_side} position at {entry_price:.2f} with size: {amount:.6f} ({position_size:.2f} USD)")
            
            # Add equity record for this bar
            equity_data.append(equity_record)
        
        # Close any remaining open trades at the last price
        final_price = data["close"].iloc[-1]
        for trade in open_trades:
            entry_price = trade["entry_price"]
            amount = trade["amount"]
            side = trade["side"]
            
            # Calculate PnL
            if side == "long":
                pnl = (final_price - entry_price) * amount
                pnl_pct = (final_price / entry_price - 1) * 100
            else:  # short
                pnl = (entry_price - final_price) * amount
                pnl_pct = (1 - final_price / entry_price) * 100
            
            # Apply commission
            commission_amount = (final_price * amount) * (self.commission / 100)
            pnl -= commission_amount
            
            # Update balance
            balance += pnl
            
            # Record the trade
            trade["exit_timestamp"] = int(data.index[-1].timestamp() * 1000)
            trade["exit_price"] = final_price
            trade["pnl"] = pnl
            trade["pnl_pct"] = pnl_pct
            trade["commission"] = commission_amount
            trade["status"] = "closed"
            trade["exit_reason"] = "end_of_data"
            
            # Add to completed trades
            self.trades.append(trade.copy())
            
            # Log the forced exit
            logger.debug(f"Forced exit of {side} position at {final_price:.2f} with PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_data)
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)
            equity_df["equity"] = equity_df["balance"] + equity_df["unrealized_pnl"]
            
            # Calculate returns
            equity_df["return"] = equity_df["equity"].pct_change()
            equity_df["return_pct"] = equity_df["return"] * 100
        
        # Record metadata
        self.metadata.update({
            "final_balance": balance,
            "total_return": balance - self.initial_balance,
            "total_return_pct": (balance / self.initial_balance - 1) * 100,
            "n_trades": len(self.trades),
            "commission_paid": sum(t.get("commission", 0) for t in self.trades),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses
        })
        
        # Log summary
        logger.info(f"Backtest completed with {len(self.trades)} trades")
        logger.info(f"Final balance: ${balance:.2f} (Return: {self.metadata['total_return_pct']:.2f}%)")
        
        # Return backtest result
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_df,
            initial_balance=self.initial_balance,
            settings=self.config,
            metadata=self.metadata
        )
    
    def _calculate_signal(self, data: pd.DataFrame, higher_tf_data: pd.DataFrame = None) -> Tuple[float, str, Dict]:
        """
        Calculate trading signal based on selected strategy
        
        Args:
            data: Current data
            higher_tf_data: Higher timeframe data
            
        Returns:
            Tuple[float, str, Dict]: Signal strength, direction, and parameters
        """
        if self.strategy_name == "auto":
            # Evaluate all strategies and return the best signal
            from strategies import evaluate_strategies
            
            try:
                result = evaluate_strategies(data, higher_tf_data, self.strategy_config)
                return result["signal"], result["direction"], result["parameters"]
            except Exception as e:
                logger.error(f"Error evaluating strategies: {e}")
                return 0.0, "", {}
                
        elif self.strategy_name == "ehlers_supertrend":
            # Use Ehlers Supertrend strategy
            try:
                signal_strength, direction, parameters = calculate_ehlers_supertrend_signal(
                    data, self.strategy_config.get("ehlers_supertrend", {})
                )
                return signal_strength, direction, parameters
            except Exception as e:
                logger.error(f"Error calculating Ehlers Supertrend signal: {e}")
                return 0.0, "", {}
                
        elif self.strategy_name == "momentum_divergence":
            # Use Momentum Divergence strategy
            try:
                signal_strength, direction, parameters = calculate_momentum_divergence_strategy(
                    data, self.strategy_config.get("momentum_divergence", {})
                )
                return signal_strength, direction, parameters
            except Exception as e:
                logger.error(f"Error calculating Momentum Divergence signal: {e}")
                return 0.0, "", {}
                
        elif self.strategy_name == "multi_timeframe_trend":
            # Use Multi-Timeframe Trend strategy
            if higher_tf_data is None:
                logger.error("Higher timeframe data required for multi_timeframe_trend strategy")
                return 0.0, "", {}
                
            try:
                signal_strength, direction, parameters = calculate_multi_timeframe_trend_strategy(
                    data, higher_tf_data, self.strategy_config.get("multi_timeframe_trend", {})
                )
                return signal_strength, direction, parameters
            except Exception as e:
                logger.error(f"Error calculating Multi-Timeframe Trend signal: {e}")
                return 0.0, "", {}
                
        elif self.strategy_name == "support_resistance_breakout":
            # Use Support/Resistance Breakout strategy
            try:
                signal_strength, direction, parameters = calculate_support_resistance_breakout_strategy(
                    data, self.strategy_config.get("support_resistance_breakout", {})
                )
                return signal_strength, direction, parameters
            except Exception as e:
                logger.error(f"Error calculating Support/Resistance Breakout signal: {e}")
                return 0.0, "", {}
        
        # Default: no signal
        logger.warning(f"Unknown strategy: {self.strategy_name}")
        return 0.0, "", {}
    
    def _check_exit_conditions(self, trade: Dict, data: pd.DataFrame, higher_tf_data: pd.DataFrame = None) -> bool:
        """
        Check if trade should be exited
        
        Args:
            trade: Trade to check
            data: Current data
            higher_tf_data: Higher timeframe data
            
        Returns:
            bool: True if trade should be exited
        """
        side = trade["side"]
        entry_price = trade["entry_price"]
        current_price = data["close"].iloc[-1]
        
        # Check stop loss
        stop_loss = trade.get("params", {}).get("stop_loss")
        if stop_loss:
            if side == "long" and current_price <= stop_loss:
                trade["exit_reason"] = "stop_loss"
                return True
            elif side == "short" and current_price >= stop_loss:
                trade["exit_reason"] = "stop_loss"
                return True
        
        # Check take profit
        take_profit = trade.get("params", {}).get("take_profit")
        if take_profit:
            if side == "long" and current_price >= take_profit:
                trade["exit_reason"] = "take_profit"
                return True
            elif side == "short" and current_price <= take_profit:
                trade["exit_reason"] = "take_profit"
                return True
        
        # Check trailing stop
        trailing_stop = trade.get("params", {}).get("trailing_stop", {})
        if trailing_stop and trailing_stop.get("enabled", False):
            activation_pct = trailing_stop.get("activation_pct", 1.0)
            trail_pct = trailing_stop.get("trail_pct", 0.5)
            
            # Get current trailing stop price
            current_trailing_stop = trade.get("trailing_stop_price")
            
            if current_trailing_stop is None:
                # Calculate activation threshold
                if side == "long":
                    activation_threshold = entry_price * (1 + activation_pct/100)
                    if current_price >= activation_threshold:
                        # Initialize trailing stop
                        current_trailing_stop = current_price * (1 - trail_pct/100)
                        trade["trailing_stop_price"] = current_trailing_stop
                else:  # short
                    activation_threshold = entry_price * (1 - activation_pct/100)
                    if current_price <= activation_threshold:
                        # Initialize trailing stop
                        current_trailing_stop = current_price * (1 + trail_pct/100)
                        trade["trailing_stop_price"] = current_trailing_stop
            else:
                # Update trailing stop
                if side == "long":
                    new_trailing_stop = current_price * (1 - trail_pct/100)
                    if new_trailing_stop > current_trailing_stop:
                        trade["trailing_stop_price"] = new_trailing_stop
                else:  # short
                    new_trailing_stop = current_price * (1 + trail_pct/100)
                    if new_trailing_stop < current_trailing_stop:
                        trade["trailing_stop_price"] = new_trailing_stop
            
            # Check if price hit trailing stop
            if current_trailing_stop is not None:
                if side == "long" and current_price <= current_trailing_stop:
                    trade["exit_reason"] = "trailing_stop"
                    return True
                elif side == "short" and current_price >= current_trailing_stop:
                    trade["exit_reason"] = "trailing_stop"
                    return True
        
        # Check time-based exit
        max_duration_hours = trade.get("params", {}).get("time_limit")
        if max_duration_hours:
            entry_timestamp = trade.get("entry_timestamp", 0)
            current_timestamp = int(data.index[-1].timestamp() * 1000)
            
            duration_hours = (current_timestamp - entry_timestamp) / (1000 * 60 * 60)
            if duration_hours >= max_duration_hours:
                trade["exit_reason"] = "time_limit"
                return True
        
        # Check strategy-specific exit signal
        if "ehlers_supertrend" in self.strategy_name:
            try:
                from ehlers_supertrend_strategy import calculate_ehlers_supertrend_exit_signal
                
                exit_signal = calculate_ehlers_supertrend_exit_signal(
                    data, 
                    trade, 
                    self.strategy_config.get("ehlers_supertrend", {})
                )
                
                if exit_signal:
                    trade["exit_reason"] = "signal"
                    return True
                    
            except Exception as e:
                logger.error(f"Error calculating Ehlers Supertrend exit signal: {e}")
        
        # Default: no exit
        return False
    
    def _calculate_position_size(self, balance: float, price: float, params: Dict) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            balance: Current account balance
            price: Current price
            params: Signal parameters
            
        Returns:
            float: Position size in quote currency
        """
        # Get risk percentage
        risk_pct = params.get("risk_per_trade_pct", 1.0)
        
        # Get stop loss distance
        stop_loss = params.get("stop_loss")
        
        if stop_loss and price > 0:
            # Calculate position size based on risk amount
            risk_amount = balance * (risk_pct / 100)
            
            side = params.get("side", "")
            if side == "long":
                stop_distance = price - stop_loss
            else:  # short
                stop_distance = stop_loss - price
            
            if stop_distance > 0:
                # Calculate position size in quote currency
                position_size = risk_amount * price / stop_distance
            else:
                # Fallback to percentage of balance
                position_size = balance * (risk_pct / 100)
        else:
            # No stop loss, use fixed percentage
            position_size = balance * (risk_pct / 100)
        
        # Apply constraints
        min_trade_size = self.config.get("min_trade_size", 10)
        max_trade_size = self.config.get("max_trade_size", balance * 0.5)
        
        if position_size < min_trade_size:
            position_size = min_trade_size
        
        if position_size > max_trade_size:
            position_size = max_trade_size
        
        # Check if sufficient balance
        if position_size > balance:
            position_size = balance * 0.95  # Use most of balance but keep some buffer
        
        return position_size
    
    def _calculate_slippage(self, price: float, side: str, is_entry: bool) -> float:
        """
        Calculate slippage based on market conditions
        
        Args:
            price: Current price
            side: Trade side ('long' or 'short')
            is_entry: Whether this is an entry (True) or exit (False)
            
        Returns:
            float: Slippage as a percentage
        """
        # Base slippage
        base_slippage = self.slippage / 100  # Convert to decimal
        
        # Add randomness to simulate market conditions
        # In reality, this would depend on factors like volatility, liquidity, etc.
        random_factor = np.random.normal(1.0, 0.3)  # Mean 1.0, std 0.3
        slippage = base_slippage * max(0.2, min(2.0, random_factor))  # Bound between 0.2x and 2x base
        
        return slippage