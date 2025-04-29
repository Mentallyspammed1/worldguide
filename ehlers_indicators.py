"""
Ehlers Indicators Module

This module implements various John Ehlers' indicators:
- Super Smoother Filter
- Roofing Filter
- Fisher Transform
- Center of Gravity Oscillator
- Cyber Cycle
- Signal-to-Noise Ratio
- Autocorrelation Periodogram

These indicators provide enhanced market analysis for detecting:
- Trends vs ranging markets
- Cycle identification
- Turning points
- Market types
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

import logging
logger = logging.getLogger("trading_bot.indicators.ehlers")

def compute_super_smoother_filter(series: pd.Series, length: int = 10) -> pd.Series:
    """
    Compute Super Smoother Filter (SSF)
    
    This is Ehlers' low-pass filter with minimal lag and no overshoot.
    
    Args:
        series: Input price data
        length: Filter length
        
    Returns:
        pd.Series: Filtered price series
    """
    # Get series values as numpy array
    price = series.values
    
    # Pre-allocate result array
    ssf = np.zeros_like(price, dtype=np.float64)
    
    # Calculate filter constants
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    # Initial values
    ssf[0] = price[0]
    ssf[1] = price[1]
    
    # Apply filter
    for i in range(2, len(price)):
        ssf[i] = c1 * (price[i] + price[i-1]) / 2 + c2 * ssf[i-1] + c3 * ssf[i-2]
    
    # Convert back to pandas Series
    return pd.Series(ssf, index=series.index)


def compute_roofing_filter(series: pd.Series, hp_length: int = 48, ss_length: int = 10) -> pd.Series:
    """
    Compute Roofing Filter
    
    This is a bandpass filter that combines a high-pass and super smoother filter.
    It helps identify cycles by removing both long-term trend and short-term noise.
    
    Args:
        series: Input price data
        hp_length: High-pass filter length
        ss_length: Super smoother filter length
        
    Returns:
        pd.Series: Roofing filter values
    """
    # Get series values as numpy array
    price = series.values
    
    # Pre-allocate result arrays
    hp = np.zeros_like(price, dtype=np.float64)
    roof = np.zeros_like(price, dtype=np.float64)
    
    # Calculate high-pass filter constant
    alpha1 = (np.cos(0.707 * 2 * np.pi / hp_length) + np.sin(0.707 * 2 * np.pi / hp_length) - 1) / np.cos(0.707 * 2 * np.pi / hp_length)
    
    # Calculate super smoother constants
    a1 = np.exp(-1.414 * np.pi / ss_length)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / ss_length)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    # Initial values
    hp[0] = price[0]
    hp[1] = price[1]
    roof[0] = 0
    roof[1] = 0
    
    # Apply high-pass filter
    for i in range(2, len(price)):
        hp[i] = (1 - alpha1/2)**2 * (price[i] - 2*price[i-1] + price[i-2]) + 2*(1-alpha1)*hp[i-1] - (1-alpha1)**2*hp[i-2]
    
    # Apply super smoother filter to high-pass
    for i in range(2, len(price)):
        roof[i] = c1 * (hp[i] + hp[i-1]) / 2 + c2 * roof[i-1] + c3 * roof[i-2]
    
    # Convert back to pandas Series
    return pd.Series(roof, index=series.index)


def compute_fisher_transform(series: pd.Series, length: int = 10) -> pd.Series:
    """
    Compute Fisher Transform
    
    The Fisher Transform converts price data to a more normalized distribution,
    making turning points and trends easier to identify.
    
    Args:
        series: Input price data
        length: Lookback length
        
    Returns:
        pd.Series: Fisher transform values
    """
    # Get series values
    price = series.values
    
    # Pre-allocate result arrays
    value1 = np.zeros_like(price, dtype=np.float64)
    value2 = np.zeros_like(price, dtype=np.float64)
    fisher = np.zeros_like(price, dtype=np.float64)
    
    # Calculate Fisher Transform
    for i in range(length, len(price)):
        # Get price slice for normalization
        temp_slice = price[i-length+1:i+1]
        
        # Find min/max for period
        max_val = np.max(temp_slice)
        min_val = np.min(temp_slice)
        
        # Avoid division by zero
        denom = max_val - min_val
        if denom == 0:
            value1[i] = 0
        else:
            # Normalize to -1 to +1 range
            value1[i] = 0.66 * ((price[i] - min_val) / denom - 0.5) + 0.67 * value1[i-1]
            
            # Bound value1
            value1[i] = max(-0.99, min(0.99, value1[i]))
        
        # Apply Fisher Transform
        value2[i] = 0.5 * np.log((1 + value1[i]) / (1 - value1[i])) + 0.5 * value2[i-1]
    
    # Smooth the Fisher Transform
    for i in range(1, len(price)):
        fisher[i] = 0.5 * value2[i] + 0.5 * fisher[i-1]
    
    # Convert back to pandas Series
    return pd.Series(fisher, index=series.index)


def compute_center_of_gravity(series: pd.Series, length: int = 10) -> pd.Series:
    """
    Compute Center of Gravity (CG) Oscillator
    
    The CG Oscillator measures the center of gravity of price over time,
    providing a leading indicator for trend reversals.
    
    Args:
        series: Input price data
        length: Lookback length
        
    Returns:
        pd.Series: CG Oscillator values
    """
    # Get series values
    price = series.values
    
    # Pre-allocate result array
    cg = np.zeros_like(price, dtype=np.float64)
    
    # Calculate CG Oscillator
    for i in range(length, len(price)):
        # Initialize numerator and denominator
        num = 0
        denom = 0
        
        # Calculate weighted sum
        for j in range(length):
            num += (j+1) * price[i-j]
            denom += price[i-j]
        
        # Avoid division by zero
        if denom != 0:
            cg[i] = -num / denom + (length + 1) / 2
    
    # Convert back to pandas Series
    return pd.Series(cg, index=series.index)


def compute_cyber_cycle(series: pd.Series, length: int = 20, alpha: float = 0.07) -> pd.Series:
    """
    Compute Cyber Cycle indicator
    
    The Cyber Cycle extracts the dominant cycle from price data,
    filtering out both trend and noise.
    
    Args:
        series: Input price data
        length: Cycle length
        alpha: Smoothing factor
        
    Returns:
        pd.Series: Cyber Cycle values
    """
    # Get series values
    price = series.values
    
    # Pre-allocate result arrays
    smooth = np.zeros_like(price, dtype=np.float64)
    cycle = np.zeros_like(price, dtype=np.float64)
    
    # Calculate alpha from length if not provided
    if alpha <= 0:
        alpha = 0.07
    
    # Initialize
    smooth[0] = price[0]
    cycle[0] = 0
    
    # Calculate Cyber Cycle
    for i in range(1, len(price)):
        # Smooth price
        smooth[i] = (1 - alpha) * price[i] + alpha * smooth[i-1]
        
        # Calculate cycle
        if i >= 4:
            cycle[i] = (price[i] - 2*price[i-1] + price[i-2]) / 4 + (4*cycle[i-1] - 6*cycle[i-2] + 4*cycle[i-3] - cycle[i-4]) / 3
    
    # Convert back to pandas Series
    return pd.Series(cycle, index=series.index)


def compute_autocorrelation_periodogram(series: pd.Series, max_period: int = 50) -> List[int]:
    """
    Compute Autocorrelation Periodogram
    
    This function identifies the dominant cycles in the price data using
    autocorrelation techniques.
    
    Args:
        series: Input price data
        max_period: Maximum period to analyze
        
    Returns:
        List[int]: Dominant cycle periods
    """
    # Get series values and ensure it's demeaned
    price = series.values
    price = price - np.mean(price)
    
    # Pre-allocate autocorrelation array
    corr = np.zeros(max_period)
    
    # Calculate autocorrelation
    for lag in range(1, max_period):
        if lag >= len(price) - 1:
            break
            
        # Standard autocorrelation formula
        numerator = np.sum(price[lag:] * price[:-lag])
        denominator = np.sqrt(np.sum(price[lag:]**2) * np.sum(price[:-lag]**2))
        
        if denominator > 0:
            corr[lag] = numerator / denominator
    
    # Find peaks in autocorrelation
    peaks = []
    
    for i in range(3, len(corr) - 3):
        # Check if current point is higher than three points on either side
        if corr[i] > 0.1 and all(corr[i] > corr[i+j] for j in [-3, -2, -1, 1, 2, 3]):
            peaks.append(i)
    
    # Return peaks in ascending order of significance (height)
    return sorted(peaks)


def compute_signal_to_noise_ratio(series: pd.Series, length: int = 20) -> pd.Series:
    """
    Compute Signal-to-Noise Ratio (SNR)
    
    This indicator measures the strength of the trend relative to noise,
    helping distinguish between trending and non-trending markets.
    
    Args:
        series: Input price data
        length: Lookback length
        
    Returns:
        pd.Series: SNR values
    """
    # Calculate the linear regression slope over the lookback period
    def rolling_slope(x):
        y = np.arange(len(x))
        slope, _, _, _, _ = np.polyfit(y, x, 1, full=True)
        return slope[0]
    
    # Get price changes
    price_changes = series.diff().fillna(0)
    
    # Calculate rolling variance (noise)
    variance = price_changes.rolling(window=length).var()
    
    # Calculate rolling slope (signal)
    slopes = series.rolling(window=length).apply(rolling_slope, raw=True)
    
    # Ensure we don't divide by zero
    variance = np.where(variance == 0, 1e-9, variance)
    
    # Calculate SNR as |slope| / sqrt(variance)
    snr = np.abs(slopes) / np.sqrt(variance)
    
    # Convert to pandas Series
    return pd.Series(snr, index=series.index)


def detect_market_type(
    fisher: float,
    cg_osc: float,
    snr: float,
    snr_threshold: float = 2.5
) -> Tuple[str, float]:
    """
    Detect market type based on various indicators
    
    Args:
        fisher: Fisher Transform value
        cg_osc: Center of Gravity Oscillator value
        snr: Signal-to-Noise Ratio
        snr_threshold: SNR threshold for trending markets
        
    Returns:
        Tuple[str, float]: Market type and confidence score
    """
    # Default market type
    market_type = "ranging"
    confidence = 0.5
    
    # Check for trending market
    if snr > snr_threshold:
        market_type = "trending"
        confidence = min(0.9, 0.5 + (snr - snr_threshold) / 5.0)
        
        # Direction of trend
        if fisher > 0.5:
            direction = "up"
        elif fisher < -0.5:
            direction = "down"
        else:
            direction = "neutral"
        
        # Check if trend is accelerating or decelerating
        if abs(cg_osc) > 0.3:
            phase = "accelerating" if cg_osc * fisher > 0 else "decelerating"
        else:
            phase = "steady"
        
        # No need to include direction and phase in return values
        # But useful for debugging
        logger.debug(f"Trending market detected: {direction} ({phase}), SNR: {snr:.2f}")
        
    # Check for volatile market (low SNR but high price movement)
    elif abs(fisher) > 1.5 and snr < 1.5:
        market_type = "volatile"
        confidence = min(0.8, 0.5 + abs(fisher) / 5.0)
        logger.debug(f"Volatile market detected, Fisher: {fisher:.2f}, SNR: {snr:.2f}")
        
    # Ranging market (default)
    else:
        # More confident about ranging if SNR is very low
        confidence = max(0.5, 1.0 - snr / snr_threshold)
        logger.debug(f"Ranging market detected, SNR: {snr:.2f}")
    
    return market_type, confidence