// src/utils/indicators.js
const { RSI, StochasticRSI, EMA, ATR, SMA } = require('technicalindicators');

/**
 * Calculates various technical indicators based on OHLCV data.
 * @param {Array<object>} ohlcv - Array of OHLCV objects { timestamp, open, high, low, close, volume }.
 * @param {object} config - Configuration object containing indicator periods.
 * @returns {object} Object containing calculated indicator values (or null if calculation failed).
 */
const calculateIndicators = (ohlcv, config) => {
    // Ensure we have data and enough for the longest period required by indicators
    const requiredLength = Math.max(
        config.indicatorPeriod || 0,
        config.atrPeriod || 0,
        config.ehlersMaPeriod || 0,
        config.stochRsiLength || 0,
        config.stochRsiStochLength || 0,
        50 // Add a general buffer
    );

    if (!ohlcv || ohlcv.length < requiredLength) {
        console.warn(`[Indicators] Insufficient data. Need at least ${requiredLength}, got ${ohlcv?.length || 0}`);
        return {}; // Return empty object if not enough data
    }

    // Extract necessary price arrays
    const closes = ohlcv.map(k => k.close);
    const highs = ohlcv.map(k => k.high);
    const lows = ohlcv.map(k => k.low);

    let indicators = {};

    // --- Calculate Indicators ---
    // Wrap each calculation in try-catch for robustness

    // RSI
    try {
        const rsiResult = RSI.calculate({ values: closes, period: config.indicatorPeriod });
        indicators.rsi = rsiResult.length ? rsiResult[rsiResult.length - 1] : null;
    } catch (e) {
        console.error("[Indicators] Error calculating RSI:", e.message);
        indicators.rsi = null;
    }

    // Stochastic RSI
    try {
        // Step 1: Calculate underlying RSI values needed for StochRSI input
        const rsiValuesForStoch = RSI.calculate({ values: closes, period: config.stochRsiLength });

        // Ensure enough RSI values for the stochastic calculation part
        if (rsiValuesForStoch.length >= config.stochRsiStochLength) {
            const stochRsiInput = {
                values: rsiValuesForStoch,
                rsiPeriod: config.stochRsiLength,       // Period used to calculate the input RSI values
                stochasticPeriod: config.stochRsiStochLength, // Period for the stochastic calculation on RSI
                kPeriod: config.stochRsiK,              // %K period for stochastic
                dPeriod: config.stochRsiD,              // %D period (smoothing of %K)
            };
            const stochRsiResult = StochasticRSI.calculate(stochRsiInput);
            // Get the last calculated { k, d } object
            indicators.stochRsi = stochRsiResult.length ? stochRsiResult[stochRsiResult.length - 1] : null;
            indicators.fullStochRsi = stochRsiResult; // Store full series if needed elsewhere
        } else {
            console.warn(`[Indicators] Not enough RSI values (${rsiValuesForStoch.length}) for StochRSI stochastic period (${config.stochRsiStochLength}).`);
            indicators.stochRsi = null;
            indicators.fullStochRsi = [];
        }
    } catch (e) {
        console.error("[Indicators] Error calculating StochRSI:", e.message);
        indicators.stochRsi = null;
        indicators.fullStochRsi = [];
    }

    // Ehlers MA (using EMA as a substitute)
    try {
        // Note: A true Ehlers MA might be more complex (e.g., Ehlers Fisher Transform, Instantaneous Trendline).
        // Using EMA for simplicity based on the config name.
        const emaResult = EMA.calculate({ values: closes, period: config.ehlersMaPeriod });
        indicators.ehlersMa = emaResult.length ? emaResult[emaResult.length - 1] : null;
        indicators.fullEhlersMa = emaResult; // Store full series
    } catch (e) {
        console.error("[Indicators] Error calculating Ehlers MA (using EMA):", e.message);
        indicators.ehlersMa = null;
        indicators.fullEhlersMa = [];
    }

    // ATR (Average True Range)
    try {
        const atrInput = { high: highs, low: lows, close: closes, period: config.atrPeriod };
        const atrResult = ATR.calculate(atrInput);
        indicators.atr = atrResult.length ? atrResult[atrResult.length - 1] : null;
    } catch (e) {
        console.error("[Indicators] Error calculating ATR:", e.message);
        indicators.atr = null;
    }

    // Optionally include raw closes if needed by the strategy
    indicators.closes = closes;

    // console.debug("[Indicators] Calculated values:", indicators); // Use debug level for verbose logs
    return indicators;
};

module.exports = { calculateIndicators };
