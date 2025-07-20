// archon_engine.js
// The Grand Ritual of Vector & Gemini: The Archon Edition
// A masterfully crafted spell for high-efficiency backtesting and analysis.

import { promises as fs } from 'fs';
import path from 'path';
import { parse } from 'csv-parse';
import { stringify } from 'csv-stringify';
import dotenv from 'dotenv';
import chalk from 'chalk';
let GoogleGenerativeAI;
let _GEMINI_AVAILABLE = false;
try {
    const geminiModule = await import('@google/generative-ai');
    GoogleGenerativeAI = geminiModule.GoogleGenerativeAI;
    _GEMINI_AVAILABLE = true;
} catch (e) {
    console.warn(chalk.yellow("Warning: @google/generative-ai not found. Gemini summary will be unavailable."));
}

// Load environment variables from .env file
dotenv.config();

// --- The Scroll of Strategy (Centralized Configuration) ---
const STRATEGY_CONFIG = {
    rsi_period: 14, adx_period: 14, aroon_period: 14,
    bb_period: 20, bb_std: 2.0,
    supertrend_period: 10, supertrend_multiplier: 3.0,
    z_score_period: 20,
    thresholds: {
        adx_strong_trend: 25, aroon_strong_momentum: 50,
        rsi_overbought: 70, rsi_oversold: 30,
        z_score_overextended: 2.0,
        signal_confidence: 0.4
    }
};

// --- The Chromatic Palette of the Spell ---
const COLOR_HEADER = chalk.cyan.bold;
const COLOR_SIGNAL_LONG = chalk.green.bold;
const COLOR_SIGNAL_SHORT = chalk.red.bold;
const COLOR_SIGNAL_HOLD = chalk.yellow;
const COLOR_CONFIDENCE = chalk.magenta.bold;
const COLOR_REASON_POS = chalk.green;
const COLOR_REASON_NEG = chalk.red;
const COLOR_STATUS = chalk.blue.bold;
const COLOR_ORACLE = chalk.hex('#FFD700'); // Gold
const COLOR_ERROR = chalk.red.bold;

// Helper function to calculate Average True Range (ATR)
function calculateATR(df, period) {
    console.log(`ATR Debug: df.length = ${df.length}, period = ${period}`);
    const trs = [];
    for (let i = 1; i < df.length; i++) {
        const highLow = df[i].high - df[i].low;
        const highPrevClose = Math.abs(df[i].high - df[i - 1].close);
        const lowPrevClose = Math.abs(df[i].low - df[i - 1].close);
        const tr = Math.max(highLow, highPrevClose, lowPrevClose);
        trs.push(tr);
    }
    console.log("ATR Debug: TRs array length:", trs.length);
    console.log("ATR Debug: TRs array:", trs.map(t => t.toFixed(4)));

    const atrs = [];
    if (trs.length > 0) {
        // Initial ATR is the average of the first 'period' TRs
        let sumTr = 0;
        for (let i = 0; i < period && i < trs.length; i++) {
            sumTr += trs[i];
        }
        const initialAtr = sumTr / period;
        atrs.push(initialAtr);
        console.log(`ATR Debug: Initial ATR (${period} period): ${initialAtr.toFixed(4)}`);

        // Subsequent ATRs
        for (let i = period; i < trs.length; i++) {
            const prevAtr = atrs[atrs.length - 1];
            const newAtr = ((prevAtr * (period - 1)) + trs[i]) / period;
            atrs.push(newAtr);
        }
    }
    console.log("ATR Debug: Final ATRs array length:", atrs.length);
    console.log("ATR Debug: Final ATRs array:", atrs.map(a => a.toFixed(4)));
    return atrs;
}

// Helper function to calculate Supertrend
function calculateSupertrend(df, period, multiplier) {
    const atrs = calculateATR(df, period);
    const supertrend = [];
    const supertrendDirection = []; // 1 for up, -1 for down

    for (let i = 0; i < df.length; i++) {
        if (i < period) {
            supertrend.push(NaN);
            supertrendDirection.push(NaN);
            continue;
        }

        const currentAtr = atrs[i - period]; // ATR is calculated from previous data
        if (isNaN(currentAtr)) {
            supertrend.push(NaN);
            supertrendDirection.push(NaN);
            continue;
        }

        const basicUpperBand = ((df[i].high + df[i].low) / 2) + (multiplier * currentAtr);
        const basicLowerBand = ((df[i].high + df[i].low) / 2) - (multiplier * currentAtr);

        let finalUpperBand = basicUpperBand;
        let finalLowerBand = basicLowerBand;

        if (i > period) {
            // Adjust bands based on previous close and band values
            const prevSupertrendDirection = supertrendDirection[i - 1];
            const prevSupertrend = supertrend[i - 1];

            if (!isNaN(prevSupertrendDirection) && !isNaN(prevSupertrend)) {
                if (prevSupertrendDirection === 1) { // Previous was uptrend
                    finalUpperBand = Math.min(basicUpperBand, prevSupertrend);
                    finalLowerBand = Math.max(basicLowerBand, prevSupertrend);
                } else if (prevSupertrendDirection === -1) { // Previous was downtrend
                    finalUpperBand = Math.min(basicUpperBand, prevSupertrend);
                    finalLowerBand = Math.max(basicLowerBand, prevSupertrend);
                }
            }
        }

        let currentSupertrend;
        let currentDirection;

        if (supertrendDirection[i - 1] === 1) { // Previous was uptrend
            if (df[i].close > finalLowerBand) {
                currentSupertrend = finalLowerBand;
                currentDirection = 1;
            } else {
                currentSupertrend = finalUpperBand;
                currentDirection = -1;
            }
        } else if (supertrendDirection[i - 1] === -1) { // Previous was downtrend
            if (df[i].close < finalUpperBand) {
                currentSupertrend = finalUpperBand;
                currentDirection = -1;
            } else {
                currentSupertrend = finalLowerBand;
                currentDirection = 1;
            }
        } else { // First valid point, or previous was NaN
            if (df[i].close > finalUpperBand) {
                currentSupertrend = finalLowerBand;
                currentDirection = 1;
            } else {
                currentSupertrend = finalUpperBand;
                currentDirection = -1;
            }
        }

        supertrend.push(currentSupertrend);
        supertrendDirection.push(currentDirection);
    }
    return { supertrend, supertrendDirection };
}

// --- DataManager: The Keeper of Scrolls and Ethers ---
class DataManager {
    constructor() {
        console.log(COLOR_HEADER("DataManager: Forged to manage data from all sources."));
    }

    async loadFromScroll(filePath) {
        try {
            console.log(chalk.white(`Unfurling the scroll: ${filePath}...`));
            const fileContent = await fs.readFile(filePath, { encoding: 'utf8' });
            const records = await new Promise((resolve, reject) => {
                parse(fileContent, {
                    columns: true,
                    skip_empty_lines: true
                }, (err, records) => {
                    if (err) reject(err);
                    else resolve(records);
                });
            });
            console.log("Type of records: ", typeof records, "Is array: ", Array.isArray(records));

            // Convert to a DataFrame-like structure (array of objects)
            // and ensure correct types and sorting
            records.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

            let df = records.map(row => ({
                timestamp: new Date(row.timestamp),
                open: parseFloat(row.open),
                high: parseFloat(row.high),
                low: parseFloat(row.low),
                close: parseFloat(row.close),
                volume: parseFloat(row.volume)
            }));
            console.log(chalk.green("Scroll successfully deciphered."));
            return df;
        } catch (e) {
            console.error(COLOR_ERROR(`A flaw was found in the scroll's inscription: ${e.message}`));
            return [];
        }
    }

    weaveIndicators(df, config) {
        console.log(chalk.white("Weaving the indicator sigils onto the data scroll..."));
        // Calculate Supertrend
        const { supertrend, supertrendDirection } = calculateSupertrend(df, config.supertrend_period, config.supertrend_multiplier);

        return df.map((row, index) => ({
            ...row,
            RSI_14: Math.random() * 100, // Dummy RSI
            ADX_14: Math.random() * 50,  // Dummy ADX
            AROONOSC_14: Math.random() * 200 - 100, // Dummy Aroon Oscillator
            BBL_20_2_0: row.close * 0.95, // Dummy Bollinger Band Lower
            BBM_20_2_0: row.close,       // Dummy Bollinger Band Middle
            BBU_20_2_0: row.close * 1.05, // Dummy Bollinger Band Upper
            SUPERT_10_3_0: supertrend[index] || NaN, // Actual Supertrend
            SUPERTd_10_3_0: supertrendDirection[index] || NaN, // Actual Supertrend Direction
            Z_SCORE: (row.close - (row.close * (1 + (Math.random() - 0.5) * 0.01))) / (row.close * 0.01) // Dummy Z-Score
        }));
    }
}

// --- GeminiOracle: The Voice of Synthetic Wisdom ---
class GeminiOracle {
    constructor(apiKey) {
        if (!_GEMINI_AVAILABLE) {
            this.model = null;
            console.error(COLOR_ERROR("Gemini Oracle: Not available due to missing dependencies."));
            return;
        }
        try {
            const genAI = new GoogleGenerativeAI(apiKey);
            this.model = genAI.getGenerativeModel({ model: "gemini-pro" });
            console.log(COLOR_ORACLE("Gemini Oracle: Consciousness awakened."));
        } catch (e) {
            this.model = null;
            console.error(COLOR_ERROR(`Failed to awaken the Gemini Oracle: ${e.message}`));
        }
    }

    async summarizeBacktest(resultsDf, symbol, config) {
        if (!this.model) return "The Oracle is silent.";
        
        const tradeSignals = resultsDf.filter(row => row.signal !== 'HOLD');
        const numLongs = tradeSignals.filter(row => row.signal === 'LONG').length;
        const numShorts = tradeSignals.filter(row => row.signal === 'SHORT').length;
        const avgConfidence = tradeSignals.length > 0 ? tradeSignals.reduce((sum, row) => sum + row.confidence, 0) / tradeSignals.length : 0;

        const prompt = `
        You are a master strategist, reviewing a completed trading simulation.
        Analyze the following backtest summary and provide a concise, expert narrative (3-4 sentences) on the strategy's overall behavior.

        Simulation Summary:
        - Asset: ${symbol}
        - Total Candles Analyzed: ${resultsDf.length}
        - Total Trade Signals Generated: ${tradeSignals.length}
        - Long Signals: ${numLongs}
        - Short Signals: ${numShorts}
        - Average Signal Confidence: ${avgConfidence.toFixed(2)}

        Strategy Parameters:
        - RSI Period: ${config.rsi_period}, Overbought: ${config.thresholds.rsi_overbought}, Oversold: ${config.thresholds.rsi_oversold}
        - ADX Period: ${config.adx_period}, Strong Trend Threshold: ${config.thresholds.adx_strong_trend}
        - Supertrend: ${config.supertrend_period}/${config.supertrend_multiplier}

        Provide your strategic overview.
        `;
        try {
            const result = await this.model.generateContent(prompt);
            const response = await result.response;
            return response.text().trim();
        } catch (e) {
            return `The Oracle's thoughts are clouded: ${e.message}`;
        }
    }
}

// --- VectorEngine: The Heart of Quantitative Analysis ---
class VectorEngine {
    constructor(config) {
        this.config = config;
        this.t = config.thresholds;
    }

    analyzeRow(primaryRow, htfDir, symbol) {
        let reasoning = [];
        let score = 0;
        let signal = "HOLD";
        
        // 1. Multi-Timeframe Context
        const htfTrend = htfDir === 1 ? "UP" : "DOWN";
        reasoning.push(`${COLOR_REASON_POS("[+]")} CONTEXT: HTF current is ${htfTrend}.`);
        if (htfDir === 1) { score = 0.2; signal = "LONG"; }
        else { score = 0.2; signal = "SHORT"; }

        // 2. Indicator Confluence & Divergence
        if (primaryRow[`ADX_${this.config.adx_period}`] > this.t.adx_strong_trend) {
            reasoning.push(`${COLOR_REASON_POS("[+]")} TREND: ADX shows a strong current.`);
            score += 0.15;
        }
        
        if (primaryRow[`AROONOSC_${this.config.aroon_period}`] > this.t.aroon_strong_momentum) {
            reasoning.push(`${COLOR_REASON_POS("[+]")} MOMENTUM: Aroon confirms bullish surge.`);
            score += 0.1;
        } else if (primaryRow[`AROONOSC_${this.config.aroon_period}`] < -this.t.aroon_strong_momentum) {
            reasoning.push(`${COLOR_REASON_POS("[+]")} MOMENTUM: Aroon confirms bearish plunge.`);
            score += 0.1;
        }

        // 3. Risk Analysis & Signal Invalidation
        if (primaryRow[`RSI_${this.config.rsi_period}`] > this.t.rsi_overbought) {
            reasoning.push(`${COLOR_REASON_NEG("[-]")} RISK: RSI is overbought.`);
            if (signal === "LONG") { signal = "HOLD"; score -= 0.3; }
        } else if (primaryRow[`RSI_${this.config.rsi_period}`] < this.t.rsi_oversold) {
            reasoning.push(`${COLOR_REASON_NEG("[-]")} RISK: RSI is oversold.`);
            if (signal === "SHORT") { signal = "HOLD"; score -= 0.3; }
        }

        if (Math.abs(primaryRow.Z_SCORE) > this.t.z_score_overextended) {
            reasoning.push(`${COLOR_REASON_NEG("[-]")} RISK: Z-Score shows extreme deviation.`);
            if ((primaryRow.Z_SCORE > 0 && signal === "LONG") || (primaryRow.Z_SCORE < 0 && signal === "SHORT")) {
                signal = "HOLD"; score -= 0.2;
            }
        }

        const stDirCol = `SUPERTd_${this.config.supertrend_period}_${this.config.supertrend_multiplier}`;
        if ((signal === "LONG" && primaryRow[stDirCol] === -1) || (signal === "SHORT" && primaryRow[stDirCol] === 1)) {
            reasoning.push(`${COLOR_REASON_NEG("[-]")} CONFLICT: Primary Supertrend opposes HTF bias.`);
            signal = "HOLD"; score -= 0.3;
        }

        // 4. Final Verdict
        const finalSignal = score >= this.t.signal_confidence ? signal : "HOLD";
        return { signal: finalSignal, confidence: Math.round(Math.max(0, score) * 100) / 100, reasoning: reasoning.map(r => r.split('] ')[1]).join(' | ') };
    }
}

// --- Chronomancer: The Master of Time Simulation ---
class Chronomancer {
    constructor(engine, verbose) {
        this.engine = engine;
        this.verbose = verbose;
        this.results = [];
    }

    async scry(dfPrimary, dfHtf, symbol) {
        console.log(COLOR_HEADER("Chronomancer begins scrying the threads of past time..."));
        
        // Align HTF data to primary timestamps for efficient lookup
        // In a real scenario, you'd need a more robust way to align timeframes
        // For this example, we'll assume dfHtf has corresponding timestamps
        for (const row of dfPrimary) {
            let htfDir = undefined;
            for (let i = dfHtf.length - 1; i >= 0; i--) {
                if (dfHtf[i].timestamp.getTime() <= row.timestamp.getTime()) {
                    htfDir = dfHtf[i][`SUPERTd_${STRATEGY_CONFIG.supertrend_period}_${STRATEGY_CONFIG.supertrend_multiplier}`];
                    break;
                }
            }
            if (htfDir === undefined) {
                continue; // Skip if no corresponding HTF data exists yet
            }

            const analysis = this.engine.analyzeRow(row, htfDir, symbol);
            this.results.push({
                timestamp: row.timestamp, close: row.close,
                signal: analysis.signal, confidence: analysis.confidence,
                reasoning: analysis.reasoning
            });
            if (this.verbose || analysis.signal !== 'HOLD') {
                this.printAnalysis(this.results[this.results.length - 1]);
            }
        }
        
        console.log(COLOR_HEADER(`Scrying complete. ${this.results.length} moments in time were analyzed.`));
    }

    async saveLedger(filename) {
        if (this.results.length === 0) return;
        console.log(chalk.white(`Inscribing the Ledger of Fates to ${filename}...`));
        const csvString = await stringify(this.results.map(row => ({
            timestamp: row.timestamp.toISOString(), // Convert Date to ISO string for CSV
            close: row.close,
            signal: row.signal,
            confidence: row.confidence,
            reasoning: row.reasoning
        })), { header: true });
        await fs.writeFile(filename, csvString);
        console.log(chalk.green("The Ledger has been written."));
    }

    printAnalysis(result) {
        const signalColor = { "LONG": COLOR_SIGNAL_LONG, "SHORT": COLOR_SIGNAL_SHORT, "HOLD": COLOR_SIGNAL_HOLD }[result.signal];
        console.log(`${result.timestamp.toISOString()} | ${COLOR_HEADER("CLOSE:")} ${result.close.toFixed(4)} | ${signalColor(`SIG: ${result.signal}`)} ${COLOR_CONFIDENCE(`(${result.confidence.toFixed(2)})`)}`);
    }
}

// --- The Main Summoning Ritual ---
async function main() {
    const args = process.argv.slice(2);
    let primaryScroll = '';
    let htfScroll = '';
    let symbol = 'HISTORICAL_ASSET';
    let output = '';
    let verbose = false;
    let geminiSummary = false;

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--symbol':
                symbol = args[++i];
                break;
            case '-o':
            case '--output':
                output = args[++i];
                break;
            case '-v':
            case '--verbose':
                verbose = true;
                break;
            case '--gemini-summary':
                geminiSummary = true;
                break;
            default:
                if (!primaryScroll) {
                    primaryScroll = args[i];
                } else if (!htfScroll) {
                    htfScroll = args[i];
                }
                break;
        }
    }

    if (!primaryScroll || !htfScroll) {
        console.error(COLOR_ERROR("Usage: node archon_engine.js <primary_scroll_path> <htf_scroll_path> [--symbol <symbol>] [-o <output_file>] [-v] [--gemini-summary]"));
        return;
    }

    // --- Initialization ---
    const dm = new DataManager();
    const engine = new VectorEngine(STRATEGY_CONFIG);
    const chronomancer = new Chronomancer(engine, verbose);

    // --- Data Preparation ---
    let dfPrimary = await dm.loadFromScroll(primaryScroll);
    let dfHtf = await dm.loadFromScroll(htfScroll);
    if (dfPrimary.length === 0 || dfHtf.length === 0) {
        console.error(COLOR_ERROR("Cannot perform chronomancy without valid scrolls. The spell is broken."));
        return;
    }

    dfPrimary = dm.weaveIndicators(dfPrimary, STRATEGY_CONFIG);
    dfHtf = dm.weaveIndicators(dfHtf, STRATEGY_CONFIG);
    console.log("dfPrimary after weaveIndicators:", dfPrimary.length, dfPrimary[0]);
    console.log("dfHtf after weaveIndicators:", dfHtf.length, dfHtf[0]);

    // --- Execution ---
    // Ensure enough data for indicators to be meaningful
    const minDataPoints = Math.max(STRATEGY_CONFIG.rsi_period, STRATEGY_CONFIG.adx_period, STRATEGY_CONFIG.bb_period, STRATEGY_CONFIG.supertrend_period, STRATEGY_CONFIG.z_score_period);
    if (dfPrimary.length < minDataPoints) {
        console.error(COLOR_ERROR(`Not enough primary data points (${dfPrimary.length}) for indicator calculation. Need at least ${minDataPoints}.`));
        return;
    }
    if (dfHtf.length < STRATEGY_CONFIG.supertrend_period + 1) {
        console.error(COLOR_ERROR(`Not enough higher timeframe data points (${dfHtf.length}) for Supertrend calculation. Need at least ${STRATEGY_CONFIG.supertrend_period + 1}.`));
        return;
    }

    await chronomancer.scry(dfPrimary, dfHtf, symbol);

    if (output) {
        await chronomancer.saveLedger(output);
    }

    if (geminiSummary) {
        if (!_GEMINI_AVAILABLE) {
            console.error(COLOR_ERROR("Cannot invoke the Oracle: @google/generative-ai is not installed."));
        } else {
            const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
            if (!GEMINI_API_KEY) {
                console.error(COLOR_ERROR("Cannot invoke the Oracle: Gemini API key is missing from .env scroll."));
            } else {
                const oracle = new GeminiOracle(GEMINI_API_KEY);
                console.log(`
${COLOR_ORACLE("Invoking the Gemini Oracle for a strategic summary...")}`);
                const summary = await oracle.summarizeBacktest(chronomancer.results, symbol, STRATEGY_CONFIG);
                console.log(chalk.white(summary));
            }
        }
    }
}

if (process.argv[1] === import.meta.url.slice(7)) {
    main().catch(e => {
        console.error(COLOR_ERROR(`A chaotic energy has disrupted the ritual: ${e.message}`));
        console.error(e);
    });
}
