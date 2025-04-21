// GridBot.js - Final Enhanced Version
import ccxt from 'ccxt';
import { Mutex } from 'async-mutex';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as nc from 'nanocolors';
import dotenv from 'dotenv'; // Import dotenv

// --- Load Environment Variables ---
// Load .env file at the very beginning
dotenv.config();

// --- Import Custom Modules ---
import { config } from './config.js';
import { log, setLogLevel, initializeLogFile, closeLogStream } from './logger.js';
import { Decimal, ZERO, ONE, MIN_ORDER_AMOUNT_THRESHOLD_DEC, safeDecimal, formatDecimalString } from './decimal_utils.js';
import { calculate_manual_sma, calculate_manual_ema, calculate_manual_atr, calculate_momentum_roc } from './indicators.js';
import { loadState, saveState } from './state_manager.js';
import { sendSmsAlert } from './sms_alert.js';
import { GracefulShutdown } from './shutdown_handler.js';

// --- Constants and Setup ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- GridBot Class Definition ---
class GridBot {
    constructor() {
        // Initialize logger first
        initializeLogFile(config.log_file || path.join(__dirname, 'grid_bot.log'));
        setLogLevel(config.log_level || 'INFO');

        log('INFO', `${nc.bold(nc.magenta('=== Grid Bot Initialization ==='))}`);

        // Validate essential config/env vars
        if (!process.env.BYBIT_API_KEY || !process.env.BYBIT_API_SECRET) {
            log('CRITICAL', 'API Key or Secret not found in environment variables. Ensure BYBIT_API_KEY and BYBIT_API_SECRET are set.');
            // Throw an error to prevent starting without credentials
            throw new Error("Missing API credentials in environment variables.");
        }
        if (!config.symbol) {
            throw new Error("Missing 'symbol' in config.js");
        }
        if (!config.exchange_id) {
            throw new Error("Missing 'exchange_id' in config.js");
        }

        this.config = config;
        this.bot_running = true;
        this.exchange = null;
        this.market_info = null;
        this.initial_balance_usd = null; // Store as Decimal
        this.realized_pnl = ZERO; // Store as Decimal
        this.active_orders = {}; // { orderId: { price: string, side: string, amount: Decimal, status: string, timestamp: number } }
        this.active_buy_prices = {}; // { formattedPriceString: orderId }
        this.active_sell_prices = {}; // { formattedPriceString: orderId }
        this.trade_pairs = {}; // Stores buy legs { key: { price, amount, fee, timestamp, remaining_amount } } or processed sells { key: { processed_sell, pnl, timestamp } }
        this.position_size = ZERO; // Base currency amount (negative for short)
        this.entry_price = ZERO; // Average entry price
        this.unrealized_pnl = ZERO; // From exchange
        this.current_price = null; // Latest price (Decimal)
        this.last_price_update_time = 0; // Unix timestamp (seconds)
        this.trend_bias = 0; // 1=bull, -1=bear, 0=neutral
        this.order_book_imbalance = new Decimal("0.5"); // 0 to 1
        this.momentum_roc = ZERO; // Rate of Change (%)
        this.current_atr = null; // Latest ATR (Decimal)
        this.last_pivot_calculation_time = null; // Unix timestamp (seconds)
        this.current_pivot_points = {}; // { pp, r1, s1, ... } (Sorted descending)
        this.api_error_counter = 0; // Consecutive REST API errors
        this.websocket_connected_ticker = false;
        this.websocket_connected_orders = false;
        this.shutdown_triggered = false;
        this.grid_center_price = null; // Price grid was last centered (Decimal)
        this.history_needed = Math.max(
            config.ema_short_period || 0,
            config.ema_long_period || 0,
            config.atr_period || 0,
            config.momentum_period || 0,
            20 // Absolute minimum periods needed
        ) + 10; // Add buffer
        this.recent_closes = []; // Array of Decimals
        this.prev_ema_short = null; // Decimal
        this.prev_ema_long = null; // Decimal
        this.prev_atr = null; // Decimal

        // Concurrency Locks
        this.order_lock = new Mutex();
        this.pnl_lock = new Mutex();
        this.state_lock = new Mutex();
        this.init_lock = new Mutex();

        this.state_file_path = path.join(__dirname, config.state_file || 'grid_bot_state.json');
        log('DEBUG', `State file path: ${this.state_file_path}`);

        // Shutdown Handler
        this.shutdownHandler = new GracefulShutdown(this);
    }

    // --- State Persistence ---
    async _load_state() { await loadState(this, this.state_file_path); }
    async _save_state() { await saveState(this, this.state_file_path); }

    /** Resets volatile bot state variables asynchronously. */
    async _async_reset_state(save_after_reset = true) {
        log('WARNING', `${nc.yellow('Resetting internal bot state (async)...')}`);
        const releaseState = await this.state_lock.acquire();
        const releaseOrder = await this.order_lock.acquire();
        const releasePnl = await this.pnl_lock.acquire();
        try {
            this.prev_ema_short = null; this.prev_ema_long = null; this.prev_atr = null;
            this.recent_closes = []; this.momentum_roc = ZERO; this.current_atr = null; this.trend_bias = 0;
            this.active_orders = {}; this.active_buy_prices = {}; this.active_sell_prices = {};
            this.trade_pairs = {}; this.realized_pnl = ZERO;
            this.grid_center_price = null; this.current_pivot_points = {}; this.last_pivot_calculation_time = null;
            this.position_size = ZERO; this.entry_price = ZERO; this.unrealized_pnl = ZERO;
            this.order_book_imbalance = new Decimal("0.5"); this.api_error_counter = 0;
            log('INFO', 'Internal bot state reset complete (async).');
        } finally {
            releasePnl(); releaseOrder(); releaseState();
        }
        if (save_after_reset) {
            log('INFO', 'Saving reset state...');
            await this._save_state();
        }
    }

    // --- Helper Functions ---

    /** Formats a numeric value according to market precision rules. */
    _format_value(value, precision_info, rounding_mode = Decimal.ROUND_DOWN) {
        let valueDec = (value instanceof Decimal) ? value : safeDecimal(value, null);
        if (valueDec === null || !valueDec.isFinite()) return valueDec; // Return null/non-finite as is
        if (precision_info === null || precision_info === undefined || valueDec.isZero()) return valueDec; // No formatting needed/possible

        try {
            const precisionMode = this.market_info?.precisionMode ?? ccxt.DECIMAL_PLACES;
            if (precisionMode === ccxt.DECIMAL_PLACES) {
                let places = -1;
                if (typeof precision_info === 'number' && Number.isInteger(precision_info) && precision_info >= 0) places = precision_info;
                else if (typeof precision_info === 'string') {
                     if (/^\d+$/.test(precision_info)) places = parseInt(precision_info, 10);
                     else if (precision_info.includes('.') && precision_info.startsWith('0.')) places = precision_info.length - 2;
                     else if (precision_info.includes('e-')) {
                         try { places = Math.abs(Math.floor(Math.log10(parseFloat(precision_info)))); } catch { /* ignore */ }
                     }
                }
                if (places < 0 || !Number.isInteger(places)) {
                     log('WARNING', `Could not determine valid decimal places from precision_info: ${precision_info}. Returning unformatted.`); return valueDec;
                }
                return valueDec.toDecimalPlaces(places, rounding_mode);
            } else if (precisionMode === ccxt.TICK_SIZE) {
                const tickSize = safeDecimal(precision_info, null);
                if (tickSize === null || !tickSize.isFinite() || tickSize.lte(ZERO)) {
                     log('WARNING', `Invalid tick size: ${precision_info}. Returning unformatted.`); return valueDec;
                }
                return valueDec.div(tickSize).toInteger(rounding_mode).mul(tickSize);
            } else {
                 log('WARNING', `Unsupported precisionMode: ${precisionMode}. Returning unformatted.`); return valueDec;
            }
        } catch (e) {
             log('ERROR', `_format_value error for value '${valueDec?.toString()}', precision '${precision_info}': ${e.message}`, e);
             return null; // Return null on error
        }
    }

    /** Formats a price using market rules. ROUND_NEAREST default for display/logic, specific for placement. */
    _format_price(price, roundingMode = Decimal.ROUND_NEAREST) {
        const prec_info = this.market_info?.precision?.price;
        return this._format_value(price, prec_info, roundingMode);
    }

    /** Formats an amount using market rules. Always ROUND_DOWN. */
    _format_amount(amount) {
        const prec_info = this.market_info?.precision?.amount;
        return this._format_value(amount, prec_info, Decimal.ROUND_DOWN);
    }

    /** Checks order against exchange amount/cost limits and step precision. */
    _check_order_limits(side, price, amount) {
        if (!price || !amount || price.lte(ZERO) || amount.lte(ZERO)) return false;
        if (!this.market_info?.limits) { log('WARNING', "Market limits info unavailable."); return true; }

        const limits = this.market_info.limits;
        const precision = this.market_info.precision;
        const min_amount = safeDecimal(limits.amount?.min, ZERO);
        const max_amount = limits.amount?.max ? safeDecimal(limits.amount.max, null) : null;
        const min_cost = safeDecimal(limits.cost?.min, ZERO);
        const max_cost = limits.cost?.max ? safeDecimal(limits.cost.max, null) : null;
        const amount_step_info = precision?.amount;
        const amount_step = (this.market_info?.precisionMode === ccxt.TICK_SIZE && amount_step_info) ? safeDecimal(amount_step_info, null) : null;
        const cost = price.mul(amount);

        let amount_ok = amount.gte(min_amount);
        if (max_amount !== null && max_amount.gt(ZERO)) amount_ok = amount_ok && amount.lte(max_amount);
        if (amount_ok && amount_step !== null && amount_step.gt(ZERO)) {
             const remainder = amount.mod(amount_step);
             const tolerance = amount_step.mul('1e-9');
             if (!(remainder.abs().lte(tolerance) || remainder.abs().sub(amount_step).abs().lte(tolerance))) amount_ok = false;
        }
        let cost_ok = cost.gte(min_cost);
        if (max_cost !== null && max_cost.gt(ZERO)) cost_ok = cost_ok && cost.lte(max_cost);

        const valid = amount_ok && cost_ok;
        if (!valid) {
            const details = [];
            if (!amount_ok) details.push(`Amount ${formatDecimalString(amount, 8)} vs Lim(${formatDecimalString(min_amount, 8)}/${max_amount ? formatDecimalString(max_amount, 8) : 'N/A'}) Step(${amount_step ? formatDecimalString(amount_step, 8) : 'N/A'})`);
            if (!cost_ok) details.push(`Cost ${formatDecimalString(cost, 4)} vs Lim(${formatDecimalString(min_cost, 4)}/${max_cost ? formatDecimalString(max_cost, 4) : 'N/A'})`);
            log('WARNING', `Order Limit Fail (${side.toUpperCase()} @ ${formatDecimalString(price, 4)}): ${details.join('; ')}`);
        }
        return valid;
    }

    // --- API Interaction ---

    /** Executes a CCXT REST API call with retries, error handling, and timeout. */
    async _execute_ccxt_rest_call(methodName, ...args) {
        // ... (Implementation from previous response - considered complete and robust)
         if (this.shutdown_triggered) {
            log('INFO', `REST call ${nc.cyan(methodName)} skipped: Shutdown triggered.`);
            return null;
        }
        if (!this.exchange) {
            log('ERROR', `REST call '${methodName}' failed: Exchange not initialized.`);
            return null;
        }
        if (typeof this.exchange[methodName] !== 'function') {
             log('ERROR', `REST call failed: Method '${methodName}' not found on exchange instance.`);
             return null;
        }

        const retries = config.rest_api_retries || 3;
        const baseDelay = (config.rest_retry_base_delay_seconds || 5);
        let last_exception = null;
        const method_info = `${nc.cyan(methodName)}`;

        for (let attempt = 1; attempt <= retries; attempt++) {
            if (this.shutdown_triggered) {
                log('INFO', `REST call ${method_info} aborted (Attempt ${attempt}) due to shutdown trigger.`);
                return null;
            }

            try {
                log('DEBUG', `REST Call: ${method_info} (Attempt ${attempt}/${retries}) Args: ${JSON.stringify(args)}`);

                const timeout_ms = (config.rest_timeout_seconds || 20) * 1000;
                const result = await Promise.race([
                    this.exchange[methodName](...args),
                    new Promise((_, reject) =>
                        setTimeout(() => reject(new ccxt.RequestTimeout(`Custom Timeout (${timeout_ms / 1000}s) exceeded for ${methodName}`)), timeout_ms)
                    )
                ]);

                this.api_error_counter = 0; // Reset error counter on success
                return result;

            } catch (e) {
                last_exception = e;
                log('DEBUG', `REST Error (${method_info}, Attempt ${attempt}): ${e?.constructor?.name}, Msg: ${e?.message}`);

                let should_retry = false;
                let return_value = null;
                let handled = false;
                const error_message_lc = e?.message?.toLowerCase() || '';

                // --- Specific Handled Non-Retry Cases ---
                if (methodName === 'setLeverage' && error_message_lc.includes("leverage not modified")) {
                    log('INFO', `Leverage for ${args[1] || config.symbol} already set - handled.`);
                    return_value = { status: 'info', message: 'Leverage not modified' };
                    should_retry = false;
                    handled = true;
                }
                // Consolidate OrderNotFound / Already Closed/Filled/Cancelled errors
                else if (e instanceof ccxt.OrderNotFound || ['invalid order_id', 'order does not exist', 'order not found', 'order already closed', 'order already cancelled', 'order has been filled', 'order status error', 'unknown order sent', 'order finally failed', 'order has been closed', 'order was canceled', 'order id not found'].some(s => error_message_lc.includes(s))) {
                    log('WARNING', `Treating API error as OrderNotFound/Inactive (${method_info}): ${e.message}`);
                    return_value = { error: 'OrderNotFound', message: e?.message || 'Order not found or inactive' };
                    should_retry = false;
                    handled = true;
                }
                else if (e instanceof ccxt.AuthenticationError) {
                    log('CRITICAL', `${nc.bold(nc.red(`REST Authentication Error (${method_info}): ${e.message}. Check API keys. Halting!`))}`);
                    await this.shutdown("Authentication Error"); // Use the instance method
                    return_value = null;
                    should_retry = false;
                    handled = true;
                }
                else if (e instanceof ccxt.InsufficientFunds) {
                    log('ERROR', `${nc.red(`REST Insufficient Funds (${method_info}): ${e.message}`)}`);
                    if (config.send_sms_alerts && config.sms_alert_on_error) {
                        sendSmsAlert(`GridBot Insufficient Funds: ${config.symbol} ${methodName}.`).catch(smsErr => log('ERROR', `SMS Alert Error: ${smsErr.message}`));
                    }
                    return_value = { error: 'InsufficientFunds', message: e.message };
                    should_retry = false;
                    handled = true;
                }
                else if (error_message_lc.includes('post-only') || error_message_lc.includes('order would immediately match')) {
                     log('WARNING', `REST Post-Only Failure (${method_info}): ${e.message}`);
                     return_value = { error: 'PostOnlyFailed', message: e.message };
                     should_retry = false;
                     handled = true;
                }
                else if (e instanceof ccxt.BadArgument || e instanceof ccxt.BadRequest || error_message_lc.includes('invalid parameter') || error_message_lc.includes('parameter error')) {
                    log('ERROR', `${nc.red(`REST Bad Argument/Request (${method_info}): ${e.message}. Check parameters/config.`)}`);
                    return_value = { error: 'BadRequest', message: e.message };
                    should_retry = false;
                    handled = true;
                }


                // --- Retryable Errors ---
                if (!handled) {
                    if (e instanceof ccxt.RateLimitExceeded) {
                        let wait_time = baseDelay * Math.pow(1.5, attempt);
                        const retryAfterMatch = error_message_lc.match(/retry after (\d+)/) || error_message_lc.match(/try again in (\d+)/) || error_message_lc.match(/limit reach.* (\d+) seconds?/) || error_message_lc.match(/rate limit.*? (\d+)ms/);
                        if (retryAfterMatch?.[1]) {
                            const suggested_wait_s = error_message_lc.includes('ms') ? parseInt(retryAfterMatch[1], 10) / 1000 : parseInt(retryAfterMatch[1], 10);
                            wait_time = Math.max(wait_time, suggested_wait_s + 1);
                        }
                        wait_time = Math.min(wait_time, 60); // Cap wait time
                        log('WARNING', `${nc.yellow(`REST Rate Limit Exceeded (${method_info}). Retrying in ${wait_time.toFixed(1)}s...`)}`);
                        await new Promise(resolve => setTimeout(resolve, wait_time * 1000));
                        should_retry = true;
                        handled = true;
                    }
                    // Group common network/timeout/availability errors
                    else if (e instanceof ccxt.NetworkError || e instanceof ccxt.RequestTimeout || e instanceof ccxt.ExchangeNotAvailable || e instanceof ccxt.OperationTimeout || e instanceof ccxt.DDoSProtection || error_message_lc.includes('timeout') || error_message_lc.includes('connection reset') || error_message_lc.includes('connection refused') || error_message_lc.includes('service unavailable') || error_message_lc.includes('cloudflare') || error_message_lc.includes('proxy error') || error_message_lc.includes('bad gateway')) {
                        log('WARNING', `${nc.yellow(`REST Network/Timeout/Availability Error (${method_info}, Attempt ${attempt}/${retries}): ${e?.constructor?.name}`)}`);
                        this.api_error_counter++;
                        should_retry = true;
                        handled = true;
                    }
                    else if (e instanceof ccxt.ExchangeError) {
                        log('ERROR', `${nc.red(`REST Exchange Error (${method_info}, Attempt ${attempt}/${retries}): ${e.message}`)}`);
                        this.api_error_counter++;
                        should_retry = true;
                        handled = true;
                    }
                }

                // --- Unhandled Errors ---
                if (!handled) {
                    log('ERROR', `${nc.red(`Unexpected/Unhandled REST error (${method_info}, Attempt ${attempt}): ${e?.constructor?.name} - ${e?.message}`)}`, e);
                    this.api_error_counter++;
                    should_retry = true;
                }

                // --- Decide Action ---
                if (should_retry && attempt < retries) {
                    if (!(e instanceof ccxt.RateLimitExceeded)) {
                        const delay_ms = baseDelay * 1000 * Math.pow(1.5, attempt -1);
                        const capped_delay_ms = Math.min(delay_ms, 30000);
                        log('DEBUG', `Waiting ${capped_delay_ms / 1000}s before retry...`);
                        await new Promise(resolve => setTimeout(resolve, capped_delay_ms));
                    }
                    continue;
                } else {
                    if (should_retry && attempt >= retries) {
                        log('ERROR', `${nc.red(`REST call ${method_info} failed after ${retries} retries. Last Error: ${last_exception?.message}`)}`);
                        this.api_error_counter++;
                        return_value = last_exception;
                    } else {
                         log('DEBUG', `Not retrying error for ${method_info}. Returning: ${JSON.stringify(return_value)}`);
                    }

                    if (this.api_error_counter >= (config.max_consecutive_api_errors || 10)) {
                        log('CRITICAL', `${nc.bold(nc.red(`Shutdown triggered due to ${this.api_error_counter} consecutive REST API errors.`))}`);
                        await this.shutdown("Repeated REST Errors"); // Use the instance method
                        return null;
                    }
                    return return_value;
                }
            } // end catch
        } // end for loop

        log('ERROR', `REST call ${method_info} exited loop unexpectedly. Last Error: ${last_exception?.message}`);
        return last_exception || null;
    }

    // --- Initialization and Setup ---

    /** Initializes the CCXT exchange instance, loads markets, and performs initial setup. */
    async _initialize_exchange() {
        const release = await this.init_lock.acquire();
        try {
            if (this.exchange) {
                log('INFO', `Exchange ${this.exchange.id} already initialized.`);
                return true;
            }
            log('INFO', `Initializing CCXT for exchange: ${config.exchange_id}...`);

            const exchangeId = config.exchange_id.toLowerCase();
            const exchangeClass = ccxt[exchangeId];
            if (!exchangeClass) throw new Error(`Exchange '${exchangeId}' is not supported by CCXT.`);

            const exchangeOptions = {
                apiKey: process.env.BYBIT_API_KEY,
                secret: process.env.BYBIT_API_SECRET,
                enableRateLimit: true,
                timeout: (config.rest_timeout_seconds || 20) * 1000,
                options: {
                    defaultType: config.market_type || 'linear',
                    adjustForTimeDifference: true,
                    recvWindow: 15000,
                }
            };
            if (config.use_testnet) {
                 log('INFO', "Attempting to configure for Testnet mode...");
                 if (new exchangeClass().has?.sandbox) {
                     exchangeOptions.options.sandboxMode = true; log('INFO', "Using CCXT 'sandboxMode = true'.");
                 } else { log('WARNING', "'sandboxMode' not explicitly supported. Relying on Testnet API keys/URLs."); }
                 log('WARNING', `${nc.yellow("Testnet mode enabled. Ensure API keys are for the test environment.")}`);
            }

            this.exchange = new exchangeClass(exchangeOptions);
            log('INFO', `Initialized CCXT instance for ${this.exchange.id} (Version: ${this.exchange.version}).`);

            log('INFO', "Loading markets...");
            const markets = await this._execute_ccxt_rest_call('loadMarkets');
            if (!markets || typeof markets !== 'object' || Object.keys(markets).length === 0) {
                if (markets instanceof Error) throw markets;
                throw new Error("Failed to load markets.");
            }
            log('INFO', `Successfully loaded ${Object.keys(markets).length} markets.`);

            const symbol = config.symbol;
            if (!this.exchange.markets[symbol]) {
                throw new Error(`Symbol '${symbol}' not found on exchange '${exchangeId}'.`);
            }
            this.market_info = this.exchange.markets[symbol];
            if (!this.market_info) throw new Error(`Market info for '${symbol}' is missing.`);

            const limits = this.market_info.limits;
            log('INFO', `Market ${nc.bold(symbol)} loaded. Type: ${this.market_info?.type}, Settle: ${this.market_info?.settle}`);
            log('DEBUG', ` Precision: Price=${JSON.stringify(this.market_info?.precision?.price)}, Amount=${JSON.stringify(this.market_info?.precision?.amount)}, Mode=${this.market_info?.precisionMode}`);
            log('DEBUG', ` Limits: Amount=${JSON.stringify(limits?.amount)}, Cost=${JSON.stringify(limits?.cost)}`);
            log('DEBUG', ` Contract Size: ${this.market_info?.contractSize || 'N/A'}`);

            await this._set_leverage();
            await this._fetch_balance();
            await this._fetch_and_update_indicator_history();

            log('INFO', `${nc.green('Exchange initialization and setup complete.')}`);
            return true;

        } catch (error) {
            log('CRITICAL', `Exchange Initialization Failed: ${error.message}`, error);
            await this._safe_exchange_close();
            this.exchange = null;
            return false;
        } finally {
            release();
        }
    }

    /** Sets leverage for the symbol if applicable. */
    async _set_leverage() {
        const is_contract_market = this.market_info?.contract || this.market_info?.swap || this.market_info?.future || this.market_info?.linear || this.market_info?.inverse;
        if (!this.market_info?.spot && is_contract_market && this.exchange?.has['setLeverage']) {
            const leverage = parseInt(config.leverage, 10);
            if (!isNaN(leverage) && leverage > 0) {
                log('INFO', `Attempting to set leverage to ${leverage}x for ${config.symbol}...`);
                const params = {};
                if (config.exchange_id === 'bybit') {
                    if (config.market_type === 'linear') params.category = 'linear';
                    else if (config.market_type === 'inverse') params.category = 'inverse';
                    params.buyLeverage = leverage.toString();
                    params.sellLeverage = leverage.toString();
                }
                const response = await this._execute_ccxt_rest_call('setLeverage', leverage, config.symbol, params);
                // Handle response logging based on common patterns
                if (response === null || response instanceof Error) log('ERROR', `Leverage setting failed. API call returned: ${response instanceof Error ? response.message : 'null'}.`);
                else if (response.status === 'info' && response.message === 'Leverage not modified') log('INFO', `Leverage already set to ${leverage}x.`);
                else if (response?.info?.retMsg === 'ok' || response?.info?.result?.leverage || response?.leverage) log('INFO', `Leverage set request processed successfully.`);
                else log('WARNING', `Leverage set request processed. Response: ${JSON.stringify(response?.info || response)} (Check exchange)`);
            } else {
                log('WARNING', `Invalid leverage value '${config.leverage}' configured. Skipping.`);
            }
        } else {
            log('INFO', `Leverage setting not applicable or supported for ${config.symbol}. Skipping.`);
        }
    }

    // --- Fetch Methods --- (Implementations from previous response are complete)
    async _fetch_position() { /* ... Full implementation ... */
        log('DEBUG', `Fetching position for ${config.symbol}...`);
        if (this.market_info?.spot) {
            const releasePnl = await this.pnl_lock.acquire();
            try {
                if (!this.position_size.isZero() || !this.entry_price.isZero()) {
                    log('INFO', "Resetting position info for SPOT market.");
                    this.position_size = ZERO;
                    this.entry_price = ZERO;
                    this.unrealized_pnl = ZERO;
                }
            } finally {
                releasePnl();
            }
            return true; // Indicate success for spot
        }

        if (!this.exchange?.has['fetchPositions'] && !this.exchange?.has['fetchPosition']) {
            log('WARNING', "Exchange does not support fetchPositions() or fetchPosition(). Cannot fetch position data.");
            const releasePnl = await this.pnl_lock.acquire();
            try {
                this.position_size = ZERO;
                this.entry_price = ZERO;
                this.unrealized_pnl = ZERO;
            } finally {
                releasePnl();
            }
            return false;
        }

        try {
            const params = {};
            if (config.exchange_id === 'bybit') {
                if (config.market_type === 'linear') params.category = 'linear';
                else if (config.market_type === 'inverse') params.category = 'inverse';
            }

            let positions;
            if (this.exchange.has['fetchPosition']) {
                 log('DEBUG', 'Using fetchPosition(symbol)...');
                 const position = await this._execute_ccxt_rest_call('fetchPosition', config.symbol, params);
                 positions = (position && position.symbol === config.symbol) ? [position] : [];
            } else {
                 log('DEBUG', 'Using fetchPositions([symbol])...');
                 positions = await this._execute_ccxt_rest_call('fetchPositions', [config.symbol], params);
            }

            if (positions === null || positions instanceof Error) {
                 log('WARNING', `fetchPosition(s) API call failed or returned error: ${positions instanceof Error ? positions.message : 'null'}.`);
                 return false;
            }
            if (!Array.isArray(positions)) {
                log('WARNING', `fetchPosition(s) returned unexpected data type: ${typeof positions}. Expected array.`);
                return false;
            }

            const pos_data = positions.find(p => p?.symbol === config.symbol);
            let updated = false;

            const releasePnl = await this.pnl_lock.acquire();
            try {
                const old_size = this.position_size;

                const contracts_raw = pos_data?.contracts ?? pos_data?.contractSize;
                if (contracts_raw !== undefined && contracts_raw !== null) {
                    const contracts = safeDecimal(contracts_raw, ZERO);
                    const entry = safeDecimal(pos_data.entryPrice, ZERO);
                    const upnl = safeDecimal(pos_data.unrealizedPnl, ZERO);
                    const side = pos_data.side?.toLowerCase();

                    const new_position_size = contracts.isZero() ? ZERO : (side === 'short' ? contracts.negated() : contracts);
                    const new_entry_price = new_position_size.isZero() ? ZERO : entry;
                    const new_unrealized_pnl = new_position_size.isZero() ? ZERO : upnl;

                    if (!this.position_size.equals(new_position_size) || !this.entry_price.equals(new_entry_price)) {
                        this.position_size = new_position_size;
                        this.entry_price = new_entry_price;
                        this.unrealized_pnl = new_unrealized_pnl;
                        updated = true;
                    } else if (!this.unrealized_pnl.equals(new_unrealized_pnl)) {
                        this.unrealized_pnl = new_unrealized_pnl;
                        // updated = true; // Uncomment if log desired on UPNL change only
                    }

                } else {
                    if (!this.position_size.isZero()) {
                        log('INFO', `Position for ${config.symbol} reported as closed or not found by API. Resetting local state.`);
                        this.position_size = ZERO;
                        this.entry_price = ZERO;
                        this.unrealized_pnl = ZERO;
                        updated = true;
                    }
                }

                if (updated) {
                    log('INFO', `Position Update: Size=${formatDecimalString(this.position_size, 4)} Entry=${formatDecimalString(this.entry_price, 4)} UPNL=${formatDecimalString(this.unrealized_pnl, 4)}`);
                } else {
                    log('DEBUG', `Position check: No change. Size=${formatDecimalString(this.position_size, 4)} UPNL=${formatDecimalString(this.unrealized_pnl, 4)}`);
                }
            } finally {
                releasePnl();
            }

            return true;
        } catch (e) {
            log('ERROR', `Error processing fetched position data: ${e.message}`, e);
            return false;
        }
     }
    async _fetch_balance() { /* ... Full implementation ... */
        log('DEBUG', "Fetching account balance...");
        if (!this.exchange?.has['fetchBalance']) {
            log('WARNING', "Exchange does not support fetchBalance(). Cannot update balance.");
            return null;
        }

        const settle_currency = this.market_info?.settle;
        if (!settle_currency) {
            log('ERROR', `Cannot fetch balance: Settle currency for symbol ${config.symbol} is unknown.`);
            return null;
        }
        log('DEBUG', `Fetching balance for settle currency: ${settle_currency}`);

        let total_balance = null;

        try {
            const params = {};
            if (config.exchange_id === 'bybit') {
                if (config.market_type === 'spot') params.accountType = 'SPOT';
                else if (config.market_type === 'linear' || config.market_type === 'inverse') {
                    params.accountType = 'UNIFIED'; // Or 'CONTRACT' based on user setup
                }
                log('DEBUG', `Using Bybit params for fetchBalance: ${JSON.stringify(params)}`);
            }

            const balance_data = await this._execute_ccxt_rest_call('fetchBalance', params);

            if (balance_data === null || balance_data instanceof Error) {
                log('WARNING', `fetchBalance API call failed or returned error: ${balance_data instanceof Error ? balance_data.message : 'null'}.`);
                return null;
            }

            let total_balance_val = null;
            let available_balance_val = null;

            if (balance_data[settle_currency]) {
                total_balance_val = balance_data[settle_currency].total;
                available_balance_val = balance_data[settle_currency].free;
                log('DEBUG', `Parsed balance from standard structure for ${settle_currency}`);
            } else if (balance_data.total?.[settle_currency] !== undefined) {
                total_balance_val = balance_data.total[settle_currency];
                available_balance_val = balance_data.free?.[settle_currency];
                log('DEBUG', `Parsed balance from total/free sub-object structure for ${settle_currency}`);
            } else if (balance_data.info) {
                log('DEBUG', "Attempting to parse balance from 'info' field...");
                try {
                    // Bybit V5 example: info.result.list[{coin, equity, availableToWithdraw,...}]
                    if (config.exchange_id === 'bybit' && Array.isArray(balance_data.info?.result?.list)) {
                        const account_info = balance_data.info.result.list.find(a => a.coin === settle_currency);
                        if (account_info) {
                            total_balance_val = account_info.equity ?? account_info.walletBalance;
                            available_balance_val = account_info.availableToWithdraw ?? account_info.availableBalance;
                            log('DEBUG', `Parsed balance from Bybit info.result.list for ${settle_currency}`);
                        }
                    }
                } catch (parse_error) {
                    log('WARNING', `Could not parse balance from 'info' field: ${parse_error.message}`);
                }
            }

            if (total_balance_val === null || total_balance_val === undefined) {
                log('WARNING', `Could not find balance information for currency ${settle_currency} in the response.`);
                log('DEBUG', `Full balance response: ${JSON.stringify(balance_data)}`);
                return null;
            }

            const total_balance_dec = safeDecimal(total_balance_val, null);
            const available_balance_dec = safeDecimal(available_balance_val, ZERO);

            if (total_balance_dec === null || !total_balance_dec.isFinite()) {
               log('ERROR', `Failed to convert total balance value '${total_balance_val}' to valid Decimal.`);
               return null;
            }

            total_balance = total_balance_dec;

            log('INFO', `Balance (${settle_currency}): Total = ${formatDecimalString(total_balance, 4)}, Available = ${formatDecimalString(available_balance_dec, 4)}`);

            const releaseState = await this.state_lock.acquire();
            let should_save_state = false;
            try {
                // Record initial balance only if not already set and total balance is positive
                if (this.initial_balance_usd === null && total_balance.gt(ZERO)) {
                    this.initial_balance_usd = total_balance;
                    log('INFO', `${nc.green(`Initial balance recorded: ${formatDecimalString(this.initial_balance_usd, 4)} ${settle_currency}`)}`);
                    should_save_state = true;
                }
            } finally {
                releaseState();
            }

            if (should_save_state) {
                await this._save_state();
            }

            return total_balance;

        } catch (e) {
            log('ERROR', `Error fetching or processing balance: ${e.message}`, e);
            return null;
        }
     }
    async _fetch_ohlcv(timeframe, limit, params = {}) { /* ... Full implementation ... */
        const logPrefix = `fetchOHLCV(${timeframe}, limit=${limit})`;
        log('DEBUG', `${logPrefix}: Fetching...`);
        if (!this.exchange?.has['fetchOHLCV']) {
            log('WARNING', `${logPrefix}: Operation not supported by exchange.`);
            return null;
        }
        const actual_limit = limit + 10; // Fetch buffer
        try {
            const fetch_params = { ...params };
            if (config.exchange_id === 'bybit') {
                if (config.market_type === 'linear') fetch_params.category = 'linear';
                else if (config.market_type === 'inverse') fetch_params.category = 'inverse';
                else if (config.market_type === 'spot') fetch_params.category = 'spot';
            }

            const ohlcv = await this._execute_ccxt_rest_call('fetchOHLCV', config.symbol, timeframe, undefined, actual_limit, fetch_params);

            if (ohlcv === null || ohlcv instanceof Error) {
                 log('WARNING', `${logPrefix}: API call failed or returned error: ${ohlcv instanceof Error ? ohlcv.message : 'null'}.`);
                 return null;
            }
            if (!Array.isArray(ohlcv)) {
                log('WARNING', `${logPrefix}: Invalid response type. Expected array, got ${typeof ohlcv}`);
                return null;
            }
            if (ohlcv.length === 0) {
                log('WARNING', `${logPrefix}: Received empty array.`);
                return []; // Return empty array, let caller handle
            }
            if (!Array.isArray(ohlcv[0]) || ohlcv[0].length < 5) {
                log('WARNING', `${logPrefix}: Invalid candle structure in response. Example: ${JSON.stringify(ohlcv[0])}`);
                return null;
            }
            // Ensure ascending sort by timestamp
            if (ohlcv.length > 1 && ohlcv[0][0] > ohlcv[ohlcv.length - 1][0]) {
                log('WARNING', `${logPrefix}: OHLCV data may not be sorted correctly. Re-sorting.`);
                ohlcv.sort((a, b) => a[0] - b[0]);
            }

            log('DEBUG', `${logPrefix}: Received ${ohlcv.length} candles. Returning last ${limit}.`);
            return ohlcv.slice(-limit);

        } catch (e) {
            log('ERROR', `${logPrefix}: Error during fetch: ${e.message}`, e);
            return null;
        }
     }
    async _fetch_order_book(limit = null) { /* ... Full implementation ... */
        log('DEBUG', "Fetching order book...");
        if (!this.exchange?.has['fetchOrderBook']) {
            log('WARNING', "fetchOrderBook() not supported by exchange.");
            return null;
        }
        const book_limit = limit ?? config.order_book_limit ?? 20;
        try {
             const params = {};
             if (config.exchange_id === 'bybit') {
                if (config.market_type === 'linear') params.category = 'linear';
                else if (config.market_type === 'inverse') params.category = 'inverse';
                else if (config.market_type === 'spot') params.category = 'spot';
             }

             const ob = await this._execute_ccxt_rest_call('fetchOrderBook', config.symbol, book_limit, params);

             if (ob === null || ob instanceof Error) {
                 log('WARNING', `fetchOrderBook API call failed or returned error: ${ob instanceof Error ? ob.message : 'null'}.`);
                 return null;
             }
             if (ob && Array.isArray(ob.bids) && Array.isArray(ob.asks)) {
                 log('DEBUG', `Fetched order book: ${ob.bids.length} bids, ${ob.asks.length} asks.`);
                 // Basic structure validation
                 if (ob.bids.length > 0 && (!Array.isArray(ob.bids[0]) || ob.bids[0].length < 2)) {
                     log('WARNING', `fetchOrderBook returned invalid bid structure: ${JSON.stringify(ob.bids[0])}`);
                     return null;
                 }
                 if (ob.asks.length > 0 && (!Array.isArray(ob.asks[0]) || ob.asks[0].length < 2)) {
                      log('WARNING', `fetchOrderBook returned invalid ask structure: ${JSON.stringify(ob.asks[0])}`);
                      return null;
                 }
                 return ob;
             } else {
                 log('WARNING', `fetchOrderBook returned invalid structure: ${JSON.stringify(ob)}`);
                 return null;
             }
        } catch (e) {
            log('ERROR', `fetchOrderBook error: ${e.message}`, e);
            return null;
        }
     }

    // --- Startup Synchronization ---
    async _sync_open_orders_from_rest() { /* ... Full implementation ... */
        if (!config.sync_orders_on_startup) {
            log('INFO', "Skipping REST order synchronization on startup as per configuration.");
            return;
        }
        if (!this.exchange?.has['fetchOpenOrders']) {
            log('ERROR', "Cannot sync orders: fetchOpenOrders() is not supported by the exchange.");
            return;
        }

        log('INFO', "Synchronizing open orders with exchange via REST API...");
        try {
            const params = {};
            if (config.exchange_id === 'bybit') {
                if (config.market_type === 'linear') params.category = 'linear';
                else if (config.market_type === 'inverse') params.category = 'inverse';
                else if (config.market_type === 'spot') params.category = 'spot';
            }
            const exchange_orders_result = await this._execute_ccxt_rest_call('fetchOpenOrders', config.symbol, undefined, undefined, params);

            let valid_exchange_orders = [];
            if (exchange_orders_result === null || exchange_orders_result instanceof Error) {
                if (exchange_orders_result?.error === 'OrderNotFound' || exchange_orders_result?.message?.toLowerCase().includes('order not found')) {
                     log('INFO', 'fetchOpenOrders indicated no open orders found.');
                     valid_exchange_orders = [];
                } else {
                     log('ERROR', `Failed to fetch open orders. Error: ${exchange_orders_result instanceof Error ? exchange_orders_result.message : 'API call failed'}`);
                     return; // Hard failure
                }
            } else if (!Array.isArray(exchange_orders_result)) {
                log('ERROR', `Received invalid data type from fetchOpenOrders: ${typeof exchange_orders_result}`);
                return; // Hard failure
            } else {
                valid_exchange_orders = exchange_orders_result;
            }

            log('INFO', `Found ${valid_exchange_orders.length} open order(s) on the exchange for ${config.symbol}.`);

            const exchange_ids = new Set(valid_exchange_orders.map(o => o.id));
            const cancel_tasks = []; // Store orders to be cancelled if untracked

            const releaseOrder = await this.order_lock.acquire();
            try {
                const local_ids = new Set(Object.keys(this.active_orders));
                let added = 0, removed = 0, updated = 0;

                // Process orders found on the exchange
                for (const o of valid_exchange_orders) {
                    const oid = o.id;
                    const side = o.side?.toLowerCase();
                    const price_dec = safeDecimal(o.price, null);
                    const amount_dec = safeDecimal(o.amount, null);
                    const status = o.status || 'open';
                    const ts_ms = o.timestamp || o.lastUpdateTimestamp || Date.now();
                    const ts_sec = Math.floor(ts_ms / 1000);

                    if (!oid || !side || !price_dec || price_dec.lte(ZERO) || !amount_dec || amount_dec.lte(ZERO)) {
                         log('WARNING', `Sync: Skipping invalid order data from exchange: ${JSON.stringify(o)}`);
                         continue;
                    }
                    // Use ROUND_NEAREST for comparison, placement formatting done in _place_order
                    const fmtPrice = this._format_price(price_dec, Decimal.ROUND_NEAREST);
                    if (!fmtPrice || fmtPrice.lte(ZERO)) {
                        log('WARNING', `Sync: Skipping order ${oid.slice(-6)} due to invalid formatted price from ${price_dec.toString()}`);
                        continue;
                    }
                    const fmtPriceStr = formatDecimalString(fmtPrice);

                    if (!local_ids.has(oid)) {
                        // Order exists on exchange but not locally
                        if (amount_dec.gt(MIN_ORDER_AMOUNT_THRESHOLD_DEC)) {
                            if (config.cancel_untracked_orders_on_sync) {
                                log('WARNING', `Sync: Found untracked open order ${oid.slice(-6)} (${side} @ ${fmtPriceStr}). Scheduling for cancellation.`);
                                cancel_tasks.push({ id: oid, price: fmtPriceStr, side: side });
                            } else {
                                log('WARNING', `Sync: Found untracked open order ${oid.slice(-6)} (${side} @ ${fmtPriceStr}). Adding to local state.`);
                                // Add using the consistently formatted price string
                                this.active_orders[oid] = { price: fmtPriceStr, side, amount: amount_dec, status, timestamp: ts_sec };
                                if (side === 'buy') this.active_buy_prices[fmtPriceStr] = oid;
                                else this.active_sell_prices[fmtPriceStr] = oid;
                                added++;
                            }
                        } else {
                            log('DEBUG', `Sync: Skipping untracked order ${oid.slice(-6)} with amount ${formatDecimalString(amount_dec)} below threshold.`);
                        }
                    } else {
                        // Order exists locally and on exchange, check for status updates
                        const local_order = this.active_orders[oid];
                        const new_status = status;
                        const new_ts = ts_sec;

                        if (local_order.status !== new_status || local_order.timestamp < new_ts) {
                            if(local_order.status !== new_status) {
                                log('INFO', `Sync: Updating status for order ${oid.slice(-6)} from '${local_order.status}' to '${new_status}'.`);
                            }
                            local_order.status = new_status;
                            local_order.timestamp = new_ts;
                            updated++;
                        }
                    }
                }

                // Process orders that exist locally but NOT on the exchange
                for (const local_id of local_ids) {
                    if (!exchange_ids.has(local_id)) {
                        const local_order = this.active_orders[local_id];
                        if (local_order) {
                            if (local_order.status === 'closed' || local_order.status === 'canceled' || local_order.status === 'cancelled') {
                                log('DEBUG', `Sync: Local order ${local_id.slice(-6)} already marked '${local_order.status}', removing stale entry.`);
                            } else {
                                log('WARNING', `Sync: Local order ${local_id.slice(-6)} (${local_order.side} @ ${local_order.price}, status: ${local_order.status}) not found open on exchange. Assuming filled/cancelled/expired. Removing from local state.`);
                            }
                            const price_str = local_order.price; // Use the stored price string
                            if (local_order.side === 'buy' && this.active_buy_prices[price_str] === local_id) delete this.active_buy_prices[price_str];
                            else if (local_order.side === 'sell' && this.active_sell_prices[price_str] === local_id) delete this.active_sell_prices[price_str];
                            delete this.active_orders[local_id];
                            removed++;
                        }
                    }
                }
                log('INFO', `Order Sync Results: Added=${added}, Removed=${removed}, Updated=${updated}. Current Local Orders: ${Object.keys(this.active_orders).length}`);

            } finally {
                releaseOrder();
            }

            // Cancel untracked orders if configured
            if (cancel_tasks.length > 0) {
                log('INFO', `Cancelling ${cancel_tasks.length} untracked orders found during sync...`);
                let cancelled_count = 0;
                const delay_ms = (config.api_call_delay_seconds || 0.3) * 1000;
                for (const task of cancel_tasks) {
                    if (this.shutdown_triggered) { log('WARNING', 'Shutdown triggered during sync cancellation.'); break; }
                    log('INFO', `Attempting cancel untracked: ID=${task.id.slice(-6)}, Side=${task.side}, Price=${task.price}`);
                    const success = await this._cancel_order(task.id, task.price, task.side); // Pass info
                    if (success) {
                        cancelled_count++;
                    } else {
                        log('ERROR', `Failed to cancel untracked order ${task.id.slice(-6)} during sync.`);
                    }
                    if (delay_ms > 0 && !this.shutdown_triggered) {
                        await new Promise(r => setTimeout(r, delay_ms));
                    }
                }
                log('INFO', `Sync cancellation results: ${cancelled_count} / ${cancel_tasks.length} succeeded.`);
            }

            log('INFO', "Order synchronization via REST complete.");

        } catch(e) {
            log('ERROR', `Error during order synchronization: ${e.message}`, e);
        }
     }


    // --- Order Placement / Cancellation --- (Implementations from previous response are complete)
    _calculate_dynamic_size_factor(side, price) { /* ... Full implementation ... */
        if (!(price instanceof Decimal) || price.lte(ZERO)) {
             log('DEBUG', `Dynamic size: Using default factor (1.0) due to invalid input price.`);
             return ONE;
        }
        if (!this.current_price || this.current_price.lte(ZERO) || (Date.now() / 1000 - this.last_price_update_time > 120)) { // 2 min staleness check
            log('DEBUG', `Dynamic size: Using default factor (1.0) due to missing or stale current price data.`);
            return ONE;
        }

        let base_factor = ONE;
        const factors_applied = []; // Keep track of adjustments made

        try {
            // 1. Volatility Adjustment (using ATR)
            if (config.volatility_size_adjustment && this.current_atr?.gt(ZERO)) {
                const atr_percent = this.current_atr.div(this.current_price).mul(100);
                let vol_factor = ONE;
                const high_thresh = safeDecimal(config.high_volatility_threshold_percent, ZERO);
                const low_thresh = safeDecimal(config.low_volatility_threshold_percent, ZERO);
                // Factor < 1 reduces size, Factor > 1 increases size
                const high_factor = safeDecimal(config.volatility_high_size_factor, ONE); // e.g., 0.5 to reduce size
                const low_factor = safeDecimal(config.volatility_low_size_factor, ONE); // e.g., 1.2 to increase size

                if (high_thresh.gt(ZERO) && atr_percent.gt(high_thresh) && high_factor.lt(ONE)) { // Reduce size in high vol
                    vol_factor = high_factor;
                    factors_applied.push(`VolHi(ATR ${formatDecimalString(atr_percent, 2)}% > ${formatDecimalString(high_thresh)}%): x${formatDecimalString(vol_factor, 2)}`);
                } else if (low_thresh.gt(ZERO) && atr_percent.lt(low_thresh) && low_factor.gt(ONE)) { // Increase size in low vol
                    vol_factor = low_factor;
                    factors_applied.push(`VolLo(ATR ${formatDecimalString(atr_percent, 2)}% < ${formatDecimalString(low_thresh)}%): x${formatDecimalString(vol_factor, 2)}`);
                }
                if (!vol_factor.equals(ONE)) {
                    base_factor = base_factor.mul(vol_factor);
                }
            }

            // 2. Momentum Adjustment (using ROC)
            if (config.momentum_size_adjustment && !this.momentum_roc.isZero()) { // Check if momentum is non-zero
                const roc = this.momentum_roc; // Already a Decimal percentage
                let mom_factor = ONE;
                // Base factor to add/subtract (e.g., 0.2 for +/- 20%)
                const adj_base = safeDecimal(config.momentum_size_factor, ZERO);
                const pos_thresh = safeDecimal(config.momentum_threshold_positive, ZERO); // e.g., 0.5
                const neg_thresh = safeDecimal(config.momentum_threshold_negative, ZERO); // e.g., -0.5 (should be negative)

                if (adj_base.gt(ZERO)) {
                    // Increase size when trading WITH strong momentum
                    if (roc.gt(pos_thresh) && side === 'buy') { // Strong positive momentum, increase buy size
                        mom_factor = ONE.add(adj_base);
                        factors_applied.push(`MomBuy(ROC ${formatDecimalString(roc, 2)}% > ${formatDecimalString(pos_thresh)}%): +${formatDecimalString(adj_base, 2)}`);
                    } else if (roc.lt(neg_thresh) && side === 'sell') { // Strong negative momentum, increase sell size
                        mom_factor = ONE.add(adj_base);
                        factors_applied.push(`MomSell(ROC ${formatDecimalString(roc, 2)}% < ${formatDecimalString(neg_thresh)}%): +${formatDecimalString(adj_base, 2)}`);
                    }
                    // Decrease size when trading AGAINST strong momentum (fading)
                    else if (roc.lt(neg_thresh) && side === 'buy') { // Strong negative momentum, decrease buy size
                        mom_factor = ONE.sub(adj_base);
                         factors_applied.push(`MomFadeBuy(ROC ${formatDecimalString(roc, 2)}% < ${formatDecimalString(neg_thresh)}%): -${formatDecimalString(adj_base, 2)}`);
                    } else if (roc.gt(pos_thresh) && side === 'sell') { // Strong positive momentum, decrease sell size
                        mom_factor = ONE.sub(adj_base);
                        factors_applied.push(`MomFadeSell(ROC ${formatDecimalString(roc, 2)}% > ${formatDecimalString(pos_thresh)}%): -${formatDecimalString(adj_base, 2)}`);
                    }
                }

                // Ensure factor is reasonable (e.g., not negative or zero)
                mom_factor = mom_factor.max(new Decimal("0.1")); // Minimum factor of 0.1
                if (!mom_factor.equals(ONE)) { // Only apply if it changed
                     base_factor = base_factor.mul(mom_factor);
                }
            }

            // 3. Order Book Imbalance (OBI) Adjustment
            if (config.obi_size_adjustment && this.order_book_imbalance) {
                const obi = this.order_book_imbalance; // Ratio 0 to 1
                let obi_factor = ONE;
                const adj_factor_val = safeDecimal(config.obi_size_factor, ONE).min(ONE);
                const buy_weak_thresh = safeDecimal(config.order_book_imbalance_filter_buy_threshold, ZERO);
                const sell_weak_thresh = safeDecimal(config.order_book_imbalance_filter_sell_threshold, ONE);

                const is_unfavorable_buy = buy_weak_thresh.gt(ZERO) && obi.lt(buy_weak_thresh);
                const is_unfavorable_sell = sell_weak_thresh.lt(ONE) && obi.gt(sell_weak_thresh);

                if ((side === 'buy' && is_unfavorable_buy) || (side === 'sell' && is_unfavorable_sell)) {
                    if (adj_factor_val.lt(ONE)) {
                        obi_factor = adj_factor_val;
                        factors_applied.push(`OBI(${formatDecimalString(obi, 3)} vs ${side === 'buy' ? '<' + formatDecimalString(buy_weak_thresh) : '>' + formatDecimalString(sell_weak_thresh)}): x${formatDecimalString(obi_factor, 2)}`);
                        base_factor = base_factor.mul(obi_factor);
                    }
                }
            }

        } catch (error) {
            log('ERROR', `Error calculating dynamic size factor: ${error.message}`, error);
            return ONE; // Return default factor on error
        }

        // Ensure final factor is within configured bounds
        const min_factor = safeDecimal(config.min_dynamic_size_factor, "0.1").max(ZERO);
        const max_factor = safeDecimal(config.max_dynamic_size_factor, "3.0");
        let final_factor = base_factor.max(min_factor).min(max_factor);

        // Final sanity check
        if (!final_factor.isFinite() || final_factor.lte(ZERO)) {
            log('WARNING', `Dynamic size calculation resulted in invalid factor (${final_factor.toString()}). Using default (1.0). Base: ${base_factor.toString()}`);
            final_factor = ONE;
        }

        if (factors_applied.length > 0 || !final_factor.equals(base_factor)) {
            log('DEBUG', `Dynamic Size Factor (${side} @ ${formatDecimalString(price, 4)}): ${nc.yellow(formatDecimalString(final_factor, 3))} (Base: ${formatDecimalString(base_factor, 3)}, Adjustments: ${factors_applied.join(', ') || 'None'}, Clamped: ${!final_factor.equals(base_factor)})`);
        }
        return final_factor;
     }
    _passes_obi_filter(side) { /* ... Full implementation ... */
        if (!config.enable_obi_filter || !this.order_book_imbalance) {
             return true;
         }

         const obi = this.order_book_imbalance; // Ratio 0 to 1
         const buy_thresh = safeDecimal(config.order_book_imbalance_filter_buy_threshold, ZERO);
         const sell_thresh = safeDecimal(config.order_book_imbalance_filter_sell_threshold, ONE);

         if (side === 'buy' && buy_thresh.gt(ZERO) && obi.lt(buy_thresh)) {
             log('INFO', `Order Filtered: BUY order blocked by Order Book Imbalance (${formatDecimalString(obi, 3)} < threshold ${formatDecimalString(buy_thresh, 3)})`);
             return false;
         }
         if (side === 'sell' && sell_thresh.lt(ONE) && obi.gt(sell_thresh)) {
             log('INFO', `Order Filtered: SELL order blocked by Order Book Imbalance (${formatDecimalString(obi, 3)} > threshold ${formatDecimalString(sell_thresh, 3)})`);
             return false;
         }
         return true;
     }
    async _place_order(side, price, base_amount) { /* ... Full implementation ... */
        const priceDec = safeDecimal(price, null);
        const baseAmountDec = safeDecimal(base_amount, null);

        if (!priceDec || priceDec.lte(ZERO)) {
            log('WARNING', `Skipping ${side} order: Invalid price (${price?.toString()}).`);
            return null;
        }
        if (!baseAmountDec || baseAmountDec.lte(ZERO)) {
             log('WARNING', `Skipping ${side} order @ ${priceDec.toString()}: Invalid base amount (${base_amount?.toString()}).`);
             return null;
        }
        if (baseAmountDec.lte(MIN_ORDER_AMOUNT_THRESHOLD_DEC)) {
             log('WARNING', `Skipping ${side} order @ ${priceDec.toString()}: Base amount ${formatDecimalString(baseAmountDec)} below minimum threshold ${formatDecimalString(MIN_ORDER_AMOUNT_THRESHOLD_DEC)}.`);
             return null;
        }

        const sideLower = side.toLowerCase();
        if (sideLower !== 'buy' && sideLower !== 'sell') {
            log('ERROR', `Invalid order side specified: ${side}`);
            return null;
        }

        // Use specific rounding for placing orders
        const placeRounding = sideLower === 'buy' ? Decimal.ROUND_DOWN : Decimal.ROUND_UP;
        const formatted_price = this._format_price(priceDec, placeRounding);
        if (!formatted_price || formatted_price.lte(ZERO)) {
            log('WARNING', `Skipping ${sideLower.toUpperCase()} order: Formatted price is invalid (${formatted_price?.toString()}). Original: ${priceDec.toString()}`);
            return null;
        }
        const price_str = formatDecimalString(formatted_price);
        const logPrefix = `${sideLower.toUpperCase()} @ ${price_str}`;


        try {
            // Pre-placement Checks
            if (config.max_order_distance_percent > 0 && this.current_price?.gt(ZERO)) {
                const max_dist_dec = safeDecimal(config.max_order_distance_percent, ZERO);
                const distance_percent = formatted_price.sub(this.current_price).abs().div(this.current_price).mul(100);
                if (distance_percent.gt(max_dist_dec)) {
                    log('INFO', `Skipping ${logPrefix}: Order distance ${formatDecimalString(distance_percent, 2)}% exceeds maximum ${formatDecimalString(max_dist_dec)}%.`);
                    return null;
                }
            }

            if (!this._passes_obi_filter(sideLower)) {
                return null;
            }

            const size_factor = this._calculate_dynamic_size_factor(sideLower, formatted_price);
            const dynamic_amount = baseAmountDec.mul(size_factor);
            const final_amount_formatted = this._format_amount(dynamic_amount); // Format final amount

            if (!final_amount_formatted || final_amount_formatted.lte(MIN_ORDER_AMOUNT_THRESHOLD_DEC)) {
                log('INFO', `Skipping ${logPrefix}: Final amount ${formatDecimalString(final_amount_formatted)} too small after factor ${formatDecimalString(size_factor, 3)} & formatting. Base: ${formatDecimalString(baseAmountDec)}`);
                return null;
            }

            // Acquire Lock and Check Internal State
            const releaseOrder = await this.order_lock.acquire();
            let order_result = null;
            try {
                const buys = Object.keys(this.active_buy_prices).length;
                const sells = Object.keys(this.active_sell_prices).length;
                if (sideLower === 'buy' && buys >= config.max_open_orders_per_side) {
                    log('INFO', `Skipping ${logPrefix}: Maximum open buy orders (${config.max_open_orders_per_side}) reached.`);
                    return null;
                }
                if (sideLower === 'sell' && sells >= config.max_open_orders_per_side) {
                    log('INFO', `Skipping ${logPrefix}: Maximum open sell orders (${config.max_open_orders_per_side}) reached.`);
                    return null;
                }

                if ((sideLower === 'buy' && this.active_buy_prices[price_str]) || (sideLower === 'sell' && this.active_sell_prices[price_str])) {
                    log('DEBUG', `Skipping ${logPrefix}: An active order already exists at this formatted price level.`);
                    return null;
                }

                if (!this._check_order_limits(sideLower, formatted_price, final_amount_formatted)) {
                    return null;
                }

                const side_color = sideLower === 'buy' ? nc.green : nc.red;
                log('INFO', `${side_color(sideLower.toUpperCase().padEnd(4))} ${nc.reset} Placing Order: Amt=${nc.blue(formatDecimalString(final_amount_formatted, 8).padEnd(12))} @ Px=${nc.blue(price_str.padEnd(10))} (Factor=${formatDecimalString(size_factor, 3)})`);

                const params = {};
                if (config.post_only_orders) {
                    params.postOnly = true;
                }
                if (config.exchange_id === 'bybit') {
                    if (config.market_type === 'linear') params.category = 'linear';
                    else if (config.market_type === 'inverse') params.category = 'inverse';
                    else if (config.market_type === 'spot') params.category = 'spot';
                }

                const amount_str = final_amount_formatted.toString();
                const price_api_str = formatted_price.toString();

                releaseOrder(); // Release before API call

                order_result = await this._execute_ccxt_rest_call(
                    'createLimitOrder',
                    config.symbol,
                    sideLower,
                    amount_str,
                    price_api_str,
                    params
                );

                await this.order_lock.acquire(); // Re-acquire lock

                 if (order_result?.id) {
                     const { id: oid, status = 'open', timestamp: ts_ms } = order_result;
                     const timestamp_sec = (ts_ms ? Math.floor(ts_ms / 1000) : Math.floor(Date.now() / 1000));
                     log('INFO', `${side_color(sideLower.toUpperCase())} Order Initiated:${nc.reset} ID=${oid.slice(-6)}, Status:${status}`);

                     if (!this.active_orders[oid]) {
                         this.active_orders[oid] = {
                             price: price_str, side: sideLower, amount: final_amount_formatted,
                             status: status, timestamp: timestamp_sec
                         };
                         if (sideLower === 'buy') this.active_buy_prices[price_str] = oid;
                         else this.active_sell_prices[price_str] = oid;
                         return order_result;
                     } else {
                          log('WARNING', `Order ${oid.slice(-6)} already exists in local state after placement attempt (likely WS update race).`);
                          return order_result;
                     }

                 } else {
                     if (order_result?.error === 'PostOnlyFailed') log('WARNING', `${logPrefix} Placement Failed: Post-Only order would have executed immediately.`);
                     else if (order_result?.error === 'InsufficientFunds') log('ERROR', `${logPrefix} Placement Failed: Insufficient funds. ${order_result.message}`);
                     else if (order_result?.error === 'BadRequest') log('ERROR', `${logPrefix} Placement Failed: Bad Request/Parameters. ${order_result.message}`);
                     else if (order_result instanceof Error) log('ERROR', `${logPrefix} Placement FAILED. API Error: ${order_result.constructor.name} - ${order_result.message}`);
                     else if (order_result === null) log('ERROR', `${logPrefix} Placement FAILED. API call failed or returned null.`);
                     else log('ERROR', `${logPrefix} Placement FAILED. API Response: ${JSON.stringify(order_result)}`);
                     return null;
                 }
            } finally {
                if (this.order_lock.isLocked()) {
                     // Check if releaseOrder function still exists (it might be null if released before await)
                     if (typeof releaseOrder === 'function') releaseOrder();
                     else this.order_lock.release(); // Manually release if re-acquired
                }
            }
        } catch (error) {
            log('ERROR', `Unhandled error in _place_order (${logPrefix}): ${error.message}`, error);
            return null;
        }
     }
    async _cancel_order(order_id, order_price_str = null, order_side = null) { /* ... Full implementation ... */
        let local_data = null;
        const releaseCheck = await this.order_lock.acquire();
        try {
            local_data = this.active_orders[order_id];
            order_price_str = order_price_str || local_data?.price || '?';
            order_side = order_side || local_data?.side || '?';
        } finally {
            releaseCheck();
        }

        const side_color = order_side === 'buy' ? nc.green : order_side === 'sell' ? nc.red : nc.yellow;
        const log_prefix = `${nc.yellow('Cancel Req:')} ID=${order_id.slice(-6)} (${side_color(order_side.toUpperCase())}@${order_price_str}${nc.reset})`;
        log('INFO', `${log_prefix} Sending cancellation request...`);

        if (!this.exchange?.has['cancelOrder']) {
            log('ERROR', `${log_prefix} - Failed: cancelOrder() operation not supported by the exchange.`);
            return false;
        }

        try {
            const params = {};
            if (config.exchange_id === 'bybit') {
                 if (config.market_type === 'linear') params.category = 'linear';
                 else if (config.market_type === 'inverse') params.category = 'inverse';
                 else if (config.market_type === 'spot') params.category = 'spot';
            }
            const response = await this._execute_ccxt_rest_call('cancelOrder', order_id, config.symbol, params);
            let cancelled_remotely = false;

            if (response?.error === 'OrderNotFound') {
                log('WARNING', `${log_prefix} - Order already not found on exchange (or closed/cancelled). Treating as success.`);
                cancelled_remotely = true;
            } else if (response instanceof Error && response.message?.toLowerCase().includes('order not found')) {
                log('WARNING', `${log_prefix} - Order likely not found on exchange (Error: ${response.message}). Treating as success.`);
                cancelled_remotely = true;
            } else if (response && typeof response === 'object' && !(response instanceof Error) && response.error === undefined && (response.id === order_id || response.info?.orderId === order_id || response.info?.result?.orderId === order_id)) {
                 log('INFO', `${log_prefix} - Cancellation request accepted/confirmed by exchange. Status: ${response.status || response.info?.orderStatus || 'N/A'}`);
                 cancelled_remotely = true;
            } else if (response instanceof Error) {
                log('ERROR', `${log_prefix} - Cancellation failed. API Error: ${response.constructor.name} - ${response.message}.`);
                cancelled_remotely = false;
            } else if (response === null) {
                log('ERROR', `${log_prefix} - Cancellation failed (API call returned null).`);
                cancelled_remotely = false;
            } else {
                log('WARNING', `${log_prefix} - Cancellation returned unexpected/ambiguous response: ${JSON.stringify(response)}. Assuming failure.`);
                cancelled_remotely = false;
            }

            if (cancelled_remotely) {
                const releaseUpdate = await this.order_lock.acquire();
                try {
                    const current_local = this.active_orders[order_id];
                    if (current_local) {
                        const price_str = current_local.price;
                        const side = current_local.side;
                        if (side === 'buy' && this.active_buy_prices[price_str] === order_id) delete this.active_buy_prices[price_str];
                        else if (side === 'sell' && this.active_sell_prices[price_str] === order_id) delete this.active_sell_prices[price_str];
                        delete this.active_orders[order_id];
                        log('DEBUG', `Removed cancelled order ${order_id.slice(-6)} from local state.`);
                        return true;
                    } else {
                        log('DEBUG', `Order ${order_id.slice(-6)} was already removed locally.`);
                        return true;
                    }
                } finally {
                    releaseUpdate();
                }
            } else {
                return false;
            }
        } catch (error) {
            log('ERROR', `${log_prefix} - Unhandled error during cancel logic: ${error.message}`, error);
            return false;
        }
     }
    async _cancel_all_orders(reason = "Generic Cancel All") { /* ... Full implementation ... */
        log('WARNING', `${nc.yellow(`# Cancelling ALL open orders for symbol ${config.symbol} (Reason: ${reason})...`)}`);
        let local_ids_to_cancel = [];
        const releaseCheck = await this.order_lock.acquire();
        try {
            local_ids_to_cancel = Object.keys(this.active_orders);
        } finally {
            releaseCheck();
        }

        if (local_ids_to_cancel.length === 0) {
            log('INFO', "No active orders found locally to cancel.");
            return true; // Nothing to do
        }

        log('INFO', `Found ${local_ids_to_cancel.length} active order(s) locally. Attempting cancellation...`);
        let overall_success = true; // Assume success initially

        // --- Attempt Bulk Cancel First (if supported) ---
        if (this.exchange?.has['cancelAllOrders']) {
            log('INFO', "Attempting bulk cancellation via cancelAllOrders()...");
            try {
                const params = {};
                if (config.exchange_id === 'bybit') {
                     if (config.market_type === 'linear') params.category = 'linear';
                     else if (config.market_type === 'inverse') params.category = 'inverse';
                     else if (config.market_type === 'spot') params.category = 'spot';
                }
                const response = await this._execute_ccxt_rest_call('cancelAllOrders', config.symbol, params);

                if (response !== null && !(response instanceof Error) && response.error === undefined) {
                    log('INFO', `Bulk cancelAllOrders request processed by exchange. Response: ${JSON.stringify(response)}. Clearing local order state.`);
                    const releaseClear = await this.order_lock.acquire();
                    let cleared_count = 0;
                    try {
                        cleared_count = Object.keys(this.active_orders).length;
                        this.active_orders = {};
                        this.active_buy_prices = {};
                        this.active_sell_prices = {};
                    } finally {
                        releaseClear();
                    }
                    log('INFO', `Local order state cleared (${cleared_count} orders). Assuming bulk cancel succeeded.`);
                    return true; // Exit successfully after bulk cancel attempt
                } else {
                    log('ERROR', `Bulk cancelAllOrders failed or returned unexpected response: ${response instanceof Error ? response.message : JSON.stringify(response)}. Falling back to individual cancellation.`);
                    overall_success = false; // Mark as failed for now, fallback will try again
                }
            } catch (error) {
                log('ERROR', `Error during cancelAllOrders(): ${error.message}. Falling back to individual cancellation.`, error);
                overall_success = false; // Mark as failed
            }
        } else {
            log('INFO', "cancelAllOrders() not supported. Using individual cancellation fallback.");
        }

        // --- Fallback: Individual Cancellation ---
        log('INFO', "Initiating individual order cancellation loop...");
        let success_count = 0;
        let failure_count = 0;
        const delay_ms = (config.api_call_delay_seconds || 0.3) * 1000;

        log('INFO', `Starting fallback cancel for ${local_ids_to_cancel.length} orders initially identified...`);

        for (const oid of local_ids_to_cancel) {
             if (this.shutdown_triggered && reason !== "Shutdown Request" && reason !== "Signal") {
                 log('WARNING', 'Shutdown triggered during individual cancellation loop.');
                 overall_success = false;
                 break;
             }
             let needs_cancel = false;
             const releaseRecheck = await this.order_lock.acquire();
             try { needs_cancel = !!this.active_orders[oid]; } finally { releaseRecheck(); }

             if (needs_cancel) {
                 const ok = await this._cancel_order(oid);
                 if (ok) {
                     success_count++;
                 } else {
                     failure_count++;
                     overall_success = false;
                 }
                 if (delay_ms > 0 && !this.shutdown_triggered) {
                     await new Promise(r => setTimeout(r, delay_ms));
                 }
             } else {
                 log('DEBUG', `Skipping individual cancel for ${oid.slice(-6)}, already removed from local state.`);
             }
        }

        log('INFO', `Individual cancellation fallback complete: ${success_count} succeeded, ${failure_count} failed.`);
        const releaseFinalCheck = await this.order_lock.acquire();
        try {
            const remaining_count = Object.keys(this.active_orders).length;
            if (remaining_count > 0) {
                 log('WARNING', `${remaining_count} orders still remain in local state after cancellation attempts.`);
                 overall_success = false;
            } else {
                 log('INFO', "Local order state is now empty.");
            }
        } finally {
             releaseFinalCheck();
        }

        return overall_success;
     }


    // --- Indicator Calculations (Delegation) ---
    _calculate_manual_sma(data, period) { return calculate_manual_sma(data, period); }
    _calculate_manual_ema(data, period, prev) { return calculate_manual_ema(data, period, prev); }
    _calculate_manual_atr(ohlcv, period, prev) { return calculate_manual_atr(ohlcv, period, prev); }
    _calculate_momentum_roc(closes, period) { return calculate_momentum_roc(closes, period); }


    // --- Analysis Data Calculation --- (Implementations from previous response are complete)
    async _fetch_and_update_indicator_history() { /* ... Full implementation ... */
         const release = await this.state_lock.acquire();
        let history_updated = false;
        try {
            const current_length = this.recent_closes.length;
            // Force fetch if not enough history OR if incremental calculation values are missing
            const needs_fetch = current_length < this.history_needed ||
                                (config.ema_short_period > 0 && !this.prev_ema_short) ||
                                (config.ema_long_period > 0 && !this.prev_ema_long) ||
                                (config.atr_period > 0 && !this.prev_atr);

            if (needs_fetch) {
                 log('INFO', `Indicator history needs update (Len:${current_length}/${this.history_needed}, Missing Prev Vals:${!this.prev_ema_short || !this.prev_ema_long || !this.prev_atr}). Fetching ${this.history_needed + 20} klines...`);
                 release(); // Release lock before long API call
                 const ohlcv_hist = await this._fetch_ohlcv(config.kline_interval, this.history_needed + 20);
                 await this.state_lock.acquire(); // Re-acquire lock

                 if (ohlcv_hist && ohlcv_hist.length >= this.history_needed) {
                      const closes = ohlcv_hist
                          .map(row => safeDecimal(row[4], null))
                          .filter(c => c !== null && c.isFinite() && c.gt(ZERO));

                      if (closes.length >= this.history_needed) {
                          this.recent_closes = closes;
                          log('INFO', `Indicator history updated with ${this.recent_closes.length} valid close prices.`);
                          this.prev_ema_short = null; this.prev_ema_long = null; this.prev_atr = null;
                          history_updated = true;
                      } else {
                          log('ERROR', `Fetched ${ohlcv_hist.length} candles, but only found ${closes.length} valid close prices. Need at least ${this.history_needed}. History not updated.`);
                      }
                 } else {
                     log('ERROR', `Failed to fetch sufficient OHLCV history. Received ${ohlcv_hist?.length || 0} candles, needed ${this.history_needed}. History not updated.`);
                 }
            } else {
                log('DEBUG', `Sufficient indicator history available (${current_length} closes).`);
            }
        } catch (error) {
            log('ERROR', `Error fetching or updating indicator history: ${error.message}`, error);
        } finally {
            if (this.state_lock.isLocked()) release();
        }
        return history_updated;
     }
    async _calculate_analysis_data() { /* ... Full implementation ... */
        log('DEBUG', "Calculating analysis data (Indicators, Trend, OBI)...");
        const start_time = Date.now();
        let history_was_updated = false;
        let latest_close_price = null; // Decimal

        try {
             history_was_updated = await this._fetch_and_update_indicator_history();

             const latest_kline_interval = config.kline_interval || '1m';
             const ohlcv_latest = await this._fetch_ohlcv(latest_kline_interval, 2);
             if (ohlcv_latest && ohlcv_latest.length > 0) {
                 const latest_complete_candle_index = ohlcv_latest.length > 1 ? ohlcv_latest.length - 2 : 0;
                 const latest_complete_candle = ohlcv_latest[latest_complete_candle_index];
                 latest_close_price = safeDecimal(latest_complete_candle[4], null);
                 if (!latest_close_price || !latest_close_price.isFinite() || latest_close_price.lte(ZERO)) {
                     log('WARNING', `Could not get valid latest close price from OHLCV. Candle: ${JSON.stringify(latest_complete_candle)}`);
                     latest_close_price = null;
                 } else {
                     log('DEBUG', `Using latest completed candle close: ${formatDecimalString(latest_close_price, 4)}`);
                 }
             } else {
                 log('WARNING', "Could not fetch latest OHLCV candle to update indicators.");
             }

             const releaseState = await this.state_lock.acquire();
             let indicators_recalculated = false;
             let trend_updated = false;
             try {
                 let new_close_added = false;
                 if (latest_close_price) {
                     const last_stored_close = this.recent_closes[this.recent_closes.length - 1];
                     if (!last_stored_close || !last_stored_close.equals(latest_close_price)) {
                         this.recent_closes.push(latest_close_price);
                         new_close_added = true;
                         if (this.recent_closes.length > this.history_needed * 1.5) {
                             this.recent_closes.shift();
                         }
                         indicators_recalculated = true;
                         log('DEBUG', `Appended latest close price ${formatDecimalString(latest_close_price, 4)} to history (${this.recent_closes.length} points).`);
                     }
                 }

                 const min_hist_for_calcs = Math.max(
                     config.ema_short_period || 0, config.ema_long_period || 0,
                     (config.momentum_period || 0) + 1, 1
                 );

                 if (this.recent_closes.length < min_hist_for_calcs) {
                     log('WARNING', `Insufficient history (${this.recent_closes.length}) for EMA/ROC calculations (need ${min_hist_for_calcs}). Skipping.`);
                 } else {
                     const closes_list = [...this.recent_closes];
                     const force_full_ema_recalc = history_was_updated || new_close_added;
                     if(force_full_ema_recalc && !history_was_updated) log('DEBUG', 'Forcing EMA recalculation due to new close price.');

                     let ema_short = null, ema_long = null;
                     if (config.ema_short_period > 0) {
                         ema_short = this._calculate_manual_ema(closes_list, config.ema_short_period, force_full_ema_recalc ? null : this.prev_ema_short);
                         if (ema_short && (!this.prev_ema_short || !this.prev_ema_short.equals(ema_short))) { this.prev_ema_short = ema_short; indicators_recalculated = true; }
                         else if (!ema_short) log('WARNING', 'Short EMA calculation failed.');
                     }
                     if (config.ema_long_period > 0) {
                         ema_long = this._calculate_manual_ema(closes_list, config.ema_long_period, force_full_ema_recalc ? null : this.prev_ema_long);
                         if (ema_long && (!this.prev_ema_long || !this.prev_ema_long.equals(ema_long))) { this.prev_ema_long = ema_long; indicators_recalculated = true; }
                         else if (!ema_long) log('WARNING', 'Long EMA calculation failed.');
                     }

                     if (config.momentum_period > 0) {
                         const new_roc = this._calculate_momentum_roc(closes_list, config.momentum_period);
                         if (new_roc !== null && new_roc.isFinite()) {
                             if (!this.momentum_roc.equals(new_roc)) { this.momentum_roc = new_roc; indicators_recalculated = true; }
                         } else if (new_roc === null) { log('WARNING', 'Momentum ROC calculation returned null.'); }
                           else { log('WARNING', `Momentum ROC calculation returned non-finite value: ${new_roc?.toString()}`); }
                     }

                     const price_for_trend = (this.current_price?.gt(ZERO) && (Date.now() / 1000 - this.last_price_update_time < 120))
                         ? this.current_price : (latest_close_price || (closes_list.length > 0 ? closes_list[closes_list.length - 1] : null));

                     const prev_bias = this.trend_bias;
                     let new_bias = 0;
                     const eS = this.prev_ema_short; const eL = this.prev_ema_long; const roc = this.momentum_roc;
                     const trend_roc_pos_thresh = safeDecimal(config.trend_roc_threshold_positive, config.momentum_threshold_positive || ZERO);
                     const trend_roc_neg_thresh = safeDecimal(config.trend_roc_threshold_negative, config.momentum_threshold_negative || ZERO);

                     if (eS && eL && price_for_trend?.gt(ZERO)) {
                         const price_gt_eS = price_for_trend.gt(eS); const price_lt_eS = price_for_trend.lt(eS);
                         const eS_gt_eL = eS.gt(eL); const eS_lt_eL = eS.lt(eL);
                         const roc_gt_pos = roc.gt(trend_roc_pos_thresh); const roc_lt_neg = roc.lt(trend_roc_neg_thresh);
                         if (price_gt_eS && eS_gt_eL && roc_gt_pos) new_bias = 1;
                         else if (price_lt_eS && eS_lt_eL && roc_lt_neg) new_bias = -1;
                     } else log('DEBUG', "Trend bias calculation skipped: Missing price or EMA data.");

                     if (this.trend_bias !== new_bias) {
                         const trendMap = { 1: nc.green('BULLISH'), '-1': nc.red('BEARISH'), 0: nc.yellow('NEUTRAL') };
                         log('INFO', `Trend Bias Updated: ${trendMap[prev_bias] || 'N/A'} -> ${nc.bold(trendMap[new_bias] || 'N/A')}`);
                         this.trend_bias = new_bias; trend_updated = true;
                     }
                 }

             } finally {
                 releaseState();
             }

             let atr_updated = false;
             if (config.atr_period > 0) {
                 const atr_period = config.atr_period || 14;
                 const atr_ohlcv = await this._fetch_ohlcv(config.kline_interval, atr_period + 5);
                 if (atr_ohlcv && atr_ohlcv.length >= atr_period) {
                     const releaseAtr = await this.state_lock.acquire();
                     try {
                         const force_atr_recalc = history_was_updated || !this.prev_atr;
                         if (force_atr_recalc) log('DEBUG', 'Forcing full ATR recalculation.');
                         const new_atr = this._calculate_manual_atr(atr_ohlcv, atr_period, force_atr_recalc ? null : this.prev_atr);
                         if (new_atr && new_atr.isFinite() && new_atr.gt(ZERO)) {
                             if (!this.current_atr || !this.current_atr.equals(new_atr)) {
                                 log('DEBUG', `ATR updated: ${formatDecimalString(this.current_atr || ZERO, 4)} -> ${formatDecimalString(new_atr, 4)}`);
                                 this.current_atr = new_atr; atr_updated = true;
                             }
                             this.prev_atr = this.current_atr;
                         } else if (new_atr === null) { log('WARNING', "ATR calculation returned null."); }
                           else { log('WARNING', `ATR calculation returned invalid value: ${new_atr?.toString()}`); }
                     } finally { releaseAtr(); }
                 } else log('WARNING', `ATR calculation skipped: Failed to fetch sufficient OHLCV data (${atr_ohlcv?.length || 0} candles).`);
             }

             let obi_updated = false;
             if (config.enable_obi_filter || config.obi_size_adjustment) {
                 const order_book = await this._fetch_order_book();
                 if (order_book?.bids && order_book?.asks && order_book.bids.length > 0 && order_book.asks.length > 0) {
                     try {
                         const depth = config.order_book_limit || 20;
                         const bids = order_book.bids.slice(0, depth); const asks = order_book.asks.slice(0, depth);
                         let bid_volume = ZERO; let ask_volume = ZERO;
                         for (const [p, a] of bids) bid_volume = bid_volume.add(safeDecimal(p, ZERO).mul(safeDecimal(a, ZERO)));
                         for (const [p, a] of asks) ask_volume = ask_volume.add(safeDecimal(p, ZERO).mul(safeDecimal(a, ZERO)));
                         const total_volume = bid_volume.add(ask_volume);
                         let new_obi = new Decimal("0.5");
                         if (total_volume.gt(ZERO)) new_obi = bid_volume.div(total_volume).toDecimalPlaces(5);
                         const change_threshold = new Decimal("0.005");
                         if (!this.order_book_imbalance.equals(new_obi) && this.order_book_imbalance.sub(new_obi).abs().gt(change_threshold)) {
                             log('DEBUG', `OBI Updated: ${formatDecimalString(this.order_book_imbalance, 3)} -> ${formatDecimalString(new_obi, 3)}`);
                             this.order_book_imbalance = new_obi; obi_updated = true;
                         }
                     } catch (e) { log('ERROR', `Order Book Imbalance calculation error: ${e.message}`, e); }
                 } else log('WARNING', "OBI calculation skipped: Order book fetch failed or returned empty/invalid data.");
             }

             const duration = (Date.now() - start_time) / 1000;
             if (indicators_recalculated || trend_updated || atr_updated || obi_updated) {
                 const atr_str = this.current_atr ? formatDecimalString(this.current_atr, 4) : 'N/A';
                 const roc_str = formatDecimalString(this.momentum_roc, 2) + '%';
                 const obi_str = (config.enable_obi_filter || config.obi_size_adjustment) ? formatDecimalString(this.order_book_imbalance, 3) : 'N/A';
                 const trendMap = { 1: nc.green('BULL'), '-1': nc.red('BEAR'), 0: nc.gray('NEUT') };
                 const trend_str = trendMap[this.trend_bias] || nc.gray('NEUT');
                 const ema_s_str = this.prev_ema_short ? formatDecimalString(this.prev_ema_short, 4) : 'N/A';
                 const ema_l_str = this.prev_ema_long ? formatDecimalString(this.prev_ema_long, 4) : 'N/A';
                 log('INFO', `Analysis Update (${duration.toFixed(2)}s): Trend=${trend_str}, EMA(S/L)=${ema_s_str}/${ema_l_str}, ATR=${atr_str}, ROC=${roc_str}, OBI=${obi_str}`);
             } else log('DEBUG', `Analysis Update (${duration.toFixed(2)}s): No significant changes detected.`);

        } catch (error) {
            log('ERROR', `Error during analysis data calculation: ${error.message}`, error);
        }
     }

    // --- Fibonacci Pivot Points ---
    async _calculate_fib_pivot_points() { /* ... Full implementation ... */
        log('INFO', "Calculating Fibonacci Pivot Points...");
        try {
             const timeframe = config.daily_kline_interval || '1d';
             const ohlcv = await this._fetch_ohlcv(timeframe, 3);

             if (!ohlcv || ohlcv.length < 2) {
                 log('WARNING', `Insufficient historical data (${ohlcv?.length || 0} candles) for timeframe ${timeframe} to calculate pivots.`);
                 return false;
             }

             const prev_period_candle_index = ohlcv.length - 2;
             const prev_period_candle = ohlcv[prev_period_candle_index];
             if (!Array.isArray(prev_period_candle) || prev_period_candle.length < 5) {
                 log('WARNING', `Invalid previous period candle data structure: ${JSON.stringify(prev_period_candle)}`);
                 return false;
             }

             const [ts_ms, , high_raw, low_raw, close_raw ] = prev_period_candle;
             const H = safeDecimal(high_raw, null); const L = safeDecimal(low_raw, null); const C = safeDecimal(close_raw, null);

             const date_str = new Date(ts_ms).toISOString().split('T')[0];
             log('INFO', `Pivot Input Data (${date_str} ${timeframe}): High=${formatDecimalString(H)}, Low=${formatDecimalString(L)}, Close=${formatDecimalString(C)}`);

             if (H === null || L === null || C === null || !H.isFinite() || !L.isFinite() || !C.isFinite() || H.lte(ZERO) || L.lte(ZERO) || C.lte(ZERO) || H.lt(L)) {
                 log('WARNING', `Invalid High/Low/Close values obtained for pivot calculation. H=${H?.toString()}, L=${L?.toString()}, C=${C?.toString()}`);
                 return false;
             }

             const PP = H.add(L).add(C).div(3);
             const R = H.sub(L);
             if (R.lte(ZERO)) {
                 log('WARNING', `Pivot calculation failed: Range (H-L) is zero or negative (${R.toString()}).`);
                 return false;
             }

             const fib_ratios = {
                 r3: ONE, r2: new Decimal("0.618"), r1: new Decimal("0.382"),
                 s1: new Decimal("0.382"), s2: new Decimal("0.618"), s3: ONE
             };

             const pivots = { pp: PP };
             pivots.r1 = PP.add(R.mul(fib_ratios.r1)); pivots.r2 = PP.add(R.mul(fib_ratios.r2)); pivots.r3 = PP.add(R.mul(fib_ratios.r3));
             pivots.s1 = PP.sub(R.mul(fib_ratios.s1)); pivots.s2 = PP.sub(R.mul(fib_ratios.s2)); pivots.s3 = PP.sub(R.mul(fib_ratios.s3));

             log('DEBUG', `Raw calculated pivots: ${JSON.stringify(Object.fromEntries(Object.entries(pivots).map(([k,v])=>[k, v?.toString()])))}`);

             const fmt_pivots = {};
             const price_precision = this.market_info?.precision?.price;

             for (const [key, value] of Object.entries(pivots)) {
                 if (value instanceof Decimal && value.isFinite()) {
                     const positiveValue = value.max(ZERO);
                     const formattedValue = this._format_price(positiveValue, Decimal.ROUND_NEAREST); // Use ROUND_NEAREST for pivot levels
                     if (formattedValue && formattedValue.isFinite() && formattedValue.gte(ZERO)) fmt_pivots[key] = formattedValue;
                     else log('WARNING', `Skipping pivot key '${key}': Formatting resulted in invalid value (${formattedValue?.toString()}).`);
                 } else log('WARNING', `Skipping invalid raw pivot value for key '${key}'.`);
             }

             if (Object.keys(fmt_pivots).length === 0) {
                 log('ERROR', "Pivot calculation resulted in no valid formatted pivot points."); return false;
             }

             const sorted_pivots_array = Object.entries(fmt_pivots).sort(([, valA], [, valB]) => valB.comparedTo(valA));
             const sorted_pivots_obj = Object.fromEntries(sorted_pivots_array);

             log('DEBUG', `Sorted formatted pivots: ${JSON.stringify(Object.fromEntries(Object.entries(sorted_pivots_obj).map(([k,v])=>[k, v?.toString()])))}`);

             const releaseState = await this.state_lock.acquire();
             let pivots_changed = false;
             try {
                 const current_keys = Object.keys(this.current_pivot_points); const new_keys = Object.keys(sorted_pivots_obj);
                 if (current_keys.length !== new_keys.length || !current_keys.every(key => new_keys.includes(key))) pivots_changed = true;
                 else {
                     for (const key of new_keys) {
                         if (!this.current_pivot_points[key] || !this.current_pivot_points[key].equals(sorted_pivots_obj[key])) { pivots_changed = true; break; }
                     }
                 }
                 if (pivots_changed || current_keys.length === 0) {
                     log('INFO', 'Fibonacci Pivot points have been updated.');
                     this.current_pivot_points = sorted_pivots_obj;
                 } else log('INFO', 'Pivots recalculated, but no significant change detected.');
                 this.last_pivot_calculation_time = Date.now() / 1000; // Update time regardless
             } finally { releaseState(); }

             if (Object.keys(this.current_pivot_points).length > 0) {
                 const log_str = Object.entries(this.current_pivot_points)
                     .map(([k, v]) => `${k.toUpperCase()}:${formatDecimalString(v, price_precision ?? 4)}`)
                     .join(' | ');
                 log('INFO', `${nc.blueBright('Pivots Updated:')} ${log_str}`);
             } else { log('WARNING', 'Pivot calculation finished but resulted in empty pivot data in state.'); return false; }
             return true;

        } catch (error) {
            log('ERROR', `Error calculating Fibonacci pivot points: ${error.message}`, error);
            return false;
        }
     }

    // --- Grid Placement / Replenishment / Reset --- (Implementations from previous response are complete)
    async place_initial_grid() { /* ... Full implementation ... */
        log('INFO', `${nc.cyan('=== Placing Initial Grid ===')}`);
        const releaseInit = await this.init_lock.acquire();
        try {
            let pivots; let tempReleaseState = null;
            try {
                tempReleaseState = await this.state_lock.acquire();
                pivots = { ...this.current_pivot_points };
                const pivots_exist = Object.keys(pivots).length > 0;
                const pivot_interval_ms = this.exchange.parseTimeframe(config.daily_kline_interval || '1d') * 1000;
                const pivot_age_s = this.last_pivot_calculation_time ? (Date.now() / 1000 - this.last_pivot_calculation_time) : Infinity;
                const pivots_stale = pivot_age_s > (pivot_interval_ms / 1000 + 3600);

                if (!pivots_exist || pivots_stale) {
                    log('WARNING', `Pivots ${!pivots_exist ? 'not available' : 'are stale'}. Recalculating...`);
                    tempReleaseState(); tempReleaseState = null;
                    const recalc_ok = await this._calculate_fib_pivot_points();
                    tempReleaseState = await this.state_lock.acquire();
                    pivots = { ...this.current_pivot_points };
                    if (!recalc_ok || Object.keys(pivots).length === 0) throw new Error("Pivot calculation failed.");
                    log('INFO', "Pivots recalculated successfully.");
                }
            } finally { if (tempReleaseState && this.state_lock.isLocked()) tempReleaseState(); }

            const price_stale = !this.current_price || this.current_price.lte(ZERO) || (Date.now() / 1000 - this.last_price_update_time > 60);
            if (price_stale) {
                log('INFO', 'Fetching latest ticker for current price...');
                const ticker = await this._execute_ccxt_rest_call('fetchTicker', config.symbol);
                if (ticker?.last) {
                    const price = safeDecimal(ticker.last, null);
                    if (price?.gt(ZERO)) {
                        this.current_price = price;
                        this.last_price_update_time = (ticker.timestamp ? Math.floor(ticker.timestamp / 1000) : Math.floor(Date.now() / 1000));
                        log('INFO', `Fetched current price: ${formatDecimalString(this.current_price, 4)}`);
                    } else throw new Error(`Fetched ticker has invalid 'last' price: ${ticker.last}`);
                } else {
                    if (ticker instanceof Error) throw new Error(`Failed fetchTicker: ${ticker.message}`);
                    throw new Error("Failed fetchTicker (null/invalid response).");
                }
            }
            log('INFO', `Using current price for grid setup: ${formatDecimalString(this.current_price, 4)}`);

            log('INFO', "Cancelling existing orders before placing new grid...");
            if (!await this._cancel_all_orders("Initial Grid Placement")) {
                log('ERROR', `${nc.red('Failed to cancel all existing orders! Proceeding cautiously.')}`);
                await new Promise(r => setTimeout(r, 5000));
            } else {
                 const waitS = config.grid_reset_delay_seconds || 2;
                 if (waitS > 0) { log('INFO', `Waiting ${waitS}s...`); await new Promise(r => setTimeout(r, waitS * 1000)); }
            }

            const releaseStateCenter = await this.state_lock.acquire();
            try {
                const pp_level = pivots.pp;
                this.grid_center_price = (pp_level instanceof Decimal && pp_level.gt(ZERO)) ? pp_level : this.current_price;
                log('INFO', `Setting Grid Center Price to: ${formatDecimalString(this.grid_center_price, 4)}`);
            } finally { releaseStateCenter(); }

            const potential_buys = []; const potential_sells = [];
            for (const [key, value] of Object.entries(pivots)) {
                if (value instanceof Decimal && value.gt(ZERO)) {
                    const fmtPivotPrice = this._format_price(value, Decimal.ROUND_NEAREST);
                    if (fmtPivotPrice && fmtPivotPrice.gt(ZERO)) {
                        if (fmtPivotPrice.lt(this.current_price)) potential_buys.push({ price: fmtPivotPrice, name: key });
                        if (fmtPivotPrice.gt(this.current_price)) potential_sells.push({ price: fmtPivotPrice, name: key });
                    }
                }
            }
            potential_buys.sort((a, b) => b.price.comparedTo(a.price));
            potential_sells.sort((a, b) => a.price.comparedTo(b.price));
            log('DEBUG', `Potential Buys (${potential_buys.length}): ${potential_buys.map(p => `${p.name}:${formatDecimalString(p.price, 4)}`).join(', ')}`);
            log('DEBUG', `Potential Sells (${potential_sells.length}): ${potential_sells.map(p => `${p.name}:${formatDecimalString(p.price, 4)}`).join(', ')}`);

            let placed_count = 0;
            const base_order_size_usd = safeDecimal(config.order_size_usd, null);
            if (!base_order_size_usd || base_order_size_usd.lte(ZERO)) throw new Error(`Invalid order_size_usd: ${config.order_size_usd}`);
            const max_orders_per_side = config.max_open_orders_per_side;
            const place_delay_ms = (config.api_call_delay_seconds || 0.3) * 1000;
            const placed_sell_prices = new Set(); const placed_buy_prices = new Set();

            log('INFO', `Placing up to ${max_orders_per_side} SELL orders...`);
            for (let i = 0; i < potential_sells.length; i++) {
                if (this.shutdown_triggered || placed_sell_prices.size >= max_orders_per_side) break;
                const target_level = potential_sells[i];
                const target_price_place = this._format_price(target_level.price, Decimal.ROUND_UP); // Format for SELL placement
                if (!target_price_place || target_price_place.lte(ZERO)) continue;
                const price_str = formatDecimalString(target_price_place);
                if (placed_sell_prices.has(price_str)) continue;
                const base_amount = base_order_size_usd.div(target_price_place);
                const order_result = await this._place_order('sell', target_price_place, base_amount);
                if (order_result?.id) {
                    placed_count++; placed_sell_prices.add(price_str);
                    if (place_delay_ms > 0 && !this.shutdown_triggered) await new Promise(r => setTimeout(r, place_delay_ms));
                } else log('WARNING', `Failed to place initial SELL order at ${target_level.name} (${price_str}).`);
            }

            log('INFO', `Placing up to ${max_orders_per_side} BUY orders...`);
            for (let i = 0; i < potential_buys.length; i++) {
                 if (this.shutdown_triggered || placed_buy_prices.size >= max_orders_per_side) break;
                 const target_level = potential_buys[i];
                 const target_price_place = this._format_price(target_level.price, Decimal.ROUND_DOWN); // Format for BUY placement
                 if (!target_price_place || target_price_place.lte(ZERO)) continue;
                 const price_str = formatDecimalString(target_price_place);
                 if (placed_buy_prices.has(price_str)) continue;
                 const base_amount = base_order_size_usd.div(target_price_place);
                 const order_result = await this._place_order('buy', target_price_place, base_amount);
                 if (order_result?.id) {
                     placed_count++; placed_buy_prices.add(price_str);
                     if (place_delay_ms > 0 && !this.shutdown_triggered) await new Promise(r => setTimeout(r, place_delay_ms));
                 } else log('WARNING', `Failed to place initial BUY order at ${target_level.name} (${price_str}).`);
            }

            log('INFO', `${nc.cyan(`=== Initial Grid Placement Complete. Placed ${placed_count} orders. ===`)}`);
            await this._save_state();
            return placed_count > 0;

        } catch (error) {
            log('CRITICAL', `Error during initial grid placement: ${error.message}`, error);
            return false;
        } finally {
            releaseInit();
        }
     }
    async _replenish_grid_level(filled_side, filled_price_decimal) { /* ... Full implementation ... */
        if (filled_side !== 'buy' && filled_side !== 'sell') { log('ERROR', `Replenish invalid side: ${filled_side}`); return; }
        if (!(filled_price_decimal instanceof Decimal) || !filled_price_decimal.isFinite() || filled_price_decimal.lte(ZERO)) { log('ERROR', `Replenish invalid price: ${filled_price_decimal?.toString()}`); return; }

        const log_prefix = `Replenish (${filled_side.toUpperCase()} fill @ ${formatDecimalString(filled_price_decimal, 4)}):`;
        log('INFO', `${log_prefix} Attempting to place opposing order...`);

        let pivots;
        const releaseState = await this.state_lock.acquire();
        try {
            if (Object.keys(this.current_pivot_points).length === 0) { log('WARNING', `${log_prefix} Cannot replenish, no pivots available.`); return; }
            pivots = { ...this.current_pivot_points };
        } finally { releaseState(); }

        const new_side = filled_side === 'buy' ? 'sell' : 'buy';
        const all_pivot_levels = Object.entries(pivots).filter(([,p]) => p instanceof Decimal && p.isFinite() && p.gt(ZERO));
        if (all_pivot_levels.length === 0) { log('WARNING', `${log_prefix} No valid pivot levels found.`); return; }

        try {
            let active_prices_for_new_side;
            const releaseOrder = await this.order_lock.acquire();
            try {
                const price_map = new_side === 'buy' ? this.active_buy_prices : this.active_sell_prices;
                active_prices_for_new_side = new Set(Object.keys(price_map));
            } finally { releaseOrder(); }

            const candidates = [];
            for (const [key, pivot_price_raw] of all_pivot_levels) {
                const fmtPivotPriceSelect = this._format_price(pivot_price_raw, Decimal.ROUND_NEAREST);
                if (!fmtPivotPriceSelect || fmtPivotPriceSelect.lte(ZERO)) continue;
                const placeRounding = new_side === 'buy' ? Decimal.ROUND_DOWN : Decimal.ROUND_UP;
                const fmtPivotPricePlace = this._format_price(fmtPivotPriceSelect, placeRounding);
                if (!fmtPivotPricePlace || fmtPivotPricePlace.lte(ZERO)) continue;
                const fmtPivotPricePlaceStr = formatDecimalString(fmtPivotPricePlace);
                if (active_prices_for_new_side.has(fmtPivotPricePlaceStr)) continue;
                const is_correct_side = (new_side === 'sell' && fmtPivotPriceSelect.gt(filled_price_decimal)) || (new_side === 'buy' && fmtPivotPriceSelect.lt(filled_price_decimal));
                if (is_correct_side) {
                    candidates.push({ distance: fmtPivotPriceSelect.sub(filled_price_decimal).abs(), price_select: fmtPivotPriceSelect, price_place: fmtPivotPricePlace, name: key });
                }
            }

            if (candidates.length === 0) { log('INFO', `${log_prefix} No suitable unoccupied pivot level found for ${new_side} order.`); return; }

            candidates.sort((a, b) => a.distance.comparedTo(b.distance));
            const target_level = candidates[0];
            const target_price_place = target_level.price_place;
            log('INFO', `${log_prefix} Selected nearest available pivot for ${new_side}: ${target_level.name} @ ${formatDecimalString(target_price_place, 4)}`);

            const base_order_size_usd = safeDecimal(config.order_size_usd, null);
            if (!base_order_size_usd || base_order_size_usd.lte(ZERO)) { log('ERROR', `${log_prefix} Invalid order size USD for replenishment.`); return; }
            const base_amount = base_order_size_usd.div(target_price_place);
            await this._place_order(new_side, target_price_place, base_amount);

        } catch (e) { log('ERROR', `${log_prefix} Error during replenishment logic: ${e.message}`, e); }
     }
    async _check_and_reset_grid_center() { /* ... Full implementation ... */
        const reset_thresh_percent = safeDecimal(config.grid_center_reset_threshold_percent, ZERO);
        if (reset_thresh_percent.lte(ZERO)) return;

        let center_px; let grid_has_active_orders;
        const releaseState = await this.state_lock.acquire();
        try { center_px = this.grid_center_price; } finally { releaseState(); }
        const releaseOrder = await this.order_lock.acquire();
        try { grid_has_active_orders = Object.keys(this.active_orders).length > 0; } finally { releaseOrder(); }

        if (!grid_has_active_orders) { log('DEBUG', "Grid center reset check skipped: No active orders."); return; }
        if (!this.current_price || !this.current_price.isFinite() || this.current_price.lte(ZERO)) { log('DEBUG', `Grid center reset check skipped: Invalid current price.`); return; }
        if (!center_px || !center_px.isFinite() || center_px.lte(ZERO)) { log('DEBUG', `Grid center reset check skipped: Invalid grid center price.`); return; }
        if (Date.now() / 1000 - this.last_price_update_time > 180) { log('WARNING', `Grid center reset check skipped: Price data stale.`); return; }

        try {
            const deviation = this.current_price.sub(center_px).abs();
            const threshold_amount = center_px.mul(reset_thresh_percent.div(100));
            log('DEBUG', `Grid Center Check: CurrentPx=${formatDecimalString(this.current_price, 4)}, CenterPx=${formatDecimalString(center_px, 4)}, Dev=${formatDecimalString(deviation, 4)}, Threshold=${formatDecimalString(threshold_amount, 4)} (${formatDecimalString(reset_thresh_percent)}%)`);

            if (deviation.gt(threshold_amount)) {
                log('WARNING', `${nc.bold(nc.yellow('!!! GRID RESET TRIGGERED !!!'))} Price deviation exceeds threshold.`);
                log('INFO', "Recalculating pivots and resetting grid...");
                const reset_success = await this.place_initial_grid(); // Uses init_lock
                if (!reset_success) {
                    log('ERROR', `${nc.red("Grid reset attempt failed to place any orders!")}`);
                     if (config.send_sms_alerts && config.sms_alert_on_error) {
                         sendSmsAlert(`GridBot CRITICAL: Grid reset failed for ${config.symbol}.`).catch(smsErr => log('ERROR', `SMS Alert Error: ${smsErr.message}`));
                     }
                } else log('INFO', `${nc.green("Grid reset completed successfully.")}`);
            }
        } catch (e) { log('ERROR', `Error during grid center check and reset logic: ${e.message}`, e); }
     }

    // --- PNL Tracking --- (Implementations from previous response are complete)
    async _track_pnl(filled_order_data) { /* ... Full implementation ... */
        const oid = filled_order_data?.id;
        const trade_id = filled_order_data?.trades?.[0]?.id;
        const fill_timestamp_ms = filled_order_data?.lastTradeTimestamp || filled_order_data?.timestamp || Date.now();
        const fill_timestamp_sec = Math.floor(fill_timestamp_ms / 1000);
        // Use trade ID if available, fallback to order ID + timestamp + random suffix for uniqueness
        const pair_key = trade_id ? `trade_${trade_id}` : `fill_${oid}_${fill_timestamp_sec}_${Math.random().toString(36).substring(2, 7)}`;

        if (!oid) { log('WARNING', "PNL Track skipped: Missing order ID."); return; }
        const side = filled_order_data.side?.toLowerCase();
        if (side !== 'buy' && side !== 'sell') { log('WARNING', `PNL Track (${oid.slice(-6)}): Invalid side '${side}'.`); return; }

        const avg_px = safeDecimal(filled_order_data.average, null);
        const limit_px = safeDecimal(filled_order_data.price, null);
        let filled_price = (avg_px && avg_px.gt(ZERO)) ? avg_px : limit_px;
        if (!filled_price || !filled_price.isFinite() || filled_price.lte(ZERO)) { log('WARNING', `PNL Track (${pair_key.slice(-10)}): Invalid fill price.`); return; }

        const filled_amt = safeDecimal(filled_order_data.filled, null);
        if (!filled_amt || !filled_amt.isFinite() || filled_amt.lte(MIN_ORDER_AMOUNT_THRESHOLD_DEC)) { log('WARNING', `PNL Track (${pair_key.slice(-10)}): Invalid fill amount ${filled_amt?.toString()}.`); return; }

        const trade_fee = filled_order_data?.trades?.[0]?.fee;
        const fee_cost_raw = trade_fee?.cost ?? filled_order_data.fee?.cost ?? filled_order_data.fees?.[0]?.cost;
        const fee_cost = safeDecimal(fee_cost_raw, ZERO);
        const fee_curr = trade_fee?.currency ?? filled_order_data.fee?.currency ?? filled_order_data.fees?.[0]?.currency;

        log('INFO', `Processing Fill for PNL: Key=${pair_key.slice(-10)} OID=${oid.slice(-6)} Side=${side.toUpperCase()} Amt=${formatDecimalString(filled_amt, 6)} @ ${formatDecimalString(filled_price, 4)} Fee=${formatDecimalString(fee_cost, 6)} ${fee_curr || 'N/A'}`);

        const settle_curr = this.market_info?.settle;
        if (fee_curr && settle_curr && fee_curr !== settle_curr) { log('WARNING', `PNL Track (${pair_key.slice(-10)}): Fee currency (${fee_curr}) != Settle currency (${settle_curr}). PNL may be inaccurate.`); }

        const releasePnl = await this.pnl_lock.acquire();
        try {
            if (this.trade_pairs[pair_key]) { log('WARNING', `PNL Track (${oid.slice(-6)}): Duplicate fill key '${pair_key}' detected. Skipping.`); return; }

            if (side === 'buy') {
                this.trade_pairs[pair_key] = { price: filled_price, amount: filled_amt, fee: fee_cost, timestamp: fill_timestamp_sec, remaining_amount: filled_amt };
                log('DEBUG', `Added buy leg ${pair_key.slice(-10)}. Total legs: ${Object.keys(this.trade_pairs).filter(k => this.trade_pairs[k].remaining_amount).length}`);
            } else if (side === 'sell') {
                let sell_amount_to_match = filled_amt;
                let pnl_from_this_sell = ZERO;
                const matched_buy_keys = []; const buy_legs_to_remove = []; const buy_legs_to_update = {};

                const sorted_buy_legs = Object.entries(this.trade_pairs)
                    .filter(([, leg]) => leg.remaining_amount && leg.remaining_amount.gt(MIN_ORDER_AMOUNT_THRESHOLD_DEC))
                    .sort(([, a], [, b]) => (a.timestamp || 0) - (b.timestamp || 0));

                if (sorted_buy_legs.length === 0) {
                    log('WARNING', `PNL Track (${pair_key.slice(-10)}): Sell fill occurred, but no available buy legs found.`);
                    this.trade_pairs[pair_key] = { processed_sell: true, pnl: ZERO, timestamp: fill_timestamp_sec };
                    return;
                }

                for (const [buy_key, buy_info] of sorted_buy_legs) {
                    if (sell_amount_to_match.lte(MIN_ORDER_AMOUNT_THRESHOLD_DEC)) break;
                    const match_amount = Decimal.min(sell_amount_to_match, buy_info.remaining_amount);
                    if (match_amount.gt(MIN_ORDER_AMOUNT_THRESHOLD_DEC)) {
                        matched_buy_keys.push(buy_key.slice(-10));
                        const pnl_gross = filled_price.sub(buy_info.price).mul(match_amount);
                        const sell_fee_prop = filled_amt.gt(ZERO) ? fee_cost.mul(match_amount).div(filled_amt) : ZERO;
                        const buy_fee_prop = buy_info.amount.gt(ZERO) ? buy_info.fee.mul(match_amount).div(buy_info.amount) : ZERO;
                        const fees_match = sell_fee_prop.add(buy_fee_prop);
                        const pnl_net_match = pnl_gross.sub(fees_match);
                        pnl_from_this_sell = pnl_from_this_sell.add(pnl_net_match);
                        log('DEBUG', `PNL Match: Sell(${formatDecimalString(match_amount, 6)}) vs Buy ${buy_key.slice(-10)}. Net=${formatDecimalString(pnl_net_match, 6)}`);
                        sell_amount_to_match = sell_amount_to_match.sub(match_amount);
                        const new_buy_rem = buy_info.remaining_amount.sub(match_amount);
                        if (new_buy_rem.lte(MIN_ORDER_AMOUNT_THRESHOLD_DEC)) buy_legs_to_remove.push(buy_key);
                        else buy_legs_to_update[buy_key] = new_buy_rem;
                    }
                }

                if (!pnl_from_this_sell.isZero()) {
                     const old_r = this.realized_pnl; this.realized_pnl = old_r.add(pnl_from_this_sell);
                     log('INFO', `Realized PNL Updated: ${formatDecimalString(old_r, 4)} -> ${nc.bold(formatDecimalString(this.realized_pnl, 4))} (${pnl_from_this_sell.isPositive() ? '+' : ''}${formatDecimalString(pnl_from_this_sell, 4)})`);
                }
                for (const k of buy_legs_to_remove) delete this.trade_pairs[k];
                for (const [k, rem] of Object.entries(buy_legs_to_update)) if (this.trade_pairs[k]) this.trade_pairs[k].remaining_amount = rem;
                this.trade_pairs[pair_key] = { processed_sell: true, pnl: pnl_from_this_sell, timestamp: fill_timestamp_sec };
                if (sell_amount_to_match.gt(MIN_ORDER_AMOUNT_THRESHOLD_DEC)) log('WARNING', `PNL Track (${pair_key.slice(-10)}): ${formatDecimalString(sell_amount_to_match, 6)} sell amount remained unmatched.`);
            }
        } catch (error) { log('ERROR', `Error during PNL tracking for fill ${pair_key}: ${error.message}`, error);
        } finally { releasePnl(); }
     }
    async log_pnl_status() { /* ... Full implementation ... */
        try {
             let rpnl, open_buy_legs_count;
             const releasePnl = await this.pnl_lock.acquire();
             try {
                 rpnl = this.realized_pnl;
                 open_buy_legs_count = Object.values(this.trade_pairs).filter(p => p.remaining_amount?.gt(MIN_ORDER_AMOUNT_THRESHOLD_DEC)).length;
             } finally { releasePnl(); }

             const upnl = this.unrealized_pnl; const pos_size = this.position_size; const entry_px = this.entry_price;
             const total_pnl = rpnl.add(upnl);
             const current_px_str = this.current_price ? formatDecimalString(this.current_price, 4) : "N/A";
             const entry_px_str = (!pos_size.isZero() && entry_px.gt(ZERO)) ? formatDecimalString(entry_px, 4) : "Flat";
             const pos_color = pos_size.gt(ZERO) ? nc.green : pos_size.lt(ZERO) ? nc.red : nc.white;
             const pos_str = `${pos_color(formatDecimalString(pos_size, 4))}${nc.reset}`;
             const rpnl_color = rpnl.gte(ZERO) ? nc.green : nc.red;
             const rpnl_str = `${rpnl_color(formatDecimalString(rpnl, 4, '+'))}${nc.reset}`;
             const upnl_color = upnl.gte(ZERO) ? nc.green : nc.red;
             const upnl_str = (!pos_size.isZero()) ? `${upnl_color(formatDecimalString(upnl, 4, '+'))}${nc.reset}` : `${nc.gray('0.0000')}`;
             const total_pnl_color = total_pnl.gte(ZERO) ? nc.green : nc.red;
             const total_pnl_str = `${total_pnl_color(formatDecimalString(total_pnl, 4, '+'))}${nc.reset}`;
             let pnl_percent_str = " (Initial Bal N/A)";
             if (this.initial_balance_usd?.gt(ZERO)) {
                 const pnl_percent = total_pnl.div(this.initial_balance_usd).mul(100);
                 pnl_percent_str = ` (${total_pnl_color(formatDecimalString(pnl_percent, 2, '+') + '%')}${nc.reset})`;
             }
             let open_buy_orders, open_sell_orders;
             const releaseOrder = await this.order_lock.acquire();
             try { open_buy_orders = Object.keys(this.active_buy_prices).length; open_sell_orders = Object.keys(this.active_sell_prices).length; } finally { releaseOrder(); }
             log('INFO', `${nc.bold('Status:')} Px=${nc.cyan(current_px_str)}, Pos=${pos_str}@${nc.cyan(entry_px_str)}, RPNL=${rpnl_str}, UPNL=${upnl_str}, Total=${total_pnl_str}${pnl_percent_str} | Ord(B/S):${nc.green(open_buy_orders)}/${nc.red(open_sell_orders)} | Legs:${nc.blue(open_buy_legs_count)}`);
        } catch (error) { log('ERROR', `Error logging PNL status: ${error.message}`, error); }
     }


    // --- Risk Management / Shutdown ---
    async _check_pnl_limits() { /* ... Full implementation ... */
        if (!config.enable_pnl_limit_checks) return false;
        if (!this.initial_balance_usd?.gt(ZERO)) { log('WARNING', "PNL limit check skipped: Initial balance N/A."); return false; }

        try {
            let rpnl; const releasePnl = await this.pnl_lock.acquire(); try { rpnl = this.realized_pnl; } finally { releasePnl(); }
            const upnl = this.unrealized_pnl; const total_pnl = rpnl.add(upnl);
            const total_pnl_percent = total_pnl.div(this.initial_balance_usd).mul(100);
            const profit_target_percent = safeDecimal(config.grid_total_profit_target_percent, ZERO);
            const loss_limit_percent = safeDecimal(config.grid_total_loss_limit_percent, ZERO); // Positive value

            let limit_hit = false; let reason = "";
            if (profit_target_percent.gt(ZERO) && total_pnl_percent.gte(profit_target_percent)) {
                reason = "Profit Target";
                log('WARNING', `${nc.green.bold(`--- ${reason.toUpperCase()} REACHED ---`)} Total PNL ${formatDecimalString(total_pnl, 4, '+')} (${formatDecimalString(total_pnl_percent, 2, '+')}%) >= Target ${formatDecimalString(profit_target_percent)}%.`);
                limit_hit = true;
            } else if (loss_limit_percent.gt(ZERO) && total_pnl_percent.lte(loss_limit_percent.negated())) {
                reason = "Stop Loss";
                log('ERROR', `${nc.red.bold(`--- ${reason.toUpperCase()} TRIGGERED ---`)} Total PNL ${formatDecimalString(total_pnl, 4, '+')} (${formatDecimalString(total_pnl_percent, 2, '+')}%) <= Limit -${formatDecimalString(loss_limit_percent)}%.`);
                limit_hit = true;
            }
            if (limit_hit) {
                if (config.send_sms_alerts && config.sms_alert_on_pnl_limit) {
                    const alert_msg = `PNL Limit Alert (${config.symbol}): ${reason} hit. Total PNL ${formatDecimalString(total_pnl_percent, 2, '+')}%. Initiating shutdown.`;
                    sendSmsAlert(alert_msg).catch(e => log('ERROR', `Failed SMS: ${e.message}`));
                }
                await this.shutdown(reason); // Trigger shutdown
                return true;
            }
            return false;
        } catch (e) { log('ERROR', `Error during PNL limit check: ${e.message}`, e); return false; }
     }
    async _close_position_and_orders() { /* ... Full implementation ... */
        log('WARNING', `${nc.yellow('Attempting to close open position and cancel all orders...')}`);
        let overall_success = true;

        if (config.cancel_orders_on_shutdown || config.cancel_orders_on_exit) {
            log('INFO', "Cancelling all open orders...");
            if (!await this._cancel_all_orders("Shutdown/Exit")) {
                log('ERROR', `${nc.red("Failed to cancel all orders during exit sequence!")}`);
                overall_success = false;
            } else {
                log('INFO', "Cancel orders request successful. Waiting briefly...");
                await new Promise(resolve => setTimeout(resolve, 1500));
            }
        } else log('INFO', "Skipping order cancellation (disabled in config).");

        if ((config.close_position_on_shutdown || config.close_position_on_exit) && !this.market_info?.spot) {
            log('INFO', "Attempting to close existing position...");
            if (!await this._fetch_position()) {
                 log('CRITICAL', `${nc.red.bold("Failed fetch position before close! MANUAL CHECK REQUIRED.")}`); return false;
            }
            const current_pos_size = this.position_size;
            log('INFO', `Current position size: ${formatDecimalString(current_pos_size, 4)}`);
            let min_close_amount = MIN_ORDER_AMOUNT_THRESHOLD_DEC;
            if (this.market_info?.limits?.amount?.min) min_close_amount = safeDecimal(this.market_info.limits.amount.min, min_close_amount);

            if (current_pos_size.abs().gte(min_close_amount)) {
                const side_to_close = current_pos_size.gt(ZERO) ? 'sell' : 'buy';
                const amount_to_close = current_pos_size.abs();
                const formatted_amount_to_close = this._format_amount(amount_to_close);
                log('WARNING', `Placing MARKET ${side_to_close.toUpperCase()} order to close ${formatDecimalString(formatted_amount_to_close, 6)} ${this.market_info?.base || ''}`);
                if (formatted_amount_to_close.lt(min_close_amount)) {
                     log('ERROR', `${nc.red.bold(`Formatted close amount too small. CANNOT AUTO-CLOSE!`)}`); return false;
                }
                try {
                    const params = { reduceOnly: true };
                    const order = await this._execute_ccxt_rest_call('createOrder', config.symbol, 'market', side_to_close, parseFloat(formatted_amount_to_close.toString()), undefined, params);
                    if (order?.id) {
                        log('INFO', `Market close order placed (ID: ${order.id.slice(-6)}). Waiting for confirmation...`);
                        const max_wait_ms = (config.market_close_max_wait_seconds || 60) * 1000;
                        const start_time = Date.now(); let position_confirmed_closed = false;
                        while (Date.now() - start_time < max_wait_ms) {
                            await new Promise(resolve => setTimeout(resolve, (config.market_close_order_wait_seconds || 10) * 1000));
                            if (this.shutdown_triggered && (Date.now() - start_time > 20000)) { log('WARNING', 'Shutdown during position close wait.'); break; }
                            if (await this._fetch_position() && this.position_size.abs().lt(min_close_amount)) {
                                log('INFO', `${nc.green('Position successfully confirmed closed.')}`);
                                this.position_size = ZERO; this.entry_price = ZERO; this.unrealized_pnl = ZERO;
                                position_confirmed_closed = true; break;
                            } else log('INFO', `Position still open. Size: ${formatDecimalString(this.position_size, 6)}. Waiting...`);
                        }
                        if (!position_confirmed_closed && !this.shutdown_triggered) {
                            log('ERROR', `${nc.red.bold(`Position close NOT confirmed! MANUAL VERIFICATION REQUIRED!`)}`);
                            overall_success = false;
                        }
                    } else { log('ERROR', `${nc.red.bold(`Market close order FAILED. MANUAL CLOSE REQUIRED!`)}`); overall_success = false; }
                } catch (e) { log('ERROR', `${nc.red.bold(`Error during market close: ${e.message}. MANUAL CLOSE REQUIRED!`)}`, e); overall_success = false; }
            } else log('INFO', "No significant position found to close.");
        } else if (this.market_info?.spot) log('INFO', "Skipping position close (Spot market).");
        else log('INFO', "Skipping position close on exit/shutdown (disabled in config).");

        log('WARNING', `Position/Order close sequence finished. Overall Success: ${overall_success}`);
        return overall_success;
     }
    async _safe_exchange_close() { /* ... Full implementation ... */
        if (this.exchange && typeof this.exchange.close === 'function') {
            try {
                log('INFO', `Closing CCXT exchange connection for ${this.exchange.id}...`);
                await this.exchange.close(); // Handles WS closure primarily
                log('INFO', "Exchange connection close requested.");
            } catch (e) {
                log('ERROR', `Error closing exchange connection: ${e.message}`, e);
            }
        }
        this.exchange = null; // Clear reference
        this.websocket_connected_ticker = false;
        this.websocket_connected_orders = false;
     }

    // --- Shutdown --- (Delegates to handler)
    async shutdown(reason = "Signal Received") {
        await this.shutdownHandler.shutdown(reason);
    }

    // --- WebSocket Handlers --- (Implementations from previous response are complete)
    async _handle_ticker_update(ticker) { /* ... Full implementation ... */
        try {
            const last_price = safeDecimal(ticker?.last, null);
            let updated_price = null;

            if (last_price?.gt(ZERO)) updated_price = last_price;
            else if (ticker?.close) {
                const close_price = safeDecimal(ticker.close, null);
                if (close_price?.gt(ZERO)) { updated_price = close_price; log('DEBUG', `Ticker WS: Using 'close' price.`); }
            }

            if (updated_price) {
                const price_changed = !this.current_price || !this.current_price.equals(updated_price);
                this.current_price = updated_price;
                this.last_price_update_time = (ticker?.timestamp ? Math.floor(ticker.timestamp / 1000) : Math.floor(Date.now() / 1000));
                if (price_changed) log('DEBUG', `Price Update (Ticker WS): ${formatDecimalString(this.current_price, 4)}`);
            } else if (ticker) log('WARNING', `Ticker WS: Received update but could not extract valid price.`);

            if (!this.websocket_connected_ticker) {
                this.websocket_connected_ticker = true; log('INFO', `${nc.green('Ticker WebSocket receiving data successfully.')}`);
            }
        } catch (e) { log('ERROR', `Error handling ticker update: ${e.message}`, e); }
     }
    async _handle_order_update(order_data) { /* ... Full implementation ... */
        const oid = order_data?.id;
        if (!oid) { log('WARNING', "WS Order Update: Received order data without ID."); return; }
        const status = order_data.status?.toLowerCase(); const side = order_data.side?.toLowerCase();
        const filled_amount_str = order_data.filled || '0'; const avg_price_str = order_data.average || 'N/A';
        log('DEBUG', `WS Order Update: ID=${oid.slice(-6)}, Status=${status}, Side=${side || 'N/A'}, Filled=${filled_amount_str}, AvgPx=${avg_price_str}`);

        const releaseOrder = await this.order_lock.acquire();
        let tempReleaseOrder = releaseOrder;
        try {
            const local_order = this.active_orders[oid];
            if (!local_order && !['closed', 'filled', 'canceled', 'cancelled', 'expired', 'rejected'].includes(status)) { log('DEBUG', `WS Order Update: Ignoring update for untracked non-terminal order ${oid.slice(-6)}.`); return; }
            if (!local_order && ['closed', 'filled', 'canceled', 'cancelled', 'expired', 'rejected'].includes(status)) { log('DEBUG', `WS Order Update: Received terminal update for untracked order ${oid.slice(-6)}. Likely already processed.`); return; }
            if (!local_order) { log('ERROR', `WS Order Update: Logic error - local order for ${oid.slice(-6)} missing.`); return; }

            let order_removed = false; let trigger_replenish = false;
            let filled_price_for_replenish = null; let pnl_data_to_process = null;

            if (status === 'closed' || status === 'filled') {
                log('INFO', `${nc.bold(nc.brightGreen('<<< ORDER FILLED >>>'))} ID=${oid.slice(-6)} Side=${local_order.side.toUpperCase()} Amt=${formatDecimalString(safeDecimal(filled_amount_str), 6)} @ AvgPx=${formatDecimalString(safeDecimal(avg_price_str), 4)}`);
                pnl_data_to_process = { ...order_data }; // Process full fill
                filled_price_for_replenish = safeDecimal(order_data.average, null) || safeDecimal(order_data.price, null);
                if (filled_price_for_replenish?.gt(ZERO)) trigger_replenish = true;
                else log('ERROR', `Cannot get valid fill price for replenishment ${oid.slice(-6)}.`);
                const price_str = local_order.price; const order_side = local_order.side;
                if (order_side === 'buy' && this.active_buy_prices[price_str] === oid) delete this.active_buy_prices[price_str];
                if (order_side === 'sell' && this.active_sell_prices[price_str] === oid) delete this.active_sell_prices[price_str];
                delete this.active_orders[oid]; order_removed = true; log('DEBUG', `Removed filled order ${oid.slice(-6)}.`);
            } else if (status === 'canceled' || status === 'cancelled' || status === 'expired' || status === 'rejected') {
                log('INFO', `${nc.yellow(`--- ORDER ${status.toUpperCase()} ---`)} ID=${oid.slice(-6)} Side=${local_order.side.toUpperCase()} Price=${local_order.price}`);
                const price_str = local_order.price; const order_side = local_order.side;
                if (order_side === 'buy' && this.active_buy_prices[price_str] === oid) delete this.active_buy_prices[price_str];
                if (order_side === 'sell' && this.active_sell_prices[price_str] === oid) delete this.active_sell_prices[price_str];
                delete this.active_orders[oid]; order_removed = true; log('DEBUG', `Removed ${status} order ${oid.slice(-6)}.`);
            } else if (status === 'open' || status === 'new' || status === 'partially_filled') {
                log('DEBUG', `WS Order Update: Status update for ${oid.slice(-6)} to '${status}'. Filled: ${filled_amount_str}`);
                local_order.status = status; local_order.timestamp = (order_data.lastTradeTimestamp || order_data.timestamp || Date.now()) / 1000;
                if (status === 'partially_filled' && order_data.trades?.length > 0) {
                     const last_trade = order_data.trades[order_data.trades.length-1];
                     log('INFO', `${nc.brightBlue('--- ORDER PARTIALLY FILLED ---')} ID=${oid.slice(-6)} Side=${local_order.side.toUpperCase()}. Last Fill Amt: ${formatDecimalString(safeDecimal(last_trade.amount), 6)} @ ${formatDecimalString(safeDecimal(last_trade.price), 4)}`);
                     pnl_data_to_process = [];
                     for (const trade of order_data.trades) {
                         const tradeAmt = safeDecimal(trade.amount, ZERO); const tradePx = safeDecimal(trade.price, ZERO);
                         if (tradeAmt.gt(ZERO) && tradePx.gt(ZERO)) {
                             pnl_data_to_process.push({ symbol: order_data.symbol, side: order_data.side, price: order_data.price, id: oid, trades: [trade], filled: trade.amount, average: trade.price, fee: trade.fee, timestamp: trade.timestamp, lastTradeTimestamp: trade.timestamp });
                         } else log('WARNING', `Skipping invalid trade in partial fill update: ${JSON.stringify(trade)}`);
                     }
                     if (pnl_data_to_process.length === 0) pnl_data_to_process = null;
                }
            } else log('WARNING', `WS Order Update: Unhandled status '${status}' for ${oid.slice(-6)}.`);

            tempReleaseOrder(); tempReleaseOrder = null; // Release lock before async calls

            if (pnl_data_to_process) {
                const processPnl = async (data) => { try { await this._track_pnl(data); } catch (e) { log('ERROR', `BG PNL Error ${oid}: ${e.message}`, e); }};
                if (Array.isArray(pnl_data_to_process)) pnl_data_to_process.forEach(d => processPnl(d)); else processPnl(pnl_data_to_process);
            }
            if (trigger_replenish && filled_price_for_replenish) {
                 this._replenish_grid_level(local_order.side, filled_price_for_replenish).catch(e => log('ERROR', `BG Replenish Error ${oid}: ${e.message}`, e));
            }
            if (order_removed && config.save_state_on_order_close) {
                this._save_state().catch(e => log('ERROR', `State save error after order update: ${e.message}`));
            }
        } catch (error) { log('ERROR', `Error handling order update ${oid}: ${error.message}`, error);
        } finally { if (tempReleaseOrder && this.order_lock.isLocked()) tempReleaseOrder(); }
     }

    // --- Main Loop Components ---

    /** Main loop for watching ticker WebSocket stream. */
    async _watch_tickers_loop() {
        log('INFO', "Starting Ticker WebSocket Loop...");
        while (this.bot_running && !this.shutdown_triggered) {
            if (!this.exchange?.has['watchTicker']) {
                log('ERROR', `watchTicker not supported. Cannot get real-time prices via WS.`);
                await new Promise(resolve => setTimeout(resolve, 300000)); // Wait 5 mins and retry check
                continue;
            }
            if (!this.websocket_connected_ticker) log('INFO', "Ticker WS: Attempting connection...");
            try {
                while (this.bot_running && !this.shutdown_triggered) {
                    const ticker = await this.exchange.watchTicker(config.symbol);
                    if (!this.bot_running || this.shutdown_triggered) break;
                    await this._handle_ticker_update(ticker);
                }
            } catch (e) {
                log('ERROR', `Ticker WS Error: ${e.message}. Retrying connection...`, e);
                this.websocket_connected_ticker = false;
                await new Promise(resolve => setTimeout(resolve, config.websocket_retry_delay_seconds * 1000 || 15000));
            }
        }
        log('INFO', "Ticker WebSocket loop stopped.");
    }

    /** Main loop for watching order WebSocket stream. */
    async _watch_orders_loop() {
        log('INFO', "Starting Orders WebSocket Loop...");
        while (this.bot_running && !this.shutdown_triggered) {
             if (!this.exchange?.has['watchOrders']) {
                 log('ERROR', `watchOrders not supported. Order updates rely on REST polling.`);
                 await new Promise(resolve => setTimeout(resolve, 300000)); // Wait 5 mins
                 continue;
             }
             if (!this.websocket_connected_orders) log('INFO', "Orders WS: Attempting connection...");
             try {
                 while (this.bot_running && !this.shutdown_triggered) {
                     const orders = await this.exchange.watchOrders(config.symbol);
                     if (!this.bot_running || this.shutdown_triggered) break;
                     if (!this.websocket_connected_orders) { this.websocket_connected_orders = true; log('INFO', `${nc.green('Orders WebSocket receiving data.')}`); }
                     if (Array.isArray(orders)) {
                          const results = await Promise.allSettled(orders.map(order => this._handle_order_update(order)));
                          results.forEach((result, index) => { if (result.status === 'rejected') log('ERROR', `Error processing WS order ${orders[index]?.id}: ${result.reason}`, result.reason); });
                     }
                 }
             } catch (e) {
                 log('ERROR', `Orders WS Error: ${e.message}. Retrying connection...`, e);
                 this.websocket_connected_orders = false;
                 await new Promise(resolve => setTimeout(resolve, config.websocket_retry_delay_seconds * 1000 || 15000));
             }
        }
        log('INFO', "Orders WebSocket loop stopped.");
    }

    /** Main processing loop of the bot. */
    async run_main_loop() {
        log('INFO', "Starting main bot loop...");
        let consecutive_errors = 0;
        const error_threshold = 5;

        while (this.bot_running && !this.shutdown_triggered) {
            try {
                const loop_start_time = Date.now();
                const loop_start_sec = Math.floor(loop_start_time / 1000);

                // Periodic Checks
                if (loop_start_sec % config.balance_fetch_interval_seconds === 0) await this._fetch_balance();
                if (loop_start_sec % config.position_fetch_interval_seconds === 0 && !this.market_info?.spot) await this._fetch_position();
                if (loop_start_sec % config.analysis_interval_seconds === 0) await this._calculate_analysis_data();
                const pivot_interval_s = this.exchange.parseTimeframe(config.daily_kline_interval || '1d');
                const pivot_check_interval_s = Math.max(3600, pivot_interval_s / 2);
                if (loop_start_sec % pivot_check_interval_s === 0) {
                     const now_s = Date.now() / 1000;
                     if (!this.last_pivot_calculation_time || (now_s - this.last_pivot_calculation_time) > pivot_interval_s) {
                         log('INFO', 'Recalculating pivots due to interval...'); await this._calculate_fib_pivot_points();
                     }
                 }
                if (loop_start_sec % config.grid_reset_check_interval_seconds === 0) await this._check_and_reset_grid_center();
                if (loop_start_sec % config.grid_health_check_interval_seconds === 0) { /* ... Grid Health Check Logic ... */
                     const releaseOrderCheck = await this.order_lock.acquire();
                     const active_buys_count = Object.keys(this.active_buy_prices).length; const active_sells_count = Object.keys(this.active_sell_prices).length;
                     releaseOrderCheck();
                     const target_orders = config.max_open_orders_per_side;
                     const threshold = Math.max(1, Math.floor(target_orders * 0.5));
                     if (target_orders > 1 && (active_buys_count < threshold || active_sells_count < threshold)) {
                          log('WARNING', `Grid health check: Underpopulated (B:${active_buys_count}/${target_orders}, S:${active_sells_count}/${target_orders}). Triggering reset...`);
                          await this.place_initial_grid();
                     } else log('DEBUG', `Grid health check OK.`);
                 }
                 if (config.enable_pnl_limit_checks && loop_start_sec % config.pnl_limit_check_interval_seconds === 0) await this._check_pnl_limits();
                 if (loop_start_sec % config.log_pnl_interval_seconds === 0) await this.log_pnl_status();
                 if (loop_start_sec % config.save_state_interval_seconds === 0) await this._save_state();

                consecutive_errors = 0; // Reset on success

                // Loop Delay
                const loop_duration_ms = Date.now() - loop_start_time;
                const delay_ms = Math.max(0, (config.main_loop_interval_seconds * 1000) - loop_duration_ms);
                if (delay_ms > 0) await new Promise(resolve => setTimeout(resolve, delay_ms));
                else log('WARNING', `Main loop iteration took longer than interval: ${loop_duration_ms}ms.`);

            } catch (error) {
                consecutive_errors++;
                log('ERROR', `Error in main bot loop (Attempt ${consecutive_errors}/${error_threshold}): ${error.message}`, error);
                if (consecutive_errors >= error_threshold) {
                     if (config.send_sms_alerts && config.sms_alert_on_error) sendSmsAlert(`GridBot Loop Error: ${config.symbol} failed ${consecutive_errors} times. Pausing. Err: ${error.message.substring(0,100)}`).catch(smsErr => log('ERROR', `SMS Alert Error: ${smsErr.message}`));
                     const pause_seconds = 60; log('ERROR', `Pausing main loop for ${pause_seconds} seconds.`);
                     await new Promise(resolve => setTimeout(resolve, pause_seconds * 1000));
                     consecutive_errors = 0; // Reset after pause
                } else await new Promise(resolve => setTimeout(resolve, 5000)); // Short delay after single error
            }
        }
        log('INFO', "Main bot loop terminated.");
        // Ensure shutdown cleanup runs if the loop terminates unexpectedly
        if (!this.shutdown_triggered) {
            await this.shutdown("Main Loop Unexpected Exit");
        }
    }


    // --- Entry Point ---
    /** Starts the grid bot operation. */
    async start() {
        log('INFO', `${nc.bold(nc.blue(`Starting Grid Bot for ${config.symbol} on ${config.exchange_id}`))}`);
        try {
            this.shutdownHandler.registerHandlers(); // Setup SIGINT/SIGTERM handling
            await this._load_state();
            if (!await this._initialize_exchange()) {
                throw new Error("Initialization failed"); // Throw error to trigger shutdown in catch block
            }
            await this._sync_open_orders_from_rest();

            const pivot_interval_s = this.exchange.parseTimeframe(config.daily_kline_interval || '1d');
            const now_s = Date.now() / 1000;
            if (!this.last_pivot_calculation_time || (now_s - this.last_pivot_calculation_time) > pivot_interval_s) {
                 log('INFO', "Calculating initial pivots..."); await this._calculate_fib_pivot_points();
            }

            let has_active_orders;
            const releaseOrderCheck = await this.order_lock.acquire();
            try { has_active_orders = Object.keys(this.active_orders).length > 0; } finally { releaseOrderCheck(); }

            if (!has_active_orders && config.place_initial_grid_on_startup) {
                 log('INFO', "Placing initial grid..."); await this.place_initial_grid();
            } else {
                 log('INFO', `Skipping initial grid placement (${has_active_orders ? 'orders exist' : 'disabled'}).`);
                 if (has_active_orders) { // Ensure center price exists if orders do
                     const releaseStateCheck = await this.state_lock.acquire();
                     try {
                         if (!this.grid_center_price || this.grid_center_price.lte(ZERO)) {
                            const center_fallback = this.current_pivot_points?.pp || this.current_price;
                            if (center_fallback?.gt(ZERO)) {
                                this.grid_center_price = center_fallback;
                                log('WARNING', `Grid center price missing, set to ${formatDecimalString(this.grid_center_price)}`);
                            } else log('ERROR', 'Cannot set initial grid center price.');
                         }
                     } finally { releaseStateCheck(); }
                 }
            }

            if (config.use_websockets) {
                log('INFO', "Starting WebSocket loops...");
                // Run concurrently, don't await
                this._watch_tickers_loop().catch(e => log('CRITICAL', 'Ticker WS loop failed critically', e));
                this._watch_orders_loop().catch(e => log('CRITICAL', 'Orders WS loop failed critically', e));
            } else log('INFO', "WebSockets disabled. Using REST polling.");

            await this.run_main_loop(); // This runs until bot_running is false

        } catch (error) {
            log('CRITICAL', `Unhandled error during bot startup sequence: ${error.message}`, error);
            await this.shutdown("Startup Error"); // Trigger graceful shutdown on startup error
        } finally {
            log('INFO', "Bot start() method finished execution.");
            // Ensure log file stream is closed if the process exits here (e.g., after shutdown)
            await closeLogStream();
        }
    }
}

export { GridBot };
