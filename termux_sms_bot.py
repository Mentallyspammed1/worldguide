# --- Add near imports ---
import subprocess
import platform # To check if running on Termux

# --- Add near configuration loading ---
ENABLE_TERMUX_SMS_ALERTS: bool = config.get("enable_termux_sms_alerts", False)
TERMUX_SMS_RECIPIENT: Optional[str] = config.get("termux_sms_recipient", None)
TERMUX_API_AVAILABLE: bool = False

# --- Check if Termux API is available ---
def check_termux_api():
    """Checks if the termux-sms-send command is likely available."""
    global TERMUX_API_AVAILABLE
    if platform.system() == "Linux" and "ANDROID_ROOT" in os.environ:
        try:
            # Try running a simple termux command like termux-info
            result = subprocess.run(['termux-info'], capture_output=True, text=True, check=False, timeout=5)
            # Check if it executed without error or with a known 'usage' error if no args given
            if result.returncode == 0 or "usage: termux-info" in result.stderr.lower():
                 # Check specifically for sms send capability (might require separate package)
                 result_sms = subprocess.run(['termux-sms-send', '--help'], capture_output=True, text=True, check=False, timeout=5)
                 if result_sms.returncode == 0 or "usage: termux-sms-send" in result_sms.stderr.lower():
                     log_info("Termux environment detected and termux-sms-send seems available.")
                     TERMUX_API_AVAILABLE = True
                     return True
                 else:
                     log_warning("Termux environment detected, but 'termux-sms-send' command failed. SMS alerts disabled. Install termux-api package?")
                     TERMUX_API_AVAILABLE = False
                     return False
            else:
                log_debug("Termux environment detected, but basic termux command failed. API likely unavailable.")
                TERMUX_API_AVAILABLE = False
                return False
        except FileNotFoundError:
            log_debug("Termux command not found. Not running in Termux or termux-api not installed.")
            TERMUX_API_AVAILABLE = False
            return False
        except subprocess.TimeoutExpired:
            log_warning("Timeout checking Termux API availability.")
            TERMUX_API_AVAILABLE = False
            return False
        except Exception as e:
            log_warning(f"Error checking Termux API: {e}")
            TERMUX_API_AVAILABLE = False
            return False
    else:
        # log_debug("Not detecting a Termux environment.")
        TERMUX_API_AVAILABLE = False
        return False

# Call the check function once during initialization
check_termux_api()

if ENABLE_TERMUX_SMS_ALERTS:
    if not TERMUX_API_AVAILABLE:
        log_error("Termux SMS alerts enabled in config, but Termux API (termux-sms-send) is not available or functional. Disabling SMS alerts.")
        ENABLE_TERMUX_SMS_ALERTS = False # Force disable
    elif not TERMUX_SMS_RECIPIENT:
        log_error("Termux SMS alerts enabled, but 'termux_sms_recipient' not set in config. Disabling SMS alerts.")
        ENABLE_TERMUX_SMS_ALERTS = False # Force disable
    else:
        log_info(f"Termux SMS Alerts ENABLED. Recipient: {TERMUX_SMS_RECIPIENT}")
else:
    log_info("Termux SMS Alerts DISABLED.")


# --- New Function: Send Termux SMS ---
def send_termux_sms(message: str, recipient: Optional[str] = TERMUX_SMS_RECIPIENT):
    """Sends an SMS using the Termux API if enabled and available."""
    if not ENABLE_TERMUX_SMS_ALERTS or not TERMUX_API_AVAILABLE or not recipient:
        # log_debug(f"SMS sending skipped (Enabled: {ENABLE_TERMUX_SMS_ALERTS}, API: {TERMUX_API_AVAILABLE}, Recipient: {recipient})")
        return

    # Basic message sanitization (optional, Termux might handle it)
    # message = message.replace('"', "'") # Replace double quotes

    # Limit message length (SMS limit is typically 160 chars, but be safe)
    max_len = 150
    if len(message) > max_len:
        message = message[:max_len-3] + "..."
        log_debug(f"Truncated SMS message to {max_len} chars.")

    command = ['termux-sms-send', '-n', recipient, message]
    log_debug(f"Executing Termux SMS command: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15) # Added timeout
        log_info(f"Termux SMS sent successfully to {recipient}.")
        log_debug(f"Termux SMS send stdout: {result.stdout}")
        log_debug(f"Termux SMS send stderr: {result.stderr}")
    except FileNotFoundError:
        log_error("Termux SMS failed: 'termux-sms-send' command not found. Disabling alerts.")
        global ENABLE_TERMUX_SMS_ALERTS, TERMUX_API_AVAILABLE
        ENABLE_TERMUX_SMS_ALERTS = False
        TERMUX_API_AVAILABLE = False
    except subprocess.CalledProcessError as e:
        log_error(f"Termux SMS failed: Command returned non-zero exit status {e.returncode}.")
        log_error(f"Stderr: {e.stderr}")
        log_error(f"Stdout: {e.stdout}")
        # Consider disabling alerts after repeated failures?
    except subprocess.TimeoutExpired:
        log_error("Termux SMS failed: Command timed out.")
    except Exception as e:
        log_error(f"Termux SMS failed with unexpected error: {e}", exc_info=True)
        # Consider disabling alerts


# --- Integrate send_termux_sms calls ---

# Example 1: Bot startup
# Inside the main script body, after initial config load and exchange connection attempt
log_info(f"Bot starting. Simulation Mode: {'ON' if SIMULATION_MODE else '!!! LIVE !!!'}")
send_termux_sms(f"TradingBot started ({symbol}). SIM: {'ON' if SIMULATION_MODE else 'LIVE!'}")

# Example 2: Entering a position
# Inside the `if long_entry_condition:` block, after successful entry order placement and SL/TP attempt:
                    if is_filled:
                        # ... (existing code to place SL/TP and update state) ...
                        save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits) # Show new status

                        # Send SMS Alert
                        sl_tp_status = f"SL/TP {'OK' if sl_order and tp_order else 'PARTIAL/FAIL'}"
                        sms_msg = (f"Trade Alert: ENTER LONG {symbol}\n"
                                   f"Qty: {filled_quantity:.{amount_precision_digits}f} @ ~{entry_price_actual:.{price_precision_digits}f}\n"
                                   f"SL: {stop_loss_price:.{price_precision_digits}f}, TP: {take_profit_price:.{price_precision_digits}f}\n"
                                   f"{sl_tp_status}")
                        send_termux_sms(sms_msg)

                        if not (sl_order and sl_order.get('id')) or not (tp_order and tp_order.get('id')):
                             log_warning("Entry successful, but SL and/or TP order placement failed or did not return ID. Monitor position closely!")
                             send_termux_sms(f"⚠️ ALERT: LONG {symbol} entry OK, but SL/TP placement failed! Manual check needed.") # Specific alert for failure

# Inside the `elif short_entry_condition:` block, after successful entry order placement and SL/TP attempt:
                    if is_filled:
                         # ... (existing code to place SL/TP and update state) ...
                        save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits)

                        # Send SMS Alert
                        sl_tp_status = f"SL/TP {'OK' if sl_order and tp_order else 'PARTIAL/FAIL'}"
                        sms_msg = (f"Trade Alert: ENTER SHORT {symbol}\n"
                                   f"Qty: {filled_quantity:.{amount_precision_digits}f} @ ~{entry_price_actual:.{price_precision_digits}f}\n"
                                   f"SL: {stop_loss_price:.{price_precision_digits}f}, TP: {take_profit_price:.{price_precision_digits}f}\n"
                                   f"{sl_tp_status}")
                        send_termux_sms(sms_msg)

                        if not (sl_order and sl_order.get('id')) or not (tp_order and tp_order.get('id')):
                             log_warning("Entry successful, but SL and/or TP order placement failed or did not return ID. Monitor position closely!")
                             send_termux_sms(f"⚠️ ALERT: SHORT {symbol} entry OK, but SL/TP placement failed! Manual check needed.")


# Example 3: Position closed (inside check_position_and_orders)
# Modify the section where position_reset_flag is set to True
        if position_reset_flag:
            # ... (existing code to cancel the other order) ...

            # Send SMS Alert about closure
            close_reason = assumed_close_reason or "Unknown" # SL or TP
            entry_p = position.get('entry_price', 0) # Get price before reset
            qty = position.get('quantity', 0)        # Get qty before reset
            pos_status = position.get('status', 'N/A') # Get status before reset

            # Basic PnL estimate (very rough, doesn't account for fees/slippage accurately)
            pnl_estimate = 0
            closed_price = 0 # We don't know the exact fill price here easily
            if close_reason == "SL": closed_price = position.get('stop_loss', 0) # Estimate using SL price
            elif close_reason == "TP": closed_price = position.get('take_profit', 0) # Estimate using TP price

            if pos_status == 'long' and entry_p > 0 and closed_price > 0 and qty > 0:
                 pnl_estimate = (closed_price - entry_p) * qty
            elif pos_status == 'short' and entry_p > 0 and closed_price > 0 and qty > 0:
                 pnl_estimate = (entry_p - closed_price) * qty

            pnl_str = f"~{pnl_estimate:.{price_precision_digits}f} {market.get('quote', '')}" if pnl_estimate != 0 else "N/A"

            sms_msg_close = (f"Trade Alert: CLOSE {pos_status.upper()} {symbol} via {close_reason}\n"
                             f"Entry: {entry_p:.{price_precision_digits}f}, Qty: {qty:.{amount_precision_digits}f}\n"
                             f"Est. PnL: {pnl_str}")
            send_termux_sms(sms_msg_close)

            log_info("Resetting local position state.")
            # ... (existing code to reset position state) ...
            save_position_state() # Save the reset state
            display_position_status(position, price_precision_digits, amount_precision_digits) # Show updated status
            return True # Indicate position was reset


# Example 4: Fallback Indicator Exit
# Inside the `if execute_indicator_exit:` block, after attempting market exit
                # Reset position state immediately...
                log_info("Resetting local position state after indicator-based market exit attempt.")
                pos_status_before_reset = position.get('status', 'N/A') # Get status before reset
                # ... (existing code to reset position) ...
                save_position_state()
                display_position_status(position, price_precision_digits, amount_precision_digits)

                # Send SMS for fallback exit
                exit_success_msg = "OK" if order_result and order_result.get('id') else "FAIL"
                sms_msg_fallback = (f"⚠️ ALERT: Fallback MKT EXIT {pos_status_before_reset.upper()} {symbol}\n"
                                    f"Reason: Indicator Signal\n"
                                    f"Order Status: {exit_success_msg}")
                send_termux_sms(sms_msg_fallback)

                # Exit loop for this cycle after attempting exit
                neon_sleep_timer(sleep_interval_seconds)
                continue # Go to next cycle immediately after exit attempt


# Example 5: Critical Error in Main Loop
# Inside the `except Exception as e:` block at the end of the main loop
    except Exception as e:
        log_error(f"CRITICAL unexpected error in main loop: {e}", exc_info=True)
        # Send SMS Alert for critical error
        error_type = type(e).__name__
        send_termux_sms(f"🚨 CRITICAL ERROR: TradingBot {symbol} encountered {error_type}. Check logs! Bot trying to recover.")
        log_info("Attempting to recover by saving state and waiting 60s before next cycle...")
        try:
            save_position_state() # Save state on critical error
        except Exception as save_e:
            log_error(f"Failed to save state during critical error handling: {save_e}", exc_info=True)
        neon_sleep_timer(60) # Wait before trying next cycle


# Example 6: Trailing Stop Update Failure
# Inside `update_trailing_stop`, if placing the new TSL order fails
                    else:
                        log_error(f"Failed to place new trailing SL order after cancelling old one {old_sl_id}. POSITION MAY BE UNPROTECTED.")
                        send_termux_sms(f"🚨 ALERT: Failed to place new TSL for {position['status'].upper()} {symbol}. Position UNPROTECTED!") # Send alert
                        position['sl_order_id'] = None # Mark SL as lost
                        position['current_trailing_sl_price'] = None
                        save_position_state()

                except Exception as place_e:
                    # Error logged by retry decorator or generic handler if non-retryable
                    log_error(f"Error placing new trailing SL order: {place_e}. POSITION MAY BE UNPROTECTED.", exc_info=True)
                    send_termux_sms(f"🚨 ALERT: Error placing TSL for {position['status'].upper()} {symbol}. Position UNPROTECTED!") # Send alert
                    position['sl_order_id'] = None
                    position['current_trailing_sl_price'] = None
                    save_position_state()


# Example 7: Configuration Reload (Simulation Mode Change)
# Inside `check_and_reload_config` function where SIMULATION_MODE is updated
                # Log changes for important parameters
                if new_sim_mode != SIMULATION_MODE:
                     log_warning(f"SIMULATION MODE changed to: {'ACTIVE' if new_sim_mode else '!!! LIVE TRADING !!!'}")
                     # Send SMS Alert for Simulation Mode Change
                     sim_status_msg = f"⚠️ CONFIG ALERT: SIMULATION MODE changed to: {'ON' if new_sim_mode else 'OFF (LIVE!)'} for {symbol}"
                     send_termux_sms(sim_status_msg) # Use the SMS function
                SIMULATION_MODE = new_sim_mode

# Example 8: Configuration Reload (Critical Change Requiring Restart)
# Inside `check_and_reload_config` function where exchange/symbol change is detected
                if new_exchange_id != exchange_id or new_symbol != symbol:
                    # Simplest approach: Log and exit, requiring manual restart
                    log_error(f"CRITICAL CHANGE DETECTED: Exchange ID or Symbol changed in config ('{exchange_id}'->'{new_exchange_id}', '{symbol}'->'{new_symbol}'). Restart required.")
                    # Optionally: Implement logic to close existing position and re-initialize exchange connection
                    # This is complex and risky, exiting might be safer.
                    send_termux_sms(f"🚨 CONFIG ALERT: Exchange/Symbol changed for {symbol}. Bot requires RESTART.") # Use the SMS function
                    sys.exit(1) # Force exit


# --- Ensure the check_and_reload_config logic exists and is called in the main loop ---
# (Keep the existing hot-reload code as provided in the previous step)

# --- Add Termux related parameters to the list of reloaded globals in check_and_reload_config ---
def check_and_reload_config(filename: str = CONFIG_FILE) -> bool:
    """Checks if the config file has been modified and reloads it if necessary."""
    global config, CONFIG_LAST_MODIFIED_TIME
    global api_key, secret, passphrase # Need to potentially reload these too if changed
    global exchange_id, symbol, timeframe, rsi_length, rsi_overbought, rsi_oversold # etc.
    global risk_percentage, stop_loss_percentage, take_profit_percentage # etc.
    global enable_atr_sl_tp, atr_length, atr_sl_multiplier, atr_tp_multiplier # etc.
    global enable_trailing_stop, trailing_stop_atr_multiplier, trailing_stop_activation_atr_multiplier # etc.
    global ob_volume_threshold_multiplier, ob_lookback # etc.
    global entry_volume_confirmation_enabled, entry_volume_ma_length, entry_volume_multiplier # etc.
    global sleep_interval_seconds, SIMULATION_MODE # etc.
    # Add Termux parameters
    global ENABLE_TERMUX_SMS_ALERTS, TERMUX_SMS_RECIPIENT, TERMUX_API_AVAILABLE

    # --- Add ALL parameters loaded from config that might change ---

    try:
        current_mod_time = os.path.getmtime(filename)
        if current_mod_time > CONFIG_LAST_MODIFIED_TIME:
            log_warning(f"Detected change in configuration file '{filename}'. Reloading...")
            try:
                new_config = load_config(filename) # Use the existing load function

                # --- Update global variables ---
                # Critical ones first (exchange/symbol change might require restart, handle carefully)
                new_exchange_id = new_config.get("exchange_id", "bybit").lower()
                new_symbol = new_config.get("symbol", "").strip().upper()

                if new_exchange_id != exchange_id or new_symbol != symbol:
                    # Simplest approach: Log and exit, requiring manual restart
                    log_error(f"CRITICAL CHANGE DETECTED: Exchange ID or Symbol changed in config ('{exchange_id}'->'{new_exchange_id}', '{symbol}'->'{new_symbol}'). Restart required.")
                    send_termux_sms(f"🚨 CONFIG ALERT: Exchange/Symbol changed for {symbol}. Bot requires RESTART.") # SMS on critical change
                    sys.exit(1) # Force exit
                                                                                       # Update other parameters (Example - add all relevant ones)                                                                                   config = new_config # Store the new config dict
                timeframe = config.get("timeframe", "1h")                              rsi_length = int(config.get("rsi_length", 14))
                risk_percentage = float(config.get("risk_percentage", 0.01))
                enable_trailing_stop = config.get("enable_trailing_stop", False)
                trailing_stop_atr_multiplier = float(config.get("trailing_stop_atr_multiplier", 1.5))
                sleep_interval_seconds = int(config.get("sleep_interval_seconds", 900))
                new_sim_mode = config.get("simulation_mode", True)

                # Update Termux parameters
                new_enable_sms = new_config.get("enable_termux_sms_alerts", False)                                                                            new_sms_recipient = new_config.get("termux_sms_recipient", None)                                                                                                                                                     # Log changes for important parameters
                if new_sim_mode != SIMULATION_MODE:
                     log_warning(f"SIMULATION MODE changed to: {'ACTIVE' if new_sim_mode else '!!! LIVE TRADING !!!'}")
                     send_termux_sms(f"⚠️ CONFIG ALERT: SIMULATION MODE changed to: {'ON' if new_sim_mode else 'OFF (LIVE!)'} for {symbol}") # Send SMS
                SIMULATION_MODE = new_sim_mode

                # Handle SMS setting changes
                if new_enable_sms != ENABLE_TERMUX_SMS_ALERTS or new_sms_recipient != TERMUX_SMS_RECIPIENT:
                    log_warning(f"Termux SMS settings changed. Enabled: {new_enable_sms}, Recipient: {new_sms_recipient}")
                    ENABLE_TERMUX_SMS_ALERTS = new_enable_sms
                    TERMUX_SMS_RECIPIENT = new_sms_recipient
                    # Re-validate if enabled
                    if ENABLE_TERMUX_SMS_ALERTS:
                        if not check_termux_api(): # Re-check API availability
                             log_error("Termux SMS re-enabled, but API check failed. Disabling again.")
                             ENABLE_TERMUX_SMS_ALERTS = False
                        elif not TERMUX_SMS_RECIPIENT:
                             log_error("Termux SMS re-enabled, but recipient is missing. Disabling again.")
                             ENABLE_TERMUX_SMS_ALERTS = False
                        else:
                             send_termux_sms(f"✅ CONFIG ALERT: Termux SMS alerts RE-ENABLED for {symbol}.") # Notify on successful re-enable
                    else:
                        log_info("Termux SMS alerts remain/now disabled.")


                # Add logging for other parameter changes if desired
                log_info(f"Risk % updated to: {risk_percentage*100:.2f}%")
                log_info(f"Trailing Stop enabled: {enable_trailing_stop}, ATR Multiplier: {trailing_stop_atr_multiplier}")
                log_info(f"Sleep Interval updated to: {sleep_interval_seconds}s")
                # ... log other reloaded params ...

                CONFIG_LAST_MODIFIED_TIME = current_mod_time
                log_info("Configuration reloaded successfully.")
                return True # Indicate reload happened

            except Exception as e:
                log_error(f"Failed to reload configuration from '{filename}': {e}", exc_info=True)
                send_termux_sms(f"🚨 ALERT: Failed to reload config for {symbol}. Check logs!") # Alert on reload failure
                # Keep using the old config
                return False # Indicate reload failed
        else:
            # log_debug("Config file unchanged.") # Optional: Log check
            return False # Indicate no reload needed

    except FileNotFoundError:
        log_error(f"Configuration file '{filename}' not found during check. Cannot reload.")
        # Keep using the current config in memory
        return False
    except Exception as e:
        log_error(f"Error checking config file modification time: {e}", exc_info=True)
        return False