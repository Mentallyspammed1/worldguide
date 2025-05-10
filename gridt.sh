#!/usr/bin/env bash

# Grid Trading Bot for Bybit (Termux Optimized - Live Trading Only)
# Features: Grid trading, Termux wake lock & notifications, secure API keys,
#           dynamic sizing, stop-loss/take-profit per grid level, log rotation, price caching.
# Prerequisites: Install Termux, curl, jq, bc, openssl, termux-api; run 'termux-setup-storage'

# --- Strict Mode ---
set -euo pipefail # Exit on error, unset variable, or pipe failure

# --- Configuration & Globals ---
CONFIG_FILE="gridbot.conf"
ENV_FILE=".env"
declare -g LOCK_FILE="/tmp/gridbot_$(basename "$0" .sh)_default.lock" # Finalized after config load
LOG_FILE="gridbot.log" # Default, can be overridden
declare -g SERVER_TIME_OFFSET=0 # Difference between server time and local time in MS
declare -g LAST_MARKET_PRICE=0 LAST_PRICE_TIMESTAMP=0 CACHE_TIMEOUT=3 # Cache market price for 3s
declare -g INSTRUMENT_INFO_CACHED=false

# --- Global Variables (Initialized in init_global_vars) ---
declare -gA ACTIVE_BUY_ORDERS   # Price -> OrderID
declare -gA ACTIVE_SELL_ORDERS  # Price -> OrderID
declare -gA TRADE_PAIRS         # BuyOrderID -> "SellOrderID,BuyPrice,SellPrice,Qty"
declare -g REALIZED_PNL_USD=0 TOTAL_PNL_USD=0 ORDER_FILL_COUNT=0
declare -g VOLATILITY_MULTIPLIER=1 TREND_BIAS=0
declare -g PRICE_PRECISION=4 QTY_PRECISION=1 MIN_ORDER_QTY="0.1" MIN_NOTIONAL_VALUE="5"
declare -g SCRIPT_RUNNING=true

# --- --- --- --- --- --- --- --- ---
# --- Core Functions             ---
# --- --- --- --- --- --- --- --- ---

# --- Function: Display Usage ---
usage() {
    cat << EOL
Grid Trading Bot for Bybit (Termux Enhanced - Live Trading)
===========================================================
Usage: ./$(basename "$0") [-h|--help]

Description:
  Automated LIVE grid trading bot for Bybit linear perpetual contracts.
  Optimized for running within the Termux environment.
  THIS SCRIPT IS FOR LIVE TRADING ONLY.

Prerequisites:
  1. Termux environment.
  2. Required packages:
     pkg install curl jq bc openssl termux-api coreutils util-linux inetutils gawk gzip
     (util-linux for 'flock', coreutils for 'timeout' and 'stat', inetutils for 'ping')
  3. Storage access (run 'termux-setup-storage' once and grant permission).
  4. API Keys:
     - Create '.env' file with BYBIT_API_KEY and BYBIT_API_SECRET, OR
     - (Recommended for Termux) Store keys using 'termux-keystore set bybit_api_key YOUR_KEY'
       and 'termux-keystore set bybit_api_secret YOUR_SECRET'. The script will prompt if not found.
  5. Configuration:
     - Review and customize 'gridbot.conf' (created on first run if missing).
     - Pay special attention to SYMBOL, ORDER_SIZE or ORDER_SIZE_PERCENTAGE (and minNotionalValue), LEVERAGE.

Features:
  - Automated LIVE grid trading strategy.
  - Termux Wake Lock to prevent device sleep.
  - Termux Notifications for critical alerts.
  - Secure API key handling options (environment file or Termux Keystore).
  - Dynamic order sizing based on available balance (configurable).
  - Stop-loss and take-profit for each grid order.
  - Log rotation to manage log file size.
  - Market price caching to reduce API load.
  - Network connectivity checks.
  - Server time synchronization.

Files:
  - Logs: ${LOG_FILE:-gridbot.log} (configurable via gridbot.conf)
  - Status: gridbot_status.txt (updated each loop)
  - Config: ${CONFIG_FILE:-gridbot.conf}
  - API Keys: ${ENV_FILE:-.env} (if not using Termux Keystore)
===========================================================
EOL
    exit 0
}

# --- Function: Manage Termux Wake Lock ---
manage_wake_lock() {
    local action="$1"
    if command -v termux-wake-lock &>/dev/null; then
        if [[ "$action" == "acquire" ]]; then
            termux-wake-lock
            log_message "INFO" "Acquired Termux wake lock."
        elif [[ "$action" == "release" ]]; then
            termux-wake-unlock
            log_message "INFO" "Released Termux wake lock."
        fi
    else
        log_message "WARNING" "termux-wake-lock not found. Power management may affect bot. (Install termux-api)"
    fi
}

# --- Function: Send Termux Notification ---
send_termux_notification() {
    local title="$1" message="$2"
    if command -v termux-notification &>/dev/null; then
        (termux-notification --title "$title" --content "$message" --id "gridbot_$(basename "$0" .sh)_${SYMBOL:-default}" --priority high --sound ) &
        log_message "INFO" "Sent Termux notification: $title - $message"
    else
        log_message "WARNING" "termux-notification not found. (Install termux-api)"
    fi
}

# --- Function: Check Termux Storage Access ---
check_storage_access() {
    log_message "DEBUG" "Checking Termux storage access..."
    local test_dir="${LOG_FILE%/*}"; [[ "$test_dir" == "$LOG_FILE" ]] && test_dir="."
    if ! mkdir -p "$test_dir" 2>/dev/null; then
        log_message "ERROR" "Cannot create log directory: '$test_dir'."
        handle_error "Storage access error for '$test_dir'." "STORAGE" "ERROR" 1
    fi
    if ! touch "${test_dir}/.gridbot_storage_test" 2>/dev/null ; then
        log_message "ERROR" "Cannot write to storage directory: '$test_dir'."
        command -v termux-setup-storage &>/dev/null && log_message "ERROR" "Run 'termux-setup-storage' and grant permissions." || log_message "ERROR" "'termux-setup-storage' not found."
        handle_error "Storage access required for logs/config." "STORAGE" "ERROR" 1
    else
        rm -f "${test_dir}/.gridbot_storage_test"
        log_message "INFO" "Storage access confirmed for directory: $test_dir"
    fi
}

# --- Function: Rotate Log File ---
rotate_log() {
    local max_size_mb=5 max_size_bytes=$((max_size_mb * 1024 * 1024))
    [[ -z "$LOG_FILE" ]] && return
    if [[ -f "$LOG_FILE" ]]; then
        local current_size; current_size=$(stat -c%s "$LOG_FILE" 2>/dev/null || stat -f %z "$LOG_FILE" 2>/dev/null || wc -c < "$LOG_FILE" | awk '{print $1}' || echo 0)
        if [[ "$current_size" -gt "$max_size_bytes" ]]; then
            local log_dir="${LOG_FILE%/*}"; [[ "$log_dir" == "$LOG_FILE" ]] && log_dir="."
            local backup_dir="${log_dir}/log_backups"; mkdir -p "$backup_dir"
            local backup_file="${backup_dir}/$(basename "$LOG_FILE").$(date +%Y%m%d_%H%M%S).gz"
            log_message "INFO" "Log file size ($current_size B) > max ($max_size_bytes B). Rotating..."
            if gzip -c "$LOG_FILE" > "$backup_file"; then : > "$LOG_FILE"; log_message "INFO" "Rotated log to $backup_file";
                find "$backup_dir" -name "$(basename "$LOG_FILE").*.gz" -type f -print0 | xargs -0 ls -t | tail -n +6 | xargs -r rm -- ;
            else log_message "ERROR" "Failed to gzip log file for rotation."; fi
        fi
    fi
}

# --- Function: Enhanced Error Handling ---
handle_error() {
    local error_message="$1" context="$2" severity="${3:-ERROR}" exit_code="${4:-1}"
    log_message "$severity" "[$context] $error_message"
    if [[ "$severity" = "ERROR" ]]; then
        send_termux_notification "Grid Bot Error: $context" "$error_message"
        [[ "${SMS_NOTIFICATIONS_ENABLED:-false}" == "true" ]] && send_sms_notification "CRITICAL ERROR: [$context] $error_message"
    fi
    if [[ "$severity" = "ERROR" && "$context" == "API" && "$exit_code" -ne 0 ]]; then
        log_message "INFO" "Pausing for API error recovery (30s)..."
        sleep 30
        if check_network && sync_server_time; then log_message "INFO" "Network/time sync OK after pause. Calling function may retry."; return 0;
        else log_message "ERROR" "API recovery failed. Exiting."; fi
    fi
    if [[ "$context" == "API" || "$context" == "TRADING" || "$context" == "MAIN" ]]; then
        log_message "INFO" "Attempting to cancel all orders due to error..." >&2
        local subshell_vars="SYMBOL BYBIT_API_URL BYBIT_API_KEY BYBIT_API_SECRET SERVER_TIME_OFFSET SMS_NOTIFICATIONS_ENABLED SMS_PHONE_NUMBER LOG_FILE LOG_LEVEL ENABLE_TIME_SYNC MIN_ORDER_QTY MIN_NOTIONAL_VALUE PRICE_PRECISION QTY_PRECISION"
        local declare_cmd=""; for var_name_in_list in $subshell_vars; do if [ -v "$var_name_in_list" ]; then declare_cmd+="$(declare -p "$var_name_in_list");"; else declare_cmd+="$var_name_in_list='';"; fi; done
        timeout 15s bash -c "$(declare -f cancel_all_orders bybit_request log_message send_sms_notification send_termux_notification handle_error check_network sync_server_time); $declare_cmd cancel_all_orders" || \
        log_message "WARNING" "Order cancellation during error handling failed or timed out." >&2
    fi
    exit "$exit_code"
}

# --- Function: Cleanup on Exit ---
cleanup() {
    local original_exit_status=$?
    SCRIPT_RUNNING=false
    log_message "INFO" "Shutdown signal (Exit: $original_exit_status). Cleaning up..." >&2
    sleep 0.5
    log_message "INFO" "Attempting to cancel all open orders on exit..." >&2
    local subshell_vars="SYMBOL BYBIT_API_URL BYBIT_API_KEY BYBIT_API_SECRET SERVER_TIME_OFFSET SMS_NOTIFICATIONS_ENABLED SMS_PHONE_NUMBER LOG_FILE LOG_LEVEL ENABLE_TIME_SYNC MIN_ORDER_QTY MIN_NOTIONAL_VALUE PRICE_PRECISION QTY_PRECISION"
    local declare_cmd=""; for var_name_in_list in $subshell_vars; do if [ -v "$var_name_in_list" ]; then declare_cmd+="$(declare -p "$var_name_in_list");"; else declare_cmd+="$var_name_in_list='';"; fi; done
    timeout 15s bash -c "$(declare -f cancel_all_orders bybit_request log_message send_sms_notification send_termux_notification handle_error check_network sync_server_time); $declare_cmd cancel_all_orders" || \
    log_message "WARNING" "Order cancellation during cleanup failed or timed out." >&2
    if [ -f "$LOCK_FILE" ]; then rm -f "$LOCK_FILE"; log_message "DEBUG" "Removed lock file: $LOCK_FILE" >&2; fi
    manage_wake_lock "release"
    log_message "INFO" "Grid bot stopped." >&2
    stty sane >/dev/null 2>&1 || true
    if [[ $original_exit_status -eq 0 || $original_exit_status -eq 130 || $original_exit_status -eq 143 ]]; then exit 0; else exit "$original_exit_status"; fi
}

# --- Function: Enhanced Logging (to stderr and file) ---
log_message() {
    local level="$1" message="$2"; local current_log_file="${LOG_FILE:-gridbot.log}" timestamp color_prefix="\033[0m" color_suffix="\033[0m"
    timestamp=$(date '+%Y-%m-%d %H:%M:%S'); local log_levels=("ERROR:1" "WARNING:2" "INFO:3" "DEBUG:4"); local log_level_val=3 msg_level_val=3
    for entry in "${log_levels[@]}"; do IFS=: read -r name val <<<"$entry"; [[ "${LOG_LEVEL:-INFO}" == "$name" ]] && log_level_val=$val; [[ "$level" == "$name" ]] && msg_level_val=$val; done
    case "$level" in DEBUG) color_prefix="\033[36m";; INFO) color_prefix="\033[32m";; WARNING) color_prefix="\033[33m";; ERROR) color_prefix="\033[91m";; *) level="INFO";color_prefix="\033[32m";; esac
    ( flock -n 200 || { echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Log flock failed: $message" >> "$current_log_file"; } ; echo "${timestamp} [${level}] ${message}" >> "$current_log_file"; ) 200>"${current_log_file}.lock"
    [[ "$msg_level_val" -le "$log_level_val" ]] && printf "%s %b[%s]%b %s\n" "$timestamp" "$color_prefix" "$level" "$color_suffix" "$message" >&2
}

# --- Function: Check Dependencies ---
check_dependencies() {
    log_message "DEBUG" "Checking dependencies..."
    local missing_deps=0; local -A dep_pkg_map=( [jq]=jq [bc]=bc [openssl]=openssl-tool [curl]=curl [date]=coreutils [printf]=coreutils [sort]=coreutils [head]=coreutils [tac]=coreutils [grep]=grep [timeout]=coreutils [awk]=gawk [paste]=coreutils [ping]=inetutils [stat]=coreutils [flock]=util-linux [gzip]=gzip [termux-wake-lock]=termux-api [termux-notification]=termux-api [termux-keystore]=termux-api [termux-sms-send]=termux-api )
    local deps_to_check=("jq" "bc" "openssl" "curl" "date" "printf" "sort" "head" "tac" "grep" "timeout" "awk" "paste" "ping" "stat" "flock" "gzip")
    [[ "${SMS_NOTIFICATIONS_ENABLED:-false}" == "true" ]] && deps_to_check+=("termux-sms-send")
    for dep in "${deps_to_check[@]}"; do if ! command -v "$dep" >/dev/null 2>&1; then local pkg_name=${dep_pkg_map[$dep]:-$dep}; log_message "ERROR" "'$dep' not found. Try: pkg install $pkg_name"; missing_deps=$((missing_deps + 1)); fi; done
    [[ "$missing_deps" -gt 0 ]] && handle_error "$missing_deps critical dependencies missing." "DEPENDENCY" "ERROR" 1
    log_message "INFO" "All critical dependencies found."
}

# --- Function: Load API Keys (Termux Keystore or .env) ---
load_api_keys() {
    log_message "DEBUG" "Loading API keys..."
    local key_source="environment variables (or not set)"
    if command -v termux-keystore &>/dev/null && { [ -z "${BYBIT_API_KEY:-}" ] || [ -z "${BYBIT_API_SECRET:-}" ]; }; then
        log_message "INFO" "Attempting to load API keys from Termux Keystore..."
        local stored_key stored_secret
        stored_key=$(termux-keystore get "bybit_api_key" 2>/dev/null || echo "")
        stored_secret=$(termux-keystore get "bybit_api_secret" 2>/dev/null || echo "")
        if [[ -n "$stored_key" && -n "$stored_secret" ]]; then
            BYBIT_API_KEY="$stored_key"; export BYBIT_API_KEY
            BYBIT_API_SECRET="$stored_secret"; export BYBIT_API_SECRET
            key_source="Termux Keystore"; log_message "INFO" "API keys loaded from Termux Keystore."
        else
            log_message "WARNING" "API keys not found in Termux Keystore."
            if [ -z "${BYBIT_API_KEY:-}" ]; then read -rsp "Enter Bybit API Key (or set in .env): " BYBIT_API_KEY_TEMP; export BYBIT_API_KEY="$BYBIT_API_KEY_TEMP"; echo; fi
            if [ -z "${BYBIT_API_SECRET:-}" ]; then read -rsp "Enter Bybit API Secret (or set in .env): " BYBIT_API_SECRET_TEMP; export BYBIT_API_SECRET="$BYBIT_API_SECRET_TEMP"; echo; fi
            if [[ -n "${BYBIT_API_KEY:-}" && -n "${BYBIT_API_SECRET:-}" ]]; then
                read -rp "Save these API keys to Termux Keystore for future use? (y/N): " save_choice
                if [[ "$save_choice" =~ ^[Yy]$ ]]; then
                    termux-keystore set "bybit_api_key" "$BYBIT_API_KEY"
                    termux-keystore set "bybit_api_secret" "$BYBIT_API_SECRET"
                    log_message "INFO" "API keys saved to Termux Keystore."
                fi
            fi; key_source="interactive input / .env"
        fi
    elif [[ -n "${BYBIT_API_KEY:-}" && -n "${BYBIT_API_SECRET:-}" ]]; then log_message "INFO" "API keys loaded from .env file."; key_source=".env file"; fi
    [ -z "${BYBIT_API_KEY:-}" ] || [ -z "${BYBIT_API_SECRET:-}" ] && handle_error "API keys missing." "ENV" "ERROR" 1
    [[ ${#BYBIT_API_KEY} -lt 16 || ${#BYBIT_API_SECRET} -lt 16 ]] && handle_error "API keys from $key_source too short." "ENV" "ERROR" 1
}

# --- Function: Load .env and API Keys ---
load_env() {
    log_message "DEBUG" "Loading environment variables..."
    if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE" || handle_error "Failed to source $ENV_FILE." "ENV" "ERROR" 1; set +a; fi
    load_api_keys
    log_message "INFO" "Environment variables and API keys loaded."
}

# --- Function: Load and Validate Configuration ---
load_config() {
    log_message "DEBUG" "Loading configuration from $CONFIG_FILE..."
    if [ ! -f "$CONFIG_FILE" ]; then
        log_message "WARNING" "Configuration file '$CONFIG_FILE' not found. Creating default."
        cat > "$CONFIG_FILE" << EOL
# Bybit Grid Trading Bot Configuration (Live Trading Focused)
# THIS BOT RUNS LIVE. THERE IS NO PAPER TRADING MODE. TEST ON TESTNET FIRST.
SYMBOL="DOTUSDT"
GRID_LEVELS=3
GRID_INTERVAL=0.05
# ORDER_SIZE_PERCENTAGE: Use a percentage of (balance * leverage) for each order's value.
# Example: 0.02 means each order (price*qty) will be ~2% of (balance*leverage).
# This value MUST result in an order value >= minNotionalValue for your chosen SYMBOL.
# (e.g., for DOTUSDT, minNotionalValue is usually 5 USDT).
# If ORDER_SIZE (fixed quantity) is set to a positive value, it overrides ORDER_SIZE_PERCENTAGE.
ORDER_SIZE_PERCENTAGE=0.02 # Example: Use 2% of leveraged balance per order.
ORDER_SIZE=0               # Set to a positive fixed quantity to override percentage.
LEVERAGE=5                 # CAUTION: Higher leverage increases risk significantly. Start low.
MAX_SPREAD_PERCENT=0.05
GRID_TOTAL_PROFIT_TARGET_USD=1.0
GRID_TOTAL_LOSS_LIMIT_USD=0.5
MAX_OPEN_ORDERS=10
CHECK_INTERVAL_SECONDS=20
LOG_FILE="gridbot.log"
LOG_LEVEL="INFO" # DEBUG, INFO, WARNING, ERROR
SMS_NOTIFICATIONS_ENABLED="false"
SMS_PHONE_NUMBER="+1234567890"
ENABLE_TIME_SYNC="true"
VOLATILITY_LOOKBACK_MINUTES=60
VOLATILITY_THRESHOLD_PERCENT=1.0
VOLATILITY_MAX_MULTIPLIER=2.0
TREND_LOOKBACK_HOURS=24
TREND_SMA_PERIOD=20
DYNAMIC_GRID_ENABLED="false"
DYNAMIC_GRID_FACTOR=0.1
# Stop Loss & Take Profit % for EACH INDIVIDUAL GRID ORDER
# Calculated from the entry price of that specific grid level. Set to 0 to disable.
GRID_ORDER_TAKE_PROFIT_PERCENT=0.03  # e.g., 3% TP from order entry
GRID_ORDER_STOP_LOSS_PERCENT=0.015 # e.g., 1.5% SL from order entry
ORDER_TYPE="Limit" # Typically "Limit" for grid strategies
BYBIT_API_URL="https://api.bybit.com"
# BYBIT_API_URL="https://api-testnet.bybit.com" # For testing with testnet keys
EOL
        handle_error "Default config '$CONFIG_FILE' created. REVIEW ALL SETTINGS CAREFULLY, especially SYMBOL, ORDER_SIZE/PERCENTAGE, and LEVERAGE, then run again. THIS BOT RUNS LIVE." "CONFIG" "INFO" 0
    fi
    source "$CONFIG_FILE" || handle_error "Failed to source $CONFIG_FILE." "CONFIG" "ERROR" 1
    local valid=true param val
    [[ -z "${SYMBOL:-}" ]] && { log_message "ERROR" "SYMBOL not set."; valid=false; }
    for param in SMS_NOTIFICATIONS_ENABLED DYNAMIC_GRID_ENABLED ENABLE_TIME_SYNC; do val="${!param:-}"; [[ "$val" != "true" && "$val" != "false" ]] && { log_message "ERROR" "Invalid boolean for $param: '$val'."; valid=false; }; done
    local int_params=("GRID_LEVELS" "LEVERAGE" "MAX_OPEN_ORDERS" "CHECK_INTERVAL_SECONDS" "VOLATILITY_LOOKBACK_MINUTES" "TREND_LOOKBACK_HOURS" "TREND_SMA_PERIOD")
    local float_params=("GRID_INTERVAL" "MAX_SPREAD_PERCENT" "GRID_TOTAL_PROFIT_TARGET_USD" "GRID_TOTAL_LOSS_LIMIT_USD" "VOLATILITY_THRESHOLD_PERCENT" "VOLATILITY_MAX_MULTIPLIER" "DYNAMIC_GRID_FACTOR" "GRID_ORDER_TAKE_PROFIT_PERCENT" "GRID_ORDER_STOP_LOSS_PERCENT")
    if [[ -n "${ORDER_SIZE_PERCENTAGE:-}" && "$(echo "${ORDER_SIZE_PERCENTAGE:-0} > 0" | bc -l)" -eq 1 ]]; then
        float_params+=("ORDER_SIZE_PERCENTAGE"); ORDER_SIZE=0
    elif [[ -n "${ORDER_SIZE:-}" && "$(echo "${ORDER_SIZE:-0} > 0" | bc -l)" -eq 1 ]]; then
        float_params+=("ORDER_SIZE"); ORDER_SIZE_PERCENTAGE=0
    else log_message "ERROR" "ORDER_SIZE or ORDER_SIZE_PERCENTAGE must be positive." ; valid=false; fi

    for param in "${int_params[@]}"; do val="${!param:-}"; ! [[ "$val" =~ ^[1-9][0-9]*$ ]] && { log_message "ERROR" "Invalid $param: '$val'. Must be positive integer."; valid=false; }; done
    for param in "${float_params[@]}"; do val="${!param:-}"; { ! [[ "$val" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ([[ "$(echo "$val <= 0" | bc -l)" -eq 1 ]] && ! [[ "$param" =~ (TAKE_PROFIT_PERCENT|STOP_LOSS_PERCENT)$ ]]) ; } && { log_message "ERROR" "Invalid $param: '$val'. Must be positive (or 0 for TP/SL)."; valid=false; }; done
    ! [[ "${LOG_LEVEL:-INFO}" =~ ^(DEBUG|INFO|WARNING|ERROR)$ ]] && { log_message "ERROR" "Invalid LOG_LEVEL: '${LOG_LEVEL:-}'."; valid=false; }
    [[ "${SMS_NOTIFICATIONS_ENABLED:-false}" == "true" && ! "${SMS_PHONE_NUMBER:-}" =~ ^\+[1-9][0-9]{10,}$ ]] && { log_message "ERROR" "Invalid SMS_PHONE_NUMBER."; valid=false; }
    ! [[ "${BYBIT_API_URL:-}" =~ ^https?:// ]] && { log_message "ERROR" "Invalid BYBIT_API_URL."; valid=false; }
    [[ "${CHECK_INTERVAL_SECONDS:-30}" -lt 5 ]] && { log_message "WARNING" "CHECK_INTERVAL_SECONDS too low. Min 5."; CHECK_INTERVAL_SECONDS=5; }
    [[ "${GRID_LEVELS:-3}" -gt 15 ]] && { log_message "WARNING" "GRID_LEVELS > 15 can be demanding. Max suggested for stability: 15."; }
    [[ "$valid" == "false" ]] && handle_error "Config validation failed." "CONFIG" "ERROR" 1
    LOG_FILE="${LOG_FILE:-gridbot.log}"; INSTRUMENT_INFO_CACHED=false
    log_message "INFO" "Configuration loaded and validated for LIVE TRADING."
}

# --- Function: Initialize Global Variables (after config) ---
init_global_vars() {
    log_message "DEBUG" "Initializing global variables..."
    ACTIVE_BUY_ORDERS=(); ACTIVE_SELL_ORDERS=(); TRADE_PAIRS=()
    REALIZED_PNL_USD=0; TOTAL_PNL_USD=0; ORDER_FILL_COUNT=0
    VOLATILITY_MULTIPLIER=1; TREND_BIAS=0
    PRICE_PRECISION=4; QTY_PRECISION=1; MIN_ORDER_QTY="0.1"; MIN_NOTIONAL_VALUE="5"
    SCRIPT_RUNNING=true; SERVER_TIME_OFFSET=0
    LAST_MARKET_PRICE=0; LAST_PRICE_TIMESTAMP=0
    log_message "DEBUG" "Global variables initialized."
}

# --- Function: Check Network Connectivity ---
check_network() {
    log_message "DEBUG" "Checking network connectivity..."
    if command -v ping &>/dev/null && ping -c 1 -W 1 8.8.8.8 &>/dev/null; then log_message "DEBUG" "Network (ping) OK."; return 0;
    elif curl --silent --head --fail --max-time 2 "https://www.google.com" &>/dev/null; then log_message "DEBUG" "Network (curl) OK."; return 0;
    else log_message "WARNING" "No network connectivity detected."; return 1; fi
}

# --- Function: Get Server Time Offset ---
sync_server_time() {
    [[ "${ENABLE_TIME_SYNC:-true}" != "true" ]] && return 0
    log_message "DEBUG" "Attempting to sync time with Bybit server..."
    check_network || { log_message "WARNING" "No network for time sync."; SERVER_TIME_OFFSET=0; return 1; }
    local response server_time_ns local_time_ms offset_ms server_time_ms
    response=$(timeout 5s curl -s "${BYBIT_API_URL}/v5/market/time" || echo "")
    if [[ -z "$response" ]]; then log_message "WARNING" "Failed to fetch server time or curl timed out."; SERVER_TIME_OFFSET=0; return 1; fi
    server_time_ns=$(echo "$response" | jq -r '.timeNano // ""'); [[ -z "$server_time_ns" || ! "$server_time_ns" =~ ^[0-9]+$ ]] && { log_message "WARNING" "Failed to parse server timeNano. Resp: $response"; SERVER_TIME_OFFSET=0; return 1; }
    server_time_ms=${server_time_ns:0:13}; local_time_ms=$(date +%s%3N); offset_ms=$((server_time_ms - local_time_ms))
    if [[ ${offset_ms#-} -gt 300 ]]; then SERVER_TIME_OFFSET="$offset_ms"; log_message "INFO" "Server time offset updated: ${SERVER_TIME_OFFSET}ms";
    else SERVER_TIME_OFFSET=0; log_message "DEBUG" "Time difference with server ${offset_ms}ms (within tolerance)."; fi
    return 0
}

# --- Function: Send SMS Notification (also calls Termux notification) ---
send_sms_notification() {
    local message="$1"; send_termux_notification "Grid Bot Alert" "$message"
    [[ "${SMS_NOTIFICATIONS_ENABLED:-false}" != "true" ]] && return 0
    ! command -v termux-sms-send &>/dev/null && { log_message "WARNING" "termux-sms-send not found."; return 1; }
    [ -z "${SMS_PHONE_NUMBER:-}" ] && { log_message "WARNING" "SMS_PHONE_NUMBER not set."; return 1; }
    log_message "INFO" "Sending SMS to ${SMS_PHONE_NUMBER}: ${message}"
    (termux-sms-send -n "$SMS_PHONE_NUMBER" "$message" &) || log_message "WARNING" "termux-sms-send execution failed."
}

# --- Function: Print Status Dashboard to file ---
print_status_dashboard() {
    local balance="$1" price="$2"; local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_dir_status="${LOG_FILE%/*}"; [[ "$log_dir_status" == "$LOG_FILE" ]] && log_dir_status="."
    local status_file="${log_dir_status}/gridbot_status.txt"
    {   printf "=== Grid Bot Status (%s) ===\n" "$timestamp"
        printf "Symbol: %s\n" "$SYMBOL"; printf "Balance: %s USDT\n" "$balance"; printf "Market Price: %s\n" "$price"
        printf "PNL: %s USD (Fills: %s)\n" "$TOTAL_PNL_USD" "$ORDER_FILL_COUNT"
        printf "Active Orders: Buys=%s, Sells=%s\n" "${#ACTIVE_BUY_ORDERS[@]}" "${#ACTIVE_SELL_ORDERS[@]}"
        printf "Volatility Multiplier: %.2f\n" "$VOLATILITY_MULTIPLIER"
        printf "Trend Bias: %s\n" "$( [[ $TREND_BIAS -eq 1 ]] && echo "Up" || [[ $TREND_BIAS -eq -1 ]] && echo "Down" || echo "Neutral" )"
        printf "=====================================\n"; } > "$status_file"
    log_message "DEBUG" "Status dashboard updated: $status_file"
}

# --- Bybit API Functions ---
# --- Function: Verify API Key Permissions ---
verify_api_permissions() {
    log_message "DEBUG" "Verifying API key permissions..."
    local response permissions
    response=$(bybit_request "/v5/user/query-api" "GET") || return 1
    permissions=$(echo "$response" | jq -r '.result.permissions.ContractTrade[]? // ""')
    if ! echo "$permissions" | grep -q -E "(Order|Position|UnifiedTrade)"; then
        handle_error "API key lacks 'ContractTrade' (Order/Position/UnifiedTrade) permission. Response: $permissions" "API" "ERROR" 1
    fi
    log_message "INFO" "API key permissions appear sufficient for contract trading."
}

# --- Function: Bybit API Request ---
bybit_request() {
    local endpoint="$1" method="${2:-GET}" data="${3:-}" acceptable_retcodes_csv="${4:-0}"
    check_network || { log_message "WARNING" "Skipping API request due to no network."; return 1; }
    local max_attempts=5 current_attempt base_delay=1 max_delay=16
    for ((current_attempt=1; current_attempt <= max_attempts; current_attempt++)); do
        local timestamp=$(( $(date +%s%3N) + SERVER_TIME_OFFSET )) recv_window=10000 # Increased recv_window
        local signature_payload params url request_body signature curl_args response_raw response http_code retCode retMsg curl_exit_code
        if [[ "$method" == "GET" ]]; then
            params="api_key=${BYBIT_API_KEY}×tamp=${timestamp}&recv_window=${recv_window}${data:+&$data}"
            signature_payload="${timestamp}${BYBIT_API_KEY}${recv_window}${params}"
            url="${BYBIT_API_URL}${endpoint}?${params}"; request_body=""
        else
            params="api_key=${BYBIT_API_KEY}×tamp=${timestamp}&recv_window=${recv_window}"
            signature_payload="${timestamp}${BYBIT_API_KEY}${recv_window}${data}"
            url="${BYBIT_API_URL}${endpoint}?${params}"; request_body="$data"
        fi
        signature=$(echo -n "$signature_payload" | openssl dgst -sha256 -hmac "$BYBIT_API_SECRET" -binary | xxd -p -c 256)
        curl_args=("-s" "-w" "\n%{http_code}" "-X" "$method" -H "X-BAPI-API-KEY: ${BYBIT_API_KEY}" -H "X-BAPI-SIGN: ${signature}" -H "X-BAPI-SIGN-TYPE: 2" -H "X-BAPI-TIMESTAMP: ${timestamp}" -H "X-BAPI-RECV-WINDOW: ${recv_window}" -H "Content-Type: application/json" -H "Accept: application/json" "--connect-timeout" "15" "--max-time" "30") # Increased curl timeouts
        [[ "$method" == "POST" ]] && curl_args+=("-d" "$request_body")
        curl_args+=("$url")
        log_message "DEBUG" "API Req (${method},Try ${current_attempt}/${max_attempts}): ${endpoint}${data:+?$data}"
        # Use a longer timeout for the entire command, shorter one for curl itself
        response_raw=$(timeout 35s curl "${curl_args[@]}" || echo "CURL_ERROR_TIMEOUT") ; curl_exit_code=$?
        
        if [[ "$response_raw" == "CURL_ERROR_TIMEOUT" || $curl_exit_code -eq 124 ]]; then log_message "WARNING" "Curl command timed out (Attempt ${current_attempt}/${max_attempts})."; retCode="-124"; response=""; # Ensure response is empty on timeout
        elif [ $curl_exit_code -ne 0 ]; then log_message "WARNING" "Curl command failed (Curl Exit:$curl_exit_code, Try ${current_attempt})."; retCode="-${curl_exit_code}"; response="";
        else
            http_code=$(echo "$response_raw" | tail -n1); response=$(echo "$response_raw" | sed '$d')
            log_message "DEBUG" "API Resp Code:${http_code}, Body Len:${#response}"
            [[ "$response" =~ ^<html ]] && log_message "WARNING" "API returned HTML, possibly a Cloudflare block or error page." && response="" # Clear HTML responses
            if [[ "$http_code" =~ ^[45] ]]; then log_message "WARNING" "API HTTP Err ${http_code} (Try ${current_attempt}). Resp:${response}"; [[ "$http_code" == "403" ]] && handle_error "HTTP 403. Check API keys/IP." "API"; retCode="-${http_code}";
            elif [ -z "$response" ]; then log_message "WARNING" "Empty API resp (HTTP:$http_code, Try ${current_attempt})."; retCode="-101";
            else retCode=$(echo "$response" | jq -r '.retCode // "-1"'); fi
        fi
        local is_acceptable=false; IFS=',' read -ra codes_array <<< "$acceptable_retcodes_csv"; for code_chk in "${codes_array[@]}"; do [[ "$retCode" -eq "$code_chk" ]] && { is_acceptable=true; break; }; done
        if $is_acceptable; then echo "$response"; return 0; fi
        
        # Safely get retMsg, even if response is not valid JSON
        if [[ -n "$response" && "$response" != "CURL_ERROR_TIMEOUT" ]]; then retMsg=$(echo "$response" | jq -r '.retMsg // "Unknown Bybit error or non-JSON response"'); else retMsg="No response body or curl error"; fi

        log_message "WARNING" "API Call Unsuccessful (retCode:${retCode}, Try ${current_attempt}): ${retMsg}"
        local non_retry_codes=(10001 10003 10004 10005 110007 110012 110014 110031 110033 110034 110035 110036 110040 110042 110044 110048 110052 110057 110060 110061 110062 110064 110067 110068 110071 110072 110073 110074 110075 110077 110078 110079 110080 110081 110082 110083 110084 110085 110086 110087 110088 110089 110090 110091 110092 110093 110094 110095 110096 110097 110098 110099 110100 110101 110102 110103 110104 110105 110106 110107 110108 110109 110110 110111 110112 110113 110114 110115 110116 110117 110118 110119 110120 110121 110122 110123 110124 110125 110126 110127 110128 110129 110130 110131 110132 110133 110134 110135 110136 110137 110138 110139 110140 110141 110142 110143 110144 110145 110146 110147 110148 110149 110150 110151 110152 110153 110154 110155 110156 110157 110158 110159 110160 110161)
        for nr_code in "${non_retry_codes[@]}"; do [[ "$retCode" == "$nr_code" ]] && handle_error "Unrecoverable API error (Code:$retCode): $retMsg" "API"; done
        [[ "$retCode" == "10002" ]] && { log_message "WARNING" "Timestamp error. Resyncing..."; sync_server_time; }
        [[ "$retCode" == "10006" ]] && { base_delay=$((base_delay * 4)); log_message "WARNING" "Rate limit. Increased backoff."; }
        if [ "$current_attempt" -lt "$max_attempts" ]; then
            local delay_jitter=$((base_delay + RANDOM % 3)); delay_jitter=$(( delay_jitter < max_delay ? delay_jitter : max_delay ))
            log_message "INFO" "Retrying API in ${delay_jitter}s..."; sleep "$delay_jitter"; base_delay=$((base_delay*2)); base_delay=$((base_delay < max_delay ? base_delay : max_delay))
        fi
    done
    handle_error "API req failed after ${max_attempts} attempts: ${endpoint}. Last retCode:${retCode:-N/A}" "API"
}

# --- Function: Get Wallet Balance (USDT) ---
get_wallet_balance() {
    log_message "DEBUG" "Fetching wallet balance..."
    local api_response balance_str
    api_response=$(bybit_request "/v5/account/wallet-balance" "GET" "accountType=UNIFIED") || return 1
    balance_str=$(echo "$api_response" | jq -r '.result.list[0].coin[]? | select(.coin=="USDT") | .walletBalance // "0"' 2>/dev/null | head -n 1)
    if ! [[ "$balance_str" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then log_message "ERROR" "Failed to parse USDT balance. Raw: '$balance_str'. API Resp: $api_response"; return 1; fi
    log_message "DEBUG" "Current USDT Balance: $balance_str"
    printf "%.4f\n" "$balance_str"; return 0
}

# --- Function: Get Instrument Info (Cached) ---
get_instrument_info() {
    if [[ "$INSTRUMENT_INFO_CACHED" == "true" ]]; then log_message "DEBUG" "Using cached instrument info."; return 0; fi
    log_message "INFO" "Fetching instrument info for ${SYMBOL}..."
    local response min_qty qty_step price_scale min_notional
    response=$(bybit_request "/v5/market/instruments-info" "GET" "category=linear&symbol=${SYMBOL}") || return 1
    min_qty=$(echo "$response" | jq -r ".result.list[0].lotSizeFilter.minOrderQty // \"${MIN_ORDER_QTY}\"")
    qty_step=$(echo "$response" | jq -r ".result.list[0].lotSizeFilter.qtyStep // \"0.1\"")
    price_scale=$(echo "$response" | jq -r ".result.list[0].priceScale // \"${PRICE_PRECISION}\"")
    min_notional=$(echo "$response" | jq -r ".result.list[0].lotSizeFilter.minNotionalValue // \"${MIN_NOTIONAL_VALUE}\"")
    for var_name in min_qty qty_step min_notional; do val="${!var_name}"; [[ ! "$val" =~ ^[0-9]+(\.[0-9]+)?$ ]] && { log_message "ERROR" "Invalid $var_name received: $val"; return 1; }; done
    [[ ! "$price_scale" =~ ^[0-9]+$ ]] && { log_message "ERROR" "Invalid priceScale received: $price_scale"; return 1; }
    MIN_ORDER_QTY="$min_qty"; PRICE_PRECISION="$price_scale"; MIN_NOTIONAL_VALUE="$min_notional"
    if [[ "$qty_step" == "1" ]]; then QTY_PRECISION=0
    elif [[ "$qty_step" =~ ^0\.([0-9]*1$) ]]; then QTY_PRECISION=${#BASH_REMATCH[1]}
    elif [[ "$qty_step" =~ \. ]]; then QTY_PRECISION=$(echo "$qty_step" | awk -F'.' '{print length($2)}'); log_message "WARNING" "Uncommon qtyStep $qty_step, using prec $QTY_PRECISION."
    else QTY_PRECISION=0; log_message "WARNING" "Unknown qtyStep $qty_step, using prec 0."; fi
    INSTRUMENT_INFO_CACHED=true
    log_message "INFO" "Instrument: MinQty=${MIN_ORDER_QTY}, MinNotional=${MIN_NOTIONAL_VALUE}, QtyPrec=${QTY_PRECISION}, PricePrec=${PRICE_PRECISION}"
    return 0
}

# --- Function: Get Current Market Price (Cached) ---
get_market_price() {
    local current_time=$(date +%s)
    if [[ $((current_time - LAST_PRICE_TIMESTAMP)) -lt $CACHE_TIMEOUT && "$LAST_MARKET_PRICE" != "0" ]]; then
        log_message "DEBUG" "Using cached market price: $LAST_MARKET_PRICE"
        printf "%.*f\n" "$PRICE_PRECISION" "$LAST_MARKET_PRICE"; return 0
    fi
    log_message "DEBUG" "Fetching market price for ${SYMBOL}..."
    local response price
    response=$(bybit_request "/v5/market/tickers" "GET" "category=linear&symbol=${SYMBOL}") || return 1
    price=$(echo "$response" | jq -r '.result.list[0].lastPrice // ""')
    if ! [[ "$price" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "$(echo "$price <= 0" | bc -l)" -eq 1 ]]; then log_message "ERROR" "Invalid market price: '$price'"; return 1; fi
    LAST_MARKET_PRICE="$price"; LAST_PRICE_TIMESTAMP="$current_time"
    log_message "DEBUG" "Market Price: $price"
    printf "%.*f\n" "$PRICE_PRECISION" "$price"; return 0
}

# --- Function: Check Market Spread ---
check_market_conditions() {
    log_message "DEBUG" "Checking market conditions for ${SYMBOL}..."
    local response bid ask spread_percent
    response=$(bybit_request "/v5/market/tickers" "GET" "category=linear&symbol=${SYMBOL}") || return 1
    bid=$(echo "$response" | jq -r '.result.list[0].bid1Price // "0"'); ask=$(echo "$response" | jq -r '.result.list[0].ask1Price // "0"')
    if ! [[ "$bid" =~ ^[0-9.]+$ && "$ask" =~ ^[0-9.]+$ ]] || [[ "$(echo "$bid <= 0 || $ask <= 0 || $ask < $bid" | bc -l)" -eq 1 ]]; then log_message "WARNING" "Invalid bid ($bid)/ask ($ask) or crossed book."; return 1; fi
    spread_percent=$(echo "scale=5; if ($ask > 0) (($ask - $bid) / $ask) * 100 else 0" | bc -l)
    log_message "DEBUG" "Bid: ${bid}, Ask: ${ask}, Spread: ${spread_percent}%"
    if [[ "$(echo "$spread_percent > $MAX_SPREAD_PERCENT" | bc -l)" -eq 1 ]]; then log_message "WARNING" "Spread (${spread_percent}%) > threshold (${MAX_SPREAD_PERCENT}%)."; return 1; fi
    log_message "INFO" "Spread OK: ${spread_percent}%"; return 0
}

# --- Function: Calculate Volatility (Internal) ---
_calculate_volatility() {
    local prices_str="$1" current_price="$2"; local prices=($prices_str); local num_prices=${#prices[@]}
    log_message "DEBUG" "Calculating volatility from $num_prices prices. Current: $current_price"
    if [[ "$num_prices" -lt 2 ]] || ! [[ "$current_price" =~ ^[0-9.]+$ ]] || [[ "$(echo "$current_price <= 0" | bc -l)" -eq 1 ]]; then log_message "WARNING" "Insufficient data/invalid price for volatility calc."; echo 1; return 1; fi
    local sum mean sum_sq_diff variance std_dev volatility_percent threshold max_multiplier multiplier result_multiplier=1
    sum=$(printf "%s\n" "${prices[@]}" | paste -sd+ | bc -l); mean=$(echo "scale=10; $sum / $num_prices" | bc -l)
    sum_sq_diff=0; for price in "${prices[@]}"; do local diff=$(echo "$price - $mean" | bc -l); sum_sq_diff=$(echo "$sum_sq_diff + ($diff * $diff)" | bc -l); done
    variance=$(echo "scale=10; if ($num_prices > 1) $sum_sq_diff / ($num_prices - 1) else 0" | bc -l)
    [[ "$(echo "$variance >= 0" | bc -l)" -eq 1 ]] && std_dev=$(echo "scale=10; sqrt($variance)" | bc -l) || std_dev=0
    volatility_percent=$(echo "scale=4; ($std_dev / $current_price) * 100" | bc -l)
    threshold="${VOLATILITY_THRESHOLD_PERCENT:-1.0}"; max_multiplier="${VOLATILITY_MAX_MULTIPLIER:-2.0}"
    if [[ "$(echo "$volatility_percent > $threshold" | bc -l)" -eq 1 ]]; then
        multiplier=$(echo "scale=4; 1 + (($volatility_percent - $threshold) / $threshold)" | bc -l)
        result_multiplier=$(echo "scale=2; define min(a,b){if(a<b)return(a);return(b)}; min($multiplier, $max_multiplier)" | bc -l)
        log_message "WARNING" "High volatility: ${volatility_percent}% (StdDev:$std_dev). Multiplier: ${result_multiplier}"
    else log_message "INFO" "Volatility OK: ${volatility_percent}% (StdDev:$std_dev). Multiplier: 1.0"; fi
    echo "$result_multiplier"; return 0
}

# --- Function: Calculate Trend (Internal) ---
_calculate_trend() {
    local prices_str="$1" current_price="$2" sma_period="$3"; local prices=($prices_str); local num_prices=${#prices[@]}
    log_message "DEBUG" "Calculating trend from $num_prices prices. Current: $current_price, Period: $sma_period"
    if [[ "$num_prices" -lt "$sma_period" ]] || ! [[ "$current_price" =~ ^[0-9.]+$ ]] || [[ "$(echo "$current_price <= 0" | bc -l)" -eq 1 ]]; then log_message "WARNING" "Insufficient data/invalid price for trend calc."; echo 0; return 1; fi
    local recent_closes=("${prices[@]: -$sma_period}"); local sma_sum sma trend_bias=0
    sma_sum=$(printf "%s\n" "${recent_closes[@]}" | paste -sd+ | bc -l); sma=$(echo "scale=$PRICE_PRECISION; $sma_sum / $sma_period" | bc -l)
    if [[ "$(echo "$current_price > $sma" | bc -l)" -eq 1 ]]; then trend_bias=1; elif [[ "$(echo "$current_price < $sma" | bc -l)" -eq 1 ]]; then trend_bias=-1; fi
    log_message "INFO" "Trend: $([[ $trend_bias -eq 1 ]] && echo "Up" || [[ $trend_bias -eq -1 ]] && echo "Down" || echo "Neutral") (Price:${current_price}, SMA(${sma_period}):${sma})"
    echo "$trend_bias"; return 0
}

# --- Function: Place Order ---
place_order() {
    local side="$1" price="$2" qty="$3" check_min_notional="${4:-true}"
    if [[ "$side" != "Buy" && "$side" != "Sell" ]]; then log_message "ERROR" "Invalid side: '$side'"; return 1; fi
    if ! [[ "$price" =~ ^[0-9.]+$ && "$qty" =~ ^[0-9.]+$ ]] || [[ "$(echo "$price <= 0 || $qty <= 0" | bc -l)" -eq 1 ]]; then log_message "ERROR" "Invalid price/qty: P=$price, Q=$qty"; return 1; fi
    if [[ "$(echo "$qty < $MIN_ORDER_QTY" | bc -l)" -eq 1 ]]; then log_message "ERROR" "Qty '$qty' < Min Qty ($MIN_ORDER_QTY)"; return 1; fi
    if "$check_min_notional"; then
        local order_value=$(echo "scale=4; $price * $qty" | bc -l)
        if [[ "$(echo "$order_value < $MIN_NOTIONAL_VALUE" | bc -l)" -eq 1 ]]; then
            log_message "ERROR" "Order value ($order_value USDT) < Min Notional ($MIN_NOTIONAL_VALUE USDT)."
            return 1
        fi
    fi
    local fmt_price=$(printf "%.*f" "$PRICE_PRECISION" "$price") fmt_qty=$(printf "%.*f" "$QTY_PRECISION" "$qty")
    local sl_price tp_price payload
    payload=$(jq -nc --arg cat "linear" --arg sym "$SYMBOL" --arg sd "$side" --arg ot "${ORDER_TYPE:-Limit}" --arg q "$fmt_qty" --arg p "$fmt_price" --arg tif "GTC" \
        '{category:$cat,symbol:$sy,side:$sd,orderType:$ot,qty:$q,price:$p,timeInForce:$tif}')
    
    local sl_percent="${GRID_ORDER_STOP_LOSS_PERCENT:-0}" tp_percent="${GRID_ORDER_TAKE_PROFIT_PERCENT:-0}"
    if [[ "$sl_percent" != "0" && "$(echo "$sl_percent > 0" | bc -l)" -eq 1 ]]; then
        if [[ "$side" == "Buy" ]]; then sl_price=$(echo "scale=$PRICE_PRECISION; $price * (1 - $sl_percent)" | bc -l)
        else sl_price=$(echo "scale=$PRICE_PRECISION; $price * (1 + $sl_percent)" | bc -l); fi
        [[ "$(echo "$sl_price > 0" | bc -l)" -eq 1 ]] && payload=$(echo "$payload" | jq --arg sl "$sl_price" '. + {stopLoss: $sl}')
    fi
    if [[ "$tp_percent" != "0" && "$(echo "$tp_percent > 0" | bc -l)" -eq 1 ]]; then
        if [[ "$side" == "Buy" ]]; then tp_price=$(echo "scale=$PRICE_PRECISION; $price * (1 + $tp_percent)" | bc -l)
        else tp_price=$(echo "scale=$PRICE_PRECISION; $price * (1 - $tp_percent)" | bc -l); fi
        [[ "$(echo "$tp_price > 0" | bc -l)" -eq 1 ]] && payload=$(echo "$payload" | jq --arg tp "$tp_price" '. + {takeProfit: $tp}')
    fi
    log_message "INFO" "Attempting $side order: Qty=${fmt_qty}, Price=${fmt_price}${sl_price:+, SL:$sl_price}${tp_price:+, TP:$tp_price}"
    
    local response order_id
    response=$(bybit_request "/v5/order/create" "POST" "$payload") || return 1
    order_id=$(echo "$response" | jq -r '.result.orderId // ""'); [ -z "$order_id" ] && { log_message "ERROR" "No order ID from $side. Resp: $response"; return 1; }
    log_message "INFO" "$side order placed: ID=${order_id}"
    [[ "$side" == "Buy" ]] && ACTIVE_BUY_ORDERS["$price"]="$order_id" || ACTIVE_SELL_ORDERS["$price"]="$order_id"
    echo "$response"; return 0
}

# --- Function: Cancel Order ---
cancel_order() {
    local order_id="$1" price_key="$2" side="$3"
    if [[ -z "$order_id" || -z "$price_key" || ("$side" != "Buy" && "$side" != "Sell") ]]; then log_message "ERROR" "cancel_order args invalid"; return 1; fi
    log_message "INFO" "Cancelling $side order ID: $order_id (Price: $price_key)"
    [[ "$side" == "Buy" ]] && unset ACTIVE_BUY_ORDERS["$price_key"] || unset ACTIVE_SELL_ORDERS["$price_key"]
    local payload=$(jq -nc --arg c "linear" --arg sy "$SYMBOL" --arg id "$order_id" '{category:$c,symbol:$sy,orderId:$id}')
    local subshell_vars="SYMBOL BYBIT_API_URL BYBIT_API_KEY BYBIT_API_SECRET SERVER_TIME_OFFSET LOG_FILE LOG_LEVEL ENABLE_TIME_SYNC"
    local declare_cmd=""; for var in $subshell_vars; do declare_cmd+="$(declare -p "$var" 2>/dev/null || echo "$var=${!var}");"; done
    timeout 10s bash -c "$(declare -f bybit_request log_message send_sms_notification send_termux_notification handle_error check_network sync_server_time); $declare_cmd bybit_request '/v5/order/cancel' 'POST' \"\$payload\" \"0,110025,110021\" " || {
        log_message "WARNING" "Cancel order $order_id failed or timed out."; return 1;
    }
    log_message "INFO" "Cancel request for order ID $order_id processed."; return 0
}

# --- Function: Cancel All Orders ---
cancel_all_orders() {
    log_message "INFO" "Cancelling ALL open orders for ${SYMBOL}..."
    ACTIVE_BUY_ORDERS=(); ACTIVE_SELL_ORDERS=(); TRADE_PAIRS=()
    local payload=$(jq -nc --arg c "linear" --arg sy "$SYMBOL" '{category:$c,symbol:$sy,orderFilter:"Order"}')
    local response exit_status
    local subshell_vars="SYMBOL BYBIT_API_URL BYBIT_API_KEY BYBIT_API_SECRET SERVER_TIME_OFFSET LOG_FILE LOG_LEVEL ENABLE_TIME_SYNC"
    local declare_cmd=""; for var in $subshell_vars; do declare_cmd+="$(declare -p "$var" 2>/dev/null || echo "$var=${!var}");"; done
    response=$(timeout 15s bash -c "$(declare -f bybit_request log_message send_sms_notification send_termux_notification handle_error check_network sync_server_time); $declare_cmd bybit_request '/v5/order/cancel-all' 'POST' \"\$payload\" \"0\" ")
    exit_status=$?
    if [ $exit_status -eq 124 ]; then log_message "WARNING" "Cancel-all orders timed out."; return 1;
    elif [ $exit_status -ne 0 ]; then log_message "WARNING" "Cancel-all request failed (Exit: $exit_status). Resp: $response"; return 1; fi
    local cancelled_count=$(echo "$response" | jq -r '.result.list | length // 0')
    log_message "INFO" "Cancel-all successful, ${cancelled_count} order(s) reported cancelled."
    return 0
}

# --- Function: Set Leverage ---
set_leverage() {
    log_message "INFO" "Setting leverage to ${LEVERAGE}x for ${SYMBOL}..."
    local payload=$(jq -nc --arg c "linear" --arg sy "$SYMBOL" --arg l "$LEVERAGE" '{category:$c,symbol:$sy,buyLeverage:$l,sellLeverage:$l}')
    local response retCode
    response=$(bybit_request "/v5/position/set-leverage" "POST" "$payload" "0,110043")
    if [ $? -eq 0 ]; then
        retCode=$(echo "$response" | jq -r '.retCode // "-1"')
        if [[ "$retCode" -eq 0 ]]; then log_message "INFO" "Leverage set to ${LEVERAGE}x."; return 0;
        elif [[ "$retCode" -eq 110043 ]]; then log_message "INFO" "Leverage already ${LEVERAGE}x."; return 0;
        else log_message "WARNING" "set_leverage: Unexpected retCode $retCode. Resp: $response"; return 1; fi
    else log_message "WARNING" "Failed to set leverage (API request command failed)."; return 1; fi
}

# --- Grid Logic Functions ---

# --- Function: Check Open Orders, Update State, Calculate PNL ---
check_orders() {
    log_message "DEBUG" "Checking orders & processing fills..."
    local response orders_json; response=$(bybit_request "/v5/order/realtime" "GET" "category=linear&symbol=${SYMBOL}&limit=50") || return 1
    declare -A current_active_buys=("${ACTIVE_BUY_ORDERS[@]}"); declare -A current_active_sells=("${ACTIVE_SELL_ORDERS[@]}"); ACTIVE_BUY_ORDERS=(); ACTIVE_SELL_ORDERS=()
    orders_json=$(echo "$response" | jq -c '.result.list[]' 2>/dev/null)
    if [ -z "$orders_json" ]; then log_message "DEBUG" "No open orders from API."; else
        while IFS= read -r order_data; do
            local id s p q eq st ap; id=$(echo "$order_data"|jq -r .orderId); s=$(echo "$order_data"|jq -r .side); p=$(echo "$order_data"|jq -r .price); q=$(echo "$order_data"|jq -r .qty); eq=$(echo "$order_data"|jq -r .cumExecQty); st=$(echo "$order_data"|jq -r .orderStatus); ap=$(echo "$order_data"|jq -r '.avgPrice//"0"'); [[ "$ap" == "0" || -z "$ap" ]] && ap="$p"
            log_message "DEBUG" "API Order: ID=$id, Side=$s, Price=$p, Qty=$q, ExecQty=$eq, Status=$st, AvgPrice=$ap"
            case "$st" in
                "New"|"PartiallyFilled") [[ "$s"=="Buy" ]] && ACTIVE_BUY_ORDERS["$p"]="$id" || ACTIVE_SELL_ORDERS["$p"]="$id"; unset current_active_buys["$p"] current_active_sells["$p"];;
                "Filled")
                    if [[ "$s" == "Buy" && -n "${current_active_buys[$p]:-}" && "${current_active_buys[$p]}" == "$id" ]]; then
                        log_message "INFO" "BUY Filled: ID=$id, Price=$ap, Qty=$eq"; TRADE_PAIRS["$id"]="0,$ap,0,$eq"; unset current_active_buys["$p"]
                    elif [[ "$s" == "Sell" && -n "${current_active_sells[$p]:-}" && "${current_active_sells[$p]}" == "$id" ]]; then
                        log_message "INFO" "SELL Filled: ID=$id, Price=$ap, Qty=$eq"; unset current_active_sells["$p"]; local fp=false
                        for buy_id in "${!TRADE_PAIRS[@]}"; do local pair_data=${TRADE_PAIRS[$buy_id]}; IFS=',' read -r ps_id pb_p ps_p p_q <<< "$pair_data"
                            [[ "$ps_id" == "0" && "$(echo "scale=5; abs($p_q - $eq) < 0.00001" | bc -l)" -eq 1 ]] && {
                                local pnl=$(echo "$eq * ($ap - $pb_p)" | bc -l); REALIZED_PNL_USD=$(echo "$REALIZED_PNL_USD + $pnl" | bc -l); ORDER_FILL_COUNT=$((ORDER_FILL_COUNT+1))
                                log_message "INFO" "TRADE COMPLETED: BuyID=$buy_id @ $pb_p -> SellID=$id @ $ap. Qty:$eq. PNL:$pnl USD. Total PNL:$REALIZED_PNL_USD"
                                send_termux_notification "Trade Completed ($SYMBOL)" "PNL: $pnl USD"
                                TRADE_PAIRS["$buy_id"]="$id,$pb_p,$ap,$eq"; fp=true; break; }; done
                        [[ "$fp" == false ]] && log_message "WARNING" "Filled Sell $id (Qty:$eq) no matching open Buy."
                    fi ;;
                "Cancelled"|"Rejected"|"Expired") log_message "INFO" "Order $id $st."; unset current_active_buys["$p"] current_active_sells["$p"];;
                *) log_message "WARNING" "Unhandled status '$st' for $id.";;
            esac
        done <<< "$orders_json"; fi
    for p in "${!current_active_buys[@]}"; do log_message "WARNING" "Tracked Buy ${current_active_buys[$p]} @ $p missing from API.";done
    for p in "${!current_active_sells[@]}"; do log_message "WARNING" "Tracked Sell ${current_active_sells[$p]} @ $p missing from API.";done
    TOTAL_PNL_USD="$REALIZED_PNL_USD"
    log_message "DEBUG" "Order check done. Buys:${#ACTIVE_BUY_ORDERS[@]}, Sells:${#ACTIVE_SELL_ORDERS[@]}, Pairs:${#TRADE_PAIRS[@]}"
    return 0
}

# --- Function: Determine Order Size ---
determine_order_size() {
    local current_price="$1" balance="$2" calculated_size final_size
    if [[ -n "${ORDER_SIZE_PERCENTAGE:-}" && "$(echo "${ORDER_SIZE_PERCENTAGE:-0} > 0" | bc -l)" -eq 1 ]]; then
        local value_per_order=$(echo "scale=8; $balance * $LEVERAGE * $ORDER_SIZE_PERCENTAGE" | bc -l)
        calculated_size=$(echo "scale=$((QTY_PRECISION + 2)); $value_per_order / $current_price" | bc -l)
        log_message "DEBUG" "Dynamic size: Value/Order=${value_per_order}, CalcSize=${calculated_size}"
    elif [[ -n "${ORDER_SIZE:-}" && "$(echo "${ORDER_SIZE:-0} > 0" | bc -l)" -eq 1 ]]; then
        calculated_size="${ORDER_SIZE}"; log_message "DEBUG" "Fixed size from config: ${calculated_size}"
    else log_message "ERROR" "Order sizing not properly configured!"; calculated_size="$MIN_ORDER_QTY"; fi
    if [[ "$(echo "$calculated_size < $MIN_ORDER_QTY" | bc -l)" -eq 1 ]]; then
        log_message "WARNING" "Calculated size ($calculated_size) < MIN_ORDER_QTY ($MIN_ORDER_QTY). Using MIN_ORDER_QTY."
        calculated_size="$MIN_ORDER_QTY"
    fi
    final_size=$(printf "%.*f" "$QTY_PRECISION" "$calculated_size")
    if [[ "$(echo "$final_size <= 0" | bc -l)" -eq 1 ]]; then
        log_message "WARNING" "Final size after precision ($final_size) is zero. Using MIN_ORDER_QTY."
        final_size="$MIN_ORDER_QTY"
    fi
    echo "$final_size"
}

# --- Function: Place Initial Grid Orders ---
place_grid_orders() {
    log_message "INFO" "Setting up initial grid..."
    local order_count=0 current_price balance grid_interval order_size order_value
    current_price=$(get_market_price) || handle_error "Failed to get market price for initial grid." "TRADING"
    get_instrument_info || handle_error "Failed to get instrument info." "TRADING"
    sync_server_time # Critical before live operations
    balance=$(get_wallet_balance) || handle_error "Failed to get wallet balance." "TRADING"
    set_leverage || handle_error "Failed to set leverage." "TRADING"
    check_market_conditions || handle_error "Initial market conditions unfavorable (spread too high)." "TRADING"

    grid_interval=$(echo "scale=$PRICE_PRECISION; $GRID_INTERVAL * $VOLATILITY_MULTIPLIER" | bc -l)
    [[ "$(echo "$grid_interval <= 0" | bc -l)" -eq 1 ]] && grid_interval="$GRID_INTERVAL"
    order_size=$(determine_order_size "$current_price" "$balance")
    log_message "INFO" "Grid Interval: $grid_interval, Final Order Size: $order_size"

    [[ "$(echo "$order_size < $MIN_ORDER_QTY" | bc -l)" -eq 1 ]] && handle_error "Final ORDER_SIZE ($order_size) < Min Qty ($MIN_ORDER_QTY)." "CONFIG"
    order_value=$(echo "scale=4; $current_price * $order_size" | bc -l)
    [[ "$(echo "$order_value < $MIN_NOTIONAL_VALUE" | bc -l)" -eq 1 ]] && handle_error "Final Order Value ($order_value USDT) < Min Notional ($MIN_NOTIONAL_VALUE USDT)." "CONFIG"
    local total_orders_to_place=$(( GRID_LEVELS * 2 ))
    local total_margin_needed=$(echo "scale=6; $order_value * $total_orders_to_place / $LEVERAGE" | bc -l)
    local available_margin=$(echo "scale=6; $balance * 0.95" | bc -l)
    log_message "INFO" "Est. total margin for grid: ${total_margin_needed} USDT. Available: ${available_margin} USDT."
    [[ "$(echo "$total_margin_needed > $available_margin" | bc -l)" -eq 1 ]] && handle_error "Insufficient balance for grid." "TRADING"

    log_message "INFO" "Cancelling existing orders for $SYMBOL..."
    cancel_all_orders || handle_error "Failed to cancel existing orders for $SYMBOL." "TRADING"
    sleep 2
    log_message "INFO" "Placing initial grid orders around $current_price"
    for (( i=1; i<=GRID_LEVELS; i++ )); do
        ! $SCRIPT_RUNNING && { log_message "INFO" "Grid placement halted."; break; }
        local buy_price sell_price
        buy_price=$(echo "scale=$PRICE_PRECISION; $current_price - ($i * $grid_interval)" | bc -l)
        sell_price=$(echo "scale=$PRICE_PRECISION; $current_price + ($i * $grid_interval)" | bc -l)
        if [[ "$(echo "$buy_price > 0" | bc -l)" -eq 1 ]]; then
            place_order "Buy" "$buy_price" "$order_size" && order_count=$((order_count + 1)) || { log_message "ERROR" "Failed: Buy L$i."; cancel_all_orders; return 1; }
            sleep 0.3
        else log_message "WARNING" "Skipping Buy L$i: Price <= 0 ($buy_price)"; fi
        ! $SCRIPT_RUNNING && break
        place_order "Sell" "$sell_price" "$order_size" && order_count=$((order_count + 1)) || { log_message "ERROR" "Failed: Sell L$i."; cancel_all_orders; return 1; }
        sleep 0.3
    done
    log_message "INFO" "Initial grid placed ${order_count} orders."
    sleep 3
    return 0
}

# --- Function: Rebalance Grid ---
rebalance_grid() {
    log_message "DEBUG" "Checking grid balance..."
    local buy_count=${#ACTIVE_BUY_ORDERS[@]} sell_count=${#ACTIVE_SELL_ORDERS[@]} target_lvls="$GRID_LEVELS" placed_this_cycle=0 max_place=2
    if [[ "$buy_count" -ge "$target_lvls" && "$sell_count" -ge "$target_lvls" ]]; then log_message "DEBUG" "Grid balanced."; return 0; fi
    log_message "INFO" "Grid imbalance. B:$buy_count/$target_lvls, S:$sell_count/$target_lvls. Rebalancing..."
    local current_price balance grid_interval order_size
    current_price=$(get_market_price) || { log_message "ERROR" "Rebalance: No market price."; return 1; }
    balance=$(get_wallet_balance) || { log_message "ERROR" "Rebalance: No balance info."; return 1; } # Needed for dynamic sizing
    grid_interval=$(echo "scale=$PRICE_PRECISION; $GRID_INTERVAL * $VOLATILITY_MULTIPLIER" | bc -l); [[ "$(echo "$grid_interval <= 0"|bc -l)" -eq 1 ]] && grid_interval="$GRID_INTERVAL"
    order_size=$(determine_order_size "$current_price" "$balance") # Recalculate size
    
    if [[ "$buy_count" -lt "$target_lvls" && "$placed_this_cycle" -lt "$max_place" ]]; then
        local lowest_buy; [[ "$buy_count" -gt 0 ]] && lowest_buy=$(printf "%s\n" "${!ACTIVE_BUY_ORDERS[@]}"|sort -n|head -n1) || lowest_buy=$(echo "$current_price - $grid_interval"|bc -l)
        for (( i=1; i<=$((target_lvls - buy_count)) && placed_this_cycle < max_place; i++ )); do
            ! $SCRIPT_RUNNING && break; local new_buy=$(echo "$lowest_buy - ($i * $grid_interval)"|bc -l)
            if [[ "$(echo "$new_buy > 0"|bc -l)" -eq 1 && -z "${ACTIVE_BUY_ORDERS[$new_buy]:-}" ]]; then
                place_order "Buy" "$new_buy" "$order_size" && placed_this_cycle=$((placed_this_cycle+1)) || { log_message "ERROR" "Rebalance Buy failed."; break; }; sleep 0.3
            fi; done; fi
    if [[ "$sell_count" -lt "$target_lvls" && "$placed_this_cycle" -lt "$max_place" ]]; then
        local highest_sell; [[ "$sell_count" -gt 0 ]] && highest_sell=$(printf "%s\n" "${!ACTIVE_SELL_ORDERS[@]}"|sort -nr|head -n1) || highest_sell=$(echo "$current_price + $grid_interval"|bc -l)
        for (( i=1; i<=$((target_lvls - sell_count)) && placed_this_cycle < max_place; i++ )); do
            ! $SCRIPT_RUNNING && break; local new_sell=$(echo "$highest_sell + ($i * $grid_interval)"|bc -l)
            if [[ -z "${ACTIVE_SELL_ORDERS[$new_sell]:-}" ]]; then
                place_order "Sell" "$new_sell" "$order_size" && placed_this_cycle=$((placed_this_cycle+1)) || { log_message "ERROR" "Rebalance Sell failed."; break; }; sleep 0.3
            fi; done; fi
    [[ "$placed_this_cycle" -gt 0 ]] && log_message "INFO" "Rebalance placed $placed_this_cycle orders."
    return 0
}

# --- Function: Adjust Grid Levels Dynamically ---
adjust_grid_levels() {
    [[ "${DYNAMIC_GRID_ENABLED:-false}" != "true" ]] && return 0
    log_message "DEBUG" "Dynamic grid adjustment check..."
    local target_pnl="${GRID_TOTAL_PROFIT_TARGET_USD}" current_pnl="${TOTAL_PNL_USD}" factor="${DYNAMIC_GRID_FACTOR:-0.1}" adj=0
    local base_lvls_for_dynamic="${GRID_LEVELS_CONFIG:-$GRID_LEVELS}"
    [[ "$(echo "$target_pnl <= 0" | bc -l)" -eq 1 ]] && return 0
    local pnl_percent=$(echo "scale=2; $current_pnl / $target_pnl * 100" | bc -l)
    if [[ "$(echo "$current_pnl > ($target_pnl * 0.5)" | bc -l)" -eq 1 ]]; then adj=1; elif [[ "$(echo "$current_pnl < (-$GRID_TOTAL_LOSS_LIMIT_USD * 0.5)" | bc -l)" -eq 1 ]]; then adj=-1; fi
    local new_lvls=$(( base_lvls_for_dynamic + adj )); local min_lvls=1 max_lvls=$(( base_lvls_for_dynamic + 3 ))
    [[ "$new_lvls" -lt "$min_lvls" ]] && new_lvls=$min_lvls; [[ "$new_lvls" -gt "$max_lvls" ]] && new_lvls=$max_lvls
    if [[ "$new_lvls" -ne "$GRID_LEVELS" ]]; then log_message "INFO" "Dynamic Grid: Adjusting target levels from $GRID_LEVELS to $new_lvls (PNL: ${pnl_percent}% of target)"; GRID_LEVELS=$new_lvls; fi
}

# --- Main Execution Loop ---
main() {
    trap cleanup INT TERM EXIT
    manage_wake_lock "acquire"
    load_env; load_config
    check_storage_access
    LOCK_FILE="/tmp/gridbot_$(basename "$0" .sh)_${SYMBOL:-default}.lock"
    if [ -e "$LOCK_FILE" ]; then local owner_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo ""); if [[ -n "$owner_pid" ]] && ps -p "$owner_pid" > /dev/null; then handle_error "Instance running (PID: $owner_pid)." "MAIN"; else log_message "WARNING" "Stale lock file ($LOCK_FILE). Overwriting."; rm -f "$LOCK_FILE"; fi; fi
    echo $$ > "$LOCK_FILE" || handle_error "Cannot create lock file." "MAIN"
    log_message "INFO" "--- Grid Bot Starting (LIVE TRADING) --- Symbol: ${SYMBOL}, Log: ${LOG_LEVEL} @ ${LOG_FILE}"
    check_dependencies; init_global_vars
    declare -g GRID_LEVELS_CONFIG="$GRID_LEVELS" # Store initial configured levels for dynamic grid adjustment
    verify_api_permissions
    sync_server_time || log_message "WARNING" "Initial time sync failed. API calls might be unreliable."
    place_grid_orders || handle_error "Initial grid setup failed. Exiting." "MAIN"

    log_message "INFO" "Entering main loop..."
    local loop_count=0
    while $SCRIPT_RUNNING; do
        loop_count=$((loop_count + 1))
        local loop_start_time=$(date +%s) balance="N/A" current_price="N/A" kline_prices_m=""
        log_message "INFO" "--- Loop Start (#${loop_count}) ---"
        rotate_log
        [[ "$((loop_count % 5))" -eq 0 && "${ENABLE_TIME_SYNC:-true}" == "true" ]] && sync_server_time

        balance_raw=$(get_wallet_balance)
        if [ $? -eq 0 ]; then balance="$balance_raw"; else log_message "ERROR" "Loop: Failed to get balance."; fi
        check_orders || log_message "ERROR" "Loop: Failed to check orders."
        sleep 0.2

        current_price_raw=$(get_market_price)
        if [ $? -eq 0 ]; then current_price="$current_price_raw"; else log_message "ERROR" "Loop: Failed to get market price. Skipping cycle."; sleep "${CHECK_INTERVAL_SECONDS:-20}"; continue; fi

        local lookback_m="${VOLATILITY_LOOKBACK_MINUTES:-60}" limit_m=$(( lookback_m < 200 ? lookback_m : 199 )); [[ "$limit_m" -lt 2 ]] && limit_m=2
        kline_resp_m=$(bybit_request "/v5/market/kline" "GET" "category=linear&symbol=${SYMBOL}&interval=1&limit=${limit_m}") || kline_resp_m=""
        [[ -n "$kline_resp_m" ]] && kline_prices_m=$(echo "$kline_resp_m" | jq -r '.result.list[] | .[4]' | tac | paste -sd' ') || log_message "WARNING" "Loop: Failed to get 1m klines."

        if [[ -n "$kline_prices_m" && -n "$current_price" && "$current_price" != "N/A" ]]; then
            VOLATILITY_MULTIPLIER=$(_calculate_volatility "$kline_prices_m" "$current_price") || VOLATILITY_MULTIPLIER=1
            TREND_BIAS=$(_calculate_trend "$kline_prices_m" "$current_price" "${TREND_SMA_PERIOD:-20}") || TREND_BIAS=0
        else VOLATILITY_MULTIPLIER=1; TREND_BIAS=0; fi

        log_message "INFO" "Status: Bal=${balance} USDT, PNL=${TOTAL_PNL_USD} USD, Fills=${ORDER_FILL_COUNT}, Orders(B/S):${#ACTIVE_BUY_ORDERS[@]}/${#ACTIVE_SELL_ORDERS[@]}, VolMult:$VOLATILITY_MULTIPLIER"
        print_status_dashboard "$balance" "$current_price"

        local profit_target="${GRID_TOTAL_PROFIT_TARGET_USD:-1}" loss_limit="${GRID_TOTAL_LOSS_LIMIT_USD:-0.5}"
        if [[ "$(echo "$TOTAL_PNL_USD >= $profit_target" | bc -l)" -eq 1 ]]; then log_message "INFO" "Profit target reached! PNL=${TOTAL_PNL_USD}"; send_sms_notification "PROFIT $SYMBOL! PNL:$TOTAL_PNL_USD"; handle_error "Profit target." "MAIN" "INFO" 0; fi
        if [[ "$(echo "$TOTAL_PNL_USD <= -$loss_limit" | bc -l)" -eq 1 ]]; then log_message "ERROR" "Loss limit hit! PNL=${TOTAL_PNL_USD}"; send_sms_notification "LOSS $SYMBOL! PNL:$TOTAL_PNL_USD"; handle_error "Loss limit." "MAIN" "ERROR" 2; fi

        adjust_grid_levels
        local total_open_orders=$(( ${#ACTIVE_BUY_ORDERS[@]} + ${#ACTIVE_SELL_ORDERS[@]} ))
        if [[ "$total_open_orders" -lt "${MAX_OPEN_ORDERS:-10}" ]]; then rebalance_grid || log_message "ERROR" "Rebalance failed."; else log_message "WARNING" "Skipping rebalance: Max open orders (${MAX_OPEN_ORDERS:-10})."; fi

        local base_sleep="${CHECK_INTERVAL_SECONDS:-20}" adjusted_sleep=$(echo "scale=2; $base_sleep / ($VOLATILITY_MULTIPLIER + 0.00001)" | bc -l)
        [[ "$(echo "$adjusted_sleep < 5" | bc -l)" -eq 1 ]] && adjusted_sleep=5
        local loop_duration=$(( $(date +%s) - loop_start_time )) sleep_time=$(echo "scale=0; ($adjusted_sleep - $loop_duration)/1" | bc -l)
        if [[ "$(echo "$sleep_time > 0" | bc -l)" -eq 1 ]]; then log_message "DEBUG" "Loop:${loop_duration}s. AdjInt:${adjusted_sleep}s. Sleep:${sleep_time}s."; sleep "$sleep_time"; else log_message "WARNING" "Loop ${loop_duration}s >= adj interval. Min pause."; sleep 1; fi
        log_message "INFO" "--- Loop End ---"
    done
}

# --- Script Entry Point ---
if [ -z "$BASH_VERSION" ]; then { printf "Error: Requires Bash.\n" >&2; exit 1; } fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; fi

main "$@"
exit 0