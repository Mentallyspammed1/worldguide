#!/bin/bash

# --- Configuration ---
CONFIG_FILE="gridbot.conf"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "\033[33mConfiguration file '$CONFIG_FILE' not found. Creating default configuration...\033[0m"
    cat > "$CONFIG_FILE" << EOL
# Bybit API Credentials
BYBIT_API_KEY="YOUR_API_KEY"
BYBIT_API_SECRET="YOUR_API_SECRET"

# Trading Parameters
SYMBOL="BTCUSDT"
GRID_LEVELS=7          # Number of buy/sell levels on each side initially
GRID_INTERVAL=100      # Price difference between grid levels (in USDT for SYMBOL)
ORDER_SIZE=0.001       # Order quantity in base currency (e.g., BTC for BTCUSDT)
LEVERAGE=20            # Desired leverage

# Market Condition Thresholds (Optional - Set high values to disable)
MAX_SPREAD_PERCENT=0.01    # Maximum allowed spread percentage ((ask-bid)/ask*100)

# Risk Management
GRID_TOTAL_PROFIT_TARGET_USD=10 # Target total profit (realized + unrealized) to stop the bot
GRID_TOTAL_LOSS_LIMIT_USD=5   # Maximum total loss (realized + unrealized) to stop the bot
MAX_OPEN_ORDERS=20            # Maximum number of concurrent open limit orders per side (Buy/Sell)

# Bot Behavior
CHECK_INTERVAL_SECONDS=30   # How often to check orders and PNL (seconds)
LOG_FILE="gridbot.log"
PAPER_TRADING_MODE="true"   # Set to "false" for live trading
SMS_NOTIFICATIONS_ENABLED="false" # Requires termux-api package
SMS_PHONE_NUMBER="+1234567890"  # Your phone number for SMS alerts

# Optional: Bybit API URL (Uncomment for Testnet)
# BYBIT_API_URL="https://api-testnet.bybit.com"
EOL
    echo -e "\033[32mDefault configuration created. Please edit $CONFIG_FILE with your settings.\033[0m"
    exit 0
fi

# --- Dependency Check ---
command -v jq >/dev/null 2>&1 || { echo -e "\033[91mError: 'jq' command not found. Please install it (pkg install jq).\033[0m"; exit 1; }
command -v bc >/dev/null 2>&1 || { echo -e "\033[91mError: 'bc' command not found. Please install it (pkg install bc).\033[0m"; exit 1; }
command -v openssl >/dev/null 2>&1 || { echo -e "\033[91mError: 'openssl' command not found. Please install it (pkg install openssl).\033[0m"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo -e "\033[91mError: 'curl' command not found. Please install it (pkg install curl).\033[0m"; exit 1; }

# Load configuration
source "$CONFIG_FILE"

# Use default API URL if not set
BYBIT_API_URL=${BYBIT_API_URL:-"https://api.bybit.com"}

# --- Global Variables ---
declare -A ACTIVE_BUY_ORDERS  # Stores active buy order IDs keyed by price
declare -A ACTIVE_SELL_ORDERS # Stores active sell order IDs keyed by price
declare -A TRADE_PAIRS        # Tracks open buy orders to match with sells for PNL calc (buy_order_id -> "buy_order_id,0,buy_price,0")
POSITION_SIZE=0
ENTRY_PRICE=0
REALIZED_PNL_USD=0
UNREALIZED_PNL_USD=0
TOTAL_PNL_USD=0
ORDER_FILL_COUNT=0
ORDER_PLACED_COUNT=0          # Counter for orders placed in one cycle
VOLATILITY_MULTIPLIER=1       # Adjusts grid interval based on volatility
TREND_BIAS=0                  # -1: Downtrend, 0: Neutral, 1: Uptrend

# --- Enhanced Logging Function ---
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local color_prefix=""
    local color_suffix="\033[0m" # Reset color

    # Ensure log file exists
    touch "$LOG_FILE"

    case "$level" in
        "INFO")    color_prefix="\033[32m" ;;    # Green
        "WARNING") color_prefix="\033[33m" ;;    # Yellow
        "ERROR")   color_prefix="\033[91m" ;;    # Bright Red
        "DEBUG")   color_prefix="\033[36m" ;;    # Cyan
        *)         level="INFO"; color_prefix="\033[32m" ;; # Default to INFO
    esac

    # Log to file
    echo -e "${timestamp} [${level}] ${message}" >> "$LOG_FILE"
    # Log ERROR, WARNING, and INFO to console
    if [[ "$level" == "ERROR" || "$level" == "WARNING" || "$level" == "INFO" ]]; then
        echo -e "${timestamp} ${color_prefix}[${level}]${color_suffix} ${message}"
    fi
}

# --- Enhanced SMS Notification ---
send_sms_notification() {
    if [ "$SMS_NOTIFICATIONS_ENABLED" == "true" ]; then
        local message="$1"
        if ! command -v termux-sms-send &> /dev/null; then
            log_message "ERROR" "Termux SMS API command 'termux-sms-send' not found. Please install termux-api package (pkg install termux-api) and run termux-setup-storage."
            return 1
        fi

        log_message "INFO" "Attempting to send SMS to $SMS_PHONE_NUMBER: $message"
        termux-sms-send -n "$SMS_PHONE_NUMBER" "$message"
        if [ $? -ne 0 ]; then
            log_message "WARNING" "SMS send command failed. Check termux-api setup and Android permissions for Termux:API app."
        else
            log_message "INFO" "SMS potentially sent successfully."
        fi
    fi
}

# --- Enhanced API Request Handler ---
bybit_request() {
    local endpoint="$1"       # e.g., /v5/order/create
    local method="${2:-GET}"  # GET or POST
    local data="${3:-}"       # JSON string for POST requests
    local retries=3
    local delay=5             # Seconds between retries

    while [ $retries -gt 0 ]; do
        local timestamp=$(date +%s%3N) # Milliseconds timestamp
        local recv_window=5000        # milliseconds

        local signature_payload=""
        local url_params=""
        local request_body=""

        # Prepare signature payload and request body/params based on method
        if [ "$method" == "GET" ]; then
            # For GET, data is appended to the endpoint as query params before signing
            local query_params=$(echo "$data" | sed 's/&/\\&/g') # Escape ampersands if data is provided for GET
            url_params="api_key=$BYBIT_API_KEY×tamp=$timestamp&recv_window=$recv_window${query_params:+&$query_params}"
            signature_payload="${timestamp}${BYBIT_API_KEY}${recv_window}${url_params}"
            url="${BYBIT_API_URL}${endpoint}?${url_params}"
        elif [ "$method" == "POST" ]; then
            # For POST, data is the request body
            url_params="api_key=$BYBIT_API_KEY×tamp=$timestamp&recv_window=$recv_window"
            signature_payload="${timestamp}${BYBIT_API_KEY}${recv_window}${data}"
            request_body="$data"
            url="${BYBIT_API_URL}${endpoint}"
        else
            log_message "ERROR" "Unsupported HTTP method: $method"
            return 1
        fi

        # Generate signature
        local signature=$(echo -n "$signature_payload" | openssl dgst -sha256 -hmac "$BYBIT_API_SECRET" -binary | xxd -p -c 256)

        # Construct curl command
        local curl_cmd="curl -s -X $method"
        curl_cmd+=" -H 'X-BAPI-API-KEY: $BYBIT_API_KEY'"
        curl_cmd+=" -H 'X-BAPI-SIGN: $signature'"
        curl_cmd+=" -H 'X-BAPI-SIGN-TYPE: 2'" # HMAC_SHA256
        curl_cmd+=" -H 'X-BAPI-TIMESTAMP: $timestamp'"
        curl_cmd+=" -H 'X-BAPI-RECV-WINDOW: $recv_window'"
        curl_cmd+=" -H 'Content-Type: application/json'"

        if [ "$method" == "POST" ]; then
             # Append query parameters for authentication to the URL for POST as well
            curl_cmd+=" -d '$request_body' \"${url}?${url_params}\""
        else # GET
            curl_cmd+=" \"$url\"" # URL already includes params
        fi

        log_message "DEBUG" "Curl command: $curl_cmd" # Be careful logging this if sensitive data is involved
        local response=$(eval $curl_cmd)
        local exit_code=$?

        log_message "DEBUG" "API Response: $response"

        if [ $exit_code -eq 0 ] && echo "$response" | jq -e '.retCode == 0' > /dev/null; then
            # Successful API call with retCode 0
            echo "$response"
            return 0
        elif [ $exit_code -ne 0 ]; then
            log_message "WARNING" "Curl command failed with exit code $exit_code. Retries left: $((retries - 1))"
        else
            # API call succeeded but returned an error retCode
            local retMsg=$(echo "$response" | jq -r '.retMsg // "Unknown error"')
            local retCode=$(echo "$response" | jq -r '.retCode // "N/A"')
            log_message "WARNING" "API Error (Code: $retCode): $retMsg. Endpoint: $endpoint. Retries left: $((retries - 1))"
            # Specific handling for order placement errors that shouldn't be retried immediately
            if [[ "$retMsg" == *"Order price is out of permissible range"* ]] || [[ "$retMsg" == *"insufficient balance"* ]]; then
                 log_message "ERROR" "Unrecoverable API error: $retMsg. Stopping retry."
                 retries=0 # Stop retrying
            fi
        fi

        retries=$((retries - 1))
        if [ $retries -gt 0 ]; then
            log_message "INFO" "Retrying in $delay seconds..."
            sleep $delay
        fi
    done

    log_message "ERROR" "API request failed after multiple retries! Endpoint: $endpoint, Method: $method" # Avoid logging data directly
    send_sms_notification "CRITICAL: Bybit API Request failed for $endpoint after retries."
    return 1
}

# --- Enhanced Market Price Function ---
get_market_price() {
    local response=$(bybit_request "/v5/market/tickers" "GET" "category=linear&symbol=$SYMBOL")
    if [ $? -ne 0 ]; then return 1; fi
    local price=$(echo "$response" | jq -r '.result.list[0].lastPrice')
    if [[ "$price" == "null" || -z "$price" ]]; then
        log_message "ERROR" "Could not parse last price from ticker response."
        return 1
    fi
    echo "$price"
    return 0
}

# --- Enhanced Ticker Info Function ---
get_ticker_info() {
    bybit_request "/v5/market/tickers" "GET" "category=linear&symbol=$SYMBOL"
}

# --- Enhanced Kline Data Function ---
get_kline_data() {
    local interval="${1:-15}" # Default to 15 minutes
    local limit="${2:-200}"   # Default to 200 data points
    bybit_request "/v5/market/kline" "GET" "category=linear&symbol=$SYMBOL&interval=$interval&limit=$limit"
}

# --- Enhanced Market Conditions Check ---
check_market_conditions() {
    log_message "DEBUG" "Checking market conditions..."
    local ticker_response=$(get_ticker_info)
    if [ $? -ne 0 ]; then
        log_message "WARNING" "Market condition check failed: Could not fetch ticker info."
        return 1 # Indicate failure
    fi
    
    # Check Spread (Handle potential null prices)
    local best_bid=$(echo "$ticker_response" | jq -r '.result.list[0].bid1Price // "0"')
    local best_ask=$(echo "$ticker_response" | jq -r '.result.list[0].ask1Price // "0"')
    if ! [[ "$best_bid" =~ ^[0-9.]+$ ]] || ! [[ "$best_ask" =~ ^[0-9.]+$ ]] || [ $(echo "$best_ask <= 0" | bc -l) -eq 1 ]; then
        log_message "WARNING" "Market condition check failed: Invalid bid/ask prices ($best_bid / $best_ask)."
        return 1 # Indicate failure
    fi
    if [ $(echo "$best_bid >= $best_ask" | bc -l) -eq 1 ]; then
         log_message "WARNING" "Market condition check failed: Bid price ($best_bid) is not less than Ask price ($best_ask)."
         return 1 # Indicate crossed book or bad data
    fi

    local spread_percent=$(echo "scale=4; if($best_ask > 0) (($best_ask - $best_bid) / $best_ask) * 100 else 0" | bc -l)
    if [ $(echo "$spread_percent > $MAX_SPREAD_PERCENT" | bc -l) -eq 1 ]; then
        log_message "WARNING" "Market Condition Fail: Spread ($spread_percent%) exceeds threshold ($MAX_SPREAD_PERCENT%). Bid: $best_bid, Ask: $best_ask."
        send_sms_notification "WARNING: $SYMBOL Spread high ($spread_percent%)."
        return 1 # Indicate failure
    fi

    log_message "INFO" "Market conditions OK ( Spread: $spread_percent%)."
    return 0 # Indicate success
}


# --- Enhanced Volatility Check (Using ATR) ---
check_volatility_and_atr() {
    log_message "DEBUG" "Calculating ATR for volatility..."
    local interval="15" # 15-minute candles
    local period=14     # Standard ATR period
    local limit=$((period + 1)) # Need 'period' candles + 1 previous for calculation

    local kline_response=$(get_kline_data "$interval" "$limit")
    if [ $? -ne 0 ]; then
        log_message "WARNING" "Volatility check failed: Could not fetch kline data."
        return 1 # Failure
    fi

    # Extract High, Low, Close prices. Note: Bybit kline data is newest first. Reverse it.
    local highs=($(echo "$kline_response" | jq -r '.result.list[].high' | tac))
    local lows=($(echo "$kline_response" | jq -r '.result.list[].low' | tac))
    local closes=($(echo "$kline_response" | jq -r '.result.list[].close' | tac))

    if [ ${#closes[@]} -lt $period ]; then
        log_message "WARNING" "Volatility check failed: Not enough kline data (${#closes[@]}/$period)."
        return 1 # Failure
    fi

    local tr_sum=0
    local atr=0

    # Calculate True Range (TR) for each period
    for (( i=1; i<${#closes[@]}; i++ )); do
        local high_i=${highs[$i]}
        local low_i=${lows[$i]}
        local close_prev=${closes[$((i-1))]}

        local tr1=$(echo "$high_i - $low_i" | bc -l)
        local tr2=$(echo "sqrt(($high_i - $close_prev)^2)" | bc -l) # Absolute value using sqrt(x^2)
        local tr3=$(echo "sqrt(($low_i - $close_prev)^2)" | bc -l) # Absolute value

        local true_range=$tr1
        if [ $(echo "$tr2 > $true_range" | bc -l) -eq 1 ]; then true_range=$tr2; fi
        if [ $(echo "$tr3 > $true_range" | bc -l) -eq 1 ]; then true_range=$tr3; fi

        # Use Wilder's smoothing for ATR
        if [ $i -lt $period ]; then
             tr_sum=$(echo "$tr_sum + $true_range" | bc -l)
             if [ $i -eq $((period - 1)) ]; then # Calculate initial SMA for ATR
                 atr=$(echo "scale=8; $tr_sum / $period" | bc -l)
             fi
        else
             atr=$(echo "scale=8; (($atr * ($period - 1)) + $true_range) / $period" | bc -l)
        fi
    done

    local current_price=$(get_market_price)
    if [ $? -ne 0 ]; then return 1; fi # Failed to get price

    # Define volatility threshold (e.g., ATR as % of price)
    local atr_percent=$(echo "scale=4; if($current_price > 0) ($atr / $current_price) * 100 else 0" | bc -l)
    local high_vol_threshold=1.5 # Example: ATR > 1.5% of price is high volatility

    log_message "INFO" "Volatility Check: Current Price: $current_price, ATR($period): $atr, ATR Percent: $atr_percent%"

    # Adjust multiplier based on volatility (example logic)
    if [ $(echo "$atr_percent > $high_vol_threshold" | bc -l) -eq 1 ]; then
        # Increase multiplier slightly in high vol, up to a max of 2x
        VOLATILITY_MULTIPLIER=$(echo "scale=2; 1 + ($atr_percent - $high_vol_threshold) / $high_vol_threshold" | bc -l)
        if [ $(echo "$VOLATILITY_MULTIPLIER > 2.0" | bc -l) -eq 1 ]; then VOLATILITY_MULTIPLIER=2.0; fi
        log_message "WARNING" "High volatility detected (ATR $atr_percent% > $high_vol_threshold%). Adjusted Grid Interval Multiplier: $VOLATILITY_MULTIPLIER"
    elif [ $(echo "$atr_percent < ($high_vol_threshold / 3)" | bc -l) -eq 1 ]; then
         # Decrease multiplier slightly in very low vol, down to a min of 0.5x
        VOLATILITY_MULTIPLIER=$(echo "scale=2; 1 - ( ($high_vol_threshold / 3) - $atr_percent ) / ($high_vol_threshold / 3)" | bc -l)
        if [ $(echo "$VOLATILITY_MULTIPLIER < 0.5" | bc -l) -eq 1 ]; then VOLATILITY_MULTIPLIER=0.5; fi
        log_message "INFO" "Low volatility detected (ATR $atr_percent% < $(echo "scale=2;$high_vol_threshold/3" | bc -l)%). Adjusted Grid Interval Multiplier: $VOLATILITY_MULTIPLIER"
    else
        VOLATILITY_MULTIPLIER=1.0
        log_message "INFO" "Normal volatility. Grid Interval Multiplier: $VOLATILITY_MULTIPLIER"
    fi

    return 0 # Success
}


# --- Enhanced EMA Calculation ---
calculate_ema() {
    local interval="${1:-15}" # Kline interval (e.g., 15 for 15min)
    local period="${2:-20}"   # EMA period
    local kline_limit=$((period * 2)) # Fetch more data for EMA stability

    log_message "DEBUG" "Calculating EMA($period) on $interval min interval..."
    local kline_response=$(get_kline_data "$interval" "$kline_limit")
    if [ $? -ne 0 ]; then
        log_message "WARNING" "EMA calculation failed: Could not fetch kline data."
        return 1 # Failure
    fi

    # jq '.result.list[] | .[4]' extracts the close price (index 4)
    # tac reverses the order (oldest first) as Bybit returns newest first
    local close_prices_str=$(echo "$kline_response" | jq -r '.result.list[].close')
    if [ -z "$close_prices_str" ]; then
        log_message "WARNING" "EMA calculation failed: No close prices found in kline data."
        return 1
    fi
    local close_prices=($(echo "$close_prices_str" | tac))


    if [ ${#close_prices[@]} -lt $period ]; then
        log_message "WARNING" "EMA calculation failed: Not enough data points (${#close_prices[@]}/$period)."
        return 1 # Failure
    fi

    local ema=0
    local k=$(echo "scale=10; 2 / ($period + 1)" | bc -l) # Smoothing factor
    local one_minus_k=$(echo "scale=10; 1 - $k" | bc -l)

    # Calculate EMA iteratively
    ema=${close_prices[0]} # Start with the first price as the initial EMA
    for (( i=1; i<${#close_prices[@]}; i++ )); do
        local price=${close_prices[$i]}
        if ! [[ "$price" =~ ^[0-9.]+$ ]]; then price=0; fi # Basic validation
        ema=$(echo "scale=8; ($price * $k) + ($ema * $one_minus_k)" | bc -l)
    done

    log_message "DEBUG" "Calculated EMA($period) on $interval min: $ema"
    echo "$ema"
    return 0 # Success
}


# --- Enhanced Trend Check ---
check_trend() {
    log_message "DEBUG" "Checking trend..."
    local ema_short=$(calculate_ema 15 20) # 20-period EMA on 15min chart
    local ema_long=$(calculate_ema 60 50)  # 50-period EMA on 1h chart

    if [ $? -ne 0 ]; then
        log_message "WARNING" "Trend check failed: EMA calculation error."
        TREND_BIAS=0 # Default to neutral on error
        return 1
    fi

    local current_price=$(get_market_price)
    if [ $? -ne 0 ]; then
        log_message "WARNING" "Trend check failed: Could not get current price."
        TREND_BIAS=0 # Default to neutral on error
        return 1
    fi

    # Basic Trend Logic: Price vs EMAs and EMA crossover
    if [ $(echo "$current_price > $ema_short" | bc -l) -eq 1 ] && \
       [ $(echo "$ema_short > $ema_long" | bc -l) -eq 1 ]; then
        TREND_BIAS=1 # Uptrend
        log_message "INFO" "Trend Check: Uptrend detected (Price=$current_price > EMA15_20=$ema_short > EMA60_50=$ema_long)."
        # send_sms_notification "INFO: $SYMBOL Uptrend detected." # Optional: Can be noisy
    elif [ $(echo "$current_price < $ema_short" | bc -l) -eq 1 ] && \
         [ $(echo "$ema_short < $ema_long" | bc -l) -eq 1 ]; then
        TREND_BIAS=-1 # Downtrend
        log_message "INFO" "Trend Check: Downtrend detected (Price=$current_price < EMA15_20=$ema_short < EMA60_50=$ema_long)."
        # send_sms_notification "INFO: $SYMBOL Downtrend detected." # Optional: Can be noisy
    else
        TREND_BIAS=0 # Neutral / Ranging
        log_message "INFO" "Trend Check: Neutral trend detected (Price=$current_price, EMA15_20=$ema_short, EMA60_50=$ema_long)."
    fi
    return 0
}


# --- Enhanced Grid Interval Adjustment ---
adjust_grid_interval_dynamic() {
    local base_interval=$GRID_INTERVAL
    local adjusted_interval=$(echo "scale=2; $base_interval * $VOLATILITY_MULTIPLIER" | bc -l)
    # Ensure minimum interval (e.g., 1 tick size - needs fetching symbol info)
    # For simplicity, let's set a minimum absolute value like 1 USDT
    if [ $(echo "$adjusted_interval < 1" | bc -l) -eq 1 ]; then
        adjusted_interval=1
    fi
    log_message "DEBUG" "Base Interval: $base_interval, Vol Multiplier: $VOLATILITY_MULTIPLIER, Adjusted Interval: $adjusted_interval"
    echo "$adjusted_interval"
}

# --- Enhanced Grid Orders Placement ---
place_grid_orders() {
    log_message "INFO" "Setting up initial grid..."
    ORDER_PLACED_COUNT=0 # Reset counter for this placement cycle

    # Perform checks only if not in paper trading mode
    if [ "$PAPER_TRADING_MODE" == "false" ]; then
        if ! check_market_conditions; then
            log_message "ERROR" "Cannot place initial grid: Market conditions unfavorable. Will retry later."
            return 1
        fi
        if ! check_volatility_and_atr; then
            log_message "WARNING" "Volatility check failed or indicated high volatility. Using calculated multiplier."
            # Proceeding, but using the calculated VOLATILITY_MULTIPLIER
        fi
        if ! check_trend; then
             log_message "WARNING" "Trend check failed. Proceeding with neutral trend bias (0)."
             TREND_BIAS=0
        fi
        # Set leverage before placing orders
        if ! set_leverage; then
             log_message "ERROR" "Cannot place initial grid: Failed to set leverage. Check API key permissions."
             return 1
        fi
    else
        log_message "INFO" "[PAPER TRADING] Skipping market checks, volatility, trend, and leverage setting."
        VOLATILITY_MULTIPLIER=1.0 # Default for paper trading
        TREND_BIAS=0              # Default for paper trading
    fi

    local current_price=$(get_market_price)
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Cannot place initial grid: Failed to get current market price."
        return 1
    fi
    log_message "INFO" "Current Market Price: $current_price"

    # Cancel any existing orders before placing a new grid (safety measure)
    log_message "INFO" "Cancelling potentially existing orders before placing new grid..."
    cancel_all_orders || { log_message "ERROR" "Failed to cancel existing orders. Aborting grid placement."; return 1; }
    sleep 2 # Allow time for cancellation to process

    local dynamic_grid_interval=$(adjust_grid_interval_dynamic)
    log_message "INFO" "Using dynamic grid interval: $dynamic_grid_interval USDT"

    # Adjust grid levels based on trend bias
    local base_levels=$GRID_LEVELS
    local buy_levels=$base_levels
    local sell_levels=$base_levels

    case $TREND_BIAS in
        1) # Uptrend: More sells, fewer buys initially
            sell_levels=$((base_levels + 2))
            buy_levels=$((base_levels > 1 ? base_levels - 1 : 1)) # Ensure at least 1 buy level
            log_message "INFO" "Applying Uptrend Bias: Placing $sell_levels sell levels and $buy_levels buy levels."
            ;;
        -1) # Downtrend: More buys, fewer sells initially
            buy_levels=$((base_levels + 2))
            sell_levels=$((base_levels > 1 ? base_levels - 1 : 1)) # Ensure at least 1 sell level
            log_message "INFO" "Applying Downtrend Bias: Placing $buy_levels buy levels and $sell_levels sell levels."
            ;;
        0) # Neutral trend
            log_message "INFO" "Applying Neutral Bias: Placing $buy_levels buy levels and $sell_levels sell levels."
            ;;
    esac


    # --- Place Sell Orders ---
    log_message "INFO" "Placing $sell_levels Sell orders..."
    for ((i=1; i<=sell_levels; i++)); do
        local sell_price=$(echo "scale=2; $current_price + ($i * $dynamic_grid_interval)" | bc -l)
         # Round sell price UP to required precision (e.g., 2 decimal places for USDT pairs) - Needs symbol info ideally
        sell_price=$(printf "%.2f" "$sell_price") # Basic rounding example

        if ! [[ "$sell_price" =~ ^[0-9.]+$ ]] || [ $(echo "$sell_price <= 0" | bc -l) -eq 1 ]; then
            log_message "WARNING" "Skipping invalid sell price: $sell_price"
            continue
        fi

        # Check against MAX_OPEN_ORDERS limit for sells
        if [ ${#ACTIVE_SELL_ORDERS[@]} -ge $MAX_OPEN_ORDERS ]; then
             log_message "WARNING" "Max open sell orders ($MAX_OPEN_ORDERS) reached. Skipping sell order at $sell_price."
             continue
        fi

        if place_order "Sell" "$sell_price" "$ORDER_SIZE"; then
            local order_id=$(parse_order_id "$response") # 'response' is global from place_order
             if [ -n "$order_id" ] && [ "$order_id" != "null" ]; then
                ACTIVE_SELL_ORDERS["$sell_price"]="$order_id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Placed SELL Order #${i} at $sell_price, Size: $ORDER_SIZE, ID: $order_id"
             else
                log_message "WARNING" "Failed to place or parse SELL order at $sell_price. Response: $response"
             fi
        else
            log_message "WARNING" "Failed to place SELL order at $sell_price."
            # Consider stopping if placement fails critically?
        fi
        sleep 0.2 # Small delay to avoid rate limits
    done

    # --- Place Buy Orders ---
    log_message "INFO" "Placing $buy_levels Buy orders..."
    for ((i=1; i<=buy_levels; i++)); do
        local buy_price=$(echo "scale=2; $current_price - ($i * $dynamic_grid_interval)" | bc -l)
         # Round buy price DOWN to required precision (e.g., 2 decimal places for USDT pairs) - Needs symbol info ideally
         buy_price=$(printf "%.2f" "$buy_price") # Basic rounding example

        if ! [[ "$buy_price" =~ ^[0-9.]+$ ]] || [ $(echo "$buy_price <= 0" | bc -l) -eq 1 ]; then
            log_message "WARNING" "Skipping invalid buy price: $buy_price"
            continue
        fi

         # Check against MAX_OPEN_ORDERS limit for buys
        if [ ${#ACTIVE_BUY_ORDERS[@]} -ge $MAX_OPEN_ORDERS ]; then
             log_message "WARNING" "Max open buy orders ($MAX_OPEN_ORDERS) reached. Skipping buy order at $buy_price."
             continue
        fi

        if place_order "Buy" "$buy_price" "$ORDER_SIZE"; then
            local order_id=$(parse_order_id "$response") # 'response' is global from place_order
             if [ -n "$order_id" ] && [ "$order_id" != "null" ]; then
                ACTIVE_BUY_ORDERS["$buy_price"]="$order_id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Placed BUY Order #${i} at $buy_price, Size: $ORDER_SIZE, ID: $order_id"
             else
                log_message "WARNING" "Failed to place or parse BUY order at $buy_price. Response: $response"
             fi
        else
            log_message "WARNING" "Failed to place BUY order at $buy_price."
             # Consider stopping if placement fails critically?
        fi
         sleep 0.2 # Small delay to avoid rate limits
    done

    log_message "INFO" "Initial grid placement attempt complete. Placed $ORDER_PLACED_COUNT orders."
    return 0
}

# --- Enhanced Leverage Setting ---
set_leverage() {
    if [ "$PAPER_TRADING_MODE" == "true" ]; then
        log_message "INFO" "[PAPER TRADING] Skipping leverage setting."
        return 0
    fi

    log_message "INFO" "Attempting to set leverage for $SYMBOL to $LEVERAGE..."
    # Bybit requires setting buy and sell leverage separately for Hedge Mode,
    # but for One-Way mode, setting it once should suffice. Let's assume One-Way mode.
    # If using Hedge Mode, you might need to call this twice with buyLeverage and sellLeverage.
    local data=$(cat <<EOF
{
    "category": "linear",
    "symbol": "$SYMBOL",
    "buyLeverage": "$LEVERAGE",
    "sellLeverage": "$LEVERAGE"
}
EOF
)
    # Remove newlines from JSON data for the API call
    data=$(echo $data | tr -d '\n' | tr -d ' ')

    local response=$(bybit_request "/v5/position/set-leverage" "POST" "$data")
    if [ $? -eq 0 ]; then
        log_message "INFO" "Leverage successfully set to $LEVERAGE x for $SYMBOL."
        return 0
    else
        log_message "ERROR" "Failed to set leverage for $SYMBOL. Response: $response"
        # Check if leverage is already set to the desired value (common 'error')
        if echo "$response" | jq -e '.retMsg | contains("Leverage not modified")' > /dev/null; then
             log_message "INFO" "Leverage was already set to $LEVERAGE x."
             return 0
        fi
        return 1
    fi
}

# --- Enhanced Order Placement ---
place_order() {
    local side="$1"      # "Buy" or "Sell"
    local price="$2"     # Desired limit price
    local qty="$3"       # Order quantity

    # Basic validation
    if ! [[ "$price" =~ ^[0-9.]+$ ]] || [ $(echo "$price <= 0" | bc -l) -eq 1 ]; then
        log_message "ERROR" "Invalid price ($price) for order placement."
        return 1
    fi
     if ! [[ "$qty" =~ ^[0-9.]+$ ]] || [ $(echo "$qty <= 0" | bc -l) -eq 1 ]; then
        log_message "ERROR" "Invalid quantity ($qty) for order placement."
        return 1
    fi

    # --- Paper Trading Simulation ---
    if [ "$PAPER_TRADING_MODE" == "true" ]; then
        local paper_order_id="PAPER_$(date +%s%N)_$RANDOM"
        # Simulate successful response structure
        response=$(cat <<EOF
{
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "orderId": "$paper_order_id",
        "orderLinkId": ""
    },
    "retExtInfo": {},
    "time": $(date +%s%3N)
}
EOF
)
        log_message "INFO" "[PAPER TRADING] Simulating $side order placement at $price, Size: $qty, Mock ID: $paper_order_id"
        return 0 # Simulate success
    fi

    # --- Live Trading Order Placement ---
    log_message "DEBUG" "Placing live $side order: Symbol=$SYMBOL, Qty=$qty, Price=$price"

    # Ensure price has correct decimal places (needs symbol info - fetch tickSize)
    # Example: Assume 2 decimal places for price formatting
    local formatted_price=$(printf "%.2f" "$price")
    # Ensure qty has correct decimal places (needs symbol info - fetch lotSize)
    # Example: Assume 3 decimal places for BTC quantity formatting
    local formatted_qty=$(printf "%.3f" "$qty")

    local orderLinkId="gridbot_$(date +%s%N)" # Unique client order ID

    local data=$(cat <<EOF
{
    "category": "linear",
    "symbol": "$SYMBOL",
    "side": "$side",
    "orderType": "Limit",
    "qty": "$formatted_qty",
    "price": "$formatted_price",
    "timeInForce": "GTC",
    "orderLinkId": "$orderLinkId"
}
EOF
)
    # Remove newlines/spaces for API call
    data=$(echo $data | tr -d '\n' | tr -d ' ')

    # Make the API request - 'response' variable will be set globally by bybit_request
    response=$(bybit_request "/v5/order/create" "POST" "$data")
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_message "INFO" "Successfully initiated $side order placement at $formatted_price (Qty: $formatted_qty). Response: $response"
        # Further check if the order was actually created (though retCode 0 usually means success)
        local order_id=$(echo "$response" | jq -r '.result.orderId // ""')
         if [ -z "$order_id" ]; then
             log_message "WARNING" "Order placement API call succeeded, but no orderId found in response. Response: $response"
             return 1 # Indicate potential issue
         fi
        return 0 # Success
    else
        log_message "ERROR" "Failed to place $side order at $formatted_price (Qty: $formatted_qty). Response: $response"
        send_sms_notification "ERROR: Failed to place $side order for $SYMBOL at $formatted_price"
        return 1 # Failure
    fi
}

# --- Enhanced Order ID Parsing ---
parse_order_id() {
    # Parses orderId from the globally set 'response' variable
    echo "$response" | jq -r '.result.orderId // ""'
}

# --- Enhanced Order Cancellation ---
cancel_order() {
    local order_id="$1"
    if [ -z "$order_id" ]; then
        log_message "WARNING" "Cancel order called with empty order ID."
        return 1
    fi

    # --- Paper Trading Simulation ---
    if [ "$PAPER_TRADING_MODE" == "true" ]; then
        log_message "INFO" "[PAPER TRADING] Simulating cancellation of order $order_id"
        # Remove from active lists if paper trading
        for price in "${!ACTIVE_BUY_ORDERS[@]}"; do
             if [ "${ACTIVE_BUY_ORDERS[$price]}" == "$order_id" ]; then
                 unset ACTIVE_BUY_ORDERS["$price"]
                 log_message "DEBUG" "[PAPER] Removed buy order $order_id at price $price."
                 break
             fi
        done
         for price in "${!ACTIVE_SELL_ORDERS[@]}"; do
             if [ "${ACTIVE_SELL_ORDERS[$price]}" == "$order_id" ]; then
                 unset ACTIVE_SELL_ORDERS["$price"]
                 log_message "DEBUG" "[PAPER] Removed sell order $order_id at price $price."
                 break
             fi
        done
        return 0 # Simulate success
    fi

    # --- Live Trading Cancellation ---
    log_message "INFO" "Attempting to cancel order $order_id for $SYMBOL..."
    local data=$(cat <<EOF
{
    "category": "linear",
    "symbol": "$SYMBOL",
    "orderId": "$order_id"
}
EOF
)
    data=$(echo $data | tr -d '\n' | tr -d ' ')

    local cancel_response=$(bybit_request "/v5/order/cancel" "POST" "$data")
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
         # Bybit cancel returns the cancelled orderId on success
         local cancelled_id=$(echo "$cancel_response" | jq -r '.result.orderId // ""')
         if [ "$cancelled_id" == "$order_id" ]; then
            log_message "INFO" "Successfully cancelled order $order_id."
            return 0 # Success
         else
            log_message "WARNING" "Order cancellation API call succeeded, but returned unexpected ID ($cancelled_id) or no ID. Expected: $order_id. Response: $cancel_response"
            # Still might be cancelled, treat as potential success but log warning
            return 0
         fi
    else
        log_message "WARNING" "Failed to cancel order $order_id. Response: $cancel_response"
        # Common reason: Order already filled or cancelled. Check message.
        if echo "$cancel_response" | jq -e '.retMsg | test("Order has been finished|Order has been canceled|Order does not exist")' > /dev/null; then
             log_message "INFO" "Order $order_id likely already filled or cancelled."
             return 0 # Treat as success in this context (order is no longer active)
        fi
        return 1 # Failure
    fi
}

# --- Enhanced All Orders Cancellation ---
cancel_all_orders() {
    # --- Paper Trading Simulation ---
    if [ "$PAPER_TRADING_MODE" == "true" ]; then
        log_message "INFO" "[PAPER TRADING] Simulating cancellation of all orders."
        ACTIVE_BUY_ORDERS=()
        ACTIVE_SELL_ORDERS=()
        TRADE_PAIRS=() # Also clear any pending trade pairs
        log_message "INFO" "[PAPER TRADING] Cleared active buy/sell order lists."
        return 0
    fi

    # --- Live Trading Cancellation ---
    log_message "INFO" "Attempting to cancel ALL open orders for $SYMBOL..."
    local data=$(cat <<EOF
{
    "category": "linear",
    "symbol": "$SYMBOL"
}
EOF
)
    data=$(echo $data | tr -d '\n' | tr -d ' ')

    local cancel_response=$(bybit_request "/v5/order/cancel-all" "POST" "$data")
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Response is a list of objects, each with an orderId
        local cancelled_count=$(echo "$cancel_response" | jq -r '.result.list | length')
        log_message "INFO" "Successfully initiated cancellation of $cancelled_count orders for $SYMBOL."
        # Clear local lists after successful API call
        ACTIVE_BUY_ORDERS=()
        ACTIVE_SELL_ORDERS=()
        TRADE_PAIRS=() # Also clear any pending trade pairs
        send_sms_notification "INFO: All $SYMBOL orders cancelled."
        return 0 # Success
    else
        log_message "ERROR" "Failed to cancel all orders for $SYMBOL. Response: $cancel_response"
        # If it failed, do not clear local lists, as orders might still be active
        send_sms_notification "ERROR: Failed to cancel all $SYMBOL orders."
        return 1 # Failure
    fi
}

# --- Enhanced Position Management ---
get_open_position() {
     # --- Paper Trading Simulation ---
     if [ "$PAPER_TRADING_MODE" == "true" ]; then
        log_message "DEBUG" "[PAPER TRADING] Simulating position fetch."
        # In paper mode, we need to calculate the position based on simulated fills
        local simulated_size=0
        local simulated_entry=0
        local buys=0
        local sells=0
        # This is a rough simulation, real PNL/entry needs proper tracking
        # Count buys and sells that were 'filled' (removed from active)
        # This part needs better implementation if paper PNL is critical.
        # For now, just return zero position.
        POSITION_SIZE=0
        ENTRY_PRICE=0
        UNREALIZED_PNL_USD=0
        echo "{ \"result\": { \"list\": [ { \"symbol\": \"$SYMBOL\", \"size\": \"$POSITION_SIZE\", \"avgPrice\": \"$ENTRY_PRICE\", \"unrealisedPnl\": \"$UNREALIZED_PNL_USD\" } ] } }"
        return 0
    fi

    # --- Live Trading Position Fetch ---
    log_message "DEBUG" "Fetching open position for $SYMBOL..."
    local response=$(bybit_request "/v5/position/list" "GET" "category=linear&symbol=$SYMBOL")
    if [ $? -ne 0 ]; then
        log_message "WARNING" "Failed to fetch position data for $SYMBOL."
        # Reset global vars on failure to avoid using stale data
        POSITION_SIZE=0
        ENTRY_PRICE=0
        UNREALIZED_PNL_USD=0
        return 1
    fi

    # Extract position details (assuming one-way mode, so only one entry in the list)
    local pos_data=$(echo "$response" | jq -r '.result.list[0] // {}')
    POSITION_SIZE=$(echo "$pos_data" | jq -r '.size // "0"')
    ENTRY_PRICE=$(echo "$pos_data" | jq -r '.avgPrice // "0"') # Note: Bybit API calls this avgPrice, not entryPrice sometimes
    UNREALIZED_PNL_USD=$(echo "$pos_data" | jq -r '.unrealisedPnl // "0"') # Ensure correct field name

    # Validate numeric values
    if ! [[ "$POSITION_SIZE" =~ ^-?[0-9.]+$ ]]; then POSITION_SIZE=0; fi
    if ! [[ "$ENTRY_PRICE" =~ ^[0-9.]+$ ]]; then ENTRY_PRICE=0; fi
    if ! [[ "$UNREALIZED_PNL_USD" =~ ^-?[0-9.]+$ ]]; then UNREALIZED_PNL_USD=0; fi

    log_message "DEBUG" "Position: Size=$POSITION_SIZE, Entry=$ENTRY_PRICE, Unrealized PNL=$UNREALIZED_PNL_USD USD"
    echo "$response" # Return the full response for potential further use
    return 0
}

# --- Enhanced Grid Orders Management ---
manage_grid_orders() {
    log_message "INFO" "--- Managing Grid Cycle Start ---"
    ORDER_PLACED_COUNT=0 # Reset counter for this cycle

    # 1. Update Position Info
    get_open_position || log_message "WARNING" "Continuing cycle despite position fetch failure."

    # 2. Get Current Price
    local current_price=$(get_market_price)
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Critical: Cannot get current market price. Skipping management cycle."
        return 1
    fi
    log_message "INFO" "Current Price: $current_price"

    # 3. Check and Replenish Orders based on Fills
    local filled_buy_count=0
    local filled_sell_count=0
    check_and_replenish_orders "Buy" "$current_price" filled_buy_count || return 1 # Pass counter by name reference
    check_and_replenish_orders "Sell" "$current_price" filled_sell_count || return 1 # Pass counter by name reference

    local total_fills=$((filled_buy_count + filled_sell_count))
    if [ $total_fills -gt 0 ]; then
         log_message "INFO" "Detected $filled_buy_count filled BUY orders and $filled_sell_count filled SELL orders this cycle."
         ORDER_FILL_COUNT=$((ORDER_FILL_COUNT + total_fills))
         # Optional: Force PNL update after fills
         get_open_position
    fi


    # 4. Calculate and Log PNL
    calculate_and_log_pnl

    # 5. Check PNL Limits
    if check_pnl_limits; then
        # PNL limit hit, close_position_and_stop was called
        log_message "INFO" "PNL limit reached. Exiting bot."
        exit 0 # Exit the script gracefully
    fi

    # 6. Optional: Re-evaluate market conditions/trend periodically
    # This could trigger a full grid reset if conditions change drastically
    # Example: if ! check_market_conditions; then cancel_all_orders; place_grid_orders; fi
    # Example: check_trend # Update TREND_BIAS for replenishment logic

    log_message "INFO" "--- Managing Grid Cycle End (Placed $ORDER_PLACED_COUNT orders) ---"
    echo # Add a blank line for readability
    return 0
}

# --- Enhanced Order Replenishment ---
# Takes side ("Buy" or "Sell"), current price, and a variable name to store fill count
check_and_replenish_orders() {
    local side="$1"
    local current_price="$2"
    local -n fill_counter_ref="$3" # Name reference to the fill counter variable
    fill_counter_ref=0             # Reset counter for this side

    local array_name="ACTIVE_${side^^}_ORDERS"
    declare -n orders_ref="$array_name" # Use name reference for the associative array

    log_message "DEBUG" "Checking ${#orders_ref[@]} active $side orders..."

    # Iterate over a copy of keys, as we might modify the array during iteration
    local prices_to_check=("${!orders_ref[@]}")

    for price in "${prices_to_check[@]}"; do
        # Check if the key still exists (might have been removed in a previous iteration step)
        if [[ -z "${orders_ref[$price]}" ]]; then
             continue
        fi

        local order_id="${orders_ref[$price]}"
        local status="Unknown"

        # --- Paper Trading Simulation ---
        if [ "$PAPER_TRADING_MODE" == "true" ]; then
            # Simulate fills based on price crossing the order level
            if [ "$side" == "Buy" ] && [ $(echo "$current_price <= $price" | bc -l) -eq 1 ]; then
                status="Filled"
                log_message "INFO" "[PAPER TRADING] Simulating BUY fill at price $price (Current: $current_price)"
            elif [ "$side" == "Sell" ] && [ $(echo "$current_price >= $price" | bc -l) -eq 1 ]; then
                status="Filled"
                log_message "INFO" "[PAPER TRADING] Simulating SELL fill at price $price (Current: $current_price)"
            else
                status="New" # Assume still active if price hasn't crossed
            fi
        else
             # --- Live Trading Order Status Check ---
            local response=$(bybit_request "/v5/order/realtime" "GET" "category=linear&symbol=$SYMBOL&orderId=$order_id")
            if [ $? -ne 0 ]; then
                log_message "WARNING" "Failed to get status for $side order $order_id at price $price. Skipping check for this order."
                continue # Skip to next order
            fi
             # Check if list is empty (order might not exist anymore)
            if [[ $(echo "$response" | jq -r '.result.list | length') == "0" ]]; then
                 log_message "WARNING" "$side order $order_id (Price: $price) not found in realtime check. Assuming Filled/Cancelled."
                 # To be safe, assume it's gone and try to replenish if needed. Treat as 'Filled' for logic flow.
                 status="Filled" # Or maybe "NotFound" -> leads to removal
            else
                status=$(echo "$response" | jq -r '.result.list[0].orderStatus // "Unknown"')
            fi
        fi


        # --- Process Order Status ---
        case "$status" in
            "Filled"|"PartiallyFilled") # Treat partially filled as filled for grid logic
                log_message "INFO" "$side order at $price (ID: $order_id) reported as $status."
                fill_counter_ref=$((fill_counter_ref + 1)) # Increment the counter via name reference

                # Track PNL (needs filled price, use order price as approximation for now)
                track_and_calculate_pnl "$side" "$price" "$order_id"

                # Replenish the grid on the opposite side
                replenish_grid_level "$side" "$price" "$current_price" || {
                    log_message "ERROR" "Failed to replenish grid after $side fill at $price. Manual intervention may be needed."
                    # Decide whether to stop the bot or continue
                }

                # Remove the filled order from the active list
                unset orders_ref["$price"]
                log_message "DEBUG" "Removed filled $side order $order_id from active list."
                ;;

            "Cancelled"|"Rejected"|"Expired")
                log_message "WARNING" "$side order at $price (ID: $order_id) has status: $status. Removing from active list."
                # Remove the inactive order
                unset orders_ref["$price"]
                # Optionally, try to replace this cancelled/rejected order? For now, just remove.
                ;;

            "New"|"Untriggered"|"Active"|"PartiallyFilledCanceled")
                # Order is still active or waiting, do nothing
                log_message "DEBUG" "$side order at $price (ID: $order_id) is active ($status)."
                ;;

            "NotFound") # Custom status if realtime check returns empty list
                 log_message "WARNING" "$side order $order_id (Price: $price) not found. Removing from active list."
                 unset orders_ref["$price"]
                 # Maybe try to replenish? Risky without knowing why it disappeared.
                 ;;
            *)
                log_message "WARNING" "Unknown status '$status' for $side order $order_id at price $price. Check API response."
                ;;
        esac
         sleep 0.1 # Small delay between checking orders
    done

    log_message "DEBUG" "Finished checking $side orders. Fill count this cycle: $fill_counter_ref"
    return 0
}

# --- Enhanced PNL Tracking ---
track_and_calculate_pnl() {
    local side="$1"       # Side of the order that got filled ("Buy" or "Sell")
    local filled_price="$2" # Price level of the filled order
    local order_id="$3"   # Order ID of the filled order

    log_message "DEBUG" "Tracking PNL for $side fill at $filled_price (Order ID: $order_id)"

    if [ "$side" == "Buy" ]; then
        # A Buy order filled. Store it, waiting for a corresponding Sell.
        # Format: "buy_order_id,0,buy_price,0" (0 indicates sell order/price not yet matched)
        if [[ -z "${TRADE_PAIRS[$order_id]}" ]]; then
             TRADE_PAIRS[$order_id]="$order_id,0,$filled_price,0"
             log_message "DEBUG" "Recorded open Buy trade leg: ID $order_id, Price $filled_price"
        else
             log_message "WARNING" "Duplicate Buy fill detected for tracking? Order ID: $order_id"
        fi

    elif [ "$side" == "Sell" ]; then
        # A Sell order filled. Try to match it with the oldest open Buy trade leg.
        local matched=false
        local oldest_buy_id=""
        local oldest_buy_price=0

        # Find the oldest unmatched Buy order (could be improved with timestamps if needed)
        # For simplicity, just iterate and find the first unmatched one.
        for buy_id in "${!TRADE_PAIRS[@]}"; do
            local pair_data_str="${TRADE_PAIRS[$buy_id]}"
            # Use parameter expansion for safer splitting
            local buy_order_id_p=${pair_data_str%%,*}
            local temp=${pair_data_str#*,}
            local sell_order_id_p=${temp%%,*}
            local temp=${temp#*,}
            local buy_price_p=${temp%%,*}
            local sell_price_p=${temp#*,}


            if [ "$sell_order_id_p" == "0" ]; then # Found an unmatched Buy leg
                oldest_buy_id="$buy_order_id_p"
                oldest_buy_price="$buy_price_p"
                matched=true
                break
            fi
        done

        if $matched; then
            # Calculate PNL for this completed pair
            # PNL = (Sell Price - Buy Price) * Quantity
            # Note: This calculates PNL per unit. Total PNL depends on position size changes.
            # The REALIZED_PNL_USD should ideally come from trade history or calculations involving fees.
            # This is a simplified grid PNL calculation.
            local pnl_per_unit=$(echo "scale=8; ($filled_price - $oldest_buy_price)" | bc -l)
            # Assuming ORDER_SIZE hasn't changed for this pair
            local pair_pnl=$(echo "scale=8; $pnl_per_unit * $ORDER_SIZE" | bc -l)

            # Accumulate REALIZED PNL (This is an approximation)
            REALIZED_PNL_USD=$(echo "$REALIZED_PNL_USD + $pair_pnl" | bc -l)

            log_message "INFO" "Trade Pair Closed: Buy ID $oldest_buy_id (Price $oldest_buy_price) -> Sell ID $order_id (Price $filled_price). Approx Pair PNL: $pair_pnl USD. Cumulative Realized PNL: $REALIZED_PNL_USD USD"

            # Remove the completed pair from tracking
            unset TRADE_PAIRS[$oldest_buy_id]
        else
            # This should ideally not happen in a pure grid strategy if started flat
            # unless initial grid had more sells, or buys were manually closed.
            log_message "WARNING" "Sell fill occurred (ID $order_id, Price $filled_price), but no open Buy trade leg found to match it. PNL cannot be calculated for this pair."
        fi
    fi
}


# --- Enhanced Grid Level Replenishment ---
replenish_grid_level() {
    local side_filled="$1"   # The side that was just filled ("Buy" or "Sell")
    local filled_price="$2"  # The price level that was filled
    local current_price="$3" # Current market price (for context, not directly used for new price)

    local dynamic_grid_interval=$(adjust_grid_interval_dynamic)
    log_message "DEBUG" "Replenishing grid level after $side_filled fill at $filled_price. Interval: $dynamic_grid_interval"

    if [ "$side_filled" == "Buy" ]; then
        # Buy filled, place a new Sell order one grid level above the filled buy price
        local new_sell_price=$(echo "scale=2; $filled_price + $dynamic_grid_interval" | bc -l)
         # Round UP
        new_sell_price=$(printf "%.2f" "$new_sell_price")

        if ! [[ "$new_sell_price" =~ ^[0-9.]+$ ]] || [ $(echo "$new_sell_price <= 0" | bc -l) -eq 1 ]; then
            log_message "WARNING" "Skipping replenishment: Invalid new sell price calculated: $new_sell_price"
            return 1
        fi

        # Check if an order already exists at this level or if max orders reached
        if [[ -n "${ACTIVE_SELL_ORDERS[$new_sell_price]}" ]]; then
             log_message "INFO" "Replenishment skipped: Sell order already exists at $new_sell_price (ID: ${ACTIVE_SELL_ORDERS[$new_sell_price]})."
             return 0
        fi
         if [ ${#ACTIVE_SELL_ORDERS[@]} -ge $MAX_OPEN_ORDERS ]; then
             log_message "WARNING" "Replenishment skipped: Max open sell orders ($MAX_OPEN_ORDERS) reached. Cannot place new sell at $new_sell_price."
             return 1
        fi

        # Place the new Sell order
        log_message "INFO" "Replenishing: Placing new SELL order at $new_sell_price (Size: $ORDER_SIZE)..."
        if place_order "Sell" "$new_sell_price" "$ORDER_SIZE"; then
            local order_id=$(parse_order_id "$response")
             if [ -n "$order_id" ] && [ "$order_id" != "null" ]; then
                ACTIVE_SELL_ORDERS[$new_sell_price]="$order_id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1)) # Increment cycle counter
                log_message "INFO" "Successfully replenished SELL order at $new_sell_price, ID: $order_id"
                return 0
             else
                 log_message "WARNING" "Failed to place or parse replenished SELL order at $new_sell_price. Response: $response"
                 return 1
             fi
        else
            log_message "ERROR" "Failed to place replenished SELL order at $new_sell_price."
            return 1
        fi

    elif [ "$side_filled" == "Sell" ]; then
        # Sell filled, place a new Buy order one grid level below the filled sell price
        local new_buy_price=$(echo "scale=2; $filled_price - $dynamic_grid_interval" | bc -l)
         # Round DOWN
        new_buy_price=$(printf "%.2f" "$new_buy_price")

        if ! [[ "$new_buy_price" =~ ^[0-9.]+$ ]] || [ $(echo "$new_buy_price <= 0" | bc -l) -eq 1 ]; then
            log_message "WARNING" "Skipping replenishment: Invalid new buy price calculated: $new_buy_price"
            return 1
        fi

        # Check if an order already exists at this level or if max orders reached
        if [[ -n "${ACTIVE_BUY_ORDERS[$new_buy_price]}" ]]; then
             log_message "INFO" "Replenishment skipped: Buy order already exists at $new_buy_price (ID: ${ACTIVE_BUY_ORDERS[$new_buy_price]})."
             return 0
        fi
         if [ ${#ACTIVE_BUY_ORDERS[@]} -ge $MAX_OPEN_ORDERS ]; then
             log_message "WARNING" "Replenishment skipped: Max open buy orders ($MAX_OPEN_ORDERS) reached. Cannot place new buy at $new_buy_price."
             return 1
        fi

        # Place the new Buy order
        log_message "INFO" "Replenishing: Placing new BUY order at $new_buy_price (Size: $ORDER_SIZE)..."
        if place_order "Buy" "$new_buy_price" "$ORDER_SIZE"; then
            local order_id=$(parse_order_id "$response")
             if [ -n "$order_id" ] && [ "$order_id" != "null" ]; then
                ACTIVE_BUY_ORDERS[$new_buy_price]="$order_id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1)) # Increment cycle counter
                log_message "INFO" "Successfully replenished BUY order at $new_buy_price, ID: $order_id"
                return 0
             else
                 log_message "WARNING" "Failed to place or parse replenished BUY order at $new_buy_price. Response: $response"
                 return 1
             fi
        else
            log_message "ERROR" "Failed to place replenished BUY order at $new_buy_price."
            return 1
        fi
    else
        log_message "ERROR" "Invalid side '$side_filled' passed to replenish_grid_level."
        return 1
    fi
}


# --- Calculate and Log PNL ---
calculate_and_log_pnl() {
    # Unrealized PNL is fetched during get_open_position()
    # Realized PNL is approximated in track_and_calculate_pnl()

    # Calculate Total PNL
    TOTAL_PNL_USD=$(echo "$REALIZED_PNL_USD + $UNREALIZED_PNL_USD" | bc -l)

    log_message "INFO" "PNL Status: Realized=$REALIZED_PNL_USD USD, Unrealized=$UNREALIZED_PNL_USD USD, Total=$TOTAL_PNL_USD USD"
    log_message "INFO" "Current Position: Size=$POSITION_SIZE $SYMBOL, Avg Entry=$ENTRY_PRICE"
    log_message "INFO" "Active Orders: ${#ACTIVE_BUY_ORDERS[@]} Buys, ${#ACTIVE_SELL_ORDERS[@]} Sells. Total Fills: $ORDER_FILL_COUNT"
}

# --- Check PNL Limits ---
# Returns 0 if limits not hit, 1 if limits hit (and triggers stop)
check_pnl_limits() {
    # Ensure TOTAL_PNL_USD is up-to-date (calculated in calculate_and_log_pnl)
    local check_needed=false
    if [ $(echo "$GRID_TOTAL_PROFIT_TARGET_USD > 0" | bc -l) -eq 1 ]; then check_needed=true; fi
    if [ $(echo "$GRID_TOTAL_LOSS_LIMIT_USD > 0" | bc -l) -eq 1 ]; then check_needed=true; fi

    if ! $check_needed; then
         # log_message "DEBUG" "PNL limit checks skipped (targets/limits not set)."
         return 0 # Limits not configured
    fi

    # Check Profit Target
    if [ $(echo "$GRID_TOTAL_PROFIT_TARGET_USD > 0" | bc -l) -eq 1 ] && \
       [ $(echo "$TOTAL_PNL_USD >= $GRID_TOTAL_PROFIT_TARGET_USD" | bc -l) -eq 1 ]; then
        log_message "WARNING" "--- PROFIT TARGET REACHED ---"
        log_message "WARNING" "Total PNL ($TOTAL_PNL_USD USD) exceeded target ($GRID_TOTAL_PROFIT_TARGET_USD USD)."
        send_sms_notification "PROFIT TARGET HIT for $SYMBOL! Total PNL: $TOTAL_PNL_USD USD."
        close_position_and_stop "Profit target reached"
        return 1 # Indicates stop triggered
    fi

    # Check Stop Loss
    # Loss limit is positive, compare against negative total PNL
    if [ $(echo "$GRID_TOTAL_LOSS_LIMIT_USD > 0" | bc -l) -eq 1 ] && \
       [ $(echo "$TOTAL_PNL_USD <= -$GRID_TOTAL_LOSS_LIMIT_USD" | bc -l) -eq 1 ]; then
        log_message "ERROR" "--- STOP LOSS TRIGGERED ---"
        log_message "ERROR" "Total PNL ($TOTAL_PNL_USD USD) breached loss limit (-$GRID_TOTAL_LOSS_LIMIT_USD USD)."
        send_sms_notification "STOP LOSS HIT for $SYMBOL! Total PNL: $TOTAL_PNL_USD USD."
        close_position_and_stop "Stop loss triggered"
        return 1 # Indicates stop triggered
    fi

    return 0 # Limits not hit
}

# --- Close Position and Stop Bot ---
close_position_and_stop() {
    local reason="$1"
    log_message "WARNING" "Initiating bot stop sequence. Reason: $reason"

    # 1. Cancel All Open Orders
    log_message "INFO" "Cancelling all open orders..."
    if ! cancel_all_orders; then
        log_message "ERROR" "Failed to cancel all orders during shutdown. Manual intervention required!"
        send_sms_notification "CRITICAL: Failed to cancel $SYMBOL orders on stop! Manual check needed."
        # Continue to attempt position close anyway? Risky. Let's exit.
        exit 1
    fi
    sleep 2 # Allow cancellations to process

    # 2. Close Existing Position (if any)
    # Fetch final position state after cancelling orders
    get_open_position || log_message "WARNING" "Could not fetch final position state before closing."

    if [ $(echo "$POSITION_SIZE != 0" | bc -l) -eq 1 ]; then
        log_message "INFO" "Closing existing position of $POSITION_SIZE $SYMBOL..."
        local close_side=""
        local close_qty=$(echo "sqrt($POSITION_SIZE^2)" | bc -l) # Absolute value for quantity

        if [ $(echo "$POSITION_SIZE > 0" | bc -l) -eq 1 ]; then
            close_side="Sell"
        else # Position size < 0
            close_side="Buy"
        fi

        log_message "INFO" "Placing Market order: Side=$close_side, Qty=$close_qty"

        # --- Paper Trading Simulation ---
        if [ "$PAPER_TRADING_MODE" == "true" ]; then
             log_message "INFO" "[PAPER TRADING] Simulating market $close_side order to close position of $POSITION_SIZE."
             POSITION_SIZE=0 # Simulate closed position
             UNREALIZED_PNL_USD=0 # Reset unrealized PNL
        else
            # --- Live Trading Market Close Order ---
            # Use order/create endpoint with orderType=Market
             local data=$(cat <<EOF
{
    "category": "linear",
    "symbol": "$SYMBOL",
    "side": "$close_side",
    "orderType": "Market",
    "qty": "$close_qty",
    "timeInForce": "GTC"
}
EOF
)
            data=$(echo $data | tr -d '\n' | tr -d ' ')
            local close_response=$(bybit_request "/v5/order/create" "POST" "$data")

             if [ $? -eq 0 ]; then
                 log_message "INFO" "Market close order placed successfully. Response: $close_response"
                 # Position might take a moment to update, PNL calculation might be slightly off immediately
             else
                 log_message "ERROR" "Failed to place market close order! Manual intervention required! Response: $close_response"
                 send_sms_notification "CRITICAL: Failed to close $SYMBOL position! Manual check needed."
                 # Exit even on failure to prevent further trading attempts
                 exit 1
             fi
        fi
    else
        log_message "INFO" "No open position to close."
    fi

    # 3. Log Final State
    log_message "INFO" "Final Realized PNL (approx): $REALIZED_PNL_USD USD"
    log_message "WARNING" "--- Grid Bot Stopped ---"
    send_sms_notification "INFO: $SYMBOL Grid Bot Stopped. Reason: $reason. Final Realized PNL: $REALIZED_PNL_USD USD."

    # 4. Exit Script
    exit 0
}


# --- Cleanup on Exit Signal ---
cleanup_and_exit() {
    echo # Newline for clarity
    log_message "WARNING" "Exit signal received (Ctrl+C or termination)."
    close_position_and_stop "Manual stop signal received"
    exit 0 # Should be already exited by close_position_and_stop
}

# --- Main Execution ---

# Trap signals for graceful shutdown
trap cleanup_and_exit SIGINT SIGTERM

log_message "INFO" "--- Termux Bybit Grid Bot Starting ---"
log_message "INFO" "Symbol: $SYMBOL, Grid Levels: $GRID_LEVELS, Interval: $GRID_INTERVAL, Order Size: $ORDER_SIZE"
log_message "INFO" "Paper Trading: $PAPER_TRADING_MODE, Check Interval: $CHECK_INTERVAL_SECONDS s"
log_message "INFO" "Profit Target: $GRID_TOTAL_PROFIT_TARGET_USD USD, Loss Limit: $GRID_TOTAL_LOSS_LIMIT_USD USD"

# Initial Setup
if [ "$PAPER_TRADING_MODE" == "false" ]; then
    # Validate API Credentials only in live mode
    if [[ "$BYBIT_API_KEY" == "YOUR_API_KEY" || "$BYBIT_API_SECRET" == "YOUR_API_SECRET" ]]; then
        log_message "ERROR" "API Key/Secret not configured in $CONFIG_FILE. Please update."
        exit 1
    fi
     # Test API connection by fetching server time or balance
     log_message "INFO" "Testing API connection..."
     api_test_resp=$(bybit_request "/v5/market/time" "GET")
     if [ $? -ne 0 ]; then
         log_message "ERROR" "API connection test failed. Check credentials, network, and API URL. Response: $api_test_resp"
         exit 1
     fi
     log_message "INFO" "API connection successful. Server Time: $(echo $api_test_resp | jq -r .timeSecond)"
else
     log_message "INFO" "Running in Paper Trading Mode. No API keys needed."
fi

# Place initial grid orders
if ! place_grid_orders; then
     log_message "ERROR" "Failed to place initial grid. Bot cannot start."
     # Attempt cleanup in case some orders were placed
     cancel_all_orders
     exit 1
fi

log_message "INFO" "Initial grid placed. Starting main monitoring loop..."

# Main Loop
while true; do
    if ! manage_grid_orders; then
        log_message "ERROR" "An error occurred during grid management cycle. Retrying after interval."
        # Consider adding a counter for consecutive errors and stopping if too many occur
    fi

    log_message "DEBUG" "Sleeping for $CHECK_INTERVAL_SECONDS seconds..."
    sleep "$CHECK_INTERVAL_SECONDS"
done

# Script should ideally exit via cleanup_and_exit or close_position_and_stop
log_message "WARNING" "Main loop exited unexpectedly."
exit 1
