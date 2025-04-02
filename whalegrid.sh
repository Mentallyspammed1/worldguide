#!/bin/bash
# --- GridBot Enhanced v8 ---
# Description: Advanced Bybit Grid Trading Bot - Trend Bias Utilization & Precise PNL Tracking
# Version: 8.0
# Disclaimer: Trading involves risk. Use this script at your own risk.
#             Test thoroughly in Paper Trading mode and on Bybit Testnet before real funds.

# --- Configuration ---

CONFIG_FILE="gridbot.conf"

# Load configuration or prompt user
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    if [ -z "$BYBIT_API_KEY" ] || [ -z "$BYBIT_API_SECRET" ] || [ -z "$SYMBOL" ] || [ -z "$GRID_LEVELS" ] || [ -z "$GRID_INTERVAL" ] || [ -z "$ORDER_SIZE" ]; then
        log_message "ERROR" "Incomplete configuration in $CONFIG_FILE. Required: BYBIT_API_KEY, BYBIT_API_SECRET, SYMBOL, GRID_LEVELS, GRID_INTERVAL, ORDER_SIZE."
        exit 1
    fi
else
    echo "Configuration file '$CONFIG_FILE' not found. Enter settings manually:"
    read -p "Enter your Bybit API Key: " BYBIT_API_KEY
    read -s -p "Enter your Bybit API Secret: " BYBIT_API_SECRET
    echo ""
    read -p "Enter trading symbol (e.g., BTCUSDT): " SYMBOL
    read -p "Enter number of grid levels (per side, e.g., 5): " GRID_LEVELS
    read -p "Enter base grid interval (price difference, e.g., 100): " GRID_INTERVAL
    read -p "Enter order size (e.g., 0.001): " ORDER_SIZE
    read -p "Enter leverage (e.g., 2): " LEVERAGE
    read -p "Enter max open orders (e.g., 20): " MAX_OPEN_ORDERS
    read -p "Enter min 24h turnover (e.g., 50000000): " MIN_24H_TURNOVER
    read -p "Enter max spread percent (e.g., 0.1): " MAX_SPREAD_PERCENT
    read -p "Enter total profit target USD (e.g., 50): " GRID_TOTAL_PROFIT_TARGET_USD
    read -p "Enter total loss limit USD (e.g., 25): " GRID_TOTAL_LOSS_LIMIT_USD
    read -p "Enter check interval seconds (e.g., 30): " CHECK_INTERVAL_SECONDS
    read -p "Enter log file name (e.g., gridbot.log): " LOG_FILE
    read -p "Enable Paper Trading Mode? (yes/no): " PAPER_TRADING_MODE_INPUT
    read -p "Enable SMS Notifications? (yes/no): " SMS_NOTIFICATIONS_ENABLED_INPUT
    if [[ "$SMS_NOTIFICATIONS_ENABLED_INPUT" == "yes" ]]; then
        read -p "Enter SMS Phone Number (e.g., +15551234567): " SMS_PHONE_NUMBER
    fi

    # Set defaults if not provided
    : ${LEVERAGE:=1}
    : ${MAX_OPEN_ORDERS:=20}
    : ${MIN_24H_TURNOVER:=50000000}
    : ${MAX_SPREAD_PERCENT:=0.1}
    : ${GRID_TOTAL_PROFIT_TARGET_USD:=50}
    : ${GRID_TOTAL_LOSS_LIMIT_USD:=25}
    : ${CHECK_INTERVAL_SECONDS:=30}
    : ${LOG_FILE:="gridbot.log"}
    : ${PAPER_TRADING_MODE:=$([[ "$PAPER_TRADING_MODE_INPUT" == "yes" ]] && echo "true" || echo "false")}
    : ${SMS_NOTIFICATIONS_ENABLED:=$([[ "$SMS_NOTIFICATIONS_ENABLED_INPUT" == "yes" ]] && echo "true" || echo "false")}
fi

# --- Script Settings & Defaults ---
: ${BYBIT_API_URL:="https://api.bybit.com"}       # Default to mainnet API URL

# --- Global Variables ---
declare -A ACTIVE_BUY_ORDERS    # Tracks active buy orders (price -> order_id)
declare -A ACTIVE_SELL_ORDERS   # Tracks active sell orders (price -> order_id)
declare -A TRADE_PAIRS          # Tracks buy-sell pairs for PNL (buy_order_id -> [sell_order_id, buy_price, sell_price])
POSITION_SIZE=0                 # Current position size
ENTRY_PRICE=0                   # Entry price of position
REALIZED_PNL_USD=0              # Accumulated realized PNL in USD
UNREALIZED_PNL_USD=0            # Unrealized PNL in USD
ORDER_FILL_COUNT=0              # Count of filled orders
ORDER_PLACED_COUNT=0            # Count of placed orders
VOLATILITY_MULTIPLIER=1         # Dynamic multiplier for grid spacing
TREND_BIAS=0                    # 0=Neutral, 1=Uptrend, -1=Downtrend

# --- Dependency Check ---
check_dependencies() {
    for cmd in curl jq bc openssl; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: $cmd is not installed. Install it (e.g., sudo apt install $cmd)."
            exit 1
        fi
    done
    if [[ "$SMS_NOTIFICATIONS_ENABLED" == "true" ]] && ! command -v termux-sms-send &> /dev/null; then
        echo "Error: termux-sms-send not installed. Install termux-api (pkg install termux-api) and grant SMS permissions."
        exit 1
    fi
    echo "Dependencies check passed."
}

# --- Logging Function ---
log_message() {
    local level="$1" message="$2" timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    printf "[$timestamp] [$level] %s\n" "$message" >> "$LOG_FILE"
    [[ "$level" == "ERROR" ]] && printf "[$timestamp] [ERROR] %s\n" "$message"
}

# --- SMS Notification ---
send_sms_notification() {
    [[ "$SMS_NOTIFICATIONS_ENABLED" != "true" ]] && return
    local message="$1"
    termux-sms-send -n "$SMS_PHONE_NUMBER" "$message"
    [[ $? -ne 0 ]] && log_message "WARNING" "Failed to send SMS notification."
}

# --- API Request Function (V5) ---
bybit_request() {
    local endpoint="$1" method="${2:-GET}" data="${3:-}" retries=3 delay=5
    while [[ $retries -gt 0 ]]; do
        local timestamp=$(date +%s%3N) recv_window=5000
        local signature_payload="$method$timestamp$recv_window$data"
        [[ "$method" != "GET" ]] && signature_payload="$method$timestamp$recv_window$endpoint$data"
        local signature=$(echo -n "$signature_payload" | openssl dgst -sha256 -hmac "$BYBIT_API_SECRET" | awk '{print $2}')
        local auth_params="api_key=$BYBIT_API_KEY&timestamp=$timestamp&recv_window=$recv_window&sign=$signature"
        local url="$BYBIT_API_URL$endpoint?$auth_params"
        local curl_cmd="curl -s -X $method -H 'Content-Type: application/json' \"$url\""
        [[ "$method" != "GET" ]] && curl_cmd="curl -s -X $method -H 'Content-Type: application/json' -d '$data' \"$url\""
        
        local response=$($curl_cmd)
        if [[ $? -eq 0 && $(echo "$response" | jq -r '.retCode') == "0" ]]; then
            echo "$response"
            return 0
        fi
        log_message "WARNING" "API Request Failed: $(echo "$response" | jq -r '.retMsg'). Retries left: $retries"
        retries=$((retries - 1))
        [[ $retries -gt 0 ]] && sleep $delay
    done
    log_message "ERROR" "API Request failed after retries. Endpoint: $endpoint, Method: $method, Data: $data"
    send_sms_notification "ERROR: API Request failed for $endpoint"
    return 1
}

# --- Get Market Price ---
get_market_price() {
    local response=$(bybit_request "/v5/market/tickers?symbol=$SYMBOL")
    [[ $? -eq 0 ]] && echo "$response" | jq -r '.result.list[0].lastPrice' || return 1
}

# --- Get Ticker Info ---
get_ticker_info() {
    bybit_request "/v5/market/tickers?symbol=$SYMBOL"
}

# --- Get Kline Data ---
get_kline_data() {
    local interval="${1:-15}" limit="${2:-200}"
    bybit_request "/v5/market/kline?symbol=$SYMBOL&interval=$interval&limit=$limit"
}

# --- Check Market Conditions ---
check_market_conditions() {
    local ticker_response=$(get_ticker_info) || { log_message "WARNING" "Failed to get ticker info."; return 1; }
    local turnover24h=$(echo "$ticker_response" | jq -r '.result.list[0].turnover24h')
    local best_bid=$(echo "$ticker_response" | jq -r '.result.list[0].bidPrice')
    local best_ask=$(echo "$ticker_response" | jq -r '.result.list[0].askPrice')
    
    [[ $(echo "$turnover24h < $MIN_24H_TURNOVER" | bc -l) -eq 1 ]] && {
        log_message "WARNING" "24h Turnover ($turnover24h) below threshold ($MIN_24H_TURNOVER)."
        send_sms_notification "WARNING: 24h Turnover below threshold."
        return 1
    }
    local spread_percent=$(echo "scale=4; (($best_ask - $best_bid) / $best_ask) * 100" | bc -l)
    [[ $(echo "$spread_percent > $MAX_SPREAD_PERCENT" | bc -l) -eq 1 ]] && {
        log_message "WARNING" "Spread ($spread_percent%) exceeds threshold ($MAX_SPREAD_PERCENT%)."
        send_sms_notification "WARNING: Spread exceeds threshold."
        return 1
    }
    echo "Market conditions OK (Turnover: $turnover24h, Spread: $spread_percent%)."
    return 0
}

# --- Check Volatility ---
check_volatility() {
    local price_history=() history_count=20
    for i in $(seq 1 $history_count); do
        local price=$(get_market_price) || { log_message "WARNING" "Failed to get price for volatility."; return 1; }
        price_history+=("$price")
        sleep 1
    done
    local sum=0 sum_sq=0 n=$history_count
    for price in "${price_history[@]}"; do
        sum=$(echo "$sum + $price" | bc)
        sum_sq=$(echo "$sum_sq + ($price * $price)" | bc)
    done
    local avg=$(echo "scale=8; $sum / $n" | bc)
    local variance=$(echo "scale=8; ($sum_sq / $n) - ($avg * $avg)" | bc)
    local std_dev=$(echo "scale=8; sqrt($variance)" | bc)
    local threshold=$(echo "scale=8; $avg * 0.005" | bc)
    
    if [[ $(echo "$std_dev > $threshold" | bc -l) -eq 1 ]]; then
        VOLATILITY_MULTIPLIER=$(echo "scale=2; $std_dev / $threshold" | bc)
        [[ $(echo "$VOLATILITY_MULTIPLIER > 2.0" | bc -l) -eq 1 ]] && VOLATILITY_MULTIPLIER=2.0
        log_message "WARNING" "High volatility (Std Dev: $std_dev). Multiplier: $VOLATILITY_MULTIPLIER"
        return 1
    else
        VOLATILITY_MULTIPLIER=1
        log_message "INFO" "Volatility OK (Std Dev: $std_dev)."
        return 0
    fi
}

# --- Calculate EMA ---
calculate_ema() {
    local interval="${1:-15}" period="${2:-20}"
    local kline_response=$(get_kline_data "$interval" "$period") || { log_message "WARNING" "Failed to get kline data."; return 1; }
    local close_prices=($(echo "$kline_response" | jq -r '.result.list[].close'))
    local ema=0 k=$(echo "scale=8; 2 / ($period + 1)" | bc)
    for price in "${close_prices[@]}"; do
        [[ "$ema" == "0" ]] && ema="$price" || ema=$(echo "scale=8; ($price * $k) + ($ema * (1 - $k))" | bc)
    done
    echo "$ema"
}

# --- Check Trend ---
check_trend() {
    local ema_short=$(calculate_ema 15 20) ema_long=$(calculate_ema 60 50) || return 1
    local current_price=$(get_market_price) || { log_message "WARNING" "Failed to get price for trend."; return 1; }
    
    if [[ $(echo "$ema_short > $ema_long" | bc -l) -eq 1 && $(echo "$current_price > $ema_short" | bc -l) -eq 1 ]]; then
        TREND_BIAS=1
        log_message "INFO" "Uptrend detected (EMA Short: $ema_short, EMA Long: $ema_long, Price: $current_price)."
        send_sms_notification "INFO: Uptrend detected."
        echo "Uptrend"
    elif [[ $(echo "$ema_short < $ema_long" | bc -l) -eq 1 && $(echo "$current_price < $ema_short" | bc -l) -eq 1 ]]; then
        TREND_BIAS=-1
        log_message "INFO" "Downtrend detected (EMA Short: $ema_short, EMA Long: $ema_long, Price: $current_price)."
        send_sms_notification "INFO: Downtrend detected."
        echo "Downtrend"
    else
        TREND_BIAS=0
        log_message "INFO" "Neutral trend (EMA Short: $ema_short, EMA Long: $ema_long, Price: $current_price)."
        echo "Neutral"
    fi
    return 0
}

# --- Adjust Grid Interval ---
adjust_grid_interval_dynamic() {
    echo "scale=2; $GRID_INTERVAL * $VOLATILITY_MULTIPLIER" | bc -l
}

# --- Place Grid Orders with Trend Bias ---
place_grid_orders() {
    [[ "$PAPER_TRADING_MODE" == "false" && $(check_market_conditions) -ne 0 ]] && { log_message "WARNING" "Market conditions unfavorable."; return 1; }
    [[ "$PAPER_TRADING_MODE" == "false" ]] && check_volatility
    check_trend || log_message "WARNING" "Trend check failed, using neutral grid."
    
    local current_price=$(get_market_price) || { log_message "ERROR" "Failed to get current price."; return 1; }
    log_message "INFO" "Current Price: $current_price"
    
    [[ "$PAPER_TRADING_MODE" == "false" ]] && set_leverage || log_message "INFO" "[PAPER TRADING] Leverage setting skipped."
    
    local dynamic_grid_interval=$(adjust_grid_interval_dynamic)
    local buy_levels=$GRID_LEVELS sell_levels=$GRID_LEVELS
    
    # Adjust grid based on trend bias
    case $TREND_BIAS in
        1)  # Uptrend: More sell levels
            sell_levels=$((GRID_LEVELS + 2))
            buy_levels=$((GRID_LEVELS - 1))
            log_message "INFO" "Uptrend bias: Sell levels increased to $sell_levels, Buy levels reduced to $buy_levels."
            ;;
        -1) # Downtrend: More buy levels
            buy_levels=$((GRID_LEVELS + 2))
            sell_levels=$((GRID_LEVELS - 1))
            log_message "INFO" "Downtrend bias: Buy levels increased to $buy_levels, Sell levels reduced to $sell_levels."
            ;;
        0)  # Neutral: Balanced
            log_message "INFO" "Neutral trend: Balanced grid."
            ;;
    esac
    
    # Place sell orders
    for ((i=1; i<=sell_levels; i++)); do
        local sell_price=$(echo "$current_price + ($i * $dynamic_grid_interval)" | bc -l)
        [[ ! "$sell_price" =~ ^[0-9.]+$ ]] && { log_message "WARNING" "Invalid sell price: $sell_price"; continue; }
        if [[ ${#ACTIVE_SELL_ORDERS[@]} -lt "$MAX_OPEN_ORDERS" && -z "${ACTIVE_SELL_ORDERS[$sell_price]}" ]]; then
            place_order "Sell" "$sell_price" "$ORDER_SIZE" && {
                local order_id=$(parse_order_id "$response")
                ACTIVE_SELL_ORDERS[$sell_price]="$order_id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Placed SELL order at $sell_price, ID: $order_id"
            }
        fi
    done
    
    # Place buy orders
    for ((i=1; i<=buy_levels; i++)); do
        local buy_price=$(echo "$current_price - ($i * $dynamic_grid_interval)" | bc -l)
        [[ ! "$buy_price" =~ ^[0-9.]+$ ]] && { log_message "WARNING" "Invalid buy price: $buy_price"; continue; }
        if [[ ${#ACTIVE_BUY_ORDERS[@]} -lt "$MAX_OPEN_ORDERS" && -z "${ACTIVE_BUY_ORDERS[$buy_price]}" ]]; then
            place_order "Buy" "$buy_price" "$ORDER_SIZE" && {
                local order_id=$(parse_order_id "$response")
                ACTIVE_BUY_ORDERS[$buy_price]="$order_id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Placed BUY order at $buy_price, ID: $order_id"
            }
        fi
    done
    log_message "INFO" "Grid placement complete. Placed $ORDER_PLACED_COUNT orders."
    ORDER_PLACED_COUNT=0
}

# --- Set Leverage ---
set_leverage() {
    local data="symbol=$SYMBOL&leverage=$LEVERAGE"
    bybit_request "/v5/position/set-leverage" "POST" "$data" && {
        log_message "INFO" "Leverage set to $LEVERAGE x."
        return 0
    } || {
        log_message "ERROR" "Failed to set leverage."
        return 1
    }
}

# --- Place Order ---
place_order() {
    local side="$1" price="$2" qty="$3"
    if [[ "$PAPER_TRADING_MODE" == "true" ]]; then
        response="{\"result\":{\"orderId\":\"PAPER_$RANDOM\"}}"
        log_message "INFO" "[PAPER TRADING] Would place $side order at $price, size: $qty"
        return 0
    fi
    local data="symbol=$SYMBOL&side=$side&orderType=Limit&qty=$qty&price=$price&timeInForce=GTC&leverage=$LEVERAGE"
    response=$(bybit_request "/v5/order/create" "POST" "$data") || {
        log_message "ERROR" "Failed to place $side order at $price."
        return 1
    }
    return 0
}

# --- Parse Order ID ---
parse_order_id() {
    echo "$1" | jq -r '.result.orderId'
}

# --- Cancel Order ---
cancel_order() {
    local order_id="$1"
    [[ "$PAPER_TRADING_MODE" == "true" ]] && { log_message "INFO" "[PAPER TRADING] Would cancel order $order_id"; return 0; }
    local data="symbol=$SYMBOL&orderId=$order_id"
    bybit_request "/v5/order/cancel" "POST" "$data" && {
        log_message "INFO" "Canceled order $order_id"
        return 0
    } || {
        log_message "WARNING" "Failed to cancel order $order_id"
        return 1
    }
}

# --- Cancel All Orders ---
cancel_all_orders() {
    [[ "$PAPER_TRADING_MODE" == "true" ]] && { ACTIVE_BUY_ORDERS=(); ACTIVE_SELL_ORDERS=(); log_message "INFO" "[PAPER TRADING] All orders canceled."; return 0; }
    local data="symbol=$SYMBOL"
    bybit_request "/v5/order/cancel-all" "POST" "$data" && {
        ACTIVE_BUY_ORDERS=()
        ACTIVE_SELL_ORDERS=()
        log_message "INFO" "All orders canceled."
        send_sms_notification "INFO: All orders canceled."
        return 0
    } || {
        log_message "WARNING" "Failed to cancel all orders."
        return 1
    }
}

# --- Get Open Position ---
get_open_position() {
    local response=$(bybit_request "/v5/position/list?symbol=$SYMBOL") || { log_message "WARNING" "Failed to get position data."; return 1; }
    echo "$response"
}

# --- Manage Grid Orders ---
manage_grid_orders() {
    local position_response=$(get_open_position)
    if [[ $? -eq 0 ]]; then
        POSITION_SIZE=$(echo "$position_response" | jq -r '.result.list[0].size')
        ENTRY_PRICE=$(echo "$position_response" | jq -r '.result.list[0].avgEntryPrice')
        UNREALIZED_PNL_USD=$(echo "$position_response" | jq -r '.result.list[0].unrealizedPnl')
    else
        POSITION_SIZE=0 ENTRY_PRICE=0 UNREALIZED_PNL_USD=0
    fi
    
    local current_price=$(get_market_price) || { log_message "WARNING" "Failed to get current price."; return 1; }
    check_and_replenish_orders "Buy" "$current_price"
    check_and_replenish_orders "Sell" "$current_price"
    calculate_and_log_pnl "$current_price"
    check_pnl_limits
}

# --- Check and Replenish Orders ---
check_and_replenish_orders() {
    local side="$1" current_price="$2" array_name="ACTIVE_${side^^}_ORDERS"
    declare -n orders="$array_name"
    
    for price in "${!orders[@]}"; do
        [[ ! "$price" =~ ^[0-9.]+$ ]] && { log_message "WARNING" "Invalid price key: $price"; continue; }
        local order_id="${orders[$price]}"
        local status=$([[ "$PAPER_TRADING_MODE" == "true" ]] && echo "Filled" || echo $(bybit_request "/v5/order/realtime?symbol=$SYMBOL&orderId=$order_id" | jq -r '.result.list[0].orderStatus'))
        
        [[ -z "$status" ]] && { log_message "WARNING" "Failed to get status for $order_id."; continue; }
        if [[ "$status" == "Filled" ]]; then
            log_message "INFO" "$side order at $price (ID: $order_id) FILLED."
            ORDER_FILL_COUNT=$((ORDER_FILL_COUNT + 1))
            track_and_calculate_pnl "$side" "$price" "$order_id" "$current_price"
            replenish_grid_level "$side" "$price" "$current_price"
            unset orders["$price"]
        elif [[ "$status" == "Cancelled" || "$status" == "Rejected" ]]; then
            log_message "WARNING" "$side order at $price (ID: $order_id) $status."
            unset orders["$price"]
        fi
    done
}

# --- Track and Calculate PNL ---
track_and_calculate_pnl() {
    local side="$1" price="$2" order_id="$3" current_price="$4"
    if [[ "$side" == "Buy" ]]; then
        TRADE_PAIRS[$order_id]="$order_id,0,$price,0" # [buy_id, sell_id, buy_price, sell_price]
    else # Sell
        for buy_id in "${!TRADE_PAIRS[@]}"; do
            local pair_data=(${TRADE_PAIRS[$buy_id]//,/ })
            if [[ "${pair_data[1]}" == "0" ]]; then # No sell order paired yet
                TRADE_PAIRS[$buy_id]="${pair_data[0]},$order_id,${pair_data[2]},$price"
                local pnl=$(echo "scale=8; ($price - ${pair_data[2]}) * $ORDER_SIZE" | bc -l)
                REALIZED_PNL_USD=$(echo "$REALIZED_PNL_USD + $pnl" | bc -l)
                log_message "INFO" "Completed trade pair: Buy at ${pair_data[2]}, Sell at $price, PNL: $pnl USD"
                unset TRADE_PAIRS[$buy_id]
                break
            fi
        done
    fi
}

# --- Replenish Grid Level ---
replenish_grid_level() {
    local side="$1" filled_price="$2" current_price="$3"
    local dynamic_grid_interval=$(adjust_grid_interval_dynamic)
    if [[ "$side" == "Buy" ]]; then
        local new_sell_price=$(echo "$filled_price + $dynamic_grid_interval" | bc -l)
        [[ ! "$new_sell_price" =~ ^[0-9.]+$ ]] && { log_message "WARNING" "Invalid new sell price: $new_sell_price"; return; }
        if [[ ${#ACTIVE_SELL_ORDERS[@]} -lt "$MAX_OPEN_ORDERS" && -z "${ACTIVE_SELL_ORDERS[$new_sell_price]}" ]]; then
            place_order "Sell" "$new_sell_price" "$ORDER_SIZE" && {
                local order_id=$(parse_order_id "$response")
                ACTIVE_SELL_ORDERS[$new_sell_price]="$order_id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Replenished SELL at $new_sell_price, ID: $order_id"
            }
        fi
    else # Sell
        local new_buy_price=$(echo "$filled_price - $dynamic_grid_interval" | bc -l)
        [[ ! "$new_buy_price" =~ ^[0-9.]+$ ]] && { log_message "WARNING" "Invalid new buy price: $new_buy_price"; return; }
        if [[ ${#ACTIVE_BUY_ORDERS[@]} -lt "$MAX_OPEN_ORDERS" && -z "${ACTIVE_BUY_ORDERS[$new_buy_price]}" ]]; then
            place_order "Buy" "$new_buy_price" "$ORDER_SIZE" && {
                local order_id=$(parse_order_id "$response")
                ACTIVE_BUY_ORDERS[$new_buy_price]="$order_id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Replenished BUY at $new_buy_price, ID: $order_id"
            }
        fi
    fi
}

# --- Calculate and Log PNL ---
calculate_and_log_pnl() {
    local current_price="$1"
    local total_pnl=$(echo "$UNREALIZED_PNL_USD + $REALIZED_PNL_USD" | bc -l)
    local fill_rate=$( [[ $ORDER_PLACED_COUNT -gt 0 ]] && echo "scale=2; ($ORDER_FILL_COUNT / $ORDER_PLACED_COUNT) * 100" | bc -l || echo 0 )
    
    log_message "INFO" "Current Price: $current_price, Position Size: $POSITION_SIZE"
    log_message "INFO" "Unrealized PNL: $(printf "%.4f" "$UNREALIZED_PNL_USD") USD, Realized PNL: $(printf "%.4f" "$REALIZED_PNL_USD") USD, Total PNL: $(printf "%.4f" "$total_pnl") USD"
    log_message "INFO" "Fill Rate: $(printf "%.2f" "$fill_rate")%, Volatility Multiplier: $VOLATILITY_MULTIPLIER, Trend Bias: $TREND_BIAS"
    
    [[ "$SMS_NOTIFICATIONS_ENABLED" == "true" ]] && send_sms_notification "Price: $current_price\nTotal PNL: $(printf "%.4f" "$total_pnl") USD\nFill Rate: $(printf "%.2f" "$fill_rate")%"
    
    ORDER_FILL_COUNT=0 ORDER_PLACED_COUNT=0
}

# --- Check PNL Limits ---
check_pnl_limits() {
    if [[ $(echo "$REALIZED_PNL_USD >= $GRID_TOTAL_PROFIT_TARGET_USD" | bc -l) -eq 1 ]]; then
        log_message "INFO" "Profit target ($GRID_TOTAL_PROFIT_TARGET_USD USD) reached!"
        send_sms_notification "INFO: Profit target reached!"
        cancel_all_orders
        close_position
        exit 0
    elif [[ $(echo "$UNREALIZED_PNL_USD <= -$GRID_TOTAL_LOSS_LIMIT_USD" | bc -l) -eq 1 ]]; then
        log_message "WARNING" "Loss limit ($GRID_TOTAL_LOSS_LIMIT_USD USD) reached!"
        send_sms_notification "WARNING: Loss limit reached!"
        cancel_all_orders
        close_position
        exit 1
    fi
}

# --- Close Position ---
close_position() {
    [[ $(echo "$POSITION_SIZE == 0" | bc -l) -eq 1 ]] && { log_message "INFO" "No position to close."; return 0; }
    local side=$([[ $(echo "$POSITION_SIZE > 0" | bc -l) -eq 1 ]] && echo "Sell" || echo "Buy")
    local qty=$(echo "$POSITION_SIZE" | sed 's/-//')
    [[ "$PAPER_TRADING_MODE" == "true" ]] && { log_message "INFO" "[PAPER TRADING] Would close position."; return 0; }
    local data="symbol=$SYMBOL&side=$side&orderType=Market&qty=$qty&timeInForce=GTC&reduceOnly=true"
    bybit_request "/v5/order/create" "POST" "$data" && {
        log_message "INFO" "Position closed with market order."
        send_sms_notification "INFO: Position closed."
        return 0
    } || {
        log_message "ERROR" "Failed to close position."
        return 1
    }
}

# --- Main Logic ---
main() {
    check_dependencies
    log_message "INFO" "--- Grid Bot Started (Version 8.0) ---"
    log_message "INFO" "Symbol: $SYMBOL, Grid Levels: $GRID_LEVELS, Interval: $GRID_INTERVAL, Order Size: $ORDER_SIZE, Leverage: $LEVERAGE"
    
    [[ "$PAPER_TRADING_MODE" == "false" && $(check_market_conditions) -ne 0 ]] && { log_message "ERROR" "Market conditions failed."; exit 1; }
    place_grid_orders || { log_message "ERROR" "Failed to place initial grid."; exit 1; }
    
    log_message "INFO" "Starting main loop..."
    send_sms_notification "INFO: Bot started."
    
    while true; do
        manage_grid_orders
        sleep "$CHECK_INTERVAL_SECONDS"
    done
}

# --- Trap Signals ---
trap 'log_message "INFO" "Shutting down..."; cancel_all_orders; close_position; log_message "INFO" "Shutdown complete."; send_sms_notification "INFO: Bot shutdown."; exit 0' SIGINT SIGTERM SIGHUP

# --- Run ---
main "$@"
