#!/bin/bash

# --- GridBot Enhanced v8.1 ---
# Description: Advanced Bybit Grid Trading Bot - Trend Bias & Precise
PNL Tracking
# Version: 8.1
# Disclaimer: Trading involves risk. Use at your own risk. Test in
Paper Trading mode first.

# --- Configuration ---

CONFIG_FILE="gridbot.conf"

# Load config or prompt user
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    if [ -z "$BYBIT_API_KEY" ] || [ -z "$BYBIT_API_SECRET" ] || [ -z
"$SYMBOL" ] || [ -z "$GRID_LEVELS" ] || [ -z "$GRID_INTERVAL" ] || [ -z
"$ORDER_SIZE" ]; then
        echo "ERROR: Incomplete configuration in $CONFIG_FILE" >&2
        exit 1
    fi
else
    echo "Config file '$CONFIG_FILE' not found. Enter settings:"
    read -p "Bybit API Key: " BYBIT_API_KEY
    read -s -p "Bybit API Secret: " BYBIT_API_SECRET; echo
    read -p "Trading symbol (e.g., BTCUSDT): " SYMBOL
    read -p "Grid levels per side (e.g., 5): " GRID_LEVELS
    read -p "Base grid interval (e.g., 100): " GRID_INTERVAL
    read -p "Order size (e.g., 0.001): " ORDER_SIZE
    read -p "Leverage (e.g., 2): " LEVERAGE
    read -p "Max open orders (e.g., 20): " MAX_OPEN_ORDERS
    read -p "Min 24h turnover (e.g., 50000000): " MIN_24H_TURNOVER
    read -p "Max spread percent (e.g., 0.1): " MAX_SPREAD_PERCENT
    read -p "Profit target USD (e.g., 50): "
GRID_TOTAL_PROFIT_TARGET_USD
    read -p "Loss limit USD (e.g., 25): " GRID_TOTAL_LOSS_LIMIT_USD
    read -p "Check interval seconds (e.g., 30): "
CHECK_INTERVAL_SECONDS
    read -p "Log file (e.g., gridbot.log): " LOG_FILE
    read -p "Enable Paper Trading? (yes/no): " PAPER_TRADING_MODE_INPUT
    read -p "Enable SMS Notifications? (yes/no): "
SMS_NOTIFICATIONS_ENABLED_INPUT
    [ "$SMS_NOTIFICATIONS_ENABLED_INPUT" = "yes" ] && read -p "SMS
Phone Number (e.g., +15551234567): " SMS_PHONE_NUMBER

    # Defaults
    LEVERAGE=${LEVERAGE:-1}
    MAX_OPEN_ORDERS=${MAX_OPEN_ORDERS:-20}
    MIN_24H_TURNOVER=${MIN_24H_TURNOVER:-50000000}
    MAX_SPREAD_PERCENT=${MAX_SPREAD_PERCENT:-0.1}
    GRID_TOTAL_PROFIT_TARGET_USD=${GRID_TOTAL_PROFIT_TARGET_USD:-50}
    GRID_TOTAL_LOSS_LIMIT_USD=${GRID_TOTAL_LOSS_LIMIT_USD:-25}
    CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-30}
    LOG_FILE=${LOG_FILE:-gridbot.log}
    PAPER_TRADING_MODE=$([ "$PAPER_TRADING_MODE_INPUT" = "yes" ] &&
echo "true" || echo "false")
    SMS_NOTIFICATIONS_ENABLED=$([ "$SMS_NOTIFICATIONS_ENABLED_INPUT" =
"yes" ] && echo "true" || echo "false")
fi

# --- Defaults ---
BYBIT_API_URL=${BYBIT_API_URL:-"https://api.bybit.com"}

# --- Global Variables ---
declare -A ACTIVE_BUY_ORDERS ACTIVE_SELL_ORDERS TRADE_PAIRS
POSITION_SIZE=0 ENTRY_PRICE=0 REALIZED_PNL_USD=0 UNREALIZED_PNL_USD=0
ORDER_FILL_COUNT=0 ORDER_PLACED_COUNT=0 VOLATILITY_MULTIPLIER=1
TREND_BIAS=0

# --- Dependency Check ---
check_dependencies() {
    for cmd in curl jq bc openssl; do
        command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not
installed" >&2; exit 1; }
    done
    if [ "$SMS_NOTIFICATIONS_ENABLED" = "true" ]; then
        command -v termux-sms-send >/dev/null 2>&1 || { echo "Error:
termux-sms-send not installed" >&2; exit 1; }
    fi
    echo "Dependencies OK"
}

# --- Logging ---
log_message() {
    local level="$1" msg="$2" ts=$(date '+%Y-%m-%d %H:%M:%S')
    printf "[%s] [%s] %s\n" "$ts" "$level" "$msg" >> "$LOG_FILE"
    [ "$level" = "ERROR" ] && printf "[%s] [ERROR] %s\n" "$ts" "$msg"
>&2
}

# --- SMS Notification ---
send_sms_notification() {
    [ "$SMS_NOTIFICATIONS_ENABLED" != "true" ] && return
    local msg="$1"
    termux-sms-send -n "$SMS_PHONE_NUMBER" "$msg" || log_message
"WARNING" "Failed to send SMS"
}

# --- API Request ---
bybit_request() {
    local endpoint="$1" method="${2:-GET}" data="${3:-}" retries=3
delay=5
    while [ $retries -gt 0 ]; do
        local ts=$(date +%s%3N) recv=5000
        local payload="$method$ts$recv$data"
        [ "$method" != "GET" ] &&
payload="$method$ts$recv$endpoint$data"
        local sign=$(echo -n "$payload" | openssl dgst -sha256 -hmac
"$BYBIT_API_SECRET" | awk '{print $2}')
        local url="$BYBIT_API_URL$endpoint?
api_key=$BYBIT_API_KEY&timestamp=$ts&recv_window=$recv&sign=$sign"
        local cmd="curl -s -X $method -H 'Content-Type: application/
json'"
        [ "$method" != "GET" ] && cmd="$cmd -d '$data'"
        cmd="$cmd '$url'"

        local resp=$($cmd)
        local curl_exit=$?
        if [ $curl_exit -eq 0 ]; then
            local ret_code=$(echo "$resp" | jq -r '.retCode' 2>/dev/
null)
            if [ "$ret_code" = "0" ]; then
                echo "$resp"
                return 0
            else
                log_message "WARNING" "API error: $(echo "$resp" | jq
-r '.retMsg')"
            fi
        else
            log_message "WARNING" "curl failed: $curl_exit"
        fi
        retries=$((retries - 1))
        [ $retries -gt 0 ] && sleep $delay
    done
    log_message "ERROR" "API failed after retries: $endpoint"
    send_sms_notification "ERROR: API failed for $endpoint"
    return 1
}

# --- Get Market Price ---
get_market_price() {
    local resp=$(bybit_request "/v5/market/tickers?symbol=$SYMBOL") ||
return 1
    echo "$resp" | jq -r '.result.list[0].lastPrice' 2>/dev/null ||
{ log_message "ERROR" "Failed to parse price"; return 1; }
}

# --- Get Ticker Info ---
get_ticker_info() {
    bybit_request "/v5/market/tickers?symbol=$SYMBOL"
}

# --- Get Kline Data ---
get_kline_data() {
    local interval="${1:-15}" limit="${2:-200}"
    bybit_request "/v5/market/kline?
symbol=$SYMBOL&interval=$interval&limit=$limit"
}

# --- Check Market Conditions ---
check_market_conditions() {
    local resp=$(get_ticker_info) || { log_message "WARNING" "Failed to
get ticker"; return 1; }
    local turnover=$(echo "$resp" | jq -r
'.result.list[0].turnover24h')
    local bid=$(echo "$resp" | jq -r '.result.list[0].bidPrice')
    local ask=$(echo "$resp" | jq -r '.result.list[0].askPrice')

    [ "$(echo "$turnover < $MIN_24H_TURNOVER" | bc -l)" -eq 1 ] && {
        log_message "WARNING" "Turnover ($turnover) below
$MIN_24H_TURNOVER"
        send_sms_notification "WARNING: Low turnover"
        return 1
    }
    local spread=$(echo "scale=4; (($ask - $bid) / $ask) * 100" | bc -l
2>/dev/null)
    [ "$(echo "$spread > $MAX_SPREAD_PERCENT" | bc -l)" -eq 1 ] && {
        log_message "WARNING" "Spread ($spread%) exceeds
$MAX_SPREAD_PERCENT%"
        send_sms_notification "WARNING: High spread"
        return 1
    }
    echo "Market OK: Turnover $turnover, Spread $spread%"
    return 0
}

# --- Check Volatility ---
check_volatility() {
    local prices=() count=20
    for i in $(seq 1 $count); do
        local price=$(get_market_price) || { log_message "WARNING"
"Failed to get price"; return 1; }
        prices+=("$price")
        sleep 1
    done
    local sum=0 sum_sq=0
    for price in "${prices[@]}"; do
        sum=$(echo "$sum + $price" | bc -l)
        sum_sq=$(echo "$sum_sq + ($price * $price)" | bc -l)
    done
    local avg=$(echo "scale=8; $sum / $count" | bc -l)
    local var=$(echo "scale=8; ($sum_sq / $count) - ($avg * $avg)" | bc
-l)
    local std=$(echo "scale=8; sqrt($var)" | bc -l 2>/dev/null)
    local thresh=$(echo "scale=8; $avg * 0.005" | bc -l)

    if [ "$(echo "$std > $thresh" | bc -l)" -eq 1 ]; then
        VOLATILITY_MULTIPLIER=$(echo "scale=2; $std / $thresh" | bc -l)
        [ "$(echo "$VOLATILITY_MULTIPLIER > 2" | bc -l)" -eq 1 ] &&
VOLATILITY_MULTIPLIER=2
        log_message "WARNING" "High volatility: Std $std, Multiplier
$VOLATILITY_MULTIPLIER"
        return 1
    fi
    VOLATILITY_MULTIPLIER=1
    log_message "INFO" "Volatility OK: Std $std"
    return 0
}

# --- Calculate EMA ---
calculate_ema() {
    local interval="${1:-15}" period="${2:-20}"
    local resp=$(get_kline_data "$interval" "$period") || { log_message
"WARNING" "Failed to get kline"; return 1; }
    local prices=($(echo "$resp" | jq -r '.result.list[].close'))
    local ema=0 k=$(echo "scale=8; 2 / ($period + 1)" | bc -l)
    for price in "${prices[@]}"; do
        [ "$ema" = "0" ] && ema="$price" || ema=$(echo "scale=8;
($price * $k) + ($ema * (1 - $k))" | bc -l)
    done
    echo "$ema"
}

# --- Check Trend ---
check_trend() {
    local ema_short=$(calculate_ema 15 20) || return 1
    local ema_long=$(calculate_ema 60 50) || return 1
    local price=$(get_market_price) || { log_message "WARNING" "Failed
to get price"; return 1; }

    if [ "$(echo "$ema_short > $ema_long" | bc -l)" -eq 1 ] &&
[ "$(echo "$price > $ema_short" | bc -l)" -eq 1 ]; then
        TREND_BIAS=1
        log_message "INFO" "Uptrend: EMA Short $ema_short, EMA Long
$ema_long, Price $price"
        send_sms_notification "INFO: Uptrend"
    elif [ "$(echo "$ema_short < $ema_long" | bc -l)" -eq 1 ] &&
[ "$(echo "$price < $ema_short" | bc -l)" -eq 1 ]; then
        TREND_BIAS=-1
        log_message "INFO" "Downtrend: EMA Short $ema_short, EMA Long
$ema_long, Price $price"
        send_sms_notification "INFO: Downtrend"
    else
        TREND_BIAS=0
        log_message "INFO" "Neutral: EMA Short $ema_short, EMA Long
$ema_long, Price $price"
    fi
    return 0
}

# --- Adjust Grid Interval ---
adjust_grid_interval() {
    echo "scale=2; $GRID_INTERVAL * $VOLATILITY_MULTIPLIER" | bc -l
}

# --- Place Grid Orders ---
place_grid_orders() {
    [ "$PAPER_TRADING_MODE" = "false" ] && { check_market_conditions ||
return 1; }
    [ "$PAPER_TRADING_MODE" = "false" ] && check_volatility
    check_trend || log_message "WARNING" "Trend check failed, using
neutral grid"

    local price=$(get_market_price) || { log_message "ERROR" "Failed to
get price"; return 1; }
    log_message "INFO" "Current Price: $price"

    [ "$PAPER_TRADING_MODE" = "false" ] && { set_leverage || return
1; }

    local interval=$(adjust_grid_interval)
    local buy_levels=$GRID_LEVELS sell_levels=$GRID_LEVELS

    case $TREND_BIAS in
        1) sell_levels=$((GRID_LEVELS + 2)); buy_levels=$((GRID_LEVELS
- 1)); log_message "INFO" "Uptrend: $sell_levels sell, $buy_levels
buy";;
        -1) buy_levels=$((GRID_LEVELS + 2)); sell_levels=$((GRID_LEVELS
- 1)); log_message "INFO" "Downtrend: $buy_levels buy, $sell_levels
sell";;
        0) log_message "INFO" "Neutral grid";;
    esac

    for i in $(seq 1 $sell_levels); do
        local sell_price=$(echo "$price + ($i * $interval)" | bc -l)
        [ -z "${ACTIVE_SELL_ORDERS[$sell_price]}" ] &&
[ ${#ACTIVE_SELL_ORDERS[@]} -lt "$MAX_OPEN_ORDERS" ] && {
            place_order "Sell" "$sell_price" "$ORDER_SIZE" && {
                local id=$(parse_order_id "$response")
                ACTIVE_SELL_ORDERS[$sell_price]="$id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Placed SELL at $sell_price, ID:
$id"
            }
        }
    done

    for i in $(seq 1 $buy_levels); do
        local buy_price=$(echo "$price - ($i * $interval)" | bc -l)
        [ -z "${ACTIVE_BUY_ORDERS[$buy_price]}" ] &&
[ ${#ACTIVE_BUY_ORDERS[@]} -lt "$MAX_OPEN_ORDERS" ] && {
            place_order "Buy" "$buy_price" "$ORDER_SIZE" && {
                local id=$(parse_order_id "$response")
                ACTIVE_BUY_ORDERS[$buy_price]="$id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Placed BUY at $buy_price, ID: $id"
            }
        }
    done
    log_message "INFO" "Grid placed: $ORDER_PLACED_COUNT orders"
    ORDER_PLACED_COUNT=0
}

# --- Set Leverage ---
set_leverage() {
    local data="symbol=$SYMBOL&leverage=$LEVERAGE"
    bybit_request "/v5/position/set-leverage" "POST" "$data" && {
        log_message "INFO" "Leverage set to $LEVERAGE"
        return 0
    } || {
        log_message "ERROR" "Failed to set leverage"
        return 1
    }
}

# --- Place Order ---
place_order() {
    local side="$1" price="$2" qty="$3"
    if [ "$PAPER_TRADING_MODE" = "true" ]; then
        response='{"retCode":0,"result":{"orderId":"PAPER_'$RANDOM'"}}'
        log_message "INFO" "[PAPER] $side order at $price, size: $qty"
        return 0
    fi
    local
data="symbol=$SYMBOL&side=$side&orderType=Limit&qty=$qty&price=$price&t
imeInForce=GTC&leverage=$LEVERAGE"
    response=$(bybit_request "/v5/order/create" "POST" "$data") || {
        log_message "ERROR" "Failed to place $side order at $price"
        return 1
    }
    return 0
}

# --- Parse Order ID ---
parse_order_id() {
    echo "$1" | jq -r '.result.orderId' 2>/dev/null || echo "ERROR"
}

# --- Cancel Order ---
cancel_order() {
    local id="$1"
    [ "$PAPER_TRADING_MODE" = "true" ] && { log_message "INFO" "[PAPER]
Canceled $id"; return 0; }
    local data="symbol=$SYMBOL&orderId=$id"
    bybit_request "/v5/order/cancel" "POST" "$data" && {
        log_message "INFO" "Canceled $id"
        return 0
    } || {
        log_message "WARNING" "Failed to cancel $id"
        return 1
    }
}

# --- Cancel All Orders ---
cancel_all_orders() {
    if [ "$PAPER_TRADING_MODE" = "true" ]; then
        ACTIVE_BUY_ORDERS=() ACTIVE_SELL_ORDERS=()
        log_message "INFO" "[PAPER] All orders canceled"
        return 0
    fi
    local data="symbol=$SYMBOL"
    bybit_request "/v5/order/cancel-all" "POST" "$data" && {
        ACTIVE_BUY_ORDERS=() ACTIVE_SELL_ORDERS=()
        log_message "INFO" "All orders canceled"
        send_sms_notification "INFO: All orders canceled"
        return 0
    } || {
        log_message "WARNING" "Failed to cancel all orders"
        return 1
    }
}

# --- Get Open Position ---
get_open_position() {
    local resp=$(bybit_request "/v5/position/list?symbol=$SYMBOL") ||
{ log_message "WARNING" "Failed to get position"; return 1; }
    echo "$resp"
}

# --- Manage Grid Orders ---
manage_grid_orders() {
    local pos=$(get_open_position)
    if [ $? -eq 0 ]; then
        POSITION_SIZE=$(echo "$pos" | jq -r '.result.list[0].size')
        ENTRY_PRICE=$(echo "$pos" | jq -r
'.result.list[0].avgEntryPrice')
        UNREALIZED_PNL_USD=$(echo "$pos" | jq -r
'.result.list[0].unrealizedPnl')
    else
        POSITION_SIZE=0 ENTRY_PRICE=0 UNREALIZED_PNL_USD=0
    fi

    local price=$(get_market_price) || { log_message "WARNING" "Failed
to get price"; return 1; }
    check_and_replenish_orders "Buy" "$price"
    check_and_replenish_orders "Sell" "$price"
    calculate_and_log_pnl "$price"
    check_pnl_limits
}

# --- Check and Replenish Orders ---
check_and_replenish_orders() {
    local side="$1" price="$2" array="ACTIVE_${side^^}_ORDERS"
    declare -n orders="$array"

    for p in "${!orders[@]}"; do
        local id="${orders[$p]}"
        local status=$([ "$PAPER_TRADING_MODE" = "true" ] && echo
"Filled" || echo $(bybit_request "/v5/order/realtime?
symbol=$SYMBOL&orderId=$id" | jq -r '.result.list[0].orderStatus'))
        [ -z "$status" ] && { log_message "WARNING" "Failed to get
status for $id"; continue; }

        if [ "$status" = "Filled" ]; then
            log_message "INFO" "$side order at $p (ID: $id) filled"
            ORDER_FILL_COUNT=$((ORDER_FILL_COUNT + 1))
            track_and_calculate_pnl "$side" "$p" "$id" "$price"
            replenish_grid_level "$side" "$p" "$price"
            unset orders["$p"]
        elif [ "$status" = "Cancelled" ] || [ "$status" = "Rejected" ];
then
            log_message "WARNING" "$side order $id $status"
            unset orders["$p"]
        fi
    done
}

# --- Track and Calculate PNL ---
track_and_calculate_pnl() {
    local side="$1" price="$2" id="$3" current="$4"
    if [ "$side" = "Buy" ]; then
        TRADE_PAIRS[$id]="$id,0,$price,0"
    else
        for buy_id in "${!TRADE_PAIRS[@]}"; do
            local pair=(${TRADE_PAIRS[$buy_id]//,/ })
            if [ "${pair[1]}" = "0" ]; then
                TRADE_PAIRS[$buy_id]="${pair[0]},$id,${pair[2]},$price"
                local pnl=$(echo "scale=8; ($price - ${pair[2]}) *
$ORDER_SIZE" | bc -l)
                REALIZED_PNL_USD=$(echo "$REALIZED_PNL_USD + $pnl" | bc
-l)
                log_message "INFO" "Trade pair: Buy ${pair[2]}, Sell
$price, PNL: $pnl USD"
                unset TRADE_PAIRS[$buy_id]
                break
            fi
        done
    fi
}

# --- Replenish Grid Level ---
replenish_grid_level() {
    local side="$1" filled="$2" current="$3"
interval=$(adjust_grid_interval)
    if [ "$side" = "Buy" ]; then
        local new=$(echo "$filled + $interval" | bc -l)
        [ ${#ACTIVE_SELL_ORDERS[@]} -lt "$MAX_OPEN_ORDERS" ] && [ -z
"${ACTIVE_SELL_ORDERS[$new]}" ] && {
            place_order "Sell" "$new" "$ORDER_SIZE" && {
                local id=$(parse_order_id "$response")
                ACTIVE_SELL_ORDERS[$new]="$id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Replenished SELL at $new, ID: $id"
            }
        }
    else
        local new=$(echo "$filled - $interval" | bc -l)
        [ ${#ACTIVE_BUY_ORDERS[@]} -lt "$MAX_OPEN_ORDERS" ] && [ -z
"${ACTIVE_BUY_ORDERS[$new]}" ] && {
            place_order "Buy" "$new" "$ORDER_SIZE" && {
                local id=$(parse_order_id "$response")
                ACTIVE_BUY_ORDERS[$new]="$id"
                ORDER_PLACED_COUNT=$((ORDER_PLACED_COUNT + 1))
                log_message "INFO" "Replenished BUY at $new, ID: $id"
            }
        }
    fi
}

# --- Calculate and Log PNL ---
calculate_and_log_pnl() {
    local price="$1" total=$(echo "$UNREALIZED_PNL_USD +
$REALIZED_PNL_USD" | bc -l)
    local fill_rate=$([ $ORDER_PLACED_COUNT -gt 0 ] && echo "scale=2;
($ORDER_FILL_COUNT / $ORDER_PLACED_COUNT) * 100" | bc -l || echo 0)

    log_message "INFO" "Price: $price, Position: $POSITION_SIZE"
    log_message "INFO" "Unrealized: $(printf "%.4f"
"$UNREALIZED_PNL_USD") USD, Realized: $(printf "%.4f"
"$REALIZED_PNL_USD") USD, Total: $(printf "%.4f" "$total") USD"
    log_message "INFO" "Fill Rate: $(printf "%.2f" "$fill_rate")%,
Volatility: $VOLATILITY_MULTIPLIER, Trend: $TREND_BIAS"

    [ "$SMS_NOTIFICATIONS_ENABLED" = "true" ] && send_sms_notification
"Price: $price\nTotal PNL: $(printf "%.4f" "$total") USD\nFill:
$(printf "%.2f" "$fill_rate")%"

    ORDER_FILL_COUNT=0 ORDER_PLACED_COUNT=0
}

# --- Check PNL Limits ---
check_pnl_limits() {
    if [ "$(echo "$REALIZED_PNL_USD >= $GRID_TOTAL_PROFIT_TARGET_USD" |
bc -l)" -eq 1 ]; then
        log_message "INFO" "Profit target $GRID_TOTAL_PROFIT_TARGET_USD
USD reached"
        send_sms_notification "INFO: Profit target reached"
        cancel_all_orders
        close_position
        exit 0
    elif [ "$(echo "$UNREALIZED_PNL_USD <= -$GRID_TOTAL_LOSS_LIMIT_USD"
| bc -l)" -eq 1 ]; then
        log_message "WARNING" "Loss limit $GRID_TOTAL_LOSS_LIMIT_USD
USD reached"
        send_sms_notification "WARNING: Loss limit reached"
        cancel_all_orders
        close_position
        exit 1
    fi
}

# --- Close Position ---
close_position() {
    [ "$(echo "$POSITION_SIZE == 0" | bc -l)" -eq 1 ] && { log_message
"INFO" "No position to close"; return 0; }
    local side=$([ "$(echo "$POSITION_SIZE > 0" | bc -l)" -eq 1 ] &&
echo "Sell" || echo "Buy")
    local qty=$(echo "$POSITION_SIZE" | sed 's/-//')
    [ "$PAPER_TRADING_MODE" = "true" ] && { log_message "INFO" "[PAPER]
Closed position"; return 0; }
    local
data="symbol=$SYMBOL&side=$side&orderType=Market&qty=$qty&timeInForce=G
TC&reduceOnly=true"
    bybit_request "/v5/order/create" "POST" "$data" && {
        log_message "INFO" "Position closed"
        send_sms_notification "INFO: Position closed"
        return 0
    } || {
        log_message "ERROR" "Failed to close position"
        return 1
    }
}

# --- Main ---
main() {
    check_dependencies
    log_message "INFO" "--- GridBot v8.1 Started ---"
    log_message "INFO" "Symbol: $SYMBOL, Levels: $GRID_LEVELS,
Interval: $GRID_INTERVAL, Size: $ORDER_SIZE, Leverage: $LEVERAGE"

    [ "$PAPER_TRADING_MODE" = "false" ] && { check_market_conditions ||
{ log_message "ERROR" "Market check failed"; exit 1; }; }
    place_grid_orders || { log_message "ERROR" "Failed to place grid";
exit 1; }

    log_message "INFO" "Starting loop"
    send_sms_notification "INFO: Bot started"

    while true; do
        manage_grid_orders
        sleep "$CHECK_INTERVAL_SECONDS"
    done
}

# --- Trap Signals ---
trap 'log_message "INFO" "Shutting down"; cancel_all_orders;
close_position; log_message "INFO" "Shutdown complete";
send_sms_notification "INFO: Shutdown"; exit 0' SIGINT SIGTERM SIGHUP

# --- Run ---
main "$@"
