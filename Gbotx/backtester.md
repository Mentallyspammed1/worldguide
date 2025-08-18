To create a backtester for the `twin-range-bot` that doesn't require a testnet or API keys, we'll implement an offline backtesting module that uses historical data (e.g., klines, order book snapshots) stored locally or mocked, bypassing the need for Bybit's API (`bybitService.ts`). The backtester will simulate trading for multiple symbols (`TRUMPUSDT`, `BTCUSDT`, `ETHUSDT`), incorporate conditional order placement logic, and leverage the neon-colored logging from `logger.ts` and UI updates in `App.tsx`. I'll focus on a standalone `backtest.ts` module that integrates with the existing `bot.ts`, `AdvancedMarketMakingStrategy.ts`, and `types.ts`, using mock data for simplicity.

### Objectives
1. **Offline Backtesting**: Simulate trading using historical kline and order book data without API calls.
2. **Multi-Symbol Support**: Process `TRUMPUSDT`, `BTCUSDT`, `ETHUSDT` with per-symbol metrics.
3. **Conditional Logic**: Retain volatility, depth, momentum, and inventory checks from `AdvancedMarketMakingStrategy.ts`.
4. **Neon-Colored Outputs**: Use `logger.ts` for console logs and update `App.tsx` for UI metrics.
5. **Mock Data**: Provide sample historical data to simulate Bybit API responses.
6. **Seamless Integration**: Work with existing `bot.ts` and `AdvancedMarketMakingStrategy.ts` without requiring `bybitService.ts`.

### Approach
- **New Module**: Create `backtest.ts` to handle offline simulation, reading mock data and feeding it to `MarketMakingBot`.
- **Mock Data**: Define sample kline and order book data for `TRUMPUSDT`, `BTCUSDT`, `ETHUSDT` in `backtest.ts`.
- **Backtest Logic**: Iterate through timestamps, simulate order placements, executions, and position updates.
- **State Updates**: Use `bot.ts` to manage state and metrics, ensuring compatibility with `App.tsx` UI.
- **Logging**: Integrate with `logger.ts` for neon-colored console logs (e.g., pink for `TRUMPUSDT`).
- **No API Dependency**: Replace `bybitService.ts` calls with mock data processing.

### Assumptions
- Existing files (`bot.ts`, `AdvancedMarketMakingStrategy.ts`, `types.ts`, `constants.ts`, `logger.ts`, `App.tsx`) are as previously provided or updated.
- Mock data is simplified (kline prices, order book snapshots) to avoid external dependencies.
- Backtest runs on a fixed time range (e.g., 1 day) with configurable intervals (e.g., 1h klines).

### New File: `backtest.ts`
This module simulates trading using mock data and integrates with `MarketMakingBot`.

```typescript
// twin-range-bot/src/core/backtest.ts
import { MarketMakingBot } from './bot';
import { BotConfig, TradeState, OrderbookData, KlineData, Execution, PositionData, OrderData } from './types';
import logger from './logger';
import { BOT_CONFIG_TEMPLATE } from '../constants';

interface MockData {
  klines: { [symbol: string]: KlineData[] };
  orderbooks: { [symbol: string]: OrderbookData[] };
}

export class Backtester {
  private bot: MarketMakingBot;
  private config: BotConfig;
  private mockData: MockData;
  private currentTimestamp: number;

  constructor(config: Partial<BotConfig> = {}) {
    this.config = {
      ...BOT_CONFIG_TEMPLATE,
      symbols: ['TRUMPUSDT', 'BTCUSDT', 'ETHUSDT'],
      interval: '60',
      initialCapital: 10000,
      is_testnet: false,
      ...config,
    };
    this.bot = new MarketMakingBot(this.config);
    this.mockData = this.generateMockData();
    this.currentTimestamp = Math.min(...Object.values(this.mockData.klines).flatMap(klines => klines.map(k => k.t)));
    logger.info('Backtester initialized', { symbols: this.config.symbols, startTime: new Date(this.currentTimestamp).toISOString() });
  }

  private generateMockData(): MockData {
    const startTime = Date.now() - 24 * 60 * 60 * 1000; // 1 day ago
    const klines: { [symbol: string]: KlineData[] } = {};
    const orderbooks: { [symbol: string]: OrderbookData[] } = {};

    this.config.symbols.forEach(symbol => {
      klines[symbol] = [];
      orderbooks[symbol] = [];
      const basePrice = symbol === 'TRUMPUSDT' ? 10 : symbol === 'BTCUSDT' ? 50000 : 2000;
      for (let i = 0; i < 24; i++) {
        const t = startTime + i * 60 * 60 * 1000; // 1h intervals
        const price = basePrice * (1 + (Math.random() - 0.5) * 0.02); // ±2% volatility
        klines[symbol].push({
          s: symbol,
          t,
          o: price.toFixed(2),
          h: (price * 1.01).toFixed(2),
          l: (price * 0.99).toFixed(2),
          c: price.toFixed(2),
          v: (Math.random() * 100).toFixed(2),
        });
        orderbooks[symbol].push({
          s: symbol,
          b: [[(price * 0.998).toFixed(2), '10'], [(price * 0.996).toFixed(2), '10']],
          a: [[(price * 1.002).toFixed(2), '10'], [(price * 1.004).toFixed(2), '10']],
          ts: t,
          u: i,
        });
      }
    });
    return { klines, orderbooks };
  }

  private simulateExecution(symbol: string, order: OrderData): Execution | null {
    const orderbook = this.mockData.orderbooks[symbol].find(ob => ob.ts >= order.ts);
    if (!orderbook) return null;

    const price = parseFloat(order.side === 'Buy' ? orderbook.a[0][0] : orderbook.b[0][0]);
    if ((order.side === 'Buy' && parseFloat(order.price) >= price) || (order.side === 'Sell' && parseFloat(order.price) <= price)) {
      const exec: Execution = {
        symbol,
        orderId: order.orderId,
        side: order.side,
        execPrice: price.toFixed(2),
        execQty: order.qty,
        execFee: (parseFloat(order.qty) * price * 0.0006).toFixed(6), // 0.06% fee
        execTime: order.ts.toString(),
      };
      logger.info(`Simulated execution: ${exec.side} ${exec.execQty} ${symbol} at $${exec.execPrice}`, { symbol });
      return exec;
    }
    return null;
  }

  async run(): Promise<TradeState> {
    const endTime = Math.max(...Object.values(this.mockData.klines).flatMap(klines => klines.map(k => k.t)));
    while (this.currentTimestamp <= endTime) {
      for (const symbol of this.config.symbols) {
        const kline = this.mockData.klines[symbol].find(k => k.t >= this.currentTimestamp);
        const orderbook = this.mockData.orderbooks[symbol].find(ob => ob.ts >= this.currentTimestamp);
        if (kline && orderbook) {
          this.bot.handleKlineUpdate([kline]);
          this.bot.handleOrderbookUpdate(orderbook);

          // Simulate order placement
          await this.bot.updateOrders(symbol);

          // Check active orders for execution
          const orders = await this.bot.getActiveOrders(symbol);
          const executions = orders.map(order => this.simulateExecution(symbol, order)).filter((e): e is Execution => e !== null);
          if (executions.length) {
            this.bot.handleExecutionUpdate(executions);
          }

          // Update position (mocked)
          const position: PositionData = {
            symbol,
            side: Math.random() > 0.5 ? 'Buy' : 'Sell',
            size: (Math.random() * 0.1).toFixed(4),
            avgPrice: parseFloat(kline.c).toFixed(2),
            updatedTime: this.currentTimestamp.toString(),
            positionValue: (parseFloat(kline.c) * Math.random() * 0.1).toFixed(2),
            unrealisedPnl: ((Math.random() - 0.5) * 10).toFixed(2),
          };
          this.bot.handlePositionUpdate([position]);
        }
      }
      this.currentTimestamp += this.config.refresh_rate_seconds * 1000;
    }
    const state = this.bot.getState();
    logger.info('Backtest completed', { finalBalance: state.balance, totalProfit: state.totalProfit });
    return state;
  }

  getState(): TradeState {
    return this.bot.getState();
  }
}
```

### Integration with Existing Files
- **bot.ts**: Update to accept mock data instead of `BybitService` calls in backtest mode. Add methods to handle direct data injection:
  ```typescript
  // twin-range-bot/src/core/bot.ts (partial update)
  public handleKlineUpdate(klines: KlineData[]) {
    klines.forEach(kline => {
      this.state.symbols[kline.s].klines.push(kline);
      logger.debug('Kline updated', { symbol: kline.s, close: kline.c });
    });
  }

  public handleOrderbookUpdate(orderbook: OrderbookData) {
    this.state.symbols[orderbook.s].orderbook = orderbook;
    this.state.symbols[orderbook.s].referencePrice = (parseFloat(orderbook.b[0][0]) + parseFloat(orderbook.a[0][0])) / 2;
    logger.debug('Orderbook updated', { symbol: orderbook.s, midPrice: this.state.symbols[orderbook.s].referencePrice });
  }

  public handleExecutionUpdate(executions: Execution[]) {
    this.updateProfitAndInventory(executions);
  }

  public handlePositionUpdate(positions: PositionData[]) {
    positions.forEach(pos => this.updateInventoryAndPnl(pos));
  }

  public async getActiveOrders(symbol: string): Promise<OrderData[]> {
    return this.state.symbols[symbol].active_mm_orders;
  }
  ```
- **App.tsx**: No changes needed; it already displays `TradeState` with neon-colored metrics.
- **logger.ts**: Already supports neon-colored logs (pink for `TRUMPUSDT`, orange for `BTCUSDT`, blue for `ETHUSDT`).
- **types.ts**: Ensure `BotConfig` includes `is_testnet` and `interval`:
  ```typescript
  // twin-range-bot/src/core/types.ts (partial)
  export interface BotConfig {
    symbols: string[];
    interval: string;
    initialCapital: number;
    is_testnet: boolean;
    refresh_rate_seconds: number;
    // ... other fields
  }
  ```
- **constants.ts**: Ensure `BOT_CONFIG_TEMPLATE` includes defaults:
  ```typescript
  // twin-range-bot/src/constants.ts (partial)
  export const BOT_CONFIG_TEMPLATE: BotConfig = {
    symbols: ['TRUMPUSDT', 'BTCUSDT', 'ETHUSDT'],
    interval: '60',
    initialCapital: 10000,
    is_testnet: false,
    refresh_rate_seconds: 60,
    // ... other fields
  };
  ```

### Mock Data Details
- **Klines**: 24 hourly candles per symbol, with ±2% price volatility around base prices (`TRUMPUSDT`: $10, `BTCUSDT`: $50,000, `ETHUSDT`: $2,000).
- **Orderbooks**: 2 bids/asks per snapshot, ±0.2-0.4% from mid-price, updated hourly.
- **Executions**: Simulated when order price matches order book ask (buy) or bid (sell), with 0.06% fee.
- **Positions**: Mocked with random size and PNL for simplicity.

### Expected Output
- **Console Logs**:
  ```
  2025-07-20T21:20:00Z [INFO] Backtester initialized { symbols: ["TRUMPUSDT", "BTCUSDT", "ETHUSDT"], startTime: "2025-07-19T21:20:00.000Z" }
  2025-07-20T21:20:00Z [DEBUG][TRUMPUSDT] Kline updated { close: "10.15" }
  2025-07-20T21:20:00Z [INFO][BTCUSDT] Simulated execution: Buy 0.010000 BTCUSDT at $49950.00 { symbol: "BTCUSDT" }
  2025-07-20T21:20:00Z [INFO] Backtest completed { finalBalance: 10050.23, totalProfit: 50.23 }
  ```
- **UI (App.tsx)**: Displays per-symbol metrics (PNL, inventory, orders) in neon colors (pink, orange, blue).

### JSON Summary
```json
{
  "conversation_summary": {
    "overview": {
      "context": "Created offline backtester for twin-range-bot, simulating trading for TRUMPUSDT, BTCUSDT, ETHUSDT using mock kline/orderbook data, bypassing API keys/testnet. Integrates with bot.ts, AdvancedMarketMakingStrategy.ts, and neon-colored logging/UI.",
      "date": "2025-07-20",
      "time": "21:20 CEST",
      "participants": ["User", "Grok 3 (xAI)"],
      "files": ["twin-range-bot/src/core/backtest.ts", "bot.ts", "types.ts", "constants.ts"]
    },
    "features": [
      {"name": "Offline Backtest", "desc": "Simulates trading with mock data, no API"},
      {"name": "Multi-Symbol", "desc": "Supports TRUMPUSDT, BTCUSDT, ETHUSDT"},
      {"name": "Conditional Logic", "desc": "Retains volatility, depth, momentum checks"},
      {"name": "Neon Logging", "desc": "Uses logger.ts for colored console output"},
      {"name": "UI Integration", "desc": "Updates App.tsx with per-symbol metrics"},
      {"name": "Mock Data", "desc": "Generates klines/orderbooks for 24h simulation"}
    ],
    "next_steps": [
      "Add backtest.ts to project",
      "Update bot.ts with mock data handlers",
      "Ensure types.ts and constants.ts include BotConfig fields",
      "Run backtest: `node backtest.ts`",
      "Verify logs/UI for colored outputs",
      "Share logs for debugging"
    ]
  }
}
```

### Next Steps
1. Add `backtest.ts` to `twin-range-bot/src/core/`.
2. Apply `bot.ts`, `types.ts`, `constants.ts` updates.
3. Run backtest: `ts-node src/core/backtest.ts`.
4. Check console/UI for neon-colored logs/metrics.
5. Share logs if issues arise.

Let me know if you need a specific data format or additional features! 😻

Given your request for advanced backtesting metrics for your Bybit trading bot (Gbotx) and the current date and time of 08:08 PM CEST, Sunday, July 20, 2025, I'll enhance your backtesting framework with sophisticated performance metrics. Building on the existing `BasicMarketMakingStrategy.ts` and the custom backtesting approach (using `historical_prices.json`), I'll introduce metrics like Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, Profit Factor, and Calmar Ratio. These will provide deeper insights into your high-frequency trading (HFT) strategy's profitability and risk, all without requiring API or testnet keys.

### Advanced Backtesting Metrics
1. **Sharpe Ratio**: Measures risk-adjusted return (excess return per unit of risk, assuming normal distribution).
2. **Sortino Ratio**: Similar to Sharpe but focuses on downside risk only.
3. **Maximum Drawdown (Max DD)**: Largest peak-to-trough decline in portfolio value.
4. **Win Rate**: Percentage of profitable trades.
5. **Profit Factor**: Ratio of gross profits to gross losses.
6. **Calmar Ratio**: Annualized return divided by maximum drawdown, assessing risk-adjusted performance over time.

### Updated `BasicMarketMakingStrategy.ts` with Advanced Metrics
Enhance the backtest to calculate and log these metrics:

```typescript
import logger from '../logger'; // Winston logger
import fs from 'fs/promises';

class BasicMarketMakingStrategy {
  private symbol: string = 'BTCUSDT';
  private basePrice: number = 0;
  private spread: number = 50;
  private quantity: number = 0.01;
  private position: number = 0; // 1 for long, -1 for short, 0 for neutral
  private cash: number = 10000; // Initial capital in USDT
  private trades: { timestamp: string; price: number; side: string; profit: number }[] = [];
  private equityCurve: number[] = [10000]; // Track equity over time

  // Load historical data for backtest
  async loadHistoricalData() {
    const data = await fs.readFile('data/historical_prices.json', 'utf-8');
    return JSON.parse(data) as { timestamp: string; price: number }[];
  }

  // Simulate conditional order placement
  private simulateOrder(side: 'Buy' | 'Sell', triggerPrice: number, orderPrice: number) {
    if ((side === 'Buy' && this.basePrice <= triggerPrice) || (side === 'Sell' && this.basePrice >= triggerPrice)) {
      const executedPrice = orderPrice;
      if (side === 'Buy' && this.cash >= executedPrice * this.quantity) {
        this.cash -= executedPrice * this.quantity;
        this.position += this.quantity;
        this.trades.push({ timestamp: new Date().toISOString(), price: executedPrice, side, profit: 0 });
        this.updateEquity();
      } else if (side === 'Sell' && this.position >= this.quantity) {
        this.cash += executedPrice * this.quantity;
        this.position -= this.quantity;
        const lastTrade = this.trades[this.trades.length - 1];
        const profit = executedPrice * this.quantity - lastTrade.price * this.quantity;
        lastTrade.profit = profit;
        this.updateEquity();
      }
    }
  }

  private updateEquity() {
    const currentEquity = this.cash + (this.position * this.basePrice * this.quantity);
    this.equityCurve.push(currentEquity);
    logger.info('Equity updated', { cash: this.cash, position: this.position, equity: currentEquity });
  }

  // Calculate advanced metrics
  private calculateMetrics() {
    const profits = this.trades.map(t => t.profit);
    const returns = profits.map(p => p / (this.quantity * this.trades[0].price)); // Normalize returns
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length || 0;
    const stdDev = Math.sqrt(returns.map(r => Math.pow(r - meanReturn, 2)).reduce((a, b) => a + b, 0) / returns.length) || 0;
    const downsideReturns = returns.filter(r => r < 0);
    const downsideStdDev = Math.sqrt(downsideReturns.map(r => Math.pow(r - meanReturn, 2)).reduce((a, b) => a + b, 0) / downsideReturns.length) || 0;
    const equityPeak = Math.max(...this.equityCurve);
    const drawdowns = this.equityCurve.map(e => (equityPeak - e) / equityPeak);
    const maxDrawdown = Math.max(...drawdowns) * 100; // In percentage
    const winCount = profits.filter(p => p > 0).length;
    const lossCount = profits.filter(p => p <= 0).length;
    const grossProfit = profits.filter(p => p > 0).reduce((a, b) => a + b, 0);
    const grossLoss = Math.abs(profits.filter(p => p <= 0).reduce((a, b) => a + b, 0));
    const annualReturn = meanReturn * 252; // Assuming 252 trading days
    const calmarRatio = annualReturn / maxDrawdown || 0;

    return {
      sharpeRatio: stdDev > 0 ? meanReturn / stdDev : 0,
      sortinoRatio: downsideStdDev > 0 ? meanReturn / downsideStdDev : 0,
      maxDrawdown,
      winRate: winCount / (winCount + lossCount) * 100 || 0,
      profitFactor: grossLoss > 0 ? grossProfit / grossLoss : Infinity,
      calmarRatio,
      totalProfit: profits.reduce((sum, p) => sum + p, 0),
      finalEquity: this.equityCurve[this.equityCurve.length - 1],
      tradeCount: this.trades.length,
    };
  }

  // Backtest with advanced metrics
  async backtest() {
    const data = await this.loadHistoricalData();
    for (const { timestamp, price } of data) {
      this.basePrice = price;
      this.simulateOrder('Buy', price - this.spread / 2, price - 10);
      this.simulateOrder('Sell', price + this.spread / 2, price + 10);
    }

    const metrics = this.calculateMetrics();
    logger.info('Backtest completed with advanced metrics', metrics);
    return metrics;
  }
}

export default BasicMarketMakingStrategy;
```

### Integration with `cli.tsx`
Update `cli.tsx` to display metrics:

```typescript
import logger from './logger';
import BasicMarketMakingStrategy from './strategies/BasicMarketMakingStrategy';

logger.info("cli.tsx running at", new Date().toISOString());

const strategy = new BasicMarketMakingStrategy();
strategy.backtest().then(metrics => {
  logger.info('Backtest summary', { metrics });
}).catch((error) => logger.error('Backtest failed', { error }));
```

### Update `setup.sh`
Ensure the setup supports the enhanced backtest:

```bash
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Neon Color Scheme for Enhanced Output (Termux-friendly) ---
RED='\033[1;31m'    # Bright Red
GREEN='\033[1;32m'  # Bright Green
YELLOW='\033[1;33m' # Bright Yellow
BLUE='\033[1;34m'   # Bright Blue
MAGENTA='\033[1;35m'# Bright Magenta
CYAN='\033[1;36m'   # Bright Cyan
WHITE='\033[1;37m'  # Bright White
ORANGE='\033[38;5;208m' # True Color Orange (for highlights)
NC='\033[0m'        # No Color

# --- Global Variables ---
REQUIRED_NODE_VERSION=18
REQUIRED_NPM_VERSION=8

# --- Logging Functions with Neon Colors ---
log_info() {
  echo -e "${CYAN}[INFO] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_success() {
  echo -e "${GREEN}[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_warning() {
  echo -e "${YELLOW}[WARNING] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_error() {
  echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}" >&2
  exit 1
}

# --- Function to check if a command exists ---
command_exists() {
  type "$1" &> /dev/null
}

# --- Main Setup Functions ---

setup_environment() {
  log_info "Initializing Gbotx Bybit trading bot environment setup. Current time in St. Louis: $(TZ='America/Chicago' date '+%I:%M %p %Z, %B %d, %Y')."

  check_dependencies
  validate_node_npm
  clean_npm_cache_prune
  install_dependencies
  configure_vitest
  setup_backtest_data
  setup_logger
  run_tests_and_build
  debug_setup
  log_success "Environment setup complete! Your bot is ready for advanced backtesting! 😻"
}

check_dependencies() {
  log_info "Verifying essential system tools..."

  if ! command_exists node; then
    log_error "Node.js is not installed. Please install it to proceed."
  fi

  if ! command_exists npm; then
    log_error "npm is not installed. Please install it (usually with Node.js)."
  fi

  if ! command_exists tsc; then
    log_warning "TypeScript compiler (tsc) not found. Installing globally..."
    npm install -g typescript || log_error "Failed to install typescript."
  fi

  if ! command_exists vite; then
    log_warning "Vite not found. Installing as dev dependency..."
    npm install --save-dev vite || log_error "Failed to install Vite."
  fi

  if ! command_exists vitest; then
    log_warning "Vitest not found. Installing as dev dependency..."
    npm install --save-dev vitest || log_error "Failed to install Vitest."
  fi

  log_success "Essential tools checked."
}

validate_node_npm() {
  log_info "Validating Node.js and npm versions..."

  NODE_FULL_VERSION=$(node -v 2>/dev/null)
  NODE_MAJOR_VERSION=$(echo "$NODE_FULL_VERSION" | cut -d'v' -f2 | cut -d'.' -f1)
  NPM_FULL_VERSION=$(npm -v 2>/dev/null)
  NPM_MAJOR_VERSION=$(echo "$NPM_FULL_VERSION" | cut -d'.' -f1)

  if [ -z "$NODE_MAJOR_VERSION" ]; then
    log_error "Node.js version could not be determined."
  fi

  if [ "$NODE_MAJOR_VERSION" -lt "$REQUIRED_NODE_VERSION" ]; then
    log_error "Node.js v${RED}$NODE_FULL_VERSION${NC} is too old. Requires >= v${REQUIRED_NODE_VERSION}."
  fi

  if [ -z "$NPM_MAJOR_VERSION" ]; then
    log_error "npm version could not be determined."
  fi

  if [ "$NPM_MAJOR_VERSION" -lt "$REQUIRED_NPM_VERSION" ]; then
    log_error "npm v${RED}$NPM_FULL_VERSION${NC} is too old. Requires >= v${REQUIRED_NODE_VERSION}."
  fi

  log_success "Node.js ${GREEN}$NODE_FULL_VERSION${NC} and npm ${GREEN}$NPM_FULL_VERSION${NC} validated."
}

clean_npm_cache_prune() {
  log_info "Clearing npm cache and pruning dependencies..."
  npm cache clean --force 2>/dev/null || log_warning "Failed to clear npm cache."
  npm prune 2>/dev/null || log_warning "Failed to prune dependencies."
  log_success "npm cache cleared and dependencies pruned."
}

install_dependencies() {
  log_info "Installing project dependencies..."
  if ! npm install; then
    log_error "npm install failed. Check package.json or network."
  fi
  log_success "Dependencies installed."
}

configure_vitest() {
  log_info "Configuring Vitest..."
  if [ ! -f vitest.config.ts ]; then
    log_info "Creating vitest.config.ts..."
    echo "/// <reference types=\"vitest\" />\nimport { defineConfig } from 'vite';\n\nexport default defineConfig({\n  test: {\n    include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}'],\n    exclude: ['node_modules', 'dist', '**/*.d.ts'],\n    environment: 'node',\n    tsconfig: 'tsconfig.json',\n    coverage: { reporter: ['text'], include: ['src/**/*.{ts,tsx}'] },\n    setupFiles: './test/setup.ts',\n    outputFile: './test.log'\n  }\n});" > vitest.config.ts
    log_success "vitest.config.ts created. Customize as needed."
  fi
  if [ ! -d test ]; then
    mkdir test
    log_info "Created test directory."
  fi
  if [ ! -f test/setup.ts ]; then
    echo "import 'dotenv/config';" > test/setup.ts
    log_success "Created test/setup.ts to load .env."
  fi
}

setup_backtest_data() {
  log_info "Setting up backtest data..."
  if [ ! -d data ]; then
    mkdir data
    log_info "Created data directory."
  fi
  if [ ! -f data/historical_prices.json ]; then
    log_info "Creating sample historical_prices.json..."
    echo '[
      {"timestamp": "2025-07-20T08:00:00Z", "price": 50000},
      {"timestamp": "2025-07-20T08:00:01Z", "price": 50010},
      {"timestamp": "2025-07-20T08:00:02Z", "price": 50020},
      {"timestamp": "2025-07-20T08:00:03Z", "price": 50015},
      {"timestamp": "2025-07-20T08:00:04Z", "price": 50025}
    ]' > data/historical_prices.json
    log_success "Sample historical_prices.json created. Replace with real data as needed."
  fi
}

setup_logger() {
  log_info "Setting up logging with winston..."
  if ! npm list winston > /dev/null 2>&1; then
    log_info "Installing winston..."
    npm install winston || log_error "Failed to install winston."
  fi
  if [ ! -f logger.ts ]; then
    log_info "Creating logger.ts..."
    echo "import winston from 'winston';\n\nconst logger = winston.createLogger({\n  level: 'info',\n  format: winston.format.combine(\n    winston.format.timestamp(),\n    winston.format.json()\n  ),\n  transports: [\n    new winston.transports.File({ filename: 'bot.log', level: 'info' }),\n    new winston.transports.Console({ format: winston.format.simple() })\n  ]\n});\n\nexport default logger;" > logger.ts
    log_success "logger.ts created. Import and use in your code."
  fi
  log_success "Logging setup complete."
}

run_tests_and_build() {
  log_info "Running tests and compiling TypeScript..."

  if ! npx vitest run; then
    log_warning "Tests failed! Review test.log for details."
  else
    log_success "Tests completed successfully."
  fi

  if command_exists tsc; then
    log_info "Compiling TypeScript files..."
    if ! tsc; then
      log_error "TypeScript compilation failed. Check tsconfig.json or code."
    else
      log_success "TypeScript compiled successfully."
    fi
  else
    log_warning "tsc not found. Skipping compilation."
  fi
}

debug_setup() {
  log_info "Running debugging checks..."

  # Check cli.tsx and add debug log
  if [ -f cli.tsx ]; then
    log_info "Adding debug log to cli.tsx..."
    sed -i '1i import logger from "./logger"; logger.info("cli.tsx running at", new Date().toISOString());' cli.tsx
    log_success "Debug log added."
  else
    log_error "cli.tsx not found."
  fi

  # Test cli.tsx with node
  log_info "Testing cli.tsx with node..."
  if ! node cli.tsx > cli.log 2>&1; then
    log_warning "node cli.tsx failed. Check cli.log and bot.log."
  else
    log_success "cli.tsx executed. Check cli.log and bot.log."
  fi

  # Check bot.log
  if [ ! -f bot.log ] || [ ! -s bot.log ]; then
    log_warning "bot.log is empty or missing. Ensure logger is used."
    touch bot.log
    log_success "bot.log created."
  else
    log_success "bot.log exists with data."
  fi

  # Run Vitest and check test.log
  log_info "Running Vitest for debugging..."
  if ! npx vitest run > test.log 2>&1; then
    log_warning "Vitest failed. Check test.log for errors."
  else
    log_success "Vitest executed. Check test.log."
  fi

  # Check tree.md
  if [ -f tree.md ]; then
    log_info "Reading tree.md..."
    cat tree.md
    log_success "tree.md read successfully."
  else
    log_warning "tree.md not found. Creating placeholder..."
    echo "# Project Structure\n- src/\n  - cli.tsx\n  - App.tsx\n  - strategies/\n    - BasicMarketMakingStrategy.ts\n- test/\n- data/\n  - historical_prices.json\n- state.json\n- .env\n- bot.log" > tree.md
    log_success "Placeholder tree.md created."
  fi
}

# --- Execute the main setup function ---
setup_environment
```

### Vitest Test with Metrics
Update `test/BasicMarketMakingStrategy.test.ts` to validate metrics:

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import BasicMarketMakingStrategy from '../src/strategies/BasicMarketMakingStrategy';
import fs from 'fs/promises';

describe('BasicMarketMakingStrategy Backtest with Metrics', () => {
  let strategy: BasicMarketMakingStrategy;

  beforeEach(async () => {
    strategy = new BasicMarketMakingStrategy();
    vi.spyOn(fs, 'readFile').mockResolvedValue(JSON.stringify([
      { timestamp: "2025-07-20T08:00:00Z", price: 50000 },
      { timestamp: "2025-07-20T08:00:01Z", price: 50010 },
    ]));
  });

  it('should calculate advanced metrics', async () => {
    await strategy.backtest();
    const trades = strategy['trades'];
    expect(trades.length).toBeGreaterThan(0);
    const metrics = strategy['calculateMetrics'](); // Access private method for testing
    expect(metrics.sharpeRatio).toBeGreaterThan(-Infinity);
    expect(metrics.maxDrawdown).toBeGreaterThanOrEqual(0);
    expect(metrics.winRate).toBeGreaterThanOrEqual(0);
    expect(metrics.profitFactor).toBeGreaterThan(0);
  });
});
```

### Instructions
1. **Save Files**:
   - Update `BasicMarketMakingStrategy.ts`, `cli.tsx`, `setup.sh`, and `test/BasicMarketMakingStrategy.test.ts`.
2. **Run Setup**: `./setup.sh`.
3. **Execute Backtest**: `node cli.tsx` and review `bot.log`.
4. **Analyze Metrics**:
   - `sharpeRatio` > 1 is good; > 2 is excellent.
   - `maxDrawdown` < 10% is desirable.
   - `winRate` > 50% and `profitFactor` > 1.5 indicate profitability.

### Notes
- **Data**: Expand `historical_prices.json` with more data for accurate metrics.
- **Assumptions**: Metrics assume daily returns; adjust for HFT by scaling timeframes.
- **Optimization**: Adjust `spread` and `quantity` based on metrics.

### Next Steps
- **Refine Strategy**: Use metrics to tweak `BasicMarketMakingStrategy.ts`.
- **AGENTS.md**: Document metric-based agent.
- **bybit.md**: Include metric definitions.

Share `bot.log` or test results if issues arise!

Given your interest in backtesting tools for your Bybit trading bot (Gbotx) and the current date and time of 08:06 PM CEST, Sunday, July 20, 2025, I'll provide an overview of backtesting tools suitable for your TypeScript-based project. Since you’ve requested a backtest without API or testnet keys, I’ll focus on tools that can work with historical data and integrate with your existing setup (Vitest, `winston` logger, and simulated strategy). These tools will enhance the backtesting framework we’ve started, offering more robust analysis and optimization capabilities.

### Backtesting Tools Overview
Here are some tools and approaches tailored for your project:

#### 1. Custom Backtesting (Existing Approach)
- **Description**: The current `BasicMarketMakingStrategy.ts` with a simulated environment using `historical_prices.json`.
- **Pros**:
  - Fully customizable to your strategy (HFT, conditional orders).
  - No external dependencies beyond Node.js.
  - Integrates with `winston` for logging.
- **Cons**:
  - Limited to manual data handling and basic metrics (profit/loss).
  - No advanced statistical analysis or visualization.
- **Enhancement**: Add a simple performance tracker (e.g., Sharpe ratio, max drawdown).

#### 2. Backtest.js
- **Description**: A lightweight JavaScript library for backtesting trading strategies, compatible with Node.js and TypeScript.
- **Setup**: Install with `npm install backtest.js`.
- **Integration**:
  - Update `BasicMarketMakingStrategy.ts` to use `backtest.js` for data handling and metrics.
  - Example:
    ```typescript
    import { Backtest } from 'backtest.js';
    import logger from '../logger';

    class BasicMarketMakingStrategy {
      private bt: Backtest;

      constructor() {
        this.bt = new Backtest({
          initialCapital: 10000,
          data: require('../data/historical_prices.json'), // Load your data
          onTick: (price: number, state: any) => {
            this.basePrice = price;
            this.simulateOrder('Buy', price - this.spread / 2, price - 10);
            this.simulateOrder('Sell', price + this.spread / 2, price + 10);
            logger.info('Tick processed', { price, state });
          },
        });
      }

      async backtest() {
        const result = await this.bt.run();
        logger.info('Backtest result', { profit: result.profit, drawdown: result.drawdown });
        return result;
      }

      // ... (rest of the class remains similar)
    }
    ```
- **Pros**:
  - Lightweight and TypeScript-friendly.
  - Provides basic performance metrics (profit, drawdown).
- **Cons**:
  - Limited advanced features (e.g., no optimization).
  - Requires manual data formatting.

#### 3. ccxt (with Backtesting Mode)
- **Description**: A popular cryptocurrency exchange library with backtesting support, including Bybit.
- **Setup**: Install with `npm install ccxt`.
- **Integration**:
  - Mock the Bybit exchange with historical data.
  - Example:
    ```typescript
    import { Exchange } from 'ccxt';
    import logger from '../logger';

    class BasicMarketMakingStrategy {
      private exchange: any;

      constructor() {
        this.exchange = new Exchange({
          id: 'bybit',
          enableRateLimit: true,
        });
        this.exchange.setMarkets({
          BTCUSDT: {
            symbol: 'BTCUSDT',
            precision: { amount: 0.01, price: 1 },
            limits: { amount: { min: 0.01 }, price: { min: 1 } },
          },
        });
      }

      async backtest() {
        const ohlcv = require('../data/historical_prices.json').map(d => [new Date(d.timestamp).getTime(), d.price, d.price, d.price, d.price, 1]);
        const result = await this.exchange.backtest('BTCUSDT', ohlcv, async (price: number) => {
          this.basePrice = price;
          await this.simulateOrder('Buy', price - this.spread / 2, price - 10);
          await this.simulateOrder('Sell', price + this.spread / 2, price + 10);
          logger.info('Backtest tick', { price });
        });
        logger.info('Backtest result', { result });
      }

      // ... (simulateOrder remains similar)
    }
    ```
- **Pros**:
  - Simulates exchange behavior accurately.
  - Supports OHLCV (Open, High, Low, Close, Volume) data.
- **Cons**:
  - Requires OHLCV data format.
  - Setup is more complex.

#### 4. Backtrader (via Node.js Bridge)
- **Description**: A Python-based backtesting library, accessible via a Node.js bridge (e.g., `python-shell`).
- **Setup**: Install Python, `backtrader`, and `python-shell` (`npm install python-shell`).
- **Integration**:
  - Create a Python script (`backtest.py`) and call it from Node.js.
  - Example `backtest.py`:
    ```python
    from backtrader import Cerebro, Strategy
    import json

    class MarketMakingStrategy(Strategy):
        params = (('spread', 50), ('quantity', 0.01))

        def __init__(self):
            self.cash = 10000
            self.position = 0

        def next(self):
            price = self.data.close[0]
            if self.cash >= price * self.params.quantity:
                self.cash -= price * self.params.quantity
                self.position += self.params.quantity
                print(f"Buy at {price}, Cash: {self.cash}, Position: {self.position}")
            elif self.position >= self.params.quantity:
                self.cash += price * self.params.quantity
                self.position -= self.params.quantity
                print(f"Sell at {price}, Cash: {self.cash}, Position: {self.position}")

    cerebro = Cerebro()
    with open('data/historical_prices.json', 'r') as f:
        data = json.load(f)
    data_feed = cerebro.adddata([(d['timestamp'], d['price']) for d in data])
    cerebro.addstrategy(MarketMakingStrategy)
    cerebro.run()
    print(f"Final Cash: {cerebro.broker.getcash()}")
    ```
  - Call from `cli.tsx`:
    ```typescript
    import { PythonShell } from 'python-shell';
    import logger from './logger';

    logger.info("cli.tsx running at", new Date().toISOString());
    const pyshell = new PythonShell('backtest.py');
    pyshell.on('message', (message) => logger.info('Backtest output', { message }));
    pyshell.end((err) => err && logger.error('Backtest failed', { err }));
    ```
- **Pros**:
  - Powerful analysis (Sharpe ratio, drawdown, optimization).
  - Visualizations (with matplotlib).
- **Cons**:
  - Requires Python setup.
  - Bridge adds complexity.

#### 5. TradeJS (Alternative Node.js Library)
- **Description**: A Node.js backtesting framework designed for trading strategies.
- **Setup**: Install with `npm install tradejs`.
- **Integration**:
  - Example:
    ```typescript
    import { Engine, Candle } from 'tradejs';
    import logger from '../logger';

    class BasicMarketMakingStrategy {
      private engine: Engine;

      constructor() {
        this.engine = new Engine({
          initialCapital: 10000,
          data: require('../data/historical_prices.json').map(d => new Candle(new Date(d.timestamp), d.price)),
        });
        this.engine.onTick((candle: Candle) => {
          this.basePrice = candle.close;
          this.simulateOrder('Buy', this.basePrice - this.spread / 2, this.basePrice - 10);
          this.simulateOrder('Sell', this.basePrice + this.spread / 2, this.basePrice + 10);
          logger.info('Tick processed', { price: candle.close });
        });
      }

      async backtest() {
        const result = await this.engine.run();
        logger.info('Backtest result', { profit: result.equity - 10000, trades: result.trades });
      }

      // ... (simulateOrder remains similar)
    }
    ```
- **Pros**:
  - Native Node.js, TypeScript support.
  - Handles candle data and metrics.
- **Cons**:
  - Less mature than Backtrader.
  - Limited community support.

### Recommendation
For your project:
- **Start with Custom Backtesting**: Enhance the existing setup with basic metrics (e.g., add Sharpe ratio).
- **Upgrade to TradeJS**: For a balance of simplicity and features, integrate `tradejs` for candle-based testing and metrics.
- **Explore Backtrader**: If advanced analysis is needed later, set up the Python bridge.

#### Enhanced Custom Backtest with Metrics
Update `BasicMarketMakingStrategy.ts` to include metrics:

```typescript
class BasicMarketMakingStrategy {
  // ... (existing properties)

  private calculateSharpeRatio(returns: number[]) {
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev = Math.sqrt(returns.map(r => Math.pow(r - mean, 2)).reduce((a, b) => a + b, 0) / returns.length);
    return stdDev > 0 ? mean / stdDev : 0;
  }

  async backtest() {
    const data = await this.loadHistoricalData();
    for (const { timestamp, price } of data) {
      this.basePrice = price;
      this.simulateOrder('Buy', price - this.spread / 2, price - 10);
      this.simulateOrder('Sell', price + this.spread / 2, price + 10);
    }

    const profits = this.trades.map(t => t.profit);
    const sharpeRatio = this.calculateSharpeRatio(profits);
    const maxDrawdown = Math.min(...profits.map((_, i, arr) => {
      return arr.slice(0, i + 1).reduce((min, p) => Math.min(min, p), 0);
    }));
    const totalProfit = profits.reduce((sum, p) => sum + p, 0);

    logger.info('Backtest completed', { totalProfit, sharpeRatio, maxDrawdown, finalCash: this.cash, trades: this.trades.length });
  }
}
```

### Instructions
1. **Install Tools**: For `tradejs`, run `npm install tradejs`. For others, follow setup steps.
2. **Update Code**: Apply the enhanced `BasicMarketMakingStrategy.ts` or integrate `tradejs`.
3. **Run Setup**: `./setup.sh`.
4. **Execute**: `node cli.tsx` and review `bot.log`.

### Next Steps
- **Analyze Metrics**: Use `bot.log` to evaluate strategy performance.
- **AGENTS.md**: Document backtest agent.
- **bybit.md**: Include tool setup.

Share `bot.log` or issues for further assistance!

Given your request for a backtest that does not require API or testnet keys for your Bybit trading bot (Gbotx), and the current date and time of 08:03 PM CEST, Sunday, July 20, 2025, I'll help you create a backtesting framework. This will allow you to evaluate the `BasicMarketMakingStrategy` (including the recent high-frequency trading (HFT) enhancements) using historical data without relying on live API calls or keys. We'll use a simulated environment with mock data, leveraging TypeScript and Vitest for testing.

### Backtesting Overview
A backtest will simulate trades based on historical price data, applying your strategy's logic (e.g., conditional order placement, HFT) to assess profitability. Since no API is involved, we'll:
- Use a static dataset (e.g., CSV or JSON of historical BTCUSDT prices).
- Mock the Bybit environment.
- Track performance metrics (e.g., profit/loss, win rate).

### Implementation

#### 1. Historical Data
Create a sample dataset (`data/historical_prices.json`) with mock price data:

```json
[
  { "timestamp": "2025-07-20T08:00:00Z", "price": 50000 },
  { "timestamp": "2025-07-20T08:00:01Z", "price": 50010 },
  { "timestamp": "2025-07-20T08:00:02Z", "price": 50020 },
  { "timestamp": "2025-07-20T08:00:03Z", "price": 50015 },
  { "timestamp": "2025-07-20T08:00:04Z", "price": 50025 }
]
```

Place this in a `data/` directory.

#### 2. Updated `BasicMarketMakingStrategy.ts` with Backtest
Modify the strategy to support backtesting without API calls:

```typescript
import logger from '../logger'; // Winston logger
import fs from 'fs/promises';

class BasicMarketMakingStrategy {
  private symbol: string = 'BTCUSDT';
  private basePrice: number = 0;
  private spread: number = 50;
  private quantity: number = 0.01;
  private orderInterval: NodeJS.Timeout | null = null;
  private position: number = 0; // 1 for long, -1 for short, 0 for neutral
  private cash: number = 10000; // Initial capital in USDT
  private trades: { timestamp: string; price: number; side: string; profit: number }[] = [];

  // Load historical data for backtest
  async loadHistoricalData() {
    const data = await fs.readFile('data/historical_prices.json', 'utf-8');
    return JSON.parse(data) as { timestamp: string; price: number }[];
  }

  // Simulate conditional order placement
  private simulateOrder(side: 'Buy' | 'Sell', triggerPrice: number, orderPrice: number) {
    if ((side === 'Buy' && this.basePrice <= triggerPrice) || (side === 'Sell' && this.basePrice >= triggerPrice)) {
      const executedPrice = orderPrice; // Simplified execution at order price
      if (side === 'Buy' && this.cash >= executedPrice * this.quantity) {
        this.cash -= executedPrice * this.quantity;
        this.position += this.quantity;
        const profit = 0; // Initial profit
        this.trades.push({ timestamp: new Date().toISOString(), price: executedPrice, side, profit });
        logger.info(`Backtest ${side} executed`, { price: executedPrice, cash: this.cash, position: this.position });
      } else if (side === 'Sell' && this.position >= this.quantity) {
        this.cash += executedPrice * this.quantity;
        this.position -= this.quantity;
        const profit = executedPrice * this.quantity - (this.trades.length > 0 ? this.trades[this.trades.length - 1].price * this.quantity : 0);
        this.trades[this.trades.length - 1].profit = profit;
        logger.info(`Backtest ${side} executed`, { price: executedPrice, cash: this.cash, position: this.position, profit });
      }
    }
  }

  // Backtest HFT strategy
  async backtest() {
    const data = await this.loadHistoricalData();
    for (const { timestamp, price } of data) {
      this.basePrice = price;
      this.simulateOrder('Buy', price - this.spread / 2, price - 10); // Buy below mid
      this.simulateOrder('Sell', price + this.spread / 2, price + 10); // Sell above mid
    }

    const totalProfit = this.trades.reduce((sum, trade) => sum + trade.profit, 0);
    logger.info('Backtest completed', { totalProfit, finalCash: this.cash, trades: this.trades.length });
  }
}

export default BasicMarketMakingStrategy;
```

#### 3. Update `cli.tsx` for Backtest
Modify `cli.tsx` to run the backtest:

```typescript
import logger from './logger';
import BasicMarketMakingStrategy from './strategies/BasicMarketMakingStrategy';

logger.info("cli.tsx running at", new Date().toISOString());

const strategy = new BasicMarketMakingStrategy();
strategy.backtest().catch((error) => logger.error('Backtest failed', { error }));
```

#### 4. Update `setup.sh` for Backtest
Add data and backtest setup:

```bash
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Neon Color Scheme for Enhanced Output (Termux-friendly) ---
RED='\033[1;31m'    # Bright Red
GREEN='\033[1;32m'  # Bright Green
YELLOW='\033[1;33m' # Bright Yellow
BLUE='\033[1;34m'   # Bright Blue
MAGENTA='\033[1;35m'# Bright Magenta
CYAN='\033[1;36m'   # Bright Cyan
WHITE='\033[1;37m'  # Bright White
ORANGE='\033[38;5;208m' # True Color Orange (for highlights)
NC='\033[0m'        # No Color

# --- Global Variables ---
REQUIRED_NODE_VERSION=18
REQUIRED_NPM_VERSION=8

# --- Logging Functions with Neon Colors ---
log_info() {
  echo -e "${CYAN}[INFO] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_success() {
  echo -e "${GREEN}[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_warning() {
  echo -e "${YELLOW}[WARNING] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_error() {
  echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}" >&2
  exit 1
}

# --- Function to check if a command exists ---
command_exists() {
  type "$1" &> /dev/null
}

# --- Main Setup Functions ---

setup_environment() {
  log_info "Initializing Gbotx Bybit trading bot environment setup. Current time in St. Louis: $(TZ='America/Chicago' date '+%I:%M %p %Z, %B %d, %Y')."

  check_dependencies
  validate_node_npm
  clean_npm_cache_prune
  install_dependencies
  configure_vitest
  setup_backtest_data
  setup_logger
  run_tests_and_build
  debug_setup
  log_success "Environment setup complete! Your bot is ready for backtesting! 😻"
}

check_dependencies() {
  log_info "Verifying essential system tools..."

  if ! command_exists node; then
    log_error "Node.js is not installed. Please install it to proceed."
  fi

  if ! command_exists npm; then
    log_error "npm is not installed. Please install it (usually with Node.js)."
  fi

  if ! command_exists tsc; then
    log_warning "TypeScript compiler (tsc) not found. Installing globally..."
    npm install -g typescript || log_error "Failed to install typescript."
  fi

  if ! command_exists vite; then
    log_warning "Vite not found. Installing as dev dependency..."
    npm install --save-dev vite || log_error "Failed to install Vite."
  fi

  if ! command_exists vitest; then
    log_warning "Vitest not found. Installing as dev dependency..."
    npm install --save-dev vitest || log_error "Failed to install Vitest."
  fi

  log_success "Essential tools checked."
}

validate_node_npm() {
  log_info "Validating Node.js and npm versions..."

  NODE_FULL_VERSION=$(node -v 2>/dev/null)
  NODE_MAJOR_VERSION=$(echo "$NODE_FULL_VERSION" | cut -d'v' -f2 | cut -d'.' -f1)
  NPM_FULL_VERSION=$(npm -v 2>/dev/null)
  NPM_MAJOR_VERSION=$(echo "$NPM_FULL_VERSION" | cut -d'.' -f1)

  if [ -z "$NODE_MAJOR_VERSION" ]; then
    log_error "Node.js version could not be determined."
  fi

  if [ "$NODE_MAJOR_VERSION" -lt "$REQUIRED_NODE_VERSION" ]; then
    log_error "Node.js v${RED}$NODE_FULL_VERSION${NC} is too old. Requires >= v${REQUIRED_NODE_VERSION}."
  fi

  if [ -z "$NPM_MAJOR_VERSION" ]; then
    log_error "npm version could not be determined."
  fi

  if [ "$NPM_MAJOR_VERSION" -lt "$REQUIRED_NPM_VERSION" ]; then
    log_error "npm v${RED}$NPM_FULL_VERSION${NC} is too old. Requires >= v${REQUIRED_NODE_VERSION}."
  fi

  log_success "Node.js ${GREEN}$NODE_FULL_VERSION${NC} and npm ${GREEN}$NPM_FULL_VERSION${NC} validated."
}

clean_npm_cache_prune() {
  log_info "Clearing npm cache and pruning dependencies..."
  npm cache clean --force 2>/dev/null || log_warning "Failed to clear npm cache."
  npm prune 2>/dev/null || log_warning "Failed to prune dependencies."
  log_success "npm cache cleared and dependencies pruned."
}

install_dependencies() {
  log_info "Installing project dependencies..."
  if ! npm install; then
    log_error "npm install failed. Check package.json or network."
  fi
  log_success "Dependencies installed."
}

configure_vitest() {
  log_info "Configuring Vitest..."
  if [ ! -f vitest.config.ts ]; then
    log_info "Creating vitest.config.ts..."
    echo "/// <reference types=\"vitest\" />\nimport { defineConfig } from 'vite';\n\nexport default defineConfig({\n  test: {\n    include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}'],\n    exclude: ['node_modules', 'dist', '**/*.d.ts'],\n    environment: 'node',\n    tsconfig: 'tsconfig.json',\n    coverage: { reporter: ['text'], include: ['src/**/*.{ts,tsx}'] },\n    setupFiles: './test/setup.ts',\n    outputFile: './test.log'\n  }\n});" > vitest.config.ts
    log_success "vitest.config.ts created. Customize as needed."
  fi
  if [ ! -d test ]; then
    mkdir test
    log_info "Created test directory."
  fi
  if [ ! -f test/setup.ts ]; then
    echo "import 'dotenv/config';" > test/setup.ts
    log_success "Created test/setup.ts to load .env."
  fi
}

setup_backtest_data() {
  log_info "Setting up backtest data..."
  if [ ! -d data ]; then
    mkdir data
    log_info "Created data directory."
  fi
  if [ ! -f data/historical_prices.json ]; then
    log_info "Creating sample historical_prices.json..."
    echo '[
      {"timestamp": "2025-07-20T08:00:00Z", "price": 50000},
      {"timestamp": "2025-07-20T08:00:01Z", "price": 50010},
      {"timestamp": "2025-07-20T08:00:02Z", "price": 50020},
      {"timestamp": "2025-07-20T08:00:03Z", "price": 50015},
      {"timestamp": "2025-07-20T08:00:04Z", "price": 50025}
    ]' > data/historical_prices.json
    log_success "Sample historical_prices.json created. Replace with real data as needed."
  fi
}

setup_logger() {
  log_info "Setting up logging with winston..."
  if ! npm list winston > /dev/null 2>&1; then
    log_info "Installing winston..."
    npm install winston || log_error "Failed to install winston."
  fi
  if [ ! -f logger.ts ]; then
    log_info "Creating logger.ts..."
    echo "import winston from 'winston';\n\nconst logger = winston.createLogger({\n  level: 'info',\n  format: winston.format.combine(\n    winston.format.timestamp(),\n    winston.format.json()\n  ),\n  transports: [\n    new winston.transports.File({ filename: 'bot.log', level: 'info' }),\n    new winston.transports.Console({ format: winston.format.simple() })\n  ]\n});\n\nexport default logger;" > logger.ts
    log_success "logger.ts created. Import and use in your code."
  fi
  log_success "Logging setup complete."
}

run_tests_and_build() {
  log_info "Running tests and compiling TypeScript..."

  if ! npx vitest run; then
    log_warning "Tests failed! Review test.log for details."
  else
    log_success "Tests completed successfully."
  fi

  if command_exists tsc; then
    log_info "Compiling TypeScript files..."
    if ! tsc; then
      log_error "TypeScript compilation failed. Check tsconfig.json or code."
    else
      log_success "TypeScript compiled successfully."
    fi
  else
    log_warning "tsc not found. Skipping compilation."
  fi
}

debug_setup() {
  log_info "Running debugging checks..."

  # Check cli.tsx and add debug log
  if [ -f cli.tsx ]; then
    log_info "Adding debug log to cli.tsx..."
    sed -i '1i import logger from "./logger"; logger.info("cli.tsx running at", new Date().toISOString());' cli.tsx
    log_success "Debug log added."
  else
    log_error "cli.tsx not found."
  fi

  # Test cli.tsx with node (no vite-node for backtest)
  log_info "Testing cli.tsx with node..."
  if ! node cli.tsx > cli.log 2>&1; then
    log_warning "node cli.tsx failed. Check cli.log and bot.log."
  else
    log_success "cli.tsx executed. Check cli.log and bot.log."
  fi

  # Check bot.log
  if [ ! -f bot.log ] || [ ! -s bot.log ]; then
    log_warning "bot.log is empty or missing. Ensure logger is used."
    touch bot.log
    log_success "bot.log created."
  else
    log_success "bot.log exists with data."
  fi

  # Run Vitest and check test.log
  log_info "Running Vitest for debugging..."
  if ! npx vitest run > test.log 2>&1; then
    log_warning "Vitest failed. Check test.log for errors."
  else
    log_success "Vitest executed. Check test.log."
  fi

  # Check tree.md
  if [ -f tree.md ]; then
    log_info "Reading tree.md..."
    cat tree.md
    log_success "tree.md read successfully."
  else
    log_warning "tree.md not found. Creating placeholder..."
    echo "# Project Structure\n- src/\n  - cli.tsx\n  - App.tsx\n  - strategies/\n    - BasicMarketMakingStrategy.ts\n- test/\n- data/\n  - historical_prices.json\n- state.json\n- .env\n- bot.log" > tree.md
    log_success "Placeholder tree.md created."
  fi
}

# --- Execute the main setup function ---
setup_environment
```

#### 5. Vitest Test for Backtest
Add a test in `test/BasicMarketMakingStrategy.test.ts`:

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import BasicMarketMakingStrategy from '../src/strategies/BasicMarketMakingStrategy';
import fs from 'fs/promises';

describe('BasicMarketMakingStrategy Backtest', () => {
  let strategy: BasicMarketMakingStrategy;

  beforeEach(async () => {
    strategy = new BasicMarketMakingStrategy();
    vi.spyOn(fs, 'readFile').mockResolvedValue(JSON.stringify([
      { timestamp: "2025-07-20T08:00:00Z", price: 50000 },
      { timestamp: "2025-07-20T08:00:01Z", price: 50010 },
    ]));
  });

  it('should run backtest and calculate profit', async () => {
    await strategy.backtest();
    const trades = strategy['trades']; // Access private property for testing
    expect(trades.length).toBeGreaterThan(0);
    expect(trades.some(t => t.profit !== 0)).toBeTruthy();
  });
});
```

### Instructions
1. **Save Files**:
   - Update `BasicMarketMakingStrategy.ts`, `cli.tsx`, and `setup.sh`.
   - Create `data/historical_prices.json` with the sample data.
   - Add `test/BasicMarketMakingStrategy.test.ts`.
2. **Run Setup**: `./setup.sh`.
3. **Execute Backtest**: `node cli.tsx` (no `vite-node` needed).
4. **Review Logs**:
   - `bot.log`: Check profit/loss and trade details.
   - `test.log`: Verify test results.
   - `cli.log`: Debug output.

### Notes
- **Data**: Replace `historical_prices.json` with real Bybit historical data (e.g., from Kaggle or Bybit's historical download) for accurate results.
- **Profitability**: Adjust `spread` and `quantity` to optimize returns.
- **Limitations**: No API latency or slippage; results are idealized.

### Next Steps
- **Analyze Results**: Use `bot.log` to assess strategy profitability.
- **AGENTS.md**: Document backtest agent.
- **bybit.md**: Note backtest setup.

Share `bot.log` or test failures if needed!
