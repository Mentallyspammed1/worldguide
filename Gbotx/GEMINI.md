# Gemini Project Documentation

This file serves as a central repository for information relevant to the Gemini CLI agent's interaction with this project.

## Gemini Added Memories
- The user's BYBIT_API_SECRET is TXztLxhYdHIcyzmN6QR2zSc2Dxj0UuQRiMzQ
- The user's BYBIT_API_KEY is NRrb4Biggi3sO7rKZ1
- The user provided a detailed conversation summary about Bybit V5 API WebSocket Functions in TypeScript. This includes information about setup, configuration, WebSocketClient for data streams (public and private), WebsocketAPIClient for sending commands (placeOrder, cancelOrder, amendOrder, etc.), example implementations, and additional notes on rate limits, authentication, connection management, endpoints, limitations, and resources. This information is crucial for understanding the project's interaction with the Bybit API.
- Pyrmethus provided a detailed guide for an enhanced Gemini chat script in Termux. The guide, written in a mystical persona, includes Node.js setup, use of 'chalk' for color, 'readline/promises' for clean async input, robust API key checks, and graceful error handling. The final script is a well-structured, user-friendly chat application with a streaming response.
- The user provided a comprehensive overview of the Bybit V5 API functions available through the `bybit-api` TypeScript SDK. This includes details on installation, key modules and functions (Market Data, Order Management, Position Management, Account Management, Asset Management, and WebSocket API), examples of usage, and additional notes on authentication, rate limits, TypeScript support, and documentation. It also highlights some limitations, such as the demo trading environment not supporting WebSocket API.

## Project-Specific Guidelines

### Code Style and Conventions
- **TypeScript**: Adhere to strict TypeScript typing.
- **Logging**: Use the `logger` utility for all logging.
- **API Interaction**: All Bybit API interactions should go through `bybitService.ts`.

### Testing
- **Unit Tests**: Use Vitest for unit testing. Test files should be placed in the `__tests__` directory, mirroring the structure of the `src` directory.
- **Backtesting**: Utilize the `backtester.ts` script for strategy backtesting.

### File Structure
- `src/`: Contains core application logic.
- `strategies/`: Contains trading strategy implementations.
- `services/`: Contains API interaction services.
- `data/`: Stores historical data for backtesting.
- `__tests__/`: Contains unit tests.

### Important Notes
- **API Keys**: API keys and secrets should be stored in `.env` and accessed securely.
- **Testnet vs. Mainnet**: Be mindful of the `is_testnet` configuration. WebSocket subscriptions are not supported on testnet.
- **Rate Limits**: Be aware of Bybit API rate limits, especially for REST calls.

## Bybit API Integration Details

### TradeState Structure
The `TradeState` interface is defined in `types.ts` and used in `bot.ts` and `App.tsx` for managing bot state and UI updates. It has a nested structure to support multi-symbol trading:

```typescript
interface PerSymbolState {
  cash: number;              // Per-symbol cash balance
  unrealizedPnl: number;     // Unrealized PNL
  inventory: number;         // Current inventory
  atr: number;               // Average True Range for volatility
  active_mm_orders: OrderData[]; // Active market-making orders
  tradeHistory: { tradeId: string; side: string; qty: number; price: number; profit: number; fee: number; timestamp: number }[];
  orderbook: OrderbookData | null; // Current order book
  klines: KlineData[];       // Historical klines
  executions: Execution[];   // Execution history
  orderStatus: string;       // Order placement status
  referencePrice: number;    // Mid-price from order book
}

interface TradeState {
  symbols: { [key: string]: PerSymbolState }; // Per-symbol state
  totalProfit: number;       // Aggregate profit
  daily_pnl: number;         // Daily PNL
  balance: number;           // Total balance
  logs: { type: string; message: string; symbol?: string }[]; // Logs
  winRate: number;           // Win rate
  profitFactor: number;      // Profit factor
  totalTrades: number;       // Total trades
  avgPnl: number;            // Average PNL per trade
  equityCurve: number[];     // Equity curve
}
```

### Backtesting Framework
The `backtester.ts` module simulates trading without API keys using mock kline and order book data. It calls `bot.backtest()` internally via the `Backtester` class, which initializes `MarketMakingBot` and processes mock data for `TRUMPUSDT`, `BTCUSDT`, `ETHUSDT`. The `Candle` interface is not used; instead, `KlineData` from `types.ts` is used for kline data.

### Bybit API Endpoints Used
- **REST**:
  - `GET /v5/position/list`: Retrieves position data (e.g., `size`, `side`, `avgPrice`, `unrealisedPnl`) for inventory and PNL tracking.
  - `POST /v5/order/create`: Places limit orders with `takeProfit` and `stopLoss` for market-making.
  - `POST /v5/order/cancel`: Cancels orders by `orderId`.
  - `GET /v5/order/realtime`: Fetches active orders.
  - `GET /v5/execution/list`: Retrieves execution history for profit calculations.
  - `GET /v5/market/kline`: Fetches historical candlestick data for volatility analysis.
  - `GET /v5/market/orderbook`: Retrieves order book depth for reference pricing.
- **WebSocket**:
  - `orderbook.50.<symbol>`: Real-time order book updates.
  - `publicTrade.<symbol>`: Recent trade data.
  - `execution`: Execution updates for profit tracking.
  - `order`: Order status updates.
  - `position`: Real-time position updates.
  - `kline.<interval>.<symbol>`: Real-time kline updates.

## Continuous Integration (CI) Workflow

A CI workflow is set up to automatically run checks on every push and pull request to ensure code quality and catch regressions early.

### Workflow Details
- **File**: `.github/workflows/ci.yml`
- **Triggers**: Pushes to `main` branch and pull requests targeting `main`.
- **Jobs**:
  - `build-and-test`: Runs on `ubuntu-latest`.
    - Checks out the repository.
    - Sets up Node.js (version 18).
    - Installs project dependencies (`npm install`).
    - Runs linting (`npm run lint`).
    - Runs tests (`npm run test`).
    - Runs build (`npm run build`).

### Setup
- Ensure `package.json` has the following scripts:
  - `"test": "vitest"`
  - `"lint": "eslint . --ext .ts,.tsx"`
  - `"build": "tsc"`
- A basic `.eslintrc.js` is provided for ESLint configuration.
