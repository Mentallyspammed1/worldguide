# Bybit V5 API Integration for twin-range-bot

## Project Overview

This project is a multi-symbol market-making trading bot built with TypeScript. It features a modular design that separates core logic, exchange communication, trading strategies, and backtesting capabilities. The bot can operate in a live/test mode connected to the Bybit V5 API or in a completely offline backtesting mode that uses locally generated mock data.

### Core Components

*   **`MarketMakingBot` (`twin-range-bot/src/core/bot.ts`):** The central orchestrator of the application. It manages the bot's state (`TradeState`), initializes the connection to the exchange (`BybitService`), loads the selected trading strategy, and handles the main event loop for both live trading and backtesting sessions.

*   **`BybitService` (`twin-range-bot/src/services/bybitService.ts`):** A dedicated service that encapsulates all communication with the Bybit V5 API. It handles REST API requests for actions like placing orders and fetching historical data, as well as WebSocket connections for real-time market data, order updates, and position changes. It includes essential features like rate limiting, WebSocket auto-reconnect, and API call metrics.

*   **Strategies (`strategies/`):** This directory contains the trading logic. The primary strategy is `AdvancedMarketMakingStrategy.ts`, which implements sophisticated logic for setting bid/ask spreads dynamically based on market volatility, order book depth, momentum, and current inventory. It also calculates order sizes based on risk parameters.

*   **`Backtester` (`twin-range-bot/src/core/backtest.ts`):** An offline simulation engine. It generates synthetic kline and order book data to test the `MarketMakingBot` and its strategies without requiring API keys or a live exchange connection. This is the entry point when running the backtest script.

*   **`cli.tsx`:** The command-line entry point for initiating the backtesting process.

*   **`App.tsx`:** A React-based web interface for visualizing the bot's performance in real-time. It displays aggregate and per-symbol metrics, including PNL, inventory, active orders, and trade history.

### Configuration and State Management

*   **`constants.ts`:** Defines the default configuration for the bot via `BOT_CONFIG_TEMPLATE`. This includes parameters for strategy, symbols, API keys, and risk management.
*   **`types.ts`:** Contains all TypeScript interfaces for the project's data structures. `BotConfig` defines the bot's configuration, while `TradeState` and `PerSymbolState` define the shape of the live state object that tracks all trading activity.

### Data Flow

*   **Live/Test Mode:** The `BybitService` fetches market data via REST or WebSocket. The `MarketMakingBot` receives this data through callbacks, updates the `TradeState`, and invokes the trading strategy. The strategy analyzes the new data and returns trading decisions (e.g., new orders to place). These decisions are sent back to the `BybitService` to be executed on the exchange.

*   **Backtest Mode:** The `Backtester` generates mock data and feeds it directly into the `MarketMakingBot`'s public data handlers (`handleKlineUpdate`, `handleOrderbookUpdate`, etc.). The bot processes this simulated data, and the strategy's logic is tested by simulating order placements and executions. This allows for rapid iteration and evaluation of the strategy's effectiveness.

## TradeState Structure
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

## Backtester
The `backtest.ts` module simulates trading without API keys using mock kline and order book data. It is run via `cli.tsx`, which initializes the `Backtester` class. The backtester in turn initializes `MarketMakingBot` and processes mock data for `TRUMPUSDT`, `BTCUSDT`, `ETHUSDT`. The `Candle` interface is not used; instead, `KlineData` from `types.ts` is used for kline data.

### Usage
To run the backtester, you can execute the `cli.tsx` file.

```bash
npx ts-node cli.tsx
```

This runs a 24-hour simulation, updating `TradeState` and logging with neon colors (pink for `TRUMPUSDT`, orange for `BTCUSDT`, blue for `ETHUSDT`).

## Gemini API Integration
The `twin-range-bot` integrates the Google Gemini API for AI-driven market analysis and code review:
- **Market Analysis**: `geminiService.ts` analyzes kline data to generate trading signals (Buy, Sell, Hold) for `TRUMPUSDT`, `BTCUSDT`, `ETHUSDT`.
- **Code Review**: A GitHub Action (`gemini-code-review.yml`) triggers on `/gemini-review` comments, using Gemini to review changed files and post comments.
- **Setup**: Install ` @google/generative-ai`, store `GEMINI_API_KEY` as a GitHub Secret, and configure `geminiService.ts`.
- **Usage**: Signals are displayed in `App.tsx` and logged via `logger.ts` with neon colors.