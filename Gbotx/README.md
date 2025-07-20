# Gemini Pro Trading Bot v2.0 (Termux Ready)

This is an advanced, AI-powered trading bot that uses the Google Gemini API to generate trading signals. It has been architected to run as a headless process (e.g., in Termux or on a server) with a separate web-based UI for real-time monitoring.

## Architecture

The application is split into two main parts:

1.  **Headless Bot (`cli.tsx`)**: The core trading logic that runs as a command-line application via Node.js. It's responsible for managing state, fetching market data, calling the Gemini API, and executing trades. The bot persists its state to `state.json` and loads it on startup. It runs continuously without a GUI.
2.  **Web Visualizer (`App.tsx`)**: A React-based web dashboard that acts as a **read-only** monitor for the bot. It periodically fetches the bot's state from a `/api/state` endpoint served by the Vite development server and displays it in a user-friendly interface.

## Features

- **CLI-First Operation**: Run the bot 24/7 in any terminal environment like Termux.
- **Web-Based Monitoring**: Check your bot's status, PnL, and trade history from any browser.
- **Persistent State**: The bot automatically saves and loads its operational state (including balance, trades, and logs) to/from `state.json`.
- **Enhanced Logging**: Detailed logs are written to `twin-range-bot/src/logs/bot.log` for easier debugging and monitoring.
- **Advanced AI Prompts**: Sophisticated prompts to get high-quality signals, including Stop Loss and Take Profit levels.
- **Professional UI**: Candlestick charts, volume data, and detailed performance statistics.

---

## Prerequisites

- **Node.js**: Required to run the bot and the web server.
- **API Keys**: You need API keys from Google Gemini and Bybit.
- **Dependencies**: Ensure all Node.js dependencies are installed (e.g., `npm install`).

## Setup

1.  **Set API Keys**: You must set your API keys as environment variables. It is recommended to use a `.env` file in the project root for this.

    Create a `.env` file in the root of your project with the following content:

    ```
    GEMINI_API_KEY="YOUR_GEMINI_KEY"
    BYBIT_API_KEY="YOUR_BYBIT_KEY"
    BYBIT_API_SECRET="YOUR_BYBIT_SECRET"
    NODE_ENV="development" # or "production"
    ```

2.  **Install Dependencies**: Navigate to the project root and run:

    ```bash
    npm install
    ```

---

## Configuration

All bot parameters are controlled by editing the `config.json` file directly. This is where you set your trading symbol, risk parameters, and strategy settings. The UI does not allow for configuration changes.

**File: `config.json`**
```json
{
  "symbol": "BTCUSDT",
  "leverage": 10,
  "risk_per_trade_percent": 1.0,
  "refresh_rate_seconds": 15,
  // ... other parameters
}
```

---

## How to Run

You must run two separate processes, preferably in two different terminal windows.

### 1. Run the Headless Bot (CLI)

This command starts the bot. It will begin analyzing the market and making trades based on your `config.json` and environment variables. You will see color-coded log output in your terminal, and detailed logs will be written to `twin-range-bot/src/logs/bot.log`.

```bash
# Ensure your .env file is correctly set up in the project root.
npm run start:cli
```

### 2. Run the Web Visualizer

This command starts a local web server to host the monitoring dashboard. The dashboard will fetch the bot's state from the `/api/state` endpoint.

```bash
npm run start:web
```

Once the server is running, it will typically provide a URL (like `http://localhost:3000`). Open this URL in your web browser to view the dashboard. The dashboard will automatically refresh every few seconds to show the latest data.