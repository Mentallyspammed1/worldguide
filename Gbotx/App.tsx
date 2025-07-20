// App.tsx
import React, { useState, useEffect } from 'react';
import { BOT_CONFIG_TEMPLATE } from './constants';
import type { TradeState } from './types';
import { GeminiService } from './src/services/geminiService';
import { Backtester } from './src/core/backtest';

const App: React.FC = () => {
  const [state, setState] = useState<TradeState | null>(null);
  const [signals, setSignals] = useState<{ [symbol: string]: string }>({});
  const geminiService = new GeminiService(process.env.GEMINI_API_KEY || null, BOT_CONFIG_TEMPLATE);

  useEffect(() => {
    const runBacktest = async () => {
      const backtester = new Backtester();
      const newState = await backtester.run();
      setState(newState);
      if (newState) {
        for (const symbol of Object.keys(newState.symbols)) {
          const signal = await geminiService.analyzeMarketData(symbol, newState.symbols[symbol].klines.slice(-5));
          setSignals(prev => ({ ...prev, [symbol]: signal }));
        }
      }
    };
    runBacktest();
  }, []);

  return (
    <div style={{ fontFamily: 'Arial', padding: '20px' }}>
      <h1 style={{ color: '#00ff00' }}>Gbotx - Multi-Symbol Market Making</h1>
      {state && (
        <>
          <h2>Aggregate Metrics</h2>
          <p><strong>Balance:</strong> ${state.balance.toFixed(2)} USDT</p>
          <p><strong>Daily PNL:</strong> ${state.daily_pnl.toFixed(2)} USDT</p>
          <p><strong>Total Profit:</strong> ${state.totalProfit.toFixed(2)} USDT</p>
          <p><strong>Win Rate:</strong> {(state.winRate * 100).toFixed(2)}%</p>
          <p><strong>Profit Factor:</strong> {state.profitFactor.toFixed(2)}</p>
          <p><strong>Total Trades:</strong> {state.totalTrades}</p>
          <p><strong>Average PNL:</strong> ${state.avgPnl.toFixed(2)} USDT</p>
          {Object.keys(state.symbols).map(symbol => (
            <div key={symbol} style={{ marginBottom: '20px', color: symbol === 'TRUMPUSDT' ? '#ff69b4' : symbol === 'BTCUSDT' ? '#ffa500' : '#1e90ff' }}>
              <h2>{symbol} Metrics</h2>
              <p><strong>Cash:</strong> ${state.symbols[symbol].cash.toFixed(2)} USDT</p>
              <p><strong>Unrealized PNL:</strong> ${state.symbols[symbol].unrealizedPnl.toFixed(2)} USDT</p>
              <p><strong>Inventory:</strong> {state.symbols[symbol].inventory.toFixed(4)}</p>
              <p><strong>Order Status:</strong> {state.symbols[symbol].orderStatus}</p>
              <p><strong>Volatility (ATR):</strong> {(state.symbols[symbol].atr * 100).toFixed(2)}%</p>
              <p><strong>Gemini Signal:</strong> {signals[symbol] || 'Pending'}</p>
              <h3>Active Orders</h3>
              <ul>
                {state.symbols[symbol].active_mm_orders.map(order => (
                  <li key={order.orderId}>{order.type.toUpperCase()} at ${order.price.toFixed(2)}</li>
                ))}
              </ul>
              <h3>Trade History</h3>
              <ul>
                {state.symbols[symbol].tradeHistory.map((trade, index) => (
                  <li key={trade.tradeId}>
                    {trade.side} {trade.qty.toFixed(4)} {symbol} at ${trade.price.toFixed(2)}, Profit: ${trade.profit.toFixed(2)}, Fee: ${trade.fee.toFixed(2)}, Time: {new Date(trade.timestamp).toLocaleString()}
                  </li>
                ))}
              </ul>
            </div>
          ))}
          <h2>Logs</h2>
          <ul>
            {state.logs.map((log, index) => (
              <li key={index}>{log.symbol ? `[${log.symbol}] ` : ''}{log.type}: {log.message}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
};

export default App;
