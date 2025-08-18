import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
// App.tsx
import { useState, useEffect } from 'react';
import { BOT_CONFIG_TEMPLATE } from './constants';
import { GeminiService } from './src/services/geminiService';
import { Backtester } from './src/core/backtest';
const App = () => {
    const [state, setState] = useState(null);
    const [signals, setSignals] = useState({});
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
    return (_jsxs("div", { style: { fontFamily: 'Arial', padding: '20px' }, children: [_jsx("h1", { style: { color: '#00ff00' }, children: "Gbotx - Multi-Symbol Market Making" }), state && (_jsxs(_Fragment, { children: [_jsx("h2", { children: "Aggregate Metrics" }), _jsxs("p", { children: [_jsx("strong", { children: "Balance:" }), " $", state.balance.toFixed(2), " USDT"] }), _jsxs("p", { children: [_jsx("strong", { children: "Daily PNL:" }), " $", state.daily_pnl.toFixed(2), " USDT"] }), _jsxs("p", { children: [_jsx("strong", { children: "Total Profit:" }), " $", state.totalProfit.toFixed(2), " USDT"] }), _jsxs("p", { children: [_jsx("strong", { children: "Win Rate:" }), " ", (state.winRate * 100).toFixed(2), "%"] }), _jsxs("p", { children: [_jsx("strong", { children: "Profit Factor:" }), " ", state.profitFactor.toFixed(2)] }), _jsxs("p", { children: [_jsx("strong", { children: "Total Trades:" }), " ", state.totalTrades] }), _jsxs("p", { children: [_jsx("strong", { children: "Average PNL:" }), " $", state.avgPnl.toFixed(2), " USDT"] }), Object.keys(state.symbols).map(symbol => (_jsxs("div", { style: { marginBottom: '20px', color: symbol === 'TRUMPUSDT' ? '#ff69b4' : symbol === 'BTCUSDT' ? '#ffa500' : '#1e90ff' }, children: [_jsxs("h2", { children: [symbol, " Metrics"] }), _jsxs("p", { children: [_jsx("strong", { children: "Cash:" }), " $", state.symbols[symbol].cash.toFixed(2), " USDT"] }), _jsxs("p", { children: [_jsx("strong", { children: "Unrealized PNL:" }), " $", state.symbols[symbol].unrealizedPnl.toFixed(2), " USDT"] }), _jsxs("p", { children: [_jsx("strong", { children: "Inventory:" }), " ", state.symbols[symbol].inventory.toFixed(4)] }), _jsxs("p", { children: [_jsx("strong", { children: "Order Status:" }), " ", state.symbols[symbol].orderStatus] }), _jsxs("p", { children: [_jsx("strong", { children: "Volatility (ATR):" }), " ", (state.symbols[symbol].atr * 100).toFixed(2), "%"] }), _jsxs("p", { children: [_jsx("strong", { children: "Gemini Signal:" }), " ", signals[symbol] || 'Pending'] }), _jsx("h3", { children: "Active Orders" }), _jsx("ul", { children: state.symbols[symbol].active_mm_orders.map(order => (_jsxs("li", { children: [order.type.toUpperCase(), " at $", order.price.toFixed(2)] }, order.orderId))) }), _jsx("h3", { children: "Trade History" }), _jsx("ul", { children: state.symbols[symbol].tradeHistory.map((trade, index) => (_jsxs("li", { children: [trade.side, " ", trade.qty.toFixed(4), " ", symbol, " at $", trade.price.toFixed(2), ", Profit: $", trade.profit.toFixed(2), ", Fee: $", trade.fee.toFixed(2), ", Time: ", new Date(trade.timestamp).toLocaleString()] }, trade.tradeId))) })] }, symbol))), _jsx("h2", { children: "Logs" }), _jsx("ul", { children: state.logs.map((log, index) => (_jsxs("li", { children: [log.symbol ? `[${log.symbol}] ` : '', log.type, ": ", log.message] }, index))) })] }))] }));
};
export default App;
