// twin-range-bot/src/core/bot.ts
import { BybitService } from '../services/bybitService';
import { logger } from './logger';
import { AdvancedMarketMakingStrategy } from '../../../strategies/AdvancedMarketMakingStrategy';
import BasicMarketMakingStrategy from '../../../strategies/BasicMarketMakingStrategy';
import { INITIAL_TRADE_STATE_TEMPLATE } from '../../../constants';
export class MarketMakingBot {
    constructor(config) {
        this.config = { ...config, dataSource: config.dataSource || 'rest' };
        logger.info('MarketMakingBot initialized', { config: this.config });
        this.currentStrategy = new (this.config.strategyType === 'BasicMarketMakingStrategy' ? BasicMarketMakingStrategy : AdvancedMarketMakingStrategy)({
            ...config,
            symbols: this.config.symbols,
        });
        this.state = {
            ...INITIAL_TRADE_STATE_TEMPLATE,
            balance: config.initialCapital || INITIAL_TRADE_STATE_TEMPLATE.balance,
            equityCurve: [config.initialCapital || INITIAL_TRADE_STATE_TEMPLATE.balance],
            symbols: Object.fromEntries(this.config.symbols.map(symbol => [
                symbol,
                {
                    ...INITIAL_TRADE_STATE_TEMPLATE.symbols[symbol],
                    cash: (config.initialCapital || INITIAL_TRADE_STATE_TEMPLATE.balance) / this.config.symbols.length,
                }
            ]))
        };
        this.bybitService = new BybitService(this.config.bybit_api_key, this.config.bybit_api_secret, this.config.is_testnet, this.config, {
            onOrderbookUpdate: this.handleOrderbookUpdate.bind(this),
            onTradeUpdate: this.handleTradeUpdate.bind(this),
            onExecutionUpdate: this.handleExecutionUpdate.bind(this),
            onOrderUpdate: this.handleOrderUpdate.bind(this),
            onPositionUpdate: this.handlePositionUpdate.bind(this),
            onKlineUpdate: this.handleKlineUpdate.bind(this),
        });
    }
    getConfig() {
        return this.config;
    }
    getState() {
        return this.state;
    }
    async start() {
        await this.initializeState();
        if (this.config.dataSource === 'rest') {
            setInterval(() => this.updateStateFromRest(), this.config.refresh_rate_seconds * 1000);
        }
        else if (this.config.dataSource === 'backtest') {
            await this.backtest();
        }
    }
    async initializeState() {
        for (const symbol of this.config.symbols) {
            const orderbook = await this.bybitService.getOrderbook(symbol);
            this.state.symbols[symbol].referencePrice = (parseFloat(orderbook.b[0][0]) + parseFloat(orderbook.a[0][0])) / 2;
            const positions = await this.bybitService.getPositions(symbol);
            if (positions.length > 0) {
                this.updateInventoryAndPnl(positions[0]);
            }
            this.state.symbols[symbol].klines = await this.bybitService.getKlines(symbol, this.config.interval);
            const executions = await this.bybitService.getExecutionHistory(symbol);
            this.updateProfitAndInventory(executions);
            await this.updateOrders(symbol);
        }
        this.updateEquity();
    }
    async updateStateFromRest() {
        for (const symbol of this.config.symbols) {
            const orderbook = await this.bybitService.getOrderbook(symbol);
            this.state.symbols[symbol].referencePrice = (parseFloat(orderbook.b[0][0]) + parseFloat(orderbook.a[0][0])) / 2;
            this.state.symbols[symbol].klines = await this.bybitService.getKlines(symbol, this.config.interval);
            const positions = await this.bybitService.getPositions(symbol);
            if (positions.length > 0) {
                this.updateInventoryAndPnl(positions[0]);
            }
            const executions = await this.bybitService.getExecutionHistory(symbol);
            this.updateProfitAndInventory(executions);
            await this.updateOrders(symbol);
        }
        this.updateEquity();
    }
    async backtest() {
        const allEvents = [];
        for (const symbol of this.config.symbols) {
            const klines = await this.bybitService.getKlines(symbol, this.config.interval, this.config.lookback_bars);
            klines.forEach(kline => {
                allEvents.push({ timestamp: parseInt(kline.t.toString()), symbol, kline });
            });
        }
        allEvents.sort((a, b) => a.timestamp - b.timestamp);
        for (const event of allEvents) {
            const { symbol, kline } = event;
            this.state.symbols[symbol].referencePrice = parseFloat(kline.c);
            this.state.symbols[symbol].klines = [kline, ...this.state.symbols[symbol].klines].slice(0, this.config.volatilityWindow);
            await this.updateOrders(symbol);
            this.updateEquity();
        }
        const metrics = this.calculateMetrics();
        logger.info('Backtest completed', { metrics });
    }
    handleOrderbookUpdate(orderbook) {
        const symbol = orderbook.s;
        if (this.config.symbols.includes(symbol)) {
            this.state.symbols[symbol].orderbook = orderbook;
            this.state.symbols[symbol].referencePrice = (parseFloat(orderbook.b[0][0]) + parseFloat(orderbook.a[0][0])) / 2;
            logger.debug('Orderbook updated', { symbol: orderbook.s, midPrice: this.state.symbols[orderbook.s].referencePrice });
            if (this.config.dataSource === 'websocket') {
                this.updateOrders(symbol);
            }
        }
    }
    handleTradeUpdate(trades) {
        if (this.config.dataSource === 'websocket') {
            for (const trade of trades) {
                const symbol = trade.s;
                if (this.config.symbols.includes(symbol)) {
                    this.state.symbols[symbol].recentTrades.push(parseFloat(trade.p));
                    if (this.state.symbols[symbol].recentTrades.length > this.config.volatilityWindow) {
                        this.state.symbols[symbol].recentTrades.shift();
                    }
                    this.updateOrders(symbol);
                }
            }
        }
    }
    handleKlineUpdate(klines) {
        klines.forEach(kline => {
            if (this.config.symbols.includes(kline.s)) {
                this.state.symbols[kline.s].klines.push(kline);
                logger.debug('Kline updated', { symbol: kline.s, close: kline.c });
                if (this.config.dataSource === 'websocket') {
                    this.updateOrders(kline.s);
                }
            }
        });
    }
    updateProfitAndInventory(executions) {
        executions.forEach(exec => {
            const symbolState = this.state.symbols[exec.symbol];
            if (symbolState) {
                const qty = parseFloat(exec.execQty);
                const price = parseFloat(exec.execPrice);
                const fee = parseFloat(exec.execFee);
                const profit = exec.side === 'Buy' ? -qty * price - fee : qty * price - fee;
                symbolState.cash += profit;
                symbolState.inventory += exec.side === 'Buy' ? qty : -qty;
                symbolState.tradeHistory.push({
                    tradeId: `trade-${Date.now()}`,
                    side: exec.side,
                    qty,
                    price,
                    profit,
                    fee,
                    timestamp: parseInt(exec.execTime),
                });
                this.state.totalProfit += profit;
                this.state.daily_pnl += profit;
                this.state.balance += profit;
                this.state.totalTrades++;
                this.state.avgPnl = this.state.totalProfit / (this.state.totalTrades || 1);
                this.state.equityCurve.push(this.state.balance);
                logger.info(`Trade executed: ${exec.side} ${qty} ${exec.symbol} at $${price}`, { symbol: exec.symbol, profit });
            }
        });
    }
    handleExecutionUpdate(executions) {
        this.updateProfitAndInventory(executions);
        if (this.config.dataSource === 'websocket') {
            const symbols = [...new Set(executions.map(e => e.symbol))];
            for (const symbol of symbols) {
                this.updateOrders(symbol);
            }
        }
    }
    updateInventoryAndPnl(position) {
        const symbolState = this.state.symbols[position.symbol];
        if (symbolState) {
            symbolState.inventory = parseFloat(position.size) * (position.side === 'Buy' ? 1 : -1);
            symbolState.unrealizedPnl = parseFloat(position.unrealisedPnl);
            logger.debug('Position updated', { symbol: position.symbol, inventory: symbolState.inventory });
        }
    }
    handlePositionUpdate(positions) {
        positions.forEach(pos => this.updateInventoryAndPnl(pos));
        if (this.config.dataSource === 'websocket') {
            const symbols = [...new Set(positions.map(p => p.symbol))];
            for (const symbol of symbols) {
                this.updateOrders(symbol);
            }
        }
    }
    handleOrderUpdate(orders) {
        if (this.config.dataSource === 'websocket') {
            for (const order of orders) {
                if (this.config.symbols.includes(order.symbol)) {
                    if (order.orderStatus === 'Filled' || order.orderStatus === 'Cancelled') {
                        this.state.symbols[order.symbol].active_mm_orders = this.state.symbols[order.symbol].active_mm_orders.filter(o => o.orderId !== order.orderId);
                        this.state.symbols[order.symbol].orderStatus = order.orderStatus === 'Filled' ? 'Filled' : 'Cancelled';
                        logger.info(`Order Update: ${order.orderId} ${order.orderStatus} at $${parseFloat(order.price).toFixed(2)}`, { symbol: order.symbol });
                        this.updateOrders(order.symbol);
                    }
                }
            }
        }
    }
    async updateOrders(symbol) {
        const state = this.state.symbols[symbol];
        if (!state.referencePrice) {
            state.orderStatus = 'No Reference Price';
            logger.error('No reference price available', { symbol });
            return;
        }
        try {
            const orderbook = this.config.dataSource === 'rest' ? await this.bybitService.getOrderbook(symbol) : undefined;
            const volatility = this.currentStrategy.calculateVolatility(symbol, state.klines, state.referencePrice, state.atr);
            state.atr = volatility * state.referencePrice; // Update ATR in state
            const momentum = state.recentTrades.length >= 2 ? (state.recentTrades[state.recentTrades.length - 1] - state.recentTrades[0]) / state.recentTrades[0] : 0;
            const isVolatilityValid = volatility >= this.config.minVolatility && volatility <= this.config.maxVolatility;
            const isDepthValid = orderbook ? orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) >= this.config.minDepthRatio && orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) <= this.config.maxDepthRatio : true;
            const isMomentumValid = Math.abs(momentum) <= this.config.maxMomentum;
            const canBuy = state.inventory < this.config.maxInventory * 0.9;
            const canSell = state.inventory > -this.config.maxInventory * 0.9;
            logger.info(`Order Conditions: Volatility: ${(volatility * 100).toFixed(2)}% (${isVolatilityValid ? 'Valid' : 'Invalid'}), Depth: ${orderbook ? (orderbook.b.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0) / orderbook.a.slice(0, 5).reduce((sum, [, qty]) => sum + parseFloat(qty), 0)).toFixed(2) : 'N/A'} (${isDepthValid ? 'Valid' : 'Invalid'}), Momentum: ${(momentum * 100).toFixed(2)}% (${isMomentumValid ? 'Valid' : 'Invalid'}), Can Buy: ${canBuy}, Can Sell: ${canSell}`, { symbol });
            if (!isVolatilityValid || !isDepthValid || !isMomentumValid) {
                state.orderStatus = 'Conditions Not Met';
                logger.info('Skipping order placement due to invalid conditions', { symbol });
                return;
            }
            const { buyPrice, sellPrice } = this.currentStrategy.calculateOrderPrices(symbol, state.referencePrice, volatility, state.inventory, this.config.maxInventory, state.recentTrades, orderbook);
            const buyQty = this.currentStrategy.calculateOrderSize(symbol, state.referencePrice, volatility, state.cash);
            const sellQty = this.currentStrategy.calculateOrderSize(symbol, state.referencePrice, volatility, state.cash);
            if (isNaN(buyPrice) || isNaN(sellPrice) || isNaN(buyQty) || isNaN(sellQty)) {
                state.orderStatus = 'Invalid Prices';
                logger.error('Invalid order prices or quantities: NaN detected', { symbol });
                return;
            }
            for (const order of state.active_mm_orders) {
                await this.bybitService.cancelOrder(symbol, order.orderId);
                logger.info(`Cancelled order: ${order.orderId} (${order.type})`, { symbol });
            }
            state.active_mm_orders = [];
            if (canBuy && buyQty > 0) {
                const buyOrder = await this.bybitService.placeMarketMakingOrder(symbol, 'Buy', buyPrice, buyQty, buyPrice * (1 + this.config.tpPercent), buyPrice * (1 - this.config.slPercent));
                state.active_mm_orders.push({ type: 'buy', price: buyPrice, orderId: buyOrder.orderId });
                logger.info(`Placed buy order: ${buyOrder.orderId} at $${buyPrice.toFixed(2)}, Qty: ${buyQty.toFixed(4)}`, { symbol });
            }
            if (canSell && sellQty > 0) {
                const sellOrder = await this.bybitService.placeMarketMakingOrder(symbol, 'Sell', sellPrice, sellQty, sellPrice * (1 - this.config.tpPercent), sellPrice * (1 + this.config.slPercent));
                state.active_mm_orders.push({ type: 'sell', price: sellPrice, orderId: sellOrder.orderId });
                logger.info(`Placed sell order: ${sellOrder.orderId} at $${sellPrice.toFixed(2)}, Qty: ${sellQty.toFixed(4)}`, { symbol });
            }
            state.orderStatus = state.active_mm_orders.length > 0 ? 'Active' : 'Idle';
        }
        catch (err) {
            state.orderStatus = 'Error';
            logger.error(`Error updating orders: ${err.message}`, { symbol, stack: err.stack });
        }
    }
    updateEquity() {
        const totalCash = Object.values(this.state.symbols).reduce((sum, s) => sum + s.cash, 0);
        const totalPositionValue = Object.entries(this.state.symbols).reduce((sum, [symbol, state]) => sum + state.inventory * state.referencePrice, 0);
        this.state.balance = totalCash + totalPositionValue;
        this.state.equityCurve.push(this.state.balance);
        logger.info('Equity updated', { totalCash: totalCash.toFixed(2), totalPositionValue: totalPositionValue.toFixed(2), balance: this.state.balance.toFixed(2) });
    }
    async getActiveOrders(symbol) {
        return this.state.symbols[symbol].active_mm_orders;
    }
    calculateMetrics() {
        const aggregateMetrics = {
            sharpeRatio: 0,
            sortinoRatio: 0,
            maxDrawdown: 0,
            winRate: 0,
            profitFactor: 0,
            calmarRatio: 0,
            totalProfit: 0,
            totalTradeCount: 0,
            averageTradeProfit: 0,
            totalVolumeTraded: 0,
        };
        const symbolMetrics = {};
        for (const symbol of this.config.symbols) {
            const closedTrades = this.state.symbols[symbol].tradeHistory.filter(t => t.profit !== 0);
            const profits = closedTrades.map(t => t.profit);
            const totalSymbolProfit = profits.reduce((sum, p) => sum + p, 0);
            const winCount = profits.filter(p => p > 0).length;
            const totalTrades = closedTrades.length;
            const grossProfit = profits.filter(p => p > 0).reduce((a, b) => a + b, 0);
            const grossLoss = Math.abs(profits.filter(p => p <= 0).reduce((a, b) => a + b, 0));
            const totalVolume = closedTrades.reduce((sum, t) => sum + t.qty * t.price, 0);
            let sharpeRatio = 0, sortinoRatio = 0, maxDrawdown = 0, winRate = 0, profitFactor = Infinity, calmarRatio = 0, averageTradeProfit = 0;
            if (totalTrades > 0) {
                winRate = (winCount / totalTrades) * 100;
                profitFactor = grossLoss > 0 ? grossProfit / grossLoss : Infinity;
                averageTradeProfit = totalSymbolProfit / totalTrades;
            }
            const returns = profits.map(p => p / (this.state.symbols[symbol].cash || this.state.balance / this.config.symbols.length));
            const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length || 0;
            const stdDev = returns.length > 0 ? Math.sqrt(returns.map(r => Math.pow(r - meanReturn, 2)).reduce((sum, r) => sum + r, 0) / returns.length) : 0;
            const downsideReturns = returns.filter(r => r < 0);
            const downsideStdDev = downsideReturns.length > 0 ? Math.sqrt(downsideReturns.map(r => Math.pow(r - meanReturn, 2)).reduce((sum, r) => sum + r, 0) / downsideReturns.length) : 0;
            sharpeRatio = stdDev > 0 ? meanReturn / stdDev : 0;
            sortinoRatio = downsideStdDev > 0 ? meanReturn / downsideStdDev : 0;
            const equityPeak = Math.max(...this.state.equityCurve);
            maxDrawdown = Math.max(...this.state.equityCurve.map(e => (equityPeak - e) / equityPeak)) * 100;
            const annualReturn = meanReturn * (252 * 24); // Rough annualization for hourly klines
            calmarRatio = maxDrawdown > 0 ? annualReturn / maxDrawdown : 0;
            symbolMetrics[symbol] = {
                totalProfit: parseFloat(totalSymbolProfit.toFixed(2)),
                tradeCount: totalTrades,
                winRate: parseFloat(winRate.toFixed(2)),
                profitFactor: parseFloat(profitFactor.toFixed(2)),
                averageTradeProfit: parseFloat(averageTradeProfit.toFixed(4)),
                sharpeRatio: parseFloat(sharpeRatio.toFixed(4)),
                sortinoRatio: parseFloat(sortinoRatio.toFixed(4)),
                maxDrawdown: parseFloat(maxDrawdown.toFixed(2)),
                calmarRatio: parseFloat(calmarRatio.toFixed(4)),
                totalVolumeTraded: parseFloat(totalVolume.toFixed(2)),
            };
            aggregateMetrics.totalProfit += totalProfit;
            aggregateMetrics.totalTradeCount += totalTrades;
            aggregateMetrics.totalVolumeTraded += totalVolume;
            aggregateMetrics.sharpeRatio += sharpeRatio / this.config.symbols.length;
            aggregateMetrics.sortinoRatio += sortinoRatio / this.config.symbols.length;
            aggregateMetrics.maxDrawdown = Math.max(aggregateMetrics.maxDrawdown, maxDrawdown);
            aggregateMetrics.winRate += winRate / this.config.symbols.length;
            aggregateMetrics.profitFactor += profitFactor / this.config.symbols.length;
            aggregateMetrics.calmarRatio += calmarRatio / this.config.symbols.length;
        }
        aggregateMetrics.averageTradeProfit = aggregateMetrics.totalTradeCount > 0 ? aggregateMetrics.totalProfit / aggregateMetrics.totalTradeCount : 0;
        return { ...symbolMetrics, aggregate: aggregateMetrics };
    }
}
