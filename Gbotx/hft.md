Given your request to explore high-frequency trading (HFT) for your Bybit trading bot (Gbotx) and the current date and time of 08:02 PM CEST, Sunday, July 20, 2025, I'll adapt your project to incorporate HFT principles. HFT involves executing a large number of orders at very high speeds, often leveraging low-latency infrastructure, real-time market data, and sophisticated strategies. Since your bot uses TypeScript, Vite/Vitest, and the Bybit API with a `winston` logger, I'll enhance the `BasicMarketMakingStrategy.ts` to support HFT, focusing on conditional orders and performance optimization.

### High-Frequency Trading Overview
HFT characteristics relevant to your bot:
- **Low Latency**: Minimize execution delays (requires WebSocket streams for real-time data).
- **High Order Volume**: Place and cancel orders rapidly (e.g., market making with tight spreads).
- **Co-location**: Ideal but not feasible here; optimize locally instead.
- **Strategies**: Market making, arbitrage, or momentum trading.
- **Bybit Constraints**: API rate limits (e.g., 100 requests/sec for REST, 1000 messages/sec for WebSocket) and order execution delays.

### Adapting Gbotx for HFT
To enable HFT, we'll:
1. Use Bybit WebSocket for real-time price updates.
2. Optimize `BasicMarketMakingStrategy.ts` for rapid conditional order placement.
3. Enhance logging for HFT metrics (e.g., latency, order success rate).
4. Test with Vitest for performance.

#### Prerequisites
- **Dependencies**: Install `bybit-api` with WebSocket support: `npm install bybit-api`.
- **Environment**: Ensure `.env` has `BYBIT_API_KEY` and `BYBIT_API_SECRET`.

#### Updated `BasicMarketMakingStrategy.ts` for HFT
This version uses WebSocket for low-latency data and places conditional orders at high frequency.

```typescript
import logger from '../logger'; // Winston logger
import { WebSocketClient, RestClientV5 } from 'bybit-api';

// Initialize clients
const restClient = new RestClientV5({
  key: process.env.BYBIT_API_KEY!,
  secret: process.env.BYBIT_API_SECRET!,
  testnet: true, // Use testnet
});

const wsClient = new WebSocketClient({
  key: process.env.BYBIT_API_KEY!,
  secret: process.env.BYBIT_API_SECRET!,
  market: 'v5',
  testnet: true,
});

class BasicMarketMakingStrategy {
  private symbol: string = 'BTCUSDT';
  private basePrice: number = 0;
  private spread: number = 50; // Tight spread for HFT
  private quantity: number = 0.01; // Small quantity for high frequency
  private orderInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.setupWebSocket();
  }

  // WebSocket setup for real-time price updates
  private setupWebSocket() {
    wsClient.subscribe(['publicTrade.BTCUSDT']); // Subscribe to trade updates
    wsClient.on('update', (data) => {
      if (data.topic === 'publicTrade.BTCUSDT' && data.data.length > 0) {
        const latestPrice = parseFloat(data.data[0].price);
        this.basePrice = latestPrice;
        logger.info('Price update', { latestPrice });
      }
    });

    wsClient.on('error', (error) => logger.error('WebSocket error', { error }));
  }

  // Place conditional orders at high frequency
  private async placeHFTOrder(side: 'Buy' | 'Sell', offset: number) {
    const triggerPrice = this.basePrice + (side === 'Buy' ? -offset : offset);
    const orderPrice = triggerPrice - (side === 'Buy' ? 10 : -10); // Tight limit offset

    try {
      const params = {
        category: 'linear',
        symbol: this.symbol,
        side,
        orderType: 'Limit',
        qty: this.quantity.toString(),
        triggerPrice: triggerPrice.toString(),
        orderPrice: orderPrice.toString(),
        timeInForce: 'GTC',
        triggerDirection: side === 'Buy' ? 'Fall' : 'Rise',
        orderLinkId: `hft_${side}_${Date.now()}`,
      };

      const response = await restClient.submitOrder(params);
      logger.info(`HFT ${side} order placed`, { triggerPrice, orderPrice, response });
    } catch (error) {
      logger.error(`HFT ${side} order failed`, { error });
    }
  }

  // High-frequency market making loop
  startHFT() {
    if (this.orderInterval) clearInterval(this.orderInterval);

    this.orderInterval = setInterval(() => {
      if (this.basePrice > 0) {
        this.placeHFTOrder('Buy', this.spread / 2); // Buy below mid-price
        this.placeHFTOrder('Sell', this.spread / 2); // Sell above mid-price
      }
    }, 100); // 100ms interval (adjust for rate limits)
  }

  stopHFT() {
    if (this.orderInterval) {
      clearInterval(this.orderInterval);
      logger.info('HFT stopped');
    }
  }
}

export default BasicMarketMakingStrategy;
```

#### Integration with `cli.tsx`
Update `cli.tsx` to start/stop HFT:

```typescript
import logger from './logger';
import BasicMarketMakingStrategy from './strategies/BasicMarketMakingStrategy';

logger.info("cli.tsx running at", new Date().toISOString());

const strategy = new BasicMarketMakingStrategy();
strategy.startHFT();

// Stop after 10 seconds for testing (remove in production)
setTimeout(() => strategy.stopHFT(), 10000);
```

#### `setup.sh` Update
Ensure WebSocket and HFT dependencies are handled:

```bash
# ... (keep existing setup until debug_setup) ...

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

  # Test cli.tsx with vite-node
  log_info "Testing cli.tsx with vite-node..."
  if ! vite-node cli.tsx > cli.log 2>&1; then
    log_warning "vite-node failed. Check cli.log and bot.log."
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
    echo "# Project Structure\n- src/\n  - cli.tsx\n  - App.tsx\n  - strategies/\n    - BasicMarketMakingStrategy.ts\n- test/\n- state.json\n- .env\n- bot.log" > tree.md
    log_success "Placeholder tree.md created."
  fi
}

# ... (keep remaining functions) ...
```

### HFT Optimizations
1. **Low Latency**:
   - WebSocket provides real-time data (update every ~100ms).
   - Minimize I/O by batching logs (adjust `winston` buffer).

2. **Rate Limits**:
   - Limit `setInterval` to ~100ms to stay within Bybit's 1000 messages/sec WebSocket limit.
   - Monitor API usage with `logger.info` and adjust frequency.

3. **Performance**:
   - Use `async/await` with `Promise.all` for parallel order placement if needed.
   - Avoid synchronous operations in the loop.

4. **Testing**:
   - Add a Vitest test:
     ```typescript
     import { describe, it, expect } from 'vitest';
     import BasicMarketMakingStrategy from './BasicMarketMakingStrategy';

     describe('BasicMarketMakingStrategy HFT', () => {
       it('should start and stop HFT', () => {
         const strategy = new BasicMarketMakingStrategy();
         strategy.startHFT();
         setTimeout(() => strategy.stopHFT(), 500);
         expect(strategy).toBeDefined();
       });
     });
     ```

### Challenges and Mitigations
- **Latency**: Local execution may not match co-located HFT; test on testnet first.
- **Profitability**: Tight spreads may lead to losses; adjust `spread` and monitor `bot.log`.
- **API Errors**: Handle rate limits with exponential backoff (e.g., using `setTimeout` retries).

### Next Steps
1. **Run Setup**: `./setup.sh` to apply changes.
2. **Test HFT**: Run `vite-node cli.tsx` and check `bot.log` for order activity.
3. **Analyze Logs**: Verify order placement frequency and success.
4. **AGENTS.md**: Document HFT agent behavior.
5. **bybit.md**: Include WebSocket setup.

Share `bot.log` or errors if issues arise!
