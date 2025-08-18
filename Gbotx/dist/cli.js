import { Backtester } from './twin-range-bot/src/core/backtest';
import { logger } from './logger';
logger.info("cli.tsx running at", new Date().toISOString());
const backtester = new Backtester();
backtester.run().then(() => {
    logger.info('Backtest finished.');
}).catch((error) => logger.error('Backtest failed', { error }));
