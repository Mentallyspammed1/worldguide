// backtester.ts
import { MarketMakingBot } from './twin-range-bot/src/core/bot';
import { BOT_CONFIG_TEMPLATE } from './constants';
import * as fs from 'fs/promises';
import * as path from 'path';
import { parse } from 'csv-parse/sync';
async function runBacktest() {
    const config = {
        ...BOT_CONFIG_TEMPLATE,
        is_testnet: false, // Backtest doesn't use live API, but keep consistent with live config
        dataSource: 'backtest',
    };
    console.log('Backtest config:', config);
    const bot = new MarketMakingBot(config);
    const historicalDataPath = path.join(process.cwd(), 'data', 'TRUMPUSDT-1m.csv');
    let historicalData = [];
    try {
        const fileContent = await fs.readFile(historicalDataPath, 'utf-8');
        historicalData = parse(fileContent, {
            columns: true,
            skip_empty_lines: true,
            cast: true,
        });
    }
    catch (error) {
        console.error('Error loading historical data:', error);
        return;
    }
    console.log(`Starting backtest with ${historicalData.length} klines...`);
    await bot.backtest(historicalData);
    console.log('Backtest complete. Final State:', bot.getState());
}
runBacktest().catch(console.error);
