
import { promises as fs } from 'fs';
import path from 'path';
import { BOT_CONFIG_TEMPLATE, INITIAL_TRADE_STATE_TEMPLATE } from './constants';

const CONFIG_PATH = path.join((process as any).cwd(), 'config.json');
const STATE_PATH = path.join((process as any).cwd(), 'state.json');

// Use an Immediately-Invoked Async Function Expression (IIAFE)
// to ensure the script awaits all async operations before exiting.
(async () => {
    console.log('Checking for necessary configuration files...');

    try {
        await fs.access(CONFIG_PATH);
        console.log('-> config.json already exists. Skipping creation.');
    } catch (error) {
        console.log('-> config.json not found, creating from template...');
        await fs.writeFile(CONFIG_PATH, JSON.stringify(BOT_CONFIG_TEMPLATE, null, 4));
        console.log('✅ config.json created successfully.');
    }

    try {
        await fs.access(STATE_PATH);
        console.log('-> state.json already exists. Skipping creation.');
    } catch (error) {
        console.log('-> state.json not found, creating from template...');
        // We need to shape the initial state correctly, including botStatus
        const initialState = {
            botStatus: 'IDLE',
            ...INITIAL_TRADE_STATE_TEMPLATE
        };
        await fs.writeFile(STATE_PATH, JSON.stringify(initialState, null, 4));
        console.log('✅ state.json created successfully.');
    }

    console.log('\nFile check complete. Web visualizer can now start.');

})().catch(err => {
    console.error('Failed to initialize files:', err);
    (process as any).exit(1);
});
