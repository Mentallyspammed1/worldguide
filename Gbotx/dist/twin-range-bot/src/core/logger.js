import fs from 'fs/promises';
import path from 'path';
const LOG_DIR = path.join(__dirname, '../logs');
const LOG_FILE = path.join(LOG_DIR, 'bot.log');
const writeLog = async (level, message, ...args) => {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level}] ${message} ${args.map(arg => JSON.stringify(arg)).join(' ')}\n`;
    try {
        await fs.appendFile(LOG_FILE, logMessage);
    }
    catch (error) {
        console.error('Failed to write to log file:', error);
    }
};
export const logger = {
    info: (message, ...args) => {
        console.log(`[INFO] ${message}`, ...args);
        writeLog('INFO', message, ...args);
    },
    error: (message, ...args) => {
        console.error(`[ERROR] ${message}`, ...args);
        writeLog('ERROR', message, ...args);
    },
};
