import chalk from 'chalk';
import { createWriteStream } from 'fs';
let logStream = null;
export const initializeLogger = (logFilePath) => {
    logStream = createWriteStream(logFilePath, { flags: 'w' });
    logStream.on('error', (err) => {
        console.error(chalk.red(`[ERROR] Failed to open log file stream: ${err.message}`));
    });
};
export const closeLogger = () => {
    if (logStream) {
        logStream.end();
        logStream = null;
    }
};
const writeToConsoleAndFile = (message, colorFn) => {
    console.log(colorFn(message));
    if (logStream) {
        logStream.write(`${message}\n`);
    }
};
export const logger = {
    info: (message) => writeToConsoleAndFile(chalk.blue(`[INFO] ${message}`), chalk.blue),
    success: (message) => writeToConsoleAndFile(chalk.green(`[SUCCESS] ${message}`), chalk.green),
    warning: (message) => writeToConsoleAndFile(chalk.yellow(`[WARNING] ${message}`), chalk.yellow),
    error: (message) => writeToConsoleAndFile(chalk.red(`[ERROR] ${message}`), chalk.red),
    divider: () => writeToConsoleAndFile(chalk.gray('----------------------------------------'), chalk.gray),
    status: (key, value) => writeToConsoleAndFile(chalk.cyan(`[STATUS] ${key}: ${value}`), chalk.cyan),
    pnl: (key, value) => writeToConsoleAndFile(chalk.magenta(`[PNL] ${key}: ${value}`), chalk.magenta),
};
