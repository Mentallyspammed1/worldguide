import chalk from 'chalk';
import { createWriteStream, type WriteStream } from 'fs';

let logStream: WriteStream | null = null;

export const initializeLogger = (logFilePath: string) => {
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

const writeToConsoleAndFile = (message: string, colorFn: (text: string) => string) => {
    console.log(colorFn(message));
    if (logStream) {
        logStream.write(`${message}\n`);
    }
};

export const logger = {
    info: (message: string) => writeToConsoleAndFile(chalk.blue(`[INFO] ${message}`), chalk.blue),
    success: (message: string) => writeToConsoleAndFile(chalk.green(`[SUCCESS] ${message}`), chalk.green),
    warning: (message: string) => writeToConsoleAndFile(chalk.yellow(`[WARNING] ${message}`), chalk.yellow),
    error: (message: string) => writeToConsoleAndFile(chalk.red(`[ERROR] ${message}`), chalk.red),
    divider: () => writeToConsoleAndFile(chalk.gray('----------------------------------------'), chalk.gray),
    status: (key: string, value: any) => writeToConsoleAndFile(chalk.cyan(`[STATUS] ${key}: ${value}`), chalk.cyan),
    pnl: (key: string, value: any) => writeToConsoleAndFile(chalk.magenta(`[PNL] ${key}: ${value}`), chalk.magenta),
};