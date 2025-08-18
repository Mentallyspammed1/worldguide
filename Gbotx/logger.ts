import fs from 'fs/promises';
import path from 'path';
import { inspect } from 'util';

const LOG_DIR = path.join(__dirname, '../logs');
const LOG_FILE = path.join(LOG_DIR, 'bot.log');

// Utility to safely stringify objects, handling circular references
const safeStringify = (obj: any) => {
  try {
    return JSON.stringify(obj, (key, value) => {
      if (typeof value === 'object' && value !== null) {
        if (value instanceof Error) {
          return { message: value.message, stack: value.stack };
        }
      }
      return value;
    });
  } catch (error) {
    // Fallback for complex circular structures not handled by simple replacers
    return inspect(obj, { depth: null, circular: true });
  }
};

const writeLog = async (level: string, message: string, ...args: any[]) => {
  const timestamp = new Date().toISOString();
  const formattedArgs = args.map(arg => 
    typeof arg === 'object' ? safeStringify(arg) : arg
  ).join(' ');
  const logMessage = `[${timestamp}] [${level}] ${message} ${formattedArgs}\n`;
  
  try {
    // Ensure log directory exists
    await fs.mkdir(LOG_DIR, { recursive: true });
    await fs.appendFile(LOG_FILE, logMessage);
  } catch (error) {
    console.error('Failed to write to log file:', error);
  }
};

export const logger = {
  info: async (message: string, ...args: any[]) => {
    const logMessage = `[INFO] ${message}`;
    console.log(logMessage, ...args);
    await writeLog('INFO', message, ...args);
  },
  warn: async (message: string, ...args: any[]) => {
    const logMessage = `[WARN] ${message}`;
    console.warn(logMessage, ...args);
    await writeLog('WARN', message, ...args);
  },
  error: async (message: string, ...args: any[]) => {
    const logMessage = `[ERROR] ${message}`;
    console.error(logMessage, ...args);
    await writeLog('ERROR', message, ...args);
  },
  debug: async (message: string, ...args: any[]) => {
    const logMessage = `[DEBUG] ${message}`;
    console.log(logMessage, ...args);
    await writeLog('DEBUG', message, ...args);
  }
};