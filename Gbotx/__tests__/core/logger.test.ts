import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { logger } from '../../twin-range-bot/src/core/logger';
import fs from 'fs/promises';

// Mock the fs/promises module to avoid actual file I/O
vi.mock('fs/promises', () => ({
  default: {
    appendFile: vi.fn(),
    mkdir: vi.fn(),
  },
}));

describe('Logger', () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.resetAllMocks();
    // Spy on console methods
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    // Restore original console methods
    vi.restoreAllMocks();
  });

  it('should log info messages to console and file', async () => {
    const message = 'This is an info message';
    const arg1 = { data: 'test' };
    await logger.info(message, arg1);

    // Check if console.log was called with the correct message
    expect(console.log).toHaveBeenCalledWith(`[INFO] ${message}`, arg1);

    // Check if fs.appendFile was called with a correctly formatted log string
    expect(fs.appendFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.stringMatching(/\\[INFO\\] This is an info message {.*data.*:.*test.*}/)
    );
  });

  it('should log warn messages to console and file', async () => {
    const message = 'This is a warning';
    const arg1 = { code: 'test' };
    await logger.warn(message, arg1);

    // Check if console.warn was called with the correct message
    expect(console.warn).toHaveBeenCalledWith(`[WARN] ${message}`, arg1);

    // Check if fs.appendFile was called with a correctly formatted log string
    expect(fs.appendFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.stringMatching(/\\[WARN\\] This is a warning {.*code.*:.*test.*}/)
    );
  });

  it('should log error messages to console and file', async () => {
    const message = 'This is an error message';
    const error = new Error('Something went wrong');
    await logger.error(message, error);

    // Check if console.error was called with the correct message
    expect(console.error).toHaveBeenCalledWith(`[ERROR] ${message}`, error);

    // Check if fs.appendFile was called with a correctly formatted log string
    // The safeStringify will include message and stack for Error objects
    expect(fs.appendFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.stringMatching(/\\[ERROR\\] This is an error message {.*"message":"Something went wrong".*}/)
    );
  });

  it('should handle multiple arguments', async () => {
    const message = 'Test with multiple args';
    const arg1 = { a: 1 };
    const arg2 = { b: 2 };
    await logger.info(message, arg1, arg2);
    expect(console.log).toHaveBeenCalledWith(`[INFO] ${message}`, arg1, arg2);
    expect(fs.appendFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.stringContaining('{"a":1} {"b":2}')
    );
  });

  it('should handle circular object references gracefully', async () => {
    const circularObj: any = {};
    circularObj.a = { b: circularObj };
    
    await logger.info('Circular test', circularObj);
    
    // Check that the log file write was attempted with a stringified version
    expect(fs.appendFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.stringContaining('[INFO] Circular test')
    );
  });
});