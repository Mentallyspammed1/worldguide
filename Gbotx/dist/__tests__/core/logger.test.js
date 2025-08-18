// __tests__/core/logger.test.ts
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { logger } from '../../twin-range-bot/src/core/logger';
import fs from 'fs/promises';
// Mock the fs/promises module to avoid actual file I/O
vi.mock('fs/promises', () => ({
    default: {
        appendFile: vi.fn(),
    },
}));
describe('Logger', () => {
    beforeEach(() => {
        // Reset mocks before each test
        vi.resetAllMocks();
        // Spy on console methods
        vi.spyOn(console, 'log').mockImplementation(() => { });
        vi.spyOn(console, 'error').mockImplementation(() => { });
    });
    afterEach(() => {
        // Restore original console methods
        vi.restoreAllMocks();
    });
    it('should log info messages to console and file', async () => {
        const message = 'This is an info message';
        const arg1 = { data: 'test' };
        logger.info(message, arg1);
        // Check if console.log was called with the correct message
        expect(console.log).toHaveBeenCalledWith(expect.stringContaining('[INFO]'), expect.stringContaining(message), arg1);
        // Check if fs.appendFile was called with a correctly formatted log string
        expect(fs.appendFile).toHaveBeenCalledWith(expect.any(String), expect.stringMatching(/\[INFO\] This is an info message {"data":"test"}/));
    });
    it('should log error messages to console and file', async () => {
        const message = 'This is an error message';
        const error = new Error('Something went wrong');
        logger.error(message, error);
        // Check if console.error was called with the correct message
        expect(console.error).toHaveBeenCalledWith(expect.stringContaining('[ERROR]'), expect.stringContaining(message), error);
        // Check if fs.appendFile was called with a correctly formatted log string
        expect(fs.appendFile).toHaveBeenCalledWith(expect.any(String), expect.stringMatching(/\[ERROR\] This is an error message {"message":"Something went wrong"}/));
    });
    it('should handle multiple arguments', () => {
        logger.info('Test with multiple args', { a: 1 }, { b: 2 });
        expect(console.log).toHaveBeenCalledWith(expect.stringContaining('Test with multiple args'), { a: 1 }, { b: 2 });
        expect(fs.appendFile).toHaveBeenCalledWith(expect.any(String), expect.stringContaining('{"a":1} {"b":2}'));
    });
    it('should handle circular object references gracefully', () => {
        const circularObj = {};
        circularObj.a = { b: circularObj };
        // This test just ensures that the logger doesn't crash on circular refs.
        // JSON.stringify will throw, so we expect our logger to handle it.
        // The current implementation will fail here, which is a good test case.
        // A robust logger should use a safe stringify function.
        expect(() => logger.info('Circular test', circularObj)).not.toThrow();
    });
});
