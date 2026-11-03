// Global test setup for Vitest
import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

// jsdom does not implement these browser APIs used by the app shell.
// Some test suites run in the Node environment, so only patch when present.
if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  });
}

if (typeof Element !== 'undefined') {
  Element.prototype.scrollIntoView = vi.fn();
}
