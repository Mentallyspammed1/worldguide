import { describe, it, expect } from 'vitest';

// Test that constants module loads without errors
describe('constants module', () => {
  it('should import successfully', async () => {
    const mod = await import('./constants');
    expect(mod).toBeDefined();
  });
});
