import { describe, expect, it } from 'vitest';
import { SITES } from './constants';

describe('source configuration', () => {
  it('defines unique, secure source origins and valid search URL builders', () => {
    const ids = new Set<string>();
    for (const [key, site] of Object.entries(SITES)) {
      expect(site.id).toBe(key);
      expect(ids.has(site.id)).toBe(false);
      ids.add(site.id);
      expect(new URL(site.base).protocol).toBe('https:');
      for (const page of [1, 2]) {
        const url = new URL(site.buildUrl('public phrase', page));
        expect(url.protocol).toBe('https:');
        expect(url.hostname).toBe(new URL(site.base).hostname);
        expect(url.href).not.toContain('public phrase');
      }
    }
    expect(Object.keys(SITES)).toHaveLength(13);
  });
});
