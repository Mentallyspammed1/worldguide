import { describe, expect, it } from 'vitest';
import { SITES } from '../constants';
import {
  parseBackupText,
  sanitizeRecentlyWatched,
  sanitizeVideoResult
} from './backup';

const safeVideo = {
  id: 'xvideos-test',
  title: 'A public listing',
  url: 'https://www.xvideos.com/video123',
  thumb: 'https://cdn.xvideos.com/thumb.jpg',
  previewVideo: 'https://cdn.xvideos.com/preview.mp4',
  site: 'xvideos',
  siteName: 'untrusted display name',
  siteColor: 'url(javascript:alert(1))'
};

describe('backup data validation', () => {
  it('normalizes video metadata from the selected source config', () => {
    expect(sanitizeVideoResult(safeVideo)).toMatchObject({
      siteName: SITES.xvideos.name,
      siteColor: SITES.xvideos.color,
      url: safeVideo.url,
      previewVideo: safeVideo.previewVideo
    });
  });

  it('rejects unsafe destinations and media schemes', () => {
    expect(sanitizeVideoResult({ ...safeVideo, url: 'javascript:alert(1)' })).toBeNull();
    const sanitized = sanitizeVideoResult({
      ...safeVideo,
      thumb: 'https://[::1]/image.jpg',
      previewVideo: 'data:text/html,bad'
    });
    expect(sanitized?.thumb).toBe('');
    expect(sanitized?.previewVideo).toBeUndefined();
  });

  it('migrates older recently-watched items that did not store a site key', () => {
    const item = sanitizeRecentlyWatched({
      id: 'old-item',
      title: 'Old title',
      siteName: 'XVideos',
      siteColor: '#ff6600',
      thumb: '',
      url: 'https://www.xvideos.com/video123',
      watchedAt: 123
    });
    expect(item).toMatchObject({ site: 'xvideos', siteName: 'XVideos', watchedAt: 123 });
  });

  it('parses a backup and skips invalid entries', () => {
    const backup = parseBackupText(JSON.stringify({
      favorites: [safeVideo, { ...safeVideo, url: 'javascript:alert(1)' }],
      history: ['  safe search  ', 'safe search', 'x'.repeat(200)],
      collections: [{ id: 'list-1', name: 'Saved', color: 'red', videoIds: [], createdAt: 1 }]
    }));
    expect(backup.favorites).toHaveLength(1);
    expect(backup.history).toEqual(['safe search', 'x'.repeat(120)]);
    expect(backup.collections?.[0].color).toBe('#ff2a6d');
  });

  it('rejects malformed backup structure', () => {
    expect(() => parseBackupText('not json')).toThrow('not valid JSON');
    expect(() => parseBackupText(JSON.stringify({ favorites: {} }))).toThrow('must be a list');
    expect(() => parseBackupText(JSON.stringify({ unknown: [] }))).toThrow('not a Pyrmsearch backup');
  });
});
