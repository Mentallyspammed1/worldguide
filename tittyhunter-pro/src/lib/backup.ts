import { SITES } from '../constants';
import type { VideoResult } from './scraper';
import { isSafeHttpsUrl } from './safe-url';

export interface SavedScreenshot {
  id: string;
  title: string;
  siteName: string;
  siteColor: string;
  thumb: string;
  dataUrl: string;
  capturedAt: number;
}

export interface RecentlyWatchedItem extends VideoResult {
  watchedAt: number;
}

export interface Collection {
  id: string;
  name: string;
  color: string;
  videoIds: string[];
  videos: VideoResult[];
  createdAt: number;
}

export interface BackupData {
  favorites?: VideoResult[];
  screenshots?: SavedScreenshot[];
  history?: string[];
  recentlyWatched?: RecentlyWatchedItem[];
  collections?: Collection[];
  exportedAt?: number;
}

const MAX_BACKUP_BYTES = 20 * 1024 * 1024;
const MAX_SCREENSHOT_CHARS = 2_000_000;
const MAX_SAVED_ITEMS = 500;
const SAFE_COLOR = /^#[0-9a-f]{6}$/i;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function boundedText(value: unknown, maxLength: number): string {
  return typeof value === 'string' ? value.trim().slice(0, maxLength) : '';
}

function belongsToSite(url: string, base: string): boolean {
  try {
    const hostname = new URL(url).hostname.toLowerCase();
    const domain = new URL(base).hostname.toLowerCase().replace(/^www\./, '');
    return hostname === domain || hostname.endsWith(`.${domain}`);
  } catch {
    return false;
  }
}

function normalizeMediaUrl(value: unknown): string | undefined {
  if (!isSafeHttpsUrl(value)) return undefined;
  try {
    const pathname = new URL(value).pathname;
    return /\.(?:mp4|webm|m4v|mov|m3u8)$/i.test(pathname) || /\/get_file\//i.test(pathname)
      ? value
      : undefined;
  } catch {
    return undefined;
  }
}

export function sanitizeVideoResult(value: unknown): VideoResult | null {
  if (!isRecord(value) || typeof value.site !== 'string') return null;
  const config = SITES[value.site];
  if (!config || !isSafeHttpsUrl(value.url) || !belongsToSite(value.url, config.base)) {
    return null;
  }

  const id = boundedText(value.id, 160) || `${config.id}-${encodeURIComponent(value.url).slice(-80)}`;
  const title = boundedText(value.title, 200) || 'Untitled video';
  const thumb = isSafeHttpsUrl(value.thumb) ? value.thumb : '';
  const previewVideo = normalizeMediaUrl(value.previewVideo);
  const duration = boundedText(value.duration, 30);
  const views = boundedText(value.views, 40);

  return {
    id,
    title,
    url: value.url,
    thumb,
    ...(previewVideo ? { previewVideo } : {}),
    ...(duration ? { duration } : {}),
    ...(views ? { views } : {}),
    site: config.id,
    siteName: config.name,
    siteColor: config.color
  };
}

export function sanitizeScreenshot(value: unknown): SavedScreenshot | null {
  if (!isRecord(value)) return null;
  const dataUrl = value.dataUrl;
  if (
    typeof dataUrl !== 'string' ||
    dataUrl.length > MAX_SCREENSHOT_CHARS ||
    !/^data:image\/(?:png|jpeg|webp);base64,[a-z0-9+/]+=*$/i.test(dataUrl)
  ) {
    return null;
  }
  const siteName = boundedText(value.siteName, 80) || 'Source site';
  const site = Object.values(SITES).find(config => config.name === siteName);
  const thumb = isSafeHttpsUrl(value.thumb) ? value.thumb : '';
  const capturedAt = typeof value.capturedAt === 'number' && Number.isFinite(value.capturedAt)
    ? value.capturedAt
    : Date.now();

  return {
    id: boundedText(value.id, 160) || `screenshot-${capturedAt}`,
    title: boundedText(value.title, 200) || 'Saved screenshot',
    siteName: site?.name || siteName,
    siteColor: site?.color || (typeof value.siteColor === 'string' && SAFE_COLOR.test(value.siteColor) ? value.siteColor : '#05d9e8'),
    thumb,
    dataUrl,
    capturedAt
  };
}

export function sanitizeRecentlyWatched(value: unknown): RecentlyWatchedItem | null {
  if (!isRecord(value)) return null;
  const siteId = typeof value.site === 'string'
    ? value.site
    : Object.values(SITES).find(config => config.name === value.siteName)?.id;
  const video = sanitizeVideoResult({ ...value, site: siteId });
  if (!video) return null;
  const watchedAt = typeof value.watchedAt === 'number' && Number.isFinite(value.watchedAt)
    ? value.watchedAt
    : Date.now();
  return { ...video, watchedAt };
}

export function sanitizeCollection(value: unknown): Collection | null {
  if (!isRecord(value)) return null;
  const name = boundedText(value.name, 80);
  if (!name) return null;
  const videos = Array.isArray(value.videos)
    ? value.videos.map(sanitizeVideoResult).filter((item): item is VideoResult => item !== null)
    : [];
  const storedIds = Array.isArray(value.videoIds)
    ? value.videoIds.filter(isSafeHttpsUrl)
    : [];
  const videoIds = [...new Set([...videos.map(video => video.url), ...storedIds])].slice(0, MAX_SAVED_ITEMS);
  const color = typeof value.color === 'string' && SAFE_COLOR.test(value.color)
    ? value.color
    : '#ff2a6d';
  const createdAt = typeof value.createdAt === 'number' && Number.isFinite(value.createdAt)
    ? value.createdAt
    : Date.now();

  return {
    id: boundedText(value.id, 160) || `collection-${createdAt}`,
    name,
    color,
    videoIds,
    videos: videos.slice(0, MAX_SAVED_ITEMS),
    createdAt
  };
}

function sanitizeArray<T>(value: unknown, sanitizer: (item: unknown) => T | null): T[] {
  if (!Array.isArray(value)) return [];
  return value
    .slice(0, MAX_SAVED_ITEMS)
    .map(sanitizer)
    .filter((item): item is T => item !== null);
}

export function sanitizeHistory(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(
    value.map(item => boundedText(item, 120)).filter(Boolean)
  )].slice(0, 50);
}

export function parseBackupText(text: string): BackupData {
  if (new TextEncoder().encode(text).byteLength > MAX_BACKUP_BYTES) {
    throw new Error('Backup file is too large (20 MB maximum).');
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error('The selected file is not valid JSON.');
  }
  if (!isRecord(parsed)) throw new Error('The backup must contain a JSON object.');

  const keys = ['favorites', 'screenshots', 'history', 'recentlyWatched', 'collections'] as const;
  if (!keys.some(key => key in parsed)) throw new Error('This file is not a Pyrmsearch backup.');
  for (const key of keys) {
    if (key in parsed && !Array.isArray(parsed[key])) {
      throw new Error(`Backup field "${key}" must be a list.`);
    }
  }

  const backup: BackupData = {};
  if ('favorites' in parsed) backup.favorites = sanitizeArray(parsed.favorites, sanitizeVideoResult);
  if ('screenshots' in parsed) backup.screenshots = sanitizeArray(parsed.screenshots, sanitizeScreenshot);
  if ('history' in parsed) {
    backup.history = [...new Set(
      (parsed.history as unknown[])
        .map(item => boundedText(item, 120))
        .filter(Boolean)
    )].slice(0, 50);
  }
  if ('recentlyWatched' in parsed) {
    backup.recentlyWatched = sanitizeArray(parsed.recentlyWatched, sanitizeRecentlyWatched);
  }
  if ('collections' in parsed) backup.collections = sanitizeArray(parsed.collections, sanitizeCollection);
  if (typeof parsed.exportedAt === 'number' && Number.isFinite(parsed.exportedAt)) {
    backup.exportedAt = parsed.exportedAt;
  }
  return backup;
}
