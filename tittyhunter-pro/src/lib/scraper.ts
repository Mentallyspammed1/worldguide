import { SITES, type SiteConfig } from '../constants';
import { isSafeHttpsUrl } from './safe-url';

const CLIENT_PROXIES = [
  (url: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
  (url: string) => `https://corsproxy.io/?url=${encodeURIComponent(url)}`,
  (url: string) => `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(url)}`
];

async function fetchViaClientProxies(url: string, signal?: AbortSignal): Promise<Response> {
  const lastError: unknown[] = [];
  for (const proxy of CLIENT_PROXIES) {
    try {
      const response = await fetch(proxy(url), {
        signal: signal || AbortSignal.timeout(15_000),
        headers: { Accept: 'application/json' }
      });
      if (response.ok) return response;
    } catch (error) {
      lastError.push(error);
    }
  }
  throw new Error(`All client proxies failed: ${lastError.map(e => e instanceof Error ? e.message : String(e)).join('; ')}`);
}

export interface VideoResult {
  id: string;
  title: string;
  url: string;
  thumb: string;
  previewVideo?: string;
  duration?: string;
  views?: string;
  site: string;
  siteName: string;
  siteColor: string;
}

export interface SiteTestResult {
  id: string;
  name: string;
  base: string;
  icon: string;
  color: string;
  status: number;
  ok: boolean;
  latencyMs: number;
  error?: string;
}

const MAX_QUERY_LENGTH = 120;
const MAX_PAGE = 100;
const MAX_RESULTS = 36;

/** Resolve a scraped link safely. Only HTTPS URLs are returned. */
export function abs(base: string, href: string | null | undefined): string {
  if (!href) return '';
  const value = href.trim();
  if (!value || value === '#' || /^(?:javascript|data|vbscript|file):/i.test(value)) {
    return '';
  }

  try {
    const url = new URL(value, base);
    if (!isSafeHttpsUrl(url.toString())) {
      return '';
    }
    return url.toString();
  } catch {
    return '';
  }
}

function belongsToSite(value: string, base: string): boolean {
  try {
    const host = new URL(value).hostname.toLowerCase().replace(/\.$/, '');
    const baseHost = new URL(base).hostname.toLowerCase().replace(/^www\./, '').replace(/\.$/, '');
    return host === baseHost || host.endsWith(`.${baseHost}`);
  } catch {
    return false;
  }
}

function isPlayableMediaUrl(value: string): boolean {
  try {
    const pathname = new URL(value).pathname;
    return (
      /\.(?:mp4|webm|m4v|mov|m3u8)$/i.test(pathname) ||
      /\/get_file\//i.test(pathname)
    );
  } catch {
    return false;
  }
}

/** Extract the best thumbnail URL from an element without accepting inline data URLs. */
export function extractThumbnail(el: Element, base: string): string {
  const blankGifs = ['lightbox-blank.gif', 'blank.gif', 'data:image/gif;base64'];
  const attrs = [
    'data-src',
    'data-thumb',
    'data-mediumthumb',
    'data-original',
    'data-preview',
    'data-thumbnail',
    'data-srcset',
    'srcset',
    'src'
  ];
  const searchEl = el.tagName === 'IMG' ? el : el.querySelector('img') || el;

  for (const attr of attrs) {
    let value = searchEl.getAttribute(attr);
    if (!value || blankGifs.some(blank => value!.includes(blank))) continue;
    if (attr.includes('srcset')) value = value.split(',')[0].trim().split(/\s+/)[0];
    if (!value || value.startsWith('data:')) continue;
    const url = abs(base, value).replace(/THUMBNUM/g, '1');
    if (url) return url;
  }

  const style = searchEl.getAttribute('style') || el.getAttribute('style') || '';
  const backgroundMatch = style.match(/url\(["']?(.*?)["']?\)/i);
  if (backgroundMatch?.[1] && !blankGifs.some(blank => backgroundMatch[1].includes(blank))) {
    return abs(base, backgroundMatch[1]);
  }

  return '';
}

/** Extract a likely short preview media URL; regular page links are ignored. */
export function extractPreviewVideo(el: Element, base: string): string {
  const previewAttrs = [
    'data-pvv',
    'data-preview-custom',
    'data-previewvideo',
    'data-preview-video',
    'data-preview',
    'data-mp4',
    'data-video-preview',
    'data-trailer',
    'data-video'
  ];
  const candidates = [el, ...Array.from(el.querySelectorAll('a, img, div, span, video, source'))];

  for (const element of candidates) {
    for (const attr of previewAttrs) {
      const value = element.getAttribute(attr);
      if (!value) continue;
      const url = abs(base, value);
      if (url && isPlayableMediaUrl(url)) return url;
    }
    if (element.tagName === 'VIDEO' || element.tagName === 'SOURCE') {
      const source = element.getAttribute('src') || element.getAttribute('data-src');
      const url = abs(base, source);
      if (url && isPlayableMediaUrl(url)) return url;
    }
  }

  return '';
}

function safeQuerySelector(parent: ParentNode, selector?: string): Element | null {
  if (!selector) return null;
  try {
    return parent.querySelector(selector);
  } catch {
    return null;
  }
}

function stableId(siteKey: string, url: string): string {
  let hash = 2_166_136_261;
  for (let index = 0; index < url.length; index += 1) {
    hash = Math.imul(hash ^ url.charCodeAt(index), 16_777_619);
  }
  return `${siteKey}-${(hash >>> 0).toString(36)}`;
}

/** Parse a site's public search-result HTML into safe, display-ready cards. */
export function parseHtml(html: string, siteKey: string, base: string): VideoResult[] {
  if (typeof DOMParser === 'undefined' || !html) return [];

  let doc: Document;
  try {
    doc = new DOMParser().parseFromString(html, 'text/html');
  } catch {
    return [];
  }
  doc.querySelectorAll('script, style, noscript, template').forEach(node => node.remove());

  const config = SITES[siteKey];
  const defaultCardSelector = 'div.thumb-block, div.video-item, div.mb, .video-card, .video-thumb';
  let elements: Element[] = [];
  try {
    elements = Array.from(doc.querySelectorAll(config?.cardSel || defaultCardSelector));
  } catch {
    elements = [];
  }
  if (elements.length === 0) {
    try {
      elements = Array.from(doc.querySelectorAll('.thumb, .card, [data-video-id], div[class*="video"]'));
    } catch {
      elements = [];
    }
  }

  const seen = new Set<string>();
  const results: VideoResult[] = [];
  const siteBase = config?.base || base;

  for (const element of elements) {
    const link =
      safeQuerySelector(element, config?.linkSel) ||
      safeQuerySelector(element, 'a[href*="/video"], a[href*="watch"], a[href]');
    const rawHref = link?.getAttribute('href');
    if (!rawHref) continue;

    const url = abs(siteBase, rawHref);
    if (!url || !belongsToSite(url, siteBase) || seen.has(url)) continue;
    seen.add(url);

    const titleElement = safeQuerySelector(element, config?.titleSel) || link;
    const image = safeQuerySelector(element, 'img');
    const title = (
      titleElement?.getAttribute('title') ||
      titleElement?.getAttribute('aria-label') ||
      titleElement?.textContent ||
      image?.getAttribute('alt') ||
      image?.getAttribute('title') ||
      'Untitled video'
    )
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 140);

    const durationElement = safeQuerySelector(element, config?.durationSel);
    const duration =
      durationElement?.textContent?.replace(/\s+/g, ' ').trim() ||
      element.textContent?.match(/(\d{1,2}:\d{2}(?::\d{2})?|\d+\s*min)/i)?.[1]?.trim() ||
      '';

    const viewsElement = safeQuerySelector(element, config?.viewsSel);
    let views = viewsElement?.textContent?.replace(/\s+/g, ' ').trim() || '';
    if (views) {
      views = views.replace(/\bviews?\b/gi, '').trim();
      if (views && !views.includes('%')) views = `${views} views`;
    } else {
      views = element.textContent?.match(/(\d+(?:\.\d+)?\s*[KM]?\s*views?)/i)?.[1]?.trim() || '';
    }

    results.push({
      id: stableId(siteKey, url),
      title: title || 'Untitled video',
      url,
      thumb: extractThumbnail(element, siteBase),
      previewVideo: extractPreviewVideo(element, siteBase),
      duration,
      views,
      site: siteKey,
      siteName: config?.name || siteKey.toUpperCase(),
      siteColor: config?.color || '#ff2a6d'
    });

    if (results.length >= MAX_RESULTS) break;
  }

  return results;
}

function readApiError(payload: unknown, status: number): string {
  if (
    payload &&
    typeof payload === 'object' &&
    'error' in payload &&
    typeof payload.error === 'string'
  ) {
    return payload.error;
  }
  return `The search service returned HTTP ${status}.`;
}

/** Search through the same-origin API so queries are never sent to third-party CORS relays. */
export async function scrapeSite(
  siteKey: string,
  query: string,
  page: number,
  signal?: AbortSignal
): Promise<VideoResult[]> {
  const config: SiteConfig | undefined = SITES[siteKey];
  if (!config) throw new Error('Choose a supported source site.');
  const normalizedQuery = query.trim();
  if (!normalizedQuery || normalizedQuery.length > MAX_QUERY_LENGTH) {
    throw new Error(`Search query must be between 1 and ${MAX_QUERY_LENGTH} characters.`);
  }
  if (!Number.isInteger(page) || page < 1 || page > MAX_PAGE) {
    throw new Error(`Page must be an integer between 1 and ${MAX_PAGE}.`);
  }

  const params = new URLSearchParams({
    site: siteKey,
    query: normalizedQuery,
    page: String(page)
  });
  const serverUrl = `/api/scrape?${params.toString()}`;
  try {
    const response = await fetch(serverUrl, {
      signal: signal || AbortSignal.timeout(18_000),
      headers: { Accept: 'application/json' }
    });
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok) throw new Error(readApiError(payload, response.status));
    if (
      !payload ||
      typeof payload !== 'object' ||
      !('success' in payload) ||
      payload.success !== true ||
      !('html' in payload) ||
      typeof payload.html !== 'string'
    ) {
      throw new Error('The search service returned an invalid response.');
    }

    return parseHtml(payload.html, siteKey, config.base);
  } catch (serverError) {
    // Fall back to client-side proxies if the server API is unavailable.
    const proxyUrl = config.buildUrl(normalizedQuery, page);
    const response = await fetchViaClientProxies(proxyUrl, signal);
    const html = await response.text();
    return parseHtml(html, siteKey, config.base);
  }
}

/** Ask the server to inspect a public result page for a playable media source. */
export async function fetchFullVideoUrl(videoPageUrl: string, siteKey: string): Promise<string> {
  if (!SITES[siteKey]) return '';
  const params = new URLSearchParams({ site: siteKey, url: videoPageUrl });
  try {
    const response = await fetch(`/api/video-source?${params.toString()}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(12_000)
    });
    if (!response.ok) return '';
    const payload: unknown = await response.json();
    if (
      !payload ||
      typeof payload !== 'object' ||
      !('url' in payload) ||
      typeof payload.url !== 'string'
    ) {
      return '';
    }
    const source = abs(videoPageUrl, payload.url);
    return source && isPlayableMediaUrl(source) ? source : '';
  } catch {
    return '';
  }
}

/** Read source-site diagnostics from the server; falls back to client-side proxies. */
export async function testAllSites(siteId?: string): Promise<SiteTestResult[]> {
  if (siteId && !SITES[siteId]) throw new Error('Choose a supported source site.');
  const configs = siteId ? [SITES[siteId]] : Object.values(SITES);
  const endpoint = siteId
    ? `/api/test-sites?${new URLSearchParams({ site: siteId }).toString()}`
    : '/api/test-sites';

  try {
    const response = await fetch(endpoint, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(siteId ? 10_000 : 30_000)
    });
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok) throw new Error(readApiError(payload, response.status));
    if (!payload || typeof payload !== 'object' || !('sites' in payload) || !Array.isArray(payload.sites)) {
      throw new Error('The diagnostics service returned an invalid response.');
    }

    return payload.sites.flatMap((value): SiteTestResult[] => {
      if (!value || typeof value !== 'object' || !('id' in value) || typeof value.id !== 'string') {
        return [];
      }
      const config = SITES[value.id];
      if (!config) return [];
      const item = value as Record<string, unknown>;
      return [
        {
          id: config.id,
          name: config.name,
          base: config.base,
          icon: config.icon,
          color: config.color,
          status: typeof item.status === 'number' ? item.status : 0,
          ok: item.ok === true,
          latencyMs: typeof item.latencyMs === 'number' ? item.latencyMs : 0,
          ...(typeof item.error === 'string' ? { error: item.error } : {})
        }
      ];
    });
  } catch (serverError) {
    // Fall back to client-side proxies if the server API is unavailable.
    const results: SiteTestResult[] = [];
    for (const config of configs) {
      const startedAt = Date.now();
      try {
        const proxyUrl = config.buildUrl('test', 1);
        const response = await fetchViaClientProxies(proxyUrl, AbortSignal.timeout(10_000));
        results.push({
          id: config.id,
          name: config.name,
          base: config.base,
          icon: config.icon,
          color: config.color,
          status: response.status,
          ok: response.ok,
          latencyMs: Date.now() - startedAt
        });
      } catch (error) {
        results.push({
          id: config.id,
          name: config.name,
          base: config.base,
          icon: config.icon,
          color: config.color,
          status: 0,
          ok: false,
          latencyMs: Date.now() - startedAt,
          error: error instanceof Error ? error.message : 'Source site could not be reached.'
        });
      }
    }
    return results;
  }
}
