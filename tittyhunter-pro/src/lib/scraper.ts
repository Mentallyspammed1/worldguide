import { SITES, PROXIES, SiteConfig } from '../constants';

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

// Convert any relative link or schema-less URL to an absolute URL
export function abs(base: string, href: string | null | undefined): string {
  if (!href) return '';
  const cleanHref = href.trim();
  if (
    cleanHref.startsWith('javascript:') ||
    cleanHref === '#' ||
    cleanHref.startsWith('data:')
  ) {
    return '';
  }
  if (cleanHref.startsWith('http://') || cleanHref.startsWith('https://')) {
    return cleanHref;
  }
  if (cleanHref.startsWith('//')) {
    return `https:${cleanHref}`;
  }
  const cleanBase = base.replace(/\/+$/, '');
  const path = cleanHref.startsWith('/') ? cleanHref : `/${cleanHref}`;
  return cleanBase + path;
}

// Helper to extract the best thumbnail URL from an element
export function extractThumbnail(el: Element, base: string): string {
  const blankGifs = [
    'lightbox-blank.gif',
    'blank.gif',
    'data:image/gif;base64'
  ];
  const attrs = [
    'data-src',
    'data-thumb',
    'data-mediumthumb',
    'data-original',
    'data-preview',
    'data-thumbnail',
    'srcset',
    'data-srcset',
    'src'
  ];

  const searchEl = el.tagName === 'IMG' ? el : el.querySelector('img') || el;

  for (const attr of attrs) {
    let val = searchEl.getAttribute(attr);
    if (val && !blankGifs.some(b => val!.includes(b))) {
      if (attr.includes('srcset')) {
        val = val.split(',')[0].trim().split(' ')[0];
      }
      if (val && !val.startsWith('data:')) {
        return abs(base, val).replace(/THUMBNUM/g, '1');
      }
    }
  }

  // Also check if any parent or child has a background-image in inline style
  const style =
    searchEl.getAttribute('style') || el.getAttribute('style') || '';
  const bgMatch = style.match(/url\(['"]?(.*?)['"]?\)/);
  if (bgMatch && bgMatch[1] && !blankGifs.some(b => bgMatch[1].includes(b))) {
    return abs(base, bgMatch[1]);
  }

  return '';
}

// Helper to extract preview video URL (.mp4 or .webm) from attributes
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

  // Check the element itself, links, images, and child elements
  const candidates = [
    el,
    ...Array.from(el.querySelectorAll('a, img, div, span, video, source'))
  ];

  for (const c of candidates) {
    for (const attr of previewAttrs) {
      const val = c.getAttribute(attr);
      if (
        val &&
        (val.includes('.mp4') ||
          val.includes('.webm') ||
          val.includes('/get_file/') ||
          val.includes('preview'))
      ) {
        return abs(base, val);
      }
    }
    // Also check standard video/source elements
    if (c.tagName === 'VIDEO' || c.tagName === 'SOURCE') {
      const src = c.getAttribute('src');
      if (src) return abs(base, src);
    }
  }

  return '';
}

// Parse HTML text into structured VideoResult items
export function parseHtml(
  html: string,
  siteKey: string,
  base: string
): VideoResult[] {
  const cfg = SITES[siteKey];
  const doc = new DOMParser().parseFromString(html, 'text/html');

  // Strip scripts and styles to prevent parsing delays
  doc.querySelectorAll('script, style, noscript').forEach(n => n.remove());

  const cardSelectors =
    cfg?.cardSel ||
    'div.thumb-block, div.video-item, div.mb, .video-card, .video-thumb';
  let elements = Array.from(doc.querySelectorAll(cardSelectors));

  // If specific selectors found nothing, try general video-item queries
  if (elements.length === 0) {
    elements = Array.from(
      doc.querySelectorAll(
        '.thumb, .card, [data-video-id], div[class*="video"]'
      )
    );
  }

  const seen = new Set<string>();
  const results: VideoResult[] = [];

  for (const el of elements) {
    // Find video link
    const linkEl =
      (cfg ? el.querySelector(cfg.linkSel) : null) ||
      el.querySelector('a[href*="/video"], a[href*="watch"], a[href]');
    if (!linkEl) continue;

    const rawHref = linkEl.getAttribute('href');
    if (!rawHref || rawHref === '#' || rawHref.startsWith('javascript:'))
      continue;

    const fullUrl = abs(base, rawHref);
    if (seen.has(fullUrl)) continue;
    seen.add(fullUrl);

    // Extract title
    let title = '';
    const titleEl = (cfg ? el.querySelector(cfg.titleSel) : null) || linkEl;
    if (titleEl) {
      title = (
        titleEl.getAttribute('title') ||
        titleEl.getAttribute('aria-label') ||
        titleEl.textContent ||
        ''
      ).trim();
    }
    if (!title) {
      const imgEl = el.querySelector('img');
      title = (
        imgEl?.getAttribute('alt') ||
        imgEl?.getAttribute('title') ||
        ''
      ).trim();
    }
    title = title.replace(/\s+/g, ' ').slice(0, 140) || 'Untitled Video';

    // Extract thumbnail
    const thumb = extractThumbnail(el, base);

    // Extract duration
    let duration = '';
    if (cfg?.durationSel) {
      const durEl = el.querySelector(cfg.durationSel);
      duration = durEl?.textContent?.trim() || '';
    }
    if (!duration) {
      const durMatch = el.textContent?.match(
        /(\d{1,2}:\d{2}(?::\d{2})?|\d+\s*min)/i
      );
      if (durMatch) duration = durMatch[1].trim();
    }

    // Extract views
    let views = '';
    if (cfg?.viewsSel) {
      const vEl = el.querySelector(cfg.viewsSel);
      views = vEl?.textContent?.replace(/views|views/gi, '')?.trim() || '';
      if (
        views &&
        !views.toLowerCase().includes('view') &&
        !views.includes('%')
      ) {
        views = `${views} views`;
      }
    }
    if (!views) {
      const viewMatch = el.textContent?.match(
        /(\d+(?:\.\d+)?[KM]?\s*(?:views|Views))/
      );
      if (viewMatch) views = viewMatch[1].trim();
    }

    // Extract preview video if available
    const previewVideo = extractPreviewVideo(el, base);

    const safeHash = Math.abs(
      fullUrl
        .split('')
        .reduce((acc, char) => (acc << 5) - acc + char.charCodeAt(0), 0)
    ).toString(36);
    const videoId = `${siteKey}-${safeHash}-${results.length}`;

    results.push({
      id: videoId,
      title,
      url: fullUrl,
      thumb,
      previewVideo,
      duration,
      views,
      site: siteKey,
      siteName: cfg?.name || siteKey.toUpperCase(),
      siteColor: cfg?.color || '#ff2a6d'
    });

    if (results.length >= 36) break;
  }

  return results;
}

// Main scrape function with multi-tier fallback (server endpoint -> CORS proxies)
export async function scrapeSite(
  siteKey: string,
  query: string,
  page: number
): Promise<VideoResult[]> {
  const cfg = SITES[siteKey] || SITES.xvideos;
  const targetUrl = cfg.buildUrl(query, page);

  // 1. Try our server-side proxy route first (bypasses browser CORS & bot restrictions)
  try {
    const serverUrl = `/api/scrape?site=${encodeURIComponent(siteKey)}&query=${encodeURIComponent(query)}&page=${page}`;
    const serverResp = await fetch(serverUrl, {
      signal: AbortSignal.timeout(8000)
    });
    if (serverResp.ok) {
      const data = await serverResp.json();
      if (data.success && data.html) {
        const parsed = parseHtml(data.html, siteKey, cfg.base);
        if (parsed.length > 0) {
          return parsed;
        }
      }
    }
  } catch {
    // Continue to fallback proxies if server endpoint isn't reachable
  }

  // 2. Client-side fallback via CORS proxies
  for (const proxyFn of PROXIES) {
    try {
      const cacheBust = targetUrl.includes('?')
        ? `&_t=${Date.now()}`
        : `?_t=${Date.now()}`;
      const proxyUrl = proxyFn(targetUrl + cacheBust);
      const resp = await fetch(proxyUrl, {
        headers: { Accept: 'text/html' },
        signal: AbortSignal.timeout(6000)
      });

      if (!resp.ok) continue;
      const html = await resp.text();
      if (!html || html.length < 500) continue;

      const parsed = parseHtml(html, siteKey, cfg.base);
      if (parsed.length > 0) {
        return parsed;
      }
    } catch {
      // Try next proxy
    }
  }

  return [];
}

// Extract the full video file URL from a video page by fetching its HTML
// and looking for common video source patterns (og:video, <video>, <source>, .mp4/.webm URLs)
export async function fetchFullVideoUrl(videoPageUrl: string): Promise<string> {
  try {
    const resp = await fetch(videoPageUrl, {
      headers: {
        Accept: 'text/html,application/xhtml+xml',
        Referer: new URL(videoPageUrl).origin,
        'User-Agent':
          'Mozilla/5.0 (Linux; Android 10; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36'
      },
      signal: AbortSignal.timeout(10000)
    });
    if (!resp.ok) return '';
    const html = await resp.text();
    if (!html || html.length < 500) return '';

    // 1. og:video meta tag
    const ogVideo = html.match(
      /<meta[^>]+property=["']og:video["'][^>]+content=["']([^"']+)["']/i
    );
    if (ogVideo?.[1]) return ogVideo[1];

    // 2. twitter:player meta tag
    const twitterPlayer = html.match(
      /<meta[^>]+name=["']twitter:player["'][^>]+content=["']([^"']+)["']/i
    );
    if (twitterPlayer?.[1]) return twitterPlayer[1];

    // 3. <video src="..."> or <source src="...">
    const videoSrc = html.match(
      /<(?:video|source)[^>]+src=["']([^"']+\.(?:mp4|webm|mov)[^"']*)["']/i
    );
    if (videoSrc?.[1]) return videoSrc[1];

    // 4. Any quoted .mp4 / .webm URL in the HTML
    const mp4Matches = html.match(
      /["'](https?:[^"']+\.(?:mp4|webm)[^"']*)["']/gi
    );
    if (mp4Matches && mp4Matches.length > 0) {
      // Prefer URLs that look like actual video files (not ads/trackers)
      const clean = mp4Matches
        .map(m => m.replace(/["']/g, ''))
        .filter(
          u =>
            !u.includes('pixel') &&
            !u.includes('track') &&
            !u.includes('beacon')
        );
      if (clean.length > 0) return clean[0];
    }

    // 5. data-src or data-video attributes
    const dataVideo = html.match(
      /data-(?:src|video|video-src)=["']([^"']+)["']/i
    );
    if (dataVideo?.[1]) return dataVideo[1];

    return '';
  } catch {
    return '';
  }
}

// Test all configured sites and return health status
export async function testAllSites(): Promise<SiteTestResult[]> {
  // First attempt: use server test endpoint
  try {
    const resp = await fetch('/api/test-sites', {
      signal: AbortSignal.timeout(10000)
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data.sites && Array.isArray(data.sites)) {
        return data.sites.map((s: any) => ({
          id: s.id,
          name: s.name,
          base: s.base,
          icon: s.icon,
          color: SITES[s.id]?.color || '#ff2a6d',
          status: s.status,
          ok: s.ok,
          latencyMs: s.latencyMs,
          error: s.error
        }));
      }
    }
  } catch {
    // Fall back to client-side test
  }

  // Fallback: client-side ping check through CORS proxy
  const tests = Object.values(SITES).map(async cfg => {
    const start = Date.now();
    try {
      const testUrl = cfg.buildUrl('test', 1);
      const proxyUrl = `https://api.allorigins.win/raw?url=${encodeURIComponent(testUrl)}`;
      const res = await fetch(proxyUrl, { signal: AbortSignal.timeout(6000) });
      const latency = Date.now() - start;
      return {
        id: cfg.id,
        name: cfg.name,
        base: cfg.base,
        icon: cfg.icon,
        color: cfg.color,
        status: res.status,
        ok: res.ok,
        latencyMs: latency
      };
    } catch (err: any) {
      return {
        id: cfg.id,
        name: cfg.name,
        base: cfg.base,
        icon: cfg.icon,
        color: cfg.color,
        status: 0,
        ok: false,
        latencyMs: Date.now() - start,
        error: err.message || 'Timeout / Offline'
      };
    }
  });

  return Promise.all(tests);
}
