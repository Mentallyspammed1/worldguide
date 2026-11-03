import 'dotenv/config';
import express, { type Request, type Response as ExpressResponse } from 'express';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import pino from 'pino';
import { randomBytes } from 'node:crypto';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { SITES, type SiteConfig } from './src/constants';
import { isSafePublicHostname } from './src/lib/safe-url';

const projectRoot = path.dirname(fileURLToPath(import.meta.url));
const logger = pino({ level: process.env.LOG_LEVEL || 'info' });
const maxHtmlBytes = 2 * 1024 * 1024;
const maxRedirects = 3;
const siteKeys = new Set(Object.keys(SITES));
const mediaHostAllowlist = (process.env.MEDIA_HOST_ALLOWLIST || '')
  .split(',')
  .map(normalizeHost)
  .filter(Boolean);

class ApiError extends Error {
  constructor(
    readonly statusCode: number,
    readonly publicMessage: string
  ) {
    super(publicMessage);
    this.name = 'ApiError';
  }
}

interface AppOptions {
  fetcher?: typeof fetch;
  isDevelopment?: boolean;
}

/** Add the per-response CSP nonce to every script emitted by Vite's dev HTML transform. */
export function withCspNonce(html: string, nonce: string): string {
  return html.replace(/<script\b/gi, `<script nonce="${nonce}"`);
}

function parseFrameAncestors(value: string | undefined): string[] {
  const sources = (value || "'self'").trim().split(/\s+/).filter(Boolean);
  return sources.length > 0 ? sources : ["'self'"];
}

function normalizeHost(host: string): string {
  return host.trim().toLowerCase().replace(/^\*?\./, '').replace(/\.$/, '');
}

function isPublicHostname(hostname: string): boolean {
  return isSafePublicHostname(hostname);
}

function hostMatches(hostname: string, domain: string): boolean {
  const host = normalizeHost(hostname);
  const allowedDomain = normalizeHost(domain);
  return host === allowedDomain || host.endsWith(`.${allowedDomain}`);
}

function siteDomain(config: SiteConfig): string {
  return normalizeHost(new URL(config.base).hostname).replace(/^www\./, '');
}

function isAllowedSiteUrl(value: string, config: SiteConfig): boolean {
  try {
    const url = new URL(value);
    return (
      url.protocol === 'https:' &&
      !url.username &&
      !url.password &&
      !url.port &&
      isPublicHostname(url.hostname) &&
      hostMatches(url.hostname, siteDomain(config))
    );
  } catch {
    return false;
  }
}

function isAllowedMediaUrl(value: string): boolean {
  try {
    const url = new URL(value);
    if (
      url.protocol !== 'https:' ||
      url.username ||
      url.password ||
      url.port ||
      !isPublicHostname(url.hostname)
    ) {
      return false;
    }

    const knownSiteDomains = Object.values(SITES).map(siteDomain);
    return [...knownSiteDomains, ...mediaHostAllowlist].some(domain =>
      hostMatches(url.hostname, domain)
    );
  } catch {
    return false;
  }
}

function getSiteConfig(value: unknown): SiteConfig {
  if (typeof value !== 'string' || !siteKeys.has(value)) {
    throw new ApiError(400, 'Choose a supported source site.');
  }
  return SITES[value];
}

function getQueryString(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.length > 2_048) {
    throw new ApiError(400, `A valid ${field} is required.`);
  }
  return value;
}

function getPageNumber(value: unknown): number {
  if (value === undefined) return 1;
  const raw = typeof value === 'string' ? value : '';
  if (!/^\d{1,3}$/.test(raw)) {
    throw new ApiError(400, 'Page must be an integer between 1 and 100.');
  }
  const page = Number(raw);
  if (!Number.isInteger(page) || page < 1 || page > 100) {
    throw new ApiError(400, 'Page must be an integer between 1 and 100.');
  }
  return page;
}

function sendApiError(res: ExpressResponse, error: unknown): void {
  const apiError =
    error instanceof ApiError
      ? error
      : new ApiError(502, 'The source site could not be reached.');
  res.status(apiError.statusCode).json({
    success: false,
    error: apiError.publicMessage
  });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && ['AbortError', 'TimeoutError'].includes(error.name);
}

async function fetchWithValidatedRedirects(
  initialUrl: string,
  isAllowedUrl: (candidate: string) => boolean,
  fetcher: typeof fetch,
  options: { timeoutMs: number; headers?: HeadersInit }
): Promise<Response> {
  let currentUrl = initialUrl;

  for (let redirectCount = 0; redirectCount <= maxRedirects; redirectCount += 1) {
    if (!isAllowedUrl(currentUrl)) {
      throw new ApiError(400, 'The requested URL is not an approved source.');
    }

    let upstream: Response;
    try {
      upstream = await fetcher(currentUrl, {
        method: 'GET',
        headers: options.headers,
        redirect: 'manual',
        signal: AbortSignal.timeout(options.timeoutMs)
      });
    } catch (error) {
      if (isAbortError(error)) {
        throw new ApiError(504, 'The source site took too long to respond.');
      }
      throw new ApiError(502, 'The source site could not be reached.');
    }

    if (![301, 302, 303, 307, 308].includes(upstream.status)) {
      return upstream;
    }

    const location = upstream.headers.get('location');
    await upstream.body?.cancel().catch(() => undefined);
    if (!location) {
      throw new ApiError(502, 'The source site returned an invalid redirect.');
    }
    if (redirectCount === maxRedirects) {
      throw new ApiError(502, 'The source site redirected too many times.');
    }

    try {
      currentUrl = new URL(location, currentUrl).toString();
    } catch {
      throw new ApiError(502, 'The source site returned an invalid redirect.');
    }
  }

  throw new ApiError(502, 'The source site could not be reached.');
}

async function readTextWithLimit(response: Response): Promise<string> {
  const declaredLength = Number(response.headers.get('content-length'));
  if (Number.isFinite(declaredLength) && declaredLength > maxHtmlBytes) {
    await response.body?.cancel().catch(() => undefined);
    throw new ApiError(413, 'The source page is larger than the allowed limit.');
  }

  if (!response.body) return '';

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxHtmlBytes) {
        await reader.cancel().catch(() => undefined);
        throw new ApiError(413, 'The source page is larger than the allowed limit.');
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(body);
}

async function fetchSiteHtml(
  url: string,
  config: SiteConfig,
  fetcher: typeof fetch,
  timeoutMs = 10_000
): Promise<{ html: string; status: number; finalUrl: string }> {
  const upstream = await fetchWithValidatedRedirects(
    url,
    candidate => isAllowedSiteUrl(candidate, config),
    fetcher,
    {
      timeoutMs,
      headers: {
        Accept: 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.1',
        'User-Agent': 'Pyrmsearch/1.0 (+https://localhost; public-page preview)'
      }
    }
  );

  if (!upstream.ok) {
    const status = upstream.status;
    await upstream.body?.cancel().catch(() => undefined);
    throw new ApiError(502, `The source site responded with HTTP ${status}.`);
  }

  const contentType = upstream.headers.get('content-type') || '';
  if (contentType && !/text\/html|application\/xhtml\+xml/i.test(contentType)) {
    await upstream.body?.cancel().catch(() => undefined);
    throw new ApiError(502, 'The source site did not return an HTML page.');
  }

  return {
    html: await readTextWithLimit(upstream),
    status: upstream.status,
    finalUrl: upstream.url || url
  };
}

function decodeHtmlAttribute(value: string): string {
  return value
    .replace(/\\\//g, '/')
    .replace(/\\u0026/gi, '&')
    .replace(/\\u003d/gi, '=')
    .replace(/&amp;/gi, '&')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&#x27;/gi, "'")
    .replace(/&#x2f;|&#47;/gi, '/');
}

function readAttributes(tag: string): Record<string, string> {
  const attributes: Record<string, string> = {};
  const pattern = /([^\s=<>/]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/g;
  for (const match of tag.matchAll(pattern)) {
    attributes[match[1].toLowerCase()] = decodeHtmlAttribute(
      match[2] ?? match[3] ?? match[4] ?? ''
    );
  }
  return attributes;
}

function isPlayableMediaPath(value: string): boolean {
  try {
    const pathName = new URL(value).pathname;
    return /\.(?:mp4|webm|m4v|mov|m3u8)$/i.test(pathName) || /\/get_file\//i.test(pathName);
  } catch {
    return false;
  }
}

function extractVideoSource(html: string, pageUrl: string): string {
  const candidates: string[] = [];
  const tags = html.matchAll(/<(meta|video|source)\b[^>]*>/gi);

  for (const match of tags) {
    const tag = match[0];
    const kind = match[1].toLowerCase();
    const attributes = readAttributes(tag);
    if (kind === 'meta') {
      const metaType = (attributes.property || attributes.name || '').toLowerCase();
      if (['og:video', 'og:video:url', 'twitter:player:stream'].includes(metaType)) {
        if (attributes.content) candidates.push(attributes.content);
      }
    } else {
      const source = attributes.src || attributes['data-src'];
      if (source) candidates.push(source);
    }
  }

  const fileUrls = html.matchAll(/["']((?:https?:)?[^"'\s<>]+\.(?:mp4|webm|m4v|mov)(?:\?[^"'\s<>]*)?)["']/gi);
  for (const match of fileUrls) candidates.push(decodeHtmlAttribute(match[1]));

  for (const candidate of candidates) {
    try {
      const url = new URL(candidate, pageUrl);
      if (
        url.protocol === 'https:' &&
        !url.username &&
        !url.password &&
        isPlayableMediaPath(url.toString()) &&
        isAllowedMediaUrl(url.toString())
      ) {
        return url.toString();
      }
    } catch {
      // Ignore malformed source attributes from third-party HTML.
    }
  }

  return '';
}

export function createApp(options: AppOptions = {}): express.Express {
  const app = express();
  const fetcher = options.fetcher || fetch;
  const isDevelopment =
    options.isDevelopment ??
    (process.argv.includes('--dev') || process.env.NODE_ENV === 'development');
  const frameAncestors = process.env.FRAME_ANCESTORS?.trim()
    ? parseFrameAncestors(process.env.FRAME_ANCESTORS)
    : isDevelopment
      ? null
      : ["'self'"];

  app.disable('x-powered-by');
  app.use((_req, res, next) => {
    res.locals.cspNonce = randomBytes(16).toString('base64url');
    next();
  });
  app.use(
    helmet({
      frameguard: false,
      hsts: isDevelopment ? false : undefined,
      crossOriginOpenerPolicy: isDevelopment ? false : undefined,
      crossOriginResourcePolicy: isDevelopment ? false : undefined,
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: isDevelopment
            ? ["'self'", "'unsafe-eval'", (_req, res) => `'nonce-${(res as ExpressResponse).locals.cspNonce}'`]
            : ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          imgSrc: ["'self'", 'https:', 'data:', 'blob:'],
          mediaSrc: ["'self'", 'https:', 'blob:'],
          connectSrc: ["'self'", 'https:', 'wss:', 'ws:'],
          fontSrc: ["'self'", 'data:'],
          workerSrc: ["'self'", 'blob:'],
          objectSrc: ["'none'"],
          baseUri: ["'self'"],
          frameAncestors,
          formAction: ["'self'", 'https:'],
          upgradeInsecureRequests: isDevelopment ? null : []
        }
      },
      referrerPolicy: { policy: 'no-referrer' }
    })
  );

  // Avoid logging query strings: search terms may be sensitive.
  app.use((req, res, next) => {
    const startedAt = Date.now();
    res.on('finish', () => {
      logger.info(
        {
          method: req.method,
          path: req.path,
          status: res.statusCode,
          durationMs: Date.now() - startedAt
        },
        'http_request'
      );
    });
    next();
  });

  const apiLimiter = rateLimit({
    windowMs: 60_000,
    limit: 60,
    standardHeaders: 'draft-8',
    legacyHeaders: false,
    handler: (_req, res) =>
      res.status(429).json({ success: false, error: 'Too many requests. Try again shortly.' })
  });
  const diagnosticsLimiter = rateLimit({
    windowMs: 60_000,
    limit: 4,
    standardHeaders: 'draft-8',
    legacyHeaders: false,
    handler: (_req, res) =>
      res.status(429).json({ success: false, error: 'Diagnostics are temporarily rate-limited.' })
  });
  app.use('/api', apiLimiter);

  app.get('/health', (_req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.json({
      status: 'ok',
      uptimeSeconds: Math.floor(process.uptime()),
      environment: isDevelopment ? 'development' : process.env.NODE_ENV || 'production',
      node: process.version
    });
  });

  app.get('/api/scrape', async (req: Request, res: ExpressResponse) => {
    try {
      const config = getSiteConfig(req.query.site);
      const query = getQueryString(req.query.query, 'search query').trim();
      const page = getPageNumber(req.query.page);
      if (!query || query.length > 120) {
        throw new ApiError(400, 'Search query must be between 1 and 120 characters.');
      }

      const targetUrl = config.buildUrl(query, page);
      const result = await fetchSiteHtml(targetUrl, config, fetcher);
      res.setHeader('Cache-Control', 'private, no-store');
      res.json({ success: true, html: result.html, status: result.status });
    } catch (error) {
      sendApiError(res, error);
    }
  });

  app.get('/api/video-source', async (req: Request, res: ExpressResponse) => {
    try {
      const config = getSiteConfig(req.query.site);
      const videoPageUrl = getQueryString(req.query.url, 'video page URL');
      if (!isAllowedSiteUrl(videoPageUrl, config)) {
        throw new ApiError(400, 'The video page is not on the selected source site.');
      }

      const result = await fetchSiteHtml(videoPageUrl, config, fetcher);
      const url = extractVideoSource(result.html, result.finalUrl);
      res.setHeader('Cache-Control', 'private, no-store');
      res.json({ success: true, url });
    } catch (error) {
      sendApiError(res, error);
    }
  });

  app.use('/api/test-sites', diagnosticsLimiter);
  app.get('/api/test-sites', async (req: Request, res: ExpressResponse) => {
    try {
      const configs = req.query.site === undefined
        ? Object.values(SITES)
        : [getSiteConfig(req.query.site)];
      res.setHeader('Cache-Control', 'private, no-store');
      const sites = await Promise.all(
        configs.map(async config => {
          const startedAt = Date.now();
          try {
            const response = await fetchWithValidatedRedirects(
              config.buildUrl('test', 1),
              candidate => isAllowedSiteUrl(candidate, config),
              fetcher,
              {
                timeoutMs: 6_000,
                headers: {
                  Accept: 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.1',
                  'User-Agent': 'Pyrmsearch/1.0 (public page connectivity check)'
                }
              }
            );
            const status = response.status;
            await response.body?.cancel().catch(() => undefined);
            return {
              id: config.id,
              name: config.name,
              base: config.base,
              icon: config.icon,
              color: config.color,
              status,
              ok: response.ok,
              latencyMs: Date.now() - startedAt,
              ...(!response.ok ? { error: `Source responded with HTTP ${status}.` } : {})
            };
          } catch (error) {
            const apiError = error instanceof ApiError ? error : null;
            return {
              id: config.id,
              name: config.name,
              base: config.base,
              icon: config.icon,
              color: config.color,
              status: 0,
              ok: false,
              latencyMs: Date.now() - startedAt,
              error: apiError?.publicMessage || 'Source site could not be reached.'
            };
          }
        })
      );
      res.json({ sites });
    } catch (error) {
      sendApiError(res, error);
    }
  });

  app.get('/api/proxy-media', async (req: Request, res: ExpressResponse) => {
    try {
      const mediaUrl = getQueryString(req.query.url, 'media URL');
      if (!isAllowedMediaUrl(mediaUrl)) {
        throw new ApiError(
          403,
          'Media proxying is limited to known source domains and configured media hosts.'
        );
      }

      const range = req.get('range');
      const rangeMatch = range?.match(/^bytes=(\d*)-(\d*)$/i);
      if (
        range &&
        (!rangeMatch ||
          (!rangeMatch[1] && !rangeMatch[2]) ||
          rangeMatch[1].length > 18 ||
          rangeMatch[2].length > 18 ||
          (rangeMatch[1] && rangeMatch[2] && BigInt(rangeMatch[1]) > BigInt(rangeMatch[2])))
      ) {
        throw new ApiError(416, 'Only a single valid byte range is supported.');
      }

      const upstream = await fetchWithValidatedRedirects(
        mediaUrl,
        isAllowedMediaUrl,
        fetcher,
        {
          timeoutMs: 15_000,
          headers: {
            Accept: 'video/*,application/octet-stream;q=0.9,*/*;q=0.1',
            ...(range ? { Range: range } : {})
          }
        }
      );
      const contentType = upstream.headers.get('content-type') || '';
      const isVideoContent =
        /^video\//i.test(contentType) ||
        /application\/(?:octet-stream|vnd\.apple\.mpegurl|x-mpegurl)/i.test(contentType);

      if (!isVideoContent && upstream.status !== 416) {
        await upstream.body?.cancel().catch(() => undefined);
        throw new ApiError(415, 'The requested source did not return video media.');
      }

      res.status(upstream.status);
      res.setHeader('Cache-Control', 'private, no-store');
      res.setHeader('X-Content-Type-Options', 'nosniff');
      if (contentType) res.setHeader('Content-Type', contentType);
      for (const header of ['accept-ranges', 'content-range', 'content-length']) {
        const value = upstream.headers.get(header);
        if (value) res.setHeader(header, value);
      }

      if (!upstream.body) {
        res.end();
        return;
      }

      try {
        await pipeline(
          Readable.fromWeb(upstream.body as import('node:stream/web').ReadableStream),
          res
        );
      } catch (error) {
        if (!res.destroyed) {
          logger.warn({ error: error instanceof Error ? error.name : 'stream_error' }, 'media_stream_failed');
          res.end();
        }
      }
    } catch (error) {
      if (!res.headersSent) sendApiError(res, error);
    }
  });

  app.use('/api', (_req, res) => {
    res.status(404).json({ success: false, error: 'API endpoint not found.' });
  });

  return app;
}

export async function startServer(): Promise<import('node:http').Server> {
  const isDevelopment =
    process.argv.includes('--dev') || process.env.NODE_ENV === 'development';
  const app = createApp({ isDevelopment });

  let closeDevServer: (() => Promise<void>) | undefined;
  if (isDevelopment) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      configFile: path.join(projectRoot, 'vite.config.ts'),
      root: projectRoot,
      server: { middlewareMode: true },
      appType: 'custom'
    });
    closeDevServer = () => vite.close();
    app.use(vite.middlewares);
    app.use(async (req: Request, res: ExpressResponse) => {
      try {
        const template = await import('node:fs/promises').then(fs =>
          fs.readFile(path.join(projectRoot, 'index.html'), 'utf8')
        );
        const transformed = await vite.transformIndexHtml(req.originalUrl, template);
        const html = withCspNonce(transformed, res.locals.cspNonce as string);
        res.status(200).type('html').send(html);
      } catch (error) {
        vite.ssrFixStacktrace(error as Error);
        logger.error({ error: error instanceof Error ? error.message : 'vite_error' }, 'dev_html_transform_failed');
        res.status(500).send('Unable to render the development app.');
      }
    });
  } else {
    const distPath = path.join(projectRoot, 'dist');
    if (!existsSync(path.join(distPath, 'index.html'))) {
      throw new Error('Production build not found. Run "npm run build" before "npm start".');
    }
    app.use(express.static(distPath, { index: false, fallthrough: true, maxAge: '1h' }));
    app.get('*', (req: Request, res: ExpressResponse, next) => {
      if (req.path.startsWith('/api/')) return next();
      res.sendFile(path.join(distPath, 'index.html'), error => {
        if (error) next(error);
      });
    });
  }

  app.use((error: unknown, _req: Request, res: ExpressResponse, _next: express.NextFunction) => {
    logger.error(
      { error: error instanceof Error ? error.message : 'unhandled_error' },
      'request_failed'
    );
    if (!res.headersSent) res.status(500).send('Internal server error.');
  });

  const port = Number(process.env.PORT || 3000);
  const host = process.env.HOST || '0.0.0.0';
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error('PORT must be an integer between 1 and 65535.');
  }

  const server = app.listen(port, host, () => {
    logger.info({ port, host, mode: isDevelopment ? 'development' : 'production' }, 'server_started');
  });

  const shutdown = () => {
    void closeDevServer?.().catch(error => {
      logger.error({ error: error instanceof Error ? error.message : 'vite_shutdown_error' }, 'dev_server_shutdown_failed');
    });
    server.close(error => {
      if (error) logger.error({ error: error.message }, 'server_shutdown_failed');
      else logger.info('server_stopped');
    });
  };
  process.once('SIGINT', shutdown);
  process.once('SIGTERM', shutdown);

  return server;
}

const entryPoint = process.argv[1] ? path.resolve(process.argv[1]) : '';
if (entryPoint === fileURLToPath(import.meta.url)) {
  startServer().catch(error => {
    logger.error({ error: error instanceof Error ? error.message : 'startup_error' }, 'server_start_failed');
    process.exitCode = 1;
  });
}
