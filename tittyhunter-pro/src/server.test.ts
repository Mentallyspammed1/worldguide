// @vitest-environment node
import { once } from 'node:events';
import type { AddressInfo } from 'node:net';
import type { Server } from 'node:http';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createApp } from '../server';

const servers: Server[] = [];

async function withServer(fetcher: typeof fetch, run: (baseUrl: string) => Promise<void>) {
  const server = createApp({ fetcher }).listen(0, '127.0.0.1');
  servers.push(server);
  await once(server, 'listening');
  const address = server.address() as AddressInfo;
  try {
    await run(`http://127.0.0.1:${address.port}`);
  } finally {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    servers.splice(servers.indexOf(server), 1);
  }
}

afterEach(() => {
  for (const server of servers) server.closeAllConnections();
  vi.restoreAllMocks();
});

describe('Express API', () => {
  it('serves a health check and basic security headers', async () => {
    await withServer(fetch, async baseUrl => {
      const response = await fetch(`${baseUrl}/health`);
      const payload = await response.json();
      expect(response.status).toBe(200);
      expect(payload.status).toBe('ok');
      expect(response.headers.get('x-powered-by')).toBeNull();
      expect(response.headers.get('content-security-policy')).toContain("default-src 'self'");
    });
  });

  it('validates source and query values before making an upstream request', async () => {
    const fetcher = vi.fn(async () => new Response('unexpected')) as unknown as typeof fetch;
    await withServer(fetcher, async baseUrl => {
      const unsupported = await fetch(`${baseUrl}/api/scrape?site=unknown&query=test`);
      expect(unsupported.status).toBe(400);
      const invalidPage = await fetch(`${baseUrl}/api/scrape?site=xvideos&query=test&page=0`);
      expect(invalidPage.status).toBe(400);
      const longQuery = await fetch(`${baseUrl}/api/scrape?site=xvideos&query=${'a'.repeat(121)}`);
      expect(longQuery.status).toBe(400);
      expect(fetcher).not.toHaveBeenCalled();
    });
  });

  it('fetches search HTML only from the selected configured source', async () => {
    const fetcher = vi.fn(async () => new Response('<html>listing</html>', {
      headers: { 'content-type': 'text/html; charset=utf-8' }
    })) as unknown as typeof fetch;

    await withServer(fetcher, async baseUrl => {
      const response = await fetch(`${baseUrl}/api/scrape?site=xvideos&query=public%20listing&page=1`);
      const payload = await response.json();
      expect(response.status).toBe(200);
      expect(payload).toMatchObject({ success: true, html: '<html>listing</html>', status: 200 });
      expect(String(vi.mocked(fetcher).mock.calls[0]?.[0])).toContain('www.xvideos.com');
    });
  });

  it('extracts only allow-listed HTTPS media from a selected source page', async () => {
    const fetcher = vi.fn(async () => new Response(`
      <meta property="og:video" content="https://cdn.xvideos.com/clip.mp4?token=sample">
      <source src="https://untrusted.example/other.mp4">
    `, { headers: { 'content-type': 'text/html' } })) as unknown as typeof fetch;

    await withServer(fetcher, async baseUrl => {
      const pageUrl = 'https://www.xvideos.com/video123';
      const response = await fetch(`${baseUrl}/api/video-source?site=xvideos&url=${encodeURIComponent(pageUrl)}`);
      const payload = await response.json();
      expect(response.status).toBe(200);
      expect(payload).toEqual({ success: true, url: 'https://cdn.xvideos.com/clip.mp4?token=sample' });
    });
  });

  it('revalidates every upstream redirect before following it', async () => {
    const fetcher = vi.fn(async () => new Response(null, {
      status: 302,
      headers: { location: 'https://127.0.0.1/private' }
    })) as unknown as typeof fetch;

    await withServer(fetcher, async baseUrl => {
      const response = await fetch(`${baseUrl}/api/scrape?site=xvideos&query=public`);
      expect(response.status).toBe(400);
      expect(fetcher).toHaveBeenCalledTimes(1);
    });
  });

  it('blocks private or unapproved hosts in the page and media proxies', async () => {
    const fetcher = vi.fn(async () => new Response('unexpected')) as unknown as typeof fetch;
    await withServer(fetcher, async baseUrl => {
      const page = await fetch(`${baseUrl}/api/video-source?site=xvideos&url=${encodeURIComponent('http://127.0.0.1/admin')}`);
      expect(page.status).toBe(400);
      for (const value of [
        'http://169.254.169.254/latest/meta-data',
        'https://127.0.0.1/video.mp4',
        'https://[::1]/video.mp4',
        'https://2130706433/video.mp4'
      ]) {
        const media = await fetch(`${baseUrl}/api/proxy-media?url=${encodeURIComponent(value)}`);
        expect(media.status).toBe(403);
      }
      expect(fetcher).not.toHaveBeenCalled();
    });
  });

  it('rejects malformed, reversed, and multiple byte ranges', async () => {
    const fetcher = vi.fn(async () => new Response('unexpected')) as unknown as typeof fetch;
    await withServer(fetcher, async baseUrl => {
      for (const range of ['bytes=-', 'bytes=4-2', 'bytes=0-1,2-3']) {
        const response = await fetch(`${baseUrl}/api/proxy-media?url=${encodeURIComponent('https://cdn.xvideos.com/video.mp4')}`, {
          headers: { Range: range }
        });
        expect(response.status).toBe(416);
      }
      expect(fetcher).not.toHaveBeenCalled();
    });
  });

  it('supports a single validated range request for approved media', async () => {
    const fetcher = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(new Headers(init?.headers).get('range')).toBe('bytes=0-3');
      return new Response('clip', {
        status: 206,
        headers: {
          'content-type': 'video/mp4',
          'content-range': 'bytes 0-3/4',
          'accept-ranges': 'bytes'
        }
      });
    }) as unknown as typeof fetch;

    await withServer(fetcher, async baseUrl => {
      const response = await fetch(`${baseUrl}/api/proxy-media?url=${encodeURIComponent('https://cdn.xvideos.com/video.mp4')}`, {
        headers: { Range: 'bytes=0-3' }
      });
      expect(response.status).toBe(206);
      expect(response.headers.get('content-type')).toBe('video/mp4');
      expect(await response.text()).toBe('clip');
    });
  });

  it('limits diagnostics to a single configured site when requested', async () => {
    const fetcher = vi.fn(async () => new Response('', { status: 200 })) as unknown as typeof fetch;
    await withServer(fetcher, async baseUrl => {
      const response = await fetch(`${baseUrl}/api/test-sites?site=xvideos`);
      const payload = await response.json();
      expect(response.status).toBe(200);
      expect(payload.sites).toHaveLength(1);
      expect(payload.sites[0]).toMatchObject({ id: 'xvideos', ok: true });
      expect(fetcher).toHaveBeenCalledTimes(1);
    });
  });
});
