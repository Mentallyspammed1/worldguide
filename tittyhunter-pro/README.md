# Pyrmsearch

A local-first progressive web app for searching public listings on adult-content platforms. **18+ only.** The app shows links and metadata; it does not host source content or bypass logins, paywalls, age checks, or other access controls.

> Use only public pages. Follow the laws where you live and each source site's terms. Availability and page layouts change; a configured site may block automated requests or return no results.

## Highlights

- Search one configured source at a time across 13 site adapters.
- Parse results into responsive cards with optional hover previews.
- Save favorites, recent items, collections, and compressed screenshots in browser storage.
- Import/export a validated JSON backup.
- Run per-source or all-source connectivity checks.
- Install as a PWA; the service worker caches only the app shell and static assets, never search responses or remote media.
- Same-origin Express API with rate limits, security headers, bounded HTML responses, and allow-listed upstream hosts.

## Requirements and setup

- Node.js 20 or newer
- npm 10 or newer

```sh
cd tittyhunter-pro
npm ci
cp .env.example .env
npm run dev
```

The development server listens on `0.0.0.0:3000` by default and serves both the React app and API. Vite HMR is served through the same HTTP port, so proxied previews only need one port exposed. Set `PORT` to use another port. Do not commit `.env` files.

### Production

```sh
npm ci
npm run build
npm ci --omit=dev  # optional: keep only runtime packages after building
npm start
```

`npm start` serves the generated `dist/` build and API from Express. Run `npm run build` first; the server now fails fast with a clear message if `dist/index.html` is missing. The server runner (`tsx`) is a runtime dependency; Vite, React, and test/build tooling are development dependencies. `npm run preview` is an alias for this full production server; it is not Vite's static-only preview server.

### Embedded previews

Production defaults to `frame-ancestors 'self'`, so browsers will block it inside a cross-origin iframe. Development keeps local previews embeddable unless `FRAME_ANCESTORS` is set. For a production Arena live preview, start the server with the Arena parent origins explicitly allowed:

```sh
FRAME_ANCESTORS="'self' https://arena.ai https://*.arena.ai" npm start
```

You can use the same setting with `npm run dev` to restrict development previews. Do not use `*` unless you intentionally want any site to embed the app.

### Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start Express with Vite middleware and hot reload |
| `npm run build` | Type-check and build the production frontend |
| `npm run start` | Serve the production build and API |
| `npm run typecheck` | Run TypeScript checks |
| `npm test` | Run unit and API tests |
| `npm run check` | Run type-check, tests, and build |
| `npm run bg` / `npm run stop` | Start/stop a detached production server |
| `npm run status` / `npm run logs` | Inspect server state/logs |
| `npm run restart` | Restart the detached production server |

The background scripts track only their own PID; `server.pid` and `server.log` are ignored by Git.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `3000` | HTTP listening port |
| `HOST` | `0.0.0.0` | Listening interface |
| `NODE_ENV` | production when started normally | Use `development` for development behavior |
| `LOG_LEVEL` | `info` | Pino log level |
| `FRAME_ANCESTORS` | `'self'` | Space-separated CSP `frame-ancestors` sources; add trusted preview parent origins when embedding |
| `MEDIA_HOST_ALLOWLIST` | empty | Optional comma-separated HTTPS media CDN hostnames, without schemes or paths |

The media proxy already permits configured source-site domains. Add only CDN hosts you trust. Redirects are revalidated against the same allow-list. Do not use this setting to allow private or internal hosts.

## API

All endpoints are same-origin. No third-party CORS relay is used, so search terms are not sent to public proxy services. Requests to a search source are made by this server on the user's behalf.

- `GET /health` — server health and runtime metadata.
- `GET /api/scrape?site=xvideos&query=example&page=1` — fetches a public listing page from a configured source. `site` must be a known key, `query` is limited to 120 characters, and `page` to 1–100.
- `GET /api/test-sites` — checks all configured sources. Add `?site=xvideos` to check just one.
- `GET /api/video-source?site=xvideos&url=https%3A%2F%2F...` — inspects a public video page on the selected site for a supported direct media URL.
- `GET /api/proxy-media?url=https%3A%2F%2F...` — streams media only from configured site domains or `MEDIA_HOST_ALLOWLIST` hosts. Supports a single byte range for seeking.

API requests are rate-limited. Upstream HTML is capped at 2 MiB, redirect targets are checked, and arbitrary URLs cannot be fetched through the scraper routes. Source-site failures are returned as errors rather than relayed through external services.

## Local data and privacy

Favorites, search history, recent items, collections, and screenshots are stored in the current browser's `localStorage`. They are not uploaded by the app, but anyone with access to that browser profile can see them. Search queries are sent to the selected source through this server; server request logs deliberately omit query strings. Use **Backup** to export or move data, and clear browser storage when appropriate. Backups may contain sensitive browsing history and screenshots—store them carefully.

The PWA can reopen its interface offline after it has been loaded once, but searches and media require a network connection. Remote thumbnails and video previews are requested from their source hosts. Browser cross-origin restrictions may prevent screenshot capture for some streams.

## Sources and maintenance

Source metadata, URL templates, and selectors live in `src/constants.ts`. Source sites change frequently, so diagnostics indicate basic reachability only—not that selectors are still current or that results will be playable. The parser ignores unsafe link schemes and only keeps result links on the selected source's domain.

## License

MIT. Use responsibly.
