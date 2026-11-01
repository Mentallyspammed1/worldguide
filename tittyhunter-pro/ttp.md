# Pyrmsearch project notes

The user-facing setup, API, privacy, and operations guide is maintained in [`README.md`](README.md). This file replaces obsolete scratch notes that described deleted proxy behavior, a bundled APK, and stale build errors.

## Architecture

- `src/App.tsx` contains the React interface and browser-local interactions.
- `src/constants.ts` owns the 13 configured public source adapters and their selectors.
- `src/lib/scraper.ts` parses bounded listing HTML and calls only the same-origin API.
- `src/lib/safe-url.ts` provides shared URL checks for public HTTPS destinations.
- `src/lib/backup.ts` validates and migrates browser backup data.
- `server.ts` serves the API and, in development, Vite middleware; production serves `dist/`.
- `public/sw.js` caches the app shell/static build assets only. It does not cache searches, page HTML, or media.

## Development and verification

Run `npm ci`, then `npm run dev`. Before shipping, run `npm run check`; it type-checks, runs tests, and builds the production frontend. `npm audit` checks the complete dependency tree. For a smaller runtime install, build first and then run `npm ci --omit=dev`; the production server runner remains a runtime dependency.

Search and diagnostics depend on third-party sites being reachable and retaining compatible public-page markup. A connectivity result means only that an HTTP response was received; it does not guarantee that parsing or playback works. Respect source-site terms and applicable law. Pyrmsearch is for adults and does not host content or bypass access controls.
