# TittyHunter Pro

**Neon Multi-Site Hunter** — A React 19 + TypeScript + Express/Vite multi-site media scraping search engine for adult content sites, optimized for Termux/Android.

> ⚠️ **Disclaimer:** This project is for educational and personal use only. Respect the terms of service of target websites and applicable laws in your jurisdiction. The authors are not responsible for misuse.

---

## Features

- **Multi-site search** across 13 adult content platforms:
  - XVideos, XNXX, Eporner, XHamster, Pornhub, RedTube, YouPorn
  - OK.XXX, Hello.Porn, Users.Porn, PornTrex, HQPorner, SpankBang
- **Live hover video previews** on search results (touch + mouse support)
- **Full video playback in teaser modal** — fetch and play the entire video from the source page (`PLAY FULL VIDEO` / `PREVIEW CLIP` toggle)
- **IntersectionObserver preloading** — videos preload as cards enter the viewport
- **Save video screenshots** — capture a frame from any video preview and save it to localStorage (`📷` icon on each card)
- **Saved Screenshots gallery** — view all captured screenshots in a dedicated tab with delete and bulk actions
- **Favorites system** (persisted via `localStorage`)
- **Site Diagnostics** tab for connectivity/latency testing per site
- **CORS proxy** for server-side HTML fetching (`/api/scrape`)
- **Media stream proxy** with `Range` support (`/api/proxy-media`)
- **PWA support** (installable as an app)
- **Rate limiting** and structured `pino` logging
- **Express + Vite middleware mode** (dev) / static file serving (prod)
- **Background server management** via npm scripts (`bg`, `stop`, `status`, `logs`, `restart`)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, TypeScript, Tailwind CSS v4, Lucide React, Motion (Framer Motion) |
| Backend | Express, Node.js, tsx |
| Build Tool | Vite 6, Vite PWA Plugin |
| Logging | Pino + Pino Pretty |
| Rate Limiting | express-rate-limit |
| Security | Helmet |
| Environment | Termux (Android), Node v26.4.0, npm 11.14.0 |

---

## Project Structure

```
tittyhunter-pro/
├── server.ts              # Express backend entry point
├── package.json           # Dependencies & scripts
├── tsconfig.json          # TypeScript config
├── vite.config.ts         # Vite build config
├── tailwind.config.js     # Tailwind CSS v4 config
├── .env.local             # Environment variables (create this)
├── server.log             # Background server logs
├── server.pid             # Background server PID file
├── public/
│   ├── pyrmsearch.apk     # Android APK download
│   ├── manifest.json      # PWA manifest
│   └── icons/             # PWA icons
├── src/
│   ├── main.tsx           # React entry point
│   ├── App.tsx            # Main app (Hunt / Favorites / Diagnostics tabs)
│   ├── constants.ts       # Site configs & CSS selectors
│   ├── index.css          # Tailwind imports & custom styles
│   ├── usePWAInstall.ts   # PWA install prompt hook
│   └── lib/
│       └── scraper.ts     # HTML parsing + proxy fallback logic
└── dist/                  # Production build output (generated)
```

---

## Installation

### Prerequisites

- Termux (Android) or any Node.js environment
- Node.js v20+ (tested on v26.4.0)
- npm v11+

### Setup

```bash
# Clone / navigate to the project
cd /data/data/com.termux/files/home/nsfw/Termux_Monkey_Projects/tittyhunter-pro

# Install dependencies (use --legacy-peer-deps on Termux/Android)
npm install --legacy-peer-deps --no-audit --no-fund

# Create environment file
cp .env.example .env.local   # if .env.example exists, otherwise create manually
```

### Environment Variables (`.env.local`)

```env
PORT=3000
NODE_ENV=development
LOG_LEVEL=info
```

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | No | Server port (default: `3000`) |
| `NODE_ENV` | No | `development` or `production` (default: `development`) |
| `LOG_LEVEL` | No | Pino log level (default: `info`) |

---

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start server in foreground with `tsx` |
| `npm run build` | Build frontend for production with Vite |
| `npm run start` | Start compiled server with `node server.ts` |
| `npm run preview` | Preview production build with Vite |
| `npm run clean` | Remove `dist/` directory |
| `npm run lint` | Type-check with `tsc --noEmit` |
| `npm run bg` | Start server in background (`nohup`, logs to `server.log`) |
| `npm run stop` | Stop background server |
| `npm run status` | Check if background server is running |
| `npm run logs` | Follow background server logs |
| `npm run restart` | Stop then start background server |

### Recommended Workflow (Termux)

```bash
# Start in background
npm run bg

# Check status
npm run status

# View logs
npm run logs

# Stop when done
npm run stop
```

---

## API Endpoints

### `GET /health`
Health check. Returns server status, uptime, port, and environment.

### `GET /api/test-sites`
Tests reachability and latency for all configured scraping sites.

### `GET /api/scrape`
Server-side HTML proxy to bypass CORS when scraping target sites.

**Query Parameters:**
- `site` — Target site key (e.g., `xvideos`, `pornhub`)
- `query` — Search query string
- `page` — Page number (default: `1`)

**Example:**
```bash
curl "http://localhost:3000/api/scrape?site=xvideos&query=test&page=1"
```

### `GET /api/proxy-media`
Proxies video media streams with `Referer`/`Origin` headers and `Range` request support.

**Query Parameters:**
- `url` — Direct media URL to proxy

**Example:**
```bash
curl "http://localhost:3000/api/proxy-media?url=https://example.com/video.mp4"
```

### `GET /pyrmsearch.apk`
Serves the Android APK file from `public/`.

---

## Frontend Tabs

### 1. Hunt
Search across all configured sites with live hover video previews. Results are parsed from target site HTML and displayed in a responsive grid. Videos preload as cards enter the viewport, and previews work on both mouse hover and touch.

Open any result in the **teaser modal** to watch the preview clip, or tap **PLAY FULL VIDEO** to fetch and play the entire video directly from the source site (when a playable video URL can be extracted). Toggle between `PREVIEW CLIP` and `PLAY FULL VIDEO` at any time.

### 2. Favorites
Bookmark favorite videos. Persisted in browser `localStorage`.

### 3. Site Diagnostics
Ping each configured site to check connectivity and measure response latency. Useful for debugging which sites are reachable from your network.

### 4. Saved Screenshots
View all captured video screenshots in a grid. Each screenshot shows the source site, title, and capture timestamp. You can delete individual screenshots or clear all at once. Screenshot data URLs can be copied to clipboard in bulk.

---

## Background Server Management

The `bg` script uses `nohup` to detach the process and writes the PID to `server.pid`:

```bash
npm run bg      # starts: nohup ./node_modules/.bin/tsx server.ts > server.log 2>&1 &
npm run status  # checks if PID is alive
npm run logs    # tail -f server.log
npm run stop    # kills the process and removes server.pid
npm run restart # stop + start
```

> **Note:** On Termux, always use `./node_modules/.bin/tsx` directly (not `npx tsx`) for background processes, as `npx` can spawn a short-lived wrapper that exits before the server binds.

---

## Troubleshooting

### Port 3000 already in use
```bash
npm run stop
# or manually kill:
pkill -f 'tsx server.ts'
```

### TypeScript errors
```bash
npm run lint
```

### Missing optional dependency (`@rollup/rollup-android-arm64`)
If `npm install` fails on Termux due to missing optional dependencies:

```bash
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps --no-audit --no-fund
```

### Server starts but port never listens (backgrounded)
Ensure you're using `npm run bg` (which uses `./node_modules/.bin/tsx` directly) rather than `npx tsx server.ts` in the background.

---

## License

MIT. Use responsibly.

---

## Credits

Built with [React](https://react.dev), [Vite](https://vitejs.dev), [Express](https://expressjs.com), [Tailwind CSS](https://tailwindcss.com), and [Pino](https://getpino.io).
