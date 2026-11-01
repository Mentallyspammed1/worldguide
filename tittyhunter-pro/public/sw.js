const CACHE_PREFIX = 'pyrmsearch-shell-';
const CACHE_NAME = `${CACHE_PREFIX}v1`;

self.addEventListener('install', event => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches
      .keys()
      .then(names =>
        Promise.all(
          names
            .filter(name => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
            .map(name => caches.delete(name))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const requestUrl = new URL(request.url);
  if (requestUrl.origin !== self.location.origin || requestUrl.pathname.startsWith('/api/')) {
    return;
  }

  const isNavigation = request.mode === 'navigate';
  const isBuiltAsset = /^\/assets\/.*\.(?:js|css|svg|png|webp|woff2?)$/i.test(requestUrl.pathname);
  const isAppIcon = /^\/(?:icon\.svg|apple-touch-icon\.png|pwa-(?:192x192|512x512|maskable-512x512)\.png)$/i.test(requestUrl.pathname);
  if (!isNavigation && !isBuiltAsset && !isAppIcon && requestUrl.pathname !== '/manifest.json') return;

  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      try {
        const response = await fetch(request);
        if (response.ok && response.type === 'basic') {
          await cache.put(request, response.clone());
        }
        return response;
      } catch {
        const cached = await cache.match(isNavigation ? '/' : request);
        return cached || Response.error();
      }
    })()
  );
});
