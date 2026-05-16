const CACHE_NAME = 'atlas-0-shell-v2';

const STATIC_SHELL = [
  '/app/',
  '/app/index.html',
  '/app/manifest.webmanifest',
  '/app/atlas-icon.svg',
  '/app/js/api.js',
  '/app/js/app.js',
  '/app/js/intelligence.js',
  '/app/js/overlay.js',
  '/app/js/product_playbooks.js',
  '/app/js/scene_viewer.js',
  '/app/js/upload.js',
];

const PUBLIC_METADATA = new Set([
  '/operator/access',
  '/product/privacy',
  '/product/upload-guidance',
  '/product/trust-proof',
]);

const PRIVATE_PREFIXES = [
  '/upload',
  '/jobs',
  '/reports',
  '/operator/settings',
  '/operator/storage',
  '/product/events',
  '/product/waitlist',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys
        .filter((key) => key !== CACHE_NAME)
        .map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);

  if (request.method !== 'GET' || url.origin !== self.location.origin) {
    return;
  }

  if (PRIVATE_PREFIXES.some((prefix) => url.pathname.startsWith(prefix))) {
    return;
  }

  if (STATIC_SHELL.includes(url.pathname)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  if (PUBLIC_METADATA.has(url.pathname)) {
    event.respondWith(networkFirst(request));
  }
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }
  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(CACHE_NAME);
    await cache.put(request, response.clone());
  }
  return response;
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(request);
    if (response.ok && !request.headers.has('authorization')) {
      await cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }
    throw new Error('Offline and no cached public metadata available.');
  }
}
