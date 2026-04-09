/**
 * api.js — Atlas-0 REST API client (relative URLs, same origin).
 */

async function json(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${body.slice(0, 120)}`);
  }
  return res.json();
}

export const fetchHealth  = ()         => json('/health');
export const fetchObjects = ()         => json('/objects');
export const fetchScene   = ()         => json('/scene');
export const fetchJob     = (id)       => json(`/jobs/${id}`);
export const fetchJobs    = ()         => json('/jobs');
export const reportPdfUrl = (id)      => `/reports/${id}.pdf`;

export function postQuery(query, maxResults = 5) {
  return json('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, max_results: maxResults }),
  });
}

export async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/upload', { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Upload failed ${res.status}: ${body.slice(0, 120)}`);
  }
  return res.json();
}
