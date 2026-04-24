/**
 * api.js — Atlas-0 REST API client (relative URLs, same origin).
 */

const ACCESS_TOKEN_KEY = 'atlas0.accessToken';

function accessToken() {
  return window.localStorage.getItem(ACCESS_TOKEN_KEY) || '';
}

function authHeaders(headers = {}) {
  const token = accessToken().trim();
  return token
    ? { ...headers, Authorization: `Bearer ${token}` }
    : headers;
}

async function errorMessage(res, fallback) {
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string' && body.detail.trim()) {
      return body.detail.trim();
    }
  } catch {}

  const text = await res.text().catch(() => '');
  return `${fallback}: ${(text || res.statusText || 'Request failed').slice(0, 160)}`;
}

async function json(url, opts = {}) {
  const res = await fetch(url, {
    ...opts,
    headers: authHeaders(opts.headers || {}),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, `HTTP ${res.status}`));
  }
  return res.json();
}

export function getAccessToken() {
  return accessToken();
}

export function setAccessToken(token) {
  window.localStorage.setItem(ACCESS_TOKEN_KEY, String(token || '').trim());
}

export function clearAccessToken() {
  window.localStorage.removeItem(ACCESS_TOKEN_KEY);
}

export function withAccessToken(url) {
  const token = accessToken().trim();
  if (!token) {
    return url;
  }
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}access_token=${encodeURIComponent(token)}`;
}

export const fetchHealth  = ()         => json('/health');
export const fetchObjects = ()         => json('/objects');
export const fetchScene   = ()         => json('/scene');
export const fetchJob     = (id)       => json(`/jobs/${id}`);
export const fetchJobs    = ()         => json('/jobs');
export const fetchSampleReport = ()    => json('/sample-report');
export const fetchAccessPolicy = ()    => json('/operator/access');
export const fetchOperatorSettings = () => json('/operator/settings');
export const pruneOperatorStorage = () => json('/operator/storage/prune', {
  method: 'POST',
});
export const fetchPrivacyPolicy = ()   => json('/product/privacy');
export const fetchUploadGuidance = ()   => json('/product/upload-guidance');
export const submitWaitlist = (payload) => json('/product/waitlist', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
export const reportPdfUrl = (id)      => withAccessToken(`/reports/${id}.pdf`);
export const sampleReportPdfUrl = ()  => '/sample-report/report.pdf';
export async function logProductEvent(payload) {
  const res = await fetch('/product/events', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, `Product event failed ${res.status}`));
  }
}
export const submitFindingFeedback = (jobId, payload) => json(`/jobs/${jobId}/feedback`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
export const submitFindingFollowUp = (jobId, payload) => json(`/jobs/${jobId}/follow-up`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
export const submitJobEvaluation = (jobId, payload) => json(`/jobs/${jobId}/evaluation`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
export const exportEvalCandidate = (jobId, payload) => json(`/jobs/${jobId}/eval-candidate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});

export function postQuery(query, maxResults = 5) {
  return json('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, max_results: maxResults }),
  });
}

export async function uploadFile(file, options = {}) {
  const form = new FormData();
  form.append('file', file);
  const roomLabel = String(options.roomLabel || '').trim();
  const audienceMode = String(options.audienceMode || '').trim();
  const headers = {};
  if (roomLabel) {
    headers['X-Room-Label'] = roomLabel;
  }
  if (audienceMode) {
    headers['X-Audience-Mode'] = audienceMode;
  }
  const res = await fetch('/upload', {
    method: 'POST',
    body: form,
    headers: authHeaders(headers),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, `Upload failed ${res.status}`));
  }
  return res.json();
}

export async function deleteJob(id) {
  const res = await fetch(`/jobs/${id}`, {
    method: 'DELETE',
    headers: authHeaders(),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, `Delete failed ${res.status}`));
  }
}
