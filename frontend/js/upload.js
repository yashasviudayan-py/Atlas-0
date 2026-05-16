/**
 * upload.js — upload-first interaction layer for the report workflow.
 */

import * as api from './api.js';

const POLL_MS = 1400;
const OFFLINE_DB_NAME = 'atlas0-offline-uploads';
const OFFLINE_DB_VERSION = 1;
const OFFLINE_STORE = 'queuedUploads';

export class UploadView {
  /**
   * @param {{
   *   dropZone: HTMLElement,
   *   fileInput: HTMLInputElement,
   *   roomLabelInput?: HTMLInputElement | null,
   *   audienceModeInput?: HTMLSelectElement | null,
   *   uploadGuidance?: any,
   *   onJobCreated?: (job: any) => void,
   *   onJobUpdate?: (job: any) => void,
   *   onJobError?: (error: Error) => void,
   *   onUploadStart?: (file: File, metadata: { roomLabel: string, audienceMode: string }) => void,
   *   onOfflineQueued?: (entry: any) => void,
   *   onOfflineReplayStart?: (entry: any) => void,
   *   onOfflineQueueChange?: (entries: any[]) => void,
   *   onPreflightFailed?: (file: File | null, error: Error) => void,
   * }} opts
   */
  constructor(opts) {
    this._dropZone = opts.dropZone;
    this._fileInput = opts.fileInput;
    this._roomLabelInput = opts.roomLabelInput || null;
    this._audienceModeInput = opts.audienceModeInput || null;
    this._uploadGuidance = opts.uploadGuidance || null;
    this._onJobCreated = opts.onJobCreated || (() => {});
    this._onJobUpdate = opts.onJobUpdate || (() => {});
    this._onJobError = opts.onJobError || (() => {});
    this._onUploadStart = opts.onUploadStart || (() => {});
    this._onOfflineQueued = opts.onOfflineQueued || (() => {});
    this._onOfflineReplayStart = opts.onOfflineReplayStart || (() => {});
    this._onOfflineQueueChange = opts.onOfflineQueueChange || (() => {});
    this._onPreflightFailed = opts.onPreflightFailed || (() => {});
    this._pollers = new Map();
    this._retryingOffline = false;
  }

  init() {
    this._dropZone.addEventListener('click', () => this._fileInput.click());
    this._dropZone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        this._fileInput.click();
      }
    });

    this._fileInput.addEventListener('change', (event) => {
      const input = /** @type {HTMLInputElement} */ (event.currentTarget);
      Array.from(input.files || []).forEach((file) => this._handle(file));
      input.value = '';
    });

    this._dropZone.addEventListener('dragover', (event) => {
      event.preventDefault();
      this._dropZone.classList.add('drag-over');
    });

    this._dropZone.addEventListener('dragleave', (event) => {
      const relatedTarget = /** @type {Node|null} */ (event.relatedTarget);
      if (!relatedTarget || !this._dropZone.contains(relatedTarget)) {
        this._dropZone.classList.remove('drag-over');
      }
    });

    this._dropZone.addEventListener('drop', (event) => {
      event.preventDefault();
      this._dropZone.classList.remove('drag-over');
      Array.from(event.dataTransfer?.files || []).forEach((file) => this._handle(file));
    });

    window.addEventListener('online', () => this.retryQueuedUploads());
    this.refreshOfflineQueue();
  }

  setGuidance(guidance) {
    this._uploadGuidance = guidance || null;
  }

  async _handle(file) {
    let validated = false;
    try {
      this._validateFile(file);
      validated = true;
      const roomLabel = this._roomLabelInput?.value?.trim() || '';
      const audienceMode = this._audienceModeInput?.value?.trim() || 'general';
      if (navigator.onLine === false) {
        const entry = await queueOfflineUpload(file, { roomLabel, audienceMode });
        this._onOfflineQueued(entry);
        await this.refreshOfflineQueue();
        return;
      }
      this._onUploadStart(file, { roomLabel, audienceMode });
      const job = await api.uploadFile(file, { roomLabel, audienceMode });
      await this._onJobCreated(job);
      await this._onJobUpdate(job);
      this._poll(job.job_id);
    } catch (error) {
      const normalized = error instanceof Error ? error : new Error(String(error));
      if (!validated) {
        this._onPreflightFailed(file || null, normalized);
      }
      this._onJobError(normalized);
    }
  }

  _validateFile(file) {
    if (!file || file.size <= 0) {
      throw new Error('Choose a non-empty image or walkthrough video to start a scan.');
    }

    const guidance = this._uploadGuidance || {};
    const acceptedPrefixes = Array.isArray(guidance.accepted_media_prefixes)
      ? guidance.accepted_media_prefixes
      : ['video/', 'image/'];
    const acceptedExtensions = Array.isArray(guidance.accepted_extensions)
      ? guidance.accepted_extensions
      : ['.mp4', '.mov', '.webm', '.jpg', '.jpeg', '.png'];
    const lowerName = String(file.name || '').toLowerCase();
    const acceptedType = acceptedPrefixes.some((prefix) => file.type.startsWith(prefix))
      || acceptedExtensions.some((extension) => lowerName.endsWith(extension));
    if (!acceptedType) {
      throw new Error('Choose an image or a walkthrough video such as MP4, MOV, or WEBM.');
    }

    const maxBytes = Number(guidance.max_upload_bytes || 0);
    if (maxBytes > 0 && file.size > maxBytes) {
      throw new Error(`This file is ${formatBytes(file.size)}. The hosted upload limit is ${formatBytes(maxBytes)}.`);
    }
  }

  _poll(jobId) {
    const existing = this._pollers.get(jobId);
    if (existing) {
      clearInterval(existing);
    }

    const timer = setInterval(async () => {
      try {
        const job = await api.fetchJob(jobId);
        this._onJobUpdate(job);
        if (job.status === 'complete' || job.status === 'error') {
          clearInterval(timer);
          this._pollers.delete(jobId);
        }
      } catch (error) {
        clearInterval(timer);
        this._pollers.delete(jobId);
        this._onJobError(
          error instanceof Error ? error : new Error(String(error)),
        );
      }
    }, POLL_MS);

    this._pollers.set(jobId, timer);
  }

  async refreshOfflineQueue() {
    try {
      this._onOfflineQueueChange(await listQueuedUploads());
    } catch {
      this._onOfflineQueueChange([]);
    }
  }

  async retryQueuedUploads() {
    if (this._retryingOffline || navigator.onLine === false) {
      return;
    }
    this._retryingOffline = true;
    try {
      const entries = await listQueuedUploads();
      for (const entry of entries) {
        if (!entry?.file) {
          await deleteQueuedUpload(entry.id);
          continue;
        }
        try {
          this._onOfflineReplayStart(entry);
          this._onUploadStart(entry.file, {
            roomLabel: entry.roomLabel || '',
            audienceMode: entry.audienceMode || 'general',
          });
          const job = await api.uploadFile(entry.file, {
            roomLabel: entry.roomLabel || '',
            audienceMode: entry.audienceMode || 'general',
          });
          await deleteQueuedUpload(entry.id);
          await this._onJobCreated(job);
          await this._onJobUpdate(job);
          this._poll(job.job_id);
        } catch (error) {
          this._onJobError(error instanceof Error ? error : new Error(String(error)));
          break;
        } finally {
          await this.refreshOfflineQueue();
        }
      }
    } finally {
      this._retryingOffline = false;
    }
  }
}

function openOfflineDb() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) {
      reject(new Error('Offline retry storage is unavailable in this browser.'));
      return;
    }
    const request = window.indexedDB.open(OFFLINE_DB_NAME, OFFLINE_DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(OFFLINE_STORE)) {
        db.createObjectStore(OFFLINE_STORE, { keyPath: 'id' });
      }
    };
    request.onerror = () => reject(request.error || new Error('Could not open offline upload queue.'));
    request.onsuccess = () => resolve(request.result);
  });
}

async function withOfflineStore(mode, callback) {
  const db = await openOfflineDb();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction(OFFLINE_STORE, mode);
      const store = transaction.objectStore(OFFLINE_STORE);
      const result = callback(store);
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () => reject(transaction.error || new Error('Offline queue operation failed.'));
      transaction.onabort = () => reject(transaction.error || new Error('Offline queue operation aborted.'));
    });
  } finally {
    db.close();
  }
}

async function queueOfflineUpload(file, metadata) {
  const entry = {
    id: `offline_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 9)}`,
    file,
    filename: file.name || 'room-scan-upload',
    fileType: file.type || 'application/octet-stream',
    fileSize: file.size || 0,
    roomLabel: metadata.roomLabel || '',
    audienceMode: metadata.audienceMode || 'general',
    queuedAt: new Date().toISOString(),
  };
  await withOfflineStore('readwrite', (store) => store.put(entry));
  return entry;
}

async function listQueuedUploads() {
  const entries = await withOfflineStore('readonly', (store) => {
    const request = store.getAll();
    return new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error || new Error('Could not read offline upload queue.'));
      request.onsuccess = () => resolve(request.result || []);
    });
  });
  return entries.sort((a, b) => String(a.queuedAt || '').localeCompare(String(b.queuedAt || '')));
}

async function deleteQueuedUpload(id) {
  await withOfflineStore('readwrite', (store) => store.delete(id));
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
