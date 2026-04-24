/**
 * upload.js — upload-first interaction layer for the report workflow.
 */

import * as api from './api.js';

const POLL_MS = 1400;

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
    this._pollers = new Map();
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
  }

  setGuidance(guidance) {
    this._uploadGuidance = guidance || null;
  }

  async _handle(file) {
    try {
      this._validateFile(file);
      const roomLabel = this._roomLabelInput?.value?.trim() || '';
      const audienceMode = this._audienceModeInput?.value?.trim() || 'general';
      const job = await api.uploadFile(file, { roomLabel, audienceMode });
      await this._onJobCreated(job);
      await this._onJobUpdate(job);
      this._poll(job.job_id);
    } catch (error) {
      this._onJobError(
        error instanceof Error ? error : new Error(String(error)),
      );
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
