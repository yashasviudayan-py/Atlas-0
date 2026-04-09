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
   *   onJobCreated?: (job: any) => void,
   *   onJobUpdate?: (job: any) => void,
   *   onJobError?: (error: Error) => void,
   * }} opts
   */
  constructor(opts) {
    this._dropZone = opts.dropZone;
    this._fileInput = opts.fileInput;
    this._onJobCreated = opts.onJobCreated || (() => {});
    this._onJobUpdate = opts.onJobUpdate || (() => {});
    this._onJobError = opts.onJobError || (() => {});
    this._pollers = new Map();
  }

  init() {
    this._dropZone.addEventListener('click', () => this._fileInput.click());

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

  async _handle(file) {
    try {
      const job = await api.uploadFile(file);
      this._onJobCreated(job);
      this._onJobUpdate(job);
      this._poll(job.job_id);
    } catch (error) {
      this._onJobError(
        error instanceof Error ? error : new Error(String(error)),
      );
    }
  }

  _poll(jobId) {
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
