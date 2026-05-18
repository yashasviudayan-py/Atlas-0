/**
 * app.js — ATLAS-0 report-first web application.
 */

import * as api from './api.js';
import {
  BETA_SHARE_PROMPTS,
  CAPTURE_COACH_MODES,
  CURIOSITY_SAMPLE_GALLERY,
  DAILY_HOME_ACTIONS,
  FIX_LIBRARY_GUIDES,
  FIX_QUEST_TEMPLATES,
  HOME_BINGO_TASKS,
  PERSONAL_SAFETY_MODES,
  REPORT_DECISION_STEPS,
  REPORT_THEMES,
  ROOM_CARE_WEEK_TEMPLATE,
  ROOM_MYSTERY_MODES,
  ROOM_PERSONALITIES,
  ROOM_PLAYBOOKS,
  ROOM_RITUALS,
  SAFETY_MISSIONS,
  SEASONAL_RITUAL_PACKS,
} from './product_playbooks.js';
import { SceneViewer } from './scene_viewer.js';
import { UploadView } from './upload.js';

const THEME_STORAGE_KEY = 'atlas0.theme';
const MOTION_STORAGE_KEY = 'atlas0.reducedMotion';
const LOW_CONFIDENCE_STORAGE_KEY = 'atlas0.showLowConfidenceDefault';
const MISSION_STORAGE_KEY = 'atlas0.dailySafetyMission';
const CHALLENGE_SELECTION_STORAGE_KEY = 'atlas0.selectedChallenge';
const CHALLENGE_JOB_STORAGE_KEY = 'atlas0.challengeJobs';
const FIX_CHECKLIST_STORAGE_KEY = 'atlas0.fixChecklist';
const CAPTURE_COACH_STORAGE_KEY = 'atlas0.captureCoach';
const SESSION_STORAGE_KEY = 'atlas0.sessionId';
const FIRST_RUN_STORAGE_KEY = 'atlas0.firstRunStarted';
const RITUAL_STORAGE_KEY = 'atlas0.roomRituals';
const RITUAL_SELECTION_STORAGE_KEY = 'atlas0.selectedRitual';
const HOME_JOURNAL_STORAGE_KEY = 'atlas0.homeJournal';
const FAVORITE_ROOMS_STORAGE_KEY = 'atlas0.favoriteRooms';
const MYSTERY_MODE_STORAGE_KEY = 'atlas0.selectedMysteryMode';
const REPORT_STYLE_STORAGE_KEY = 'atlas0.reportStyle';
const DEFAULT_AUDIENCE_STORAGE_KEY = 'atlas0.defaultAudienceMode';
const DEFAULT_ROOM_LABEL_STORAGE_KEY = 'atlas0.defaultRoomLabel';
const DEFAULT_MYSTERY_MODE_STORAGE_KEY = 'atlas0.defaultMysteryMode';
const RESCAN_REMINDER_STORAGE_KEY = 'atlas0.rescanReminderCadence';
const LARGE_TEXT_STORAGE_KEY = 'atlas0.largeText';
const HIGH_CONTRAST_STORAGE_KEY = 'atlas0.highContrast';
const LAYOUT_DENSITY_STORAGE_KEY = 'atlas0.layoutDensity';
const FOCUS_MODE_STORAGE_KEY = 'atlas0.alwaysShowFocus';
const ROOM_PLAYBOOK_STORAGE_KEY = 'atlas0.selectedRoomPlaybook';
const WELCOME_TOUR_STORAGE_KEY = 'atlas0.welcomeTourDismissed';
const SHARE_CARD_STYLE_STORAGE_KEY = 'atlas0.shareCardStyle';
const ACTIVE_EVIDENCE_STORAGE_KEY = 'atlas0.activeEvidenceFrame';
const VERIFICATION_STORAGE_KEY = 'atlas0.fixVerificationState';
const HOME_BINGO_STORAGE_KEY = 'atlas0.homeBingo';
const FIX_QUEST_STORAGE_KEY = 'atlas0.fixQuests';
const PERSONAL_MODE_STORAGE_KEY = 'atlas0.selectedPersonalMode';
const REPORT_THEME_STORAGE_KEY = 'atlas0.reportTheme';
const DAILY_ACTION_STORAGE_KEY = 'atlas0.dailyAction';
const ROOM_CARE_CALENDAR_STORAGE_KEY = 'atlas0.roomCareCalendar';
const ROOM_CARE_COMPLETED_STORAGE_KEY = 'atlas0.roomCareCompleted';
const CARE_CADENCE_STORAGE_KEY = 'atlas0.careCadence';
const FIX_GUIDE_STORAGE_KEY = 'atlas0.activeFixGuide';
const LIVE_CAPTURE_COACH_STORAGE_KEY = 'atlas0.liveCaptureCoach';
const REPORT_QA_HISTORY_STORAGE_KEY = 'atlas0.reportQuestionHistory';
const PRIVACY_RECEIPT_STORAGE_KEY = 'atlas0.privacyReceiptEvidence';
const OFFLINE_ACK_STORAGE_KEY = 'atlas0.offlineReadyAcknowledged';
const REFERRAL_STORAGE_KEY = 'atlas0.betaReferralCode';

const SETTINGS_LOCAL_KEYS = [
  THEME_STORAGE_KEY,
  MOTION_STORAGE_KEY,
  LOW_CONFIDENCE_STORAGE_KEY,
  MISSION_STORAGE_KEY,
  CHALLENGE_SELECTION_STORAGE_KEY,
  CHALLENGE_JOB_STORAGE_KEY,
  FIX_CHECKLIST_STORAGE_KEY,
  RITUAL_STORAGE_KEY,
  RITUAL_SELECTION_STORAGE_KEY,
  HOME_JOURNAL_STORAGE_KEY,
  FAVORITE_ROOMS_STORAGE_KEY,
  MYSTERY_MODE_STORAGE_KEY,
  REPORT_STYLE_STORAGE_KEY,
  DEFAULT_AUDIENCE_STORAGE_KEY,
  DEFAULT_ROOM_LABEL_STORAGE_KEY,
  DEFAULT_MYSTERY_MODE_STORAGE_KEY,
  RESCAN_REMINDER_STORAGE_KEY,
  LARGE_TEXT_STORAGE_KEY,
  HIGH_CONTRAST_STORAGE_KEY,
  LAYOUT_DENSITY_STORAGE_KEY,
  FOCUS_MODE_STORAGE_KEY,
  ROOM_PLAYBOOK_STORAGE_KEY,
  WELCOME_TOUR_STORAGE_KEY,
  SHARE_CARD_STYLE_STORAGE_KEY,
  ACTIVE_EVIDENCE_STORAGE_KEY,
  VERIFICATION_STORAGE_KEY,
  HOME_BINGO_STORAGE_KEY,
  FIX_QUEST_STORAGE_KEY,
  PERSONAL_MODE_STORAGE_KEY,
  REPORT_THEME_STORAGE_KEY,
  DAILY_ACTION_STORAGE_KEY,
  ROOM_CARE_CALENDAR_STORAGE_KEY,
  ROOM_CARE_COMPLETED_STORAGE_KEY,
  CARE_CADENCE_STORAGE_KEY,
  FIX_GUIDE_STORAGE_KEY,
  LIVE_CAPTURE_COACH_STORAGE_KEY,
  REPORT_QA_HISTORY_STORAGE_KEY,
  PRIVACY_RECEIPT_STORAGE_KEY,
  OFFLINE_ACK_STORAGE_KEY,
  REFERRAL_STORAGE_KEY,
];

function readStoredPreference(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStoredPreference(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {}
}

function removeStoredPreference(key) {
  try {
    window.localStorage.removeItem(key);
  } catch {}
}

function betaSessionId() {
  const existing = readStoredPreference(SESSION_STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const generated = `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
  writeStoredPreference(SESSION_STORAGE_KEY, generated);
  return generated;
}

function urlAttribution() {
  const params = new URLSearchParams(window.location.search);
  return {
    referrer: document.referrer || null,
    utm_source: params.get('utm_source'),
    utm_campaign: params.get('utm_campaign'),
  };
}

function betaReferralCode() {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = (params.get('ref') || params.get('referral') || '').trim().slice(0, 80);
  if (fromUrl) {
    writeStoredPreference(REFERRAL_STORAGE_KEY, fromUrl);
    return fromUrl;
  }
  return (readStoredPreference(REFERRAL_STORAGE_KEY) || '').slice(0, 80);
}

function betaPersona() {
  const mode = selectedAudienceMode();
  if (mode === 'toddler') return 'parent_or_caregiver';
  if (mode === 'pet') return 'pet_owner';
  if (mode === 'renter') return 'renter';
  return 'home_user';
}

const VIEW_LABELS = {
  scan: 'Scan',
  report: 'Report',
  journal: 'Journal',
  scene: 'Scene',
  settings: 'Settings',
};

const state = {
  activeView: 'scan',
  activeJobId: null,
  activeSampleKey: null,
  jobs: new Map(),
  showLowConfidence: readStoredPreference(LOW_CONFIDENCE_STORAGE_KEY) === 'true',
  accessPolicy: null,
  privacyPolicy: null,
  uploadGuidance: null,
  trustProof: null,
  operatorSettings: null,
  reportViewEvents: new Set(),
  uploadCompleteEvents: new Set(),
  privacyReceiptEvents: new Set(),
  activeChallengeId: readStoredPreference(CHALLENGE_SELECTION_STORAGE_KEY) || null,
  activeRitualId: readStoredPreference(RITUAL_SELECTION_STORAGE_KEY) || null,
  activeMysteryModeId: readStoredPreference(MYSTERY_MODE_STORAGE_KEY) || null,
  activeRoomPlaybookId: readStoredPreference(ROOM_PLAYBOOK_STORAGE_KEY) || null,
  activePersonalModeId: readStoredPreference(PERSONAL_MODE_STORAGE_KEY) || null,
  activeEvidenceIndex: Number(readStoredPreference(ACTIVE_EVIDENCE_STORAGE_KEY) || 0),
  pendingUploadChallengeId: null,
  activeReportQuestion: null,
  activeReportAnswer: '',
};

const navButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('.nav-btn[data-view]')
);
const jumpButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('[data-jump-view]')
);
const sampleButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('[data-load-sample]')
);
const useCaseButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('[data-use-case-mode]')
);
const viewElements = document.querySelectorAll('.view');
const viewLabel = document.getElementById('hdr-view-label');

const healthStatus = document.getElementById('health-status');
const uploadStatus = document.getElementById('upload-status');
const reportStatus = document.getElementById('report-status');

const processStage = document.getElementById('process-stage');
const processCopy = document.getElementById('process-copy');
const processBar = document.getElementById('process-bar');
const processMeta = document.getElementById('process-meta');
const processGuidance = document.getElementById('process-guidance');
const roomLabelInput = /** @type {HTMLInputElement} */ (document.getElementById('room-label-input'));
const audienceModeInput = /** @type {HTMLSelectElement} */ (document.getElementById('audience-mode-input'));

const recentList = document.getElementById('upload-list');
const recentEmpty = document.getElementById('upload-empty');
const accessBanner = document.getElementById('access-banner');
const accessHelp = document.getElementById('access-help');
const uploadGuidanceCopy = document.getElementById('upload-guidance-copy');
const uploadDurationPill = document.getElementById('upload-duration-pill');
const uploadSizePill = document.getElementById('upload-size-pill');
const scanWizardStatus = document.getElementById('scan-wizard-status');
const welcomeTourCard = document.getElementById('welcome-tour-card');
const welcomeTourCompleteBtn = /** @type {HTMLButtonElement} */ (document.getElementById('welcome-tour-complete'));
const roomPlaybookGrid = document.getElementById('room-playbook-grid');
const challengeLibraryGrid = document.getElementById('challenge-library-grid');
const challengeStreakSummary = document.getElementById('challenge-streak-summary');
const privacyPolicy = document.getElementById('privacy-policy');
const operatorPolicy = document.getElementById('operator-policy');
const operatorQueue = document.getElementById('operator-queue');
const operatorSystem = document.getElementById('operator-system');
const operatorEval = document.getElementById('operator-eval');
const operatorProduct = document.getElementById('operator-product');
const operatorPruneBtn = /** @type {HTMLButtonElement} */ (document.getElementById('operator-prune-btn'));
const accessTokenInput = /** @type {HTMLInputElement} */ (document.getElementById('access-token-input'));
const accessTokenSave = /** @type {HTMLButtonElement} */ (document.getElementById('access-token-save'));
const accessTokenClear = /** @type {HTMLButtonElement} */ (document.getElementById('access-token-clear'));
const waitlistEmailInput = /** @type {HTMLInputElement} */ (document.getElementById('waitlist-email-input'));
const waitlistUseCaseInput = /** @type {HTMLInputElement} */ (document.getElementById('waitlist-use-case-input'));
const waitlistReferralInput = /** @type {HTMLInputElement} */ (document.getElementById('waitlist-referral-input'));
const waitlistSubmitBtn = /** @type {HTMLButtonElement} */ (document.getElementById('waitlist-submit-btn'));
const waitlistNote = document.getElementById('waitlist-note');

const reportHero = document.getElementById('report-hero');
const reportHeroMeta = document.getElementById('report-hero-meta');
const briefExecutiveTitle = document.getElementById('brief-executive-title');
const briefExecutiveCopy = document.getElementById('brief-executive-copy');
const briefExecutiveActions = document.getElementById('brief-executive-actions');
const briefConfidenceLabel = document.getElementById('brief-confidence-label');
const briefConfidenceMeter = document.getElementById('brief-confidence-meter');
const briefConfidenceCopy = document.getElementById('brief-confidence-copy');
const briefConfidenceDetails = /** @type {HTMLDetailsElement} */ (
  document.getElementById('brief-confidence-details')
);
const summaryObjects = document.getElementById('summary-objects');
const summaryHazards = document.getElementById('summary-hazards');
const summarySeverity = document.getElementById('summary-severity');
const summaryConfidence = document.getElementById('summary-confidence');
const summaryCoverage = document.getElementById('summary-coverage');
const summarySource = document.getElementById('summary-source');
const roomPassportPanel = document.getElementById('room-passport-panel');
const roomScorecard = document.getElementById('room-scorecard');
const fixVerificationPanel = document.getElementById('fix-verification-panel');
const reportActionLoop = document.getElementById('report-action-loop');
const fixFirstList = document.getElementById('fix-first-list');
const scanQualityCard = document.getElementById('scan-quality-card');
const reportPostureCard = document.getElementById('report-posture-card');
const reportEvalCard = document.getElementById('report-eval-card');
const weekendFixList = document.getElementById('weekend-fix-list');
const roomWinsList = document.getElementById('room-wins-list');
const fixChecklistList = document.getElementById('fix-checklist-list');
const challengeResultCard = document.getElementById('challenge-result-card');
const lowConfidenceToggle = /** @type {HTMLInputElement} */ (document.getElementById('low-confidence-toggle'));
const settingsLowConfidenceToggle = /** @type {HTMLInputElement} */ (
  document.getElementById('settings-low-confidence-toggle')
);
const findingToggleNote = document.getElementById('finding-toggle-note');
const shareLinkNote = document.getElementById('share-link-note');
const reportHeadline = document.getElementById('report-headline');
const reportSubhead = document.getElementById('report-subhead');
const reportHazards = document.getElementById('risk-report-list');
const reportRecommendations = document.getElementById('rec-list');
const reportEvidence = document.getElementById('evidence-grid');
const evidenceTimeline = document.getElementById('evidence-timeline');
const trustNotes = document.getElementById('trust-notes');
const exportPdfBtn = /** @type {HTMLAnchorElement} */ (document.getElementById('export-pdf-btn'));
const copyShareBtn = /** @type {HTMLButtonElement} */ (document.getElementById('copy-share-btn'));
const deleteJobBtn = /** @type {HTMLButtonElement} */ (document.getElementById('delete-job-btn'));
const copyShareCardBtn = /** @type {HTMLButtonElement} */ (document.getElementById('copy-share-card-btn'));
const shareCardStyleInput = /** @type {HTMLSelectElement} */ (document.getElementById('share-card-style'));
const shareCardCopy = document.getElementById('share-card-copy');
const themeToggle = /** @type {HTMLInputElement} */ (document.getElementById('theme-toggle'));
const themeStatus = document.getElementById('theme-status');
const motionToggle = /** @type {HTMLInputElement} */ (document.getElementById('motion-toggle'));
const motionStatus = document.getElementById('motion-status');
const settingsTokenStatus = document.getElementById('settings-token-status');
const settingsTokenClear = /** @type {HTMLButtonElement} */ (document.getElementById('settings-token-clear'));
const settingsSampleBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-sample-btn'));
const settingsOverviewGrid = document.getElementById('settings-overview-grid');
const settingsControlStatus = document.getElementById('settings-control-status');
const settingsReportStyleInput = /** @type {HTMLSelectElement} */ (document.getElementById('settings-report-style'));
const settingsReportThemeInput = /** @type {HTMLSelectElement} */ (document.getElementById('settings-report-theme'));
const settingsCareCadenceInput = /** @type {HTMLSelectElement} */ (document.getElementById('settings-care-cadence'));
const settingsDefaultAudienceInput = /** @type {HTMLSelectElement} */ (document.getElementById('settings-default-audience'));
const settingsDefaultRoomLabelInput = /** @type {HTMLInputElement} */ (document.getElementById('settings-default-room-label'));
const settingsDefaultMysteryInput = /** @type {HTMLSelectElement} */ (document.getElementById('settings-default-mystery'));
const settingsRescanReminderInput = /** @type {HTMLSelectElement} */ (document.getElementById('settings-rescan-reminder'));
const settingsLargeTextToggle = /** @type {HTMLInputElement} */ (document.getElementById('settings-large-text-toggle'));
const settingsHighContrastToggle = /** @type {HTMLInputElement} */ (document.getElementById('settings-high-contrast-toggle'));
const settingsLayoutDensityInput = /** @type {HTMLSelectElement} */ (document.getElementById('settings-layout-density'));
const settingsFocusToggle = /** @type {HTMLInputElement} */ (document.getElementById('settings-focus-toggle'));
const settingsClearJournalBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-clear-journal'));
const settingsClearRitualsBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-clear-rituals'));
const settingsClearCompanionBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-clear-companion'));
const settingsClearDailyValueBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-clear-daily-value'));
const settingsClearDefaultsBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-clear-defaults'));
const settingsClearAllLocalBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-clear-all-local'));
const settingsOpenCurrentReportBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-open-current-report'));
const settingsDeleteCurrentReportBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-delete-current-report'));
const settingsRegenerateCareWeekBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-regenerate-care-week'));
const settingsFeedbackCopyBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-feedback-copy'));
const settingsBadResultBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-bad-result'));
const settingsFeatureRequestBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-feature-request'));
const settingsBetaInviteBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-beta-invite'));
const settingsWaitlistBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-waitlist'));
const settingsReplayWelcomeBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-replay-welcome-tour'));
const settingsReplayWeeklyBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-replay-weekly-recap'));
const settingsExportBackupBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-export-backup'));
const settingsImportBackupBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-import-backup'));
const settingsImportFileInput = /** @type {HTMLInputElement} */ (document.getElementById('settings-import-file'));
const settingsPrivacySummaryBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-privacy-summary'));
const dailyMissionTitle = document.getElementById('daily-mission-title');
const dailyMissionCopy = document.getElementById('daily-mission-copy');
const dailyMissionSteps = document.getElementById('daily-mission-steps');
const dailyMissionProgress = document.getElementById('daily-mission-progress');
const dailyMissionStart = /** @type {HTMLButtonElement} */ (document.getElementById('daily-mission-start'));
const dailyMissionComplete = /** @type {HTMLButtonElement} */ (document.getElementById('daily-mission-complete'));
const ritualTodayTitle = document.getElementById('ritual-today-title');
const ritualTodayCopy = document.getElementById('ritual-today-copy');
const ritualTodayPrompt = document.getElementById('ritual-today-prompt');
const ritualTodayMeta = document.getElementById('ritual-today-meta');
const ritualStreakSummary = document.getElementById('ritual-streak-summary');
const ritualGrid = document.getElementById('ritual-grid');
const seasonalPackGrid = document.getElementById('seasonal-pack-grid');
const ritualStartBtn = /** @type {HTMLButtonElement} */ (document.getElementById('ritual-start-btn'));
const ritualCompleteBtn = /** @type {HTMLButtonElement} */ (document.getElementById('ritual-complete-btn'));
const homePulseCard = document.getElementById('home-pulse-card');
const homeCompanionPanel = document.getElementById('home-companion-panel');
const oneThingTodayCard = document.getElementById('one-thing-today-card');
const weeklyRecapCard = document.getElementById('weekly-recap-card');
const homeBingoGrid = document.getElementById('home-bingo-grid');
const roomCareCalendar = document.getElementById('room-care-calendar');
const roomCareRegenerateBtn = /** @type {HTMLButtonElement} */ (document.getElementById('room-care-regenerate-btn'));
const fixLibraryGrid = document.getElementById('fix-library-grid');
const personalModeGrid = document.getElementById('personal-mode-grid');
const mysteryModeGrid = document.getElementById('mystery-mode-grid');
const mysteryPromptCopy = document.getElementById('mystery-prompt-copy');
const curiositySampleGrid = document.getElementById('curiosity-sample-grid');
const trustProofDashboard = document.getElementById('trust-proof-dashboard');
const trustProofMetrics = document.getElementById('trust-proof-metrics');
const trustProofDetails = document.getElementById('trust-proof-details');
const captureCoachTitle = document.getElementById('capture-coach-title');
const captureCoachCopy = document.getElementById('capture-coach-copy');
const captureCoachRoute = document.getElementById('capture-coach-route');
const captureCoachChecks = document.getElementById('capture-coach-checks');
const captureCoachPrompt = document.getElementById('capture-coach-prompt');
const captureCoachStatus = document.getElementById('capture-coach-status');
const captureCoachMeter = document.getElementById('capture-coach-meter');
const liveCaptureCoach = document.getElementById('live-capture-coach');
const liveCaptureVideo = /** @type {HTMLVideoElement} */ (document.getElementById('live-capture-video'));
const liveCaptureCanvas = /** @type {HTMLCanvasElement} */ (document.getElementById('live-capture-canvas'));
const liveCapturePreview = document.getElementById('live-capture-preview');
const liveCaptureStatus = document.getElementById('live-capture-status');
const liveCaptureStats = document.getElementById('live-capture-stats');
const liveCaptureGuidance = document.getElementById('live-capture-guidance');
const liveCaptureStartBtn = /** @type {HTMLButtonElement} */ (document.getElementById('live-capture-start'));
const liveCaptureStopBtn = /** @type {HTMLButtonElement} */ (document.getElementById('live-capture-stop'));
const betaShareCopy = document.getElementById('beta-share-copy');
const copyBetaInviteBtn = /** @type {HTMLButtonElement} */ (document.getElementById('copy-beta-invite-btn'));
const briefTriageStrip = document.getElementById('brief-triage-strip');
const fieldNotesPanel = document.getElementById('field-notes-panel');
const roomMapPreview = document.getElementById('room-map-preview');
const beforeAfterStory = document.getElementById('before-after-story');
const reportThemePanel = document.getElementById('report-theme-panel');
const reportThemeInput = /** @type {HTMLSelectElement} */ (document.getElementById('report-theme-style'));
const reportThemeCopy = document.getElementById('report-theme-copy');
const fixQuestPanel = document.getElementById('fix-quest-panel');
const roomComparePanel = document.getElementById('room-compare-panel');
const smartRescanCoach = document.getElementById('smart-rescan-coach');
const evidenceStoryPanel = document.getElementById('evidence-story-panel');
const reportQuestionList = document.getElementById('report-question-list');
const reportQuestionAnswer = document.getElementById('report-question-answer');
const copyReportAnswerBtn = /** @type {HTMLButtonElement} */ (document.getElementById('copy-report-answer-btn'));
const privacyReceiptPanel = document.getElementById('privacy-receipt-panel');
const privacyReceiptSummary = document.getElementById('privacy-receipt-summary');
const privacyEvidenceList = document.getElementById('privacy-evidence-list');
const copyPrivacyReceiptBtn = /** @type {HTMLButtonElement} */ (document.getElementById('copy-privacy-receipt-btn'));
const downloadPrivacyReceiptBtn = /** @type {HTMLButtonElement} */ (document.getElementById('download-privacy-receipt-btn'));
const homeJournalSummary = document.getElementById('home-journal-summary');
const roomPersonalityPanel = document.getElementById('room-personality-panel');
const roomHealthTimelinePanel = document.getElementById('room-health-timeline-panel');
const homeJournalGrid = document.getElementById('home-journal-grid');
const homeJournalEmpty = document.getElementById('home-journal-empty');

const sceneCanvas = /** @type {HTMLCanvasElement} */ (document.getElementById('scene-canvas'));
const sceneEmpty = document.getElementById('scene-empty');
const sceneObjList = document.getElementById('scene-obj-list');
const sceneViewer = new SceneViewer(sceneCanvas, sceneEmpty, sceneObjList);
let sceneReady = false;
let liveCaptureStream = null;
let liveCaptureTimer = null;
let liveCaptureStartedAt = 0;
let liveCaptureLastFrame = null;
let liveCaptureLastEventAt = 0;
let liveCaptureCoverageCells = new Set();

const toast = document.getElementById('toast');
const offlineBanner = document.getElementById('offline-banner');
let toastTimer = null;

function showToast(message, timeout = 2600) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), timeout);
}

function updateOfflineBanner() {
  if (!offlineBanner) {
    return;
  }
  offlineBanner.classList.toggle('show', navigator.onLine === false);
  if (navigator.onLine === false) {
    offlineBanner.textContent = 'Offline mode: local journal/settings remain available. New uploads will be saved for retry when you reconnect.';
  } else if (offlineBanner.dataset.queueCount && offlineBanner.dataset.queueCount !== '0') {
    offlineBanner.classList.add('show');
    offlineBanner.textContent = `${offlineBanner.dataset.queueCount} upload${offlineBanner.dataset.queueCount === '1' ? '' : 's'} waiting to retry. Keep this tab open while ATLAS-0 reconnects.`;
  }
}

async function registerServiceWorker() {
  if (!('serviceWorker' in navigator) || window.location.protocol === 'file:') {
    return;
  }
  try {
    const registration = await navigator.serviceWorker.register('/app/service-worker.js', {
      scope: '/app/',
    });
    await navigator.serviceWorker.ready;
    if (!readStoredPreference(OFFLINE_ACK_STORAGE_KEY)) {
      writeStoredPreference(OFFLINE_ACK_STORAGE_KEY, 'true');
      trackProductEvent('pwa_offline_ready', {
        surface: 'service_worker',
        reason: registration.active ? 'active' : 'registered',
      });
    }
  } catch {
    // PWA support is progressive enhancement; never block scanning or reports.
  }
}

function requestedJobId() {
  return new URLSearchParams(window.location.search).get('job');
}

function requestedSampleKey() {
  return new URLSearchParams(window.location.search).get('sample');
}

function requestedView() {
  const value = new URLSearchParams(window.location.search).get('view');
  return value && VIEW_LABELS[value] ? value : null;
}

function reportDeepLink(job) {
  if (!job?.job_id) {
    return '';
  }
  const relative = job.share_url || (job.sample_key
    ? `/app?view=report&sample=${encodeURIComponent(job.sample_key)}`
    : `/app?view=report&job=${encodeURIComponent(job.job_id)}`);
  return new URL(relative, window.location.origin).toString();
}

function syncUrlState() {
  const url = new URL(window.location.href);
  if (state.activeSampleKey) {
    url.searchParams.delete('job');
    url.searchParams.set('sample', state.activeSampleKey);
  } else if (state.activeJobId) {
    url.searchParams.set('job', state.activeJobId);
    url.searchParams.delete('sample');
  } else {
    url.searchParams.delete('job');
    url.searchParams.delete('sample');
  }

  const view = state.activeView || 'scan';
  if (view === 'scan' && !state.activeJobId) {
    url.searchParams.delete('view');
  } else {
    url.searchParams.set('view', view);
  }

  const next = `${url.pathname}${url.search}${url.hash}`;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next !== current) {
    window.history.replaceState({}, '', next);
  }
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const ghost = document.createElement('textarea');
  ghost.value = value;
  ghost.setAttribute('readonly', 'true');
  ghost.style.position = 'absolute';
  ghost.style.left = '-9999px';
  document.body.appendChild(ghost);
  ghost.select();
  document.execCommand('copy');
  document.body.removeChild(ghost);
}

async function trackProductEvent(eventName, extra = {}) {
  try {
    await api.logProductEvent({
      event_name: eventName,
      session_id: betaSessionId(),
      client_ts: new Date().toISOString(),
      audience_mode: selectedAudienceMode(),
      room_label: roomLabelInput?.value?.trim() || null,
      room_labeled: Boolean(roomLabelInput?.value?.trim()),
      persona: betaPersona(),
      referral_code: betaReferralCode() || null,
      ...urlAttribution(),
      ...extra,
    });
  } catch {}
}

function markFirstRunStarted(surface = 'unknown') {
  if (readStoredPreference(FIRST_RUN_STORAGE_KEY)) {
    return;
  }
  writeStoredPreference(FIRST_RUN_STORAGE_KEY, new Date().toISOString());
  trackProductEvent('first_run_started', {
    surface,
    mission_id: activeChallenge().id,
    challenge_id: activeChallenge().id,
  });
}

function applyThemePreference(theme) {
  const nextTheme = theme === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = nextTheme;
  writeStoredPreference(THEME_STORAGE_KEY, nextTheme);

  if (themeToggle) {
    themeToggle.checked = nextTheme === 'dark';
  }
  if (themeStatus) {
    themeStatus.textContent = nextTheme === 'dark' ? 'Using dark mode' : 'Using light mode';
  }
}

function applyMotionPreference(reduced) {
  const enabled = Boolean(reduced);
  if (enabled) {
    document.documentElement.dataset.reducedMotion = 'true';
  } else {
    delete document.documentElement.dataset.reducedMotion;
  }
  writeStoredPreference(MOTION_STORAGE_KEY, enabled ? 'true' : 'false');

  if (motionToggle) {
    motionToggle.checked = enabled;
  }
  if (motionStatus) {
    motionStatus.textContent = enabled ? 'Animations are reduced' : 'Animations are on';
  }
}

function storedBoolean(key) {
  return readStoredPreference(key) === 'true';
}

function storedChoice(key, allowed, fallback) {
  const value = readStoredPreference(key);
  return allowed.includes(value) ? value : fallback;
}

function currentReportStyle() {
  return storedChoice(REPORT_STYLE_STORAGE_KEY, ['action-first', 'evidence-detailed', 'calm-brief'], 'action-first');
}

function currentDefaultAudience() {
  return storedChoice(DEFAULT_AUDIENCE_STORAGE_KEY, Object.keys(CAPTURE_COACH_MODES), 'general');
}

function currentLayoutDensity() {
  return storedChoice(LAYOUT_DENSITY_STORAGE_KEY, ['spacious', 'compact'], 'spacious');
}

function currentRescanReminder() {
  return storedChoice(RESCAN_REMINDER_STORAGE_KEY, ['off', 'weekly', 'monthly'], 'off');
}

function currentReportTheme() {
  return storedChoice(REPORT_THEME_STORAGE_KEY, REPORT_THEMES.map((theme) => theme.id), 'calm-brief');
}

function currentCareCadence() {
  return storedChoice(CARE_CADENCE_STORAGE_KEY, ['weekly', 'twice-weekly', 'daily'], 'weekly');
}

function currentDefaultMysteryMode() {
  const value = readStoredPreference(DEFAULT_MYSTERY_MODE_STORAGE_KEY);
  return ROOM_MYSTERY_MODES.some((mode) => mode.id === value) ? value : '';
}

function setRootDatasetFlag(name, enabled) {
  if (enabled) {
    document.documentElement.dataset[name] = 'true';
  } else {
    delete document.documentElement.dataset[name];
  }
}

function applyAccessibilityPreferences() {
  setRootDatasetFlag('largeText', storedBoolean(LARGE_TEXT_STORAGE_KEY));
  setRootDatasetFlag('highContrast', storedBoolean(HIGH_CONTRAST_STORAGE_KEY));
  setRootDatasetFlag('alwaysFocus', storedBoolean(FOCUS_MODE_STORAGE_KEY));
  document.documentElement.dataset.layoutDensity = currentLayoutDensity();

  if (settingsLargeTextToggle) {
    settingsLargeTextToggle.checked = storedBoolean(LARGE_TEXT_STORAGE_KEY);
  }
  if (settingsHighContrastToggle) {
    settingsHighContrastToggle.checked = storedBoolean(HIGH_CONTRAST_STORAGE_KEY);
  }
  if (settingsFocusToggle) {
    settingsFocusToggle.checked = storedBoolean(FOCUS_MODE_STORAGE_KEY);
  }
  if (settingsLayoutDensityInput) {
    settingsLayoutDensityInput.value = currentLayoutDensity();
  }
}

function syncSettingsPreferenceControls() {
  if (settingsReportStyleInput) {
    settingsReportStyleInput.value = currentReportStyle();
  }
  if (settingsReportThemeInput) {
    settingsReportThemeInput.value = currentReportTheme();
  }
  if (settingsCareCadenceInput) {
    settingsCareCadenceInput.value = currentCareCadence();
  }
  if (reportThemeInput) {
    reportThemeInput.value = currentReportTheme();
  }
  if (settingsDefaultAudienceInput) {
    settingsDefaultAudienceInput.value = currentDefaultAudience();
  }
  if (settingsDefaultRoomLabelInput) {
    settingsDefaultRoomLabelInput.value = readStoredPreference(DEFAULT_ROOM_LABEL_STORAGE_KEY) || '';
  }
  if (settingsDefaultMysteryInput) {
    settingsDefaultMysteryInput.innerHTML = [
      '<option value="">Daily rotating prompt</option>',
      ...ROOM_MYSTERY_MODES.map((mode) => (
        `<option value="${escapeHtml(mode.id)}">${escapeHtml(mode.title)}</option>`
      )),
    ].join('');
    settingsDefaultMysteryInput.value = currentDefaultMysteryMode();
  }
  if (settingsRescanReminderInput) {
    settingsRescanReminderInput.value = currentRescanReminder();
  }
  if (shareCardStyleInput) {
    shareCardStyleInput.value = currentShareCardStyle();
  }
  applyAccessibilityPreferences();
}

function applyDefaultScanPreferences(force = false) {
  const defaultAudience = currentDefaultAudience();
  const defaultLabel = readStoredPreference(DEFAULT_ROOM_LABEL_STORAGE_KEY) || '';
  const defaultMystery = currentDefaultMysteryMode();

  if (audienceModeInput && (force || !audienceModeInput.value || audienceModeInput.value === 'general')) {
    audienceModeInput.value = defaultAudience;
  }
  if (roomLabelInput && defaultLabel && (force || !roomLabelInput.value.trim())) {
    roomLabelInput.value = defaultLabel;
  }
  if (defaultMystery) {
    state.activeMysteryModeId = defaultMystery;
    writeStoredPreference(MYSTERY_MODE_STORAGE_KEY, defaultMystery);
  }
  renderMysteryModes();
  renderCaptureCoach();
  renderSettingsControlCenter();
}

function localDataCounts() {
  const journal = Object.values(readHomeJournal());
  const favorites = readFavoriteRooms();
  const ritualState = readRitualState();
  const challengeRaw = readStoredPreference(CHALLENGE_JOB_STORAGE_KEY);
  let challengeCount = 0;
  try {
    challengeCount = Object.keys(JSON.parse(challengeRaw || '{}')).length;
  } catch {
    challengeCount = 0;
  }
  let bingoDone = 0;
  try {
    bingoDone = Object.keys(JSON.parse(readStoredPreference(HOME_BINGO_STORAGE_KEY) || '{}')).length;
  } catch {
    bingoDone = 0;
  }
  let careDone = 0;
  try {
    careDone = Object.keys(JSON.parse(readStoredPreference(ROOM_CARE_COMPLETED_STORAGE_KEY) || '{}')).length;
  } catch {
    careDone = 0;
  }
  return {
    rooms: journal.length,
    favorites: favorites.size,
    rituals: ritualState.completedDates.length,
    challengeJobs: challengeCount,
    bingoDone,
    careDone,
  };
}

function reportStyleLabel(style = currentReportStyle()) {
  return {
    'action-first': 'Action-first',
    'evidence-detailed': 'Detailed evidence',
    'calm-brief': 'Calm brief',
  }[style] || 'Action-first';
}

function renderSettingsControlCenter() {
  if (!settingsOverviewGrid) {
    return;
  }
  const counts = localDataCounts();
  const active = activeJob();
  const privacy = state.privacyPolicy;
  const tokenStored = Boolean(api.getAccessToken());
  const theme = readStoredPreference(THEME_STORAGE_KEY) || document.documentElement.dataset.theme || 'light';
  const motion = storedBoolean(MOTION_STORAGE_KEY) ? 'Reduced' : 'On';
  const accessibility = [
    storedBoolean(LARGE_TEXT_STORAGE_KEY) ? 'Large text' : null,
    storedBoolean(HIGH_CONTRAST_STORAGE_KEY) ? 'High contrast' : null,
    currentLayoutDensity() === 'compact' ? 'Compact' : null,
    storedBoolean(FOCUS_MODE_STORAGE_KEY) ? 'Focus always visible' : null,
  ].filter(Boolean).join(' · ') || 'Standard';
  settingsOverviewGrid.innerHTML = [
    { label: 'Interface', value: `${capitalize(theme)} · ${motion}`, detail: accessibility },
    { label: 'Report default', value: reportStyleLabel(), detail: `${reportThemeLabel()} · ${state.showLowConfidence ? 'Lower-confidence findings visible' : 'Cleaner high-confidence view'}` },
    { label: 'Scan default', value: CAPTURE_COACH_MODES[currentDefaultAudience()]?.title || 'General home safety', detail: readStoredPreference(DEFAULT_ROOM_LABEL_STORAGE_KEY) || 'No default room label' },
    { label: 'Local journal', value: `${counts.rooms} room${counts.rooms === 1 ? '' : 's'}`, detail: `${counts.favorites} favorite${counts.favorites === 1 ? '' : 's'} · ${counts.rituals} ritual day${counts.rituals === 1 ? '' : 's'} · ${counts.bingoDone} bingo · ${counts.careDone} care tasks` },
    { label: 'Privacy posture', value: privacy ? `${privacy.retention_days} day retention` : 'Policy unavailable', detail: privacy?.delete_supported ? 'Delete controls available' : 'Report delete status unknown' },
    { label: 'Access', value: tokenStored ? 'Token stored locally' : 'No token stored', detail: active?.status === 'complete' ? 'Current report ready' : 'No completed active report' },
  ].map((item) => `
    <article class="settings-overview-card">
      <span>${escapeHtml(item.label)}</span>
      <strong>${escapeHtml(item.value)}</strong>
      <small>${escapeHtml(item.detail)}</small>
    </article>
  `).join('');

  if (settingsControlStatus) {
    settingsControlStatus.textContent = `Local only · ${counts.rooms} room journal entr${counts.rooms === 1 ? 'y' : 'ies'} · ${currentCareCadence()} care cadence`;
  }
  if (settingsOpenCurrentReportBtn) {
    settingsOpenCurrentReportBtn.disabled = !active;
  }
  if (settingsDeleteCurrentReportBtn) {
    settingsDeleteCurrentReportBtn.disabled = !active || Boolean(active.is_sample);
  }
}

function clearLocalKeys(keys) {
  keys.forEach(removeStoredPreference);
}

function clearLocalPrefixes(prefixes) {
  try {
    Object.keys(window.localStorage)
      .filter((key) => prefixes.some((prefix) => key === prefix || key.startsWith(`${prefix}.`)))
      .forEach((key) => window.localStorage.removeItem(key));
  } catch {}
}

function resetLocalRuntimeState() {
  state.activeChallengeId = null;
  state.activeRitualId = null;
  state.activeMysteryModeId = null;
  state.activeRoomPlaybookId = null;
  state.activePersonalModeId = null;
  state.pendingUploadChallengeId = null;
  state.showLowConfidence = false;
  syncLowConfidenceControls();
  renderDailyMission();
  renderChallengeLibrary();
  renderRoomRituals();
  renderMysteryModes();
  renderRoomPlaybooks();
  renderHomeCompanion();
  renderWelcomeTour();
  renderHomeJournal();
  renderHomePulse();
  renderCaptureCoach();
  renderSettingsControlCenter();
}

function localBackupPayload() {
  const preferences = {};
  SETTINGS_LOCAL_KEYS.forEach((key) => {
    const value = readStoredPreference(key);
    if (value !== null) {
      preferences[key] = value;
    }
  });
  return {
    product: 'ATLAS-0',
    schema: 'settings-local-backup-v1',
    exportedAt: new Date().toISOString(),
    preferences,
    homeJournal: readHomeJournal(),
    favoriteRooms: [...readFavoriteRooms()],
  };
}

function downloadTextFile(filename, text, type = 'application/json') {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function newestJournalEntry(existing, incoming) {
  const existingTs = Date.parse(existing?.lastCheckedAt || '');
  const incomingTs = Date.parse(incoming?.lastCheckedAt || '');
  return Number.isFinite(incomingTs) && (!Number.isFinite(existingTs) || incomingTs >= existingTs)
    ? incoming
    : existing;
}

function importLocalBackup(payload) {
  if (!payload || payload.product !== 'ATLAS-0' || payload.schema !== 'settings-local-backup-v1') {
    throw new Error('This is not a valid ATLAS-0 local backup.');
  }
  const preferences = payload.preferences && typeof payload.preferences === 'object' ? payload.preferences : {};
  Object.entries(preferences).forEach(([key, value]) => {
    if (SETTINGS_LOCAL_KEYS.includes(key) && typeof value === 'string') {
      writeStoredPreference(key, value);
    }
  });

  const currentJournal = readHomeJournal();
  const incomingJournal = payload.homeJournal && typeof payload.homeJournal === 'object' && !Array.isArray(payload.homeJournal)
    ? payload.homeJournal
    : {};
  Object.entries(incomingJournal).forEach(([key, entry]) => {
    if (entry && typeof entry === 'object') {
      currentJournal[key] = newestJournalEntry(currentJournal[key], entry);
    }
  });
  writeHomeJournal(currentJournal);

  if (Array.isArray(payload.favoriteRooms)) {
    writeFavoriteRooms(new Set([...readFavoriteRooms(), ...payload.favoriteRooms.filter((item) => typeof item === 'string')]));
  }

  state.showLowConfidence = storedBoolean(LOW_CONFIDENCE_STORAGE_KEY);
  state.activeChallengeId = readStoredPreference(CHALLENGE_SELECTION_STORAGE_KEY) || null;
  state.activeRitualId = readStoredPreference(RITUAL_SELECTION_STORAGE_KEY) || null;
  state.activeMysteryModeId = readStoredPreference(MYSTERY_MODE_STORAGE_KEY) || currentDefaultMysteryMode() || null;
  state.activeRoomPlaybookId = readStoredPreference(ROOM_PLAYBOOK_STORAGE_KEY) || null;
  state.activePersonalModeId = readStoredPreference(PERSONAL_MODE_STORAGE_KEY) || null;
  applyThemePreference(readStoredPreference(THEME_STORAGE_KEY) || 'light');
  applyMotionPreference(storedBoolean(MOTION_STORAGE_KEY));
  applyAccessibilityPreferences();
  syncSettingsPreferenceControls();
  applyDefaultScanPreferences(true);
  syncLowConfidenceControls();
  renderDailyMission();
  renderChallengeLibrary();
  renderRoomRituals();
  renderMysteryModes();
  renderRoomPlaybooks();
  renderHomeCompanion();
  renderWelcomeTour();
  renderHomeJournal();
  renderHomePulse();
  renderReport(activeJob());
  renderSettingsControlCenter();
}

function privacySummaryText() {
  const privacy = state.privacyPolicy;
  return [
    'ATLAS-0 privacy summary',
    '',
    'ATLAS-0 is decision support, not safety certification.',
    privacy?.summary || 'Privacy policy is unavailable from the API right now.',
    `Retention window: ${privacy?.retention_days ?? 'unknown'} day(s).`,
    `Original uploads saved: ${privacy?.save_original_uploads ? 'yes' : 'no by default'}.`,
    `Text redaction: ${privacy?.text_redaction_enabled ? 'enabled' : 'unknown'}.`,
    `Delete support: ${privacy?.delete_supported ? 'available from report view' : 'unknown'}.`,
    '',
    'Local settings, Home Journal, rituals, and streaks live only in this browser unless you export a local backup.',
  ].join('\n');
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('Could not read the selected backup file.'));
    reader.readAsText(file);
  });
}

function syncLowConfidenceControls() {
  if (lowConfidenceToggle) {
    lowConfidenceToggle.checked = state.showLowConfidence;
  }
  if (settingsLowConfidenceToggle) {
    settingsLowConfidenceToggle.checked = state.showLowConfidence;
  }
}

function setLowConfidenceVisibility(enabled, persist = true) {
  state.showLowConfidence = Boolean(enabled);
  if (persist) {
    writeStoredPreference(LOW_CONFIDENCE_STORAGE_KEY, state.showLowConfidence ? 'true' : 'false');
  }
  syncLowConfidenceControls();
  renderReport(activeJob());
}

function syncSettingsAccessStatus() {
  const tokenStored = Boolean(api.getAccessToken());
  if (settingsTokenStatus) {
    settingsTokenStatus.textContent = tokenStored
      ? 'Private-beta token stored locally'
      : 'No private-beta token stored';
  }
  if (settingsTokenClear) {
    settingsTokenClear.disabled = !tokenStored;
  }
  renderSettingsControlCenter();
}

async function loadSampleReport(sampleJourneyId = 'walkthrough', surface = 'sample_report') {
  try {
    markFirstRunStarted('sample_report');
    if (sampleJourneyId !== 'walkthrough') {
      await trackProductEvent('sample_journey_opened', {
        surface,
        sample_key: sampleJourneyId,
      });
    }
    const sample = await api.fetchSampleReport();
    upsertJob(sample);
    setActiveJob(sample.job_id);
    switchView('report');
    showToast('Sample report loaded.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not load the sample report.', 3600);
  }
}

function localDateKey(date = new Date()) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function dailyMissionForDate(date = new Date()) {
  const dayNumber = Math.floor(new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime() / 86_400_000);
  return SAFETY_MISSIONS[dayNumber % SAFETY_MISSIONS.length];
}

function challengeById(id) {
  return SAFETY_MISSIONS.find((challenge) => challenge.id === id) || null;
}

function activeChallenge() {
  return challengeById(state.activeChallengeId) || dailyMissionForDate();
}

function dailyMissionStreak(completedDates, fromDate = new Date()) {
  const completed = new Set(completedDates);
  let streak = 0;
  const cursor = new Date(fromDate.getFullYear(), fromDate.getMonth(), fromDate.getDate());
  while (completed.has(localDateKey(cursor))) {
    streak += 1;
    cursor.setDate(cursor.getDate() - 1);
  }
  return streak;
}

function readDailyMissionState() {
  const raw = readStoredPreference(MISSION_STORAGE_KEY);
  if (!raw) {
    return { completedDates: [], completedChallengeIds: [] };
  }
  try {
    const parsed = JSON.parse(raw);
    return {
      completedDates: Array.isArray(parsed.completedDates) ? parsed.completedDates.slice(-45) : [],
      completedChallengeIds: Array.isArray(parsed.completedChallengeIds) ? parsed.completedChallengeIds.slice(-80) : [],
      lastMissionId: typeof parsed.lastMissionId === 'string' ? parsed.lastMissionId : null,
    };
  } catch {
    return { completedDates: [], completedChallengeIds: [] };
  }
}

function writeDailyMissionState(nextState) {
  writeStoredPreference(MISSION_STORAGE_KEY, JSON.stringify({
    completedDates: Array.isArray(nextState.completedDates) ? nextState.completedDates.slice(-45) : [],
    completedChallengeIds: Array.isArray(nextState.completedChallengeIds) ? nextState.completedChallengeIds.slice(-80) : [],
    lastMissionId: nextState.lastMissionId || null,
  }));
}

function readChallengeJobMap() {
  const raw = readStoredPreference(CHALLENGE_JOB_STORAGE_KEY);
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeChallengeJobMap(map) {
  const entries = Object.entries(map).slice(-80);
  writeStoredPreference(CHALLENGE_JOB_STORAGE_KEY, JSON.stringify(Object.fromEntries(entries)));
}

function assignChallengeToJob(jobId, challengeId) {
  if (!jobId || !challengeById(challengeId)) {
    return;
  }
  const map = readChallengeJobMap();
  map[jobId] = challengeId;
  writeChallengeJobMap(map);
}

function challengeForJob(job) {
  if (!job) {
    return activeChallenge();
  }
  const map = readChallengeJobMap();
  const mapped = challengeById(map[job.job_id]);
  if (mapped) {
    return mapped;
  }
  const roomLabel = String(job.room_label || job.summary?.room_label || '').toLowerCase();
  const mode = job.audience_mode || job.summary?.audience_mode || 'general';
  return SAFETY_MISSIONS.find((challenge) => (
    challenge.roomLabel.toLowerCase() === roomLabel
    || (challenge.audienceMode === mode && roomLabel.includes(challenge.roomLabel.toLowerCase()))
  )) || activeChallenge();
}

function challengeStreakCopy() {
  const missionState = readDailyMissionState();
  const today = localDateKey();
  const completedToday = missionState.completedDates.includes(today);
  const streak = completedToday ? dailyMissionStreak(missionState.completedDates) : 0;
  const uniqueChallenges = new Set(missionState.completedChallengeIds || []).size;
  if (completedToday) {
    return `${streak}-day streak · ${uniqueChallenges} challenge${uniqueChallenges === 1 ? '' : 's'} tried`;
  }
  if (uniqueChallenges) {
    return `${uniqueChallenges} challenge${uniqueChallenges === 1 ? '' : 's'} tried · pick one today`;
  }
  return 'No streak yet · start with one tiny win';
}

function ritualForDate(date = new Date()) {
  const dayNumber = Math.floor(new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime() / 86_400_000);
  return ROOM_RITUALS[dayNumber % ROOM_RITUALS.length];
}

function ritualById(id) {
  return ROOM_RITUALS.find((ritual) => ritual.id === id) || null;
}

function activeRitual() {
  return ritualById(state.activeRitualId) || ritualForDate();
}

function readRitualState() {
  const raw = readStoredPreference(RITUAL_STORAGE_KEY);
  if (!raw) {
    return { completedDates: [], completedRitualIds: [], lastCompletedAt: null };
  }
  try {
    const parsed = JSON.parse(raw);
    return {
      completedDates: Array.isArray(parsed.completedDates) ? parsed.completedDates.slice(-60) : [],
      completedRitualIds: Array.isArray(parsed.completedRitualIds) ? parsed.completedRitualIds.slice(-120) : [],
      lastCompletedAt: typeof parsed.lastCompletedAt === 'string' ? parsed.lastCompletedAt : null,
    };
  } catch {
    return { completedDates: [], completedRitualIds: [], lastCompletedAt: null };
  }
}

function writeRitualState(nextState) {
  writeStoredPreference(RITUAL_STORAGE_KEY, JSON.stringify({
    completedDates: Array.isArray(nextState.completedDates) ? nextState.completedDates.slice(-60) : [],
    completedRitualIds: Array.isArray(nextState.completedRitualIds) ? nextState.completedRitualIds.slice(-120) : [],
    lastCompletedAt: nextState.lastCompletedAt || null,
  }));
}

function ritualStreakCopy() {
  const ritualState = readRitualState();
  const completedDates = ritualState.completedDates || [];
  const completedToday = completedDates.includes(localDateKey());
  const streak = completedToday ? dailyMissionStreak(completedDates) : 0;
  const uniqueRituals = new Set(ritualState.completedRitualIds || []).size;
  if (completedToday) {
    return `${streak}-day ritual streak · ${uniqueRituals} routine${uniqueRituals === 1 ? '' : 's'} tried`;
  }
  if (uniqueRituals) {
    return `${uniqueRituals} routine${uniqueRituals === 1 ? '' : 's'} tried · keep it tiny today`;
  }
  return 'No ritual streak yet · start with five calm minutes';
}

function ritualIconLabel(icon) {
  const labels = {
    sun: 'SUN',
    paw: 'PAW',
    hand: 'HAND',
    key: 'KEY',
  };
  return labels[icon] || 'A0';
}

function renderRoomRituals() {
  if (!ritualTodayTitle || !ritualTodayCopy || !ritualTodayPrompt || !ritualTodayMeta) {
    return;
  }

  const ritual = activeRitual();
  const ritualState = readRitualState();
  const completedToday = ritualState.completedDates.includes(localDateKey());
  const completedIds = new Set(ritualState.completedRitualIds || []);

  ritualTodayTitle.textContent = ritual.title;
  ritualTodayCopy.textContent = ritual.copy;
  ritualTodayPrompt.textContent = ritual.prompt;
  ritualTodayMeta.innerHTML = [
    ritual.cadence,
    ritual.duration,
    ritual.season,
    CAPTURE_COACH_MODES[ritual.audienceMode]?.title || 'General room safety',
  ].map((label) => `<span class="soft-badge">${escapeHtml(label)}</span>`).join('');

  if (ritualStreakSummary) {
    ritualStreakSummary.textContent = ritualStreakCopy();
  }
  if (ritualCompleteBtn) {
    ritualCompleteBtn.disabled = completedToday;
    ritualCompleteBtn.textContent = completedToday ? 'Ritual done today' : 'Mark ritual done';
  }
  if (ritualGrid) {
    ritualGrid.innerHTML = ROOM_RITUALS.map((item) => `
      <button class="ritual-card ${item.id === ritual.id ? 'active' : ''}" type="button" data-room-ritual="${escapeHtml(item.id)}">
        <span class="ritual-icon">${escapeHtml(ritualIconLabel(item.icon))}</span>
        <span class="guide-kicker">${escapeHtml(item.cadence)} · ${escapeHtml(item.duration)}</span>
        <strong>${escapeHtml(item.title)}</strong>
        <span>${escapeHtml(item.copy)}</span>
        <span class="soft-badge">${completedIds.has(item.id) ? 'Tried locally' : item.season}</span>
      </button>
    `).join('');
  }
  if (seasonalPackGrid) {
    seasonalPackGrid.innerHTML = SEASONAL_RITUAL_PACKS.map((pack) => `
      <button class="seasonal-card" type="button" data-seasonal-pack="${escapeHtml(pack.id)}">
        <span class="guide-kicker">${escapeHtml(pack.season)}</span>
        <strong>${escapeHtml(pack.title)}</strong>
        <span>${escapeHtml(pack.copy)}</span>
      </button>
    `).join('');
  }
}

function startRoomRitual(ritual, source = 'ritual_dashboard') {
  markFirstRunStarted(source);
  state.activeRitualId = ritual.id;
  writeStoredPreference(RITUAL_SELECTION_STORAGE_KEY, ritual.id);
  if (roomLabelInput) {
    const currentLabel = roomLabelInput.value.trim();
    const isExistingRitualLabel = ROOM_RITUALS.some((item) => item.roomLabel === currentLabel);
    if (!currentLabel || isExistingRitualLabel) {
      roomLabelInput.value = ritual.roomLabel;
    }
  }
  if (audienceModeInput) {
    audienceModeInput.value = ritual.audienceMode;
  }
  renderRoomRituals();
  renderCaptureCoach();
  trackProductEvent('room_ritual_started', {
    surface: source,
    ritual_id: ritual.id,
    audience_mode: ritual.audienceMode,
    room_label: ritual.roomLabel,
    room_labeled: true,
  });
  switchView('scan');
  showToast(`${ritual.title} loaded. Scan one room, then fix one small thing.`);
}

function completeRoomRitual(ritual = activeRitual()) {
  const today = localDateKey();
  const ritualState = readRitualState();
  const completedDates = new Set(ritualState.completedDates || []);
  const completedRitualIds = new Set(ritualState.completedRitualIds || []);
  completedDates.add(today);
  completedRitualIds.add(ritual.id);
  writeRitualState({
    completedDates: [...completedDates].sort(),
    completedRitualIds: [...completedRitualIds],
    lastCompletedAt: new Date().toISOString(),
  });
  renderRoomRituals();
  trackProductEvent('room_ritual_completed', {
    ritual_id: ritual.id,
    completion_count: completedDates.size,
    audience_mode: ritual.audienceMode,
  });
  showToast(`${ritual.title} logged locally. The home-care streak lives in this browser.`);
}

function mysteryModeForDate(date = new Date()) {
  const dayNumber = Math.floor(new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime() / 86_400_000);
  return ROOM_MYSTERY_MODES[dayNumber % ROOM_MYSTERY_MODES.length];
}

function mysteryModeById(id) {
  return ROOM_MYSTERY_MODES.find((mode) => mode.id === id) || null;
}

function activeMysteryMode() {
  return mysteryModeById(state.activeMysteryModeId) || mysteryModeForDate();
}

function renderMysteryModes() {
  if (!mysteryModeGrid || !mysteryPromptCopy) {
    return;
  }
  const activeMode = activeMysteryMode();
  mysteryPromptCopy.textContent = activeMode.prompt;
  mysteryModeGrid.innerHTML = ROOM_MYSTERY_MODES.map((mode) => `
    <button class="mystery-card ${mode.id === activeMode.id ? 'active' : ''}" type="button" data-mystery-mode="${escapeHtml(mode.id)}">
      <span class="guide-kicker">${escapeHtml(CAPTURE_COACH_MODES[mode.audienceMode]?.title || 'Room discovery')}</span>
      <strong>${escapeHtml(mode.title)}</strong>
      <span>${escapeHtml(mode.prompt)}</span>
      <span class="soft-badge">Discovery prompt</span>
    </button>
  `).join('');
}

function startMysteryMode(mode, source = 'mystery_mode_grid') {
  markFirstRunStarted(source);
  state.activeMysteryModeId = mode.id;
  writeStoredPreference(MYSTERY_MODE_STORAGE_KEY, mode.id);
  if (roomLabelInput) {
    const currentLabel = roomLabelInput.value.trim();
    const isExistingMysteryLabel = ROOM_MYSTERY_MODES.some((item) => item.roomLabel === currentLabel);
    if (!currentLabel || isExistingMysteryLabel) {
      roomLabelInput.value = mode.roomLabel;
    }
  }
  if (audienceModeInput) {
    audienceModeInput.value = mode.audienceMode;
  }
  if (uploadGuidanceCopy) {
    uploadGuidanceCopy.textContent = mode.uploadHint;
  }
  renderMysteryModes();
  renderCaptureCoach();
  trackProductEvent('mystery_mode_started', {
    surface: source,
    mystery_mode_id: mode.id,
    audience_mode: mode.audienceMode,
    room_label: mode.roomLabel,
    room_labeled: true,
  });
  switchView('scan');
  showToast(`${mode.title} loaded. The report will stay evidence-backed.`);
}

function renderCuriositySampleGallery() {
  if (!curiositySampleGrid) {
    return;
  }
  curiositySampleGrid.innerHTML = CURIOSITY_SAMPLE_GALLERY.map((sample) => `
    <button class="curiosity-sample-card" type="button" data-curiosity-sample="${escapeHtml(sample.id)}">
      <span class="guide-kicker">${escapeHtml(CAPTURE_COACH_MODES[sample.audienceMode]?.title || 'Sample report')}</span>
      <strong>${escapeHtml(sample.title)}</strong>
      <span>${escapeHtml(sample.lesson)}</span>
      <div class="sample-journey-meta">
        <span><strong>Fix first:</strong> ${escapeHtml(sample.topFix || 'Review the top action and evidence.')}</span>
        <span><strong>Uncertainty:</strong> ${escapeHtml(sample.uncertainty || 'Approximate evidence, not certification.')}</span>
        <span><strong>Before/after:</strong> ${escapeHtml(sample.beforeAfter || 'Scan, fix one thing, then rescan the same room label.')}</span>
      </div>
      <span class="soft-badge">Try without upload</span>
    </button>
  `).join('');
}

function roomPlaybookById(id) {
  return ROOM_PLAYBOOKS.find((playbook) => playbook.id === id) || null;
}

function activeRoomPlaybook() {
  return roomPlaybookById(state.activeRoomPlaybookId) || null;
}

function renderWelcomeTour() {
  if (!welcomeTourCard) {
    return;
  }
  const dismissed = readStoredPreference(WELCOME_TOUR_STORAGE_KEY) === 'true';
  welcomeTourCard.classList.toggle('is-dismissed', dismissed);
  welcomeTourCard.hidden = dismissed;
}

function completeWelcomeTour() {
  writeStoredPreference(WELCOME_TOUR_STORAGE_KEY, 'true');
  renderWelcomeTour();
  trackProductEvent('welcome_tour_completed', {
    surface: 'scan_onboarding',
    playbook_id: activeRoomPlaybook()?.id || null,
  });
  showToast('Welcome tour tucked away. You can replay it from Settings.');
}

function renderRoomPlaybooks() {
  if (!roomPlaybookGrid) {
    return;
  }
  const active = activeRoomPlaybook();
  roomPlaybookGrid.innerHTML = ROOM_PLAYBOOKS.map((playbook) => `
    <button class="room-playbook-card ${active?.id === playbook.id ? 'active' : ''}" type="button" data-room-playbook="${escapeHtml(playbook.id)}">
      <span class="ritual-icon" aria-hidden="true">${escapeHtml(playbook.badge.slice(0, 2))}</span>
      <span class="guide-kicker">${escapeHtml(playbook.badge)}</span>
      <strong>${escapeHtml(playbook.title)}</strong>
      <span>${escapeHtml(playbook.prompt)}</span>
      <small>${escapeHtml(playbook.captureFocus)}</small>
    </button>
  `).join('');
}

function startRoomPlaybook(playbook, source = 'playbook_grid') {
  markFirstRunStarted(source);
  state.activeRoomPlaybookId = playbook.id;
  state.activeMysteryModeId = playbook.mysteryModeId;
  state.activeRitualId = playbook.ritualId;
  writeStoredPreference(ROOM_PLAYBOOK_STORAGE_KEY, playbook.id);
  writeStoredPreference(MYSTERY_MODE_STORAGE_KEY, playbook.mysteryModeId);
  writeStoredPreference(RITUAL_SELECTION_STORAGE_KEY, playbook.ritualId);
  if (audienceModeInput) {
    audienceModeInput.value = playbook.audienceMode;
  }
  if (roomLabelInput) {
    roomLabelInput.value = playbook.roomLabel;
  }
  if (uploadGuidanceCopy) {
    uploadGuidanceCopy.textContent = playbook.captureFocus;
  }
  renderRoomPlaybooks();
  renderMysteryModes();
  renderRoomRituals();
  renderCaptureCoach();
  trackProductEvent('room_playbook_started', {
    surface: source,
    playbook_id: playbook.id,
    audience_mode: playbook.audienceMode,
    room_label: playbook.roomLabel,
    room_labeled: true,
  });
  switchView('scan');
  showToast(`${playbook.title} loaded. Record with the same room label for cleaner before/after checks.`);
}

function reportThemeById(id) {
  return REPORT_THEMES.find((theme) => theme.id === id) || REPORT_THEMES[0];
}

function reportThemeLabel(id = currentReportTheme()) {
  return reportThemeById(id)?.title || 'Calm Brief';
}

function personalModeById(id) {
  return PERSONAL_SAFETY_MODES.find((mode) => mode.id === id) || null;
}

function renderPersonalModes() {
  if (!personalModeGrid) {
    return;
  }
  const activeId = state.activePersonalModeId;
  personalModeGrid.innerHTML = PERSONAL_SAFETY_MODES.map((mode) => `
    <button class="personal-mode-card ${mode.id === activeId ? 'active' : ''}" type="button" data-personal-mode="${escapeHtml(mode.id)}">
      <span class="guide-kicker">${escapeHtml(CAPTURE_COACH_MODES[mode.audienceMode]?.title || 'Personal mode')}</span>
      <strong>${escapeHtml(mode.title)}</strong>
      <span>${escapeHtml(mode.copy)}</span>
      <small>${escapeHtml(mode.captureFocus)}</small>
    </button>
  `).join('');
}

function startPersonalMode(mode) {
  markFirstRunStarted('personal_safety_mode');
  state.activePersonalModeId = mode.id;
  state.activeMysteryModeId = mode.mysteryModeId;
  writeStoredPreference(PERSONAL_MODE_STORAGE_KEY, mode.id);
  writeStoredPreference(MYSTERY_MODE_STORAGE_KEY, mode.mysteryModeId);
  markBingoTask('personal-mode');
  if (audienceModeInput) {
    audienceModeInput.value = mode.audienceMode;
  }
  if (roomLabelInput) {
    roomLabelInput.value = mode.roomLabel;
  }
  if (uploadGuidanceCopy) {
    uploadGuidanceCopy.textContent = mode.captureFocus;
  }
  renderPersonalModes();
  renderMysteryModes();
  renderCaptureCoach();
  renderHomeCompanion();
  trackProductEvent('personal_mode_selected', {
    personal_mode_id: mode.id,
    audience_mode: mode.audienceMode,
    room_label: mode.roomLabel,
    room_labeled: true,
  });
  switchView('scan');
  showToast(`${mode.title} loaded. The scan guidance is tuned, but the report stays honest.`);
}

function readBingoState() {
  const raw = readStoredPreference(HOME_BINGO_STORAGE_KEY);
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeBingoState(nextState) {
  writeStoredPreference(HOME_BINGO_STORAGE_KEY, JSON.stringify(nextState));
}

function markBingoTask(taskId, shouldTrack = false) {
  if (!HOME_BINGO_TASKS.some((task) => task.id === taskId)) {
    return;
  }
  const stateMap = readBingoState();
  if (!stateMap[taskId]) {
    stateMap[taskId] = new Date().toISOString();
    writeBingoState(stateMap);
    if (shouldTrack) {
      trackProductEvent('home_bingo_task_completed', { task_id: taskId });
    }
  }
}

function journalEntries() {
  return Object.values(readHomeJournal())
    .sort((a, b) => String(b.lastCheckedAt || '').localeCompare(String(a.lastCheckedAt || '')));
}

function roomPersonalityForEntry(entry = {}) {
  const haystack = [
    entry.roomLabel,
    entry.audienceLabel,
    entry.topAction,
    entry.confidenceLabel,
    entry.rescanRecommended ? 'rescan' : '',
    (entry.lastScore || 0) < 70 ? 'high watch rescan' : 'calm',
  ].join(' ').toLowerCase();
  return ROOM_PERSONALITIES.find((personality) => (
    personality.match.some((token) => haystack.includes(token))
  )) || ROOM_PERSONALITIES[0];
}

function weeklyRecapData() {
  const entries = journalEntries();
  const scoreDeltas = entries.map((entry) => {
    const scores = Array.isArray(entry.scores) ? entry.scores : [];
    return {
      entry,
      delta: scores.length >= 2 ? scores[scores.length - 1] - scores[0] : 0,
    };
  });
  const best = scoreDeltas.sort((a, b) => b.delta - a.delta)[0] || null;
  const attentionCounts = {};
  entries.forEach((entry) => {
    const key = entry.topAction || 'Review the Safety Brief';
    attentionCounts[key] = (attentionCounts[key] || 0) + 1;
  });
  const commonAttention = Object.entries(attentionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'No recurring attention area yet';
  const completedFixes = entries.reduce((sum, entry) => sum + Number(entry.completedFixes || 0), 0);
  const nextRoom = entries.find((entry) => entry.rescanRecommended)?.roomLabel
    || entries[0]?.roomLabel
    || activeMysteryMode().roomLabel;
  return {
    entries,
    roomsChecked: entries.length,
    bestRoom: best?.entry?.roomLabel || 'No room trend yet',
    bestDelta: best?.delta || 0,
    commonAttention,
    completedFixes,
    nextRoom,
  };
}

function dayOfYear(date = new Date()) {
  const start = new Date(date.getFullYear(), 0, 0);
  return Math.floor((date - start) / 86_400_000);
}

function careWeekKey(date = new Date()) {
  return `${date.getFullYear()}-w${Math.ceil(dayOfYear(date) / 7)}`;
}

function actionById(actionId) {
  return DAILY_HOME_ACTIONS.find((action) => action.id === actionId) || DAILY_HOME_ACTIONS[0];
}

function suggestedRoomLabel(fallback = 'Daily room check') {
  const entries = journalEntries();
  return entries.find((entry) => entry.rescanRecommended)?.roomLabel
    || entries[0]?.roomLabel
    || activeRitual().roomLabel
    || fallback;
}

function readJsonObject(key) {
  const raw = readStoredPreference(key);
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeJsonObject(key, value) {
  writeStoredPreference(key, JSON.stringify(value));
}

function selectDailyAction() {
  const entries = journalEntries();
  if (entries.some((entry) => entry.rescanRecommended)) {
    return actionById('rescan-needed-room');
  }
  const today = localDateKey();
  const stored = readJsonObject(DAILY_ACTION_STORAGE_KEY);
  const storedAction = DAILY_HOME_ACTIONS.find((action) => action.id === stored.actionId);
  if (stored.date === today && storedAction) {
    return storedAction;
  }
  const index = dayOfYear() % DAILY_HOME_ACTIONS.length;
  const action = DAILY_HOME_ACTIONS[index];
  writeJsonObject(DAILY_ACTION_STORAGE_KEY, { date: today, actionId: action.id, completed: false });
  return action;
}

function dailyActionState() {
  return readJsonObject(DAILY_ACTION_STORAGE_KEY);
}

function buildRoomCareWeek(force = false) {
  const weekKey = careWeekKey();
  const existing = readJsonObject(ROOM_CARE_CALENDAR_STORAGE_KEY);
  if (!force && existing.weekKey === weekKey && Array.isArray(existing.tasks)) {
    return existing;
  }
  const rooms = journalEntries();
  const seasonal = SEASONAL_RITUAL_PACKS;
  const tasks = ROOM_CARE_WEEK_TEMPLATE.map((template, index) => {
    const action = actionById(template.actionId);
    const room = rooms[index % Math.max(1, rooms.length)];
    const pack = seasonal[index % seasonal.length];
    return {
      id: `${weekKey}-${template.day.toLowerCase()}`,
      day: template.day,
      title: template.title,
      actionId: action.id,
      action: action.action,
      roomLabel: room?.roomLabel || pack?.roomLabel || action.roomLabel,
      audienceMode: room?.audienceMode || pack?.audienceMode || action.audienceMode,
    };
  });
  const next = { weekKey, generatedAt: new Date().toISOString(), cadence: currentCareCadence(), tasks };
  writeJsonObject(ROOM_CARE_CALENDAR_STORAGE_KEY, next);
  return next;
}

function readRoomCareCompleted() {
  return readJsonObject(ROOM_CARE_COMPLETED_STORAGE_KEY);
}

function fixGuideForText(text = '') {
  const value = String(text || '').toLowerCase();
  return FIX_LIBRARY_GUIDES.find((guide) => guide.match.some((token) => value.includes(token)))
    || FIX_LIBRARY_GUIDES[0];
}

function activeFixGuideId() {
  const stored = readStoredPreference(FIX_GUIDE_STORAGE_KEY);
  return FIX_LIBRARY_GUIDES.some((guide) => guide.id === stored) ? stored : FIX_LIBRARY_GUIDES[0].id;
}

function renderFixLibrary() {
  if (!fixLibraryGrid) {
    return;
  }
  const activeId = activeFixGuideId();
  fixLibraryGrid.innerHTML = FIX_LIBRARY_GUIDES.map((guide) => `
    <button class="fix-guide-card ${guide.id === activeId ? 'active' : ''}" type="button" data-open-fix-guide="${escapeHtml(guide.id)}">
      <span class="guide-kicker">${escapeHtml(guide.badge)}</span>
      <strong>${escapeHtml(guide.title)}</strong>
      <p>${escapeHtml(guide.summary)}</p>
    </button>
  `).join('');
}

function renderOneThingToday() {
  if (!oneThingTodayCard) {
    return;
  }
  const action = selectDailyAction();
  const stateMap = dailyActionState();
  const completed = stateMap.date === localDateKey() && stateMap.completed === true;
  const roomLabel = suggestedRoomLabel(action.roomLabel);
  oneThingTodayCard.innerHTML = `
    <span class="guide-kicker">One Thing Today · ${escapeHtml(action.badge)}</span>
    <h4>${escapeHtml(action.title)}</h4>
    <p>${escapeHtml(action.action)}</p>
    <p>${escapeHtml(action.reason)}</p>
    <div class="journal-meta">
      <span class="soft-badge">${escapeHtml(roomLabel)}</span>
      <span class="soft-badge">${escapeHtml(CAPTURE_COACH_MODES[action.audienceMode]?.title || 'General home safety')}</span>
      <span class="soft-badge">${completed ? 'Completed locally' : 'Local prompt'}</span>
    </div>
    <div class="daily-value-actions">
      <button class="button-link" type="button" data-start-one-thing="${escapeHtml(action.id)}">Use this prompt</button>
      <button class="button-link ghost" type="button" data-complete-one-thing="${escapeHtml(action.id)}" ${completed ? 'disabled' : ''}>${completed ? 'Done today' : 'Mark done'}</button>
    </div>
  `;
}

function renderRoomCareCalendar() {
  if (!roomCareCalendar) {
    return;
  }
  const week = buildRoomCareWeek();
  const completed = readRoomCareCompleted();
  roomCareCalendar.innerHTML = week.tasks.map((task) => `
    <article class="room-care-card ${completed[task.id] ? 'done' : ''}">
      <span class="guide-kicker">${escapeHtml(task.day)} · ${completed[task.id] ? 'Done' : currentCareCadence()}</span>
      <strong>${escapeHtml(task.title)}</strong>
      <p>${escapeHtml(task.action)}</p>
      <div class="journal-meta">
        <span class="soft-badge">${escapeHtml(task.roomLabel)}</span>
        <span class="soft-badge">${escapeHtml(CAPTURE_COACH_MODES[task.audienceMode]?.title || 'General')}</span>
      </div>
      <div class="daily-value-actions">
        <button class="button-link ghost" type="button" data-start-room-care-task="${escapeHtml(task.id)}">Use</button>
        <button class="button-link ghost" type="button" data-complete-room-care-task="${escapeHtml(task.id)}" ${completed[task.id] ? 'disabled' : ''}>${completed[task.id] ? 'Done' : 'Mark done'}</button>
      </div>
    </article>
  `).join('');
}

function regenerateRoomCareWeek(surface = 'home_companion') {
  buildRoomCareWeek(true);
  writeJsonObject(ROOM_CARE_COMPLETED_STORAGE_KEY, {});
  renderHomeCompanion();
  renderSettingsControlCenter();
  trackProductEvent('room_care_week_regenerated', { surface });
  showToast('Room Care Calendar regenerated locally.');
}

function renderRoomHealthTimeline() {
  if (!roomHealthTimelinePanel) {
    return;
  }
  const entries = journalEntries();
  if (!entries.length) {
    roomHealthTimelinePanel.innerHTML = emptyMarkup('Room Health Timeline will appear once saved rooms have local scan history.');
    return;
  }
  roomHealthTimelinePanel.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Room Health Timeline</span>
        <h4>See how each room is changing locally.</h4>
        <p>Timeline cards use saved Calm Scores, completed fixes, recurring attention areas, and next due copy from this browser.</p>
      </div>
      <button class="button-link ghost" type="button" data-room-health-timeline-open="true">Log timeline view</button>
    </div>
    <div class="room-health-timeline-grid">
      ${entries.slice(0, 6).map((entry) => {
        const scores = Array.isArray(entry.scores) ? entry.scores : [];
        const trend = scores.length >= 2 ? scores[scores.length - 1] - scores[0] : 0;
        const checked = entry.lastCheckedAt ? new Date(entry.lastCheckedAt).toLocaleDateString() : 'Recently';
        const nextDue = entry.rescanRecommended ? 'Rescan after one fix' : `${currentCareCadence()} check`;
        return `
          <article class="timeline-card">
            <span class="guide-kicker">${escapeHtml(checked)} · ${escapeHtml(nextDue)}</span>
            <strong>${escapeHtml(entry.roomLabel || 'Saved room')}</strong>
            <p>${escapeHtml(entry.topAction || 'Review the Safety Brief')}</p>
            <div class="journal-meta">
              <span class="soft-badge">${escapeHtml(entry.lastScore === null || entry.lastScore === undefined ? 'Score pending' : `${entry.lastScore}/100 Calm Score`)}</span>
              <span class="soft-badge">${escapeHtml(trend ? `${trend > 0 ? '+' : ''}${trend} trend` : 'Baseline')}</span>
              <span class="soft-badge">${escapeHtml(`${entry.completedFixes || 0} fixes`)}</span>
            </div>
          </article>
        `;
      }).join('')}
    </div>
  `;
}

function buildWeeklyRecapText() {
  const recap = weeklyRecapData();
  if (!recap.roomsChecked) {
    return 'ATLAS-0 weekly home win: I am starting with one room scan, one evidence-backed Safety Brief, and one small fix.';
  }
  const delta = recap.bestDelta ? ` (${recap.bestDelta > 0 ? '+' : ''}${recap.bestDelta} Calm Score)` : '';
  return `ATLAS-0 weekly home win: checked ${recap.roomsChecked} room${recap.roomsChecked === 1 ? '' : 's'}, logged ${recap.completedFixes} fix${recap.completedFixes === 1 ? '' : 'es'}, top attention area was "${recap.commonAttention}", and next suggested room is ${recap.nextRoom}.${delta}`;
}

function renderHomeCompanion() {
  if (!homeCompanionPanel) {
    return;
  }
  const recap = weeklyRecapData();
  renderOneThingToday();
  if (weeklyRecapCard) {
    weeklyRecapCard.innerHTML = `
      <span class="guide-kicker">Weekly Home Pulse Recap</span>
      <h4>${escapeHtml(recap.roomsChecked ? `${recap.roomsChecked} room${recap.roomsChecked === 1 ? '' : 's'} checked` : 'Start this week with one room')}</h4>
      <div class="home-pulse-grid">
        <span class="companion-stat"><strong>${escapeHtml(recap.bestDelta ? `${recap.bestDelta > 0 ? '+' : ''}${recap.bestDelta}` : '—')}</strong><small>Best Calm Score change</small></span>
        <span class="companion-stat"><strong>${escapeHtml(String(recap.completedFixes))}</strong><small>Completed local fixes</small></span>
        <span class="companion-stat"><strong>${escapeHtml(recap.nextRoom || 'Pick a room')}</strong><small>Next room</small></span>
      </div>
      <p>${escapeHtml(recap.commonAttention)}</p>
      <button class="button-link ghost" type="button" data-copy-weekly-recap="true">Copy weekly home win</button>
    `;
  }
  if (homeBingoGrid) {
    const done = readBingoState();
    homeBingoGrid.innerHTML = HOME_BINGO_TASKS.map((task) => `
      <button class="bingo-card ${done[task.id] ? 'done' : ''}" type="button" data-home-bingo-task="${escapeHtml(task.id)}">
        <span class="guide-kicker">${done[task.id] ? 'Done' : 'Try'}</span>
        <strong>${escapeHtml(task.title)}</strong>
        <span>${escapeHtml(task.copy)}</span>
      </button>
    `).join('');
  }
  renderRoomCareCalendar();
  renderFixLibrary();
  renderPersonalModes();
}

function renderRoomPersonalityPanel() {
  if (!roomPersonalityPanel) {
    return;
  }
  const entries = journalEntries();
  if (!entries.length) {
    roomPersonalityPanel.innerHTML = emptyMarkup('Room Personality Profiles will appear once you save at least one room locally.');
    return;
  }
  roomPersonalityPanel.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Room Personality Profiles</span>
        <h4>Saved rooms now have a small local identity.</h4>
        <p>Profiles are derived from room label, mode, recurring top risk, Calm Score, completed fixes, and rescan state.</p>
      </div>
    </div>
    <div class="room-personality-grid">
      ${entries.slice(0, 6).map((entry) => {
        const personality = roomPersonalityForEntry(entry);
        return `
          <article class="personality-card">
            <span class="guide-kicker">${escapeHtml(personality.badge)}</span>
            <strong>${escapeHtml(entry.roomLabel || 'Saved room')} · ${escapeHtml(personality.title)}</strong>
            <p>${escapeHtml(personality.copy)}</p>
            <p>${escapeHtml(personality.nextPrompt)}</p>
            <div class="journal-meta">
              <span class="soft-badge">${escapeHtml(entry.lastScore === null || entry.lastScore === undefined ? 'Calm Score pending' : `${entry.lastScore}/100 Calm Score`)}</span>
              <span class="soft-badge">${escapeHtml(`${entry.completedFixes || 0} fixes`)}</span>
            </div>
          </article>
        `;
      }).join('')}
    </div>
  `;
}

function renderHomePulse() {
  if (!homePulseCard) {
    return;
  }
  const entries = Object.values(readHomeJournal());
  const latest = entries
    .sort((a, b) => String(b.lastCheckedAt || '').localeCompare(String(a.lastCheckedAt || '')))[0];
  const scoreValues = entries
    .map((entry) => entry.lastScore)
    .filter((score) => typeof score === 'number');
  const calmScore = scoreValues.length
    ? Math.round(scoreValues.reduce((sum, score) => sum + score, 0) / scoreValues.length)
    : null;
  const nextRoom = latest?.rescanRecommended
    ? latest.roomLabel
    : activeMysteryMode().roomLabel || activeRitual().roomLabel;
  homePulseCard.innerHTML = `
    <div class="home-pulse-head">
      <div>
        <span class="guide-kicker">Home Pulse</span>
        <strong>${entries.length ? `${entries.length} room${entries.length === 1 ? '' : 's'} checked` : 'Start your first room pulse'}</strong>
      </div>
      <button class="button-link ghost" type="button" data-open-home-pulse="true">Open Journal</button>
    </div>
    <div class="home-pulse-grid">
      <span><strong>${escapeHtml(calmScore === null ? '—' : `${calmScore}/100`)}</strong><small>Weekly calm score</small></span>
      <span><strong>${escapeHtml(latest?.topAction || 'No room win yet')}</strong><small>Last room win</small></span>
      <span><strong>${escapeHtml(nextRoom || 'Pick a room')}</strong><small>Next suggested room</small></span>
    </div>
  `;
}

function readHomeJournal() {
  const raw = readStoredPreference(HOME_JOURNAL_STORAGE_KEY);
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeHomeJournal(journal) {
  const entries = Object.entries(journal)
    .sort(([, a], [, b]) => String(b.lastCheckedAt || '').localeCompare(String(a.lastCheckedAt || '')))
    .slice(0, 40);
  writeStoredPreference(HOME_JOURNAL_STORAGE_KEY, JSON.stringify(Object.fromEntries(entries)));
}

function readFavoriteRooms() {
  const raw = readStoredPreference(FAVORITE_ROOMS_STORAGE_KEY);
  if (!raw) {
    return new Set();
  }
  try {
    const parsed = JSON.parse(raw);
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

function writeFavoriteRooms(favorites) {
  writeStoredPreference(FAVORITE_ROOMS_STORAGE_KEY, JSON.stringify([...favorites].slice(0, 40)));
}

function roomJournalKey(job) {
  const summary = job?.summary || {};
  return String(job?.room_label || summary.room_label || job?.filename || 'Unlabeled room')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 64) || 'unlabeled-room';
}

function upsertHomeJournalFromJob(job) {
  if (!job || job.status !== 'complete') {
    return;
  }
  const summary = job.summary || {};
  const key = roomJournalKey(job);
  const journal = readHomeJournal();
  const existing = journal[key] || {};
  const score = typeof summary.room_score === 'number' ? Math.round(summary.room_score) : null;
  const previousScores = Array.isArray(existing.scores) ? existing.scores : [];
  const progress = checklistProgress(
    job,
    job.fix_first || [],
    job.recommendations || [],
    state.showLowConfidence ? job.risks || [] : (job.risks || []).filter((risk) => !isLowConfidenceRisk(risk)),
  );
  journal[key] = {
    key,
    roomLabel: job.room_label || summary.room_label || job.filename || 'Unlabeled room',
    audienceLabel: summary.audience_label || CAPTURE_COACH_MODES[job.audience_mode || 'general']?.title || 'General home safety',
    lastCheckedAt: new Date().toISOString(),
    lastJobId: job.job_id,
    sampleKey: job.sample_key || null,
    filename: job.filename || 'scan',
    topAction: (job.fix_first || [])[0]?.title || (job.recommendations || [])[0]?.title || summary.top_hazard_label || 'Review the Safety Brief',
    confidenceLabel: summary.confidence_label || summary.scan_quality_label || 'Approximate confidence',
    hazardCount: Number(summary.hazard_count || (job.risks || []).length || 0),
    completedFixes: Math.max(Number(existing.completedFixes || 0), progress.done || 0),
    scores: score === null ? previousScores.slice(-6) : [...previousScores, score].slice(-6),
    lastScore: score,
    rescanRecommended: Boolean(summary.rescan_recommended || job.scan_quality?.rescan_recommended),
    activePlaybookId: state.activeRoomPlaybookId || existing.activePlaybookId || null,
    activePersonalModeId: state.activePersonalModeId || existing.activePersonalModeId || null,
    reportThemeId: currentReportTheme(),
  };
  writeHomeJournal(journal);
}

function renderHomeJournal() {
  if (!homeJournalSummary || !homeJournalGrid || !homeJournalEmpty) {
    return;
  }
  const journal = readHomeJournal();
  const favorites = readFavoriteRooms();
  const entries = Object.values(journal)
    .sort((a, b) => {
      const favoriteDelta = Number(favorites.has(b.key)) - Number(favorites.has(a.key));
      return favoriteDelta || String(b.lastCheckedAt || '').localeCompare(String(a.lastCheckedAt || ''));
    });
  const scannedRooms = entries.length;
  const favoriteCount = entries.filter((entry) => favorites.has(entry.key)).length;
  const scoreValues = entries
    .map((entry) => entry.lastScore)
    .filter((score) => typeof score === 'number');
  const avgScore = scoreValues.length
    ? scoreValues.reduce((sum, score) => sum + score, 0) / scoreValues.length
    : null;

  homeJournalSummary.innerHTML = entries.length
    ? `
      <span class="guide-kicker">Local home rhythm</span>
      <strong>${scannedRooms} room${scannedRooms === 1 ? '' : 's'} checked locally</strong>
      <p>${escapeHtml(favoriteCount
        ? `${favoriteCount} favorite room${favoriteCount === 1 ? '' : 's'} pinned. Average Calm Score ${avgScore === null ? 'pending' : `${Math.round(avgScore)}/100`}.`
        : `Average Calm Score ${avgScore === null ? 'pending' : `${Math.round(avgScore)}/100`}. Favorite a room to make the next check faster.`)}</p>
      <div class="journal-meta">
        <span class="soft-badge">${escapeHtml(ritualStreakCopy())}</span>
        <span class="soft-badge">Local browser history</span>
        <span class="soft-badge">No account required</span>
      </div>
    `
    : `
      <strong>No rooms saved yet</strong>
      <p>Run a scan with a room label and completed reports will become local room journal entries.</p>
    `;

  homeJournalEmpty.style.display = entries.length ? 'none' : '';
  homeJournalGrid.innerHTML = entries.map((entry) => {
    const isFavorite = favorites.has(entry.key);
    const scores = Array.isArray(entry.scores) ? entry.scores : [];
    const trend = scores.length >= 2 ? scores[scores.length - 1] - scores[0] : 0;
    const checked = entry.lastCheckedAt ? new Date(entry.lastCheckedAt).toLocaleDateString() : 'Recently';
    return `
      <article class="journal-room-card" data-journal-room="${escapeHtml(entry.key)}">
        <span class="guide-kicker">${escapeHtml(isFavorite ? 'Favorite passport' : 'Room Health Passport')}</span>
        <h4 class="journal-room-title">${escapeHtml(entry.roomLabel || 'Room')}</h4>
        <p>${escapeHtml(entry.topAction || 'Review the Safety Brief')}</p>
        <div class="journal-meta">
          <span class="soft-badge">${escapeHtml(entry.lastScore === null || entry.lastScore === undefined ? 'Calm Score pending' : `${entry.lastScore}/100 Calm Score`)}</span>
          <span class="soft-badge">${escapeHtml(trend ? `${trend > 0 ? '+' : ''}${trend} trend` : 'Baseline')}</span>
          <span class="soft-badge">${escapeHtml(`${entry.hazardCount || 0} attention areas`)}</span>
          <span class="soft-badge">${escapeHtml(`${entry.completedFixes || 0} fixes`)}</span>
        </div>
        <p>${escapeHtml(`Last checked ${checked}. ${entry.rescanRecommended ? 'Rescan recommended.' : 'Ready for a calm follow-up.'}`)}</p>
        <div class="journal-actions">
          <button class="button-link ghost" type="button" data-open-journal-report="${escapeHtml(entry.lastJobId || '')}" data-sample-key="${escapeHtml(entry.sampleKey || '')}">Open report</button>
          <button class="button-link ghost" type="button" data-copy-journal-passport="${escapeHtml(entry.key)}">Copy passport</button>
          <button class="button-link ghost" type="button" data-favorite-room="${escapeHtml(entry.key)}">${isFavorite ? 'Unfavorite' : 'Favorite'}</button>
          <button class="button-link ghost" type="button" data-room-reminder="${escapeHtml(entry.key)}">Plan reminder</button>
        </div>
      </article>
    `;
  }).join('');
  renderHomePulse();
  renderSettingsControlCenter();
}

function renderDailyMission() {
  if (!dailyMissionTitle || !dailyMissionCopy || !dailyMissionSteps || !dailyMissionProgress) {
    return;
  }

  const mission = dailyMissionForDate();
  const today = localDateKey();
  const missionState = readDailyMissionState();
  const completedToday = missionState.completedDates.includes(today);
  const totalCompleted = missionState.completedDates.length;
  const streak = completedToday ? dailyMissionStreak(missionState.completedDates) : 0;

  dailyMissionTitle.textContent = mission.title;
  dailyMissionCopy.textContent = mission.copy;
  dailyMissionSteps.innerHTML = mission.steps.map((step) => (
    `<div class="mission-step">${escapeHtml(step)}</div>`
  )).join('');
  dailyMissionProgress.classList.toggle('complete', completedToday);
  dailyMissionProgress.textContent = completedToday
    ? `Mission tried today. ${streak}-day streak, ${totalCompleted} total mission${totalCompleted === 1 ? '' : 's'} logged on this browser.`
    : totalCompleted
      ? `${totalCompleted} mission${totalCompleted === 1 ? '' : 's'} logged on this browser. Try today’s one for a tiny streak.`
      : 'No missions tried yet. Start with one small room win.';

  if (dailyMissionComplete) {
    dailyMissionComplete.disabled = completedToday;
    dailyMissionComplete.textContent = completedToday ? 'Tried today' : 'Mark tried today';
  }
  if (challengeStreakSummary) {
    challengeStreakSummary.textContent = challengeStreakCopy();
  }
}

function renderChallengeLibrary() {
  if (!challengeLibraryGrid) {
    return;
  }
  const missionState = readDailyMissionState();
  const completedIds = new Set(missionState.completedChallengeIds || []);
  const currentChallenge = activeChallenge();
  challengeLibraryGrid.innerHTML = SAFETY_MISSIONS.map((challenge) => `
    <button class="challenge-card ${challenge.id === currentChallenge.id ? 'active' : ''}" type="button" data-safety-challenge="${escapeHtml(challenge.id)}">
      <span class="guide-kicker">${escapeHtml(challenge.badge || challenge.audienceMode)}</span>
      <strong>${escapeHtml(challenge.title)}</strong>
      <span>${escapeHtml(challenge.copy)}</span>
      <span class="soft-badge">${completedIds.has(challenge.id) ? 'Tried locally' : challenge.winLabel || 'Room win'}</span>
    </button>
  `).join('');
  challengeLibraryGrid.querySelectorAll('[data-safety-challenge]').forEach((button) => {
    button.addEventListener('click', () => {
      const challenge = challengeById(button.dataset.safetyChallenge);
      if (challenge) {
        startChallenge(challenge, 'challenge_library');
      }
    });
  });
}

function startChallenge(challenge, source = 'daily_card') {
  markFirstRunStarted(source);
  state.activeChallengeId = challenge.id;
  state.pendingUploadChallengeId = challenge.id;
  writeStoredPreference(CHALLENGE_SELECTION_STORAGE_KEY, challenge.id);
  if (roomLabelInput) {
    const currentLabel = roomLabelInput.value.trim();
    const isExistingChallengeLabel = SAFETY_MISSIONS.some((item) => item.roomLabel === currentLabel);
    if (!currentLabel || isExistingChallengeLabel) {
      roomLabelInput.value = challenge.roomLabel;
    }
  }
  if (audienceModeInput) {
    audienceModeInput.value = challenge.audienceMode;
  }
  renderCaptureCoach();
  renderDailyMission();
  renderChallengeLibrary();
  if (uploadGuidanceCopy) {
    uploadGuidanceCopy.textContent = challenge.uploadHint;
  }
  trackProductEvent('daily_mission_started', {
    mission_id: challenge.id,
    challenge_id: challenge.id,
    surface: source,
    audience_mode: challenge.audienceMode,
  });
  switchView('scan');
  showToast(`${challenge.title} loaded. Record one room when you are ready.`);
}

function startDailyMission() {
  startChallenge(dailyMissionForDate(), 'daily_mission_card');
}

function completeDailyMission(challenge = activeChallenge()) {
  const today = localDateKey();
  const missionState = readDailyMissionState();
  const completedDates = new Set(missionState.completedDates);
  const completedChallengeIds = new Set(missionState.completedChallengeIds || []);
  completedDates.add(today);
  completedChallengeIds.add(challenge.id);
  writeDailyMissionState({
    completedDates: [...completedDates].sort(),
    completedChallengeIds: [...completedChallengeIds],
    lastMissionId: challenge.id,
  });
  renderDailyMission();
  renderChallengeLibrary();
  trackProductEvent('daily_mission_completed', {
    mission_id: challenge.id,
    challenge_id: challenge.id,
    completion_count: completedDates.size,
  });
  trackProductEvent('weekly_challenge_completed', {
    surface: 'home_companion',
    mission_id: challenge.id,
    challenge_id: challenge.id,
    completion_count: completedChallengeIds.size,
  });
  showToast(`${challenge.title} logged locally. Tiny room win counted.`);
}

function selectedAudienceMode() {
  const mode = audienceModeInput?.value || 'general';
  return CAPTURE_COACH_MODES[mode] ? mode : 'general';
}

function captureCoachStorageKey(mode = selectedAudienceMode()) {
  return `${CAPTURE_COACH_STORAGE_KEY}.${localDateKey()}.${mode}`;
}

function readCaptureCoachState(mode = selectedAudienceMode()) {
  const raw = readStoredPreference(captureCoachStorageKey(mode));
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeCaptureCoachState(mode, nextState) {
  writeStoredPreference(captureCoachStorageKey(mode), JSON.stringify(nextState));
}

function renderCaptureCoach() {
  if (!captureCoachTitle || !captureCoachCopy || !captureCoachRoute || !captureCoachChecks) {
    return;
  }

  const mode = selectedAudienceMode();
  const coach = CAPTURE_COACH_MODES[mode] || CAPTURE_COACH_MODES.general;
  const coachState = readCaptureCoachState(mode);
  const checkedCount = coach.checks.filter((check) => coachState[check]).length;
  const progress = coach.checks.length ? Math.round((checkedCount / coach.checks.length) * 100) : 0;

  captureCoachTitle.textContent = coach.title;
  captureCoachCopy.textContent = coach.promise;
  captureCoachRoute.innerHTML = coach.route
    .map((step) => `<li>${escapeHtml(step)}</li>`)
    .join('');
  captureCoachChecks.innerHTML = coach.checks
    .map((check) => `
      <li class="coach-check">
        <input type="checkbox" data-capture-check="${escapeHtml(check)}" ${coachState[check] ? 'checked' : ''} />
        <span>${escapeHtml(check)}</span>
      </li>
    `)
    .join('');
  if (captureCoachPrompt) {
    captureCoachPrompt.textContent = coach.funPrompt;
  }
  if (captureCoachStatus) {
    captureCoachStatus.textContent = `${checkedCount} of ${coach.checks.length} ready`;
  }
  if (captureCoachMeter) {
    captureCoachMeter.style.width = `${progress}%`;
  }
  useCaseButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.useCaseMode === mode);
  });
  if (scanWizardStatus) {
    scanWizardStatus.textContent = progress >= 80
      ? 'Great capture setup. Upload when your walkthrough is ready, then use the report as decision support, not certification.'
      : `${checkedCount} of ${coach.checks.length} capture checks are ready. Finish the checklist to improve report quality before uploading.`;
  }
}

function renderTrustProofDashboard() {
  if (!trustProofMetrics || !trustProofDetails) {
    return;
  }
  const proof = state.trustProof;
  if (!proof) {
    trustProofMetrics.innerHTML = `
      <article class="proof-metric"><strong>—</strong><span>Completed scans</span></article>
      <article class="proof-metric"><strong>—</strong><span>Evidence-backed reports</span></article>
      <article class="proof-metric"><strong>—</strong><span>Weak scans flagged</span></article>
    `;
    trustProofDetails.innerHTML = emptyMarkup('Trust proof signals are unavailable right now. The product still shows scan-level trust notes inside each report.');
    return;
  }
  trustProofMetrics.innerHTML = `
    <article class="proof-metric"><strong>${Number(proof.completed_scans || 0)}</strong><span>Completed scans</span></article>
    <article class="proof-metric"><strong>${Number(proof.evidence_backed_reports || 0)}</strong><span>Evidence-backed reports</span></article>
    <article class="proof-metric"><strong>${Number(proof.rejected_or_downgraded_scans || 0)}</strong><span>Weak scans flagged</span></article>
  `;
  const proofPoints = Array.isArray(proof.proof_points) ? proof.proof_points : [];
  const knownLimits = Array.isArray(proof.known_limits) ? proof.known_limits : [];
  const privacyNotes = Array.isArray(proof.privacy_notes) ? proof.privacy_notes : [];
  trustProofDetails.innerHTML = `
    ${proofPoints.length ? `<p class="subsection-label">Quality signals</p>${renderPolicyItems(proofPoints.map((item) => ({ label: item.label, value: item.value })))}` : ''}
    ${knownLimits.length ? `<p class="subsection-label">Known limits</p><ul class="settings-list">${knownLimits.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}
    ${privacyNotes.length ? `<p class="subsection-label">Privacy posture</p><ul class="settings-list">${privacyNotes.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}
  `;
}

async function startLiveCaptureCoach() {
  if (!navigator.mediaDevices?.getUserMedia || !liveCaptureVideo || !liveCaptureCanvas) {
    if (liveCaptureGuidance) {
      liveCaptureGuidance.textContent = 'This browser does not expose camera preview APIs. Use the checklist and upload preflight instead.';
    }
    showToast('Live camera preview is unavailable in this browser.', 3400);
    return;
  }
  try {
    liveCaptureStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    liveCaptureVideo.srcObject = liveCaptureStream;
    await liveCaptureVideo.play();
    liveCaptureStartedAt = Date.now();
    liveCaptureLastFrame = null;
    liveCaptureCoverageCells = new Set();
    liveCapturePreview?.classList.add('active');
    if (liveCaptureStartBtn) liveCaptureStartBtn.disabled = true;
    if (liveCaptureStopBtn) liveCaptureStopBtn.disabled = false;
    if (liveCaptureStatus) liveCaptureStatus.textContent = 'Checking quality';
    writeStoredPreference(LIVE_CAPTURE_COACH_STORAGE_KEY, 'started');
    trackProductEvent('live_capture_coach_started', { surface: 'guided_scan_wizard' });
    liveCaptureTimer = window.setInterval(updateLiveCaptureQuality, 900);
    updateLiveCaptureQuality();
  } catch (error) {
    if (liveCaptureGuidance) {
      liveCaptureGuidance.textContent = 'Camera permission was blocked or unavailable. No problem: use the normal upload checklist and file preflight.';
    }
    showToast(error instanceof Error ? error.message : 'Could not start live capture coach.', 3600);
  }
}

function stopLiveCaptureCoach() {
  if (liveCaptureTimer) {
    window.clearInterval(liveCaptureTimer);
    liveCaptureTimer = null;
  }
  if (liveCaptureStream) {
    liveCaptureStream.getTracks().forEach((track) => track.stop());
    liveCaptureStream = null;
  }
  if (liveCaptureVideo) {
    liveCaptureVideo.srcObject = null;
  }
  liveCapturePreview?.classList.remove('active');
  if (liveCaptureStartBtn) liveCaptureStartBtn.disabled = false;
  if (liveCaptureStopBtn) liveCaptureStopBtn.disabled = true;
  if (liveCaptureStatus) liveCaptureStatus.textContent = 'Camera off';
}

function updateLiveCaptureQuality() {
  if (!liveCaptureVideo || !liveCaptureCanvas || !liveCaptureStats || liveCaptureVideo.readyState < 2) {
    return;
  }
  const context = liveCaptureCanvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    return;
  }
  context.drawImage(liveCaptureVideo, 0, 0, liveCaptureCanvas.width, liveCaptureCanvas.height);
  const data = context.getImageData(0, 0, liveCaptureCanvas.width, liveCaptureCanvas.height).data;
  const coverage = analyzeLiveCaptureCoverage(data, liveCaptureCanvas.width, liveCaptureCanvas.height);
  coverage.cells.forEach((cell) => liveCaptureCoverageCells.add(cell));
  let brightness = 0;
  let diff = 0;
  for (let index = 0; index < data.length; index += 4) {
    const luminance = (data[index] + data[index + 1] + data[index + 2]) / 3;
    brightness += luminance;
    if (liveCaptureLastFrame) {
      diff += Math.abs(luminance - liveCaptureLastFrame[index / 4]);
    }
  }
  const pixels = data.length / 4;
  const brightnessScore = brightness / pixels;
  const motionScore = liveCaptureLastFrame ? diff / pixels : 0;
  liveCaptureLastFrame = Array.from({ length: pixels }, (_, index) => {
    const offset = index * 4;
    return (data[offset] + data[offset + 1] + data[offset + 2]) / 3;
  });
  const elapsed = Math.round((Date.now() - liveCaptureStartedAt) / 1000);
  const lightLabel = brightnessScore < 70 ? 'Too dark' : brightnessScore > 215 ? 'Too bright' : 'Good';
  const motionLabel = motionScore > 28 ? 'Move slower' : elapsed < 3 ? 'Starting' : 'Steady';
  const coveragePercent = Math.round((liveCaptureCoverageCells.size / 9) * 100);
  const aggregateCoverage = {
    floorVisible: [...liveCaptureCoverageCells].some((cell) => cell.startsWith('2-')),
    cornerVisible: ['0-0', '0-2', '2-0', '2-2'].some((cell) => liveCaptureCoverageCells.has(cell)),
  };
  const coverageLabel = liveCoverageLabel(coveragePercent, aggregateCoverage);
  const durationLabel = elapsed >= 20 ? 'Upload-ready' : `${elapsed}s`;
  liveCaptureStats.innerHTML = `
    <article class="live-coach-stat"><strong>${escapeHtml(lightLabel)}</strong><span>${Math.round(brightnessScore)} brightness score</span></article>
    <article class="live-coach-stat"><strong>${escapeHtml(motionLabel)}</strong><span>${Math.round(motionScore)} motion delta</span></article>
    <article class="live-coach-stat"><strong>${escapeHtml(coverageLabel)}</strong><span>${coveragePercent}% room coverage cue</span></article>
    <article class="live-coach-stat"><strong>${escapeHtml(durationLabel)}</strong><span>Target 20-60 seconds</span></article>
  `;
  if (liveCaptureGuidance) {
    liveCaptureGuidance.textContent = liveCaptureGuidanceText(lightLabel, motionLabel, coverageLabel, elapsed, coveragePercent);
  }
  if (liveCaptureStatus) {
    liveCaptureStatus.textContent = lightLabel === 'Good' && motionLabel === 'Steady' && coveragePercent >= 66 && elapsed >= 10
      ? 'Looks usable'
      : 'Keep improving';
  }
  if (Date.now() - liveCaptureLastEventAt > 8000) {
    liveCaptureLastEventAt = Date.now();
    trackProductEvent('live_capture_quality_checked', {
      surface: 'guided_scan_wizard',
      reason: `${lightLabel}; ${motionLabel}; ${coverageLabel}; ${elapsed}s`,
      coverage_percent: coveragePercent,
    });
  }
}

function analyzeLiveCaptureCoverage(data, width, height) {
  const grid = 3;
  const cellWidth = Math.max(1, Math.floor(width / grid));
  const cellHeight = Math.max(1, Math.floor(height / grid));
  const cells = [];
  let floorVisible = false;
  let cornerVisible = false;

  for (let row = 0; row < grid; row += 1) {
    for (let col = 0; col < grid; col += 1) {
      let luminance = 0;
      let contrast = 0;
      let samples = 0;
      const startX = col * cellWidth;
      const startY = row * cellHeight;
      const endX = Math.min(width, startX + cellWidth);
      const endY = Math.min(height, startY + cellHeight);
      for (let y = startY; y < endY; y += 6) {
        let previous = null;
        for (let x = startX; x < endX; x += 6) {
          const offset = (y * width + x) * 4;
          const value = (data[offset] + data[offset + 1] + data[offset + 2]) / 3;
          luminance += value;
          if (previous !== null) {
            contrast += Math.abs(value - previous);
          }
          previous = value;
          samples += 1;
        }
      }
      const avg = samples ? luminance / samples : 0;
      const edge = samples ? contrast / samples : 0;
      if (avg > 45 && avg < 235 && edge > 2.5) {
        cells.push(`${row}-${col}`);
        if (row === 2) floorVisible = true;
        if ((row === 0 || row === 2) && (col === 0 || col === 2)) cornerVisible = true;
      }
    }
  }

  return { cells, floorVisible, cornerVisible };
}

function liveCoverageLabel(coveragePercent, coverage) {
  if (!coverage.floorVisible) return 'Add floor path';
  if (!coverage.cornerVisible) return 'Find corners';
  if (coveragePercent < 55) return 'Turn once more';
  if (coveragePercent < 78) return 'Nearly covered';
  return 'Good coverage';
}

function liveCaptureGuidanceText(lightLabel, motionLabel, coverageLabel, elapsed, coveragePercent) {
  if (lightLabel !== 'Good') {
    return lightLabel === 'Too dark'
      ? 'Turn on room lights or face away from bright windows before recording.'
      : 'Avoid pointing directly at bright windows or lamps; let the room surfaces stay visible.';
  }
  if (motionLabel === 'Move slower') {
    return 'Move more slowly and pause on corners, floor paths, shelves, and reachable objects.';
  }
  if (coverageLabel === 'Add floor path') {
    return 'Tilt down briefly so ATLAS-0 sees the walking path, rug edges, cords, and low obstacles.';
  }
  if (coverageLabel === 'Find corners') {
    return 'Pause on at least two room corners so the scan has a stronger sense of the room boundary.';
  }
  if (coveragePercent < 66) {
    return 'Turn once more across shelves, floor path, doorway, and reachable surfaces before uploading.';
  }
  if (elapsed < 20) {
    return 'Quality looks usable. Keep recording until at least 20 seconds so ATLAS-0 sees the full room route.';
  }
  return 'Good preflight. Upload the recorded walkthrough when you are ready; this preview was not uploaded.';
}

function updateCaptureCoachCheck(check, enabled) {
  const mode = selectedAudienceMode();
  const nextState = readCaptureCoachState(mode);
  nextState[check] = Boolean(enabled);
  writeCaptureCoachState(mode, nextState);
  renderCaptureCoach();
  trackProductEvent('capture_coach_checked', {
    audience_mode: mode,
    check,
    checked: Boolean(enabled),
  });
}

function applyUseCase(mode, label) {
  markFirstRunStarted('use_case_card');
  if (audienceModeInput && CAPTURE_COACH_MODES[mode]) {
    audienceModeInput.value = mode;
  }
  if (roomLabelInput && label && !roomLabelInput.value.trim()) {
    roomLabelInput.value = label;
  }
  renderCaptureCoach();
  trackProductEvent('capture_mode_changed', {
    surface: 'use_case_card',
    audience_mode: selectedAudienceMode(),
    room_label: roomLabelInput?.value?.trim() || null,
    room_labeled: Boolean(roomLabelInput?.value?.trim()),
  });
  trackProductEvent('beta_onboarding_started', {
    surface: 'use_case_card',
    persona: betaPersona(),
    use_case: label || mode,
  });
  switchView('scan');
  showToast(`${CAPTURE_COACH_MODES[selectedAudienceMode()].title} selected.`);
}

function initLandingSectionTracking() {
  const sections = [...document.querySelectorAll('[data-landing-section]')];
  if (!sections.length || !('IntersectionObserver' in window)) {
    return;
  }

  const seen = new Set();
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }
      const section = entry.target.getAttribute('data-landing-section');
      if (!section || seen.has(section)) {
        return;
      }
      seen.add(section);
      trackProductEvent('landing_section_viewed', {
        surface: section,
      });
      if (section === 'trust_dashboard') {
        trackProductEvent('trust_dashboard_opened', { surface: section });
      }
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.55 });

  sections.forEach((section) => observer.observe(section));
}

function switchView(id) {
  state.activeView = id;
  navButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.view === id);
  });
  viewElements.forEach((element) => {
    element.classList.toggle('active', element.id === `view-${id}`);
  });
  viewLabel.textContent = VIEW_LABELS[id] || id;

  if (id === 'scene') {
    ensureScene();
    sceneViewer.refresh();
  }
  if (id === 'journal') {
    renderHomeJournal();
    trackProductEvent('home_journal_opened', { surface: 'journal_nav' });
    trackProductEvent('room_personality_viewed', {
      surface: 'journal_nav',
      room_count: journalEntries().length,
    });
  }
  syncUrlState();
}

function ensureScene() {
  if (!sceneReady) {
    sceneViewer.init();
    sceneReady = true;
  }
}

function activeJob() {
  return state.activeJobId ? state.jobs.get(state.activeJobId) : null;
}

function setActiveJob(jobId) {
  state.activeJobId = jobId;
  state.activeSampleKey = state.jobs.get(jobId)?.sample_key || null;
  renderUploads();
  renderProcessing(activeJob());
  renderReport(activeJob());
  renderHomeJournal();
  renderSettingsControlCenter();
  syncUrlState();
}

function upsertJob(job) {
  state.jobs.set(job.job_id, job);
  if (!state.activeJobId || job.job_id === state.activeJobId || job.status === 'complete') {
    state.activeJobId = job.job_id;
  }

  renderUploads();
  renderProcessing(job);

  if (job.status === 'complete') {
    upsertHomeJournalFromJob(job);
    renderHomeJournal();
    renderReport(job);
    switchView('report');
    showToast(job.is_sample ? 'Sample report ready.' : 'Scan complete. Report is ready.');
  } else if (job.status === 'error') {
    showToast(job.error || 'Scan failed.', 3600);
  }
}

function removeJob(jobId) {
  state.jobs.delete(jobId);
  if (state.activeJobId === jobId) {
    const nextJob = [...state.jobs.values()].sort((a, b) => b.job_id.localeCompare(a.job_id))[0] || null;
    state.activeJobId = nextJob?.job_id || null;
  }
  renderUploads();
  renderProcessing(activeJob());
  renderReport(activeJob());
  renderHomeJournal();
  renderSettingsControlCenter();
}

function renderUploads() {
  const jobs = [...state.jobs.values()]
    .filter((job) => !job.is_sample)
    .sort((a, b) => b.job_id.localeCompare(a.job_id));
  recentEmpty.style.display = jobs.length ? 'none' : '';
  recentEmpty.textContent = jobs.length
    ? ''
    : 'No scans yet. Start with one room you know well, keep the camera steady, and the first report will appear here.';

  recentList.innerHTML = jobs.map((job) => {
    const summary = job.summary || {};
    const hazardLabel = summary.top_hazard_label || 'No hazards yet';
    const roomLabel = job.room_label || summary.room_label || '';
    const roomScore = typeof summary.room_score === 'number' ? `${summary.room_score}/100 room score` : '';
    const audienceLabel = summary.audience_label || '';
    const activeClass = job.job_id === state.activeJobId ? 'active' : '';
    return `
      <button class="scan-card ${activeClass}" data-job-id="${job.job_id}">
        <div class="scan-card-top">
          <span class="scan-file">${escapeHtml(job.filename)}</span>
          <span class="scan-pill ${job.status}">${escapeHtml(job.status)}</span>
        </div>
        <div class="scan-card-meta">
          <span>${Math.round((job.progress || 0) * 100)}%</span>
          <span>${escapeHtml(job.stage || 'queued')}</span>
          <span>${escapeHtml(roomLabel || hazardLabel)}</span>
          ${audienceLabel ? `<span>${escapeHtml(audienceLabel)}</span>` : ''}
          ${roomScore ? `<span>${escapeHtml(roomScore)}</span>` : ''}
        </div>
      </button>
    `;
  }).join('');

  recentList.querySelectorAll('.scan-card').forEach((button) => {
    button.addEventListener('click', () => {
      setActiveJob(button.dataset.jobId);
      if (activeJob()?.status === 'complete') {
        switchView('report');
      }
    });
  });
}

function renderProcessing(job) {
  if (!job) {
    processStage.textContent = 'Ready for upload';
    processCopy.textContent = 'Upload a short walkthrough and ATLAS-0 will build a calmer, evidence-backed room safety report.';
    processBar.style.width = '0%';
    processMeta.textContent = 'No active scan yet';
    processGuidance.innerHTML = renderProcessGuidance(null);
    uploadStatus.textContent = 'Ready for first scan';
    reportStatus.textContent = 'No report yet';
    return;
  }

  if (job.is_sample) {
    processStage.textContent = 'Sample report loaded';
    processCopy.textContent = 'The built-in walkthrough is open so you can explore the report before recording your own room.';
    processBar.style.width = '100%';
    processMeta.textContent = `${escapeHtml(job.room_label || 'Sample room')} · ${escapeHtml(job.summary?.audience_label || 'General home safety')} · sample`;
    processGuidance.innerHTML = `
      <article class="guidance-card">
        <strong>Use this as a reference</strong>
        <p>This sample shows the tone, evidence, and follow-through structure we want first-time users to understand quickly.</p>
      </article>
      <article class="guidance-card">
        <strong>What to do next</strong>
        <p>When you are ready, switch back to Scan and record one real room with the same calm, steady walkthrough style.</p>
      </article>
    `;
    uploadStatus.textContent = 'Sample report loaded';
    reportStatus.textContent = 'Sample report ready';
    return;
  }

  const statusLabel = `${capitalize(job.stage || 'upload')} · ${Math.round((job.progress || 0) * 100)}%`;
  processStage.textContent = statusLabel;
  processBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;

  if (job.status === 'complete') {
    processCopy.textContent = 'Your report is ready with top hazards, evidence frames, and practical next steps.';
  } else if (job.status === 'error') {
    processCopy.textContent = job.error || 'The scan could not be processed.';
  } else {
    processCopy.textContent = 'ATLAS-0 is analyzing the upload, grounding the findings, and assembling the report.';
  }

  processMeta.textContent = `${escapeHtml(job.filename)} · ${escapeHtml(job.room_label || 'Unlabeled room')} · ${escapeHtml(job.summary?.audience_label || 'General home safety')} · ${escapeHtml(job.status)}`;
  processGuidance.innerHTML = renderProcessGuidance(job);
  uploadStatus.textContent = job.status === 'complete' ? 'Latest scan complete' : 'Scan in progress';
  reportStatus.textContent = job.status === 'complete' ? 'Report ready' : 'Waiting for report';
}

function renderReport(job) {
  syncLowConfidenceControls();
  if (!job || job.status !== 'complete') {
    reportHero.classList.add('empty');
    reportHeadline.textContent = 'No report yet';
    reportSubhead.textContent = 'Run one room scan to review hazards, evidence, and practical next steps in a single place.';
    reportHeroMeta.innerHTML = `
      <span class="soft-badge">Action-first</span>
      <span class="soft-badge">Evidence-backed</span>
      <span class="soft-badge">Shareable PDF</span>
    `;
    renderBriefExecutive(null);
    renderBriefTriage(null);
    renderFieldNotes(null);
    renderRoomMapPreview(null);
    renderBeforeAfterStory(null);
    renderShareCardPreview(null);
    renderRoomPassport(null);
    renderFixVerification(null);
    renderReportThemePanel(null);
    renderFixQuestPanel(null);
    renderRoomComparePanel(null);
    renderSmartRescanCoach(null);
    renderEvidenceStoryPanel(null);
    renderReportQuestionPanel(null);
    renderPrivacyReceipt(null);
    summaryObjects.textContent = '0';
    summaryHazards.textContent = '0';
    summarySeverity.textContent = '—';
    summaryConfidence.textContent = '—';
    summaryCoverage.textContent = '—';
    summarySource.textContent = '—';
    fixFirstList.innerHTML = emptyMarkup('Priority actions will appear here once a scan finishes.');
    scanQualityCard.innerHTML = emptyMarkup('Scan quality diagnostics will appear here after the upload is processed.');
    reportPostureCard.innerHTML = emptyMarkup('Report posture details will appear here after a completed scan.');
    reportEvalCard.innerHTML = emptyMarkup('Feedback and review coverage will appear here after a completed scan.');
    weekendFixList.innerHTML = emptyMarkup('Weekend-friendly fixes will appear here after a completed scan.');
    roomWinsList.innerHTML = emptyMarkup('Positive scan signals will appear here after a completed scan.');
    roomScorecard.innerHTML = emptyMarkup('Calm Score card will appear after a completed scan.');
    renderChallengeResultCard(null);
    reportActionLoop.innerHTML = emptyMarkup('The fix-and-rescan loop will appear after a completed scan.');
    fixChecklistList.innerHTML = emptyMarkup('Checklist items will appear after a completed scan.');
    reportHazards.innerHTML = emptyMarkup('Hazards will appear here once ATLAS-0 has something evidence-backed to show.');
    reportRecommendations.innerHTML = emptyMarkup('Recommendations will appear here after a completed scan.');
    reportEvidence.innerHTML = emptyMarkup('Evidence frames will appear here after a completed scan.');
    evidenceTimeline.innerHTML = '';
    trustNotes.innerHTML = emptyMarkup('Trust notes will appear here after a completed scan.');
    findingToggleNote.textContent = 'Low-confidence findings stay hidden until a report is ready.';
    shareLinkNote.textContent = 'Share links open the exact report view. Hosted environments still respect token protection.';
    exportPdfBtn.removeAttribute('href');
    exportPdfBtn.classList.add('disabled');
    copyShareBtn.classList.add('disabled');
    copyShareBtn.disabled = true;
    copyShareCardBtn?.classList.add('disabled');
    if (copyShareCardBtn) {
      copyShareCardBtn.disabled = true;
    }
    deleteJobBtn.classList.add('disabled');
    deleteJobBtn.disabled = true;
    return;
  }

  reportHero.classList.remove('empty');
  const summary = job.summary || {};
  const hazards = job.risks || [];
  const visibleHazards = state.showLowConfidence ? hazards : hazards.filter((risk) => !isLowConfidenceRisk(risk));
  const hiddenCount = Math.max(0, hazards.length - visibleHazards.length);
  const fixFirst = job.fix_first || [];
  const recommendations = job.recommendations || [];
  const evidence = job.evidence_frames || [];
  const scanQuality = job.scan_quality || {};
  const notes = job.trust_notes || [];
  const evaluation = job.evaluation_summary || {};
  const resolution = job.resolution_summary || {};
  const comparison = job.room_comparison || null;
  const roomLabel = job.room_label || summary.room_label || '';
  const audienceLabel = summary.audience_label || 'General home safety';
  const weekendFixes = job.weekend_fix_list || [];
  const roomWins = job.room_wins || [];
  const viewKey = `${job.sample_key || 'job'}:${job.job_id}`;
  if (!state.reportViewEvents.has(viewKey)) {
    state.reportViewEvents.add(viewKey);
    trackProductEvent('report_viewed', {
      surface: 'report_view',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      audience_mode: job.audience_mode || 'general',
      room_labeled: Boolean(roomLabel),
    });
  }

  reportHeadline.textContent = summary.headline || (summary.top_hazard_label
    ? `Top concern: ${summary.top_hazard_label}`
    : 'Hazard report');
  reportSubhead.textContent = summary.overview || `${roomLabel || job.filename} · ${audienceLabel} · ${summary.confidence_label || 'Approximate grounding'}`;
  reportHeroMeta.innerHTML = renderHeroBadges(summary, roomLabel, comparison, resolution, job.is_sample);
  renderBriefExecutive(job, summary, visibleHazards, fixFirst, recommendations, scanQuality);
  renderBriefTriage(job, summary, visibleHazards, fixFirst, recommendations, scanQuality);
  renderFieldNotes(job, summary, visibleHazards, fixFirst, recommendations, scanQuality);
  renderRoomMapPreview(job, summary, visibleHazards, evidence);
  renderBeforeAfterStory(job, summary, visibleHazards, fixFirst, recommendations, comparison);
  renderShareCardPreview(job);
  renderRoomPassport(job, summary, visibleHazards, fixFirst, comparison);
  renderFixVerification(job, summary, visibleHazards, fixFirst, recommendations, comparison);
  renderReportThemePanel(job);
  renderFixQuestPanel(job, summary, visibleHazards, fixFirst, recommendations);
  renderRoomComparePanel(job, summary, visibleHazards, comparison);
  renderSmartRescanCoach(job, summary, visibleHazards, fixFirst, recommendations, scanQuality, comparison);
  renderEvidenceStoryPanel(job, summary, visibleHazards, evidence, scanQuality);
  renderReportQuestionPanel(job, summary, visibleHazards, fixFirst, recommendations, evidence, scanQuality);
  renderPrivacyReceipt(job, summary, evidence);

  summaryObjects.textContent = String(summary.object_count || 0);
  summaryHazards.textContent = String(summary.hazard_count || 0);
  summarySeverity.textContent = capitalize(summary.top_severity || 'none');
  summaryConfidence.textContent = summary.scan_quality_label
    ? `${summary.confidence_label || 'Approximate grounding'} · ${summary.scan_quality_label} scan`
    : summary.confidence_label || 'Approximate grounding';
  summaryCoverage.textContent = capitalize(summary.coverage_label || 'unknown');
  summarySource.textContent = summary.scene_source || 'unknown';
  findingToggleNote.textContent = hiddenCount > 0 && !state.showLowConfidence
    ? `${hiddenCount} lower-confidence finding${hiddenCount === 1 ? '' : 's'} hidden to keep the report focused.`
    : 'Showing every finding, including weak or approximate ones.';

  roomScorecard.innerHTML = renderRoomScorecard(job, summary, visibleHazards, fixFirst, recommendations, comparison, scanQuality);
  renderChallengeResultCard(job, summary, visibleHazards, fixFirst, recommendations, comparison);
  reportActionLoop.innerHTML = renderReportActionLoop(job, summary, visibleHazards, fixFirst, recommendations, evidence, comparison);

  fixFirstList.innerHTML = fixFirst.length
    ? fixFirst.map((action, index) => `
        <article class="fix-first-card">
          <div class="report-card-top">
            <h3>${index + 1}. ${escapeHtml(action.title || 'Fix first')}</h3>
            <span class="severity-pill ${action.severity || 'low'}">${escapeHtml(action.severity || 'low')}</span>
          </div>
          <div class="report-copy-block">
            <strong>What to do next</strong>
            <span>${escapeHtml(action.action || '')}</span>
          </div>
          <div class="report-copy-block">
            <strong>Why this moved to the top</strong>
            <span>${escapeHtml(action.why || '')}</span>
          </div>
          <div class="report-card-meta">
            <span>${escapeHtml(action.location || 'scan area')}</span>
            <span>${escapeHtml(action.confidence_label || 'weak')} evidence</span>
            <span>${escapeHtml(fixDifficultyLabel(action, index))}</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No high-priority actions were generated for this scan.');

  scanQualityCard.innerHTML = renderScanQuality(scanQuality);
  reportPostureCard.innerHTML = renderReportPosture(summary, scanQuality, job);
  reportEvalCard.innerHTML = renderEvaluationSummary(evaluation, job);
  weekendFixList.innerHTML = weekendFixes.length
    ? weekendFixes.map((item, index) => `
        <article class="report-card recommendation">
          <div class="report-card-top">
            <h3>${index + 1}. ${escapeHtml(item.title || 'Weekend fix')}</h3>
            <span class="severity-pill medium">${escapeHtml(item.effort || '20-30 minutes')}</span>
          </div>
          <div class="report-copy-block">
            <strong>Quick task</strong>
            <span>${escapeHtml(item.task || '')}</span>
          </div>
          <div class="report-copy-block">
            <strong>Why it helps</strong>
            <span>${escapeHtml(item.benefit || '')}</span>
          </div>
          <div class="report-card-meta">
            <span>${escapeHtml(item.location || 'scan area')}</span>
            <span>${escapeHtml(item.audience_label || audienceLabel)}</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No weekend-friendly fixes were generated for this scan.');
  roomWinsList.innerHTML = roomWins.length
    ? roomWins.map((win) => `
        <article class="report-card recommendation">
          <div class="report-card-top">
            <h3>${escapeHtml(win.title || 'Positive sign')}</h3>
            <span class="severity-pill low">Calm signal</span>
          </div>
          <p>${escapeHtml(win.detail || '')}</p>
        </article>
      `).join('')
    : emptyMarkup('No positive scan signals were generated for this report.');

  reportHazards.innerHTML = visibleHazards.length
    ? visibleHazards.map((risk) => `
        <article class="report-card hazard ${risk.severity || 'low'}">
          <div class="report-card-top">
            <h3>${escapeHtml(risk.hazard_title || risk.object_label || 'Object')}</h3>
            <span class="severity-pill ${risk.severity || 'low'}">${escapeHtml(risk.severity || 'low')}</span>
          </div>
          <div class="report-copy-block">
            <strong>What is wrong</strong>
            <span>${escapeHtml(risk.what || risk.description || '')}</span>
          </div>
          <div class="report-copy-block">
            <strong>Why it matters</strong>
            <span>${escapeHtml(risk.why_it_matters || risk.why || '')}</span>
          </div>
          <div class="report-copy-block">
            <strong>What to do next</strong>
            <span>${escapeHtml(risk.what_to_do_next || risk.recommendation || '')}</span>
          </div>
          <ul class="signal-list">
            ${(risk.reasoning?.signals || []).slice(0, 3).map((signal) => `<li>${escapeHtml(signal)}</li>`).join('')}
          </ul>
          ${renderReplayPreview(risk)}
          ${renderReasoningPanel(risk)}
          ${renderConfidenceExplainer(risk, job, evidence)}
          <div class="report-card-meta">
            <span>${Math.round((risk.risk_score || 0) * 100)} risk score</span>
            <span>${escapeHtml(risk.location_label || 'Approximate location')}</span>
            <span>${escapeHtml(risk.hazard_code || 'finding')}</span>
            <span>${escapeHtml(risk.confidence_label || 'weak')} evidence</span>
            <span>${escapeHtml(risk.reasoning?.support_summary || 'Limited support')}</span>
            ${risk.follow_up_status ? `<span>${escapeHtml(formatFollowUpLabel(risk.follow_up_status))}</span>` : ''}
          </div>
          ${job.is_sample ? '' : `
            <div class="feedback-row" data-follow-up-controls data-job-id="${job.job_id}" data-hazard-code="${escapeHtml(risk.hazard_code || '')}" data-object-id="${escapeHtml(risk.object_id || '')}" data-active-status="${escapeHtml(risk.follow_up_status || '')}">
              ${renderFollowUpButton('resolved', risk.follow_up_status)}
              ${renderFollowUpButton('monitor', risk.follow_up_status)}
              ${renderFollowUpButton('ignored', risk.follow_up_status)}
            </div>
            <div class="feedback-row" data-feedback-controls data-job-id="${job.job_id}" data-hazard-code="${escapeHtml(risk.hazard_code || '')}" data-object-id="${escapeHtml(risk.object_id || '')}">
              ${renderFeedbackButton('useful', risk.latest_feedback)}
              ${renderFeedbackButton('wrong', risk.latest_feedback)}
              ${renderFeedbackButton('duplicate', risk.latest_feedback)}
            </div>
          `}
        </article>
      `).join('')
    : emptyMarkup(
        summary.analysis_outcome === 'rejected'
          ? 'ATLAS-0 rejected this scan as a normal room report. Follow the retry guidance and rescan before trusting any “all clear” takeaway.'
          : hazards.length
          ? 'Only lower-confidence findings remain. Toggle them on if you want the full raw report.'
          : summary.rescan_recommended
            ? 'No high-confidence hazards were detected, but this scan had limited coverage. Rescan before treating the room as low risk.'
            : 'No high-confidence hazards were detected. This is still a screening result, not a safety clearance.',
      );

  reportRecommendations.innerHTML = recommendations.length
    ? recommendations.map((rec) => `
        <article class="report-card recommendation">
          <div class="report-card-top">
            <h3>${escapeHtml(rec.title || 'Recommendation')}</h3>
            <span class="severity-pill ${rec.priority || 'low'}">${escapeHtml(rec.priority || 'low')}</span>
          </div>
          <p>${escapeHtml(rec.action || '')}</p>
          <div class="report-card-meta">
            <span>${escapeHtml(rec.location || 'scan area')}</span>
            <span>${escapeHtml(rec.why || '')}</span>
          </div>
          <div class="daily-value-actions">
            <button class="button-link ghost" type="button" data-open-fix-guide="${escapeHtml(fixGuideForText(`${rec.title} ${rec.action} ${rec.why}`).id)}">Open fix guide</button>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No actions were generated for this scan.');

  fixChecklistList.innerHTML = renderFixChecklist(job, fixFirst, recommendations, visibleHazards);

  evidenceTimeline.innerHTML = renderEvidenceTimeline(evidence, visibleHazards);
  reportEvidence.innerHTML = evidence.length
    ? evidence.map((frame, index) => `
        <article class="evidence-card" data-evidence-card="${index}">
          <div class="evidence-frame-shell">
            <img src="${api.withAccessToken(frame.image_url || '')}" alt="${escapeHtml(frame.caption || 'Evidence frame')}" />
            ${renderEvidenceFrameOverlay(frame)}
          </div>
          <div class="evidence-copy">
            <strong>${escapeHtml(frame.caption || 'Evidence frame')}</strong>
            <span>${Math.round((frame.confidence || 0) * 100)}% label confidence</span>
            <span>${escapeHtml(formatEvidenceMeta(frame))}</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No evidence frames were stored for this scan.');

  trustNotes.innerHTML = notes.length
    ? notes.map((note) => `<li>${escapeHtml(note)}</li>`).join('')
    : '<li>No additional trust notes.</li>';

  exportPdfBtn.href = job.is_sample ? api.sampleReportPdfUrl() : api.reportPdfUrl(job.job_id);
  exportPdfBtn.classList.remove('disabled');
  copyShareBtn.classList.remove('disabled');
  copyShareBtn.disabled = false;
  copyShareCardBtn?.classList.remove('disabled');
  if (copyShareCardBtn) {
    copyShareCardBtn.disabled = false;
  }
  deleteJobBtn.classList.toggle('disabled', Boolean(job.is_sample));
  deleteJobBtn.disabled = Boolean(job.is_sample);
  const expiryNote = job.expires_at
    ? ` Artifacts are scheduled to expire on ${new Date(job.expires_at).toLocaleDateString()}.`
    : '';
  shareLinkNote.textContent = job.is_sample
    ? 'Share link opens this built-in sample report view.'
    : `${summary.share_summary || 'Share link opens this exact report view.'}${expiryNote}`;
  if (!job.is_sample) {
    attachFollowUpHandlers(job.job_id);
    attachFeedbackHandlers(job.job_id);
    attachEvaluationHandlers(job.job_id);
  }
  attachFixChecklistHandlers(job.job_id);
  attachEvidenceTimelineHandlers();
  attachConfidenceExplainerHandlers(job);
}

function renderReportQuestionPanel(job, summary = {}, hazards = [], fixFirst = [], recommendations = [], evidence = [], scanQuality = {}) {
  if (!reportQuestionAnswer || !copyReportAnswerBtn) {
    return;
  }
  reportQuestionList?.querySelectorAll('[data-report-question]').forEach((button) => {
    button.classList.toggle('active', button.dataset.reportQuestion === state.activeReportQuestion);
    button.disabled = !job || job.status !== 'complete';
  });
  if (!job || job.status !== 'complete') {
    state.activeReportAnswer = '';
    reportQuestionAnswer.textContent = 'Open a completed report, then choose a question to get a cited, report-grounded answer.';
    copyReportAnswerBtn.disabled = true;
    copyReportAnswerBtn.classList.add('disabled');
    return;
  }

  const questionId = state.activeReportQuestion || 'fix_first';
  const answer = answerReportQuestion(questionId, job, summary, hazards, fixFirst, recommendations, evidence, scanQuality);
  state.activeReportAnswer = answer.text;
  reportQuestionAnswer.innerHTML = `
    <strong>${escapeHtml(answer.title)}</strong>
    <p>${escapeHtml(answer.body)}</p>
    <div class="report-card-meta">
      ${answer.citations.map((citation) => `<span>${escapeHtml(citation)}</span>`).join('')}
    </div>
  `;
  copyReportAnswerBtn.disabled = false;
  copyReportAnswerBtn.classList.remove('disabled');
}

function answerReportQuestion(questionId, job, summary, hazards, fixFirst, recommendations, evidence, scanQuality) {
  const topAction = fixFirst[0] || recommendations[0] || hazards[0] || {};
  const topHazard = hazards[0] || {};
  const evidenceIds = evidence
    .map((frame, index) => frame.evidence_id || `frame-${index + 1}`)
    .slice(0, 3);
  const confidence = summary.confidence_label || scanQuality.capture_summary || 'approximate confidence';
  const rescanCopy = scanQuality.rescan_recommended || summary.rescan_recommended
    ? 'A steadier follow-up scan is recommended before over-trusting smaller details.'
    : 'A rescan is useful after you complete the top fix, especially if you reuse the same room label.';
  const citations = [
    topHazard.hazard_code ? `Finding: ${topHazard.hazard_code}` : null,
    topHazard.confidence_label ? `Confidence: ${topHazard.confidence_label}` : `Confidence: ${confidence}`,
    evidenceIds.length ? `Evidence: ${evidenceIds.join(', ')}` : 'Evidence: none stored',
  ].filter(Boolean);
  const answers = {
    fix_first: {
      title: 'Fix first',
      body: topAction.title
        ? `${topAction.title}. ${topAction.action || topAction.what_to_do_next || topAction.recommendation || 'Review the top finding and choose the smallest safe fix.'}`
        : 'This report does not contain a clear priority action. Treat it as a prompt to review evidence and rescan if the room still feels uncertain.',
    },
    why_risky: {
      title: 'Why it is risky',
      body: topHazard.why_it_matters || topHazard.why || topHazard.description
        || 'The current report does not include enough supported risk reasoning for a stronger explanation.',
    },
    rescan: {
      title: 'What to rescan',
      body: `${rescanCopy} Keep the same room label, repeat the same route, and keep the top action area in frame.`,
    },
    share: {
      title: 'Can you share this?',
      body: 'Yes, if the recipient understands this is decision support, not safety certification. Use the Privacy Receipt to review room label, evidence inclusion, redaction status, retention, and delete controls before sharing.',
    },
    low_confidence: {
      title: 'What low confidence means',
      body: 'Low confidence means the scan evidence, lighting, coverage, localization, or object support is weaker. It should guide what to inspect or rescan, not create a final safety conclusion.',
    },
  };
  const selected = answers[questionId] || answers.fix_first;
  return {
    title: selected.title,
    body: selected.body,
    citations,
    text: [
      `ATLAS-0 answer: ${selected.title}`,
      selected.body,
      citations.length ? `Citations: ${citations.join(' · ')}` : '',
      `Report: ${reportDeepLink(job)}`,
      'Decision support only, not safety certification.',
    ].filter(Boolean).join('\n'),
  };
}

function privacyReceiptState(jobId) {
  const stateForJob = readJsonObject(PRIVACY_RECEIPT_STORAGE_KEY)[jobId] || {};
  return {
    excludedEvidence: Array.isArray(stateForJob.excludedEvidence) ? stateForJob.excludedEvidence : [],
    blurredEvidence: Array.isArray(stateForJob.blurredEvidence) ? stateForJob.blurredEvidence : [],
  };
}

function writePrivacyReceiptState(jobId, value) {
  const all = readJsonObject(PRIVACY_RECEIPT_STORAGE_KEY);
  all[jobId] = value;
  writeJsonObject(PRIVACY_RECEIPT_STORAGE_KEY, all);
}

function evidencePrivacyId(frame, index) {
  return String(frame.evidence_id || `frame-${index + 1}`);
}

function selectedEvidenceFrames(job) {
  const evidence = job?.evidence_frames || [];
  if (!job?.job_id) {
    return [];
  }
  const excluded = new Set(privacyReceiptState(job.job_id).excludedEvidence || []);
  return evidence.filter((frame, index) => !excluded.has(evidencePrivacyId(frame, index)));
}

function blurredEvidenceIds(job) {
  if (!job?.job_id) {
    return new Set();
  }
  return new Set(privacyReceiptState(job.job_id).blurredEvidence || []);
}

function renderPrivacyReceipt(job, summary = {}, evidence = []) {
  if (!privacyReceiptSummary || !privacyEvidenceList || !copyPrivacyReceiptBtn || !downloadPrivacyReceiptBtn) {
    return;
  }
  if (!job || job.status !== 'complete') {
    privacyReceiptSummary.innerHTML = `
      <article class="privacy-receipt-item"><strong>—</strong><span>Room label</span></article>
      <article class="privacy-receipt-item"><strong>—</strong><span>Evidence selected</span></article>
      <article class="privacy-receipt-item"><strong>—</strong><span>Retention</span></article>
    `;
    privacyEvidenceList.innerHTML = emptyMarkup('Evidence inclusion toggles appear after a completed report.');
    copyPrivacyReceiptBtn.disabled = true;
    downloadPrivacyReceiptBtn.disabled = true;
    copyPrivacyReceiptBtn.classList.add('disabled');
    downloadPrivacyReceiptBtn.classList.add('disabled');
    return;
  }

  const selectedEvidence = selectedEvidenceFrames(job);
  const blurredEvidence = blurredEvidenceIds(job);
  const roomLabel = job.room_label || summary.room_label || 'Unlabeled room';
  const retention = state.privacyPolicy?.retention_days ?? 'unknown';
  if (!state.privacyReceiptEvents.has(job.job_id)) {
    state.privacyReceiptEvents.add(job.job_id);
    trackProductEvent('privacy_receipt_opened', {
      surface: 'privacy_receipt',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
    });
  }
  privacyReceiptSummary.innerHTML = `
    <article class="privacy-receipt-item"><strong>${escapeHtml(roomLabel)}</strong><span>Room label included in report/share text</span></article>
    <article class="privacy-receipt-item"><strong>${selectedEvidence.length}/${evidence.length}</strong><span>Evidence frames selected for local share wording</span></article>
    <article class="privacy-receipt-item"><strong>${blurredEvidence.size}</strong><span>Local blur previews before copying/sharing</span></article>
    <article class="privacy-receipt-item"><strong>${escapeHtml(String(retention))}</strong><span>${retention === 'unknown' ? 'Retention unavailable' : 'day retention window'}</span></article>
  `;
  privacyEvidenceList.innerHTML = evidence.length
    ? evidence.map((frame, index) => {
        const id = evidencePrivacyId(frame, index);
        const excluded = new Set(privacyReceiptState(job.job_id).excludedEvidence || []);
        const checked = !excluded.has(id);
        const blurred = blurredEvidence.has(id);
        const imageUrl = frame.image_url ? api.withAccessToken(frame.image_url) : '';
        return `
          <article class="privacy-evidence-toggle">
            ${imageUrl ? `<img class="privacy-evidence-thumb ${blurred ? 'blurred' : ''}" src="${escapeHtml(imageUrl)}" alt="" loading="lazy" />` : '<div class="privacy-evidence-thumb placeholder" aria-hidden="true">No image</div>'}
            <div class="privacy-evidence-copy">
              <strong>${escapeHtml(frame.caption || id)}</strong>
              <span>${escapeHtml(frame.redacted ? 'Text-heavy region redacted before storage.' : 'No server-side redaction flag on this frame.')}</span>
              <span>${escapeHtml(blurred ? 'Local preview blur is on for copy/share review.' : 'Local preview blur is off.')}</span>
              <div class="privacy-evidence-controls">
                <label><input type="checkbox" data-privacy-evidence="${escapeHtml(id)}" ${checked ? 'checked' : ''} /> Include in share wording</label>
                <label><input type="checkbox" data-privacy-blur="${escapeHtml(id)}" ${blurred ? 'checked' : ''} /> Preview blur locally</label>
              </div>
            </div>
          </article>
        `;
      }).join('')
    : emptyMarkup('This report has no stored evidence frames. Share only the summary and trust notes.');
  copyPrivacyReceiptBtn.disabled = false;
  downloadPrivacyReceiptBtn.disabled = false;
  copyPrivacyReceiptBtn.classList.remove('disabled');
  downloadPrivacyReceiptBtn.classList.remove('disabled');
}

function privacyReceiptText(job) {
  if (!job || job.status !== 'complete') {
    return 'ATLAS-0 Privacy Receipt: no completed report is active.';
  }
  const summary = job.summary || {};
  const evidence = job.evidence_frames || [];
  const selectedEvidence = selectedEvidenceFrames(job);
  const blurred = blurredEvidenceIds(job).size;
  const redacted = evidence.filter((frame) => frame.redacted).length;
  return [
    'ATLAS-0 Privacy Receipt',
    `Room label: ${job.room_label || summary.room_label || 'Unlabeled room'}`,
    `Calm Score: ${typeof summary.room_score === 'number' ? `${summary.room_score}/100` : 'not available'}`,
    `Findings: ${(job.risks || []).length}`,
    `Evidence selected locally: ${selectedEvidence.length}/${evidence.length}`,
    `Local blur/redaction previews enabled: ${blurred}`,
    `Redacted evidence frames: ${redacted}`,
    `Retention: ${state.privacyPolicy?.retention_days ?? 'unknown'} day(s)`,
    `Delete controls: ${state.privacyPolicy?.delete_supported ? 'available in report view' : 'unknown'}`,
    'This receipt affects local share/copy wording only. It does not mutate stored server artifacts.',
    'Decision support only, not safety certification.',
  ].join('\n');
}

function renderHeroBadges(summary, roomLabel, comparison, resolution, isSample) {
  const badges = [
    summary.audience_label || 'General home safety',
    summary.report_posture || 'screening',
    summary.coverage_label ? `${summary.coverage_label} coverage` : 'Coverage pending',
    summary.rescan_recommended ? 'Rescan recommended' : 'Evidence-backed',
  ];
  if (isSample) {
    badges.unshift('Built-in sample');
  }
  if (roomLabel) {
    badges.unshift(roomLabel);
  }
  if (typeof summary.room_score === 'number') {
    badges.push(`${summary.room_score}/100 Calm Score`);
  }
  if (Number(resolution?.resolved_count || 0) > 0) {
    badges.push(`${resolution.resolved_count} resolved`);
  }
  if (comparison?.score_delta) {
    const delta = Number(comparison.score_delta || 0);
    badges.push(`${delta > 0 ? '+' : ''}${delta} vs last scan`);
  }
  return badges.map((badge) => `<span class="soft-badge">${escapeHtml(badge)}</span>`).join('');
}

function renderBriefExecutive(job, summary = {}, hazards = [], fixFirst = [], recommendations = [], scanQuality = {}) {
  if (!briefExecutiveTitle || !briefExecutiveCopy || !briefExecutiveActions || !briefConfidenceLabel || !briefConfidenceMeter || !briefConfidenceCopy) {
    return;
  }

  if (!job || job.status !== 'complete') {
    briefExecutiveTitle.textContent = 'No Safety Brief yet';
    briefExecutiveCopy.textContent = 'Upload a focused room walkthrough or open the sample report to see top actions, evidence confidence, and practical next steps in one premium brief.';
    briefExecutiveActions.innerHTML = `
      <span class="soft-badge">Top 3 actions</span>
      <span class="soft-badge">Evidence rail</span>
      <span class="soft-badge">Decision support</span>
    `;
    briefConfidenceLabel.textContent = 'Waiting for scan';
    briefConfidenceMeter.style.width = '0%';
    briefConfidenceCopy.textContent = 'ATLAS-0 shows confidence and scan-quality notes so users know when to act, when to rescan, and when not to over-trust a result.';
    if (briefConfidenceDetails) {
      briefConfidenceDetails.open = false;
      const paragraph = briefConfidenceDetails.querySelector('p');
      if (paragraph) {
        paragraph.textContent = 'Upload quality, evidence count, finding confidence, and trust notes explain why this brief is useful for triage rather than safety certification.';
      }
    }
    return;
  }

  const roomLabel = job.room_label || summary.room_label || 'Room';
  const topAction = fixFirst[0]?.title
    || recommendations[0]?.title
    || hazards[0]?.hazard_title
    || summary.top_hazard_label
    || 'Review the top finding';
  const hazardCount = Number(summary.hazard_count || hazards.length || 0);
  const score = typeof summary.room_score === 'number' ? `${Math.round(summary.room_score)}/100` : 'Screened';
  const confidenceLabel = summary.confidence_label
    || summary.scan_quality_label
    || (scanQuality.status ? `${capitalize(scanQuality.status)} scan` : 'Approximate grounding');
  const meterValue = typeof scanQuality.score === 'number'
    ? scanQuality.score
    : typeof summary.average_confidence === 'number'
      ? summary.average_confidence
      : typeof hazards[0]?.confidence === 'number'
        ? hazards[0].confidence
        : 0.66;
  const meterPercent = Math.max(0, Math.min(100, Math.round(meterValue * 100)));
  const rescanCopy = summary.rescan_recommended || scanQuality.rescan_recommended
    ? 'This brief is useful for triage, but scan quality suggests a rescan before trusting smaller details.'
    : 'This brief is ready for practical next steps. Keep the confidence notes visible before making bigger decisions.';

  briefExecutiveTitle.textContent = `${roomLabel} Safety Brief`;
  briefExecutiveCopy.textContent = `${score} Calm Score. Start with "${topAction}" and use the evidence rail to confirm what ATLAS-0 saw before acting.`;
  briefExecutiveActions.innerHTML = `
    <span class="soft-badge">${escapeHtml(topAction)}</span>
    <span class="soft-badge">${hazardCount} finding${hazardCount === 1 ? '' : 's'}</span>
    <span class="soft-badge">Decision support only</span>
  `;
  briefConfidenceLabel.textContent = confidenceLabel;
  briefConfidenceMeter.style.width = `${meterPercent}%`;
  briefConfidenceCopy.textContent = rescanCopy;
  if (briefConfidenceDetails) {
    const paragraph = briefConfidenceDetails.querySelector('p');
    if (paragraph) {
      const evidenceCount = Number(job.evidence_frames?.length || 0);
      const visibleCount = hazards.length;
      paragraph.textContent = [
        `${visibleCount} visible finding${visibleCount === 1 ? '' : 's'} and ${evidenceCount} evidence frame${evidenceCount === 1 ? '' : 's'} support this brief.`,
        scanQuality.capture_summary || summary.coverage_summary || 'Scan quality details are limited, so trust notes should stay visible.',
        'Use this as decision support, not safety certification.',
      ].join(' ');
    }
  }
}

function renderBriefTriage(job, summary = {}, hazards = [], fixFirst = [], recommendations = [], scanQuality = {}) {
  if (!briefTriageStrip) {
    return;
  }
  if (!job || job.status !== 'complete') {
    briefTriageStrip.innerHTML = emptyMarkup('Fix Today, Watch Later, and Rescan Needed guidance will appear after a completed scan.');
    return;
  }

  const fixToday = fixFirst[0] || recommendations[0] || hazards[0] || {};
  const watchLater = hazards.find((risk) => risk.follow_up_status === 'monitor')
    || hazards.find((risk) => Number(risk.confidence || 0) < 0.72)
    || hazards[1]
    || {};
  const needsRescan = Boolean(summary.rescan_recommended || scanQuality.rescan_recommended);
  const rescanCopy = needsRescan
    ? scanQuality.capture_summary || summary.coverage_summary || 'Scan confidence is weak enough that a steadier follow-up is worth doing.'
    : 'No urgent rescan required for this first-pass brief. Reuse the same room label after one fix to make progress visible.';
  const fixTitle = fixToday.title || fixToday.hazard_title || fixToday.object_label || summary.top_hazard_label || 'Pick one practical fix';
  const fixCopy = fixToday.action || fixToday.what_to_do_next || fixToday.recommendation || fixToday.why || 'Start with the top evidence-backed action before trying to fix the whole room.';
  const watchTitle = watchLater.hazard_title || watchLater.title || watchLater.object_label || 'Watch weaker findings';
  const watchCopy = watchLater.why_it_matters || watchLater.why || watchLater.description || 'Review lower-confidence or monitor items after the top fix and evidence frames.';

  briefTriageStrip.innerHTML = `
    <div class="triage-head">
      <div>
        <span class="guide-kicker">Safety Brief 2.0</span>
        <h3>Fix today, watch later, rescan only when it helps.</h3>
        <p>ATLAS-0 turns the report into a home-care order of operations instead of a wall of warnings.</p>
      </div>
      <span class="pill">${escapeHtml(summary.report_posture || 'Decision support')}</span>
    </div>
    <div class="triage-grid">
      <article class="triage-card fix">
        <span class="guide-kicker">Fix Today</span>
        <strong>${escapeHtml(fixTitle)}</strong>
        <p>${escapeHtml(fixCopy)}</p>
        <button class="button-link ghost" type="button" data-copy-fix-today="true">Copy fix today</button>
      </article>
      <article class="triage-card watch">
        <span class="guide-kicker">Watch Later</span>
        <strong>${escapeHtml(watchTitle)}</strong>
        <p>${escapeHtml(watchCopy)}</p>
      </article>
      <article class="triage-card rescan">
        <span class="guide-kicker">Rescan Needed</span>
        <strong>${escapeHtml(needsRescan ? 'Yes, improve capture' : 'Not immediately')}</strong>
        <p>${escapeHtml(rescanCopy)}</p>
        <button class="button-link ghost" type="button" data-start-rescan="true">Prepare rescan</button>
      </article>
    </div>
  `;
}

function buildFieldNotes(summary = {}, hazards = [], fixFirst = [], recommendations = [], scanQuality = {}) {
  const topAction = fixFirst[0] || recommendations[0] || hazards[0] || {};
  const notes = [];
  const difficulty = fixDifficultyLabel(topAction, 0);
  if (topAction.title || topAction.hazard_title || topAction.object_label) {
    notes.push({
      id: 'quick-fix',
      title: difficulty === 'Quick fix' ? 'This looks like a quick room win.' : 'This is worth a deliberate fix.',
      copy: topAction.action || topAction.what_to_do_next || topAction.recommendation || topAction.why || 'Start with the top evidence-backed action before widening the checklist.',
    });
  }
  if (summary.rescan_recommended || scanQuality.rescan_recommended) {
    notes.push({
      id: 'rescan',
      title: 'This is a discovery prompt, not a final verdict.',
      copy: scanQuality.capture_summary || summary.coverage_summary || 'A steadier follow-up scan would make smaller findings easier to trust.',
    });
  } else {
    notes.push({
      id: 'calm',
      title: 'Useful does not have to feel dramatic.',
      copy: 'The best next step is one visible fix, then a same-room rescan if you want progress to show up.',
    });
  }
  const lowConfidenceCount = hazards.filter((risk) => isLowConfidenceRisk(risk)).length;
  if (lowConfidenceCount) {
    notes.push({
      id: 'watch',
      title: 'Some findings are for watching, not panicking.',
      copy: `${lowConfidenceCount} visible finding${lowConfidenceCount === 1 ? '' : 's'} have weaker support. Check evidence before acting.`,
    });
  }
  notes.push({
    id: 'mystery',
    title: activeMysteryMode().title,
    copy: activeMysteryMode().resultFrame,
  });
  return notes.slice(0, 4);
}

function renderFieldNotes(job, summary = {}, hazards = [], fixFirst = [], recommendations = [], scanQuality = {}) {
  if (!fieldNotesPanel) {
    return;
  }
  if (!job || job.status !== 'complete') {
    fieldNotesPanel.innerHTML = emptyMarkup('ATLAS Field Notes will appear after a completed scan.');
    return;
  }
  const notes = buildFieldNotes(summary, hazards, fixFirst, recommendations, scanQuality);
  fieldNotesPanel.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">ATLAS Field Notes</span>
        <h3>Small observations that make the report easier to explore.</h3>
        <p>Human-readable notes derived from findings, confidence, scan quality, and the active discovery prompt.</p>
      </div>
      <span class="pill">Evidence-aware</span>
    </div>
    <div class="field-note-grid">
      ${notes.map((note) => `
        <details class="field-note-card" data-field-note="${escapeHtml(note.id)}">
          <summary>${escapeHtml(note.title)}</summary>
          <p>${escapeHtml(note.copy)}</p>
        </details>
      `).join('')}
    </div>
  `;
  fieldNotesPanel.querySelectorAll('[data-field-note]').forEach((details) => {
    details.addEventListener('toggle', () => {
      if (details.open) {
        trackProductEvent('field_note_expanded', {
          field_note_id: details.dataset.fieldNote || null,
          job_id: job.job_id,
          sample_key: job.sample_key || null,
          audience_mode: job.audience_mode || 'general',
        });
      }
    });
  });
}

function markerPosition(index, total) {
  const safeTotal = Math.max(1, total);
  const x = 16 + ((index * 23) % 68);
  const y = 18 + ((index * 31 + safeTotal * 7) % 58);
  return { x, y };
}

function renderRoomMapPreview(job, summary = {}, hazards = [], evidence = []) {
  if (!roomMapPreview) {
    return;
  }
  if (!job || job.status !== 'complete') {
    roomMapPreview.innerHTML = emptyMarkup('Approximate evidence map will appear after a completed scan.');
    return;
  }
  const markers = hazards.slice(0, 6).map((risk, index) => ({
    title: risk.hazard_title || risk.object_label || `Finding ${index + 1}`,
    location: risk.location_label || risk.location || 'approximate scan zone',
    severity: risk.severity || 'low',
    ...markerPosition(index, hazards.length || evidence.length || 1),
  }));
  roomMapPreview.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Approximate evidence map</span>
        <h3>A tactile preview of where findings cluster.</h3>
        <p>This is an evidence map, not measured 3D reconstruction. Use it to orient the report, then confirm with frames.</p>
      </div>
      <button class="button-link ghost" type="button" data-room-map-preview="true">Explore map</button>
    </div>
    <div class="room-map-shell" aria-label="Approximate room evidence map">
      <div class="room-map-canvas" aria-hidden="true">
        ${markers.map((marker, index) => `
          <span class="room-map-marker ${escapeHtml(marker.severity)}" style="left:${marker.x}%; top:${marker.y}%;">${index + 1}</span>
        `).join('')}
      </div>
      <div class="room-map-list">
        ${markers.length ? markers.map((marker, index) => `
          <button class="room-map-item" type="button" data-room-map-preview="true" data-map-marker="${index + 1}">
            <strong>${index + 1}. ${escapeHtml(marker.title)}</strong>
            <span>${escapeHtml(marker.location)} · ${escapeHtml(marker.severity)} severity</span>
          </button>
        `).join('') : '<div class="empty-card">No hazard markers were generated for this scan.</div>'}
      </div>
    </div>
  `;
}

function buildBeforeAfterStoryText(job) {
  const summary = job.summary || {};
  const comparison = job.room_comparison || {};
  const roomLabel = job.room_label || summary.room_label || 'Room';
  const score = typeof summary.room_score === 'number' ? `${Math.round(summary.room_score)}/100` : 'screened';
  const delta = typeof comparison.score_delta === 'number'
    ? `${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta}`
    : 'baseline';
  const topAction = (job.fix_first || [])[0]?.title || (job.recommendations || [])[0]?.title || summary.top_hazard_label || 'review the top action';
  return [
    `ATLAS-0 before/after story: ${roomLabel}`,
    `Current score: ${score}. Change: ${delta}.`,
    `Top action: ${topAction}.`,
    comparison.summary || 'Reuse this room label after one fix to make progress visible.',
    'Decision support only, not safety certification.',
  ].join('\n');
}

function comparisonEvidenceImage(frame) {
  return frame?.image_url ? api.withAccessToken(frame.image_url) : '';
}

function renderComparisonEvidenceCard(frame, label) {
  if (!frame) {
    return `
      <article class="compare-evidence-card empty">
        <div class="compare-evidence-image placeholder">No frame</div>
        <strong>${escapeHtml(label)}</strong>
        <p>Evidence will appear here after a same-room scan has saved frames.</p>
      </article>
    `;
  }
  const imageUrl = comparisonEvidenceImage(frame);
  const confidence = typeof frame.confidence === 'number'
    ? `${Math.round(frame.confidence * 100)}% confidence`
    : 'Approximate evidence';
  return `
    <article class="compare-evidence-card">
      ${imageUrl ? `<img class="compare-evidence-image" src="${escapeHtml(imageUrl)}" alt="" loading="lazy" />` : '<div class="compare-evidence-image placeholder">No image</div>'}
      <span class="guide-kicker">${escapeHtml(label)}</span>
      <strong>${escapeHtml(frame.caption || frame.evidence_id || 'Evidence frame')}</strong>
      <p>${escapeHtml(`${confidence}${frame.redacted ? ' · redacted flag' : ''}`)}</p>
    </article>
  `;
}

function renderVisualBeforeAfter(comparison = null) {
  const previous = Array.isArray(comparison?.previous_evidence) ? comparison.previous_evidence[0] : null;
  const current = Array.isArray(comparison?.current_evidence) ? comparison.current_evidence[0] : null;
  if (!previous && !current) {
    return `
      <div class="visual-compare-strip muted">
        <div class="compare-evidence-card empty">
          <div class="compare-evidence-image placeholder">Baseline</div>
          <strong>Visual comparison locked</strong>
          <p>Save a same-room scan with evidence frames, then rescan after one fix to compare what changed.</p>
        </div>
      </div>
    `;
  }
  return `
    <div class="visual-compare-strip" aria-label="Visual before and after evidence comparison">
      ${renderComparisonEvidenceCard(previous, 'Before evidence')}
      <div class="compare-delta-pill">${escapeHtml(comparison?.trend || 'compare')}</div>
      ${renderComparisonEvidenceCard(current, 'After evidence')}
    </div>
  `;
}

function renderBeforeAfterStory(job, summary = {}, hazards = [], fixFirst = [], recommendations = [], comparison = null) {
  if (!beforeAfterStory) {
    return;
  }
  if (!job || job.status !== 'complete') {
    beforeAfterStory.innerHTML = emptyMarkup('Before/after story card will appear after a completed scan.');
    return;
  }
  const score = typeof summary.room_score === 'number' ? `${Math.round(summary.room_score)}/100` : 'Screened';
  const topAction = fixFirst[0]?.title || recommendations[0]?.title || hazards[0]?.hazard_title || summary.top_hazard_label || 'Review the top action';
  const delta = comparison && typeof comparison.score_delta === 'number'
    ? `${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta} score change`
    : 'Baseline saved';
  beforeAfterStory.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Before / After Story</span>
        <h3>${escapeHtml(comparison ? 'A shareable progress artifact.' : 'Your baseline is ready.')}</h3>
        <p>${escapeHtml(comparison?.summary || 'Rescan with the same room label after one fix to unlock a visible before/after story.')}</p>
      </div>
      <button class="button-link ghost" type="button" data-copy-before-after-story="true" ${comparison ? '' : 'disabled'}>${comparison ? 'Copy story card' : 'Rescan to unlock'}</button>
    </div>
    <article class="story-card">
      <span class="guide-kicker">ATLAS-0 room story</span>
      <strong>${escapeHtml(job.room_label || summary.room_label || 'Room')} · ${escapeHtml(score)}</strong>
      <p>${escapeHtml(topAction)}.</p>
      <div class="journal-meta">
        <span class="soft-badge">${escapeHtml(delta)}</span>
        <span class="soft-badge">${escapeHtml(`${hazards.length} finding${hazards.length === 1 ? '' : 's'}`)}</span>
        <span class="soft-badge">${escapeHtml(summary.confidence_label || 'Approximate confidence')}</span>
      </div>
    </article>
    ${renderVisualBeforeAfter(comparison)}
  `;
}

function fixDifficultyLabel(item, index = 0) {
  const text = `${item.title || ''} ${item.action || ''} ${item.why || ''}`.toLowerCase();
  if (text.match(/\b(anchor|mount|bracket|secure|install|contractor|repair)\b/)) {
    return 'Heavier fix';
  }
  if (text.match(/\b(move|clear|lower|tuck|route|remove|shift|relocate)\b/)) {
    return 'Quick fix';
  }
  return index === 0 ? 'Start here' : 'Review';
}

function roomPassportStorageKey(jobId) {
  return `${VERIFICATION_STORAGE_KEY}.${jobId}`;
}

function readVerificationState(jobId) {
  const raw = readStoredPreference(roomPassportStorageKey(jobId));
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeVerificationState(jobId, nextState) {
  writeStoredPreference(roomPassportStorageKey(jobId), JSON.stringify(nextState));
}

function roomPassportSummary(job, summary = {}, hazards = [], fixFirst = [], comparison = null) {
  const roomLabel = job?.room_label || summary.room_label || 'Room';
  const journal = readHomeJournal();
  const key = roomJournalKey({
    room_label: roomLabel,
    summary: { room_label: roomLabel, audience_label: summary.audience_label || job?.audience_mode || 'general' },
  });
  const entry = journal[key] || {};
  const scores = Array.isArray(entry.scores) ? entry.scores : [];
  const recurringRisk = entry.topAction || hazards[0]?.hazard_title || summary.top_hazard_label || 'No recurring risk yet';
  const completedFixes = Number(entry.completedFixes || 0);
  const favorite = readFavoriteRooms().has(key);
  const nextDue = currentRescanReminder() === 'off'
    ? 'No reminder set'
    : currentRescanReminder() === 'weekly'
      ? 'Check again next week'
      : 'Check again next month';
  return {
    key,
    roomLabel,
    favorite,
    recurringRisk,
    completedFixes,
    nextDue,
    lastChecked: entry.lastCheckedAt || new Date().toISOString(),
    trend: scores.length >= 2 ? `${scores.at(-2)} → ${scores.at(-1)}` : (typeof summary.room_score === 'number' ? `${summary.room_score}/100 baseline` : 'Baseline pending'),
    topAction: fixFirst[0]?.title || recurringRisk,
    delta: comparison && typeof comparison.score_delta === 'number'
      ? `${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta}`
      : 'baseline',
  };
}

function buildRoomPassportText(job) {
  const summary = job.summary || {};
  const passport = roomPassportSummary(job, summary, job.risks || [], job.fix_first || [], job.room_comparison || null);
  return [
    `ATLAS-0 Room Health Passport: ${passport.roomLabel}`,
    `Calm Score trend: ${passport.trend}.`,
    `Recurring attention area: ${passport.recurringRisk}.`,
    `Completed fixes tracked locally: ${passport.completedFixes}.`,
    `Next check: ${passport.nextDue}. Decision support only, not safety certification.`,
  ].join('\n');
}

function renderRoomPassport(job, summary = {}, hazards = [], fixFirst = [], comparison = null) {
  if (!roomPassportPanel) {
    return;
  }
  if (!job || job.status !== 'complete') {
    roomPassportPanel.innerHTML = emptyMarkup('Room Health Passport appears after a completed scan and keeps the room history local to this browser.');
    return;
  }
  const passport = roomPassportSummary(job, summary, hazards, fixFirst, comparison);
  roomPassportPanel.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Room Health Passport</span>
        <h3>${escapeHtml(passport.roomLabel)} companion card.</h3>
        <p>Per-room identity, trend, recurring attention area, completed fixes, and next-check posture without accounts.</p>
      </div>
      <span class="pill">${passport.favorite ? 'Favorite room' : 'Local passport'}</span>
    </div>
    <div class="passport-grid">
      <article class="passport-card">
        <span class="room-motif" aria-hidden="true"></span>
        <span class="guide-kicker">Calm Score trend</span>
        <strong>${escapeHtml(passport.trend)}</strong>
        <p>${escapeHtml(passport.delta === 'baseline' ? 'First saved baseline for this room.' : `${passport.delta} score movement from the previous matching room label.`)}</p>
      </article>
      <article class="passport-card">
        <span class="room-motif evidence" aria-hidden="true"></span>
        <span class="guide-kicker">Recurring attention area</span>
        <strong>${escapeHtml(passport.recurringRisk)}</strong>
        <p>Watch whether this appears again after a same-room rescan.</p>
      </article>
      <article class="passport-card">
        <span class="room-motif fix" aria-hidden="true"></span>
        <span class="guide-kicker">Completed fixes</span>
        <strong>${passport.completedFixes}</strong>
        <p>${escapeHtml(passport.completedFixes ? 'Local checklist progress is feeding the room story.' : 'Mark one fix done to make the next scan more meaningful.')}</p>
      </article>
      <article class="passport-card">
        <span class="room-motif playbook" aria-hidden="true"></span>
        <span class="guide-kicker">Next check due</span>
        <strong>${escapeHtml(passport.nextDue)}</strong>
        <p>${escapeHtml(activeRoomPlaybook()?.title || activeMysteryMode().title)}</p>
      </article>
    </div>
    <div class="report-loop-actions">
      <button class="button-link ghost" type="button" data-copy-room-passport="true">Copy room summary</button>
      <button class="button-link ghost" type="button" data-start-rescan="true">Rescan same room</button>
      <button class="button-link ghost" type="button" data-jump-view="journal">Open Home Journal</button>
    </div>
  `;
}

function buildFixVerificationText(job) {
  const summary = job.summary || {};
  const progress = checklistProgress(job, job.fix_first || [], job.recommendations || [], job.risks || []);
  const comparison = job.room_comparison || null;
  const status = comparison && Number(comparison.score_delta || 0) > 0
    ? 'fixed or improved'
    : progress.done > 0
      ? 'still watch after local fix'
      : 'rescan needed after one fix';
  return [
    `ATLAS-0 fix verification for ${job.room_label || summary.room_label || 'Room'}`,
    `Status: ${status}. Checklist: ${progress.done}/${progress.total}.`,
    comparison?.summary || 'Reuse the same room label for before/after validation.',
    'Decision support only, not safety certification.',
  ].join('\n');
}

function renderFixVerification(job, summary = {}, hazards = [], fixFirst = [], recommendations = [], comparison = null) {
  if (!fixVerificationPanel) {
    return;
  }
  if (!job || job.status !== 'complete') {
    fixVerificationPanel.innerHTML = emptyMarkup('Fix Verification Mode appears after a completed scan.');
    return;
  }
  const progress = checklistProgress(job, fixFirst, recommendations, hazards);
  const scoreDelta = comparison && typeof comparison.score_delta === 'number' ? comparison.score_delta : null;
  const verification = readVerificationState(job.job_id);
  const status = scoreDelta !== null && scoreDelta > 0
    ? 'fixed'
    : progress.done > 0
      ? 'still-watch'
      : 'rescan-needed';
  const copy = status === 'fixed'
    ? 'The same-room comparison improved. Still confirm evidence before declaring the room done.'
    : status === 'still-watch'
      ? 'You marked local checklist progress. A same-room rescan can show whether the evidence changed.'
      : 'Pick one fix, then rescan with the same room label for before/after validation.';
  writeVerificationState(job.job_id, { ...verification, status, updatedAt: new Date().toISOString() });
  fixVerificationPanel.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Fix Verification Mode</span>
        <h3>${escapeHtml(status === 'fixed' ? 'Fixed signal detected.' : status === 'still-watch' ? 'Fix logged. Verification pending.' : 'Rescan needed after one fix.')}</h3>
        <p>${escapeHtml(copy)}</p>
      </div>
      <span class="pill">${escapeHtml(status.replace('-', ' '))}</span>
    </div>
    <div class="verification-grid">
      <article class="verification-card">
        <span class="guide-kicker">Checklist progress</span>
        <strong>${progress.done}/${progress.total || 0}</strong>
        <p>${escapeHtml(progress.total ? 'Local checklist state is stored in this browser.' : 'No checklist items were generated for this report.')}</p>
      </article>
      <article class="verification-card">
        <span class="guide-kicker">Score delta</span>
        <strong>${scoreDelta === null ? 'Baseline' : `${scoreDelta > 0 ? '+' : ''}${scoreDelta}`}</strong>
        <p>${escapeHtml(comparison?.summary || 'Same-room comparison unlocks when the room label is reused.')}</p>
      </article>
      <article class="verification-card">
        <span class="guide-kicker">Next posture</span>
        <strong>${escapeHtml(status === 'fixed' ? 'Fixed' : status === 'still-watch' ? 'Still watch' : 'Rescan needed')}</strong>
        <p>Use this as a verification prompt, not proof of certification.</p>
      </article>
    </div>
    <div class="report-loop-actions">
      <button class="button-link ghost" type="button" data-start-fix-verification="true">Start verification</button>
      <button class="button-link ghost" type="button" data-copy-fix-verification="true">Copy verification note</button>
      <button class="button-link ghost" type="button" data-start-rescan="true">Prepare same-room rescan</button>
    </div>
  `;
}

function fixQuestStorageKey(jobId) {
  return `${FIX_QUEST_STORAGE_KEY}.${jobId}`;
}

function readFixQuestState(jobId) {
  const raw = readStoredPreference(fixQuestStorageKey(jobId));
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeFixQuestState(jobId, nextState) {
  writeStoredPreference(fixQuestStorageKey(jobId), JSON.stringify(nextState));
}

function buildFixQuests(job, fixFirst = [], recommendations = [], hazards = []) {
  const items = buildChecklistItems(fixFirst, recommendations, hazards);
  const source = items.length ? items : [{
    title: job?.summary?.top_hazard_label || 'Review the Safety Brief',
    action: 'Pick one small evidence-backed improvement, then rescan the same room.',
    source: 'Brief',
  }];
  return source.slice(0, 4).map((item, index) => {
    const template = FIX_QUEST_TEMPLATES[index % FIX_QUEST_TEMPLATES.length];
    return {
      id: checklistItemId(item, index),
      title: item.title || template.label,
      action: item.action || template.copy,
      label: template.label,
      copy: template.copy,
      source: item.source || 'Report',
    };
  });
}

function renderReportThemePanel(job) {
  if (!reportThemePanel || !reportThemeCopy) {
    return;
  }
  const theme = reportThemeById(currentReportTheme());
  if (reportThemeInput) {
    reportThemeInput.value = theme.id;
  }
  if (!job || job.status !== 'complete') {
    reportThemeCopy.innerHTML = `
      <strong>${escapeHtml(theme.title)}</strong>
      <p>${escapeHtml(theme.copy)} Run a scan or open the sample report to preview this theme against a Safety Brief.</p>
    `;
    return;
  }
  const summary = job.summary || {};
  const topAction = (job.fix_first || [])[0]?.title || summary.top_hazard_label || 'Fix one thing';
  reportThemeCopy.innerHTML = `
    <strong>${escapeHtml(theme.title)} · ${escapeHtml(theme.badge)}</strong>
    <p>${escapeHtml(theme.copy)}</p>
    <p>${escapeHtml(`Preview: ${job.room_label || summary.room_label || 'This room'} has ${summary.room_score ?? 'a pending'} Calm Score. Start with "${topAction}" and treat the report as decision support, not certification.`)}</p>
  `;
}

function renderFixQuestPanel(job, summary = {}, hazards = [], fixFirst = [], recommendations = []) {
  if (!fixQuestPanel) {
    return;
  }
  if (!job || job.status !== 'complete') {
    fixQuestPanel.innerHTML = emptyMarkup('Fix Quest Cards will appear after a completed scan.');
    return;
  }
  const quests = buildFixQuests(job, fixFirst, recommendations, hazards);
  const questState = readFixQuestState(job.job_id);
  fixQuestPanel.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Fix Quest Cards</span>
        <h3>Turn the brief into one doable home win.</h3>
        <p>Quests are local progress helpers built from fix-first actions, recommendations, and findings.</p>
      </div>
      <span class="pill">${escapeHtml(`${quests.filter((quest) => questState[quest.id]).length}/${quests.length} done`)}</span>
    </div>
    <div class="fix-quest-grid">
      ${quests.map((quest) => `
        <article class="fix-quest-card ${questState[quest.id] ? 'done' : ''}">
          <span class="guide-kicker">${escapeHtml(questState[quest.id] ? 'Completed' : quest.label)}</span>
          <strong>${escapeHtml(quest.title)}</strong>
          <span>${escapeHtml(quest.action || quest.copy)}</span>
          <small>${escapeHtml(`${quest.source} · Rescan to verify, not certify.`)}</small>
          <div class="daily-value-actions">
            <button class="button-link ghost" type="button" data-complete-fix-quest="${escapeHtml(quest.id)}">${questState[quest.id] ? 'Reopen' : 'Mark done'}</button>
            <button class="button-link ghost" type="button" data-open-fix-guide="${escapeHtml(fixGuideForText(`${quest.title} ${quest.action}`).id)}">Open fix guide</button>
          </div>
        </article>
      `).join('')}
    </div>
  `;
}

function renderRoomComparePanel(job, summary = {}, hazards = [], comparison = null) {
  if (!roomComparePanel) {
    return;
  }
  if (!job || job.status !== 'complete') {
    roomComparePanel.innerHTML = emptyMarkup('Room Compare Mode appears when a report or Home Journal has a useful comparison.');
    return;
  }
  const entries = journalEntries().filter((entry) => entry.lastJobId !== job.job_id);
  const currentScore = typeof summary.room_score === 'number' ? Math.round(summary.room_score) : null;
  const peer = entries.find((entry) => entry.roomLabel !== (job.room_label || summary.room_label)) || entries[0] || null;
  const delta = comparison && typeof comparison.score_delta === 'number'
    ? `${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta} vs last same-room scan`
    : 'No same-room delta yet';
  roomComparePanel.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Room Compare Mode</span>
        <h3>Compare progress without pretending it is a measurement lab.</h3>
        <p>Use same-room labels for before/after checks; use room-vs-room comparison only as a prioritization nudge.</p>
      </div>
      <button class="button-link ghost" type="button" data-room-compare-open="true">Log compare view</button>
    </div>
    <div class="room-compare-grid">
      <article class="compare-card">
        <span class="guide-kicker">Current room</span>
        <strong>${escapeHtml(job.room_label || summary.room_label || 'Current report')}</strong>
        <p>${escapeHtml(currentScore === null ? 'Calm Score pending' : `${currentScore}/100 Calm Score`)}</p>
      </article>
      <article class="compare-card">
        <span class="guide-kicker">Same-room progress</span>
        <strong>${escapeHtml(delta)}</strong>
        <p>${escapeHtml(comparison?.summary || 'Rescan with the same room label to unlock a cleaner before/after comparison.')}</p>
      </article>
      <article class="compare-card">
        <span class="guide-kicker">Room-vs-room nudge</span>
        <strong>${escapeHtml(peer?.roomLabel || 'Save another room')}</strong>
        <p>${escapeHtml(peer ? `${peer.lastScore ?? 'Pending'} Calm Score · ${peer.topAction || 'Review brief'}` : 'Home Journal needs another saved room before room-vs-room comparison is useful.')}</p>
      </article>
    </div>
    ${renderVisualBeforeAfter(comparison)}
  `;
}

function renderSmartRescanCoach(job, summary = {}, hazards = [], fixFirst = [], recommendations = [], scanQuality = {}, comparison = null) {
  if (!smartRescanCoach) {
    return;
  }
  if (!job || job.status !== 'complete') {
    smartRescanCoach.innerHTML = emptyMarkup('Smart Rescan Coach will explain how to repeat the scan after a fix.');
    return;
  }
  const roomLabel = job.room_label || summary.room_label || 'this room';
  const topFix = fixFirst[0]?.title || recommendations[0]?.title || hazards[0]?.hazard_title || 'the first visible fix';
  const qualityNote = scanQuality.status === 'weak' || summary.rescan_recommended
    ? 'The last scan looked weak, so keep the route slower and the evidence area in frame longer.'
    : 'The last scan can serve as a baseline if you repeat the same path.';
  smartRescanCoach.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Smart Rescan Coach</span>
        <h3>Make the before/after check fair.</h3>
        <p>${escapeHtml(qualityNote)}</p>
      </div>
      <button class="button-link ghost" type="button" data-smart-rescan-open="true">Use coach</button>
    </div>
    <ul class="rescan-coach-list">
      <li>Reuse the exact room label: <strong>${escapeHtml(roomLabel)}</strong>.</li>
      <li>Start from the same doorway or corner and move at the same pace.</li>
      <li>Keep <strong>${escapeHtml(topFix)}</strong> visible for a full pause after the fix.</li>
      <li>Look for a clearer path, fewer edge objects, or a higher Calm Score. If not visible, treat it as “watch later.”</li>
      <li>${escapeHtml(comparison?.summary || 'No same-room comparison yet. The next scan can become the baseline proof point.')}</li>
    </ul>
  `;
}

function renderEvidenceStoryPanel(job, summary = {}, hazards = [], evidence = [], scanQuality = {}) {
  if (!evidenceStoryPanel) {
    return;
  }
  if (!job || job.status !== 'complete') {
    evidenceStoryPanel.innerHTML = emptyMarkup('Evidence Story Mode will turn evidence frames into a short what-ATLAS-noticed sequence.');
    return;
  }
  if (!evidence.length) {
    evidenceStoryPanel.innerHTML = `
      <div class="section-head compact">
        <div>
          <span class="guide-kicker">Evidence Story Mode</span>
          <h3>No evidence story yet.</h3>
          <p>This report has no stored evidence frames, so treat it as a rescan prompt rather than proof.</p>
        </div>
      </div>
    `;
    return;
  }
  const strongest = [...evidence].sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))[0];
  const uncertainty = scanQuality.status === 'weak' || summary.rescan_recommended
    ? 'Scan quality or coverage was weak, so ATLAS-0 recommends a cautious follow-up.'
    : summary.confidence_label || 'Evidence is approximate and should be reviewed.';
  const topFix = (job.fix_first || [])[0]?.title || (job.recommendations || [])[0]?.title || hazards[0]?.hazard_title || 'Pick one small fix';
  const story = [
    { label: 'First useful frame', title: evidence[0]?.caption || 'First evidence frame', copy: renderEvidenceWhy(evidence[0], 0, hazards) },
    { label: 'Strongest evidence', title: strongest?.caption || 'Strongest frame', copy: `${Math.round(Number(strongest?.confidence || 0) * 100)}% frame confidence. Review before acting.` },
    { label: 'Uncertainty note', title: summary.confidence_label || 'Approximate grounding', copy: uncertainty },
    { label: 'Recommended fix', title: topFix, copy: 'Make one small change, then rescan the same route to verify progress.' },
  ];
  evidenceStoryPanel.innerHTML = `
    <div class="section-head compact">
      <div>
        <span class="guide-kicker">Evidence Story Mode</span>
        <h3>What ATLAS noticed, in sequence.</h3>
        <p>This is a narrative over stored evidence frames, not a stronger model claim.</p>
      </div>
      <button class="button-link ghost" type="button" data-evidence-story-open="true">Log story view</button>
    </div>
    <div class="evidence-story-grid">
      ${story.map((step, index) => `
        <article class="story-step-card">
          <span class="guide-kicker">${String(index + 1).padStart(2, '0')} · ${escapeHtml(step.label)}</span>
          <strong>${escapeHtml(step.title)}</strong>
          <p>${escapeHtml(step.copy)}</p>
        </article>
      `).join('')}
    </div>
  `;
}

function renderRoomScorecard(job, summary, hazards, fixFirst, recommendations, comparison, scanQuality) {
  const score = typeof summary.room_score === 'number' ? Math.round(summary.room_score) : null;
  const scoreLabel = score === null ? 'Pending' : `${score}/100`;
  const topRisk = hazards[0]?.hazard_title || hazards[0]?.object_label || summary.top_hazard_label || 'No top risk yet';
  const quickWin = fixFirst[0]?.title || recommendations[0]?.title || 'Review the report and pick one small fix';
  const confidence = summary.confidence_label || (scanQuality.status ? `${capitalize(scanQuality.status)} scan` : 'Unknown');
  const delta = comparison && typeof comparison.score_delta === 'number'
    ? `${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta} vs last scan`
    : 'First saved baseline';

  return `
    <div class="section-head compact">
      <div>
        <h3>Calm Score card</h3>
        <p>A shareable, human-readable summary of what changed, attention areas, and what to fix today.</p>
      </div>
      <span class="pill">${escapeHtml(job.is_sample ? 'Sample scorecard' : 'Share-ready')}</span>
    </div>
    <div class="scorecard-grid">
      <article class="score-tile">
        <span>Calm Score</span>
        <strong>${escapeHtml(scoreLabel)}</strong>
        <p>${escapeHtml(summary.room_score_summary || summary.room_score_band || 'Score appears after a completed scan.')}</p>
      </article>
      <article class="score-tile">
        <span>Progress</span>
        <strong>${escapeHtml(delta)}</strong>
        <p>${escapeHtml(comparison?.summary || 'Reuse the same room label to compare before and after scans.')}</p>
      </article>
      <article class="score-tile">
        <span>Attention area</span>
        <strong>${escapeHtml(topRisk)}</strong>
        <p>${escapeHtml(summary.top_severity ? `${capitalize(summary.top_severity)} severity` : 'No high-confidence severity yet.')}</p>
      </article>
      <article class="score-tile">
        <span>Confidence</span>
        <strong>${escapeHtml(confidence)}</strong>
        <p>${escapeHtml(scanQuality.capture_summary || summary.coverage_summary || 'Trust notes explain where the scan is weaker.')}</p>
      </article>
    </div>
    <article class="room-win-card">
      <div>
        <span>Quick win</span>
        <p>${escapeHtml(quickWin)}</p>
      </div>
      <button class="button-link ghost" type="button" data-share-room-win="true">Copy room win</button>
    </article>
  `;
}

function renderChallengeResultCard(job, summary = {}, hazards = [], fixFirst = [], recommendations = [], comparison = null) {
  if (!challengeResultCard) {
    return;
  }
  if (!job || job.status !== 'complete') {
    challengeResultCard.innerHTML = emptyMarkup('Complete a scan from a Room Safety Challenge to unlock a shareable room-win card.');
    return;
  }

  const challenge = challengeForJob(job);
  const topAction = fixFirst[0]?.title
    || recommendations[0]?.title
    || hazards[0]?.hazard_title
    || summary.top_hazard_label
    || 'Review the top finding';
  const score = typeof summary.room_score === 'number' ? `${Math.round(summary.room_score)}/100` : 'Screened';
  const delta = comparison && typeof comparison.score_delta === 'number'
    ? `${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta} vs last scan`
    : 'Baseline ready';
  const completed = new Set(readDailyMissionState().completedChallengeIds || []).has(challenge.id);

  challengeResultCard.innerHTML = `
    <div class="challenge-result-head">
      <div>
        <span class="guide-kicker">Challenge completed</span>
        <h3>${escapeHtml(challenge.title)}</h3>
        <p>${escapeHtml(challenge.resultPrompt || 'Share one practical room win, then rescan after the fix.')}</p>
      </div>
      <span class="pill">${escapeHtml(completed ? 'Logged locally' : challenge.badge || 'Room win')}</span>
    </div>
    <div class="challenge-result-grid">
      <article class="challenge-result-main">
        <span class="guide-kicker">${escapeHtml(challenge.winLabel || 'Room win')}</span>
        <strong>${escapeHtml(topAction)}</strong>
        <p>${escapeHtml(`${score} room score. ${summary.confidence_label || 'Approximate grounding'}. Decision support only, not safety certification.`)}</p>
      </article>
      <article class="challenge-result-side">
        <span class="guide-kicker">Before / after loop</span>
        <strong>${escapeHtml(delta)}</strong>
        <p>${escapeHtml(comparison?.summary || 'Reuse the same room label after one fix to compare the next scan against this baseline.')}</p>
      </article>
    </div>
    <div class="report-loop-actions">
      <button class="button-link ghost" type="button" data-copy-challenge-win="true">Copy challenge win</button>
      <button class="button-link ghost" type="button" data-start-rescan="true">Start before/after rescan</button>
      <button class="button-link ghost ${completed ? 'disabled' : ''}" type="button" data-complete-challenge="${escapeHtml(challenge.id)}" ${completed ? 'disabled' : ''}>${completed ? 'Challenge logged' : 'Mark challenge complete'}</button>
    </div>
  `;
}

function buildChecklistItems(fixFirst, recommendations, hazards) {
  const items = [
    ...fixFirst.map((item) => ({
      title: item.title || 'Fix first',
      action: item.action || item.why || '',
      source: 'Priority',
    })),
    ...recommendations.map((item) => ({
      title: item.title || 'Recommendation',
      action: item.action || item.why || '',
      source: 'Recommendation',
    })),
  ];

  if (!items.length && hazards.length) {
    items.push(...hazards.slice(0, 3).map((risk) => ({
      title: risk.hazard_title || risk.object_label || 'Review finding',
      action: risk.what_to_do_next || risk.recommendation || risk.why_it_matters || 'Review this finding before rescanning.',
      source: 'Finding',
    })));
  }

  return items.slice(0, 6);
}

function checklistProgress(job, fixFirst, recommendations, hazards) {
  if (!job?.job_id) {
    return { total: 0, done: 0 };
  }

  const items = buildChecklistItems(fixFirst, recommendations, hazards);
  const checklistState = readChecklistState(job.job_id);
  const done = items.filter((item, index) => checklistState[checklistItemId(item, index)]).length;
  return { total: items.length, done };
}

function renderReportActionLoop(job, summary, hazards, fixFirst, recommendations, evidence, comparison) {
  const progress = checklistProgress(job, fixFirst, recommendations, hazards);
  const roomLabel = job.room_label || summary.room_label || '';
  const hasEvidence = evidence.length > 0;
  const canCompare = Boolean(roomLabel);
  const completion = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;
  const stepState = {
    fix: progress.done > 0,
    verify: hasEvidence,
    rescan: Boolean(comparison),
    share: job.status === 'complete',
  };

  return `
    <div class="report-loop-head">
      <div>
        <span class="guide-kicker">The real product loop</span>
        <h3>Fix, verify, rescan, and show progress.</h3>
        <p>${escapeHtml(progress.total
          ? `${progress.done} of ${progress.total} local checklist item${progress.total === 1 ? '' : 's'} completed.`
          : 'No checklist items yet, but you can still review evidence and rescan the same room.')}</p>
      </div>
      <span class="pill">${escapeHtml(completion ? `${completion}% fixed` : canCompare ? 'Baseline ready' : 'Add room label')}</span>
    </div>
    <div class="report-loop-grid">
      ${REPORT_DECISION_STEPS.map((step, index) => `
        <article class="loop-step ${stepState[step.id] ? 'done' : ''}">
          <span class="loop-index">${String(index + 1).padStart(2, '0')}</span>
          <div>
            <strong>${escapeHtml(step.title)}</strong>
            <p>${escapeHtml(step.copy)}</p>
          </div>
        </article>
      `).join('')}
    </div>
    <div class="coach-prompt">
      ${escapeHtml(hasEvidence
        ? 'Evidence frames are your trust anchor. If the frame does not support the claim, mark it wrong or rescan before acting.'
        : 'This report has limited visual evidence. Treat it as a prompt to rescan, not as proof.')}
    </div>
    <div class="report-loop-actions">
      <button class="button-link ghost" type="button" data-copy-fix-plan="true">Copy fix plan</button>
      <button class="button-link ghost" type="button" data-start-rescan="true">Start same-room rescan</button>
      <button class="button-link ghost" type="button" data-copy-beta-invite="true">Copy beta invite</button>
    </div>
  `;
}

function checklistStorageKey(jobId) {
  return `${FIX_CHECKLIST_STORAGE_KEY}.${jobId}`;
}

function readChecklistState(jobId) {
  const raw = readStoredPreference(checklistStorageKey(jobId));
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeChecklistState(jobId, nextState) {
  writeStoredPreference(checklistStorageKey(jobId), JSON.stringify(nextState));
}

function checklistItemId(item, index) {
  return `${index}:${String(item.title || item.hazard_title || item.object_label || item.action || 'item')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 48)}`;
}

function renderFixChecklist(job, fixFirst, recommendations, hazards) {
  const items = buildChecklistItems(fixFirst, recommendations, hazards);

  if (!items.length) {
    return emptyMarkup('No checklist items were generated for this report.');
  }

  const checklistState = readChecklistState(job.job_id);
  return items.map((item, index) => {
    const itemId = checklistItemId(item, index);
    const checked = Boolean(checklistState[itemId]);
    return `
      <label class="fix-check-item ${checked ? 'done' : ''}" data-checklist-item="${escapeHtml(itemId)}">
        <input type="checkbox" ${checked ? 'checked' : ''} />
        <span class="fix-check-copy">
          <strong>${escapeHtml(item.title)}</strong>
          <span>${escapeHtml(item.action || 'Mark this once you have reviewed or fixed it.')}</span>
          <span class="report-card-meta">${escapeHtml(item.source)}</span>
        </span>
      </label>
    `;
  }).join('');
  renderRoomPersonalityPanel();
  renderRoomHealthTimeline();
  renderHomeCompanion();
}

function renderEvidenceWhy(frame, index, hazards = []) {
  const matchingRisk = hazards.find((risk) => (
    risk.object_id && frame.object_id && risk.object_id === frame.object_id
  )) || hazards[index % Math.max(1, hazards.length)] || {};
  if (frame.caption) {
    return frame.caption;
  }
  if (matchingRisk.hazard_title || matchingRisk.object_label) {
    return `${matchingRisk.hazard_title || matchingRisk.object_label} support frame`;
  }
  return 'Evidence crop supporting the current room brief.';
}

function renderEvidenceTimeline(evidence, hazards = []) {
  if (!evidence.length) {
    return `
      <div class="timeline-empty">
        <strong>No evidence timeline yet</strong>
        <span>Low-evidence reports should be treated as prompts to rescan, not proof.</span>
      </div>
    `;
  }
  return evidence.map((frame, index) => {
    const label = typeof frame.timestamp_s === 'number'
      ? `${frame.timestamp_s.toFixed(1)}s`
      : typeof frame.frame_index === 'number'
        ? `Frame ${frame.frame_index}`
        : `Frame ${index + 1}`;
    const confidence = typeof frame.confidence === 'number' ? `${Math.round(frame.confidence * 100)}%` : 'approx';
    const severity = (hazards[index]?.severity || frame.severity || 'low').toLowerCase();
    const active = Number(state.activeEvidenceIndex || 0) === index;
    return `
      <button class="timeline-marker ${escapeHtml(severity)} ${active ? 'active' : ''}" type="button" data-evidence-target="${index}">
        <span class="timeline-dot"></span>
        <span class="timeline-copy">
          <strong>${escapeHtml(label)}</strong>
          <small>${escapeHtml(confidence)} · ${escapeHtml(renderEvidenceWhy(frame, index, hazards))}</small>
        </span>
      </button>
    `;
  }).join('');
}

function renderConfidenceExplainer(risk, job, evidence = []) {
  const reasoning = risk?.reasoning || {};
  const evidenceIds = Array.isArray(reasoning.evidence_ids) ? reasoning.evidence_ids : [];
  const evidenceCount = evidenceIds.length || evidence.filter((frame) => (
    risk.object_id && frame.object_id && risk.object_id === frame.object_id
  )).length || Number(risk.replay?.frame_count || 0);
  const confidence = typeof risk.confidence === 'number'
    ? `${Math.round(risk.confidence * 100)}%`
    : risk.confidence_label || 'Approximate';
  const locationConfidence = risk.location_confidence_label || risk.location_label || 'Approximate location';
  const uncertainty = Array.isArray(reasoning.confidence_reasons) && reasoning.confidence_reasons.length
    ? reasoning.confidence_reasons.slice(0, 2).join(' ')
    : isLowConfidenceRisk(risk)
      ? 'Limited evidence or scan quality makes this finding weaker.'
      : 'Confidence is based on available evidence frames and rule support.';
  const action = job?.summary?.rescan_recommended || isLowConfidenceRisk(risk)
    ? 'Rescan or watch before acting on small details.'
    : risk.severity === 'high'
      ? 'Act on the top fix after confirming the evidence frame.'
      : 'Review, fix if easy, and keep it on the watch list.';
  return `
    <details class="confidence-explainer" data-confidence-explainer="${escapeHtml(risk.hazard_code || risk.object_id || risk.hazard_title || 'finding')}">
      <summary>Confidence explainer</summary>
      <div class="confidence-explainer-grid">
        <span><strong>${evidenceCount}</strong><small>evidence references</small></span>
        <span><strong>${escapeHtml(confidence)}</strong><small>finding confidence</small></span>
        <span><strong>${escapeHtml(locationConfidence)}</strong><small>location confidence</small></span>
      </div>
      <p>${escapeHtml(uncertainty)}</p>
      <p><strong>Act / watch / rescan:</strong> ${escapeHtml(action)}</p>
    </details>
  `;
}

function renderEvidenceFrameOverlay(frame) {
  const labelParts = [
    frame.object_label || frame.hazard_title || 'Evidence',
    typeof frame.confidence === 'number' ? `${Math.round(frame.confidence * 100)}%` : '',
  ].filter(Boolean);
  if (!labelParts.length) {
    return '';
  }

  return `
    <span class="evidence-frame-overlay" aria-hidden="true"></span>
    <span class="evidence-frame-label">${escapeHtml(labelParts.join(' · '))}</span>
  `;
}

function buildRoomWinShareText(job) {
  const summary = job.summary || {};
  const comparison = job.room_comparison || null;
  const roomLabel = job.room_label || summary.room_label || 'Room';
  const score = typeof summary.room_score === 'number' ? `${summary.room_score}/100` : 'screened';
  const delta = comparison && typeof comparison.score_delta === 'number'
    ? ` (${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta} vs last scan)`
    : '';
  const topFix = (job.fix_first || [])[0]?.title || (job.recommendations || [])[0]?.title || summary.top_hazard_label || 'one practical next step';
  const confidence = summary.confidence_label || summary.scan_quality_label || 'approximate confidence';
  return [
    `ATLAS-0 room win: ${roomLabel} scored ${score}${delta}.`,
    `Quick win: ${topFix}.`,
    `Confidence: ${confidence}. Decision support only, not safety certification.`,
  ].join(' ');
}

function buildChallengeWinText(job) {
  const summary = job.summary || {};
  const challenge = challengeForJob(job);
  const comparison = job.room_comparison || null;
  const roomLabel = job.room_label || summary.room_label || challenge.roomLabel || 'Room';
  const score = typeof summary.room_score === 'number' ? `${Math.round(summary.room_score)}/100` : 'screened';
  const delta = comparison && typeof comparison.score_delta === 'number'
    ? ` (${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta} vs last scan)`
    : '';
  const topAction = (job.fix_first || [])[0]?.title
    || (job.recommendations || [])[0]?.title
    || summary.top_hazard_label
    || 'one practical next step';
  return [
    `ATLAS-0 challenge: ${challenge.title}`,
    `${roomLabel} scored ${score}${delta}.`,
    `${challenge.winLabel || 'Room win'}: ${topAction}.`,
    'Decision support only, not safety certification.',
  ].join('\n');
}

function currentShareCardStyle() {
  const style = readStoredPreference(SHARE_CARD_STYLE_STORAGE_KEY) || 'quick-win';
  return ['landlord', 'family', 'quick-win', 'before-after', 'private-pdf'].includes(style) ? style : 'quick-win';
}

function shareStyleLabel(style = currentShareCardStyle()) {
  return {
    landlord: 'Landlord summary',
    family: 'Family summary',
    'quick-win': 'Quick room win',
    'before-after': 'Before/after card',
    'private-pdf': 'Private PDF wording',
  }[style] || 'Quick room win';
}

function buildShareCardText(job, style = currentShareCardStyle()) {
  if (!job || job.status !== 'complete') {
    return 'ATLAS-0 Room Safety Brief: upload one room walkthrough to get top actions, evidence frames, confidence signals, and a decision-support PDF.';
  }

  const summary = job.summary || {};
  const roomLabel = job.room_label || summary.room_label || 'Room';
  const score = typeof summary.room_score === 'number' ? `${summary.room_score}/100 Calm Score` : 'screened';
  const topAction = (job.fix_first || [])[0]?.title
    || (job.recommendations || [])[0]?.title
    || summary.top_hazard_label
    || 'review the top finding';
  const confidence = summary.confidence_label || summary.scan_quality_label || 'approximate confidence';
  const evidence = job.evidence_frames || [];
  const selectedEvidence = selectedEvidenceFrames(job);
  const blurredEvidence = blurredEvidenceIds(job);
  const evidenceLine = evidence.length
    ? `Evidence selected for local sharing: ${selectedEvidence.length}/${evidence.length}. Local blur previews: ${blurredEvidence.size}.`
    : 'No evidence frames are selected for local sharing.';
  const link = reportDeepLink(job);
  const comparison = job.room_comparison || null;
  const delta = comparison && typeof comparison.score_delta === 'number'
    ? `${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta}`
    : 'baseline';
  const templates = {
    landlord: [
      `ATLAS-0 room note for ${roomLabel}`,
      `Calm Score: ${score}. Main attention area: ${topAction}.`,
      evidenceLine,
      'This is a tenant-generated decision-support note with evidence references, not a certification or inspection.',
      link ? `Report/PDF: ${link}` : '',
    ],
    family: [
      `Room check for ${roomLabel}`,
      `Top thing to fix: ${topAction}.`,
      `Confidence: ${confidence}. Please confirm the evidence frame before acting.`,
      evidenceLine,
      'ATLAS-0 helps prioritize, but it does not certify safety.',
      link ? `Report: ${link}` : '',
    ],
    'quick-win': [
      `ATLAS-0 room win: ${roomLabel}`,
      `Calm Score: ${score}. Quick fix: ${topAction}.`,
      evidenceLine,
      'Fix one thing, then rescan with the same room label to show progress.',
      `Confidence: ${confidence}. Decision support only.`,
      link ? `Report: ${link}` : '',
    ],
    'before-after': [
      `ATLAS-0 before/after card: ${roomLabel}`,
      `Current Calm Score: ${score}. Change: ${delta}.`,
      `Top action: ${topAction}.`,
      evidenceLine,
      comparison?.summary || 'Reuse this same room label after a fix to unlock a comparison.',
      'Decision support only, not safety certification.',
    ],
    'private-pdf': [
      `Private ATLAS-0 Safety Brief: ${roomLabel}`,
      `Calm Score: ${score}. Top action: ${topAction}. Confidence: ${confidence}.`,
      evidenceLine,
      'Keep this wording private unless you want to share the PDF. Uploaded/report artifacts follow the configured retention policy.',
      link ? `PDF/report: ${link}` : '',
    ],
  };
  return (templates[style] || templates['quick-win']).filter(Boolean).join('\n');
}

function renderShareCardPreview(job) {
  if (!shareCardCopy) {
    return;
  }

  if (!job || job.status !== 'complete') {
    shareCardCopy.innerHTML = `
      <strong>No room brief yet</strong>
      <p>Run a scan or open the sample report to generate a concise summary with score, top action, confidence, and decision-support wording.</p>
    `;
    return;
  }

  const summary = job.summary || {};
  const roomLabel = job.room_label || summary.room_label || 'Room';
  const topAction = (job.fix_first || [])[0]?.title
    || (job.recommendations || [])[0]?.title
    || summary.top_hazard_label
    || 'Review the top finding';
  const score = typeof summary.room_score === 'number' ? `${summary.room_score}/100 Calm Score` : 'Screened';
  const evidence = job.evidence_frames || [];
  const selectedEvidence = selectedEvidenceFrames(job);
  const blurredEvidence = blurredEvidenceIds(job);
  const style = currentShareCardStyle();
  if (shareCardStyleInput) {
    shareCardStyleInput.value = style;
  }
  shareCardCopy.innerHTML = `
    <span class="guide-kicker">ATLAS-0 ${escapeHtml(shareStyleLabel(style))}</span>
    <strong>${escapeHtml(roomLabel)} · ${escapeHtml(score)}</strong>
    <p>${escapeHtml(topAction)}. ${escapeHtml(summary.confidence_label || 'Approximate grounding')}. Evidence selected ${selectedEvidence.length}/${evidence.length}; local blur previews ${blurredEvidence.size}. Decision support only, not safety certification.</p>
  `;
}

function betaSharePrompt() {
  const index = Math.floor(Date.now() / 86_400_000) % BETA_SHARE_PROMPTS.length;
  return BETA_SHARE_PROMPTS[index];
}

function buildFixPlanText(job) {
  const summary = job.summary || {};
  const roomLabel = job.room_label || summary.room_label || 'Room';
  const actions = buildChecklistItems(job.fix_first || [], job.recommendations || [], job.risks || [])
    .slice(0, 4)
    .map((item, index) => `${index + 1}. ${item.title}: ${item.action || 'Review and fix if needed.'}`);
  const header = `ATLAS-0 fix plan for ${roomLabel}`;
  return [header, ...actions, `Report: ${reportDeepLink(job)}`].filter(Boolean).join('\n');
}

function buildFixTodayText(job) {
  const summary = job.summary || {};
  const first = (job.fix_first || [])[0] || (job.recommendations || [])[0] || (job.risks || [])[0] || {};
  const roomLabel = job.room_label || summary.room_label || 'Room';
  const title = first.title || first.hazard_title || first.object_label || summary.top_hazard_label || 'Fix one thing';
  const action = first.action || first.what_to_do_next || first.recommendation || first.why || 'Review the top finding and make the smallest practical fix.';
  return [
    `ATLAS-0 Fix Today for ${roomLabel}`,
    `${title}: ${action}`,
    'After the fix, rescan with the same room label if you want the before/after to show up.',
    'Decision support only, not safety certification.',
  ].join('\n');
}

async function copyBetaInvite(surface = 'unknown') {
  const text = betaSharePrompt();
  await copyText(text);
  await trackProductEvent('beta_invite_copied', { surface });
  showToast('Beta invite copied.');
}

function renderReplayPreview(risk) {
  const replay = risk?.replay;
  if (!replay?.image_url) {
    return '';
  }

  return `
    <div class="finding-replay">
      <div class="finding-replay-copy">
        <strong>Evidence replay</strong>
        <span>${escapeHtml(replay.caption || 'Short replay from the strongest supporting crops.')}</span>
      </div>
      <img src="${api.withAccessToken(replay.image_url)}" alt="${escapeHtml(replay.caption || 'Finding replay')}" />
      <div class="finding-replay-meta">
        <span>${Number(replay.frame_count || 0)} supporting frame${Number(replay.frame_count || 0) === 1 ? '' : 's'}</span>
        <span>${escapeHtml(risk.location_label || 'scan area')}</span>
      </div>
    </div>
  `;
}

function renderReasoningPanel(risk) {
  const reasoning = risk?.reasoning || {};
  const objectSnapshot = reasoning.object_snapshot || {};
  const ruleHits = Array.isArray(reasoning.rule_hits) ? reasoning.rule_hits : [];
  const evidenceIds = Array.isArray(reasoning.evidence_ids) ? reasoning.evidence_ids : [];
  const confidenceReasons = Array.isArray(reasoning.confidence_reasons) ? reasoning.confidence_reasons : [];
  const facts = [];

  if (objectSnapshot.material) {
    facts.push(`Material: ${objectSnapshot.material}`);
  }
  if (Number(objectSnapshot.estimated_height_m || 0) > 0) {
    facts.push(`Estimated height ${Number(objectSnapshot.estimated_height_m).toFixed(2)} m`);
  }
  if (Number(objectSnapshot.estimated_width_m || 0) > 0) {
    facts.push(`Estimated width ${Number(objectSnapshot.estimated_width_m).toFixed(2)} m`);
  }
  if (Number(objectSnapshot.observation_count || 0) > 0) {
    facts.push(`${Number(objectSnapshot.observation_count)} supporting observation${Number(objectSnapshot.observation_count) === 1 ? '' : 's'}`);
  }

  if (!ruleHits.length && !facts.length && !evidenceIds.length && !confidenceReasons.length) {
    return '';
  }

  return `
    <details class="reasoning-panel">
      <summary>Why ATLAS-0 surfaced this finding</summary>
      <div class="reasoning-grid">
        ${ruleHits.length ? `
          <div class="reasoning-block">
            <strong>Triggered rules</strong>
            <ul>
              ${ruleHits.map((hit) => `<li>${escapeHtml(hit)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
        ${facts.length ? `
          <div class="reasoning-block">
            <strong>Object snapshot</strong>
            <ul>
              ${facts.map((fact) => `<li>${escapeHtml(fact)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
        ${evidenceIds.length ? `
          <div class="reasoning-block">
            <strong>Evidence references</strong>
            <div class="reasoning-chips">
              ${evidenceIds.map((id) => `<span>${escapeHtml(id)}</span>`).join('')}
            </div>
          </div>
        ` : ''}
        ${confidenceReasons.length ? `
          <div class="reasoning-block">
            <strong>Confidence calibration</strong>
            <ul>
              ${confidenceReasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
      </div>
    </details>
  `;
}

function renderProcessGuidance(job) {
  if (!job) {
    return `
      <article class="guidance-card">
        <strong>Best first scan</strong>
        <p>Choose one room, move steadily, and keep shelves, tables, and corners visible for a moment so weaker hazards do not look stronger than the scan supports.</p>
      </article>
      <article class="guidance-card">
        <strong>What you’ll get</strong>
        <p>A ranked screening report with top hazards, evidence crops, approximate locations, confidence labels, and a downloadable PDF.</p>
      </article>
    `;
  }

  if (job.status === 'error') {
    return `
      <article class="guidance-card">
        <strong>Scan needs another try</strong>
        <p>${escapeHtml(job.error || 'The scan could not be processed this time.')}</p>
        <ul class="guidance-list">
          <li>Retry with one room only.</li>
          <li>Keep the motion slower and steadier than feels natural.</li>
          <li>If lighting was poor, add light before rescanning.</li>
        </ul>
      </article>
    `;
  }

  if (job.status === 'complete') {
    const scanQuality = job.scan_quality || {};
    const guidance = scanQuality.retry_guidance || [];
    const summary = job.summary || {};
    return `
      <article class="guidance-card">
        <strong>${escapeHtml(scanQuality.rescan_recommended ? 'Use this report carefully' : 'Report is ready')}</strong>
        <p>${escapeHtml(summary.screening_statement || 'This report flags likely hazards from the uploaded scan. It does not certify that the room is safe.')}</p>
      </article>
      <article class="guidance-card">
        <strong>Next best move</strong>
        <p>${escapeHtml(scanQuality.capture_summary || 'Review the top hazards and trust notes before acting on smaller details.')}</p>
        ${guidance.length ? `<ul class="guidance-list">${guidance.slice(0, 2).map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}
      </article>
    `;
  }

  return `
    <article class="guidance-card">
      <strong>What ATLAS-0 is doing now</strong>
      <p>${escapeHtml(stageExplanation(job.stage))}</p>
    </article>
    <article class="guidance-card">
      <strong>While you wait</strong>
      <p>Keep this tab open. When the report is ready, we’ll switch you straight into the report view.</p>
    </article>
  `;
}

function emptyMarkup(message) {
  return `<div class="empty-card">${escapeHtml(message)}</div>`;
}

function renderPolicyItems(items) {
  return items.map((item) => `
    <div class="policy-item">
      <span>${escapeHtml(item.label)}</span>
      <strong>${escapeHtml(item.value)}</strong>
    </div>
  `).join('');
}

function renderReleaseGates(releaseGates) {
  if (!releaseGates || !Array.isArray(releaseGates.gates)) {
    return '';
  }

  return `
    <p class="subsection-label">Release gates</p>
    ${renderPolicyItems(releaseGates.gates.map((gate) => ({
      label: gate.label || gate.id || 'Gate',
      value: `${gate.passed ? 'Pass' : 'Open'} · ${formatGateValue(gate.actual)} / ${formatGateValue(gate.target)}`,
    })))}
    <p class="meta-copy">${escapeHtml(releaseGates.summary || '')}</p>
  `;
}

function applyUploadGuidance(guidance) {
  const recommended = guidance?.recommended_duration_seconds || {};
  const minSeconds = Number(recommended.min || 20);
  const maxSeconds = Number(recommended.max || guidance?.max_video_duration_seconds || 60);
  if (uploadDurationPill) {
    uploadDurationPill.textContent = `${minSeconds}-${maxSeconds} seconds`;
  }
  if (uploadSizePill) {
    uploadSizePill.textContent = guidance?.max_upload_bytes
      ? `${formatBytes(guidance.max_upload_bytes)} max`
      : 'Limit checked on upload';
  }
  if (uploadGuidanceCopy) {
    const checklist = Array.isArray(guidance?.checklist) ? guidance.checklist : [];
    uploadGuidanceCopy.textContent = checklist[1]
      || 'Record one bright, steady room walkthrough before uploading.';
  }
}

function renderAccessPanels(errorMessage = '') {
  const access = state.accessPolicy;
  const privacy = state.privacyPolicy;
  const settings = state.operatorSettings;
  const tokenStored = Boolean(api.getAccessToken());

  renderSettingsControlCenter();
  accessTokenClear.disabled = !tokenStored;
  syncSettingsAccessStatus();

  if (!privacy) {
    privacyPolicy.innerHTML = emptyMarkup('Privacy defaults are unavailable right now.');
  } else {
    privacyPolicy.innerHTML = `
      <p class="subsection-label">User-visible privacy</p>
      ${renderPolicyItems([
        { label: 'Retention window', value: `${privacy.retention_days} day(s)` },
        { label: 'Keep originals', value: privacy.save_original_uploads ? 'Yes' : 'No by default' },
        { label: 'Text redaction', value: privacy.text_redaction_enabled ? 'Enabled' : 'Disabled' },
        { label: 'Delete support', value: privacy.delete_supported ? 'Available in report view' : 'Unavailable' },
      ])}
      <p class="meta-copy">${escapeHtml(privacy.summary || '')}</p>
    `;
  }

  if (!access) {
    accessBanner.className = 'status-banner';
    accessBanner.innerHTML = '<strong>Could not load access policy.</strong>';
    accessHelp.textContent = 'The hosted access policy could not be loaded from the API.';
    operatorPolicy.innerHTML = emptyMarkup('Operator policy details are unavailable right now.');
    operatorQueue.innerHTML = emptyMarkup('Queue diagnostics are unavailable right now.');
    operatorSystem.innerHTML = emptyMarkup('Deployment diagnostics are unavailable right now.');
    operatorEval.innerHTML = emptyMarkup('Evaluation metrics are unavailable right now.');
    operatorProduct.innerHTML = emptyMarkup('Product metrics are unavailable right now.');
    operatorPruneBtn.disabled = true;
    return;
  }

  const locked = access.requires_token && !settings;
  accessBanner.className = `status-banner ${locked ? 'locked' : 'ready'}`;
  accessBanner.innerHTML = locked
    ? '<strong>Hosted upload/report access is locked.</strong> Add the private-beta token to use protected scans, reports, and diagnostics.'
    : access.requires_token
      ? '<strong>Hosted access is unlocked.</strong> Protected upload/report endpoints are available with the stored token.'
      : access.mode === 'loopback'
        ? '<strong>Local access is open.</strong> Loopback requests can use upload/report flows without a token.'
        : '<strong>Upload/report access is restricted.</strong> This environment is not accepting unauthenticated hosted requests.';

  accessHelp.textContent = locked
    ? (errorMessage || 'Your token stays in this browser only and is sent as a Bearer token for protected Atlas-0 endpoints.')
    : tokenStored
      ? 'A token is stored locally for this browser session and will be appended to protected API requests.'
      : 'No token is stored locally. You only need one when the hosted environment requires it.';

  if (!settings) {
    operatorPolicy.innerHTML = emptyMarkup(
      access.requires_token
        ? 'Enter a valid token to load retention, queue, and access settings.'
        : 'Operator settings are not available yet.',
    );
    operatorQueue.innerHTML = emptyMarkup('Queue diagnostics will appear here once operator access is available.');
    operatorSystem.innerHTML = emptyMarkup('Deployment diagnostics will appear here once operator access is available.');
    operatorEval.innerHTML = emptyMarkup('Evaluation metrics will appear here once operator access is available.');
    operatorProduct.innerHTML = emptyMarkup('Product metrics will appear here once operator access is available.');
    operatorPruneBtn.disabled = true;
    return;
  }

  operatorPruneBtn.disabled = false;

  operatorPolicy.innerHTML = renderPolicyItems([
    { label: 'Access mode', value: settings.access.mode === 'token' ? 'Token protected' : settings.access.mode === 'loopback' ? 'Loopback-friendly' : 'Restricted' },
    { label: 'Primary provider', value: settings.providers.primary_provider || 'unknown' },
    { label: 'Fallback provider', value: settings.providers.fallback_provider || 'None' },
    { label: 'Worker mode', value: settings.uploads.worker_mode || settings.system?.worker_mode || 'unknown' },
    { label: 'Job listing', value: settings.access.enable_job_listing ? 'Enabled' : 'Direct job IDs only' },
    { label: 'Retention window', value: `${settings.uploads.retention_days} day(s)` },
    { label: 'Keep originals', value: settings.uploads.save_original_uploads ? 'Yes' : 'No' },
    { label: 'Artifact backend', value: settings.uploads.artifact_backend || 'unknown' },
    { label: 'Queue depth limit', value: String(settings.uploads.max_queue_depth) },
    { label: 'Retry budget', value: `${settings.uploads.max_job_attempts} attempt(s)` },
  ]);

  operatorQueue.innerHTML = renderPolicyItems([
    { label: 'Workers', value: String(settings.queue.worker_count) },
    { label: 'Configured capacity', value: String(settings.queue.configured_capacity || settings.uploads.max_concurrent_jobs || 0) },
    { label: 'Queued jobs', value: String(settings.queue.queued_jobs) },
    { label: 'Processing jobs', value: String(settings.queue.processing_jobs) },
    { label: 'Failed jobs', value: String(settings.queue.failed_jobs) },
    { label: 'Active claims', value: String(settings.storage.active_claims || 0) },
    { label: 'Stored jobs', value: String(settings.storage.persisted_jobs) },
    { label: 'Storage budget', value: formatBytes(settings.storage.byte_budget || 0) },
    { label: 'Disk used', value: formatBytes(settings.storage.bytes_used || 0) },
    { label: 'Budget used', value: `${settings.storage.usage_percent || 0}%` },
  ]);

  const startupChecks = Array.isArray(settings.system?.startup_checks) ? settings.system.startup_checks : [];
  const startupSummary = settings.system?.startup_summary || '';
  const recentFailures = Array.isArray(settings.system?.recent_failures) ? settings.system.recent_failures : [];
  operatorSystem.innerHTML = `
    <p class="subsection-label">Deployment readiness</p>
    ${renderPolicyItems([
      { label: 'Status', value: settings.system?.deployment_ready ? 'Ready' : 'Needs operator fixes' },
      { label: 'Worker mode', value: settings.system?.worker_mode || 'unknown' },
      { label: 'Service uptime', value: `${Math.round(settings.system?.uptime_seconds || 0)}s` },
      { label: 'Active workers', value: String(settings.system?.active_workers || 0) },
      { label: 'Artifact backend', value: settings.system?.artifact_backend || settings.uploads.artifact_backend || 'unknown' },
      { label: 'Object store root', value: settings.system?.artifact_object_dir || settings.uploads.artifact_object_dir || 'local job storage' },
      { label: 'Storage root', value: settings.system?.storage_root || 'unknown' },
      { label: 'Recent failures', value: String(recentFailures.length) },
    ])}
    <p class="meta-copy">${escapeHtml(startupSummary)}</p>
    ${startupChecks.length ? renderPolicyItems(startupChecks.map((check) => ({
      label: check.name || 'Check',
      value: `${check.status || 'unknown'} · ${check.detail || ''}`,
    }))) : ''}
    ${recentFailures.length ? `<p class="subsection-label">Recent job failures</p>${renderPolicyItems(recentFailures.map((failure) => ({
      label: `${failure.job_id || 'job'} · ${failure.stage || 'stage'}`,
      value: `${failure.will_retry ? 'Retrying' : 'Terminal'} · ${failure.error || 'Unknown failure'}`,
    })))}` : '<p class="meta-copy">No recent terminal worker failures recorded.</p>'}
  `;

  operatorEval.innerHTML = `
    <p class="subsection-label">Evaluation loop</p>
    ${renderPolicyItems([
      { label: 'Reviewed jobs', value: String(settings.evaluation.reviewed_jobs || 0) },
      { label: 'Benchmarked jobs', value: String(settings.evaluation.benchmarked_jobs || 0) },
      { label: 'Benchmark match rate', value: `${Math.round((settings.evaluation.benchmark_match_rate || 0) * 100)}%` },
      { label: 'Missed-hazard jobs', value: String(settings.evaluation.jobs_with_missed_hazards || 0) },
      { label: 'False-positive job rate', value: `${Math.round((settings.evaluation.false_positive_job_rate || 0) * 100)}%` },
      { label: 'Average review coverage', value: `${Math.round((settings.evaluation.avg_review_coverage || 0) * 100)}%` },
      { label: 'Committed eval fixtures', value: String(settings.evaluation.seed_fixture_count || 0) },
      { label: 'Saved eval candidates', value: String(settings.evaluation.saved_eval_candidates || 0) },
      { label: 'Review-ready eval cases', value: `${settings.evaluation.available_eval_cases || 0} / ${settings.evaluation.target_corpus_size || 0}` },
    ])}
    ${renderReleaseGates(settings.evaluation.release_gates)}
  `;

  operatorProduct.innerHTML = `
    <p class="subsection-label">Beta product loop</p>
    ${renderPolicyItems([
      { label: 'Upload success rate', value: `${Math.round((settings.product.upload_success_rate || 0) * 100)}%` },
      { label: 'Rescan recommended rate', value: `${Math.round((settings.product.rescan_recommended_rate || 0) * 100)}%` },
      { label: 'Report usefulness rate', value: `${Math.round((settings.product.report_usefulness_rate || 0) * 100)}%` },
      { label: 'Average report time', value: `${settings.product.avg_report_seconds || 0}s` },
      { label: 'Completed jobs', value: String(settings.product.completed_jobs || 0) },
      { label: 'Terminal jobs', value: String(settings.product.terminal_jobs || 0) },
      { label: 'Labeled rooms', value: String(settings.product.labeled_rooms || 0) },
      { label: 'Repeat-scan rooms', value: String(settings.product.repeat_scan_rooms || 0) },
      { label: 'Waitlist signups', value: String(settings.product.waitlist_signups || 0) },
      { label: 'Beta onboarding starts', value: String(settings.product.beta_onboarding_events || 0) },
      { label: 'Sample report opens', value: String(settings.product.sample_report_opens || 0) },
      { label: 'Sample journey opens', value: String(settings.product.sample_journey_events || 0) },
      { label: 'Sample CTA taps', value: String(settings.product.sample_cta_events || 0) },
      { label: 'First-run starts', value: String(settings.product.first_run_events || 0) },
      { label: 'Landing sections viewed', value: String(settings.product.landing_section_events || 0) },
      { label: 'Share events', value: String(settings.product.share_events || 0) },
      { label: 'Share card copies', value: String(settings.product.share_card_events || 0) },
      { label: 'Report theme changes', value: String(settings.product.report_theme_events || 0) },
      { label: 'Confidence inspector opens', value: String(settings.product.confidence_inspector_events || 0) },
      { label: 'Beta invite copies', value: String(settings.product.beta_invite_events || 0) },
      { label: 'Room win cards shared', value: String(settings.product.room_win_card_shared_events || 0) },
      { label: 'Room win copies', value: String(settings.product.room_win_events || 0) },
      { label: 'Post-report feedback', value: String(settings.product.post_report_feedback_events || 0) },
      { label: 'Fix plan copies', value: String(settings.product.fix_plan_events || 0) },
      { label: 'Fix Today copies', value: String(settings.product.fix_today_events || 0) },
      { label: 'Fix Quest completions', value: String(settings.product.fix_quest_events || 0) },
      { label: 'Fix Library opens', value: String(settings.product.fix_library_events || 0) },
      { label: 'Fix Guide opens', value: String(settings.product.fix_guide_events || 0) },
      { label: 'One Thing starts', value: String(settings.product.one_thing_started_events || 0) },
      { label: 'One Thing completions', value: String(settings.product.one_thing_completed_events || 0) },
      { label: 'Care calendar opens', value: String(settings.product.room_care_calendar_events || 0) },
      { label: 'Care task completions', value: String(settings.product.room_care_completed_events || 0) },
      { label: 'Care week regenerations', value: String(settings.product.room_care_regenerated_events || 0) },
      { label: 'Room timeline opens', value: String(settings.product.room_health_timeline_events || 0) },
      { label: 'Daily Value settings', value: String(settings.product.settings_daily_value_events || 0) },
      { label: 'Before/after cards', value: String(settings.product.before_after_card_events || 0) },
      { label: 'Mystery mode starts', value: String(settings.product.mystery_mode_events || 0) },
      { label: 'Personal modes selected', value: String(settings.product.personal_mode_events || 0) },
      { label: 'Sample gallery opens', value: String(settings.product.sample_gallery_events || 0) },
      { label: 'Field note expands', value: String(settings.product.field_note_events || 0) },
      { label: 'Evidence stories opened', value: String(settings.product.evidence_story_events || 0) },
      { label: 'Map preview opens', value: String(settings.product.room_map_preview_events || 0) },
      { label: 'Room Compare opens', value: String(settings.product.room_compare_events || 0) },
      { label: 'Room Passport opens', value: String(settings.product.room_passport_events || 0) },
      { label: 'Room personalities viewed', value: String(settings.product.room_personality_events || 0) },
      { label: 'Room Playbooks starts', value: String(settings.product.room_playbook_events || 0) },
      { label: 'Fix verification starts', value: String(settings.product.fix_verification_events || 0) },
      { label: 'Fix verification copies', value: String(settings.product.fix_verification_copy_events || 0) },
      { label: 'Evidence frame focuses', value: String(settings.product.evidence_frame_focus_events || 0) },
      { label: 'Share studio copies', value: String(settings.product.share_card_studio_events || 0) },
      { label: 'Confidence explainers', value: String(settings.product.confidence_explainer_events || 0) },
      { label: 'Welcome tour completions', value: String(settings.product.welcome_tour_events || 0) },
      { label: 'Home Pulse opens', value: String(settings.product.home_pulse_events || 0) },
      { label: 'Weekly recap copies', value: String(settings.product.weekly_recap_events || 0) },
      { label: 'Weekly challenges completed', value: String(settings.product.weekly_challenge_events || 0) },
      { label: 'Home Bingo completions', value: String(settings.product.home_bingo_events || 0) },
      { label: 'Daily mission starts', value: String(settings.product.daily_mission_events || 0) },
      { label: 'Room ritual starts', value: String(settings.product.room_ritual_events || 0) },
      { label: 'Room ritual completions', value: String(settings.product.room_ritual_completed_events || 0) },
      { label: 'Home Journal opens', value: String(settings.product.home_journal_events || 0) },
      { label: 'Reminder clicks', value: String(settings.product.room_reminder_events || 0) },
      { label: 'Seasonal pack starts', value: String(settings.product.seasonal_pack_events || 0) },
      { label: 'Seasonal packs selected', value: String(settings.product.seasonal_pack_selected_events || 0) },
      { label: 'Smart Rescan Coach opens', value: String(settings.product.smart_rescan_coach_events || 0) },
      { label: 'Capture coach checks', value: String(settings.product.capture_coach_events || 0) },
      { label: 'Same-room rescan starts', value: String(settings.product.same_room_rescan_events || 0) },
      { label: 'Rescan prompt clicks', value: String(settings.product.rescan_prompt_events || 0) },
      { label: 'PDF downloads', value: String(settings.product.pdf_download_events || 0) },
      { label: 'PDF CTA taps', value: String(settings.product.pdf_export_click_events || 0) },
      { label: 'Preflight failures', value: String(settings.product.scan_preflight_failed_events || 0) },
      { label: 'CTA start-scan taps', value: String(settings.product.cta_start_scan_events || 0) },
    ])}
    ${renderBetaInbox(settings.beta_inbox)}
  `;
}

function renderBetaInbox(inbox) {
  if (!inbox) {
    return '<p class="meta-copy">Beta inbox will appear after operator access is available.</p>';
  }
  const funnel = inbox.funnel || {};
  const waitlist = Array.isArray(inbox.recent_waitlist) ? inbox.recent_waitlist : [];
  const failures = Array.isArray(inbox.failed_uploads) ? inbox.failed_uploads : [];
  const negative = Array.isArray(inbox.negative_feedback_reports) ? inbox.negative_feedback_reports : [];
  const reviewNeeded = Array.isArray(inbox.review_needed_reports) ? inbox.review_needed_reports : [];
  const missed = Array.isArray(inbox.missed_hazard_notes) ? inbox.missed_hazard_notes : [];
  const readiness = inbox.eval_candidate_readiness || {};
  return `
    <p class="subsection-label">Beta inbox</p>
    <p class="meta-copy">${escapeHtml(inbox.summary || 'Review beta demand, failures, feedback, and eval readiness before inviting more users.')}</p>
    ${renderPolicyItems([
      { label: 'CTA to upload', value: String(funnel.cta_start_scan || 0) },
      { label: 'Upload starts', value: String(funnel.upload_started || 0) },
      { label: 'Upload completes', value: String(funnel.upload_completed || 0) },
      { label: 'Report views', value: String(funnel.report_viewed || 0) },
      { label: 'Report theme changes', value: String(funnel.report_theme_changed || 0) },
      { label: 'First-run starts', value: String(funnel.first_run_started || 0) },
      { label: 'Beta onboarding starts', value: String(funnel.beta_onboarding_started || 0) },
      { label: 'Sample journeys opened', value: String(funnel.sample_journey_opened || 0) },
      { label: 'Mystery mode starts', value: String(funnel.mystery_mode_started || 0) },
      { label: 'Personal modes selected', value: String(funnel.personal_mode_selected || 0) },
      { label: 'Sample gallery opens', value: String(funnel.sample_gallery_opened || 0) },
      { label: 'Before/after copies', value: String(funnel.before_after_card_copied || 0) },
      { label: 'Field note expands', value: String(funnel.field_note_expanded || 0) },
      { label: 'Evidence stories opened', value: String(funnel.evidence_story_opened || 0) },
      { label: 'Map preview opens', value: String(funnel.room_map_preview_opened || 0) },
      { label: 'Room Compare opens', value: String(funnel.room_compare_opened || 0) },
      { label: 'Room Passport opens', value: String(funnel.room_passport_opened || 0) },
      { label: 'Room personalities viewed', value: String(funnel.room_personality_viewed || 0) },
      { label: 'Room Playbook starts', value: String(funnel.room_playbook_started || 0) },
      { label: 'Fix verification starts', value: String(funnel.fix_verification_started || 0) },
      { label: 'Fix Quest completions', value: String(funnel.fix_quest_completed || 0) },
      { label: 'Fix Library opens', value: String(funnel.fix_library_opened || 0) },
      { label: 'Fix Guide opens', value: String(funnel.fix_guide_opened || 0) },
      { label: 'One Thing starts', value: String(funnel.one_thing_today_started || 0) },
      { label: 'One Thing completions', value: String(funnel.one_thing_today_completed || 0) },
      { label: 'Care calendar opens', value: String(funnel.room_care_calendar_opened || 0) },
      { label: 'Care task completions', value: String(funnel.room_care_task_completed || 0) },
      { label: 'Care week regenerations', value: String(funnel.room_care_week_regenerated || 0) },
      { label: 'Room timeline opens', value: String(funnel.room_health_timeline_opened || 0) },
      { label: 'Daily Value settings', value: String(funnel.settings_daily_value_changed || 0) },
      { label: 'Share studio copies', value: String(funnel.share_card_studio_copied || 0) },
      { label: 'Confidence explainers', value: String(funnel.confidence_explainer_opened || 0) },
      { label: 'Welcome tours done', value: String(funnel.welcome_tour_completed || 0) },
      { label: 'Home Pulse opens', value: String(funnel.home_pulse_opened || 0) },
      { label: 'Weekly recap copies', value: String(funnel.weekly_recap_copied || 0) },
      { label: 'Weekly challenges done', value: String(funnel.weekly_challenge_completed || 0) },
      { label: 'Home Bingo completions', value: String(funnel.home_bingo_task_completed || 0) },
      { label: 'Room ritual starts', value: String(funnel.room_ritual_started || 0) },
      { label: 'Room ritual done', value: String(funnel.room_ritual_completed || 0) },
      { label: 'Seasonal packs selected', value: String(funnel.seasonal_pack_selected || 0) },
      { label: 'Smart Rescan Coach opens', value: String(funnel.smart_rescan_coach_opened || 0) },
      { label: 'Home Journal opens', value: String(funnel.home_journal_opened || 0) },
      { label: 'Fix Today copies', value: String(funnel.fix_today_copied || 0) },
      { label: 'PDF downloads', value: String(funnel.pdf_downloads || 0) },
      { label: 'Share card copies', value: String(funnel.share_card_copies || 0) },
      { label: 'Room win cards shared', value: String(funnel.room_win_card_shared || 0) },
      { label: 'Post-report feedback', value: String(funnel.post_report_feedback_submitted || 0) },
      { label: 'Confidence opens', value: String(funnel.confidence_inspector_opened || 0) },
      { label: 'Preflight failures', value: String(funnel.scan_preflight_failed || 0) },
      { label: 'Rescan prompt clicks', value: String(funnel.rescan_prompt_clicked || 0) },
      { label: 'Completion rate', value: `${Math.round((funnel.completion_rate || 0) * 100)}%` },
      { label: 'Eval-ready reports', value: String(readiness.review_ready_reports || 0) },
      { label: 'Eval candidates', value: `${readiness.saved_eval_candidates || 0} / ${readiness.target_corpus_size || 0}` },
    ])}
    ${waitlist.length ? `<p class="subsection-label">Recent waitlist</p>${renderPolicyItems(waitlist.map((entry) => ({
      label: entry.email || 'waitlist',
      value: `${entry.audience_mode || 'general'} · ${entry.use_case || entry.source || 'No use case'}`,
    })))}` : '<p class="meta-copy">No waitlist submissions yet.</p>'}
    ${failures.length ? `<p class="subsection-label">Failed uploads</p>${renderPolicyItems(failures.map((job) => ({
      label: job.filename || job.job_id || 'failed job',
      value: `${job.stage || 'stage unknown'} · ${job.error || 'Unknown failure'}`,
    })))}` : '<p class="meta-copy">No failed uploads in the beta inbox.</p>'}
    ${negative.length ? `<p class="subsection-label">Negative feedback</p>${renderPolicyItems(negative.map((job) => ({
      label: job.filename || job.job_id || 'report',
      value: `${job.wrong || 0} wrong · ${job.duplicate || 0} duplicate`,
    })))}` : '<p class="meta-copy">No wrong/duplicate report feedback yet.</p>'}
    ${reviewNeeded.length ? `<p class="subsection-label">Review queue</p>${renderPolicyItems(reviewNeeded.map((job) => ({
      label: job.filename || job.job_id || 'report',
      value: job.review_ready_for_eval ? 'Ready for eval export' : (job.summary || 'Needs human review'),
    })))}` : '<p class="meta-copy">No reports currently need review.</p>'}
    ${missed.length ? `<p class="subsection-label">Missed hazard notes</p>${renderPolicyItems(missed.map((item) => ({
      label: item.room_label || item.job_id || 'missed hazard',
      value: item.note || 'No note',
    })))}` : ''}
  `;
}

async function pollHealth() {
  try {
    const health = await api.fetchHealth();
    if (Array.isArray(health.warnings) && health.warnings.length) {
      healthStatus.textContent = `Needs attention · ${health.warnings[0]}`;
    } else {
      healthStatus.textContent = health.slam_active ? 'Live scene connected' : 'Upload-first mode';
    }
  } catch {
    healthStatus.textContent = 'API unavailable';
  }
}

async function bootstrapJobs() {
  if (!state.operatorSettings?.access?.enable_job_listing) {
    renderUploads();
    renderProcessing(activeJob());
    renderReport(activeJob());
    renderHomeJournal();
    return;
  }

  try {
    const jobs = await api.fetchJobs();
    state.jobs.clear();
    jobs.forEach((job) => state.jobs.set(job.job_id, job));
    const latest = jobs.at(-1) || jobs[jobs.length - 1];
    if (latest) {
      setActiveJob(latest.job_id);
      renderProcessing(latest);
      if (latest.status === 'complete') {
        jobs.filter((job) => job.status === 'complete').forEach(upsertHomeJournalFromJob);
        renderHomeJournal();
        renderReport(latest);
      }
    } else {
      renderUploads();
      renderProcessing(null);
      renderReport(null);
      renderHomeJournal();
    }
  } catch {
    renderUploads();
    renderProcessing(null);
    renderReport(null);
    renderHomeJournal();
  }
}

async function refreshOperatorState(errorMessage = '') {
  try {
    state.operatorSettings = await api.fetchOperatorSettings();
  } catch {
    state.operatorSettings = null;
  }
  renderAccessPanels(errorMessage);
}

async function bootstrapApp() {
  try {
    state.accessPolicy = await api.fetchAccessPolicy();
  } catch {
    state.accessPolicy = null;
  }
  try {
    state.privacyPolicy = await api.fetchPrivacyPolicy();
  } catch {
    state.privacyPolicy = null;
  }
  try {
    state.uploadGuidance = await api.fetchUploadGuidance();
    uploadView.setGuidance(state.uploadGuidance);
    applyUploadGuidance(state.uploadGuidance);
  } catch {
    state.uploadGuidance = null;
    applyUploadGuidance(null);
  }
  try {
    state.trustProof = await api.fetchTrustProof();
  } catch {
    state.trustProof = null;
  }
  renderTrustProofDashboard();
  await refreshOperatorState();
  await bootstrapJobs();
  const sampleKey = requestedSampleKey();
  if (sampleKey) {
    try {
      const sample = await api.fetchSampleReport();
      upsertJob(sample);
      setActiveJob(sample.job_id);
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not open the sample report.', 3600);
    }
  }
  const jobId = requestedJobId();
  if (jobId) {
    try {
      upsertJob(await api.fetchJob(jobId));
      setActiveJob(jobId);
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not open the shared report link.', 3600);
    }
  }
  switchView(requestedView() || (activeJob()?.status === 'complete' ? 'report' : 'scan'));
}

navButtons.forEach((button) => {
  button.addEventListener('click', () => switchView(button.dataset.view || 'scan'));
});

jumpButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const destination = button.dataset.jumpView || 'scan';
    if (destination === 'scan') {
      markFirstRunStarted('hero_scan_cta');
      trackProductEvent('cta_start_scan', { surface: 'hero' });
      trackProductEvent('beta_onboarding_started', {
        surface: 'hero_scan_cta',
        use_case: roomLabelInput?.value?.trim() || null,
      });
    }
    switchView(destination);
  });
});

sampleButtons.forEach((button) => {
  button.addEventListener('click', () => {
    trackProductEvent('sample_cta_clicked', {
      surface: button.closest('.hero') ? 'hero' : button.closest('.settings-details') ? 'settings' : 'scan_onboarding',
      sample_key: button.dataset.loadSample || 'walkthrough',
    });
    loadSampleReport();
  });
});

useCaseButtons.forEach((button) => {
  button.addEventListener('click', () => {
    applyUseCase(button.dataset.useCaseMode || 'general', button.dataset.useCaseLabel || '');
  });
});

dailyMissionStart?.addEventListener('click', startDailyMission);
dailyMissionComplete?.addEventListener('click', () => completeDailyMission(activeChallenge()));
ritualStartBtn?.addEventListener('click', () => startRoomRitual(activeRitual(), 'ritual_today_card'));
ritualCompleteBtn?.addEventListener('click', () => completeRoomRitual(activeRitual()));
ritualGrid?.addEventListener('click', (event) => {
  const button = event.target instanceof Element ? event.target.closest('[data-room-ritual]') : null;
  const ritual = ritualById(button?.dataset.roomRitual);
  if (ritual) {
    startRoomRitual(ritual, 'ritual_grid');
  }
});
welcomeTourCompleteBtn?.addEventListener('click', completeWelcomeTour);
roomPlaybookGrid?.addEventListener('click', (event) => {
  const button = event.target instanceof Element ? event.target.closest('[data-room-playbook]') : null;
  const playbook = roomPlaybookById(button?.dataset.roomPlaybook);
  if (playbook) {
    markBingoTask('try-playbook');
    startRoomPlaybook(playbook, 'room_playbook_grid');
    renderHomeCompanion();
  }
});
mysteryModeGrid?.addEventListener('click', (event) => {
  const button = event.target instanceof Element ? event.target.closest('[data-mystery-mode]') : null;
  const mode = mysteryModeById(button?.dataset.mysteryMode);
  if (mode) {
    startMysteryMode(mode, 'mystery_mode_grid');
  }
});
homeCompanionPanel?.addEventListener('click', async (event) => {
  const target = event.target instanceof Element ? event.target : null;
  const startOneThingButton = target?.closest('[data-start-one-thing]');
  if (startOneThingButton) {
    const action = actionById(startOneThingButton.dataset.startOneThing);
    if (roomLabelInput) {
      roomLabelInput.value = suggestedRoomLabel(action.roomLabel);
    }
    if (audienceModeInput) {
      audienceModeInput.value = action.audienceMode;
    }
    renderCaptureCoach();
    trackProductEvent('one_thing_today_started', {
      action_id: action.id,
      room_label: roomLabelInput?.value || action.roomLabel,
      room_labeled: true,
      audience_mode: action.audienceMode,
    });
    switchView('scan');
    showToast(`${action.title} loaded as today's tiny home-care prompt.`);
    return;
  }

  const completeOneThingButton = target?.closest('[data-complete-one-thing]');
  if (completeOneThingButton) {
    const action = actionById(completeOneThingButton.dataset.completeOneThing);
    writeJsonObject(DAILY_ACTION_STORAGE_KEY, {
      date: localDateKey(),
      actionId: action.id,
      completed: true,
      completedAt: new Date().toISOString(),
    });
    renderHomeCompanion();
    renderSettingsControlCenter();
    trackProductEvent('one_thing_today_completed', { action_id: action.id });
    showToast('One Thing Today marked done locally.');
    return;
  }

  const careStartButton = target?.closest('[data-start-room-care-task]');
  if (careStartButton) {
    const task = buildRoomCareWeek().tasks.find((item) => item.id === careStartButton.dataset.startRoomCareTask);
    if (!task) {
      return;
    }
    if (roomLabelInput) {
      roomLabelInput.value = task.roomLabel;
    }
    if (audienceModeInput) {
      audienceModeInput.value = task.audienceMode;
    }
    renderCaptureCoach();
    trackProductEvent('room_care_calendar_opened', {
      surface: 'room_care_task',
      task_id: task.id,
      action_id: task.actionId,
      room_label: task.roomLabel,
      room_labeled: true,
      audience_mode: task.audienceMode,
    });
    switchView('scan');
    showToast(`${task.title} loaded. Use it as a local care prompt, not a certification checklist.`);
    return;
  }

  const careDoneButton = target?.closest('[data-complete-room-care-task]');
  if (careDoneButton) {
    const taskId = careDoneButton.dataset.completeRoomCareTask || '';
    const completed = readRoomCareCompleted();
    completed[taskId] = new Date().toISOString();
    writeJsonObject(ROOM_CARE_COMPLETED_STORAGE_KEY, completed);
    renderHomeCompanion();
    renderSettingsControlCenter();
    trackProductEvent('room_care_task_completed', { task_id: taskId });
    showToast('Room care task completed locally.');
    return;
  }

  const weeklyButton = target?.closest('[data-copy-weekly-recap]');
  if (weeklyButton) {
    await copyText(buildWeeklyRecapText());
    markBingoTask('copy-room-win');
    renderHomeCompanion();
    trackProductEvent('weekly_recap_copied', { room_count: weeklyRecapData().roomsChecked });
    showToast('Weekly home win copied.');
    return;
  }

  const bingoButton = target?.closest('[data-home-bingo-task]');
  if (bingoButton) {
    const taskId = bingoButton.dataset.homeBingoTask || '';
    markBingoTask(taskId, true);
    renderHomeCompanion();
    renderSettingsControlCenter();
    showToast('Calm Home Bingo updated locally.');
    return;
  }

  const personalButton = target?.closest('[data-personal-mode]');
  const mode = personalModeById(personalButton?.dataset.personalMode);
  if (mode) {
    startPersonalMode(mode);
  }
});
roomCareRegenerateBtn?.addEventListener('click', () => {
  regenerateRoomCareWeek('home_companion');
});
curiositySampleGrid?.addEventListener('click', (event) => {
  const button = event.target instanceof Element ? event.target.closest('[data-curiosity-sample]') : null;
  const sample = CURIOSITY_SAMPLE_GALLERY.find((item) => item.id === button?.dataset.curiositySample);
  if (!sample) {
    return;
  }
  if (audienceModeInput) {
    audienceModeInput.value = sample.audienceMode;
  }
  if (roomLabelInput) {
    roomLabelInput.value = sample.roomLabel;
  }
  renderCaptureCoach();
  trackProductEvent('sample_gallery_opened', {
    sample_gallery_id: sample.id,
    audience_mode: sample.audienceMode,
    room_label: sample.roomLabel,
    room_labeled: true,
    use_case: sample.title,
  });
  trackProductEvent('sample_cta_clicked', {
    surface: 'curiosity_sample_gallery',
    sample_key: sample.id,
  });
  loadSampleReport(sample.id, 'curiosity_sample_gallery');
});
seasonalPackGrid?.addEventListener('click', (event) => {
  const button = event.target instanceof Element ? event.target.closest('[data-seasonal-pack]') : null;
  const pack = SEASONAL_RITUAL_PACKS.find((item) => item.id === button?.dataset.seasonalPack);
  if (!pack) {
    return;
  }
  const ritual = ritualById(pack.ritualId) || activeRitual();
  if (roomLabelInput) {
    roomLabelInput.value = pack.roomLabel || ritual.roomLabel;
  }
  if (audienceModeInput) {
    audienceModeInput.value = pack.audienceMode || ritual.audienceMode;
  }
  markBingoTask('seasonal-pack');
  trackProductEvent('seasonal_pack_selected', {
    seasonal_pack_id: pack.id,
    audience_mode: pack.audienceMode || ritual.audienceMode,
    room_label: pack.roomLabel || ritual.roomLabel,
    room_labeled: true,
  });
  trackProductEvent('seasonal_pack_started', {
    seasonal_pack_id: pack.id,
    ritual_id: ritual.id,
    audience_mode: ritual.audienceMode,
  });
  startRoomRitual(ritual, 'seasonal_pack');
  renderHomeCompanion();
  showToast(`${pack.title} loaded as today’s room-care lens.`);
});

audienceModeInput?.addEventListener('change', () => {
  renderCaptureCoach();
  trackProductEvent('capture_mode_changed', { audience_mode: selectedAudienceMode() });
});

captureCoachChecks?.addEventListener('change', (event) => {
  const checkbox = event.target instanceof HTMLInputElement
    ? event.target.closest('[data-capture-check]')
    : null;
  if (!checkbox || !(checkbox instanceof HTMLInputElement)) {
    return;
  }
  updateCaptureCoachCheck(checkbox.dataset.captureCheck || '', checkbox.checked);
});

liveCaptureStartBtn?.addEventListener('click', () => {
  startLiveCaptureCoach();
});

liveCaptureStopBtn?.addEventListener('click', () => {
  stopLiveCaptureCoach();
});

document.getElementById('scene-refresh-btn')?.addEventListener('click', () => {
  ensureScene();
  sceneViewer.refresh();
});

const uploadView = new UploadView({
  dropZone: document.getElementById('drop-zone'),
  fileInput: /** @type {HTMLInputElement} */ (document.getElementById('file-input')),
  roomLabelInput,
  audienceModeInput,
  onUploadStart: (file, metadata) => {
    markFirstRunStarted('upload_dropzone');
    state.pendingUploadChallengeId = activeChallenge().id;
    if (scanWizardStatus) {
      scanWizardStatus.textContent = `Uploading ${file.name}. Keep this tab open while ATLAS-0 builds the report.`;
    }
    trackProductEvent('upload_started', {
      surface: 'guided_scan_wizard',
      audience_mode: metadata.audienceMode,
      mission_id: state.pendingUploadChallengeId,
      challenge_id: state.pendingUploadChallengeId,
      ritual_id: activeRitual().id,
      playbook_id: activeRoomPlaybook()?.id || null,
      personal_mode_id: state.activePersonalModeId || null,
      mystery_mode_id: activeMysteryMode().id,
      room_label: metadata.roomLabel || null,
      room_labeled: Boolean(metadata.roomLabel),
    });
  },
  onOfflineQueued: (entry) => {
    markFirstRunStarted('offline_upload_queue');
    state.pendingUploadChallengeId = activeChallenge().id;
    if (scanWizardStatus) {
      scanWizardStatus.textContent = `${entry.filename} is saved locally and will retry when your connection returns.`;
    }
    trackProductEvent('offline_upload_queued', {
      surface: 'guided_scan_wizard',
      file_type: entry.fileType || null,
      file_size: entry.fileSize || null,
      audience_mode: entry.audienceMode || selectedAudienceMode(),
      room_label: entry.roomLabel || null,
      room_labeled: Boolean(entry.roomLabel),
    });
    showToast('Upload queued locally for retry when you reconnect.', 3800);
  },
  onOfflineReplayStart: (entry) => {
    if (scanWizardStatus) {
      scanWizardStatus.textContent = `Connection restored. Retrying ${entry.filename || 'queued upload'} now.`;
    }
    trackProductEvent('offline_upload_retried', {
      surface: 'guided_scan_wizard',
      file_type: entry.fileType || null,
      file_size: entry.fileSize || null,
      audience_mode: entry.audienceMode || selectedAudienceMode(),
      room_label: entry.roomLabel || null,
      room_labeled: Boolean(entry.roomLabel),
    });
  },
  onOfflineQueueChange: (entries) => {
    if (offlineBanner) {
      offlineBanner.dataset.queueCount = String(entries.length);
    }
    updateOfflineBanner();
  },
  onJobCreated: async (job) => {
    assignChallengeToJob(job.job_id, state.pendingUploadChallengeId || activeChallenge().id);
    upsertJob(job);
    await refreshOperatorState();
    switchView('scan');
  },
  onJobUpdate: (job) => {
    upsertJob(job);
    if (job.status === 'complete' && !state.uploadCompleteEvents.has(job.job_id)) {
      state.uploadCompleteEvents.add(job.job_id);
      trackProductEvent('upload_completed', {
        surface: 'upload_worker',
        job_id: job.job_id,
        mission_id: challengeForJob(job).id,
        challenge_id: challengeForJob(job).id,
        ritual_id: activeRitual().id,
        playbook_id: activeRoomPlaybook()?.id || null,
        personal_mode_id: state.activePersonalModeId || null,
        mystery_mode_id: activeMysteryMode().id,
        audience_mode: job.audience_mode || selectedAudienceMode(),
        room_label: job.room_label || job.summary?.room_label || null,
        room_labeled: Boolean(job.room_label || job.summary?.room_label),
      });
    }
  },
  onJobError: (error) => {
    if (scanWizardStatus) {
      scanWizardStatus.textContent = error.message;
    }
    showToast(error.message, 3600);
  },
  onPreflightFailed: (file, error) => {
    trackProductEvent('scan_preflight_failed', {
      surface: 'guided_scan_wizard',
      file_type: file?.type || null,
      file_size: file?.size || null,
      reason: error.message,
    });
  },
});

uploadView.init();
applyThemePreference(readStoredPreference(THEME_STORAGE_KEY) || document.documentElement.dataset.theme || 'light');
applyMotionPreference(readStoredPreference(MOTION_STORAGE_KEY) === 'true');
if (waitlistReferralInput) {
  waitlistReferralInput.value = betaReferralCode();
}
syncSettingsPreferenceControls();
applyAccessibilityPreferences();
applyDefaultScanPreferences(false);
syncLowConfidenceControls();
syncSettingsAccessStatus();
renderDailyMission();
renderChallengeLibrary();
renderRoomRituals();
renderWelcomeTour();
renderRoomPlaybooks();
renderMysteryModes();
renderCuriositySampleGallery();
renderHomeCompanion();
renderCaptureCoach();
renderHomeJournal();
renderHomePulse();
initLandingSectionTracking();
updateOfflineBanner();
window.addEventListener('online', updateOfflineBanner);
window.addEventListener('offline', updateOfflineBanner);
registerServiceWorker();
if (betaShareCopy) {
  betaShareCopy.textContent = betaSharePrompt();
}
bootstrapApp();
pollHealth();
setInterval(pollHealth, 6000);

lowConfidenceToggle?.addEventListener('change', (event) => {
  setLowConfidenceVisibility(/** @type {HTMLInputElement} */ (event.currentTarget).checked);
});

settingsLowConfidenceToggle?.addEventListener('change', (event) => {
  setLowConfidenceVisibility(/** @type {HTMLInputElement} */ (event.currentTarget).checked);
  trackProductEvent('settings_report_preferences_changed', {
    preference: 'show_low_confidence',
    enabled: state.showLowConfidence,
  });
  renderSettingsControlCenter();
});

themeToggle?.addEventListener('change', (event) => {
  const enabled = /** @type {HTMLInputElement} */ (event.currentTarget).checked;
  applyThemePreference(enabled ? 'dark' : 'light');
  trackProductEvent('settings_theme_changed', { theme: enabled ? 'dark' : 'light' });
});

motionToggle?.addEventListener('change', (event) => {
  const enabled = /** @type {HTMLInputElement} */ (event.currentTarget).checked;
  applyMotionPreference(enabled);
  trackProductEvent('settings_motion_changed', { reduced_motion: enabled });
  trackProductEvent('settings_accessibility_changed', { preference: 'reduced_motion', enabled });
  renderSettingsControlCenter();
});

settingsReportStyleInput?.addEventListener('change', (event) => {
  const value = /** @type {HTMLSelectElement} */ (event.currentTarget).value;
  writeStoredPreference(REPORT_STYLE_STORAGE_KEY, value);
  trackProductEvent('settings_report_preferences_changed', { preference: 'report_style', value });
  renderSettingsControlCenter();
  showToast(`${reportStyleLabel(value)} report style saved locally.`);
});

function updateReportTheme(value, surface = 'report') {
  writeStoredPreference(REPORT_THEME_STORAGE_KEY, value);
  syncSettingsPreferenceControls();
  renderReportThemePanel(activeJob());
  renderSettingsControlCenter();
  trackProductEvent('report_theme_changed', {
    surface,
    report_theme: value,
    job_id: activeJob()?.job_id || null,
    sample_key: activeJob()?.sample_key || null,
  });
  trackProductEvent('settings_report_preferences_changed', { preference: 'report_theme', value });
  showToast(`${reportThemeLabel(value)} theme saved locally.`);
}

settingsReportThemeInput?.addEventListener('change', (event) => {
  updateReportTheme(/** @type {HTMLSelectElement} */ (event.currentTarget).value, 'settings');
});

reportThemeInput?.addEventListener('change', (event) => {
  updateReportTheme(/** @type {HTMLSelectElement} */ (event.currentTarget).value, 'report_theme_panel');
});

settingsRescanReminderInput?.addEventListener('change', (event) => {
  const value = /** @type {HTMLSelectElement} */ (event.currentTarget).value;
  writeStoredPreference(RESCAN_REMINDER_STORAGE_KEY, value);
  trackProductEvent('settings_report_preferences_changed', { preference: 'rescan_reminder', value });
  renderSettingsControlCenter();
});

settingsCareCadenceInput?.addEventListener('change', (event) => {
  const value = /** @type {HTMLSelectElement} */ (event.currentTarget).value;
  writeStoredPreference(CARE_CADENCE_STORAGE_KEY, value);
  buildRoomCareWeek(true);
  renderHomeCompanion();
  renderRoomHealthTimeline();
  renderSettingsControlCenter();
  trackProductEvent('settings_daily_value_changed', { preference: 'care_cadence', value });
  showToast('Daily Value cadence saved locally.');
});

settingsDefaultAudienceInput?.addEventListener('change', (event) => {
  const value = /** @type {HTMLSelectElement} */ (event.currentTarget).value;
  writeStoredPreference(DEFAULT_AUDIENCE_STORAGE_KEY, value);
  applyDefaultScanPreferences(false);
  trackProductEvent('settings_default_scan_changed', { preference: 'audience_mode', value });
});

settingsDefaultRoomLabelInput?.addEventListener('change', (event) => {
  const value = /** @type {HTMLInputElement} */ (event.currentTarget).value.trim().slice(0, 80);
  if (value) {
    writeStoredPreference(DEFAULT_ROOM_LABEL_STORAGE_KEY, value);
  } else {
    removeStoredPreference(DEFAULT_ROOM_LABEL_STORAGE_KEY);
  }
  applyDefaultScanPreferences(false);
  trackProductEvent('settings_default_scan_changed', { preference: 'room_label', room_labeled: Boolean(value) });
});

settingsDefaultMysteryInput?.addEventListener('change', (event) => {
  const value = /** @type {HTMLSelectElement} */ (event.currentTarget).value;
  if (value) {
    writeStoredPreference(DEFAULT_MYSTERY_MODE_STORAGE_KEY, value);
  } else {
    removeStoredPreference(DEFAULT_MYSTERY_MODE_STORAGE_KEY);
  }
  applyDefaultScanPreferences(false);
  trackProductEvent('settings_default_scan_changed', { preference: 'mystery_mode', mystery_mode_id: value || null });
});

shareCardStyleInput?.addEventListener('change', (event) => {
  const value = /** @type {HTMLSelectElement} */ (event.currentTarget).value;
  writeStoredPreference(SHARE_CARD_STYLE_STORAGE_KEY, value);
  renderShareCardPreview(activeJob());
  trackProductEvent('share_card_style_changed', {
    share_style: value,
    job_id: activeJob()?.job_id || null,
    sample_key: activeJob()?.sample_key || null,
  });
  showToast(`${shareStyleLabel(value)} saved for this browser.`);
});

settingsLargeTextToggle?.addEventListener('change', (event) => {
  const enabled = /** @type {HTMLInputElement} */ (event.currentTarget).checked;
  writeStoredPreference(LARGE_TEXT_STORAGE_KEY, enabled ? 'true' : 'false');
  applyAccessibilityPreferences();
  trackProductEvent('settings_accessibility_changed', { preference: 'large_text', enabled });
  renderSettingsControlCenter();
});

settingsHighContrastToggle?.addEventListener('change', (event) => {
  const enabled = /** @type {HTMLInputElement} */ (event.currentTarget).checked;
  writeStoredPreference(HIGH_CONTRAST_STORAGE_KEY, enabled ? 'true' : 'false');
  applyAccessibilityPreferences();
  trackProductEvent('settings_accessibility_changed', { preference: 'high_contrast', enabled });
  renderSettingsControlCenter();
});

settingsFocusToggle?.addEventListener('change', (event) => {
  const enabled = /** @type {HTMLInputElement} */ (event.currentTarget).checked;
  writeStoredPreference(FOCUS_MODE_STORAGE_KEY, enabled ? 'true' : 'false');
  applyAccessibilityPreferences();
  trackProductEvent('settings_accessibility_changed', { preference: 'always_focus', enabled });
  renderSettingsControlCenter();
});

settingsLayoutDensityInput?.addEventListener('change', (event) => {
  const value = /** @type {HTMLSelectElement} */ (event.currentTarget).value;
  writeStoredPreference(LAYOUT_DENSITY_STORAGE_KEY, value);
  applyAccessibilityPreferences();
  trackProductEvent('settings_accessibility_changed', { preference: 'layout_density', value });
  renderSettingsControlCenter();
});

briefConfidenceDetails?.addEventListener('toggle', () => {
  if (!briefConfidenceDetails.open) {
    return;
  }
  const job = activeJob();
  trackProductEvent('confidence_inspector_opened', {
    surface: 'brief_confidence_strip',
    job_id: job?.job_id || null,
    sample_key: job?.sample_key || null,
    audience_mode: job?.audience_mode || selectedAudienceMode(),
    room_labeled: Boolean(job?.room_label || job?.summary?.room_label || roomLabelInput?.value?.trim()),
  });
});

settingsSampleBtn?.addEventListener('click', () => {
  trackProductEvent('sample_cta_clicked', {
    surface: 'settings',
    sample_key: 'walkthrough',
  });
  trackProductEvent('settings_sample_opened');
  loadSampleReport();
});

settingsClearJournalBtn?.addEventListener('click', () => {
  if (!window.confirm('Clear the local Home Journal and favorite rooms from this browser?')) {
    return;
  }
  clearLocalKeys([HOME_JOURNAL_STORAGE_KEY, FAVORITE_ROOMS_STORAGE_KEY]);
  renderHomeJournal();
  renderHomePulse();
  renderSettingsControlCenter();
  trackProductEvent('settings_data_cleared', { scope: 'home_journal' });
  showToast('Local Home Journal cleared.');
});

settingsClearRitualsBtn?.addEventListener('click', () => {
  if (!window.confirm('Clear local missions, rituals, streaks, challenge progress, and scan checklist progress?')) {
    return;
  }
  clearLocalKeys([
    MISSION_STORAGE_KEY,
    CHALLENGE_SELECTION_STORAGE_KEY,
    CHALLENGE_JOB_STORAGE_KEY,
    FIX_CHECKLIST_STORAGE_KEY,
    VERIFICATION_STORAGE_KEY,
    ACTIVE_EVIDENCE_STORAGE_KEY,
    RITUAL_STORAGE_KEY,
    RITUAL_SELECTION_STORAGE_KEY,
  ]);
  resetLocalRuntimeState();
  trackProductEvent('settings_data_cleared', { scope: 'rituals_and_streaks' });
  showToast('Local rituals and streaks cleared.');
});

settingsClearCompanionBtn?.addEventListener('click', () => {
  if (!window.confirm('Clear local Home Companion progress, including bingo, fix quests, personal mode, and report theme?')) {
    return;
  }
  clearLocalKeys([
    HOME_BINGO_STORAGE_KEY,
    PERSONAL_MODE_STORAGE_KEY,
    REPORT_THEME_STORAGE_KEY,
  ]);
  clearLocalPrefixes([FIX_QUEST_STORAGE_KEY]);
  state.activePersonalModeId = null;
  syncSettingsPreferenceControls();
  renderHomeCompanion();
  renderReport(activeJob());
  renderSettingsControlCenter();
  trackProductEvent('settings_data_cleared', { scope: 'home_companion' });
  showToast('Home Companion progress cleared locally.');
});

settingsClearDailyValueBtn?.addEventListener('click', () => {
  if (!window.confirm('Clear local One Thing Today, Room Care Calendar, completed care tasks, cadence, and active fix guide?')) {
    return;
  }
  clearLocalKeys([
    DAILY_ACTION_STORAGE_KEY,
    ROOM_CARE_CALENDAR_STORAGE_KEY,
    ROOM_CARE_COMPLETED_STORAGE_KEY,
    CARE_CADENCE_STORAGE_KEY,
    FIX_GUIDE_STORAGE_KEY,
  ]);
  syncSettingsPreferenceControls();
  renderHomeCompanion();
  renderRoomHealthTimeline();
  renderSettingsControlCenter();
  trackProductEvent('settings_data_cleared', { scope: 'daily_value' });
  showToast('Daily Value progress cleared locally.');
});

settingsClearDefaultsBtn?.addEventListener('click', () => {
  if (!window.confirm('Clear saved report and scan defaults from this browser?')) {
    return;
  }
  clearLocalKeys([
    LOW_CONFIDENCE_STORAGE_KEY,
    REPORT_STYLE_STORAGE_KEY,
    DEFAULT_AUDIENCE_STORAGE_KEY,
    DEFAULT_ROOM_LABEL_STORAGE_KEY,
    DEFAULT_MYSTERY_MODE_STORAGE_KEY,
    RESCAN_REMINDER_STORAGE_KEY,
    MYSTERY_MODE_STORAGE_KEY,
    ROOM_PLAYBOOK_STORAGE_KEY,
    SHARE_CARD_STYLE_STORAGE_KEY,
    PERSONAL_MODE_STORAGE_KEY,
    REPORT_THEME_STORAGE_KEY,
    CARE_CADENCE_STORAGE_KEY,
    FIX_GUIDE_STORAGE_KEY,
  ]);
  state.showLowConfidence = false;
  state.activeMysteryModeId = null;
  state.activeRoomPlaybookId = null;
  state.activePersonalModeId = null;
  syncLowConfidenceControls();
  syncSettingsPreferenceControls();
  applyDefaultScanPreferences(false);
  renderRoomPlaybooks();
  renderHomeCompanion();
  renderRoomHealthTimeline();
  renderReport(activeJob());
  trackProductEvent('settings_data_cleared', { scope: 'saved_defaults' });
  showToast('Report and scan defaults cleared.');
});

settingsClearAllLocalBtn?.addEventListener('click', async () => {
  if (!window.confirm('Clear all local ATLAS-0 browser data, including settings, journal, streaks, session, and token?')) {
    return;
  }
  await trackProductEvent('settings_data_cleared', { scope: 'all_local_data' });
  clearLocalKeys([...SETTINGS_LOCAL_KEYS, SESSION_STORAGE_KEY]);
  clearLocalPrefixes([FIX_CHECKLIST_STORAGE_KEY, FIX_QUEST_STORAGE_KEY, VERIFICATION_STORAGE_KEY]);
  api.clearAccessToken();
  applyThemePreference('light');
  removeStoredPreference(THEME_STORAGE_KEY);
  applyMotionPreference(false);
  removeStoredPreference(MOTION_STORAGE_KEY);
  syncSettingsPreferenceControls();
  applyAccessibilityPreferences();
  resetLocalRuntimeState();
  await refreshOperatorState();
  showToast('All local ATLAS-0 browser data cleared.');
});

settingsOpenCurrentReportBtn?.addEventListener('click', () => {
  if (!activeJob()) {
    showToast('No current report is available yet.');
    return;
  }
  switchView('report');
});

settingsRegenerateCareWeekBtn?.addEventListener('click', () => {
  regenerateRoomCareWeek('settings');
});

settingsDeleteCurrentReportBtn?.addEventListener('click', async () => {
  const job = activeJob();
  if (!job || job.is_sample) {
    showToast('Only non-sample reports can be deleted here.');
    return;
  }
  if (!window.confirm('Delete this server-side report and its artifacts?')) {
    return;
  }
  try {
    await api.deleteJob(job.job_id);
    removeJob(job.job_id);
    await refreshOperatorState();
    trackProductEvent('settings_data_cleared', { scope: 'current_report', job_id: job.job_id });
    showToast('Current report deleted.');
    switchView('scan');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not delete current report.', 3600);
  }
});

settingsFeedbackCopyBtn?.addEventListener('click', async () => {
  const job = activeJob();
  const template = [
    'ATLAS-0 beta feedback',
    `Surface: settings`,
    `Report: ${job?.job_id || 'no active report'}`,
    `Audience mode: ${selectedAudienceMode()}`,
    '',
    'What I tried:',
    'What felt useful:',
    'What felt confusing or wrong:',
    'What I expected next:',
  ].join('\n');
  try {
    await copyText(template);
    await trackProductEvent('settings_feedback_clicked', { feedback_type: 'general_template' });
    showToast('Feedback template copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy feedback template.', 3600);
  }
});

settingsBadResultBtn?.addEventListener('click', async () => {
  const job = activeJob();
  try {
    await copyText([
      'ATLAS-0 bad result report',
      `Report: ${job?.job_id || 'no active report'}`,
      `Room: ${job?.room_label || roomLabelInput?.value || 'unknown'}`,
      '',
      'What ATLAS-0 got wrong:',
      'What evidence frame should be reviewed:',
      'What hazard was missed or overstated:',
    ].join('\n'));
    await trackProductEvent('settings_feedback_clicked', { feedback_type: 'bad_result', job_id: job?.job_id || null });
    showToast('Bad-result template copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy bad-result template.', 3600);
  }
});

settingsFeatureRequestBtn?.addEventListener('click', async () => {
  try {
    await copyText([
      'ATLAS-0 feature request',
      '',
      'I wish ATLAS-0 could:',
      'This would help because:',
      'My room/use case is:',
    ].join('\n'));
    await trackProductEvent('settings_feedback_clicked', { feedback_type: 'feature_request' });
    showToast('Feature request template copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy feature request.', 3600);
  }
});

settingsBetaInviteBtn?.addEventListener('click', async () => {
  try {
    await copyBetaInvite('settings_support');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy beta invite.', 3600);
  }
});

settingsWaitlistBtn?.addEventListener('click', () => {
  trackProductEvent('settings_feedback_clicked', { feedback_type: 'waitlist_shortcut' });
  switchView('scan');
  waitlistEmailInput?.focus();
  showToast('Beta waitlist is ready in the scan view.');
});

settingsReplayWelcomeBtn?.addEventListener('click', () => {
  removeStoredPreference(WELCOME_TOUR_STORAGE_KEY);
  renderWelcomeTour();
  switchView('scan');
  welcomeTourCard?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  trackProductEvent('settings_feedback_clicked', { feedback_type: 'replay_welcome_tour' });
  showToast('Welcome tour is back in the scan view.');
});

settingsReplayWeeklyBtn?.addEventListener('click', () => {
  renderHomeCompanion();
  switchView('scan');
  weeklyRecapCard?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  trackProductEvent('settings_feedback_clicked', { feedback_type: 'replay_weekly_recap' });
  showToast('Weekly Home Pulse recap is ready in the scan view.');
});

settingsExportBackupBtn?.addEventListener('click', () => {
  downloadTextFile(`atlas-0-local-backup-${localDateKey()}.json`, JSON.stringify(localBackupPayload(), null, 2));
  trackProductEvent('settings_local_backup_exported');
  showToast('Local backup downloaded.');
});

settingsImportBackupBtn?.addEventListener('click', () => {
  settingsImportFileInput?.click();
});

settingsImportFileInput?.addEventListener('change', async (event) => {
  const input = /** @type {HTMLInputElement} */ (event.currentTarget);
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  try {
    const text = await readFileAsText(file);
    importLocalBackup(JSON.parse(text));
    await trackProductEvent('settings_local_backup_imported');
    showToast('Local backup imported.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not import local backup.', 4200);
  } finally {
    input.value = '';
  }
});

settingsPrivacySummaryBtn?.addEventListener('click', () => {
  downloadTextFile(`atlas-0-privacy-summary-${localDateKey()}.txt`, privacySummaryText(), 'text/plain');
});

copyBetaInviteBtn?.addEventListener('click', async () => {
  try {
    await copyBetaInvite('scan_beta_card');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy beta invite.', 3600);
  }
});

accessTokenSave?.addEventListener('click', async () => {
  const token = accessTokenInput.value.trim();
  if (!token) {
    showToast('Enter a token before saving.', 3200);
    return;
  }

  api.setAccessToken(token);
  accessTokenInput.value = '';
  await refreshOperatorState('The stored token did not unlock operator access.');
  await bootstrapJobs();
  if (requestedJobId()) {
    try {
      upsertJob(await api.fetchJob(requestedJobId()));
    } catch {}
  }
  if (requestedSampleKey()) {
    try {
      upsertJob(await api.fetchSampleReport());
    } catch {}
  }
  syncSettingsAccessStatus();
  showToast(state.operatorSettings ? 'Access token saved.' : 'Token saved, but access is still blocked.', 3200);
});

async function clearStoredAccessToken() {
  api.clearAccessToken();
  accessTokenInput.value = '';
  state.operatorSettings = null;
  state.jobs.clear();
  state.activeJobId = null;
  state.activeSampleKey = null;
  renderAccessPanels();
  renderUploads();
  renderProcessing(null);
  renderReport(null);
  syncSettingsAccessStatus();
  switchView('scan');
  showToast('Stored access token cleared.');
}

accessTokenClear?.addEventListener('click', async () => {
  await clearStoredAccessToken();
});

settingsTokenClear?.addEventListener('click', async () => {
  await clearStoredAccessToken();
});

operatorPruneBtn?.addEventListener('click', async () => {
  try {
    const result = await api.pruneOperatorStorage();
    await refreshOperatorState();
    showToast(
      result.deleted_jobs
        ? `Pruned ${result.deleted_jobs} job(s) and reclaimed ${formatBytes(result.bytes_reclaimed || 0)}.`
        : 'Storage prune completed with no expired jobs removed.',
      3400,
    );
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not run storage prune.', 3600);
  }
});

waitlistSubmitBtn?.addEventListener('click', async () => {
  const email = waitlistEmailInput.value.trim();
  const useCase = waitlistUseCaseInput.value.trim();
  const referralCode = (waitlistReferralInput?.value || betaReferralCode()).trim().slice(0, 80);
  if (!email) {
    showToast('Enter an email to join the beta waitlist.', 3200);
    return;
  }

  try {
    const response = await api.submitWaitlist({
      email,
      use_case: useCase || null,
      source: 'hero_waitlist',
      audience_mode: selectedAudienceMode(),
      persona: betaPersona(),
      referral_code: referralCode || null,
    });
    waitlistEmailInput.value = '';
    waitlistUseCaseInput.value = '';
    if (waitlistReferralInput && referralCode) {
      writeStoredPreference(REFERRAL_STORAGE_KEY, referralCode);
    }
    if (waitlistNote) {
      waitlistNote.textContent = response.message;
    }
    await refreshOperatorState();
    showToast(`Waitlist joined. Position ${response.waitlist_count}.`, 3200);
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not join the waitlist.', 3600);
  }
});

deleteJobBtn?.addEventListener('click', async () => {
  const job = activeJob();
  if (!job) {
    return;
  }
  if (!window.confirm('Delete this report and its artifacts?')) {
    return;
  }

  try {
    await api.deleteJob(job.job_id);
    removeJob(job.job_id);
    await refreshOperatorState();
    showToast('Report deleted.');
    switchView('scan');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not delete report.', 3600);
  }
});

copyShareBtn?.addEventListener('click', async () => {
  const job = activeJob();
  if (!job || job.status !== 'complete') {
    return;
  }

  try {
    const link = reportDeepLink(job);
    await copyText(link);
    await trackProductEvent('report_share_copied', {
      surface: 'report_toolbar',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      audience_mode: job.audience_mode || 'general',
      room_labeled: Boolean(job.room_label || job.summary?.room_label),
    });
    showToast('Report link copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy report link.', 3600);
  }
});

exportPdfBtn?.addEventListener('click', () => {
  const job = activeJob();
  if (!job || job.status !== 'complete') {
    return;
  }
  trackProductEvent('pdf_export_clicked', {
    surface: 'report_toolbar',
    job_id: job.job_id,
    sample_key: job.sample_key || null,
    audience_mode: job.audience_mode || 'general',
    room_labeled: Boolean(job.room_label || job.summary?.room_label),
  });
  trackProductEvent('report_pdf_downloaded', {
    surface: 'report_toolbar',
    job_id: job.job_id,
    sample_key: job.sample_key || null,
    audience_mode: job.audience_mode || 'general',
    room_labeled: Boolean(job.room_label || job.summary?.room_label),
  });
});

copyShareCardBtn?.addEventListener('click', async () => {
  const job = activeJob();
  if (!job || job.status !== 'complete') {
    return;
  }
  try {
    await copyText(buildShareCardText(job));
    await trackProductEvent('report_share_card_copied', {
      surface: 'share_preview',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      audience_mode: job.audience_mode || 'general',
      share_style: currentShareCardStyle(),
      room_labeled: Boolean(job.room_label || job.summary?.room_label),
    });
    await trackProductEvent('room_win_card_shared', {
      surface: 'share_card_studio',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      share_style: currentShareCardStyle(),
      audience_mode: job.audience_mode || 'general',
    });
    await trackProductEvent('share_card_studio_copied', {
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      share_style: currentShareCardStyle(),
      audience_mode: job.audience_mode || 'general',
    });
    showToast('Share card copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy share card.', 3600);
  }
});

reportQuestionList?.addEventListener('click', (event) => {
  const button = event.target instanceof Element ? event.target.closest('[data-report-question]') : null;
  const job = activeJob();
  if (!button || !job || job.status !== 'complete') {
    return;
  }
  const questionId = button.dataset.reportQuestion || 'fix_first';
  state.activeReportQuestion = questionId;
  const history = readJsonObject(REPORT_QA_HISTORY_STORAGE_KEY);
  history[job.job_id] = [
    questionId,
    ...((history[job.job_id] || []).filter((item) => item !== questionId)),
  ].slice(0, 8);
  writeJsonObject(REPORT_QA_HISTORY_STORAGE_KEY, history);
  renderReport(job);
  trackProductEvent('report_question_asked', {
    surface: 'report_qa_panel',
    job_id: job.job_id,
    sample_key: job.sample_key || null,
    reason: questionId,
  });
});

copyReportAnswerBtn?.addEventListener('click', async () => {
  const job = activeJob();
  if (!job || !state.activeReportAnswer) {
    return;
  }
  try {
    await copyText(state.activeReportAnswer);
    await trackProductEvent('report_answer_copied', {
      surface: 'report_qa_panel',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      reason: state.activeReportQuestion || 'fix_first',
    });
    showToast('Report answer copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy report answer.', 3600);
  }
});

privacyEvidenceList?.addEventListener('change', (event) => {
  const checkbox = event.target instanceof HTMLInputElement ? event.target : null;
  const job = activeJob();
  if (!checkbox || !job?.job_id || (!checkbox.dataset.privacyEvidence && !checkbox.dataset.privacyBlur)) {
    return;
  }
  const current = privacyReceiptState(job.job_id);
  const excluded = new Set(current.excludedEvidence || []);
  const blurred = new Set(current.blurredEvidence || []);
  let reason = 'updated';

  if (checkbox.dataset.privacyEvidence) {
    const id = checkbox.dataset.privacyEvidence || '';
    if (checkbox.checked) {
      excluded.delete(id);
      reason = 'included';
    } else {
      excluded.add(id);
      reason = 'excluded';
    }
  } else {
    const id = checkbox.dataset.privacyBlur || '';
    if (checkbox.checked) {
      blurred.add(id);
      reason = 'blurred';
    } else {
      blurred.delete(id);
      reason = 'unblurred';
    }
  }
  writePrivacyReceiptState(job.job_id, {
    excludedEvidence: [...excluded],
    blurredEvidence: [...blurred],
  });
  renderReport(job);
  trackProductEvent('evidence_privacy_toggled', {
    surface: 'privacy_receipt',
    job_id: job.job_id,
    sample_key: job.sample_key || null,
    reason,
  });
});

copyPrivacyReceiptBtn?.addEventListener('click', async () => {
  const job = activeJob();
  if (!job || job.status !== 'complete') {
    return;
  }
  try {
    await copyText(privacyReceiptText(job));
    await trackProductEvent('privacy_receipt_copied', {
      surface: 'privacy_receipt',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
    });
    showToast('Privacy receipt copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy privacy receipt.', 3600);
  }
});

downloadPrivacyReceiptBtn?.addEventListener('click', () => {
  const job = activeJob();
  if (!job || job.status !== 'complete') {
    return;
  }
  trackProductEvent('privacy_receipt_opened', {
    surface: 'privacy_receipt_download',
    job_id: job.job_id,
    sample_key: job.sample_key || null,
  });
  downloadTextFile(`atlas-0-privacy-receipt-${job.job_id}.txt`, privacyReceiptText(job), 'text/plain');
});

function attachFixChecklistHandlers(jobId) {
  fixChecklistList.querySelectorAll('[data-checklist-item] input').forEach((input) => {
    input.addEventListener('change', (event) => {
      const checkbox = /** @type {HTMLInputElement} */ (event.currentTarget);
      const item = checkbox.closest('[data-checklist-item]');
      const itemId = item?.dataset.checklistItem;
      if (!itemId) {
        return;
      }
      const nextState = readChecklistState(jobId);
      nextState[itemId] = checkbox.checked;
      writeChecklistState(jobId, nextState);
      item.classList.toggle('done', checkbox.checked);
      const job = activeJob();
      if (job?.job_id === jobId) {
        reportActionLoop.innerHTML = renderReportActionLoop(
          job,
          job.summary || {},
          state.showLowConfidence ? job.risks || [] : (job.risks || []).filter((risk) => !isLowConfidenceRisk(risk)),
          job.fix_first || [],
          job.recommendations || [],
          job.evidence_frames || [],
          job.room_comparison || null,
        );
        renderFixVerification(
          job,
          job.summary || {},
          state.showLowConfidence ? job.risks || [] : (job.risks || []).filter((risk) => !isLowConfidenceRisk(risk)),
          job.fix_first || [],
          job.recommendations || [],
          job.room_comparison || null,
        );
        renderFixQuestPanel(
          job,
          job.summary || {},
          state.showLowConfidence ? job.risks || [] : (job.risks || []).filter((risk) => !isLowConfidenceRisk(risk)),
          job.fix_first || [],
          job.recommendations || [],
        );
        upsertHomeJournalFromJob(job);
        renderHomeJournal();
      }
      if (checkbox.checked) {
        markBingoTask('complete-fix-quest');
        renderHomeCompanion();
      }
      trackProductEvent('fix_checklist_toggled', {
        job_id: jobId,
        item_id: itemId,
        checked: checkbox.checked,
      });
    });
  });
}

function attachEvidenceTimelineHandlers() {
  evidenceTimeline.querySelectorAll('[data-evidence-target]').forEach((button) => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.evidenceTarget || 0);
      state.activeEvidenceIndex = index;
      writeStoredPreference(ACTIVE_EVIDENCE_STORAGE_KEY, String(index));
      evidenceTimeline.querySelectorAll('[data-evidence-target]').forEach((marker) => {
        marker.classList.toggle('active', marker === button);
      });
      const target = reportEvidence.querySelector(`[data-evidence-card="${button.dataset.evidenceTarget}"]`);
      target?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      const job = activeJob();
      trackProductEvent('evidence_frame_focused', {
        job_id: job?.job_id || null,
        sample_key: job?.sample_key || null,
        evidence_index: index,
        audience_mode: job?.audience_mode || selectedAudienceMode(),
      });
    });
  });
}

function attachConfidenceExplainerHandlers(job) {
  reportHazards.querySelectorAll('[data-confidence-explainer]').forEach((details) => {
    details.addEventListener('toggle', () => {
      if (!details.open) {
        return;
      }
      trackProductEvent('confidence_explainer_opened', {
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        finding_id: details.dataset.confidenceExplainer || null,
        audience_mode: job.audience_mode || 'general',
      });
      trackProductEvent('confidence_inspector_opened', {
        surface: 'finding_card',
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        audience_mode: job.audience_mode || 'general',
        room_labeled: Boolean(job.room_label || job.summary?.room_label),
      });
    });
  });
}

document.addEventListener('click', async (event) => {
  const button = event.target instanceof Element ? event.target.closest('[data-share-room-win]') : null;
  if (!button) {
    return;
  }
  const job = activeJob();
  if (!job || job.status !== 'complete') {
    return;
  }
  try {
    await copyText(buildRoomWinShareText(job));
    await trackProductEvent('room_win_copied', {
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      audience_mode: job.audience_mode || 'general',
    });
    await trackProductEvent('room_win_card_shared', {
      surface: 'room_scorecard',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      audience_mode: job.audience_mode || 'general',
      share_style: 'quick-win',
    });
    showToast('Room win copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy room win.', 3600);
  }
});

document.addEventListener('click', async (event) => {
  const target = event.target instanceof Element ? event.target : null;
  const postReportFeedback = target?.closest('[data-post-report-feedback]');
  if (postReportFeedback) {
    const job = activeJob();
    await trackProductEvent('post_report_feedback_submitted', {
      surface: 'post_report_prompt',
      job_id: job?.job_id || null,
      sample_key: job?.sample_key || null,
      reason: postReportFeedback.dataset.postReportFeedback || 'unknown',
      audience_mode: job?.audience_mode || selectedAudienceMode(),
    });
    showToast('Beta feedback noted. Thank you for helping ATLAS-0 learn.');
    return;
  }

  const jump = target?.closest('[data-jump-view]');
  if (jump instanceof HTMLElement) {
    switchView(jump.dataset.jumpView || 'scan');
    return;
  }

  const fixGuideButton = target?.closest('[data-open-fix-guide]');
  if (fixGuideButton) {
    const guideId = fixGuideButton.dataset.openFixGuide || FIX_LIBRARY_GUIDES[0].id;
    const guide = FIX_LIBRARY_GUIDES.find((item) => item.id === guideId) || FIX_LIBRARY_GUIDES[0];
    writeStoredPreference(FIX_GUIDE_STORAGE_KEY, guide.id);
    renderFixLibrary();
    await trackProductEvent('fix_library_opened', { surface: fixGuideButton.closest('#fix-library-panel') ? 'home_companion' : 'report' });
    await trackProductEvent('fix_guide_opened', { guide_id: guide.id });
    showToast(`${guide.title}: ${guide.steps[0]} Rescan to verify progress.`);
    return;
  }

  const timelineButton = target?.closest('[data-room-health-timeline-open]');
  if (timelineButton) {
    await trackProductEvent('room_health_timeline_opened', { room_count: journalEntries().length });
    showToast('Room Health Timeline is local history, not a safety certificate.');
    return;
  }

  const copyPassport = target?.closest('[data-copy-room-passport]');
  if (copyPassport) {
    const job = activeJob();
    if (!job || job.status !== 'complete') {
      return;
    }
    try {
      await copyText(buildRoomPassportText(job));
      await trackProductEvent('room_passport_opened', {
        surface: 'copy_passport',
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        audience_mode: job.audience_mode || 'general',
      });
      showToast('Room passport summary copied.');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not copy room passport.', 3600);
    }
    return;
  }

  const startVerification = target?.closest('[data-start-fix-verification]');
  if (startVerification) {
    const job = activeJob();
    if (!job || job.status !== 'complete') {
      return;
    }
    writeVerificationState(job.job_id, { startedAt: new Date().toISOString(), status: 'started' });
    renderFixVerification(job, job.summary || {}, job.risks || [], job.fix_first || [], job.recommendations || [], job.room_comparison || null);
    await trackProductEvent('fix_verification_started', {
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      audience_mode: job.audience_mode || 'general',
    });
    showToast('Fix Verification Mode started. Reuse this room label after your fix.');
    return;
  }

  const copyVerification = target?.closest('[data-copy-fix-verification]');
  if (copyVerification) {
    const job = activeJob();
    if (!job || job.status !== 'complete') {
      return;
    }
    try {
      await copyText(buildFixVerificationText(job));
      await trackProductEvent('fix_verification_copied', {
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        audience_mode: job.audience_mode || 'general',
      });
      showToast('Fix verification note copied.');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not copy verification note.', 3600);
    }
    return;
  }

  const questButton = target?.closest('[data-complete-fix-quest]');
  if (questButton) {
    const job = activeJob();
    if (!job || job.status !== 'complete') {
      return;
    }
    const questId = questButton.dataset.completeFixQuest || '';
    const nextState = readFixQuestState(job.job_id);
    nextState[questId] = nextState[questId] ? null : new Date().toISOString();
    if (!nextState[questId]) {
      delete nextState[questId];
    }
    writeFixQuestState(job.job_id, nextState);
    if (nextState[questId]) {
      markBingoTask('complete-fix-quest');
      await trackProductEvent('fix_quest_completed', {
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        quest_id: questId,
        audience_mode: job.audience_mode || 'general',
      });
    }
    renderFixQuestPanel(
      job,
      job.summary || {},
      state.showLowConfidence ? job.risks || [] : (job.risks || []).filter((risk) => !isLowConfidenceRisk(risk)),
      job.fix_first || [],
      job.recommendations || [],
    );
    upsertHomeJournalFromJob(job);
    renderHomeJournal();
    renderHomeCompanion();
    showToast(nextState[questId] ? 'Fix Quest completed locally.' : 'Fix Quest reopened.');
    return;
  }

  const compareButton = target?.closest('[data-room-compare-open]');
  if (compareButton) {
    const job = activeJob();
    await trackProductEvent('room_compare_opened', {
      job_id: job?.job_id || null,
      sample_key: job?.sample_key || null,
      audience_mode: job?.audience_mode || selectedAudienceMode(),
    });
    showToast('Room Compare Mode is a prioritization nudge, not a measurement certificate.');
    return;
  }

  const smartRescanButton = target?.closest('[data-smart-rescan-open]');
  if (smartRescanButton) {
    const job = activeJob();
    await trackProductEvent('smart_rescan_coach_opened', {
      job_id: job?.job_id || null,
      sample_key: job?.sample_key || null,
      audience_mode: job?.audience_mode || selectedAudienceMode(),
    });
    showToast('Smart Rescan Coach opened. Repeat the room label and route after one fix.');
    return;
  }

  const evidenceStoryButton = target?.closest('[data-evidence-story-open]');
  if (evidenceStoryButton) {
    const job = activeJob();
    await trackProductEvent('evidence_story_opened', {
      job_id: job?.job_id || null,
      sample_key: job?.sample_key || null,
      audience_mode: job?.audience_mode || selectedAudienceMode(),
    });
    showToast('Evidence Story opened. Frames are still approximate support, not proof.');
  }
});

document.addEventListener('click', async (event) => {
  const target = event.target instanceof Element ? event.target : null;
  if (!target) {
    return;
  }

  const challengeWinButton = target.closest('[data-copy-challenge-win]');
  if (challengeWinButton) {
    const job = activeJob();
    if (!job || job.status !== 'complete') {
      return;
    }
    try {
      const challenge = challengeForJob(job);
      await copyText(buildChallengeWinText(job));
      await trackProductEvent('room_win_copied', {
        surface: 'challenge_result_card',
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        mission_id: challenge.id,
        challenge_id: challenge.id,
        audience_mode: job.audience_mode || 'general',
      });
      await trackProductEvent('room_win_card_shared', {
        surface: 'challenge_result_card',
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        mission_id: challenge.id,
        challenge_id: challenge.id,
        audience_mode: job.audience_mode || 'general',
        share_style: 'weekly-challenge',
      });
      showToast('Challenge win copied.');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not copy challenge win.', 3600);
    }
    return;
  }

  const completeChallengeButton = target.closest('[data-complete-challenge]');
  if (completeChallengeButton) {
    const challenge = challengeById(completeChallengeButton.dataset.completeChallenge) || activeChallenge();
    completeDailyMission(challenge);
    const job = activeJob();
    if (job?.status === 'complete') {
      renderChallengeResultCard(
        job,
        job.summary || {},
        state.showLowConfidence ? job.risks || [] : (job.risks || []).filter((risk) => !isLowConfidenceRisk(risk)),
        job.fix_first || [],
        job.recommendations || [],
        job.room_comparison || null,
      );
    }
    return;
  }

  const betaButton = target.closest('[data-copy-beta-invite]');
  if (betaButton) {
    try {
      await copyBetaInvite('report_action_loop');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not copy beta invite.', 3600);
    }
    return;
  }

  const fixPlanButton = target.closest('[data-copy-fix-plan]');
  if (fixPlanButton) {
    const job = activeJob();
    if (!job || job.status !== 'complete') {
      return;
    }
    try {
      await copyText(buildFixPlanText(job));
      await trackProductEvent('fix_plan_copied', {
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        audience_mode: job.audience_mode || 'general',
      });
      showToast('Fix plan copied.');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not copy fix plan.', 3600);
    }
    return;
  }

  const fixTodayButton = target.closest('[data-copy-fix-today]');
  if (fixTodayButton) {
    const job = activeJob();
    if (!job || job.status !== 'complete') {
      return;
    }
    try {
      await copyText(buildFixTodayText(job));
      await trackProductEvent('fix_today_copied', {
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        audience_mode: job.audience_mode || 'general',
      });
      showToast('Fix Today copied.');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not copy Fix Today.', 3600);
    }
    return;
  }

  const openJournalReportButton = target.closest('[data-open-journal-report]');
  if (openJournalReportButton) {
    const jobId = openJournalReportButton.dataset.openJournalReport;
    const sampleKey = openJournalReportButton.dataset.sampleKey;
    if (sampleKey) {
      await loadSampleReport();
      return;
    }
    if (jobId && state.jobs.has(jobId)) {
      setActiveJob(jobId);
      switchView('report');
      return;
    }
    if (jobId) {
      try {
        upsertJob(await api.fetchJob(jobId));
        setActiveJob(jobId);
        switchView('report');
      } catch (error) {
        showToast(error instanceof Error ? error.message : 'Could not open journal report.', 3600);
      }
    }
    return;
  }

  const copyJournalPassportButton = target.closest('[data-copy-journal-passport]');
  if (copyJournalPassportButton) {
    const roomKey = copyJournalPassportButton.dataset.copyJournalPassport;
    const entry = readHomeJournal()[roomKey];
    if (!entry) {
      showToast('No local room passport found yet.');
      return;
    }
    try {
      await copyText([
        `ATLAS-0 Room Health Passport: ${entry.roomLabel || 'Room'}`,
        `Calm Score: ${entry.lastScore === null || entry.lastScore === undefined ? 'pending' : `${entry.lastScore}/100`}.`,
        `Recurring attention area: ${entry.topAction || 'Review the Safety Brief'}.`,
        `Completed fixes: ${entry.completedFixes || 0}. Last checked: ${entry.lastCheckedAt ? new Date(entry.lastCheckedAt).toLocaleDateString() : 'recently'}.`,
        'Decision support only, not safety certification.',
      ].join('\n'));
      await trackProductEvent('room_passport_opened', {
        surface: 'home_journal_copy',
        room_key: roomKey || null,
        room_label: entry.roomLabel || null,
        room_labeled: Boolean(entry.roomLabel),
      });
      showToast('Room passport copied.');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not copy room passport.', 3600);
    }
    return;
  }

  const favoriteRoomButton = target.closest('[data-favorite-room]');
  if (favoriteRoomButton) {
    const roomKey = favoriteRoomButton.dataset.favoriteRoom;
    const favorites = readFavoriteRooms();
    if (favorites.has(roomKey)) {
      favorites.delete(roomKey);
    } else if (roomKey) {
      favorites.add(roomKey);
    }
    writeFavoriteRooms(favorites);
    renderHomeJournal();
    showToast(favorites.has(roomKey) ? 'Room favorited locally.' : 'Room removed from favorites.');
    return;
  }

  const roomReminderButton = target.closest('[data-room-reminder]');
  if (roomReminderButton) {
    const roomKey = roomReminderButton.dataset.roomReminder;
    const entry = readHomeJournal()[roomKey];
    await trackProductEvent('room_reminder_clicked', {
      surface: 'home_journal',
      room_key: roomKey || null,
      room_label: entry?.roomLabel || null,
      room_labeled: Boolean(entry?.roomLabel),
    });
    showToast('Reminder idea saved mentally for now: rescan this room next week with the same label.', 3800);
    return;
  }

  const homePulseButton = target.closest('[data-open-home-pulse]');
  if (homePulseButton) {
    await trackProductEvent('home_pulse_opened', { surface: 'home_pulse_card' });
    switchView('journal');
    showToast('Home Journal opened from Home Pulse.');
    return;
  }

  const roomMapButton = target.closest('[data-room-map-preview]');
  if (roomMapButton) {
    const job = activeJob();
    await trackProductEvent('room_map_preview_opened', {
      surface: roomMapButton.dataset.mapMarker ? 'map_marker' : 'map_panel',
      map_marker: roomMapButton.dataset.mapMarker || null,
      job_id: job?.job_id || null,
      sample_key: job?.sample_key || null,
      audience_mode: job?.audience_mode || selectedAudienceMode(),
    });
    showToast('Approximate evidence map is for orientation only. Confirm with frames.');
    return;
  }

  const beforeAfterButton = target.closest('[data-copy-before-after-story]');
  if (beforeAfterButton) {
    const job = activeJob();
    if (!job || job.status !== 'complete' || !job.room_comparison) {
      return;
    }
    try {
      await copyText(buildBeforeAfterStoryText(job));
      await trackProductEvent('before_after_card_copied', {
        job_id: job.job_id,
        sample_key: job.sample_key || null,
        audience_mode: job.audience_mode || 'general',
      });
      showToast('Before/after story card copied.');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not copy before/after story.', 3600);
    }
    return;
  }

  const rescanButton = target.closest('[data-start-rescan]');
  if (rescanButton) {
    const job = activeJob();
    const summary = job?.summary || {};
    if (job) {
      const challenge = challengeForJob(job);
      state.activeChallengeId = challenge.id;
      state.pendingUploadChallengeId = challenge.id;
      writeStoredPreference(CHALLENGE_SELECTION_STORAGE_KEY, challenge.id);
      renderChallengeLibrary();
    }
    if (roomLabelInput && job) {
      roomLabelInput.value = job.room_label || summary.room_label || roomLabelInput.value;
    }
    if (audienceModeInput && job?.audience_mode) {
      audienceModeInput.value = job.audience_mode;
      renderCaptureCoach();
    }
    await trackProductEvent('same_room_rescan_started', {
      job_id: job?.job_id || null,
      mission_id: job ? challengeForJob(job).id : activeChallenge().id,
      challenge_id: job ? challengeForJob(job).id : activeChallenge().id,
      room_labeled: Boolean(roomLabelInput?.value),
    });
    await trackProductEvent('rescan_prompt_clicked', {
      surface: rescanButton.closest('#challenge-result-card') ? 'challenge_result_card' : 'report_action_loop',
      job_id: job?.job_id || null,
      mission_id: job ? challengeForJob(job).id : activeChallenge().id,
      challenge_id: job ? challengeForJob(job).id : activeChallenge().id,
      room_labeled: Boolean(roomLabelInput?.value),
    });
    switchView('scan');
    showToast('Same-room rescan is ready. Reuse the room label and upload the follow-up walkthrough.');
  }
});

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function capitalize(value) {
  const text = String(value || '');
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function formatEvidenceMeta(frame) {
  const parts = [];
  if (typeof frame.frame_index === 'number') {
    parts.push(`Frame ${frame.frame_index}`);
  }
  if (typeof frame.timestamp_s === 'number') {
    parts.push(`${frame.timestamp_s.toFixed(1)}s`);
  }
  if (frame.object_label) {
    parts.push(String(frame.object_label));
  }
  if (frame.redacted) {
    parts.push('Text-heavy crop blurred');
  }
  return parts.join(' · ') || 'Stored evidence crop';
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

function formatGateValue(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value ?? '—');
  }
  if (numeric >= 0 && numeric <= 1 && !Number.isInteger(numeric)) {
    return `${Math.round(numeric * 100)}%`;
  }
  return String(Math.round(numeric * 100) / 100);
}

function isLowConfidenceRisk(risk) {
  return Number(risk?.confidence || 0) < 0.6 || Number(risk?.reasoning?.grounding_confidence || 0) < 0.55;
}

function renderFeedbackButton(verdict, activeVerdict) {
  const activeClass = verdict === activeVerdict ? 'active' : '';
  return `<button class="feedback-btn ${activeClass}" type="button" data-feedback="${verdict}">${capitalize(verdict)}</button>`;
}

function renderEvalButton(status, activeStatus) {
  const activeClass = status === activeStatus ? 'active' : '';
  const label = status === 'needs_review'
    ? 'Needs Review'
    : status === 'missed_hazard'
      ? 'Missed Hazard'
      : 'Confirmed';
  return `<button class="feedback-btn ${activeClass}" type="button" data-eval-status="${status}">${label}</button>`;
}

function renderFollowUpButton(status, activeStatus) {
  const activeClass = status === activeStatus ? 'active' : '';
  return `<button class="feedback-btn ${activeClass}" type="button" data-follow-up="${status}">${formatFollowUpLabel(status)}</button>`;
}

function formatFollowUpLabel(status) {
  if (status === 'resolved') return 'Resolved';
  if (status === 'monitor') return 'Monitor';
  if (status === 'ignored') return 'Ignored';
  return 'Open';
}

function renderScanQuality(scanQuality) {
  if (!scanQuality || Object.keys(scanQuality).length === 0) {
    return emptyMarkup('No scan quality diagnostics were recorded for this job.');
  }

  const warnings = scanQuality.warnings || [];
  const guidance = scanQuality.retry_guidance || [];
  const metrics = scanQuality.metrics || {};

  return `
    <div class="quality-score">
      <strong>${Math.round((scanQuality.score || 0) * 100)}</strong>
      <span class="severity-pill ${qualityTone(scanQuality.status)}">${escapeHtml(scanQuality.status || 'unknown')}</span>
    </div>
    <div class="quality-copy">
      <strong>What this means</strong>
      <span>${escapeHtml(scanQuality.capture_summary || (scanQuality.usable ? 'This scan is usable, but better lighting, steadier motion, and fuller coverage can still strengthen the report.' : 'This scan is likely to produce weaker findings and should ideally be rescanned before trusting smaller details.'))}</span>
    </div>
    <div class="report-card-meta">
      <span>${escapeHtml(capitalize(scanQuality.reportability || 'accepted'))} reportability</span>
      <span>${scanQuality.hard_reject ? 'Normal report refused' : scanQuality.rescan_recommended ? 'Report downgraded' : 'Normal report allowed'}</span>
    </div>
    <div class="report-card-meta">
      <span>${escapeHtml(`${metrics.frame_count || 0} sampled frame(s)`)}</span>
      <span>${escapeHtml(`${Math.round((metrics.motion_coverage || 0) * 100)}% motion coverage`)}</span>
      <span>${escapeHtml(`${Math.round((metrics.saliency_coverage || 0) * 100)}% object coverage`)}</span>
    </div>
    <div class="quality-copy">
      <strong>Coach tip</strong>
      <span>${escapeHtml(scanQuality.rescan_recommended
        ? 'For a stronger follow-up scan, move slower near shelves and counters, include the full room perimeter, and add light before rescanning.'
        : 'This scan is usable. A before/after rescan with the same room label will make progress easier to see.')}</span>
    </div>
    ${warnings.length ? `<ul class="quality-warning-list">${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('')}</ul>` : '<div class="empty-card">No major scan-quality warnings were detected.</div>'}
    ${scanQuality.rejection_reasons?.length ? `<ul class="quality-warning-list">${scanQuality.rejection_reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join('')}</ul>` : ''}
    ${guidance.length ? `<div class="quality-copy"><strong>Retry guidance</strong><span>${escapeHtml(guidance.slice(0, 2).join(' '))}</span></div>` : ''}
  `;
}

function renderReportPosture(summary, scanQuality, job) {
  const posture = summary.report_posture || 'screening';
  const guidance = scanQuality.retry_guidance || [];
  const comparison = job.room_comparison || null;
  const history = Array.isArray(job.room_history) ? job.room_history : [];
  const resolution = job.resolution_summary || {};

  return `
    <div class="report-copy-block">
      <strong>What this report supports</strong>
      <span>${escapeHtml(summary.overview || 'This upload produced a first-pass room hazard screen.')}</span>
    </div>
    <div class="report-copy-block">
      <strong>Coverage summary</strong>
      <span>${escapeHtml(summary.coverage_summary || 'Coverage details were not recorded for this scan.')}</span>
    </div>
    <div class="report-copy-block">
      <strong>What it does not claim</strong>
      <span>${escapeHtml(summary.screening_statement || 'This report flags likely hazards from the uploaded scan. It does not certify that the room is safe.')}</span>
    </div>
    ${typeof summary.room_score === 'number' ? `
      <div class="report-copy-block">
        <strong>Room safety score foundation</strong>
        <span>${escapeHtml(`${summary.room_score}/100 · ${summary.room_score_band || 'screening score'}. ${summary.room_score_summary || ''}`)}</span>
      </div>
    ` : ''}
    ${comparison ? `
      <div class="report-copy-block">
        <strong>Before / after comparison</strong>
        <span>${escapeHtml(`${comparison.summary} Score change: ${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta}. Hazard delta: ${comparison.hazard_delta > 0 ? '+' : ''}${comparison.hazard_delta}.`)}</span>
      </div>
    ` : ''}
    ${resolution.total_findings ? `
      <div class="report-copy-block">
        <strong>Follow-through state</strong>
        <span>${escapeHtml(resolution.summary || 'No follow-up state yet.')}</span>
      </div>
    ` : ''}
    <div class="report-card-meta">
      <span>${escapeHtml(posture)}</span>
      <span>${escapeHtml(summary.coverage_label || 'Unknown')} coverage</span>
      <span>${summary.rescan_recommended ? 'Rescan recommended' : 'No rescan required for first-pass review'}</span>
    </div>
    ${resolution.total_findings ? `
      <div class="report-card-meta">
        <span>${escapeHtml(String(resolution.resolved_count || 0))} resolved</span>
        <span>${escapeHtml(String(resolution.monitor_count || 0))} monitor</span>
        <span>${escapeHtml(String(resolution.ignored_count || 0))} ignored</span>
        <span>${escapeHtml(String(resolution.open_count || 0))} open</span>
      </div>
    ` : ''}
    ${history.length ? `<div class="report-card-meta">${history.slice(0, 3).map((entry) => `<span>${escapeHtml(`${entry.filename} · ${entry.room_score ?? '—'}/100`)}</span>`).join('')}</div>` : ''}
    ${guidance.length ? `<ul class="quality-warning-list">${guidance.slice(0, 2).map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}
  `;
}

function renderEvaluationSummary(evaluation, job) {
  if (!evaluation || Object.keys(evaluation).length === 0) {
    return emptyMarkup('No review coverage has been recorded for this report yet.');
  }

  const controls = job.is_sample
    ? '<p class="meta-copy">Sample reports stay read-only so the built-in walkthrough remains stable for every visitor.</p>'
    : `
      <div class="evaluation-controls" data-eval-controls data-job-id="${job.job_id}">
        <div class="feedback-row">
          ${renderEvalButton('confirmed', evaluation.human_status)}
          ${renderEvalButton('needs_review', evaluation.human_status)}
          ${renderEvalButton('missed_hazard', evaluation.human_status)}
        </div>
        <div class="feedback-row">
          <button class="feedback-btn" type="button" data-export-eval-candidate="true">Save Eval Case</button>
        </div>
        <p class="meta-copy">Use these controls to build the eval set, log missed hazards, and keep the beta report loop honest.</p>
      </div>
    `;
  const evalActions = Array.isArray(evaluation.eval_corpus_actions) ? evaluation.eval_corpus_actions : [];

  return `
    <div class="report-copy-block">
      <strong>Review status</strong>
      <span>${escapeHtml(evaluation.summary || 'No review summary available.')}</span>
    </div>
    <div class="meta-grid">
      <div class="meta-tile">
        <span>Review coverage</span>
        <strong>${Math.round((evaluation.review_coverage || 0) * 100)}%</strong>
      </div>
      <div class="meta-tile">
        <span>Pending findings</span>
        <strong>${escapeHtml(String(evaluation.pending_findings ?? 0))}</strong>
      </div>
      <div class="meta-tile">
        <span>Marked useful</span>
        <strong>${escapeHtml(String(evaluation.useful_events ?? 0))}</strong>
      </div>
      <div class="meta-tile">
        <span>Wrong or duplicate</span>
        <strong>${escapeHtml(String((evaluation.wrong_events || 0) + (evaluation.duplicate_events || 0)))}</strong>
      </div>
      <div class="meta-tile">
        <span>Precision proxy</span>
        <strong>${Math.round((evaluation.precision_proxy || 0) * 100)}%</strong>
      </div>
      <div class="meta-tile">
        <span>Recall proxy</span>
        <strong>${Math.round((evaluation.recall_proxy || 0) * 100)}%</strong>
      </div>
    </div>
    <div class="report-card-meta">
      <span>${escapeHtml(String(evaluation.high_priority_pending ?? 0))} high-priority findings unreviewed</span>
      <span>${escapeHtml(String(evaluation.missed_hazard_count ?? 0))} missed hazards logged</span>
      <span>${escapeHtml(evaluation.benchmark_label ? `${evaluation.benchmark_label} benchmark` : 'No benchmark tag')}</span>
      <span>${evaluation.needs_review ? 'More review still needed' : 'Review loop covered current findings'}</span>
    </div>
    <div class="evaluation-status">
      <span>Human verdict: ${escapeHtml(evaluation.human_status || 'not set')}</span>
      <span>${escapeHtml(evaluation.benchmark_match === true ? 'Benchmark matched' : evaluation.benchmark_match === false ? 'Benchmark mismatch' : 'No benchmark comparison')}</span>
      <span>${escapeHtml(`Eval priority: ${evaluation.eval_priority || 'unknown'}`)}</span>
    </div>
    <div class="eval-corpus-tooling">
      <strong>Eval corpus next label</strong>
      <p>${escapeHtml(evaluation.eval_candidate_reason || 'Use this report to grow the labeled eval set after review.')}</p>
      <ul class="settings-list">
        ${evalActions.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}
      </ul>
    </div>
    ${controls}
  `;
}

function qualityTone(status) {
  if (status === 'good') return 'low';
  if (status === 'fair') return 'medium';
  return 'high';
}

function stageExplanation(stage) {
  if (stage === 'upload') return 'The upload was accepted and queued for processing.';
  if (stage === 'ingest') return 'The media is being unpacked so frames can be sampled cleanly.';
  if (stage === 'vlm') return 'ATLAS-0 is labeling observations and grounding them across the scan.';
  if (stage === 'risk') return 'Findings, recommendations, and evidence artifacts are being assembled into the report.';
  if (stage === 'complete') return 'The report is ready to review.';
  return 'ATLAS-0 is processing the scan.';
}

function attachFeedbackHandlers(jobId) {
  reportHazards.querySelectorAll('[data-feedback-controls]').forEach((container) => {
    container.querySelectorAll('[data-feedback]').forEach((button) => {
      button.addEventListener('click', async () => {
        try {
          const updated = await api.submitFindingFeedback(jobId, {
            hazard_code: container.dataset.hazardCode,
            object_id: container.dataset.objectId || null,
            verdict: button.dataset.feedback,
          });
          await trackProductEvent('post_report_feedback_submitted', {
            surface: 'finding_feedback',
            job_id: jobId,
            reason: button.dataset.feedback,
          });
          upsertJob(updated);
          showToast('Feedback saved.');
        } catch (error) {
          showToast(error instanceof Error ? error.message : 'Could not save feedback.', 3600);
        }
      });
    });
  });
}

function attachFollowUpHandlers(jobId) {
  reportHazards.querySelectorAll('[data-follow-up-controls]').forEach((container) => {
    container.querySelectorAll('[data-follow-up]').forEach((button) => {
      button.addEventListener('click', async () => {
        const nextStatus = button.dataset.followUp || '';
        const activeStatus = container.dataset.activeStatus || '';
        const status = nextStatus === activeStatus ? 'open' : nextStatus;
        try {
          const updated = await api.submitFindingFollowUp(jobId, {
            hazard_code: container.dataset.hazardCode,
            object_id: container.dataset.objectId || null,
            status,
          });
          upsertJob(updated);
          showToast(status === 'open' ? 'Finding reset to open.' : 'Follow-up saved.');
        } catch (error) {
          showToast(error instanceof Error ? error.message : 'Could not save follow-up.', 3600);
        }
      });
    });
  });
}

function attachEvaluationHandlers(jobId) {
  reportEvalCard.querySelectorAll('[data-eval-controls]').forEach((container) => {
    container.querySelectorAll('[data-export-eval-candidate]').forEach((button) => {
      button.addEventListener('click', async () => {
        const suggested = `${jobId}-eval`;
        const label = window.prompt(
          'Optional eval-case label. Leave it as-is or clear it to use the job ID.',
          suggested,
        );
        if (label === null) {
          return;
        }

        try {
          const updated = await api.exportEvalCandidate(jobId, {
            label: label.trim() || null,
          });
          upsertJob(updated);
          await refreshOperatorState();
          showToast('Eval case saved.');
        } catch (error) {
          showToast(error instanceof Error ? error.message : 'Could not save eval case.', 3600);
        }
      });
    });

    container.querySelectorAll('[data-eval-status]').forEach((button) => {
      button.addEventListener('click', async () => {
        const status = button.dataset.evalStatus;
        if (!status) {
          return;
        }

        let missedHazards = [];
        let note = '';
        if (status === 'missed_hazard') {
          const answer = window.prompt(
            'List any missed hazards as a comma-separated note. Leave blank if you just want to flag a miss.',
            '',
          );
          if (answer === null) {
            return;
          }
          note = answer.trim();
          missedHazards = note
            ? note.split(',').map((item) => item.trim()).filter(Boolean)
            : [];
        }

        try {
          const updated = await api.submitJobEvaluation(jobId, {
            status,
            missed_hazards: missedHazards,
            note: note || null,
          });
          upsertJob(updated);
          await refreshOperatorState();
          showToast('Evaluation saved.');
        } catch (error) {
          showToast(error instanceof Error ? error.message : 'Could not save evaluation.', 3600);
        }
      });
    });
  });
}
