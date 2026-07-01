// ============================================================
// store/appStore.ts
// Single Zustand store. All global UI and domain state lives here.
// Components destructure only what they need — Zustand's selector
// subscriptions prevent unnecessary re-renders.
// ============================================================

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { persist } from 'zustand/middleware';

import type {
  ViewMode,
  DetectRequest,
  DetectBatchResponse,
  AnomalyExplanationResponse,
  ChecklistStep,
  MeterFormState,
} from '@/types';

// ── State shape ───────────────────────────────────────────────

interface AppState {
  // ── Global UI preferences (persisted to localStorage) ──────
  viewMode:               ViewMode;
  decisionEngineEnabled:  boolean;

  // ── Detection slice ─────────────────────────────────────────
  detection: {
    lastRequest:       DetectRequest | null;
    lastResponse:      DetectBatchResponse | null;
    checklistSteps:    ChecklistStep[];
    isLoading:         boolean;
    error:             string | null;
    sessionAnomalyIds: number[];   // collected across all detect runs
  };

  // ── Explanation slice ────────────────────────────────────────
  explanation: {
    inputAnomalyId:   string;                          // controlled input
    response:         AnomalyExplanationResponse | null;
    checklistSteps:   ChecklistStep[];
    isLoading:        boolean;
    isPolling:        boolean;
    pollElapsedMs:    number;
    timedOut:         boolean;
    error:            string | null;
  };

  // ── Form state (non-technical detection input) ───────────────
  meterForm: MeterFormState;
}

// ── Actions ───────────────────────────────────────────────────

interface AppActions {
  // Global
  setViewMode:              (mode: ViewMode) => void;
  setDecisionEngineEnabled: (enabled: boolean) => void;

  // Detection
  setDetectionLoading:    (loading: boolean) => void;
  setDetectionRequest:    (req: DetectRequest) => void;
  setDetectionResponse:   (res: DetectBatchResponse) => void;
  setDetectionError:      (err: string | null) => void;
  setDetectionChecklist:  (steps: ChecklistStep[]) => void;
  updateDetectionStep:    (id: string, status: ChecklistStep['status']) => void;
  resetDetection:         () => void;
  addSessionAnomalyId:    (id: number) => void;

  // Explanation
  setExplanationInputId:    (id: string) => void;
  setExplanationLoading:    (loading: boolean) => void;
  setExplanationPolling:    (polling: boolean) => void;
  setExplanationResponse:   (res: AnomalyExplanationResponse) => void;
  setExplanationError:      (err: string | null) => void;
  setExplanationChecklist:  (steps: ChecklistStep[]) => void;
  updateExplanationStep:    (id: string, status: ChecklistStep['status']) => void;
  tickPollElapsed:          (deltaMs: number) => void;
  setExplanationTimedOut:   (timedOut: boolean) => void;
  resetExplanation:         () => void;

  // Meter form
  setMeterFormField:      (field: keyof MeterFormState, value: MeterFormState[keyof MeterFormState]) => void;
  setMeterFormGroup:      (group: string, features: string[]) => void;
  setMeterFormFeature:    (feature: string, value: string) => void;
  setMeterFormAdvanced:   (field: keyof MeterFormState['advanced'], value: string) => void;
}

// ── Initial state factories ───────────────────────────────────

const initialDetection: AppState['detection'] = {
  lastRequest:       null,
  lastResponse:      null,
  checklistSteps:    [],
  isLoading:         false,
  error:             null,
  sessionAnomalyIds: [],
};

const initialExplanation: AppState['explanation'] = {
  inputAnomalyId: '',
  response:        null,
  checklistSteps:  [],
  isLoading:       false,
  isPolling:       false,
  pollElapsedMs:   0,
  timedOut:        false,
  error:           null,
};

const initialMeterForm: MeterFormState = {
  meterSerial:   '',
  selectedGroup: 'group_A',
  fieldValues:   {},
  advanced: {
    id:        '',
    entryId:   '1',
    obisCode:  '1.0.99.1.0.255',
    timestamp: '',
  },
};

// ── Store ─────────────────────────────────────────────────────

export const useAppStore = create<AppState & AppActions>()(
  persist(
    immer((set) => ({
      // ── Initial state ──────────────────────────────────────
      viewMode:              'technical',
      decisionEngineEnabled: true,
      detection:             initialDetection,
      explanation:           initialExplanation,
      meterForm:             initialMeterForm,

      // ── Global actions ──────────────────────────────────────
      setViewMode: (mode) => set((s) => { s.viewMode = mode; }),
      setDecisionEngineEnabled: (enabled) => set((s) => { s.decisionEngineEnabled = enabled; }),

      // ── Detection actions ───────────────────────────────────
      setDetectionLoading: (loading) => set((s) => { s.detection.isLoading = loading; }),
      setDetectionRequest: (req)     => set((s) => { s.detection.lastRequest = req; }),
      setDetectionResponse: (res)    => set((s) => {
        s.detection.lastResponse = res;
        // Collect all new anomaly IDs from this batch into session list
        res.results.forEach((r) => {
          if (r.anomaly_id && !s.detection.sessionAnomalyIds.includes(r.anomaly_id)) {
            s.detection.sessionAnomalyIds.push(r.anomaly_id);
          }
        });
      }),
      setDetectionError: (err)       => set((s) => { s.detection.error = err; }),
      setDetectionChecklist: (steps) => set((s) => { s.detection.checklistSteps = steps; }),
      updateDetectionStep: (id, status) => set((s) => {
        const step = s.detection.checklistSteps.find((st) => st.id === id);
        if (step) step.status = status;
      }),
      resetDetection: () => set((s) => {
        s.detection = { ...initialDetection, sessionAnomalyIds: s.detection.sessionAnomalyIds };
      }),
      addSessionAnomalyId: (id) => set((s) => {
        if (!s.detection.sessionAnomalyIds.includes(id)) {
          s.detection.sessionAnomalyIds.push(id);
        }
      }),

      // ── Explanation actions ──────────────────────────────────
      setExplanationInputId: (id)   => set((s) => { s.explanation.inputAnomalyId = id; }),
      setExplanationLoading: (l)    => set((s) => { s.explanation.isLoading = l; }),
      setExplanationPolling: (p)    => set((s) => { s.explanation.isPolling = p; }),
      setExplanationResponse: (res) => set((s) => { s.explanation.response = res; }),
      setExplanationError: (err)    => set((s) => { s.explanation.error = err; }),
      setExplanationChecklist: (steps) => set((s) => { s.explanation.checklistSteps = steps; }),
      updateExplanationStep: (id, status) => set((s) => {
        const step = s.explanation.checklistSteps.find((st) => st.id === id);
        if (step) step.status = status;
      }),
      tickPollElapsed: (deltaMs) => set((s) => { s.explanation.pollElapsedMs += deltaMs; }),
      setExplanationTimedOut: (timedOut) => set((s) => { s.explanation.timedOut = timedOut; }),
      resetExplanation: () => set((s) => {
        s.explanation = { ...initialExplanation, inputAnomalyId: s.explanation.inputAnomalyId };
      }),

      // ── Meter form actions ──────────────────────────────────
      setMeterFormField: (field, value) => set((s) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (s.meterForm as any)[field] = value;
      }),
      setMeterFormGroup: (group, features) => set((s) => {
        s.meterForm.selectedGroup = group;
        // Reset field values when group changes — keep only overlapping fields
        const next: Record<string, string> = {};
        features.forEach((f) => {
          next[f] = s.meterForm.fieldValues[f] ?? '';
        });
        s.meterForm.fieldValues = next;
      }),
      setMeterFormFeature: (feature, value) => set((s) => {
        s.meterForm.fieldValues[feature] = value;
      }),
      setMeterFormAdvanced: (field, value) => set((s) => {
        s.meterForm.advanced[field] = value;
      }),
    })),
    {
      name: 'ecosentinel-prefs',
      // Only persist user preferences, not transient request/response data
      partialize: (state) => ({
        viewMode:              state.viewMode,
        decisionEngineEnabled: state.decisionEngineEnabled,
      }),
    },
  ),
);
