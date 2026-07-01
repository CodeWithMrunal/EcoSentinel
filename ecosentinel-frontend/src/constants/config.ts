// ============================================================
// constants/config.ts
// Single source of truth for all static configuration.
//
// EXTENSIBILITY GUIDE:
//   New capability group  → add to CAPABILITY_GROUPS
//   New OBIS code/label   → add to OBIS_HUMAN_LABELS
//   New LLM model         → add to LLM_MODEL_GROUPS
//   New detection layer   → add to DETECTION_LAYER_LABELS
//   New rule violation    → add to RULE_VIOLATION_LABELS
// ============================================================

import type { CapabilityGroupDef, LLMModelGroup } from '@/types';

// ── Capability Groups ─────────────────────────────────────────
// Mirrors settings.py CAPABILITY_GROUPS exactly.
// Adding a new group here automatically adds it to the
// GroupSelector and DynamicMeterForm in the non-tech input.

export const CAPABILITY_GROUPS: Record<string, CapabilityGroupDef> = {
  group_A: {
    label: 'Group A',
    description: 'Energy + Voltage + Current + Power Factor',
    features: ['energy_consumption', 'voltage', 'current', 'power_factor'],
  },
  group_B: {
    label: 'Group B',
    description: 'Energy + Apparent Energy + Voltage',
    features: ['energy_consumption', 'apparent_import_energy', 'voltage'],
  },
  group_C: {
    label: 'Group C',
    description: 'Energy + Current',
    features: ['energy_consumption', 'current'],
  },
  group_D: {
    label: 'Group D',
    description: 'Full Feature Set (all electrical parameters)',
    features: [
      'energy_consumption',
      'active_export_energy',
      'apparent_import_energy',
      'voltage',
      'current',
      'power_factor',
      'frequency',
    ],
  },
  group_E: {
    label: 'Group E',
    description: 'Energy Only',
    features: ['energy_consumption'],
  },
  group_V: {
    label: 'Group V',
    description: 'Voltage + Current (no energy)',
    features: ['voltage', 'current'],
  },
};

// ── OBIS Human-Readable Labels ─────────────────────────────────
// Maps canonical feature name → human label + unit for the
// non-technical form and result display.
// Adding a new OBIS code here automatically labels it everywhere.

export const OBIS_HUMAN_LABELS: Record<string, { label: string; unit: string; placeholder: string }> = {
  energy_consumption: {
    label: 'Active Import Energy',
    unit: 'Wh',
    placeholder: 'e.g. 1.6',
  },
  active_export_energy: {
    label: 'Active Export Energy',
    unit: 'Wh',
    placeholder: 'e.g. 0.0',
  },
  apparent_import_energy: {
    label: 'Apparent Import Energy',
    unit: 'VAh',
    placeholder: 'e.g. 1.8',
  },
  apparent_export_energy: {
    label: 'Apparent Export Energy',
    unit: 'VAh',
    placeholder: 'e.g. 0.0',
  },
  reactive_import_energy: {
    label: 'Reactive Import Energy',
    unit: 'VARh',
    placeholder: 'e.g. 0.5',
  },
  reactive_export_energy: {
    label: 'Reactive Export Energy',
    unit: 'VARh',
    placeholder: 'e.g. 0.0',
  },
  active_import_power: {
    label: 'Active Import Power',
    unit: 'W',
    placeholder: 'e.g. 320',
  },
  active_export_power: {
    label: 'Active Export Power',
    unit: 'W',
    placeholder: 'e.g. 0',
  },
  voltage: {
    label: 'Voltage',
    unit: 'V',
    placeholder: 'e.g. 225.9',
  },
  current: {
    label: 'Current',
    unit: 'A',
    placeholder: 'e.g. 1.4',
  },
  power_factor: {
    label: 'Power Factor',
    unit: '',
    placeholder: 'e.g. 0.92',
  },
  frequency: {
    label: 'Frequency',
    unit: 'Hz',
    placeholder: 'e.g. 50.0',
  },
};

// ── Detection Layer Display Labels ────────────────────────────
// Maps internal layer keys → human-readable labels.
// Adding a new detection layer here labels it in the results UI.

export const DETECTION_LAYER_LABELS: Record<string, string> = {
  rule_based: 'Rule-Based',
  zscore: 'Statistical',
  isolation_forest: 'ML Model',
};

// ── Rule Violation Human Labels ───────────────────────────────
// Maps raw rule violation strings → plain English descriptions.
// Adding a new rule violation here labels it in non-tech view.

export const RULE_VIOLATION_LABELS: Record<string, string> = {
  negative_energy:           'Negative energy reading detected',
  voltage_out_of_range:      'Voltage outside safe operating range',
  invalid_power_factor:      'Power factor outside valid range (0–1)',
  zero_consumption_sequence: 'Extended period of zero consumption',
  energy_spike:              'Sudden energy consumption spike',
  // Add new rule violation labels here
};

// ── LLM Model Options ─────────────────────────────────────────
// Grouped dropdown for the TopBar model selector.
// Adding a model here automatically adds it to the dropdown.

export const LLM_MODEL_GROUPS: LLMModelGroup[] = [
  {
    group: 'Local (Ollama)',
    models: [
      { value: 'llama3.1:8b',  label: 'Llama 3.1 8B',   available: true },
      { value: 'gemma4:latest',    label: 'Gemma 4 9B',      available: true },
      { value: 'mistral:7b',   label: 'Mistral 7B',      available: true },
      { value: 'phi3:mini',    label: 'Phi-3 Mini',      available: true },
    ],
  },
  {
    group: 'Cloud (Future — configure via env vars)',
    models: [
      {
        value: 'azure/gpt-4o',
        label: 'Azure OpenAI GPT-4o',
        available: false,
        note: 'Set LLM_PROVIDER=azure in backend .env',
      },
      {
        value: 'openai/gpt-4o-mini',
        label: 'OpenAI GPT-4o Mini',
        available: false,
        note: 'Set LLM_PROVIDER=openai in backend .env',
      },
      {
        value: 'anthropic/claude-sonnet-4-6',
        label: 'Anthropic Claude Sonnet',
        available: false,
        note: 'Set LLM_PROVIDER=anthropic in backend .env',
      },
    ],
  },
];

// ── Polling Configuration ─────────────────────────────────────
// Controls explanation polling behaviour.
// Change these to adjust timeout without touching hook logic.

export const POLLING_CONFIG = {
  intervalMs:  parseInt(import.meta.env.VITE_POLL_INTERVAL_MS  ?? '3000'),
  maxAttempts: parseInt(import.meta.env.VITE_POLL_MAX_ATTEMPTS ?? '20'),
} as const;

// ── Default OBIS Profile Code ─────────────────────────────────
// Used as the default obisCode in the non-technical form's
// advanced settings.

export const DEFAULT_OBIS_PROFILE = '1.0.99.1.0.255';

// ── Feature OBIS Lookup ───────────────────────────────────────
// Maps canonical feature name → the OBIS code that produces it.
// Used by the payload builder to construct rawValue strings.

export const FEATURE_OBIS_MAP: Record<string, string> = {
  energy_consumption:     '1.0.1.29.0.255',
  active_export_energy:   '1.0.2.29.0.255',
  apparent_import_energy: '1.0.9.29.0.255',
  apparent_export_energy: '1.0.10.29.0.255',
  reactive_import_energy: '1.0.3.29.0.255',
  reactive_export_energy: '1.0.4.29.0.255',
  active_import_power:    '1.0.1.27.0.255',
  active_export_power:    '1.0.2.27.0.255',
  voltage:                '1.0.12.27.0.255',
  current:                '1.0.11.27.0.255',
  power_factor:           '1.0.13.27.0.255',
  frequency:              '1.0.14.27.0.255',
};

// ── Feature Units for rawValue builder ───────────────────────
// Maps canonical feature name → unit string used in pipe-delimited rawValue.

export const FEATURE_UNITS: Record<string, string> = {
  energy_consumption:     'Wh',
  active_export_energy:   'Wh',
  apparent_import_energy: 'VAh',
  apparent_export_energy: 'VAh',
  reactive_import_energy: 'VARh',
  reactive_export_energy: 'VARh',
  active_import_power:    'W',
  active_export_power:    'W',
  voltage:                'V',
  current:                'A',
  power_factor:           '',
  frequency:              'Hz',
};

// ── Detection Thresholds (for ops display reference) ─────────
export const DETECTION_THRESHOLDS = {
  zscore_threshold:               3.0,
  same_hour_deviation_threshold:  0.40,
  voltage_min:                    180.0,
  voltage_max:                    270.0,
  power_factor_min:               0.0,
  power_factor_max:               1.0,
  zero_consumption_window:        3,
  if_contamination:               0.07,
  rolling_window_size:            5,
} as const;
