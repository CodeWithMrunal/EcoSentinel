// ============================================================
// types/index.ts
// All TypeScript interfaces for EcoSentinel.
// Mirrors backend Pydantic schemas in api/schemas.py and
// decision_engine/schemas.py exactly.
// ============================================================

// ── Detection — Request ──────────────────────────────────────

export interface MeterRecord {
  id: number;
  meterSerial: string;
  timestamp: string;          // ISO 8601
  obisCode: string;
  entryId: number;
  rawValue: string;
}

export interface DetectRequest {
  records: MeterRecord[];
}

// ── Detection — Response ─────────────────────────────────────

export interface RuleLayerResult {
  layer: string;
  is_anomaly: boolean;
  violations: string[];
  details: Record<string, unknown>;
}

export interface ZScoreLayerResult {
  layer: string;
  is_anomaly: boolean;
  z_score: number | null;
  spike_ratio: number | null;
  triggers: string[];
  details: Record<string, unknown>;
}

export interface IFLayerResult {
  layer: string;
  is_anomaly: boolean;
  anomaly_score: number | null;
  prediction: number | null;
  model_used: string | null;
  features_used: string[] | null;
}

export interface DetectionLayers {
  rule_based: RuleLayerResult;
  zscore: ZScoreLayerResult;
  isolation_forest: IFLayerResult;
}

export interface DetectResponse {
  meter_serial: string;
  interval_timestamp: string;
  is_anomaly: boolean;
  layers: DetectionLayers | null;
  features: Record<string, number | null> | null;
  error: string | null;
  anomaly_id: number | null;
  explanation_status: 'pending' | 'completed' | 'failed' | null;
}

export interface DetectBatchResponse {
  total: number;
  anomalies: number;
  results: DetectResponse[];
}

// ── Decision Engine — Explanation ────────────────────────────

export interface AnomalyExplanation {
  anomaly_explanation: string;
  supporting_factors: string[];
  possible_false_positive_scenarios: string[];
  confidence: 'High' | 'Medium' | 'Low';
  limitations: string | null;
  llm_provider: string | null;
  llm_model: string | null;
}

export interface AnomalyExplanationResponse {
  anomaly_id: number;
  meter_serial: string;
  interval_timestamp: string;
  explanation_status: 'pending' | 'completed' | 'failed' | null;
  explanation: AnomalyExplanation | null;
  explanation_generated_at: string | null;
  explanation_error: string | null;
  rule_violations: string[] | null;
  zscore_value: number | null;
  if_score: number | null;
}

// ── Ops ───────────────────────────────────────────────────────

export interface HealthComponents {
  model_artifacts: 'ok' | 'missing';
  database: 'ok' | 'unavailable' | 'not_configured';
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  timestamp: string;
  components: HealthComponents;
  llm_model?: string;
  llm_provider?: string;
}

export interface ModelInfoResponse {
  feature_schema: string[];
  detection_config: Record<string, number | boolean>;
  rolling_window: number;
  artifact_paths: Record<string, string>;
}

export interface ModelReloadResponse {
  status: string;
  timestamp: string;
  artifacts: Record<string, string>;
}

// ── Non-technical display shapes (produced by adapters.ts) ───

export interface NonTechDetectionDisplay {
  meterSerial: string;
  timestamp: string;              // human-formatted
  isAnomaly: boolean;
  anomalyId: number | null;
  explanationStatus: DetectResponse['explanation_status'];
  layers: {
    label: string;
    key: string;
    fired: boolean;
  }[];
  ruleViolations: string[];       // human-readable violation labels
  zscoreLabel: string | null;     // e.g. "3.2× statistical average"
  ifSeverity: number | null;      // normalised 0–1 for severity bar
  ifSeverityLabel: string | null; // e.g. "High anomaly score (−0.312)"
  error: string | null;
}

export interface NonTechExplanationDisplay {
  anomalyId: number;
  meterSerial: string;
  timestamp: string;
  status: AnomalyExplanationResponse['explanation_status'];
  explanation: string | null;
  supportingFactors: string[];
  falsePositiveScenarios: string[];
  confidence: 'High' | 'Medium' | 'Low' | null;
  limitations: string | null;
  modelAttribution: string | null;   // e.g. "llama3.1:8b via ollama"
  generatedAt: string | null;
  error: string | null;
}

// ── UI State ──────────────────────────────────────────────────

export type ViewMode = 'technical' | 'non-technical';

export type ChecklistStatus = 'waiting' | 'running' | 'done' | 'error' | 'skipped';

export interface ChecklistStep {
  id: string;
  label: string;
  status: ChecklistStatus;
}

// Non-technical form state for building DetectRequest
export interface MeterFormState {
  meterSerial: string;
  selectedGroup: string;             // e.g. "group_A"
  fieldValues: Record<string, string>; // canonical_name → raw string value
  // Advanced overrides (optional)
  advanced: {
    id: string;
    entryId: string;
    obisCode: string;
    timestamp: string;
  };
}

// Capability group definition (mirrors settings.py CAPABILITY_GROUPS)
export interface CapabilityGroupDef {
  label: string;
  features: string[];
  description: string;
}

// LLM model option for the TopBar dropdown
export interface LLMModelOption {
  value: string;
  label: string;
  available: boolean;    // false = cloud/future provider, grayed out
  note?: string;         // e.g. "Configure via env vars"
}

export interface LLMModelGroup {
  group: string;
  models: LLMModelOption[];
}
