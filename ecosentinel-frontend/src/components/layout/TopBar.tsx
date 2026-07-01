// ============================================================
// components/layout/TopBar.tsx
// Top bar: view toggle, decision engine toggle, LLM selector,
// API health indicator.
// ============================================================

import { useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import ViewModeToggle from '@/components/shared/ViewModeToggle';
import { HealthDot } from '@/components/shared/StatusBadge';
import { useAppStore } from '@/store/appStore';
import { getHealth } from '@/api/api';
import type { HealthResponse } from '@/types';
import { cn } from '@/lib/utils';

export default function TopBar() {
  const decisionEngineEnabled    = useAppStore((s) => s.decisionEngineEnabled);
  const setDecisionEngineEnabled = useAppStore((s) => s.setDecisionEngineEnabled);

  const [health, setHealth]           = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  // Health polling every 30 seconds
  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const h = await getHealth();
        if (mounted) { setHealth(h); setHealthError(false); }
      } catch {
        if (mounted) setHealthError(true);
      }
    };
    check();
    const interval = setInterval(check, 30_000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  const healthStatus = healthError
    ? 'offline'
    : health?.status === 'ok'
      ? 'ok'
      : health?.status === 'degraded'
        ? 'degraded'
        : 'offline';

  const healthLabel = healthError ? 'API Offline' : health ? `API ${health.status}` : 'Connecting…';

  return (
    <header className="
      h-11 flex items-center gap-3 px-4
      bg-surface-card border-b border-surface-border
      shrink-0
    ">
      {/* Health indicator */}
      <HealthDot
        status={healthStatus}
        label={healthLabel}
        className="mr-1"
      />

      <div className="w-px h-4 bg-surface-border" />

      {/* View mode toggle */}
      <ViewModeToggle />

      <div className="w-px h-4 bg-surface-border" />

      {/* Decision Engine toggle */}
      <label className="flex items-center gap-2 cursor-pointer select-none">
        <span className="text-2xs font-mono text-text-secondary">Decision Engine</span>
        <button
          role="switch"
          aria-checked={decisionEngineEnabled}
          onClick={() => setDecisionEngineEnabled(!decisionEngineEnabled)}
          className={cn(
            'relative inline-flex w-8 h-4 rounded-full transition-colors duration-200 shrink-0',
            decisionEngineEnabled ? 'bg-brand' : 'bg-surface-border',
          )}
        >
          <span
            className={cn(
              'absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm transition-transform duration-200',
              decisionEngineEnabled ? 'translate-x-4' : 'translate-x-0.5',
            )}
          />
        </button>
        <span className={cn(
          'text-2xs font-mono font-semibold',
          decisionEngineEnabled ? 'text-brand-dark' : 'text-text-muted',
        )}>
          {decisionEngineEnabled ? 'ON' : 'OFF'}
        </span>
      </label>

      <div className="w-px h-4 bg-surface-border" />

      {/* LLM Model display — read-only, sourced from backend /health */}
      <div
        title="LLM model is set via LLM_MODEL in the backend .env file"
        className={cn(
          'flex items-center gap-1.5 px-2.5 py-1 rounded-sm border text-2xs font-mono cursor-default',
          'border-surface-border bg-surface-raised text-text-secondary',
        )}
      >
        <span className="text-text-muted">Model:</span>
        <span className="text-brand-dark max-w-[140px] truncate">
          {health?.llm_model ?? '—'}
        </span>
        {health?.llm_provider && (
          <span className="text-text-muted">via {health.llm_provider}</span>
        )}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Reload health */}
      <button
        onClick={() => getHealth().then(setHealth).catch(() => setHealthError(true))}
        title="Refresh health status"
        className="p-1 rounded-sm text-text-muted hover:text-text-primary hover:bg-surface-hover transition-colors"
      >
        <RefreshCw size={11} />
      </button>
    </header>
  );
}
