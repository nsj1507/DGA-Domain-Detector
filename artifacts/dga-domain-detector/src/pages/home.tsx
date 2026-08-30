import { useMemo, useState } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  Activity,
  AlertCircle,
  ArrowUpRight,
  Check,
  ChevronDown,
  ChevronUp,
  Clipboard,
  Clock3,
  Database,
  Gauge,
  History,
  Info,
  LockKeyhole,
  Radar,
  RotateCcw,
  Search,
  ShieldCheck,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Terminal,
  Waypoints,
  X,
} from 'lucide-react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import {
  type DomainPrediction,
  type ModelInfo,
  useGetModelInfo,
  useHealthCheck,
  usePredictDomain,
} from '@workspace/api-client-react';
import { Form, FormControl, FormField, FormItem } from '@/components/ui/form';

const domainSchema = z.object({
  domain: z
    .string()
    .trim()
    .min(1, 'Enter a domain to begin.')
    .max(253, 'Domains must be 253 characters or fewer.')
    .refine((value) => !/\s/.test(value), 'Use a domain without spaces.')
    .refine((value) => value.includes('.') && !value.startsWith('.') && !value.endsWith('.'), 'Enter a fully qualified domain, such as example.org.'),
});

type DomainFormValues = z.infer<typeof domainSchema>;

const formatFeatureName = (name: string) =>
  name
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const formatPercent = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`;

const formatNumber = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : value.toFixed(3);

function StatusPill({ status, isLoading }: { status?: string; isLoading: boolean }) {
  const isHealthy = status?.toLowerCase() === 'ok' || status?.toLowerCase() === 'healthy';
  return (
    <div className="flex items-center gap-2 rounded-full border border-[#36515a] bg-[#1d343d] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#dfeade]" data-testid="status-system-health">
      <span className={`relative flex h-2 w-2 rounded-full ${isLoading ? 'bg-[#e2a84b]' : isHealthy ? 'bg-[#b8d95c]' : 'bg-[#db766d]'}`}>
        {!isLoading && isHealthy && <span className="absolute inset-0 animate-ping rounded-full bg-[#b8d95c] opacity-40" />}
      </span>
      {isLoading ? 'Checking system' : isHealthy ? 'System nominal' : 'Model attention'}
    </div>
  );
}

function LoadingBlock({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-[hsl(var(--muted))] ${className}`} aria-label="Loading" />;
}

function ResultPanel({
  result,
  onCopy,
  copied,
}: {
  result: DomainPrediction;
  onCopy: () => void;
  copied: boolean;
}) {
  const [showAllFeatures, setShowAllFeatures] = useState(false);
  const isMalicious = result.prediction === 'Malicious';
  const visibleFeatures = Object.entries(result.features).slice(0, showAllFeatures ? 50 : 8);
  const riskTone = result.risk_level.toLowerCase();

  return (
    <section className="reveal-in overflow-hidden rounded-xl border border-[hsl(var(--card-border))] bg-[hsl(var(--card))] shadow-[var(--shadow-md)]" data-testid="section-analysis-result">
      <div className={`scan-line flex items-start justify-between gap-4 border-b px-5 py-4 sm:px-7 ${isMalicious ? 'border-[#e5b9b3] bg-[#f9eeeb]' : 'border-[#dbe5bb] bg-[#f2f6e3]'}`}>
        <div className="flex items-start gap-3">
          <div className={`mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${isMalicious ? 'bg-[#f5d9d5] text-[#a8453b]' : 'bg-[#dce9b1] text-[#46611c]'}`}>
            {isMalicious ? <ShieldAlert size={21} strokeWidth={1.8} /> : <ShieldCheck size={21} strokeWidth={1.8} />}
          </div>
          <div>
            <div className="mb-1 flex items-center gap-2">
              <span className="font-mono text-[10px] font-medium uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">Verdict</span>
              <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${isMalicious ? 'bg-[#f3d0cb] text-[#9f3e34]' : 'bg-[#d7e8a5] text-[#416019]'}`}>{result.risk_level} risk</span>
            </div>
            <h2 className={`text-xl font-extrabold tracking-[-0.03em] ${isMalicious ? 'text-[#9f3e34]' : 'text-[#46611c]'}`} data-testid="text-prediction">{result.prediction}</h2>
          </div>
        </div>
        <button type="button" onClick={onCopy} className="button-lift inline-flex shrink-0 items-center gap-2 rounded-md border border-current/15 bg-[hsl(var(--card))]/60 px-3 py-2 text-xs font-bold text-[hsl(var(--foreground))]" data-testid="button-copy-result">
          {copied ? <Check size={14} /> : <Clipboard size={14} />}
          {copied ? 'Copied' : 'Copy readout'}
        </button>
      </div>

      <div className="grid gap-6 p-5 sm:p-7 lg:grid-cols-[1fr_220px]">
        <div>
          <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">Analyzed domain</p>
              <p className="break-all font-mono text-lg font-medium text-[hsl(var(--foreground))]" data-testid="text-result-domain">{result.normalized_domain || result.domain}</p>
            </div>
            <div className="text-right">
              <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">Decision score</p>
              <p className="font-mono text-lg font-medium text-[hsl(var(--foreground))]" data-testid="text-decision-score">{formatNumber(result.decision_score)}</p>
            </div>
          </div>

          <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--background))] p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs font-bold text-[hsl(var(--foreground))]">Confidence signal</span>
              <span className="font-mono text-sm font-medium text-[hsl(var(--foreground))]" data-testid="text-confidence">{formatPercent(result.confidence)}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-[hsl(var(--muted))]">
              <div className={`metric-bar h-full rounded-full ${isMalicious ? 'bg-[#c75b50]' : 'bg-[#7e9f32]'}`} style={{ width: `${Math.min(100, Math.max(0, (result.confidence ?? 0) * 100))}%` }} />
            </div>
            <p className="mt-3 flex items-center gap-1.5 text-[11px] leading-relaxed text-[hsl(var(--muted-foreground))]">
              <Info size={13} />
              {result.confidence === null ? 'Confidence was not returned by the active model.' : result.confidence_is_calibrated ? 'Calibrated against held-out evaluation data.' : 'Raw model confidence; calibration is not available.'}
            </p>
          </div>

          {result.reasons.length > 0 && (
            <div className="mt-6">
              <div className="mb-3 flex items-center gap-2">
                <Waypoints size={15} className="text-[hsl(var(--primary))]" />
                <h3 className="text-sm font-extrabold">Signals behind the read</h3>
              </div>
              <ul className="grid gap-2 sm:grid-cols-2" data-testid="list-analysis-reasons">
                {result.reasons.map((reason, index) => (
                  <li key={`${reason}-${index}`} className="flex gap-2 rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--background))] px-3 py-2.5 text-xs leading-relaxed text-[hsl(var(--muted-foreground))]">
                    <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[hsl(var(--accent))]" />
                    <span data-testid={`text-reason-${index}`}>{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <aside className="border-t border-[hsl(var(--border))] pt-5 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0">
          <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">Feature extraction</p>
          <p className="font-mono text-3xl font-medium text-[hsl(var(--foreground))]" data-testid="text-result-feature-count">{result.feature_count}</p>
          <p className="mt-1 text-xs text-[hsl(var(--muted-foreground))]">signals evaluated</p>
          <div className="mt-5 space-y-2.5" data-testid="list-result-features">
            {visibleFeatures.map(([name, value]) => (
              <div key={name} className="flex items-center justify-between gap-3 text-xs">
                <span className="truncate text-[hsl(var(--muted-foreground))]" title={name}>{formatFeatureName(name)}</span>
                <span className="font-mono text-[hsl(var(--foreground))]">{value.toFixed(3)}</span>
              </div>
            ))}
          </div>
          {Object.keys(result.features).length > 8 && (
            <button type="button" onClick={() => setShowAllFeatures((current) => !current)} className="mt-4 inline-flex items-center gap-1 text-xs font-bold text-[hsl(var(--primary))] underline-offset-4 hover:underline" data-testid="button-toggle-features">
              {showAllFeatures ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              {showAllFeatures ? 'Show fewer' : `Show all ${Object.keys(result.features).length} features`}
            </button>
          )}
          <div className="mt-6 border-t border-[hsl(var(--border))] pt-4">
            <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">Active model</p>
            <p className="break-words text-xs font-bold text-[hsl(var(--foreground))]" data-testid="text-result-model">{result.model_name}</p>
          </div>
        </aside>
      </div>
    </section>
  );
}

function ModelSummary({ modelInfo, isLoading, isError }: { modelInfo?: ModelInfo; isLoading: boolean; isError: boolean }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <section className="rounded-xl border border-[hsl(var(--card-border))] bg-[hsl(var(--card))] p-5 shadow-[var(--shadow-sm)] sm:p-6" data-testid="section-model-summary">
      <div className="mb-5 flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[hsl(var(--secondary))] text-[hsl(var(--primary))]"><Database size={17} /></div>
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">Detector profile</p>
            <h2 className="mt-1 text-base font-extrabold">Model summary</h2>
          </div>
        </div>
        <span className="rounded-full border border-[hsl(var(--border))] px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))]">Live</span>
      </div>

      {isLoading ? (
        <div className="space-y-3" data-testid="loading-model-summary"><LoadingBlock className="h-5 w-3/4" /><LoadingBlock className="h-4 w-full" /><LoadingBlock className="h-4 w-5/6" /></div>
      ) : isError || !modelInfo ? (
        <div className="rounded-lg border border-dashed border-[#e5b9b3] bg-[#f9eeeb] p-4" data-testid="state-model-unavailable">
          <div className="flex gap-3">
            <AlertCircle size={17} className="mt-0.5 shrink-0 text-[#a8453b]" />
            <div><p className="text-sm font-bold text-[#8f3f36]">Model information unavailable</p><p className="mt-1 text-xs leading-relaxed text-[#9f625b]">The detector is not ready to accept analysis. Try again when the model service is online.</p></div>
          </div>
        </div>
      ) : (
        <>
          <p className="break-words font-mono text-sm font-medium text-[hsl(var(--foreground))]" data-testid="text-model-name">{modelInfo.model_name}</p>
          <p className="mt-1 text-xs text-[hsl(var(--muted-foreground))]">Combined lexical and character n-gram representation</p>
          <div className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--border))]">
            <div className="bg-[hsl(var(--background))] p-3"><p className="font-mono text-xl" data-testid="text-model-feature-count">{modelInfo.feature_count}</p><p className="mt-1 text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))]">Total features</p></div>
            <div className="bg-[hsl(var(--background))] p-3"><p className="font-mono text-xl" data-testid="text-model-handcrafted-count">{modelInfo.handcrafted_feature_count}</p><p className="mt-1 text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))]">Handcrafted</p></div>
          </div>
          <button type="button" onClick={() => setExpanded((current) => !current)} className="mt-4 flex w-full items-center justify-between border-t border-[hsl(var(--border))] pt-4 text-left text-xs font-bold text-[hsl(var(--primary))]" data-testid="button-toggle-model-details">
            <span className="flex items-center gap-2"><SlidersHorizontal size={14} /> Configuration & metrics</span>
            {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>
          {expanded && (
            <div className="reveal-in mt-4 space-y-4" data-testid="panel-model-details">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded bg-[hsl(var(--muted))] p-2.5"><span className="text-[hsl(var(--muted-foreground))]">Analyzer</span><strong className="mt-1 block font-mono">{modelInfo.vectorizer.analyzer}</strong></div>
                <div className="rounded bg-[hsl(var(--muted))] p-2.5"><span className="text-[hsl(var(--muted-foreground))]">N-gram range</span><strong className="mt-1 block font-mono">{modelInfo.vectorizer.ngram_min}–{modelInfo.vectorizer.ngram_max}</strong></div>
                <div className="rounded bg-[hsl(var(--muted))] p-2.5"><span className="text-[hsl(var(--muted-foreground))]">N-gram features</span><strong className="mt-1 block font-mono">{modelInfo.ngram_feature_count}</strong></div>
                <div className="rounded bg-[hsl(var(--muted))] p-2.5"><span className="text-[hsl(var(--muted-foreground))]">Max features</span><strong className="mt-1 block font-mono">{modelInfo.vectorizer.max_features}</strong></div>
              </div>
              <div>
                <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.14em] text-[hsl(var(--muted-foreground))]">Evaluation set</p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                  {Object.entries(modelInfo.metrics).map(([key, value]) => <div key={key} className="flex justify-between gap-2 border-b border-[hsl(var(--border))] pb-1.5"><span className="text-[hsl(var(--muted-foreground))]">{formatFeatureName(key)}</span><span className="font-mono">{formatPercent(value)}</span></div>)}
                </div>
              </div>
              <div>
                <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.14em] text-[hsl(var(--muted-foreground))]">Handcrafted signals</p>
                <p className="text-xs leading-relaxed text-[hsl(var(--muted-foreground))]" data-testid="text-handcrafted-features">{modelInfo.handcrafted_feature_names.map(formatFeatureName).join(' · ')}</p>
              </div>
              <div>
                <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.14em] text-[hsl(var(--muted-foreground))]">Explainability</p>
                <p className="text-xs leading-relaxed text-[hsl(var(--muted-foreground))]" data-testid="text-explainability-method">{modelInfo.explainability_method}</p>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function HistoryPanel({ history, onClear }: { history: DomainPrediction[]; onClear: () => void }) {
  return (
    <section className="rounded-xl border border-[hsl(var(--card-border))] bg-[hsl(var(--card))] p-5 shadow-[var(--shadow-sm)] sm:p-6" data-testid="section-session-history">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3"><div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[hsl(var(--secondary))] text-[hsl(var(--primary))]"><History size={17} /></div><div><p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">This session</p><h2 className="mt-1 text-base font-extrabold">Analysis history</h2></div></div>
        {history.length > 0 && <button type="button" onClick={onClear} className="button-lift rounded-md px-2 py-1.5 text-xs font-bold text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--muted))] hover:text-[hsl(var(--foreground))]" data-testid="button-clear-history">Clear</button>}
      </div>
      {history.length === 0 ? (
        <div className="flex min-h-[112px] flex-col items-center justify-center rounded-lg border border-dashed border-[hsl(var(--border))] px-4 text-center" data-testid="state-history-empty">
          <Clock3 size={17} className="mb-2 text-[hsl(var(--muted-foreground))]" /><p className="text-xs font-semibold text-[hsl(var(--muted-foreground))]">No domains analyzed yet.</p><p className="mt-1 text-[11px] text-[hsl(var(--muted-foreground))]">Your readouts will stay here until you clear this session.</p>
        </div>
      ) : (
        <div className="space-y-1.5" data-testid="list-session-history">
          {history.map((item, index) => <div key={`${item.normalized_domain}-${index}`} className="flex items-center justify-between gap-3 rounded-lg border border-transparent px-2 py-2.5 hover:border-[hsl(var(--border))] hover:bg-[hsl(var(--background))]" data-testid={`row-history-${index}`}><div className="min-w-0"><p className="truncate font-mono text-xs font-medium" data-testid={`text-history-domain-${index}`}>{item.normalized_domain || item.domain}</p><p className="mt-1 text-[10px] text-[hsl(var(--muted-foreground))]">{item.model_name}</p></div><span className={`shrink-0 rounded px-2 py-1 font-mono text-[10px] uppercase tracking-wider ${item.prediction === 'Malicious' ? 'bg-[#f3d0cb] text-[#9f3e34]' : 'bg-[#dce9b1] text-[#46611c]'}`} data-testid={`text-history-prediction-${index}`}>{item.prediction}</span></div>)}
        </div>
      )}
    </section>
  );
}

export default function Home() {
  const [result, setResult] = useState<DomainPrediction | undefined>();
  const [history, setHistory] = useState<DomainPrediction[]>([]);
  const [copied, setCopied] = useState(false);
  const form = useForm<DomainFormValues>({ resolver: zodResolver(domainSchema), defaultValues: { domain: '' }, mode: 'onSubmit' });
  const health = useHealthCheck();
  const model = useGetModelInfo();
  const prediction = usePredictDomain();
  const modelReady = Boolean(model.data && !model.isError);
  const systemIssue = Boolean(health.isError || model.isError || (health.data && !['ok', 'healthy'].includes(health.data.status.toLowerCase())));
  const buttonLabel = prediction.isPending ? 'Extracting signals' : 'Run analysis';
  const activeState = useMemo(() => {
    if (prediction.isPending) return 'running';
    if (result) return 'complete';
    return 'idle';
  }, [prediction.isPending, result]);

  const onSubmit = (values: DomainFormValues) => {
    setCopied(false);
    prediction.mutate({ data: { domain: values.domain.trim() } }, {
      onSuccess: (data) => {
        setResult(data);
        setHistory((current) => [data, ...current.filter((entry) => entry.normalized_domain !== data.normalized_domain)].slice(0, 10));
      },
    });
  };

  const resetWorkspace = () => {
    form.reset();
    setResult(undefined);
    prediction.reset();
    setCopied(false);
  };

  const copyResult = async () => {
    if (!result) return;
    await navigator.clipboard?.writeText(`${result.normalized_domain || result.domain} — ${result.prediction} (${result.risk_level} risk)`);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="instrument-shell relative min-h-[100dvh] overflow-hidden text-[hsl(var(--foreground))]">
      <header className="relative z-10 border-b border-[#36515a] bg-[hsl(var(--sidebar))] text-[hsl(var(--sidebar-foreground))]">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-4 px-5 py-4 sm:px-8 lg:px-12">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[hsl(var(--accent))] text-[hsl(var(--accent-foreground))]"><Radar size={20} strokeWidth={2.2} /></div>
            <div><div className="flex items-center gap-2"><span className="font-mono text-sm font-medium tracking-tight">DGA / DETECTOR</span><span className="hidden rounded border border-[#49646b] px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[#afc0bb] sm:inline">Analyst workspace</span></div><p className="mt-0.5 hidden text-[11px] text-[#a9bab4] sm:block">Domain intelligence for the uncertain edge</p></div>
          </div>
          <StatusPill status={health.data?.status} isLoading={health.isLoading} />
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-[1440px] px-5 pb-14 pt-8 sm:px-8 sm:pt-10 lg:px-12">
        <div className="mb-8 flex flex-col justify-between gap-5 border-b border-[hsl(var(--border))] pb-7 sm:flex-row sm:items-end">
          <div className="reveal-in max-w-2xl">
            <div className="mb-3 flex items-center gap-2 font-mono text-[10px] font-medium uppercase tracking-[0.19em] text-[hsl(var(--muted-foreground))]"><span className="h-1.5 w-1.5 rounded-full bg-[hsl(var(--accent))]" />Signal inspection / 01</div>
            <h1 className="max-w-2xl text-3xl font-extrabold leading-[1.05] tracking-[-0.055em] sm:text-5xl">Is this domain <span className="text-[#738f31]">machine-made?</span></h1>
            <p className="mt-4 max-w-xl text-sm leading-relaxed text-[hsl(var(--muted-foreground))] sm:text-base">Run a transparent lexical analysis against the active DGA detector. Start with the domain; the model handles the rest.</p>
          </div>
          <div className="reveal-in-delay flex items-center gap-2 self-start rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))]/70 px-3 py-2 sm:self-end" data-testid="status-analysis-mode"><Activity size={15} className={activeState === 'running' ? 'text-[#c28b2a]' : 'text-[#738f31]'} /><div><p className="font-mono text-[9px] uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">Instrument</p><p className="text-xs font-bold">{activeState === 'running' ? 'Processing input' : activeState === 'complete' ? 'Readout ready' : 'Awaiting input'}</p></div></div>
        </div>

        <div className="grid items-start gap-7 lg:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
          <div className="space-y-7">
            <section className="reveal-in rounded-xl border border-[hsl(var(--card-border))] bg-[hsl(var(--card))] p-5 shadow-[var(--shadow-sm)] sm:p-7" data-testid="section-domain-input">
              <div className="mb-6 flex items-start justify-between gap-4">
                <div><p className="mb-1 font-mono text-[10px] uppercase tracking-[0.16em] text-[hsl(var(--muted-foreground))]">Input channel</p><h2 className="text-xl font-extrabold tracking-[-0.03em]">Inspect a domain</h2></div>
                <div className="hidden rounded-md bg-[hsl(var(--muted))] p-2 text-[hsl(var(--muted-foreground))] sm:block"><LockKeyhole size={17} /></div>
              </div>
              <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4" data-testid="form-domain-analysis">
                  <FormField control={form.control} name="domain" render={({ field }) => (
                    <FormItem>
                      <div className="relative">
                        <FormControl>
                          <input {...field} type="text" autoComplete="off" spellCheck={false} placeholder="domain.example" className="h-14 w-full rounded-lg border border-[hsl(var(--input))] bg-[hsl(var(--background))] px-4 pr-12 font-mono text-base outline-none transition-colors placeholder:text-[hsl(var(--muted-foreground))] focus:border-[hsl(var(--ring))] focus:ring-2 focus:ring-[hsl(var(--ring))]/15" data-testid="input-domain" />
                        </FormControl>
                        <Search className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-[hsl(var(--muted-foreground))]" size={19} />
                      </div>
                      {form.formState.errors.domain && <p className="flex items-center gap-1.5 px-1 text-xs font-medium text-[#a8453b]" data-testid="status-invalid-input"><AlertCircle size={13} />{form.formState.errors.domain.message}</p>}
                    </FormItem>
                  )} />
                  {prediction.isError && <div className="flex items-start gap-2 rounded-lg border border-[#e5b9b3] bg-[#f9eeeb] px-3 py-2.5 text-xs text-[#93483f]" data-testid="status-analysis-error"><X size={15} className="mt-0.5 shrink-0" /><span>Analysis could not be completed. Check the domain and try again.</span></div>}
                  {systemIssue && <div className="flex items-start gap-2 rounded-lg border border-[#e6d7ae] bg-[#fbf6e6] px-3 py-2.5 text-xs text-[#866526]" data-testid="status-model-unavailable"><AlertCircle size={15} className="mt-0.5 shrink-0" /><span>The model service is currently unavailable. Analysis is paused until it reports healthy.</span></div>}
                  <div className="flex flex-col gap-3 pt-1 sm:flex-row">
                    <button type="submit" disabled={prediction.isPending || !modelReady || systemIssue} className="button-lift inline-flex h-11 flex-1 items-center justify-center gap-2 rounded-lg bg-[hsl(var(--primary))] px-5 text-sm font-bold text-[hsl(var(--primary-foreground))] shadow-sm disabled:cursor-not-allowed disabled:opacity-50" data-testid="button-submit-analysis"><Sparkles size={16} />{buttonLabel}<ArrowUpRight size={16} /></button>
                    <button type="button" onClick={resetWorkspace} className="button-lift inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-[hsl(var(--border))] px-5 text-sm font-bold text-[hsl(var(--foreground))] hover:bg-[hsl(var(--muted))]" data-testid="button-reset-analysis"><RotateCcw size={15} />Reset</button>
                  </div>
                  <p className="flex items-center gap-1.5 pt-1 text-[11px] text-[hsl(var(--muted-foreground))]"><Terminal size={13} />No DNS lookup or external enrichment. This read uses the supplied domain string only.</p>
                </form>
              </Form>
            </section>
            {result ? <ResultPanel result={result} onCopy={copyResult} copied={copied} /> : (
              <section className="reveal-in-delay flex min-h-[280px] flex-col items-center justify-center rounded-xl border border-dashed border-[hsl(var(--border))] bg-[hsl(var(--card))]/45 px-6 py-10 text-center" data-testid="state-analysis-empty">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-[hsl(var(--border))] bg-[hsl(var(--card))] text-[hsl(var(--muted-foreground))]"><Gauge size={22} /></div>
                <h2 className="text-lg font-extrabold tracking-[-0.02em]">Your readout will appear here</h2><p className="mt-2 max-w-sm text-xs leading-relaxed text-[hsl(var(--muted-foreground))]">Submit a domain to see the verdict, model confidence, contributing signals, and extracted feature values.</p>
              </section>
            )}
          </div>
          <div className="space-y-7">
            <ModelSummary modelInfo={model.data} isLoading={model.isLoading} isError={model.isError} />
            <HistoryPanel history={history} onClear={() => setHistory([])} />
          </div>
        </div>
      </main>
      <footer className="relative z-10 mx-auto flex max-w-[1440px] flex-col gap-2 border-t border-[hsl(var(--border))] px-5 py-5 font-mono text-[10px] uppercase tracking-[0.14em] text-[hsl(var(--muted-foreground))] sm:flex-row sm:items-center sm:justify-between sm:px-8 lg:px-12"><span data-testid="text-footer-label">DGA Domain Detector / analyst build</span><span data-testid="text-footer-feature-note">Lexical features · character n-grams · model score</span></footer>
    </div>
  );
}