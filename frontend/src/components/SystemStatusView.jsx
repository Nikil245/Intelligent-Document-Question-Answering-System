import { Activity, AlertCircle, CheckCircle2, Cpu, RefreshCw, Server } from 'lucide-react'
import { getApiBaseUrl } from '../services/api'
import { formatDateTime } from '../utils/formatters'

export default function SystemStatusView({
  health,
  isCheckingHealth,
  connectionError,
  lastCheckedAt,
  onRefreshHealth,
}) {
  const apiBaseUrl = getApiBaseUrl()
  const isOnline = Boolean(health && !connectionError)
  const isReady = Boolean(health?.model_loaded)

  return (
    <div className="space-y-5">
      {/* Connection banner */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-card p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <div
              className={`p-2.5 rounded-xl border ${
                isOnline && isReady
                  ? 'bg-emerald-50 border-emerald-200 text-emerald-600'
                  : isOnline
                    ? 'bg-amber-50 border-amber-200 text-amber-600'
                    : 'bg-rose-50 border-rose-200 text-rose-600'
              }`}
            >
              {isOnline && isReady ? (
                <CheckCircle2 className="w-5 h-5" />
              ) : (
                <AlertCircle className="w-5 h-5" />
              )}
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                {isOnline
                  ? isReady
                    ? 'FastAPI Backend Connected & Model Ready'
                    : 'Backend Connected — Model Degraded'
                  : 'Cannot Connect to Backend Server'}
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Endpoint: <span className="font-mono text-slate-700">{apiBaseUrl}/api/health</span>
                {lastCheckedAt && ` · Last checked ${formatDateTime(lastCheckedAt.toISOString())}`}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => onRefreshHealth()}
            disabled={isCheckingHealth}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 disabled:opacity-50 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isCheckingHealth ? 'animate-spin' : ''}`} />
            Refresh Status
          </button>
        </div>

        {connectionError && (
          <div className="mt-4 rounded-lg bg-rose-50 border border-rose-200 p-3.5 text-xs text-rose-800">
            <p className="font-semibold text-rose-900">Connection Error</p>
            <p className="mt-1">{connectionError.message}</p>
          </div>
        )}

        {health?.model_error && (
          <div className="mt-4 rounded-lg bg-amber-50 border border-amber-200 p-3.5 text-xs text-amber-900">
            <p className="font-semibold">Model Error</p>
            <p className="mt-1">{health.model_error}</p>
          </div>
        )}

        {health && (
          <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
            <div className="rounded-lg bg-slate-50 border border-slate-200/80 p-3.5">
              <div className="flex items-center gap-1.5 text-xs text-slate-500">
                <Activity className="w-3.5 h-3.5" />
                <span>Backend Status</span>
              </div>
              <p className="mt-1 text-sm font-semibold text-slate-900 uppercase">{health.status}</p>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Model status: {health.model_status}
              </p>
            </div>

            <div className="rounded-lg bg-slate-50 border border-slate-200/80 p-3.5">
              <div className="flex items-center gap-1.5 text-xs text-slate-500">
                <Server className="w-3.5 h-3.5" />
                <span>Pretrained Model</span>
              </div>
              <p className="mt-1 text-xs font-mono font-medium text-slate-900 break-all">
                {health.model_name}
              </p>
            </div>

            <div className="rounded-lg bg-slate-50 border border-slate-200/80 p-3.5">
              <div className="flex items-center gap-1.5 text-xs text-slate-500">
                <Cpu className="w-3.5 h-3.5" />
                <span>Inference Device</span>
              </div>
              <p className="mt-1 text-sm font-semibold text-slate-900 uppercase">{health.device}</p>
              <p className="text-[11px] text-slate-500 mt-0.5 truncate" title={health.device_name || ''}>
                {health.device_name || 'Standard CPU execution'}
              </p>
            </div>

            <div className="rounded-lg bg-slate-50 border border-slate-200/80 p-3.5">
              <p className="text-xs text-slate-500">Documents in Memory</p>
              <p className="mt-1 text-sm font-semibold text-slate-900">
                {health.documents_in_memory}
              </p>
              <p className="text-[11px] text-slate-500 mt-0.5">In-memory UUID registry</p>
            </div>
          </div>
        )}
      </div>

      {/* NLP Architecture reference */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-card p-6">
        <h3 className="text-sm font-semibold text-slate-900">
          NLP Processing Pipeline
        </h3>
        <p className="text-xs text-slate-500 mt-1">
          Extractive question answering powered by TF-IDF retrieval and DistilBERT (not generative text synthesis).
        </p>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="rounded-lg border border-slate-200 p-4">
            <span className="text-[11px] font-semibold text-sky-700 bg-sky-50 px-2 py-0.5 rounded">
              1. PDF Extraction & Chunking
            </span>
            <p className="mt-2 text-xs text-slate-600 leading-relaxed">
              PyMuPDF extracts text page by page while preserving 1-based page numbers. Long pages are split into overlapping 384-token windows using the DistilBERT fast tokenizer.
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 p-4">
            <span className="text-[11px] font-semibold text-sky-700 bg-sky-50 px-2 py-0.5 rounded">
              2. TF-IDF + Cosine Retrieval
            </span>
            <p className="mt-2 text-xs text-slate-600 leading-relaxed">
              Scikit-learn vectorizes chunks and the user question into unigram/bigram TF-IDF vectors and selects the top matching passages by cosine similarity.
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 p-4">
            <span className="text-[11px] font-semibold text-sky-700 bg-sky-50 px-2 py-0.5 rounded">
              3. Extractive DistilBERT QA
            </span>
            <p className="mt-2 text-xs text-slate-600 leading-relaxed">
              Each retrieved chunk is evaluated by <span className="font-mono">distilbert-base-cased-distilled-squad</span> to extract the highest-scoring verbatim answer span and page reference.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

