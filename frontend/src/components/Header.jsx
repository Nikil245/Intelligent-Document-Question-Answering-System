import { Cpu, Menu, RefreshCw } from 'lucide-react'

const VIEW_TITLES = {
  workspace: {
    title: 'Document Question Answering',
    subtitle: 'Upload a PDF, retrieve relevant passages with TF-IDF, and extract answers with DistilBERT.',
  },
  document: {
    title: 'Document Management & Details',
    subtitle: 'Inspect page counts, chunk indexing metadata, or delete the uploaded document.',
  },
  system: {
    title: 'Backend & Model Diagnostics',
    subtitle: 'Live health status, inference hardware, and NLP pipeline architecture.',
  },
}

export default function Header({
  activeTab,
  health,
  isCheckingHealth,
  connectionError,
  onRefreshHealth,
  onOpenSidebar,
}) {
  const heading = VIEW_TITLES[activeTab] || VIEW_TITLES.workspace
  const isOnline = Boolean(health && !connectionError)
  const isModelReady = Boolean(health?.model_loaded)

  let badgeClasses = 'bg-rose-50 text-rose-700 border-rose-200'
  let badgeLabel = 'API Offline'
  if (isOnline && isModelReady) {
    badgeClasses = 'bg-emerald-50 text-emerald-700 border-emerald-200'
    badgeLabel = 'Model Ready'
  } else if (isOnline) {
    badgeClasses = 'bg-amber-50 text-amber-700 border-amber-200'
    badgeLabel = `Model ${health?.model_status || 'Degraded'}`
  }

  return (
    <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200 px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onOpenSidebar}
          className="lg:hidden p-2 -ml-2 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100"
          aria-label="Open menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="min-w-0">
          <h1 className="text-base sm:text-lg font-semibold text-slate-900 truncate">
            {heading.title}
          </h1>
          <p className="hidden md:block text-xs text-slate-500 truncate">{heading.subtitle}</p>
        </div>
      </div>

      <div className="flex items-center gap-2.5 shrink-0">
        {isOnline && health.model_name && (
          <div
            className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 border border-slate-200 text-xs font-mono text-slate-700"
            title={health.model_name}
          >
            <span>{health.model_name}</span>
          </div>
        )}

        {isOnline && (
          <div
            className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-xs font-medium text-slate-700"
            title={health.device_name || health.device}
          >
            <Cpu className="w-3.5 h-3.5 text-slate-500" />
            <span className="uppercase">{health.device}</span>
          </div>
        )}

        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium ${badgeClasses}`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isOnline && isModelReady
                ? 'bg-emerald-500'
                : isOnline
                  ? 'bg-amber-500'
                  : 'bg-rose-500'
            }`}
          />
          {badgeLabel}
        </span>

        <button
          type="button"
          onClick={() => onRefreshHealth()}
          disabled={isCheckingHealth}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-slate-200 bg-white text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-60 transition-colors"
          title="Refresh backend connection status"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isCheckingHealth ? 'animate-spin' : ''}`} />
          <span className="hidden sm:inline">Check</span>
        </button>
      </div>
    </header>
  )
}

