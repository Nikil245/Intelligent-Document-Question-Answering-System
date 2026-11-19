import {
  Activity,
  FileText,
  LayoutDashboard,
  X,
} from 'lucide-react'
import logoSvg from '../assets/logo.svg'

const NAV_ITEMS = [
  {
    id: 'workspace',
    label: 'QA Workspace',
    description: 'Upload PDF & ask questions',
    icon: LayoutDashboard,
  },
  {
    id: 'document',
    label: 'Document Details',
    description: 'Pages, chunks & metadata',
    icon: FileText,
  },
  {
    id: 'system',
    label: 'Model & System',
    description: 'Health, device & pipeline',
    icon: Activity,
  },
]

export default function Sidebar({
  activeTab,
  onSelectTab,
  health,
  connectionError,
  document,
  isOpen,
  onClose,
}) {
  const isOnline = Boolean(health && !connectionError)
  const isModelReady = Boolean(health?.model_loaded)

  let statusDot = 'bg-rose-500'
  let statusTitle = 'Backend Offline'
  let statusSubtitle = 'Start FastAPI on port 8000'

  if (isOnline && isModelReady) {
    statusDot = 'bg-emerald-500'
    statusTitle = `Online · ${health.device?.toUpperCase() || 'READY'}`
    statusSubtitle = health.device_name || 'DistilBERT QA ready'
  } else if (isOnline && !isModelReady) {
    statusDot = 'bg-amber-500'
    statusTitle = `Model ${health.model_status || 'Degraded'}`
    statusSubtitle = 'Check system diagnostics'
  }

  const content = (
    <div className="flex h-full flex-col bg-white border-r border-slate-200">
      {/* Brand header */}
      <div className="flex items-center justify-between px-5 h-16 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <img src={logoSvg} alt="DocuMind AI logo" className="w-9 h-9 rounded-lg shadow-subtle" />
          <div>
            <span className="block text-[15px] font-semibold tracking-tight text-slate-900">
              DocuMind AI
            </span>
            <span className="block text-[11px] font-medium text-slate-500">
              Document QA Assistant
            </span>
          </div>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="lg:hidden p-1.5 rounded-md text-slate-400 hover:text-slate-700 hover:bg-slate-100"
            aria-label="Close navigation"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-5 space-y-1.5" aria-label="Primary navigation">
        <p className="px-3 mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          Workspace
        </p>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          const isActive = activeTab === item.id
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onSelectTab(item.id)
                onClose?.()
              }}
              className={`w-full flex items-start gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
                isActive
                  ? 'bg-slate-900 text-white shadow-subtle'
                  : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
              }`}
            >
              <Icon
                className={`w-4 h-4 mt-0.5 shrink-0 ${
                  isActive ? 'text-sky-400' : 'text-slate-400'
                }`}
              />
              <div className="min-w-0">
                <div className="text-sm font-medium leading-tight">{item.label}</div>
                <div
                  className={`text-[11px] mt-0.5 truncate ${
                    isActive ? 'text-slate-300' : 'text-slate-400'
                  }`}
                >
                  {item.description}
                </div>
              </div>
            </button>
          )
        })}

        {/* Active document summary pill */}
        <div className="pt-5 mt-5 border-t border-slate-100">
          <p className="px-3 mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Active Document
          </p>
          {document ? (
            <div className="mx-1 p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-sky-600 shrink-0" />
                <span
                  className="text-xs font-medium text-slate-800 truncate"
                  title={document.filename}
                >
                  {document.filename}
                </span>
              </div>
              <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
                <span>{document.num_pages} pages</span>
                <span>{document.num_chunks} chunks</span>
              </div>
            </div>
          ) : (
            <div className="mx-1 p-3 rounded-lg border border-dashed border-slate-200 text-xs text-slate-400">
              No PDF uploaded yet
            </div>
          )}
        </div>
      </nav>

      {/* Footer backend connection status */}
      <div className="p-4 border-t border-slate-200 bg-slate-50/60">
        <div className="flex items-start gap-2.5">
          <span className={`w-2.5 h-2.5 rounded-full mt-1 shrink-0 ${statusDot}`} />
          <div className="min-w-0">
            <p className="text-xs font-semibold text-slate-800 truncate">{statusTitle}</p>
            <p className="text-[11px] text-slate-500 truncate" title={statusSubtitle}>
              {statusSubtitle}
            </p>
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <>
      {/* Desktop fixed sidebar */}
      <aside className="hidden lg:block lg:w-64 lg:shrink-0">{content}</aside>

      {/* Mobile slide-over sidebar */}
      {isOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/40 backdrop-blur-[1px]"
            onClick={onClose}
            aria-hidden="true"
          />
          <div className="relative w-72 max-w-[85vw] h-full z-10 shadow-xl">{content}</div>
        </div>
      )}
    </>
  )
}

