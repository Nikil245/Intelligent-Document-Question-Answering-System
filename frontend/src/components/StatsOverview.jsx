import { Cpu, FileText, Layers, MessageSquare } from 'lucide-react'

export default function StatsOverview({ health, connectionError, document, qaItems }) {
  const isOnline = Boolean(health && !connectionError)
  const isModelReady = Boolean(health?.model_loaded)

  const answeredCount = qaItems.filter((item) => item.response?.success).length
  const totalQuestions = qaItems.length

  const cards = [
    {
      label: 'QA Model Status',
      value: isOnline ? (isModelReady ? 'DistilBERT Ready' : 'Model Degraded') : 'Offline',
      detail: isOnline
        ? 'distilbert-base-cased-distilled-squad'
        : 'Backend API unreachable',
      icon: Layers,
      accent: isOnline && isModelReady
        ? 'text-emerald-600 bg-emerald-50 border-emerald-100'
        : isOnline
          ? 'text-amber-600 bg-amber-50 border-amber-100'
          : 'text-rose-600 bg-rose-50 border-rose-100',
    },
    {
      label: 'Compute Device',
      value: isOnline ? (health.device === 'cuda' ? 'CUDA GPU' : 'CPU') : '—',
      detail: isOnline
        ? health.device_name || `Running on ${health.device?.toUpperCase()}`
        : 'Waiting for connection',
      icon: Cpu,
      accent: 'text-sky-600 bg-sky-50 border-sky-100',
    },
    {
      label: 'Active Document',
      value: document ? `${document.num_pages} Pages` : 'None Uploaded',
      detail: document
        ? `${document.num_chunks} TF-IDF chunks · ${document.filename}`
        : 'Upload a PDF up to 20 MB',
      icon: FileText,
      accent: 'text-indigo-600 bg-indigo-50 border-indigo-100',
    },
    {
      label: 'Questions Asked',
      value: `${totalQuestions}`,
      detail:
        totalQuestions > 0
          ? `${answeredCount} answered with grounded spans`
          : 'Ask multiple questions per PDF',
      icon: MessageSquare,
      accent: 'text-slate-700 bg-slate-100 border-slate-200',
    },
  ]

  return (
    <section aria-label="Dashboard overview" className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3.5">
      {cards.map((card) => {
        const Icon = card.icon
        return (
          <div
            key={card.label}
            className="bg-white rounded-xl border border-slate-200 p-4 shadow-subtle flex items-start justify-between gap-3"
          >
            <div className="min-w-0">
              <p className="text-xs font-medium text-slate-500">{card.label}</p>
              <p className="mt-1 text-lg font-semibold text-slate-900 truncate">{card.value}</p>
              <p className="mt-0.5 text-xs text-slate-500 truncate" title={card.detail}>
                {card.detail}
              </p>
            </div>
            <div className={`p-2 rounded-lg border shrink-0 ${card.accent}`}>
              <Icon className="w-4 h-4" />
            </div>
          </div>
        )
      })}
    </section>
  )
}

