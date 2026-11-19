import { useState } from 'react'
import {
  AlertTriangle,
  Check,
  Copy,
  FileText,
  Loader2,
  RefreshCw,
  Trash2,
} from 'lucide-react'
import { formatBytes, formatDateTime } from '../utils/formatters'

export default function DocumentCard({
  document,
  documentExpired,
  isRefreshing,
  isDeleting,
  docActionError,
  docActionNotice,
  onRefresh,
  onDelete,
  onClearNotice,
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [copiedId, setCopiedId] = useState(false)

  if (!document) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 shadow-card p-5">
        <h2 className="text-sm font-semibold text-slate-900">Document Information</h2>
        {docActionNotice ? (
          <div className="mt-3 rounded-lg bg-slate-50 border border-slate-200 px-3 py-2.5 text-xs text-slate-700 flex items-center justify-between gap-2">
            <span>{docActionNotice}</span>
            <button
              type="button"
              onClick={onClearNotice}
              className="text-slate-500 hover:text-slate-800 font-medium shrink-0"
            >
              Dismiss
            </button>
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-500 leading-relaxed">
            No document is currently loaded. Upload a PDF above to inspect its page count, chunk breakdown, and ask questions.
          </p>
        )}
      </div>
    )
  }

  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(document.document_id)
      setCopiedId(true)
      window.setTimeout(() => setCopiedId(false), 1800)
    } catch {
      // Clipboard API may be unavailable in restricted contexts.
    }
  }

  const handleConfirmDelete = async () => {
    const deleted = await onDelete()
    if (deleted) {
      setConfirmingDelete(false)
    }
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-card p-5 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0">
          <div className="p-2.5 rounded-lg bg-sky-50 border border-sky-100 text-sky-600 shrink-0">
            <FileText className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-slate-900 truncate" title={document.filename}>
              {document.filename}
            </h2>
            <div className="mt-1 flex items-center gap-1.5">
              <span className="font-mono text-[11px] text-slate-500 truncate max-w-[165px]">
                {document.document_id}
              </span>
              <button
                type="button"
                onClick={copyId}
                className="p-0.5 rounded text-slate-400 hover:text-slate-700"
                title="Copy Document UUID"
                aria-label="Copy Document UUID"
              >
                {copiedId ? (
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </div>
        </div>

        <span
          className={`px-2 py-0.5 rounded-full text-[11px] font-medium border shrink-0 capitalize ${
            documentExpired
              ? 'bg-rose-50 text-rose-700 border-rose-200'
              : 'bg-emerald-50 text-emerald-700 border-emerald-200'
          }`}
        >
          {documentExpired ? 'expired' : document.status}
        </span>
      </div>

      {/* Key metrics grid */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="rounded-lg bg-slate-50 border border-slate-100 p-2.5">
          <p className="text-[11px] text-slate-500">Total Pages</p>
          <p className="text-sm font-semibold text-slate-900 mt-0.5">
            {document.num_pages}
            <span className="text-xs font-normal text-slate-500 ml-1">
              ({document.pages_with_text} with text)
            </span>
          </p>
        </div>

        <div className="rounded-lg bg-slate-50 border border-slate-100 p-2.5">
          <p className="text-[11px] text-slate-500">Text Chunks</p>
          <p className="text-sm font-semibold text-slate-900 mt-0.5">{document.num_chunks}</p>
        </div>

        <div className="rounded-lg bg-slate-50 border border-slate-100 p-2.5">
          <p className="text-[11px] text-slate-500">File Size</p>
          <p className="text-sm font-semibold text-slate-900 mt-0.5">
            {formatBytes(document.file_size_bytes)}
          </p>
        </div>

        <div className="rounded-lg bg-slate-50 border border-slate-100 p-2.5">
          <p className="text-[11px] text-slate-500">Extracted Text</p>
          <p className="text-sm font-semibold text-slate-900 mt-0.5">
            {document.total_characters !== null && document.total_characters !== undefined
              ? `${document.total_characters.toLocaleString()} chars`
              : '—'}
          </p>
        </div>
      </div>

      <div className="text-[11px] text-slate-500 flex items-center justify-between pt-1 border-t border-slate-100">
        <span>Uploaded</span>
        <span>{formatDateTime(document.uploaded_at)}</span>
      </div>

      {/* Empty pages warning */}
      {Array.isArray(document.empty_pages) && document.empty_pages.length > 0 && (
        <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-xs text-amber-800 flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
          <span>
            Pages with no extractable text skipped: {document.empty_pages.join(', ')}.
          </span>
        </div>
      )}

      {/* Notices / Errors */}
      {docActionError && (
        <div className="rounded-lg bg-rose-50 border border-rose-200 px-3 py-2 text-xs text-rose-800">
          {docActionError.message}
        </div>
      )}

      {docActionNotice && !docActionError && (
        <div className="rounded-lg bg-slate-50 border border-slate-200 px-3 py-2 text-xs text-slate-700 flex items-center justify-between gap-2">
          <span>{docActionNotice}</span>
          <button
            type="button"
            onClick={onClearNotice}
            className="text-slate-500 hover:text-slate-800 font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Actions */}
      <div className="pt-1 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onRefresh}
          disabled={isRefreshing || isDeleting}
          className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-slate-200 bg-white text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          Refresh Details
        </button>

        {!confirmingDelete ? (
          <button
            type="button"
            onClick={() => setConfirmingDelete(true)}
            disabled={isDeleting || isRefreshing}
            className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-rose-200 bg-rose-50/70 text-xs font-medium text-rose-700 hover:bg-rose-100 disabled:opacity-50 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Delete PDF
          </button>
        ) : (
          <div className="w-full flex items-center justify-between gap-2 rounded-lg bg-rose-50 border border-rose-200 p-2">
            <span className="text-xs font-medium text-rose-900">Remove document & file?</span>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setConfirmingDelete(false)}
                disabled={isDeleting}
                className="px-2.5 py-1 rounded text-xs font-medium text-slate-600 hover:bg-white"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmDelete}
                disabled={isDeleting}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-rose-600 text-white text-xs font-medium hover:bg-rose-700 disabled:opacity-50"
              >
                {isDeleting && <Loader2 className="w-3 h-3 animate-spin" />}
                Confirm
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

