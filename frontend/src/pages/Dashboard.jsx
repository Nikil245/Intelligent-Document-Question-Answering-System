import { useState } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'
import ChatInterface from '../components/ChatInterface'
import DocumentCard from '../components/DocumentCard'
import DocumentUpload from '../components/DocumentUpload'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import SourcePanel from '../components/SourcePanel'
import StatsOverview from '../components/StatsOverview'
import SystemStatusView from '../components/SystemStatusView'
import { useBackendHealth } from '../hooks/useBackendHealth'
import { useDocumentQA } from '../hooks/useDocumentQA'
import { formatBytes, formatDateTime } from '../utils/formatters'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('workspace')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const {
    health,
    isChecking: isCheckingHealth,
    connectionError,
    lastCheckedAt,
    refreshHealth,
  } = useBackendHealth()

  const qa = useDocumentQA({
    onDocumentsChanged: () => refreshHealth({ silent: true }),
  })

  const isBackendReady = Boolean(health && !connectionError && health.model_loaded)

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900">
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        health={health}
        connectionError={connectionError}
        document={qa.document}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex-1 min-w-0 flex flex-col">
        <Header
          activeTab={activeTab}
          health={health}
          isCheckingHealth={isCheckingHealth}
          connectionError={connectionError}
          onRefreshHealth={() => refreshHealth()}
          onOpenSidebar={() => setSidebarOpen(true)}
        />

        <main className="flex-1 p-4 sm:p-6 max-w-[1600px] w-full mx-auto space-y-5">
          {/* Global offline / degraded alert */}
          {connectionError && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs text-rose-900">
              <div className="flex items-start gap-2.5">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold">Backend Connection Unavailable: </span>
                  <span>{connectionError.message}</span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => refreshHealth()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-rose-200 text-rose-800 font-medium hover:bg-rose-100 transition-colors"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isCheckingHealth ? 'animate-spin' : ''}`} />
                Retry Connection
              </button>
            </div>
          )}

          {!connectionError && health && !health.model_loaded && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 flex items-start gap-2.5 text-xs text-amber-900">
              <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold">QA Model Not Ready ({health.model_status}): </span>
                <span>{health.model_error || 'The pretrained DistilBERT model is not loaded yet.'}</span>
              </div>
            </div>
          )}

          <StatsOverview
            health={health}
            connectionError={connectionError}
            document={qa.document}
            qaItems={qa.qaItems}
          />

          {activeTab === 'workspace' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
              {/* Left column: upload + active document */}
              <div className="lg:col-span-4 xl:col-span-3 space-y-5">
                <DocumentUpload
                  selectedFile={qa.selectedFile}
                  isUploading={qa.isUploading}
                  uploadProgress={qa.uploadProgress}
                  uploadError={qa.uploadError}
                  uploadSuccessMessage={qa.uploadSuccessMessage}
                  isBackendReady={isBackendReady}
                  hasActiveDocument={Boolean(qa.document)}
                  onUpload={qa.handleUpload}
                  onClearFeedback={qa.clearUploadFeedback}
                />

                <DocumentCard
                  document={qa.document}
                  documentExpired={qa.documentExpired}
                  isRefreshing={qa.isRefreshingDoc}
                  isDeleting={qa.isDeletingDoc}
                  docActionError={qa.docActionError}
                  docActionNotice={qa.docActionNotice}
                  onRefresh={qa.handleRefreshDocument}
                  onDelete={qa.handleDeleteDocument}
                  onClearNotice={qa.clearDocNotice}
                />
              </div>

              {/* Center column: Q&A chat interface */}
              <div className="lg:col-span-8 xl:col-span-6">
                <ChatInterface
                  document={qa.document}
                  documentExpired={qa.documentExpired}
                  isBackendReady={isBackendReady}
                  qaItems={qa.qaItems}
                  selectedQaId={qa.selectedQaId}
                  isAnswering={qa.isAnswering}
                  onAskQuestion={qa.handleAskQuestion}
                  onSelectSource={qa.setSelectedQaId}
                />
              </div>

              {/* Right column: source passage reference */}
              <div className="lg:col-span-12 xl:col-span-3 xl:sticky xl:top-20">
                <SourcePanel
                  document={qa.document}
                  qaItems={qa.qaItems}
                  selectedQaItem={qa.selectedQaItem}
                  onSelectQaId={qa.setSelectedQaId}
                />
              </div>
            </div>
          )}

          {activeTab === 'document' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
              <div className="lg:col-span-5 space-y-5">
                <DocumentCard
                  document={qa.document}
                  documentExpired={qa.documentExpired}
                  isRefreshing={qa.isRefreshingDoc}
                  isDeleting={qa.isDeletingDoc}
                  docActionError={qa.docActionError}
                  docActionNotice={qa.docActionNotice}
                  onRefresh={qa.handleRefreshDocument}
                  onDelete={qa.handleDeleteDocument}
                  onClearNotice={qa.clearDocNotice}
                />

                <DocumentUpload
                  selectedFile={qa.selectedFile}
                  isUploading={qa.isUploading}
                  uploadProgress={qa.uploadProgress}
                  uploadError={qa.uploadError}
                  uploadSuccessMessage={qa.uploadSuccessMessage}
                  isBackendReady={isBackendReady}
                  hasActiveDocument={Boolean(qa.document)}
                  onUpload={qa.handleUpload}
                  onClearFeedback={qa.clearUploadFeedback}
                />
              </div>

              <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200 shadow-card p-6">
                <h2 className="text-base font-semibold text-slate-900">
                  Document Processing Details
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  Detailed breakdown returned by the FastAPI document registry.
                </p>

                {qa.document ? (
                  <dl className="mt-5 divide-y divide-slate-100 text-sm">
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Original Filename</dt>
                      <dd className="font-medium text-slate-900 text-right break-all">{qa.document.filename}</dd>
                    </div>
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Document UUID</dt>
                      <dd className="font-mono text-xs text-slate-800 text-right break-all">{qa.document.document_id}</dd>
                    </div>
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Processing Status</dt>
                      <dd className="font-medium text-emerald-700 capitalize">{qa.document.status}</dd>
                    </div>
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Total Pages / With Text</dt>
                      <dd className="font-medium text-slate-900">
                        {qa.document.num_pages} total · {qa.document.pages_with_text} with text
                      </dd>
                    </div>
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Empty Pages Skipped</dt>
                      <dd className="font-medium text-slate-900">
                        {qa.document.empty_pages?.length ? qa.document.empty_pages.join(', ') : 'None'}
                      </dd>
                    </div>
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Token-Aware Chunks</dt>
                      <dd className="font-medium text-slate-900">{qa.document.num_chunks}</dd>
                    </div>
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Extracted Characters</dt>
                      <dd className="font-medium text-slate-900">
                        {qa.document.total_characters !== null && qa.document.total_characters !== undefined
                          ? qa.document.total_characters.toLocaleString()
                          : '—'}
                      </dd>
                    </div>
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Uploaded File Size</dt>
                      <dd className="font-medium text-slate-900">{formatBytes(qa.document.file_size_bytes)}</dd>
                    </div>
                    <div className="py-3 flex justify-between gap-4">
                      <dt className="text-slate-500">Uploaded Timestamp</dt>
                      <dd className="font-medium text-slate-900">{formatDateTime(qa.document.uploaded_at)}</dd>
                    </div>
                  </dl>
                ) : (
                  <p className="mt-5 text-xs text-slate-500">
                    Upload a PDF document to view its full extraction and indexing details.
                  </p>
                )}
              </div>
            </div>
          )}

          {activeTab === 'system' && (
            <SystemStatusView
              health={health}
              isCheckingHealth={isCheckingHealth}
              connectionError={connectionError}
              lastCheckedAt={lastCheckedAt}
              onRefreshHealth={refreshHealth}
            />
          )}
        </main>
      </div>
    </div>
  )
}

