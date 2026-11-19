import { useRef, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  FileUp,
  Loader2,
  Trash2,
  UploadCloud,
} from 'lucide-react'
import { formatBytes } from '../utils/formatters'

export default function DocumentUpload({
  selectedFile,
  isUploading,
  uploadProgress,
  uploadError,
  uploadSuccessMessage,
  isBackendReady,
  hasActiveDocument,
  onUpload,
  onClearFeedback,
}) {
  const fileInputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const triggerBrowse = () => {
    if (isUploading) return
    fileInputRef.current?.click()
  }

  const processFile = async (file) => {
    if (!file || isUploading) return
    await onUpload(file)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    processFile(file)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    event.stopPropagation()
    if (!isUploading) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (event) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(false)
    const file = event.dataTransfer?.files?.[0]
    processFile(file)
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-card p-5">
      <div className="flex items-center justify-between gap-2 mb-3.5">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            {hasActiveDocument ? 'Upload Another PDF' : 'Upload PDF Document'}
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Selectable-text PDFs up to 20 MB (OCR not supported).
          </p>
        </div>
      </div>

      {/* Dropzone */}
      <div
        role="button"
        tabIndex={0}
        onClick={triggerBrowse}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            triggerBrowse()
          }
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative rounded-xl border-2 border-dashed p-5 text-center transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-1 ${
          isDragging
            ? 'border-sky-500 bg-sky-50/70'
            : isUploading
              ? 'border-slate-200 bg-slate-50 cursor-wait'
              : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/60'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileChange}
          disabled={isUploading}
          className="hidden"
          aria-label="Select PDF file"
        />

        <div className="flex flex-col items-center">
          <div
            className={`w-10 h-10 rounded-full flex items-center justify-center mb-3 ${
              isDragging ? 'bg-sky-100 text-sky-600' : 'bg-slate-100 text-slate-600'
            }`}
          >
            {isUploading ? (
              <Loader2 className="w-5 h-5 animate-spin text-sky-600" />
            ) : (
              <UploadCloud className="w-5 h-5" />
            )}
          </div>

          <p className="text-sm font-medium text-slate-800">
            {isUploading
              ? 'Uploading & indexing PDF...'
              : 'Drag & drop your PDF here'}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            or click to browse files from your computer
          </p>

          <button
            type="button"
            disabled={isUploading}
            onClick={(e) => {
              e.stopPropagation()
              triggerBrowse()
            }}
            className="mt-3.5 inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 disabled:opacity-50 transition-colors"
          >
            <FileUp className="w-3.5 h-3.5" />
            Browse Files
          </button>
        </div>
      </div>

      {/* Backend unavailable warning */}
      {!isBackendReady && (
        <div className="mt-3 flex items-start gap-2 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2.5 text-xs text-amber-800">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
          <span>
            The backend or QA model is not currently ready. Uploads require the backend server and DistilBERT tokenizer to be loaded.
          </span>
        </div>
      )}

      {/* Selected file & upload progress */}
      {isUploading && selectedFile && (
        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="flex items-center justify-between gap-2 text-xs">
            <span className="font-medium text-slate-800 truncate">{selectedFile.name}</span>
            <span className="text-slate-500 shrink-0">{formatBytes(selectedFile.size)}</span>
          </div>
          <div className="mt-2 h-1.5 w-full rounded-full bg-slate-200 overflow-hidden">
            <div
              className="h-full bg-sky-600 transition-all duration-200"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p className="mt-1.5 text-[11px] text-slate-500">
            {uploadProgress < 100
              ? `Uploading file... ${uploadProgress}%`
              : 'Extracting pages, chunking tokens & building TF-IDF index...'}
          </p>
        </div>
      )}

      {/* Upload success */}
      {!isUploading && uploadSuccessMessage && (
        <div className="mt-3.5 flex items-start justify-between gap-2 rounded-lg bg-emerald-50 border border-emerald-200 px-3 py-2.5 text-xs text-emerald-800">
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-600" />
            <span>{uploadSuccessMessage}</span>
          </div>
          <button
            type="button"
            onClick={onClearFeedback}
            className="text-emerald-700 hover:text-emerald-900 text-[11px] font-medium shrink-0"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Upload error with clear/remove selected file option */}
      {!isUploading && uploadError && (
        <div className="mt-3.5 rounded-lg bg-rose-50 border border-rose-200 p-3 text-xs text-rose-800">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
            <div className="flex-1 min-w-0">
              {selectedFile && (
                <p className="font-semibold text-rose-900 truncate mb-0.5">
                  {selectedFile.name} ({formatBytes(selectedFile.size)})
                </p>
              )}
              <p>{uploadError.message}</p>
            </div>
          </div>
          <div className="mt-2 flex justify-end">
            <button
              type="button"
              onClick={onClearFeedback}
              className="inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium text-rose-700 hover:bg-rose-100 transition-colors"
            >
              <Trash2 className="w-3 h-3" />
              Remove Selection
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

