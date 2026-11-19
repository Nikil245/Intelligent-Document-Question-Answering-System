import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  BookOpen,
  FileSearch,
  HelpCircle,
  Info,
  Loader2,
  Send,
  Sparkles,
} from 'lucide-react'
import { formatConfidence, formatTimeOnly } from '../utils/formatters'

const SUGGESTED_QUESTIONS = [
  'What is the main topic of this document?',
  'Who created DistilBERT?',
  'When was the Transformer architecture introduced?',
  'What is TF-IDF?',
]

export default function ChatInterface({
  document,
  documentExpired,
  isBackendReady,
  qaItems,
  selectedQaId,
  isAnswering,
  onAskQuestion,
  onSelectSource,
}) {
  const [input, setInput] = useState('')
  const [validationError, setValidationError] = useState(null)
  const feedEndRef = useRef(null)

  const canAsk = Boolean(document && !documentExpired && isBackendReady && !isAnswering)

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [qaItems.length, isAnswering])

  const submitQuestion = async (questionText) => {
    setValidationError(null)
    const result = await onAskQuestion(questionText)
    if (result?.validationError) {
      setValidationError(result.validationError)
      return
    }
    if (result?.ok) {
      setInput('')
    }
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    if (!document || documentExpired || !isBackendReady || isAnswering) return
    submitQuestion(input)
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-card flex flex-col h-[620px] xl:h-[680px]">
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            Question & Answer Assistant
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            {document
              ? `Ask questions about "${document.filename}"`
              : 'Upload a PDF document to enable question answering'}
          </p>
        </div>

        {document && (
          <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 text-[11px] font-medium">
            <BookOpen className="w-3.5 h-3.5 text-slate-500" />
            {document.num_pages} {document.num_pages === 1 ? 'page' : 'pages'} indexed
          </span>
        )}
      </div>

      {/* Conversation feed */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-5 space-y-5 bg-slate-50/40">
        {!document ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto px-4">
            <div className="w-12 h-12 rounded-2xl bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-500 mb-3.5">
              <FileSearch className="w-6 h-6" />
            </div>
            <h3 className="text-sm font-semibold text-slate-900">
              No Document Active
            </h3>
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              Upload a PDF on the left to extract its text page by page and build a TF-IDF index. Once processed, you can ask as many questions as you want without re-uploading.
            </p>
          </div>
        ) : qaItems.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto px-4">
            <div className="w-11 h-11 rounded-2xl bg-sky-50 border border-sky-100 flex items-center justify-center text-sky-600 mb-3.5">
              <Sparkles className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-semibold text-slate-900">
              Ready for Your Questions
            </h3>
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              Relevant chunks are selected via TF-IDF cosine similarity, and DistilBERT extracts exact answer spans directly from the text.
            </p>

            <div className="mt-5 w-full">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">
                Try asking
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTED_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    disabled={!canAsk}
                    onClick={() => submitQuestion(q)}
                    className="px-3 py-1.5 rounded-lg bg-white border border-slate-200 text-xs text-slate-700 hover:border-sky-300 hover:bg-sky-50/50 disabled:opacity-50 transition-colors text-left"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          qaItems.map((item) => {
            const res = item.response
            const hasSource = Boolean(res?.source)
            const isSelected = selectedQaId === item.id

            return (
              <div key={item.id} className="space-y-3">
                {/* User question */}
                <div className="flex justify-end">
                  <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-slate-900 text-white px-4 py-2.5 shadow-subtle">
                    <p className="text-sm leading-relaxed">{item.question}</p>
                    <p className="text-[10px] text-slate-300 mt-1 text-right">
                      {formatTimeOnly(item.askedAt)}
                    </p>
                  </div>
                </div>

                {/* Assistant response */}
                <div className="flex justify-start">
                  <div className="w-full max-w-[92%]">
                    {item.isLoading ? (
                      <div className="rounded-2xl rounded-tl-sm bg-white border border-slate-200 px-4 py-3.5 shadow-subtle flex items-center gap-3">
                        <Loader2 className="w-4 h-4 text-sky-600 animate-spin shrink-0" />
                        <div>
                          <p className="text-xs font-medium text-slate-800">
                            Searching document & running DistilBERT QA...
                          </p>
                          <p className="text-[11px] text-slate-500 mt-0.5">
                            Ranking chunks with TF-IDF cosine similarity and extracting answer span
                          </p>
                        </div>
                      </div>
                    ) : item.error ? (
                      <div className="rounded-2xl rounded-tl-sm bg-rose-50 border border-rose-200 px-4 py-3 text-xs text-rose-800 flex items-start gap-2.5">
                        <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                        <div>
                          <p className="font-semibold text-rose-900">Could not process question</p>
                          <p className="mt-0.5">{item.error.message}</p>
                        </div>
                      </div>
                    ) : res?.success ? (
                      <div
                        className={`rounded-2xl rounded-tl-sm bg-white border px-4 py-3.5 shadow-subtle transition-colors ${
                          isSelected ? 'border-sky-300 ring-1 ring-sky-200' : 'border-slate-200'
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                          <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-emerald-700">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                            {res.question_type === 'descriptive'
                              ? 'Extractive Key Span · Descriptive Question'
                              : 'Extracted Answer Span'}
                          </span>
                          {res.source && (
                            <span className="px-2 py-0.5 rounded-md bg-sky-50 border border-sky-200 text-[11px] font-medium text-sky-800">
                              Page {res.source.page_number}
                            </span>
                          )}
                        </div>

                        <p className="text-sm sm:text-[15px] font-medium text-slate-900 leading-relaxed">
                          {res.answer}
                        </p>

                        {res.question_type === 'descriptive' && (
                          <p className="mt-1.5 text-[11px] text-slate-500 leading-relaxed">
                            Note: DistilBERT is an extractive QA model and highlights a verbatim span rather than generating a complete prose explanation. See the supporting passages across the retrieved PDF pages below.
                          </p>
                        )}

                        {/* Confidence + metrics row */}
                        <div className="mt-3 pt-2.5 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <span
                              className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 text-[11px] font-medium text-slate-700"
                              title="DistilBERT span score is a model confidence score, not a calibrated probability of correctness."
                            >
                              Model Score: {formatConfidence(res.confidence)}
                            </span>
                            {res.source && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded bg-slate-100 text-[11px] text-slate-600">
                                TF-IDF Sim: {res.source.retrieval_score.toFixed(3)}
                              </span>
                            )}
                          </div>

                          {hasSource && (
                            <button
                              type="button"
                              onClick={() => onSelectSource(item.id)}
                              className={`text-xs font-medium px-2.5 py-1 rounded-md transition-colors ${
                                isSelected
                                  ? 'bg-sky-600 text-white'
                                  : 'text-sky-700 hover:bg-sky-50'
                              }`}
                            >
                              {isSelected ? 'Viewing Source' : 'Inspect Source'}
                            </button>
                          )}
                        </div>

                        {/* Supporting passage snippet */}
                        {res.source && (
                          <div className="mt-2.5 rounded-lg bg-slate-50 border border-slate-200/80 p-2.5 text-xs text-slate-600 leading-relaxed">
                            <span className="font-medium text-slate-700">Primary supporting passage (p. {res.source.page_number}): </span>
                            {res.source.passage}
                          </div>
                        )}

                        {/* Additional relevant passages for multi-concept / descriptive queries */}
                        {Array.isArray(res.supporting_passages) && res.supporting_passages.length > 1 && (
                          <div className="mt-2.5 space-y-1.5">
                            <p className="text-[11px] font-semibold text-slate-600">
                              Additional Relevant Supporting Passages ({res.supporting_passages.length - 1})
                            </p>
                            {res.supporting_passages.slice(1).map((sp) => (
                              <div
                                key={`${sp.chunk_id}-${sp.page_number}`}
                                className="rounded-lg bg-slate-50/70 border border-slate-200/70 p-2.5 text-[11px] text-slate-600 leading-relaxed"
                              >
                                <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                                  <span className="font-semibold text-slate-800">
                                    Page {sp.page_number} · Chunk #{sp.chunk_id}
                                  </span>
                                  <span className="text-slate-500">
                                    TF-IDF Sim: {sp.retrieval_score.toFixed(3)}
                                  </span>
                                </div>
                                <p>{sp.passage}</p>
                              </div>
                            ))}
                          </div>
                        )}

                        <p className="mt-2 flex items-start gap-1.5 text-[11px] text-slate-400">
                          <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                          <span>
                            Model score reflects DistilBERT&apos;s span confidence on the retrieved chunk, not a calibrated probability of factual correctness.
                          </span>
                        </p>
                      </div>
                    ) : (
                      /* Valid request, but no_relevant_content / low_confidence / no_answer */
                      <div
                        className={`rounded-2xl rounded-tl-sm border px-4 py-3.5 shadow-subtle ${
                          res?.status === 'low_confidence'
                            ? 'bg-amber-50/70 border-amber-200'
                            : 'bg-white border-slate-200'
                        }`}
                      >
                        <div className="flex items-start gap-2.5">
                          <HelpCircle
                            className={`w-4 h-4 shrink-0 mt-0.5 ${
                              res?.status === 'low_confidence'
                                ? 'text-amber-600'
                                : 'text-slate-500'
                            }`}
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between gap-2">
                              <p className="text-xs font-semibold text-slate-900">
                                {res?.status === 'low_confidence'
                                  ? 'Low Confidence — No Reliable Answer'
                                  : 'No Relevant Answer Found'}
                              </p>
                              {res?.confidence !== null && res?.confidence !== undefined && (
                                <span className="text-[11px] font-medium text-amber-800 bg-amber-100 px-2 py-0.5 rounded">
                                  Score: {formatConfidence(res.confidence)}
                                </span>
                              )}
                            </div>
                            <p className="mt-1 text-xs text-slate-700 leading-relaxed">
                              {res?.message}
                            </p>

                            {hasSource && (
                              <div className="mt-2.5 flex items-center justify-between gap-2 pt-2 border-t border-amber-200/70">
                                <span className="text-[11px] text-amber-900">
                                  Closest passage found on Page {res.source.page_number}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => onSelectSource(item.id)}
                                  className="text-xs font-medium text-amber-900 underline hover:text-amber-700"
                                >
                                  {isSelected ? 'Viewing Reference' : 'View Reference'}
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })
        )}
        <div ref={feedEndRef} />
      </div>

      {/* Question input bar */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-slate-200 bg-white rounded-b-xl">
        {documentExpired && (
          <p className="mb-2 text-xs text-rose-600">
            This document is no longer available on the backend. Upload the PDF again to continue asking questions.
          </p>
        )}
        {validationError && (
          <p className="mb-2 text-xs text-rose-600">{validationError}</p>
        )}

        <div className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            maxLength={500}
            disabled={!document || documentExpired || !isBackendReady || isAnswering}
            onChange={(e) => {
              setInput(e.target.value)
              if (validationError) setValidationError(null)
            }}
            placeholder={
              !document
                ? 'Upload a PDF first to start asking questions...'
                : !isBackendReady
                  ? 'Waiting for backend & DistilBERT model...'
                  : 'Ask a question about the uploaded PDF...'
            }
            className="flex-1 rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:bg-white focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-500/20 disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
            aria-label="Question input"
          />

          <button
            type="submit"
            disabled={!canAsk}
            className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-lg bg-slate-900 text-white text-sm font-medium hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
          >
            {isAnswering ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            <span>Ask</span>
          </button>
        </div>

        <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-400">
          <span>Extractive QA: answers are copied verbatim from the PDF.</span>
          <span>{input.length}/500</span>
        </div>
      </form>
    </div>
  )
}

