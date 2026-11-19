import { useState } from 'react'
import {
  BookOpenCheck,
  ChevronDown,
  ChevronUp,
  FileText,
  Hash,
  Info,
  Search,
} from 'lucide-react'
import { formatConfidence } from '../utils/formatters'

function HighlightedPassage({ passage, answerStart, answerEnd }) {
  const hasValidSpan =
    typeof answerStart === 'number' &&
    typeof answerEnd === 'number' &&
    answerStart >= 0 &&
    answerEnd > answerStart &&
    answerEnd <= passage.length

  if (!hasValidSpan) {
    return <p className="text-xs sm:text-[13px] text-slate-700 leading-relaxed">{passage}</p>
  }

  const before = passage.slice(0, answerStart)
  const match = passage.slice(answerStart, answerEnd)
  const after = passage.slice(answerEnd)

  return (
    <p className="text-xs sm:text-[13px] text-slate-700 leading-relaxed">
      {before}
      <mark className="px-1 py-0.5 rounded bg-amber-200/80 text-slate-950 font-medium">
        {match}
      </mark>
      {after}
    </p>
  )
}

export default function SourcePanel({
  document,
  qaItems,
  selectedQaItem,
  onSelectQaId,
}) {
  const [showDebug, setShowDebug] = useState(false)

  const itemsWithSource = qaItems.filter((item) => item.response?.source)
  const selectedResponse = selectedQaItem?.response
  const additionalPassages = Array.isArray(selectedResponse?.supporting_passages)
    ? selectedResponse.supporting_passages.slice(1)
    : []
  const retrievedChunks = Array.isArray(selectedResponse?.retrieved_chunks)
    ? selectedResponse.retrieved_chunks
    : []

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-card p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <BookOpenCheck className="w-4 h-4 text-sky-600" />
          <h2 className="text-sm font-semibold text-slate-900">Source Reference</h2>
        </div>
        {selectedResponse?.source && (
          <span className="px-2 py-0.5 rounded-full bg-sky-50 border border-sky-200 text-xs font-semibold text-sky-800">
            Page {selectedResponse.source.page_number}
          </span>
        )}
      </div>

      {!selectedResponse?.source ? (
        <div className="py-10 px-3 text-center">
          <div className="w-10 h-10 rounded-xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto mb-3">
            <Search className="w-5 h-5" />
          </div>
          <p className="text-xs font-medium text-slate-700">
            No Source Passage Selected
          </p>
          <p className="text-xs text-slate-500 mt-1 leading-relaxed">
            {document
              ? 'Ask a question to inspect the retrieved PDF chunks, page numbers, TF-IDF similarity, and highlighted answer span.'
              : 'Upload a PDF and ask a question to inspect grounded source citations.'}
          </p>
        </div>
      ) : (
        <>
          {/* Turn switcher when multiple source-backed answers exist */}
          {itemsWithSource.length > 1 && (
            <div>
              <p className="text-[11px] font-medium text-slate-500 mb-1.5">
                Switch cited question ({itemsWithSource.length})
              </p>
              <div className="flex flex-wrap gap-1.5">
                {itemsWithSource.map((item, idx) => {
                  const active = item.id === selectedQaItem.id
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => onSelectQaId(item.id)}
                      className={`px-2 py-1 rounded-md text-[11px] font-medium border transition-colors ${
                        active
                          ? 'bg-slate-900 text-white border-slate-900'
                          : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
                      }`}
                      title={item.question}
                    >
                      Q{idx + 1} · p.{item.response.source.page_number}
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {/* Selected question context */}
          <div className="rounded-lg bg-slate-50 border border-slate-200/80 p-3">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
              Question
            </p>
            <p className="text-xs font-medium text-slate-900 mt-1">
              {selectedQaItem.question}
            </p>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-2.5">
            <div className="rounded-lg border border-slate-200 p-2.5">
              <div className="flex items-center gap-1 text-[11px] text-slate-500">
                <FileText className="w-3.5 h-3.5" />
                <span>PDF Page</span>
              </div>
              <p className="text-sm font-semibold text-slate-900 mt-0.5">
                Page {selectedResponse.source.page_number}
              </p>
            </div>

            <div className="rounded-lg border border-slate-200 p-2.5">
              <div className="flex items-center gap-1 text-[11px] text-slate-500">
                <Hash className="w-3.5 h-3.5" />
                <span>Chunk Index</span>
              </div>
              <p className="text-sm font-semibold text-slate-900 mt-0.5">
                Chunk #{selectedResponse.source.chunk_id}
              </p>
            </div>

            <div className="rounded-lg border border-slate-200 p-2.5">
              <p className="text-[11px] text-slate-500">TF-IDF Cosine Sim</p>
              <p className="text-sm font-semibold text-slate-900 mt-0.5">
                {selectedResponse.source.retrieval_score.toFixed(4)}
              </p>
            </div>

            <div className="rounded-lg border border-slate-200 p-2.5">
              <p className="text-[11px] text-slate-500">Model Score</p>
              <p className="text-sm font-semibold text-slate-900 mt-0.5">
                {formatConfidence(selectedResponse.confidence)}
              </p>
            </div>
          </div>

          {/* Primary supporting passage */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-slate-800">
                Supporting Passage
              </span>
              {selectedResponse.success && (
                <span className="text-[11px] text-amber-800 bg-amber-100 px-1.5 py-0.5 rounded font-medium">
                  Extracted span highlighted
                </span>
              )}
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3.5 max-h-56 overflow-y-auto custom-scrollbar">
              <HighlightedPassage
                passage={selectedResponse.source.passage}
                answerStart={selectedResponse.source.answer_start}
                answerEnd={selectedResponse.source.answer_end}
              />
            </div>
          </div>

          {/* Additional relevant passages from other retrieved pages */}
          {additionalPassages.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-800">
                Additional Relevant Passages ({additionalPassages.length})
              </p>
              <div className="space-y-2 max-h-56 overflow-y-auto custom-scrollbar pr-0.5">
                {additionalPassages.map((sp) => (
                  <div
                    key={`${sp.chunk_id}-${sp.page_number}`}
                    className="rounded-lg border border-slate-200 bg-slate-50/50 p-2.5 space-y-1.5"
                  >
                    <div className="flex items-center justify-between gap-2 text-[11px]">
                      <span className="font-semibold text-slate-800">
                        Page {sp.page_number} · Chunk #{sp.chunk_id}
                      </span>
                      <span className="text-slate-500">
                        Sim: {sp.retrieval_score.toFixed(3)}
                        {sp.confidence !== null && sp.confidence !== undefined
                          ? ` · Score: ${sp.confidence.toFixed(3)}`
                          : ''}
                      </span>
                    </div>
                    <HighlightedPassage
                      passage={sp.passage}
                      answerStart={sp.answer_start}
                      answerEnd={sp.answer_end}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Collapsible retrieval & QA debug details */}
          {retrievedChunks.length > 0 && (
            <div className="border-t border-slate-100 pt-3">
              <button
                type="button"
                onClick={() => setShowDebug((prev) => !prev)}
                className="w-full flex items-center justify-between text-xs font-semibold text-slate-700 hover:text-slate-900 py-1"
              >
                <span>Retrieval & QA Debug Details ({retrievedChunks.length} chunks)</span>
                {showDebug ? (
                  <ChevronUp className="w-4 h-4 text-slate-500" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-slate-500" />
                )}
              </button>

              {showDebug && (
                <div className="mt-2 space-y-2 max-h-72 overflow-y-auto custom-scrollbar pr-0.5">
                  {retrievedChunks.map((rc) => (
                    <div
                      key={`${rc.rank}-${rc.chunk_id}`}
                      className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-[11px] space-y-1"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-1">
                        <span className="font-semibold text-slate-900">
                          #{rc.rank} · Page {rc.page_number} (Chunk {rc.chunk_id}, {rc.num_tokens} tok)
                        </span>
                        {rc.is_heading_only && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">
                            Heading-only
                          </span>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-slate-600">
                        <span>Final Sim: {rc.retrieval_score.toFixed(4)}</span>
                        <span>Raw TF-IDF: {rc.raw_tfidf_score.toFixed(4)}</span>
                        <span>Concept Cov: {(rc.concept_coverage * 100).toFixed(0)}%</span>
                      </div>
                      {rc.matched_concepts?.length > 0 && (
                        <p className="text-slate-500">
                          <span className="font-medium text-slate-700">Concepts: </span>
                          {rc.matched_concepts.join(', ')}
                        </p>
                      )}
                      {rc.candidate_answer && (
                        <p className="text-slate-700">
                          <span className="font-medium">Span ({rc.candidate_score?.toFixed(3)}): </span>
                          &ldquo;{rc.candidate_answer}&rdquo;
                        </p>
                      )}
                      <p className="text-slate-500 line-clamp-2">{rc.text_preview}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Academic clarity note */}
          <div className="rounded-lg bg-sky-50/70 border border-sky-100 p-3 text-[11px] text-slate-600 flex items-start gap-2">
            <Info className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-medium text-slate-800">How this reference was found</p>
              <p className="leading-relaxed">
                TF-IDF cosine similarity + concept coverage ranked the PDF chunks and selected Chunk #{selectedResponse.source.chunk_id} from Page {selectedResponse.source.page_number}. DistilBERT then extracted the verbatim answer span inside this text. The model score is an uncalibrated confidence score, not a calibrated probability.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

