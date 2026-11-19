import { useCallback, useState } from 'react'
import {
  askQuestion,
  deleteDocument,
  getDocumentDetails,
  uploadDocument,
} from '../services/api'

const MAX_QUESTION_CHARS = 500
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024

export function useDocumentQA({ onDocumentsChanged } = {}) {
  const [document, setDocument] = useState(null)
  const [documentExpired, setDocumentExpired] = useState(false)

  const [selectedFile, setSelectedFile] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadError, setUploadError] = useState(null)
  const [uploadSuccessMessage, setUploadSuccessMessage] = useState(null)

  const [isRefreshingDoc, setIsRefreshingDoc] = useState(false)
  const [isDeletingDoc, setIsDeletingDoc] = useState(false)
  const [docActionError, setDocActionError] = useState(null)
  const [docActionNotice, setDocActionNotice] = useState(null)

  const [qaItems, setQaItems] = useState([])
  const [selectedQaId, setSelectedQaId] = useState(null)

  const clearUploadFeedback = useCallback(() => {
    setUploadError(null)
    setUploadSuccessMessage(null)
  }, [])

  const clearDocNotice = useCallback(() => {
    setDocActionError(null)
    setDocActionNotice(null)
  }, [])

  const handleUpload = useCallback(
    async (file) => {
      if (!file) return false

      setSelectedFile({ name: file.name, size: file.size })
      setUploadError(null)
      setUploadSuccessMessage(null)
      setDocActionError(null)
      setDocActionNotice(null)

      // Quick pre-flight client validation for immediate feedback;
      // backend validation still acts as the source of truth.
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setUploadError({
          code: 'invalid_file_type',
          message: 'Only .pdf files are accepted. Please select a valid PDF document.',
        })
        return false
      }
      if (file.size === 0) {
        setUploadError({
          code: 'empty_file',
          message: 'The selected PDF file is empty (0 bytes).',
        })
        return false
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        setUploadError({
          code: 'file_too_large',
          message: `The selected file is ${(file.size / (1024 * 1024)).toFixed(1)} MB, which exceeds the 20 MB upload limit.`,
        })
        return false
      }

      setIsUploading(true)
      setUploadProgress(0)

      try {
        const uploadRes = await uploadDocument(file, setUploadProgress)
        setUploadProgress(100)

        let fullDoc = {
          ...uploadRes,
          file_size_bytes: file.size,
          total_characters: null,
          uploaded_at: new Date().toISOString(),
          uploadMessage: uploadRes.message,
        }

        try {
          const detailsRes = await getDocumentDetails(uploadRes.document_id)
          fullDoc = {
            ...detailsRes,
            uploadMessage: uploadRes.message,
          }
        } catch {
          // Keep upload response metadata if details lookup fails transiently.
        }

        setDocument(fullDoc)
        setDocumentExpired(false)
        setUploadSuccessMessage(uploadRes.message)
        setQaItems([])
        setSelectedQaId(null)
        onDocumentsChanged?.()
        return true
      } catch (err) {
        setUploadError(err)
        return false
      } finally {
        setIsUploading(false)
      }
    },
    [onDocumentsChanged],
  )

  const handleRefreshDocument = useCallback(async () => {
    if (!document?.document_id) return
    setIsRefreshingDoc(true)
    setDocActionError(null)
    setDocActionNotice(null)

    try {
      const detailsRes = await getDocumentDetails(document.document_id)
      setDocument((prev) => ({
        ...detailsRes,
        uploadMessage: prev?.uploadMessage || null,
      }))
      setDocumentExpired(false)
      setDocActionNotice('Document metadata synced with backend.')
    } catch (err) {
      if (err.code === 'document_not_found') {
        setDocumentExpired(true)
      }
      setDocActionError(err)
    } finally {
      setIsRefreshingDoc(false)
    }
  }, [document?.document_id])

  const handleDeleteDocument = useCallback(async () => {
    if (!document?.document_id) return false
    setIsDeletingDoc(true)
    setDocActionError(null)
    setDocActionNotice(null)

    const removedName = document.filename
    try {
      const res = await deleteDocument(document.document_id)
      setDocument(null)
      setDocumentExpired(false)
      setSelectedFile(null)
      setQaItems([])
      setSelectedQaId(null)
      setUploadSuccessMessage(null)
      setUploadError(null)
      setDocActionNotice(res.message || `Document '${removedName}' was deleted.`)
      onDocumentsChanged?.()
      return true
    } catch (err) {
      if (err.code === 'document_not_found') {
        // Document was already cleared on the server (e.g., server restart).
        setDocument(null)
        setDocumentExpired(false)
        setSelectedFile(null)
        setQaItems([])
        setSelectedQaId(null)
        setDocActionNotice(
          `Document '${removedName}' was no longer in backend memory and has been cleared locally.`,
        )
        onDocumentsChanged?.()
        return true
      }
      setDocActionError(err)
      return false
    } finally {
      setIsDeletingDoc(false)
    }
  }, [document?.document_id, document?.filename, onDocumentsChanged])

  const handleAskQuestion = useCallback(
    async (rawQuestion) => {
      if (!document?.document_id) return { ok: false, validationError: 'Please upload a PDF document first.' }

      const trimmed = rawQuestion.trim().replace(/\s+/g, ' ')
      if (!trimmed) {
        return { ok: false, validationError: 'Question must not be empty.' }
      }
      if (trimmed.length > MAX_QUESTION_CHARS) {
        return {
          ok: false,
          validationError: `Question exceeds ${MAX_QUESTION_CHARS} characters (${trimmed.length}/${MAX_QUESTION_CHARS}).`,
        }
      }

      const itemId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
      const pendingItem = {
        id: itemId,
        question: trimmed,
        askedAt: new Date().toISOString(),
        isLoading: true,
        response: null,
        error: null,
      }

      setQaItems((prev) => [...prev, pendingItem])

      try {
        const res = await askQuestion(document.document_id, trimmed)
        setQaItems((prev) =>
          prev.map((item) =>
            item.id === itemId ? { ...item, isLoading: false, response: res } : item,
          ),
        )
        if (res.source) {
          setSelectedQaId(itemId)
        }
        return { ok: true, response: res }
      } catch (err) {
        if (err.code === 'document_not_found') {
          setDocumentExpired(true)
          setDocActionError({
            code: 'document_not_found',
            message:
              'This document is no longer in the backend registry (the server may have restarted). Please upload the PDF again.',
          })
        }
        setQaItems((prev) =>
          prev.map((item) =>
            item.id === itemId ? { ...item, isLoading: false, error: err } : item,
          ),
        )
        return { ok: false, error: err }
      }
    },
    [document?.document_id],
  )

  const selectedQaItem = qaItems.find((item) => item.id === selectedQaId) || null
  const isAnswering = qaItems.some((item) => item.isLoading)

  return {
    document,
    documentExpired,
    selectedFile,
    isUploading,
    uploadProgress,
    uploadError,
    uploadSuccessMessage,
    isRefreshingDoc,
    isDeletingDoc,
    docActionError,
    docActionNotice,
    qaItems,
    selectedQaId,
    selectedQaItem,
    isAnswering,
    setSelectedQaId,
    clearUploadFeedback,
    clearDocNotice,
    handleUpload,
    handleRefreshDocument,
    handleDeleteDocument,
    handleAskQuestion,
  }
}

