import axios from 'axios'

const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, '')

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
})

export class ApiError extends Error {
  constructor({ code, message, details = null, httpStatus = null, isNetworkError = false }) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.details = details
    this.httpStatus = httpStatus
    this.isNetworkError = isNetworkError
  }
}

export function normalizeApiError(error) {
  if (error instanceof ApiError) {
    return error
  }

  // Request never received a response (backend offline, CORS block, timeout, etc.)
  if (!error?.response) {
    const isTimeout = error?.code === 'ECONNABORTED'
    return new ApiError({
      code: isTimeout ? 'request_timeout' : 'network_error',
      message: isTimeout
        ? 'The request timed out while waiting for the backend server.'
        : `Cannot reach the backend server at ${API_BASE_URL}. Make sure FastAPI is running.`,
      isNetworkError: true,
    })
  }

  const { status, data } = error.response
  const backendCode = data?.error || `http_${status}`
  const backendMessage =
    data?.message ||
    data?.detail ||
    `Request failed with HTTP status ${status}.`
  const details = Array.isArray(data?.details) ? data.details : null

  return new ApiError({
    code: backendCode,
    message: backendMessage,
    details,
    httpStatus: status,
    isNetworkError: false,
  })
}

export function getApiBaseUrl() {
  return API_BASE_URL
}

export async function getHealth() {
  try {
    const { data } = await apiClient.get('/api/health')
    return data
  } catch (err) {
    throw normalizeApiError(err)
  }
}

export async function uploadDocument(file, onProgress) {
  try {
    const formData = new FormData()
    formData.append('file', file)

    const { data } = await apiClient.post('/api/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (!onProgress) return
        if (progressEvent.total && progressEvent.total > 0) {
          const percent = Math.min(100, Math.round((progressEvent.loaded * 100) / progressEvent.total))
          onProgress(percent)
        }
      },
    })
    return data
  } catch (err) {
    throw normalizeApiError(err)
  }
}

export async function getDocumentDetails(documentId) {
  try {
    const { data } = await apiClient.get(`/api/documents/${encodeURIComponent(documentId)}`)
    return data
  } catch (err) {
    throw normalizeApiError(err)
  }
}

export async function deleteDocument(documentId) {
  try {
    const { data } = await apiClient.delete(`/api/documents/${encodeURIComponent(documentId)}`)
    return data
  } catch (err) {
    throw normalizeApiError(err)
  }
}

export async function askQuestion(documentId, question) {
  try {
    const { data } = await apiClient.post('/api/questions/ask', {
      document_id: documentId,
      question,
    })
    return data
  } catch (err) {
    throw normalizeApiError(err)
  }
}

