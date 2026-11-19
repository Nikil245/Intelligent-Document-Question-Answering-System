export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) {
    return '—'
  }
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export function formatDateTime(isoString) {
  if (!isoString) return '—'
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatTimeOnly(isoString) {
  if (!isoString) return ''
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatConfidence(score) {
  if (score === null || score === undefined || Number.isNaN(score)) {
    return '—'
  }
  return `${(score * 100).toFixed(1)}% (${score.toFixed(3)})`
}

