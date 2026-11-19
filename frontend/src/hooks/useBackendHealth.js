import { useCallback, useEffect, useRef, useState } from 'react'
import { getHealth } from '../services/api'

const POLL_INTERVAL_MS = 30000

export function useBackendHealth() {
  const [health, setHealth] = useState(null)
  const [isChecking, setIsChecking] = useState(true)
  const [connectionError, setConnectionError] = useState(null)
  const [lastCheckedAt, setLastCheckedAt] = useState(null)
  const isMountedRef = useRef(true)

  const refreshHealth = useCallback(async ({ silent = false } = {}) => {
    if (!silent) {
      setIsChecking(true)
    }
    try {
      const data = await getHealth()
      if (!isMountedRef.current) return null
      setHealth(data)
      setConnectionError(null)
      setLastCheckedAt(new Date())
      return data
    } catch (err) {
      if (!isMountedRef.current) return null
      setHealth(null)
      setConnectionError(err)
      setLastCheckedAt(new Date())
      return null
    } finally {
      if (isMountedRef.current) {
        setIsChecking(false)
      }
    }
  }, [])

  useEffect(() => {
    isMountedRef.current = true
    refreshHealth()

    const timerId = window.setInterval(() => {
      refreshHealth({ silent: true })
    }, POLL_INTERVAL_MS)

    return () => {
      isMountedRef.current = false
      window.clearInterval(timerId)
    }
  }, [refreshHealth])

  return {
    health,
    isChecking,
    connectionError,
    lastCheckedAt,
    refreshHealth,
  }
}

