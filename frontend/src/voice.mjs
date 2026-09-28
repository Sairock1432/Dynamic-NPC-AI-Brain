import { languageLocale, normalizeLanguage } from './language.mjs'

export function getSpeechRecognitionConstructor(environment) {
  return environment?.SpeechRecognition ?? environment?.webkitSpeechRecognition ?? null
}

export function speechRecognitionErrorMessage(errorCode) {
  const messages = {
    'not-allowed': 'Microphone permission was denied. You can type your message instead.',
    'service-not-allowed': 'Speech recognition is not allowed. You can type your message instead.',
    'audio-capture': 'No microphone is available. You can type your message instead.',
    'no-speech': 'No speech was recognized. Try again or type your message.',
    network: 'Speech recognition is unavailable right now. You can type your message instead.',
    empty: 'No speech was recognized. Try again or type your message.',
    unsupported: 'Speech input is not supported in this browser. You can type your message instead.',
  }
  return messages[errorCode] ?? 'Speech recognition failed. You can type your message instead.'
}

export function extractSpeechTranscript(event) {
  const results = event?.results
  if (!results || typeof results.length !== 'number') return ''
  const startIndex = Math.max(0, Number(event.resultIndex) || 0)
  for (let index = results.length - 1; index >= startIndex; index -= 1) {
    const result = results[index]
    if (result?.isFinal === false) continue
    const transcript = result?.[0]?.transcript
    if (typeof transcript === 'string' && transcript.trim()) return transcript.trim()
  }
  return ''
}

export function createSpeechRecognitionController(
  Constructor,
  language,
  { onTranscript, onError, onEnd } = {},
) {
  if (typeof Constructor !== 'function') return null

  let recognition
  try {
    recognition = new Constructor()
  } catch {
    onError?.(speechRecognitionErrorMessage('unsupported'))
    return null
  }

  let isActive = false
  recognition.lang = languageLocale(language)
  recognition.interimResults = false
  recognition.continuous = false
  recognition.maxAlternatives = 1
  recognition.onresult = (event) => {
    const transcript = extractSpeechTranscript(event)
    if (transcript) onTranscript?.(transcript)
    else onError?.(speechRecognitionErrorMessage('empty'))
  }
  recognition.onerror = (event) => {
    onError?.(speechRecognitionErrorMessage(event?.error))
  }
  recognition.onend = () => {
    isActive = false
    onEnd?.()
  }

  return {
    start() {
      if (isActive) return true
      try {
        recognition.start()
        isActive = true
        return true
      } catch (error) {
        onError?.(speechRecognitionErrorMessage(error?.name === 'NotAllowedError'
          ? 'not-allowed'
          : 'unsupported'))
        return false
      }
    },
    stop() {
      if (!isActive) return
      try {
        recognition.stop()
      } catch {
        isActive = false
        onEnd?.()
      }
    },
  }
}

export function selectSpeechVoice(voices, language) {
  const languageCode = normalizeLanguage(language)
  const locale = languageLocale(language).toLowerCase()
  return voices.find((voice) => voice.lang?.toLowerCase() === locale)
    ?? voices.find((voice) => voice.lang?.toLowerCase().startsWith(`${languageCode}-`))
    ?? voices.find((voice) => voice.default)
    ?? voices[0]
}

export function isTextToSpeechAvailable(environment) {
  return Boolean(
    environment?.speechSynthesis
    && typeof environment.speechSynthesis.speak === 'function'
    && typeof environment.speechSynthesis.cancel === 'function'
    && typeof environment?.SpeechSynthesisUtterance === 'function',
  )
}

export function speakText({
  synthesis,
  UtteranceConstructor,
  text,
  language,
  onStart,
  onEnd,
  onError,
}) {
  if (!synthesis || typeof synthesis.speak !== 'function' || typeof synthesis.cancel !== 'function'
    || typeof UtteranceConstructor !== 'function') {
    onError?.('unsupported')
    return false
  }
  if (typeof text !== 'string' || !text.trim()) {
    onError?.('empty')
    return false
  }

  try {
    synthesis.cancel()
    const utterance = new UtteranceConstructor(text)
    utterance.lang = languageLocale(language)
    const voice = selectSpeechVoice(synthesis.getVoices?.() ?? [], language)
    if (voice) utterance.voice = voice
    utterance.onstart = () => onStart?.()
    utterance.onend = () => onEnd?.()
    utterance.onerror = () => onError?.('playback')
    synthesis.speak(utterance)
    return true
  } catch {
    onError?.('playback')
    return false
  }
}

export function stopSpeaking(synthesis) {
  if (!synthesis || typeof synthesis.cancel !== 'function') return false
  try {
    synthesis.cancel()
    return true
  } catch {
    return false
  }
}