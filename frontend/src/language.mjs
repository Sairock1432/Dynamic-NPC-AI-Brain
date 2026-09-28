export const DEFAULT_LANGUAGE = 'en'

export const LANGUAGE_OPTIONS = Object.freeze([
  Object.freeze({ code: 'en', label: 'English', speechLocale: 'en-US' }),
  Object.freeze({ code: 'te', label: 'తెలుగు', speechLocale: 'te-IN' }),
  Object.freeze({ code: 'hi', label: 'हिन्दी', speechLocale: 'hi-IN' }),
])

const LANGUAGE_CODES = new Set(LANGUAGE_OPTIONS.map((language) => language.code))

export function normalizeLanguage(value) {
  if (typeof value !== 'string') return DEFAULT_LANGUAGE
  const code = value.trim().toLowerCase().replaceAll('_', '-').split('-', 1)[0]
  return LANGUAGE_CODES.has(code) ? code : DEFAULT_LANGUAGE
}

export function languageLocale(value) {
  const language = normalizeLanguage(value)
  return LANGUAGE_OPTIONS.find((option) => option.code === language).speechLocale
}

function preferenceKey(sessionId, preference) {
  return `mystery-village:${encodeURIComponent(sessionId)}:${preference}`
}

export function loadSessionLanguage(storage, sessionId) {
  try {
    return normalizeLanguage(storage?.getItem(preferenceKey(sessionId, 'language')))
  } catch {
    return DEFAULT_LANGUAGE
  }
}

export function saveSessionLanguage(storage, sessionId, language) {
  const selectedLanguage = normalizeLanguage(language)
  try {
    storage?.setItem(preferenceKey(sessionId, 'language'), selectedLanguage)
  } catch {
    return selectedLanguage
  }
  return selectedLanguage
}

export function loadSessionVoiceEnabled(storage, sessionId) {
  try {
    return storage?.getItem(preferenceKey(sessionId, 'voice-enabled')) === 'true'
  } catch {
    return false
  }
}

export function saveSessionVoiceEnabled(storage, sessionId, enabled) {
  const voiceEnabled = enabled === true
  try {
    storage?.setItem(preferenceKey(sessionId, 'voice-enabled'), String(voiceEnabled))
  } catch {
    return voiceEnabled
  }
  return voiceEnabled
}