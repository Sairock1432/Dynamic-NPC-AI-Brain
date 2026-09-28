export type LanguageCode = 'en' | 'te' | 'hi'

export const DEFAULT_LANGUAGE: LanguageCode
export const LANGUAGE_OPTIONS: readonly {
  readonly code: LanguageCode
  readonly label: string
  readonly speechLocale: string
}[]

export function normalizeLanguage(value: unknown): LanguageCode
export function languageLocale(value: unknown): string
export function loadSessionLanguage(storage: Storage | undefined, sessionId: string): LanguageCode
export function saveSessionLanguage(
  storage: Storage | undefined,
  sessionId: string,
  language: unknown,
): LanguageCode
export function loadSessionVoiceEnabled(storage: Storage | undefined, sessionId: string): boolean
export function saveSessionVoiceEnabled(
  storage: Storage | undefined,
  sessionId: string,
  enabled: boolean,
): boolean