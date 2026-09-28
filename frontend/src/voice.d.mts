import type { LanguageCode } from './language.mjs'

export interface SpeechRecognitionResultLike {
  readonly isFinal?: boolean
  readonly 0?: { readonly transcript?: string }
}

export interface SpeechRecognitionEventLike {
  readonly resultIndex?: number
  readonly results?: ArrayLike<SpeechRecognitionResultLike>
}

export interface SpeechRecognitionLike {
  lang: string
  interimResults: boolean
  continuous: boolean
  maxAlternatives: number
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: ((event: { error?: string }) => void) | null
  onend: (() => void) | null
  start(): void
  stop(): void
}

export type SpeechRecognitionConstructor = new () => SpeechRecognitionLike
export interface SpeechRecognitionEnvironment {
  SpeechRecognition?: SpeechRecognitionConstructor
  webkitSpeechRecognition?: SpeechRecognitionConstructor
}
export interface SpeechRecognitionController {
  start(): boolean
  stop(): void
}
export interface SpeechRecognitionCallbacks {
  onTranscript?: (transcript: string) => void
  onError?: (message: string) => void
  onEnd?: () => void
}

export function getSpeechRecognitionConstructor(
  environment: SpeechRecognitionEnvironment,
): SpeechRecognitionConstructor | null
export function speechRecognitionErrorMessage(errorCode: string): string
export function extractSpeechTranscript(event: SpeechRecognitionEventLike): string
export function createSpeechRecognitionController(
  Constructor: SpeechRecognitionConstructor | null | undefined,
  language: LanguageCode,
  callbacks?: SpeechRecognitionCallbacks,
): SpeechRecognitionController | null
export function selectSpeechVoice<T extends SpeechSynthesisVoice>(
  voices: readonly T[],
  language: LanguageCode,
): T | undefined
export function isTextToSpeechAvailable(environment: {
  speechSynthesis?: Pick<SpeechSynthesis, 'speak' | 'cancel'>
  SpeechSynthesisUtterance?: unknown
}): boolean
export function speakText(options: {
  synthesis: Pick<SpeechSynthesis, 'speak' | 'cancel' | 'getVoices'> | undefined
  UtteranceConstructor: new (text: string) => SpeechSynthesisUtterance
  text: string
  language: LanguageCode
  onStart?: () => void
  onEnd?: () => void
  onError?: (error: string) => void
}): boolean
export function stopSpeaking(synthesis: Pick<SpeechSynthesis, 'cancel'> | undefined): boolean