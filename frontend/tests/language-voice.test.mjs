import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import {
  DEFAULT_LANGUAGE,
  LANGUAGE_OPTIONS,
  languageLocale,
  loadSessionLanguage,
  loadSessionVoiceEnabled,
  normalizeLanguage,
  saveSessionLanguage,
  saveSessionVoiceEnabled,
} from '../src/language.mjs'
import {
  createSpeechRecognitionController,
  extractSpeechTranscript,
  getSpeechRecognitionConstructor,
  isTextToSpeechAvailable,
  selectSpeechVoice,
  speakText,
  speechRecognitionErrorMessage,
  stopSpeaking,
} from '../src/voice.mjs'
import {
  fetchConversation,
  fetchConversationSummaries,
  formatConversationDate,
} from '../src/history.mjs'

class MemoryStorage {
  values = new Map()

  getItem(key) {
    return this.values.get(key) ?? null
  }

  setItem(key, value) {
    this.values.set(key, String(value))
  }
}

test('language selector options include English, Telugu, and Hindi', () => {
  assert.deepEqual(LANGUAGE_OPTIONS.map(({ code }) => code), ['en', 'te', 'hi'])
  assert.deepEqual(LANGUAGE_OPTIONS.map(({ label }) => label), ['English', 'తెలుగు', 'हिन्दी'])
  assert.equal(DEFAULT_LANGUAGE, 'en')
})

test('language normalization accepts supported locales and falls back to English', () => {
  assert.equal(normalizeLanguage('en'), 'en')
  assert.equal(normalizeLanguage('te-IN'), 'te')
  assert.equal(normalizeLanguage('hi_IN'), 'hi')
  assert.equal(normalizeLanguage('fr'), 'en')
  assert.equal(normalizeLanguage(null), 'en')
  assert.equal(languageLocale('te'), 'te-IN')
  assert.equal(languageLocale('hi'), 'hi-IN')
})

test('language and voice preferences are scoped to the current session', () => {
  const storage = new MemoryStorage()
  saveSessionLanguage(storage, 'session-a', 'te')
  saveSessionVoiceEnabled(storage, 'session-a', true)

  assert.equal(loadSessionLanguage(storage, 'session-a'), 'te')
  assert.equal(loadSessionVoiceEnabled(storage, 'session-a'), true)
  assert.equal(loadSessionLanguage(storage, 'session-b'), 'en')
  assert.equal(loadSessionVoiceEnabled(storage, 'session-b'), false)
  assert.equal(saveSessionLanguage(storage, 'session-a', 'xx'), 'en')
  assert.equal(loadSessionLanguage(storage, 'session-a'), 'en')
})

test('microphone support resolves standard and prefixed browser APIs', () => {
  class StandardRecognition {}
  class PrefixedRecognition {}
  assert.equal(getSpeechRecognitionConstructor({ SpeechRecognition: StandardRecognition }), StandardRecognition)
  assert.equal(getSpeechRecognitionConstructor({ webkitSpeechRecognition: PrefixedRecognition }), PrefixedRecognition)
  assert.equal(getSpeechRecognitionConstructor({}), null)
  assert.equal(createSpeechRecognitionController(null, 'en'), null)
})

test('speech recognition uses the selected language and returns reviewable text only', () => {
  let recognizer
  const transcripts = []
  const errors = []
  const ends = []
  class FakeRecognition {
    constructor() {
      recognizer = this
    }

    start() {}
    stop() {}
  }

  const controller = createSpeechRecognitionController(FakeRecognition, 'te', {
    onTranscript: (transcript) => transcripts.push(transcript),
    onError: (message) => errors.push(message),
    onEnd: () => ends.push(true),
  })
  assert.equal(controller.start(), true)
  assert.equal(recognizer.lang, 'te-IN')
  assert.equal(recognizer.interimResults, false)
  recognizer.onresult({
    results: [[{ transcript: '  మీరు ఎలా ఉన్నారు?  ', isFinal: true }]],
  })
  assert.deepEqual(transcripts, ['మీరు ఎలా ఉన్నారు?'])
  assert.deepEqual(ends, [])
  controller.stop()
})

test('microphone permission denial and empty speech are handled without sending', () => {
  const errors = []
  const transcripts = []
  let recognizer
  class FakeRecognition {
    constructor() {
      recognizer = this
    }

    start() {}
    stop() {}
  }
  const controller = createSpeechRecognitionController(FakeRecognition, 'hi', {
    onTranscript: (value) => transcripts.push(value),
    onError: (message) => errors.push(message),
  })
  controller.start()
  recognizer.onerror({ error: 'not-allowed' })
  recognizer.onresult({ results: [[{ transcript: '  ', isFinal: true }]] })

  assert.match(errors[0], /permission was denied/i)
  assert.match(errors[1], /no speech was recognized/i)
  assert.deepEqual(transcripts, [])
  assert.equal(extractSpeechTranscript({ results: [] }), '')
  assert.match(speechRecognitionErrorMessage('audio-capture'), /no microphone/i)
})

test('speech voices prefer the selected locale and fall back safely', () => {
  const voices = [
    { lang: 'en-US', name: 'English', default: true },
    { lang: 'hi-IN', name: 'Hindi', default: false },
  ]
  assert.equal(selectSpeechVoice(voices, 'hi').name, 'Hindi')
  assert.equal(selectSpeechVoice(voices, 'te').name, 'English')
  assert.equal(selectSpeechVoice([], 'te'), undefined)
})

test('text-to-speech cancels overlap and speaks with the selected locale', () => {
  const calls = []
  let utterance
  class FakeUtterance {
    constructor(text) {
      this.text = text
      utterance = this
    }
  }
  const voice = { lang: 'hi-IN', name: 'Hindi', default: false }
  const synthesis = {
    cancel: () => calls.push('cancel'),
    getVoices: () => [voice],
    speak: (value) => calls.push(value),
  }

  assert.equal(isTextToSpeechAvailable({
    speechSynthesis: synthesis,
    SpeechSynthesisUtterance: FakeUtterance,
  }), true)
  assert.equal(speakText({
    synthesis,
    UtteranceConstructor: FakeUtterance,
    text: 'आप कैसे हैं?',
    language: 'hi',
  }), true)
  assert.deepEqual(calls, ['cancel', utterance])
  assert.equal(utterance.lang, 'hi-IN')
  assert.equal(utterance.voice, voice)
  assert.equal(stopSpeaking(synthesis), true)
  assert.equal(calls.at(-1), 'cancel')
})

test('speech playback fails gracefully when unavailable or given empty text', () => {
  const errors = []
  assert.equal(isTextToSpeechAvailable({}), false)
  assert.equal(speakText({
    synthesis: undefined,
    UtteranceConstructor: undefined,
    text: 'Hello',
    language: 'en',
    onError: (error) => errors.push(error),
  }), false)
  assert.equal(speakText({
    synthesis: { cancel() {}, speak() {} },
    UtteranceConstructor: class {},
    text: '  ',
    language: 'en',
    onError: (error) => errors.push(error),
  }), false)
  assert.deepEqual(errors, ['unsupported', 'empty'])
})

test('chat UI wires a language selector, microphone review, and voice controls', () => {
  const appSource = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8')
  assert.match(appSource, /aria-label="Response language"/)
  assert.match(appSource, /Start speech recognition/)
  assert.match(appSource, /aria-pressed=\{voiceEnabled\}/)
  assert.match(appSource, /onTranscript:\s*\(transcript\)\s*=>\s*\{[\s\S]*?setDraft\(/)
  assert.match(appSource, /aria-label="Stop speech output"/)
  assert.doesNotMatch(appSource, /onTranscript[^\n]*sendMessage/)
  assert.match(appSource, /language: responseLanguage/)
})

test('game world presents selectable NPCs and compact dialogue instead of a chat panel', () => {
  const appSource = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8')
  assert.match(appSource, /className="world-scene"/)
  assert.match(appSource, /function NpcCharacter/)
  assert.match(appSource, /className="npc-talk-button"/)
  assert.match(appSource, /className="dialogue-panel"/)
  assert.doesNotMatch(appSource, /className="message-list"/)
  assert.doesNotMatch(appSource, /fetch\([^)]*(?:movement|position|relationship)/i)
})

test('history fetches are scoped to the existing session and keep stored text intact', async () => {
  const requests = []
  const fetcher = async (url) => {
    requests.push(url)
    return {
      ok: true,
      json: async () => url.endsWith('/ravi')
        ? {
            conversation_id: 'ravi',
            npc_id: 'ravi',
            npc_name: 'Ravi',
            messages: [
              { id: 1, role: 'user', content: 'నమస్కారం', created_at: '2026-09-26 08:00:00' },
              { id: 2, role: 'assistant', content: 'రవి సమాధానం', created_at: '2026-09-26 08:00:01' },
            ],
          }
        : [{
            conversation_id: 'ravi',
            npc_id: 'ravi',
            npc_name: 'Ravi',
            title: 'Conversation with Ravi',
            message_count: 2,
            started_at: '2026-09-26 08:00:00',
            updated_at: '2026-09-26 08:00:01',
            preview: 'నమస్కారం',
          }],
    }
  }

  const summaries = await fetchConversationSummaries(fetcher, 'session a')
  const conversation = await fetchConversation(fetcher, 'session a', 'ravi')
  assert.equal(requests[0], 'http://127.0.0.1:8000/history/session%20a')
  assert.equal(requests[1], 'http://127.0.0.1:8000/history/session%20a/ravi')
  assert.equal(summaries[0].preview, 'నమస్కారం')
  assert.deepEqual(conversation.messages.map((message) => message.content), [
    'నమస్కారం',
    'రవి సమాధానం',
  ])
})

test('history handles empty lists, failed requests, and invalid transcript data', async () => {
  const emptyFetcher = async () => ({ ok: true, json: async () => [] })
  assert.deepEqual(await fetchConversationSummaries(emptyFetcher, 'session-a'), [])
  await assert.rejects(
    fetchConversationSummaries(async () => ({ ok: false, json: async () => ({ detail: 'private' }) }), 'session-a'),
    /History request failed/,
  )
  await assert.rejects(
    fetchConversation(async () => ({ ok: true, json: async () => ({}) }), 'session-a', 'ravi'),
    /Conversation response is invalid/,
  )
  await assert.rejects(
    fetchConversation(emptyFetcher, 'session-a', 'unknown'),
    /Conversation is invalid/,
  )
})

test('history dates use the selected locale and preserve unknown timestamps', () => {
  assert.equal(
    formatConversationDate('not-a-date', 'te'),
    'not-a-date',
  )
  assert.notEqual(
    formatConversationDate('2026-09-26 08:00:00', 'hi'),
    '2026-09-26 08:00:00',
  )
})

test('history UI has button, panel states, transcript selection, and preserves current chat on close', () => {
  const appSource = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8')
  assert.match(appSource, /aria-label="Open chat history"/)
  assert.match(appSource, /setIsHistoryOpen\(true\)/)
  assert.match(appSource, /aria-labelledby="history-heading"/)
  assert.match(appSource, /Loading conversation history\.\.\./)
  assert.match(appSource, /No conversation history yet\./)
  assert.match(appSource, /Unable to load conversation history\. Please try again\./)
  assert.match(appSource, /onClick=\{\(\) => void openConversation\(item\.conversation_id\)\}/)
  const openConversationBody = appSource.slice(
    appSource.indexOf('async function openConversation'),
    appSource.indexOf('function closeHistory'),
  )
  assert.doesNotMatch(openConversationBody, /playNpcReply\(/)
  const closeHistoryBody = appSource.slice(
    appSource.indexOf('function closeHistory'),
    appSource.indexOf('async function sendMessage'),
  )
  assert.doesNotMatch(closeHistoryBody, /setMessagesByNpc|setLanguage|setActiveNpc/)
  assert.match(appSource, /language: responseLanguage/)
  assert.match(appSource, /Start speech recognition/)
  assert.match(appSource, /Voice \{voiceEnabled \? 'ON' : 'OFF'\}/)
})