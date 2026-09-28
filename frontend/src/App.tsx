import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import {
  LANGUAGE_OPTIONS,
  loadSessionLanguage,
  loadSessionVoiceEnabled,
  saveSessionLanguage,
  saveSessionVoiceEnabled,
} from './language.mjs'
import type { LanguageCode } from './language.mjs'
import {
  createSpeechRecognitionController,
  getSpeechRecognitionConstructor,
  isTextToSpeechAvailable,
  speechRecognitionErrorMessage,
  speakText,
  stopSpeaking,
} from './voice.mjs'
import type {
  SpeechRecognitionController,
  SpeechRecognitionEnvironment,
} from './voice.mjs'
import {
  fetchConversation,
  fetchConversationSummaries,
  formatConversationDate,
} from './history.mjs'
import type {
  ConversationSummary,
  StoredConversationMessage,
} from './history.mjs'
import './GameWorld.css'

type NpcId = 'ravi' | 'meera'

type ChatMessage = {
  id: string
  author: 'you' | NpcId
  text: string
}

type ChatResponse = {
  message: string
}

type InventoryItem = {
  item_name: string
  quantity: number
}

type GameState = {
  session_id: string
  location: string
  health: number
  gold: number
  game_time: string
  inventory: InventoryItem[]
}

const NPC_DETAILS: Record<NpcId, { name: string; role: string; location: string }> = {
  ravi: { name: 'Ravi', role: 'Blacksmith', location: 'At the village forge' },
  meera: { name: 'Meera', role: 'Healer', location: 'At the village infirmary' },
}

type NpcCharacterProps = {
  npcId: NpcId
  details: (typeof NPC_DETAILS)[NpcId]
  isSelected: boolean
  status: string
  disabled: boolean
  onSelect: () => void
  onTalk: () => void
}

function NpcCharacter({
  npcId,
  details,
  isSelected,
  status,
  disabled,
  onSelect,
  onTalk,
}: NpcCharacterProps) {
  return (
    <article className={`npc-marker npc-marker--${npcId}${isSelected ? ' selected' : ''}`}>
      <button
        aria-label={`Select ${details.name}, ${details.role}`}
        aria-pressed={isSelected}
        className="npc-character-button"
        disabled={disabled}
        onClick={onSelect}
        type="button"
      >
        <span className={`npc-figure npc-figure--${npcId}`} aria-hidden="true">
          <span className="npc-figure-head" />
          <span className="npc-figure-body" />
          <span className="npc-figure-cloak" />
        </span>
        <span className="npc-nameplate">
          <strong>{details.name}</strong>
          <small>{status}</small>
        </span>
      </button>
      {isSelected && (
        <button className="npc-talk-button" onClick={onTalk} type="button">
          Talk <span aria-hidden="true">↗</span>
        </button>
      )}
    </article>
  )
}

function App() {
  const [sessionId] = useState(() => {
    const storageKey = 'mystery-village-session-id'
    const savedSessionId = sessionStorage.getItem(storageKey)
    if (savedSessionId) return savedSessionId

    const newSessionId = `session-${crypto.randomUUID()}`
    sessionStorage.setItem(storageKey, newSessionId)
    return newSessionId
  })
  const [language, setLanguage] = useState<LanguageCode>(() =>
    loadSessionLanguage(window.sessionStorage, sessionId),
  )
  const speechEnvironment = window as typeof window & SpeechRecognitionEnvironment
  const speechRecognitionConstructor = getSpeechRecognitionConstructor(speechEnvironment)
  const textToSpeechAvailable = isTextToSpeechAvailable(speechEnvironment)
  const [voiceEnabled, setVoiceEnabled] = useState(
    () => textToSpeechAvailable && loadSessionVoiceEnabled(window.sessionStorage, sessionId),
  )
  const voiceEnabledRef = useRef(voiceEnabled)
  const speechSequenceRef = useRef(0)
  const recognitionRef = useRef<SpeechRecognitionController | null>(null)
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [microphoneError, setMicrophoneError] = useState('')
  const [voiceError, setVoiceError] = useState('')
  const [activeNpc, setActiveNpc] = useState<NpcId>('ravi')
  const [messagesByNpc, setMessagesByNpc] = useState<Record<NpcId, ChatMessage[]>>({
    ravi: [],
    meera: [],
  })
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [gameStateError, setGameStateError] = useState(false)
  const [draft, setDraft] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')
  const [lastNpcReplyByNpc, setLastNpcReplyByNpc] = useState<
    Record<NpcId, { text: string; language: LanguageCode } | null>
  >({ ravi: null, meera: null })
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [historyItems, setHistoryItems] = useState<ConversationSummary[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState(false)
  const [loadingConversationId, setLoadingConversationId] = useState<string | null>(null)
  const [isDialogueOpen, setIsDialogueOpen] = useState(false)
  const messages = messagesByNpc[activeNpc]
  const npcDetails = NPC_DETAILS[activeNpc]
  const lastNpcReply = lastNpcReplyByNpc[activeNpc]
  const latestNpcMessage = [...messages].reverse().find((message) => message.author !== 'you')
  const latestPlayerMessage = [...messages].reverse().find((message) => message.author === 'you')

  useEffect(() => () => {
    recognitionRef.current?.stop()
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      stopSpeaking(window.speechSynthesis)
    }
  }, [])

  useEffect(() => {
    let isCurrent = true

    async function loadGameState() {
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/game-state/${encodeURIComponent(sessionId)}`,
        )
        if (!response.ok) throw new Error('Game state request failed.')
        const data = (await response.json()) as GameState
        if (isCurrent) setGameState(data)
      } catch {
        if (isCurrent) setGameStateError(true)
      }
    }

    void loadGameState()
    return () => {
      isCurrent = false
    }
  }, [sessionId])

  function handleLanguageChange(event: ChangeEvent<HTMLSelectElement>) {
    recognitionRef.current?.stop()
    recognitionRef.current = null
    setIsListening(false)
    const nextLanguage = saveSessionLanguage(
      window.sessionStorage,
      sessionId,
      event.target.value,
    )
    setLanguage(nextLanguage)
    setMicrophoneError('')
  }

  function stopSpeech() {
    speechSequenceRef.current += 1
    if (window.speechSynthesis) stopSpeaking(window.speechSynthesis)
    setIsSpeaking(false)
  }

  function playNpcReply(text: string, responseLanguage: LanguageCode) {
    if (!voiceEnabledRef.current) return
    if (!isTextToSpeechAvailable(window)) {
      setVoiceError('Voice output is not supported in this browser.')
      return
    }

    const sequence = ++speechSequenceRef.current
    const started = speakText({
      synthesis: window.speechSynthesis,
      UtteranceConstructor: window.SpeechSynthesisUtterance,
      text,
      language: responseLanguage,
      onStart: () => {
        if (speechSequenceRef.current === sequence) setIsSpeaking(true)
      },
      onEnd: () => {
        if (speechSequenceRef.current === sequence) setIsSpeaking(false)
      },
      onError: () => {
        if (speechSequenceRef.current === sequence) {
          setIsSpeaking(false)
          setVoiceError('Could not play this response aloud.')
        }
      },
    })
    if (started) {
      setIsSpeaking(true)
      setVoiceError('')
    } else {
      setIsSpeaking(false)
    }
  }

  function handleVoiceToggle() {
    const nextEnabled = !voiceEnabled
    if (nextEnabled && !textToSpeechAvailable) {
      setVoiceError('Voice output is not supported in this browser.')
      return
    }
    const savedPreference = saveSessionVoiceEnabled(
      window.sessionStorage,
      sessionId,
      nextEnabled,
    )
    voiceEnabledRef.current = savedPreference
    setVoiceEnabled(savedPreference)
    setVoiceError('')
    if (!savedPreference) stopSpeech()
  }

  function handleMicrophoneToggle() {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
      setIsListening(false)
      return
    }
    if (!speechRecognitionConstructor) {
      setMicrophoneError(speechRecognitionErrorMessage('unsupported'))
      return
    }

    const controller = createSpeechRecognitionController(
      speechRecognitionConstructor,
      language,
      {
        onTranscript: (transcript) => {
          setDraft((currentDraft) => currentDraft
            ? `${currentDraft} ${transcript}`
            : transcript)
          setMicrophoneError('')
        },
        onError: (message) => {
          setMicrophoneError(message)
          recognitionRef.current = null
          setIsListening(false)
        },
        onEnd: () => {
          setIsListening(false)
          recognitionRef.current = null
        },
      },
    )
    if (!controller) {
      setMicrophoneError(speechRecognitionErrorMessage('unsupported'))
      return
    }

    recognitionRef.current = controller
    setMicrophoneError('')
    setIsListening(true)
    if (!controller.start()) {
      recognitionRef.current = null
      setIsListening(false)
    }
  }

  async function loadHistory() {
    setHistoryLoading(true)
    setHistoryError(false)
    try {
      const summaries = await fetchConversationSummaries(fetch, sessionId)
      setHistoryItems(summaries)
    } catch {
      setHistoryError(true)
    } finally {
      setHistoryLoading(false)
    }
  }

  function openHistory() {
    setIsHistoryOpen(true)
    void loadHistory()
  }

  async function openConversation(conversationId: string) {
    setLoadingConversationId(conversationId)
    setHistoryError(false)
    try {
      const conversation = await fetchConversation(fetch, sessionId, conversationId)
      const npcId = conversation.npc_id
      const historyMessages = conversation.messages.map(
        (message: StoredConversationMessage) => ({
          id: String(message.id),
          author: message.role === 'user' ? 'you' as const : npcId,
          text: message.content,
        }),
      )
      setMessagesByNpc((currentMessages) => ({
        ...currentMessages,
        [npcId]: historyMessages,
      }))
      setActiveNpc(npcId)
      setIsHistoryOpen(false)
    } catch {
      setHistoryError(true)
    } finally {
      setLoadingConversationId(null)
    }
  }

  function closeHistory() {
    setIsHistoryOpen(false)
    setHistoryError(false)
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const message = draft.trim()
    if (!message || isSending || isListening) return
    const npcId = activeNpc
    const responseLanguage = language

    setMessagesByNpc((currentMessages) => ({
      ...currentMessages,
      [npcId]: [
        ...currentMessages[npcId],
        { id: crypto.randomUUID(), author: 'you', text: message },
      ],
    }))
    setDraft('')
    setError('')
    setIsSending(true)

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          npc_id: npcId,
          message,
          language: responseLanguage,
        }),
      })

      if (!response.ok) {
        throw new Error(`Chat request failed with status ${response.status}`)
      }

      const data = (await response.json()) as Partial<ChatResponse>
      if (typeof data.message !== 'string') {
        throw new Error('The backend returned an invalid chat response.')
      }

      const replyMessage = data.message
      setLastNpcReplyByNpc((currentReplies) => ({
        ...currentReplies,
        [npcId]: { text: replyMessage, language: responseLanguage },
      }))
      setMessagesByNpc((currentMessages) => ({
        ...currentMessages,
        [npcId]: [
          ...currentMessages[npcId],
          { id: crypto.randomUUID(), author: npcId, text: replyMessage },
        ],
      }))
      playNpcReply(replyMessage, responseLanguage)

      try {
        const stateResponse = await fetch(
          `http://127.0.0.1:8000/game-state/${encodeURIComponent(sessionId)}`,
        )
        if (stateResponse.ok) {
          setGameState((await stateResponse.json()) as GameState)
          setGameStateError(false)
        }
      } catch {
        setGameStateError(true)
      }
    } catch {
      setError('Could not get a reply from the backend. Please try again.')
    } finally {
      setIsSending(false)
    }
  }

  return (
    <main className="game-shell">
      <section className="world-scene" aria-label="Mystery Village game world">
        <div className="world-backdrop" aria-hidden="true" />
        <div className="world-atmosphere" aria-hidden="true" />
        <div className="village-landmarks" aria-hidden="true">
          <div className="landmark landmark--inn"><span>THE INN</span></div>
          <div className="landmark landmark--forge"><span>FORGE</span></div>
          <div className="landmark landmark--healer"><span>HEALER</span></div>
          <div className="tree tree--one" />
          <div className="tree tree--two" />
          <div className="tree tree--three" />
          <div className="village-path" />
        </div>

        <header className="player-hud">
          <div className="hud-world-mark">
            <span className="world-mark-icon" aria-hidden="true">✦</span>
            <span className="world-mark-copy"><strong>Mystery Village</strong><small>THE OLD ROAD</small></span>
          </div>
          <div className="hud-location" aria-live="polite">
            <span className="hud-kicker">You are in</span>
            <strong>{gameState?.location ?? (gameStateError ? 'Location unavailable' : 'Finding your way...')}</strong>
            <span className="hud-time">{gameState?.game_time ?? ''}</span>
          </div>
          <div className="hud-actions" aria-label="World controls">
            <label className="language-control">
              <span className="visually-hidden">Response language</span>
              <select
                aria-label="Response language"
                value={language}
                onChange={handleLanguageChange}
              >
                {LANGUAGE_OPTIONS.map((option) => (
                  <option key={option.code} value={option.code}>{option.label}</option>
                ))}
              </select>
            </label>
            <button
              aria-label="Open chat history"
              className="hud-button"
              disabled={isSending}
              onClick={openHistory}
              type="button"
            >
              <span aria-hidden="true">▤</span><span>Journal</span>
            </button>
            <button
              aria-pressed={voiceEnabled}
              className={voiceEnabled ? 'hud-button voice-toggle enabled' : 'hud-button voice-toggle'}
              onClick={handleVoiceToggle}
              type="button"
            >
              <span aria-hidden="true">♫</span><span>Voice {voiceEnabled ? 'ON' : 'OFF'}</span>
            </button>
          </div>
        </header>

        <div className="world-caption" aria-live="polite">
          <span className="caption-rule" />
          <span>{gameState?.location ?? 'A village on the old road'}</span>
        </div>

        <div className="npc-population" aria-label="Village characters">
          {(Object.keys(NPC_DETAILS) as NpcId[]).map((npcId) => {
            const isActive = activeNpc === npcId
            const status = isSending && isActive
              ? 'Busy'
              : isDialogueOpen && isActive
                ? 'Talking'
                : NPC_DETAILS[npcId].location
            return (
              <NpcCharacter
                details={NPC_DETAILS[npcId]}
                disabled={isSending}
                isSelected={isActive}
                key={npcId}
                npcId={npcId}
                onSelect={() => {
                  setActiveNpc(npcId)
                  setError('')
                }}
                onTalk={() => {
                  setActiveNpc(npcId)
                  setIsDialogueOpen(true)
                  setError('')
                }}
                status={status}
              />
            )
          })}
        </div>

        {gameState && (
          <aside className="player-vitals" aria-label="Player status" aria-live="polite">
            <span className="vitals-label">TRAVELER</span>
            <div className="vital-item"><span aria-hidden="true">♥</span><strong>{gameState.health}</strong></div>
            <div className="vital-item"><span aria-hidden="true">◈</span><strong>{gameState.gold}</strong></div>
            <span className="inventory-summary" title={gameState.inventory.length
              ? gameState.inventory.map((item) => `${item.item_name} x${item.quantity}`).join(', ')
              : 'No items'}>
              {gameState.inventory.length
                ? gameState.inventory.map((item) => `${item.item_name} ×${item.quantity}`).join(' · ')
                : 'No items'}
            </span>
          </aside>
        )}
        {!gameState && gameStateError && (
          <p className="world-state-error" role="status">Player state is unavailable</p>
        )}

        {isDialogueOpen && (
          <section className="dialogue-panel" aria-label={`Dialogue with ${npcDetails.name}`}>
            <header className="dialogue-header">
              <div className={`dialogue-portrait dialogue-portrait--${activeNpc}`} aria-hidden="true">
                <span className="portrait-initial">{npcDetails.name[0]}</span>
              </div>
              <div className="dialogue-identity">
                <span className="dialogue-role">{npcDetails.role}</span>
                <h1>{npcDetails.name}</h1>
                <p><span className="status-light" />{isSending ? 'Busy' : isSpeaking ? 'Speaking' : 'Here with you'}</p>
              </div>
              <div className="dialogue-actions">
                {lastNpcReply && (
                  <button
                    aria-label="Play latest NPC response"
                    className="dialogue-icon-button"
                    disabled={!voiceEnabled || !textToSpeechAvailable}
                    onClick={() => playNpcReply(lastNpcReply.text, lastNpcReply.language)}
                    title="Play reply"
                    type="button"
                  >
                    <span aria-hidden="true">♫</span>
                  </button>
                )}
                <button
                  aria-label="Stop speech output"
                  className="dialogue-icon-button"
                  disabled={!isSpeaking}
                  onClick={stopSpeech}
                  title="Stop speech"
                  type="button"
                >
                  <span aria-hidden="true">■</span>
                </button>
                <button
                  aria-label="Close dialogue"
                  className="dialogue-icon-button dialogue-close"
                  onClick={() => setIsDialogueOpen(false)}
                  title="Close dialogue"
                  type="button"
                >
                  <span aria-hidden="true">×</span>
                </button>
              </div>
            </header>

            <div className="dialogue-copy" role="log" aria-live="polite" aria-relevant="additions">
              {latestPlayerMessage && (
                <p className="player-utterance"><span aria-hidden="true">“</span>{latestPlayerMessage.text}</p>
              )}
              {latestNpcMessage ? (
                <p className="npc-utterance">{latestNpcMessage.text}</p>
              ) : (
                <p className="npc-utterance npc-utterance--quiet">
                  {isSending ? `${npcDetails.name} is gathering a reply...` : `${npcDetails.name} turns toward you.`}
                </p>
              )}
            </div>

            <div className="composer-wrap">
              {error && <p className="error-message" role="alert">{error}</p>}
              {microphoneError && <p className="error-message" role="alert">{microphoneError}</p>}
              {voiceError && <p className="voice-status" role="status">{voiceError}</p>}
              {!speechRecognitionConstructor && !microphoneError && (
                <p className="voice-status" role="status">Speech input is unavailable in this browser. You can type instead.</p>
              )}
              {isListening && (
                <p className="voice-status listening-status" role="status">
                  <span className="listening-pulse" /> Listening. Your words will appear here before sending.
                </p>
              )}
              <form className="composer" onSubmit={sendMessage}>
                <label className="visually-hidden" htmlFor="chat-input">Message {npcDetails.name}</label>
                <button
                  aria-label={isListening ? 'Stop speech recognition' : 'Start speech recognition'}
                  aria-pressed={isListening}
                  className={isListening ? 'microphone-button listening' : 'microphone-button'}
                  disabled={isSending || (!speechRecognitionConstructor && !isListening)}
                  onClick={handleMicrophoneToggle}
                  title={isListening ? 'Stop listening' : 'Use microphone'}
                  type="button"
                >
                  <span aria-hidden="true">{isListening ? '■' : '◉'}</span>
                  <span>{isListening ? 'Listening' : 'Speak'}</span>
                </button>
                <input
                  id="chat-input"
                  type="text"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder={language === 'te'
                    ? `${npcDetails.name}కు సందేశం...`
                    : language === 'hi'
                      ? `${npcDetails.name} को संदेश...`
                      : `Say something to ${npcDetails.name}...`}
                  autoComplete="off"
                  disabled={isSending}
                />
                <button
                  className="send-button"
                  type="submit"
                  disabled={isSending || isListening || !draft.trim()}
                  aria-label={isSending ? 'Sending message' : 'Send message'}
                >
                  <span>{isSending ? '...' : 'Send'}</span><span aria-hidden="true">↗</span>
                </button>
              </form>
            </div>
          </section>
        )}

        {isHistoryOpen && (
          <div
            className="history-overlay"
            role="presentation"
            onMouseDown={(event) => {
              if (event.target === event.currentTarget) closeHistory()
            }}
            onKeyDown={(event) => {
              if (event.key === 'Escape') closeHistory()
            }}
          >
            <section
              aria-labelledby="history-heading"
              aria-modal="true"
              className="history-dialog"
              role="dialog"
            >
              <header className="history-header">
                <div>
                  <p className="journal-kicker">FIELD NOTES · THIS SESSION</p>
                  <h2 id="history-heading">Journal</h2>
                </div>
                <button
                  aria-label="Close chat history"
                  className="history-close"
                  onClick={closeHistory}
                  type="button"
                >
                  <span aria-hidden="true">×</span>
                </button>
              </header>
              {historyError && (
                <div className="history-error" role="alert">
                  <p>Unable to load conversation history. Please try again.</p>
                  <button className="history-retry" onClick={() => void loadHistory()} type="button">
                    Retry
                  </button>
                </div>
              )}
              {historyLoading ? (
                <p className="history-status" role="status">Loading conversation history...</p>
              ) : historyItems.length === 0 && !historyError ? (
                <p className="history-empty">No conversation history yet.</p>
              ) : (
                <div className="history-list" aria-label="Previous conversations">
                  {historyItems.map((item) => (
                    <button
                      className="history-item"
                      disabled={loadingConversationId !== null}
                      key={item.conversation_id}
                      onClick={() => void openConversation(item.conversation_id)}
                      type="button"
                    >
                      <span className="history-item-heading">
                        <strong>{item.npc_name || item.title}</strong>
                        <small>{item.message_count} entries</small>
                      </span>
                      <time dateTime={item.updated_at}>
                        {formatConversationDate(item.updated_at, language)}
                      </time>
                      <span className="history-preview">{item.preview}</span>
                      {loadingConversationId === item.conversation_id && (
                        <span className="history-status" role="status">Restoring conversation...</span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </section>
          </div>
        )}
      </section>
    </main>
  )
}

export default App
