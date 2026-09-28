const API_BASE_URL = 'http://127.0.0.1:8000'
const NPC_IDS = new Set(['ravi', 'meera'])

async function readJson(response) {
  if (!response.ok) throw new Error('History request failed.')
  return response.json()
}

export async function fetchConversationSummaries(fetcher, sessionId) {
  const response = await fetcher(
    `${API_BASE_URL}/history/${encodeURIComponent(sessionId)}`,
  )
  const summaries = await readJson(response)
  if (!Array.isArray(summaries)) throw new Error('History response is invalid.')
  return summaries
}

export async function fetchConversation(fetcher, sessionId, conversationId) {
  if (!NPC_IDS.has(conversationId)) throw new Error('Conversation is invalid.')
  const response = await fetcher(
    `${API_BASE_URL}/history/${encodeURIComponent(sessionId)}/${encodeURIComponent(conversationId)}`,
  )
  const conversation = await readJson(response)
  if (
    !conversation
    || conversation.conversation_id !== conversationId
    || conversation.npc_id !== conversationId
    || !Array.isArray(conversation.messages)
  ) {
    throw new Error('Conversation response is invalid.')
  }
  return conversation
}

export function formatConversationDate(value, language) {
  if (typeof value !== 'string' || !value.trim()) return ''
  const dateValue = value.includes('T') ? value : `${value.replace(' ', 'T')}Z`
  const date = new Date(dateValue)
  if (Number.isNaN(date.getTime())) return value
  const locale = { en: 'en-US', te: 'te-IN', hi: 'hi-IN' }[language] ?? 'en-US'
  return new Intl.DateTimeFormat(locale, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}