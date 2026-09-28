export interface ConversationSummary {
  conversation_id: string
  npc_id: 'ravi' | 'meera'
  npc_name: string
  title: string
  message_count: number
  started_at: string
  updated_at: string
  preview: string
}

export interface StoredConversationMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export interface StoredConversation {
  conversation_id: 'ravi' | 'meera'
  npc_id: 'ravi' | 'meera'
  npc_name: string
  messages: StoredConversationMessage[]
}

export type HistoryFetcher = (url: string) => Promise<{
  ok: boolean
  json(): Promise<unknown>
}>

export function fetchConversationSummaries(
  fetcher: HistoryFetcher,
  sessionId: string,
): Promise<ConversationSummary[]>
export function fetchConversation(
  fetcher: HistoryFetcher,
  sessionId: string,
  conversationId: string,
): Promise<StoredConversation>
export function formatConversationDate(value: string, language: string): string