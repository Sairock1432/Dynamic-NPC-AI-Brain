# Mystery Village Backend

FastAPI backend for the Mystery Village NPCs. Ravi is the friendly, hardworking,
proud village blacksmith, with knowledge of metalwork, weapons, armor, and
village life. Meera is the calm, caring, observant healer, with knowledge of
herbs, village healing, basic medieval medicine, health, and village life.
Both are helpful to respectful travelers and remain in the medieval setting.
Their fixed personas live in `backend/app/npcs.py` and are selected by
allowlisted NPC IDs; request text is never used to build a persona.

Conversation history, player state, and inventory are stored separately in
SQLite at `backend/data/mystery_village.db`. Conversation history is isolated
by both `session_id` and `npc_id`; pre-existing rows are assigned to Ravi by an
additive migration. SQLite remains the source of truth for conversation
history. Curated long-term facts are stored separately in local ChromaDB at
`backend/data/chroma/`, with retrieval filtered by both IDs. Both stores are
backend-only.

## Configure the Gemini API key

Create `backend/.env` locally with your Gemini API key. Keep the real `.env`
file private; it is ignored by Git. Do not put the key in frontend code.

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Install dependencies

```powershell
py -3.10 -m venv backend/venv
backend/venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

## Run

```powershell
cd backend
venv/Scripts/python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The health check is available at <http://127.0.0.1:8000/health>. Interactive
API documentation is available at <http://127.0.0.1:8000/docs>. In a second
terminal, start the React frontend:

```powershell
cd frontend
npm install
npm run dev
```

The NPC selector displays the active character and sends its configured ID with
each chat request. The backend loads that NPC's conversation and long-term
memories independently, so switching away and back preserves each history
without cross-NPC retrieval. For example:

```json
{"session_id":"session-example","npc_id":"ravi","message":"Hello Ravi"}
```

Use `"npc_id":"meera"` to address Meera. `npc_id` is required and only
configured IDs are accepted; unknown IDs receive a 400 response. Only the
selected NPC's persona is sent as system instructions. The existing
`GET /health`, `GET /game-state/{session_id}`, and `GET /inventory/{session_id}`
routes remain available.
Read the current player state at `GET /game-state/{session_id}` and inventory at
`GET /inventory/{session_id}`. Game state and inventory have no HTTP mutation
endpoints for direct client changes.

## Language and Browser Voice

The `/chat` request accepts an optional `language` code: `en`, `te`, or `hi`.
Missing or unsupported values resolve to English. The frontend keeps the
selection in `sessionStorage` alongside the existing browser session ID and
sends it on every chat request, so changing language does not create a new
conversation, NPC memory scope, or relationship.

NPC personas, memories, relationship state, plans, and game state remain
language-independent. The backend adds one fixed language instruction to the
existing Gemini prompt; it does not create translated personas or translate
memory records. The language also passes through the existing bounded peer
consultation path.

Microphone input and speech output use the browser Web Speech APIs only. Speech
recognition places finalized text in the editable chat field and never submits
automatically; microphone/audio data is not sent to or stored by the backend.
Recognition is unavailable in browsers that do not implement
`SpeechRecognition` or `webkitSpeechRecognition`; text input remains available.
Voice output is controlled by a session-scoped toggle, cancels overlapping
speech, and prefers `en-US`, `te-IN`, or `hi-IN` voices, falling back to the
browser default. If speech synthesis or the requested voice is unavailable,
chat text continues to work normally.

## NPC Relationships and Communication

The relationship registry is centralized in `backend/app/npcs.py`. Ravi knows
Meera as the trusted village healer; Meera knows Ravi as the trusted village
blacksmith who maintains tools and equipment. User messages cannot change these
definitions.

During `/chat`, the selected NPC may use the internal
`request_npc_message` tool to consult a configured relationship. The backend
validates both NPC IDs, the relationship, session ID, and a non-empty message
of at most 500 characters. The recipient gets its own persona and relationship
context, the explicit question, and read-only current game state/inventory. It
does not get either NPC's SQLite conversation history or Chroma memories, and
the recipient has no tools, so it cannot start another consultation or mutate
game state. Only the selected NPC receives the result and generates the final
player-facing response.

At most one consultation is accepted per player request. The existing Gemini
tool-round limit also remains in force. Communication is transient and is not
stored in SQLite; no public NPC-message endpoint is exposed. For example, a
player can ask Ravi for medicine advice; Ravi can consult Meera, then explain
her answer without giving Meera access to Ravi's private conversation or
memories.

## Event-Driven NPC Behavior

`backend/app/services/world_events.py` defines the only accepted events:
`morning`, `afternoon`, `evening`, `player_low_health`, and `village_event`.
Backend behavior and schedule definitions live in
`backend/app/services/npc_behavior.py`; call `dispatch_world_event(session_id,
event_name)` internally when an explicit world event occurs. There is no public
event endpoint or background agent loop.

NPC schedules are deterministic and use the existing `player_state.game_time`;
events do not create or advance a separate clock. Ravi is at the forge from
08:00–12:00 and 14:00–18:00, in the village from 12:00–14:00, and home outside
those hours. Meera is at the healer hut from 08:00–12:00 and 14:00–18:00, in
the village from 12:00–14:00, and at the healer hut outside those hours.

For example, `dispatch_world_event(session_id, "player_low_health")` runs only
when health is below 40. Ravi may consult Meera through the existing bounded
communication service, and Meera can offer support, but neither can heal or
otherwise mutate state. Event reactions are not written to conversation or
long-term memory. Low-health state remains governed by the existing allowlisted
`heal_player` action.

## NPC Goals and Decisions

NPC goals are immutable configuration in
`backend/app/services/npc_goals.py`: Ravi maintains the forge, helps travelers
with blacksmithing, and maintains metal/wood supplies; Meera protects health,
helps injured travelers, and maintains medicine/herb supplies. Goal priority is
deterministic; goal state is not persisted or writable by Gemini.

`backend/app/services/npc_decision.py` selects only allowlisted decisions from
the configured goal, event, read-only state/inventory, schedule, and centralized
relationship trust/cooperation values. For example, the morning event keeps
Ravi at the forge and Meera at the healer hut. Below 40 health, Meera's decision
is `offer_help`; Ravi may request advice from Meera using the existing
`request_npc_message` service and its one-consultation validation.

At most two decisions run in one event. Gemini receives the backend-selected
decision and goal alongside relevant event, relationship, schedule, and
read-only game context to produce a short narrative reaction. Event generation
has no tools, cannot choose actions or mutate state, and receives no player
history or semantic memories. Event calls are explicit internal dispatches,
not an autonomous background scheduler. Existing game actions remain the only
way to change health, gold, or inventory.

## NPC Learning and Privacy

Successful player chats may add one low-risk, validated learning fact to the
existing Chroma collection. Supported categories are `PLAYER_PREFERENCE`,
`PLAYER_INTEREST`, `NPC_INTERACTION_PATTERN`, and `PLAYER_ACCEPTED_HELP`.
Learning is restricted to known village topics and explicit positive signals;
sensitive topics, credentials, contact details, raw messages, event results,
tool traces, and NPC-to-NPC messages are not learned.

Memories remain scoped to `(session_id, npc_id)`. Exact duplicates are skipped
or classified in place, each interaction adds at most one memory, and each
session/NPC pair has a maximum of 20 categorized learning memories. Existing
conservative semantic-memory extraction remains in place. No public
memory-management endpoint is provided.

Categorized preferences can influence only fixed morning decisions such as
`offer_sword_information` for Ravi or `offer_herb_information` for Meera. The
decision allowlist remains backend-owned. Gemini receives only the selected
decision and a fixed topic label to narrate; it cannot create memories, select
decisions, change relationships, or mutate game state. Peer consultations
continue to receive only the explicit request, never either NPC's private
learning or conversation history.

The Ravi/Meera trust and cooperation configuration remains fixed. The bounded
relationship progression and its transition table live in
`backend/app/services/npc_relationship_progression.py`; the NPC registry and
central relationship configuration remain authoritative. No arbitrary
relationship scores or goal state are stored.

## Step 17: Bounded Relationship Progression

The backend derives a player's relationship context for one configured NPC
from that same `(session_id, npc_id)`'s existing conversation history and
categorized learning memories. It accepts only canonical `PLAYER_ACCEPTED_HELP`
facts for that NPC; repeated positive interaction requires at least two
explicit positive acknowledgments in that NPC's scoped conversation history.
Repeated-topic-interest memories alone do not advance a relationship. Player
preferences and interests are separately checked against the topic allowlist
in the centralized NPC registry. The existing SQLite/Chroma filters and NPC
registry remain the source of scope and identity. Step 17 itself introduced no
relationship persistence; Step 19 below adds a bounded session/NPC trust score
in SQLite. No new memory store, background job, or public relationship endpoint
is added. Relationship metadata is never returned to the frontend.

The only stages are `NEUTRAL`, `FRIENDLY`, and `TRUSTING`. From `NEUTRAL`,
successful help, repeated positive interaction, or successful NPC-to-NPC
communication advances to `FRIENDLY`. From `FRIENDLY`, successful help keeps
the stage `FRIENDLY`, while another repeated positive interaction or successful
NPC communication advances it to `TRUSTING`. `TRUSTING` is absorbing. Unknown
stages, events, and transitions are rejected. Stage evaluation is reconstructed
from scoped positive conversation signals and learning facts on demand.
Successful help and communication during an event affect only that dispatch;
they are not written as relationship data.

Backend rules may select `greet_warmly`, `offer_help`,
`share_basic_information`, `request_assistance`, or `acknowledge_trust`. A
relevant, validated preference can select `share_basic_information`; friendly
NPCs greet warmly, offer help during a low-health event, or request assistance
only when the registry configures a trusted healer; trusting NPCs acknowledge
trust. Meera may offer help during a low-health event even while neutral. These
are narration decisions, not new execution steps. Event work still goes
through the existing bounded planner and executor; successful help and
`request_npc_message` results are evaluated only after those backend operations
succeed.

Gemini receives only the backend-selected social decision and, for
`share_basic_information`, its one validated topic. It receives no relationship
stage, relationship event list, or private learning memory on event narration,
and the reaction call has no tools. Gemini cannot select a decision, advance a
stage, or mutate game state. NPC communication still uses
`request_npc_message`, remains limited to one consultation per player turn and
the existing event decision cap, and never supplies another NPC's conversation
history or semantic memories.

## Step 18: Relationship-Aware Chat Dialogue

The `/chat` route derives relationship context through the Step 17 progression
service for the current `(session_id, npc_id)`. The existing NPC decision module
selects only fixed dialogue decisions for exact greetings, validated prior-help
acknowledgments, and the planner's existing allowlisted request intents.
Greeting labels vary by `NEUTRAL`, `FRIENDLY`, or `TRUSTING`; help and information
decisions require a recognized planner intent and registry-approved capability.
Previous-help acknowledgments require a validated positive signal. Unsupported
or ambiguous messages do not enter this path and retain ordinary chat behavior.

Relationship-aware narration uses the existing `generate_npc_reply` service in
a tool-free mode. Gemini receives the selected dialogue decision, its optional
allowlisted topic, the current player message, and only when applicable the
already validated planner completion or explicitly shared peer response. It
receives no relationship stage, conversation history, game state, inventory,
or raw Chroma memories in this mode. It cannot select another decision,
invoke a game action, start a consultation, or write relationship state.

Explicit buy/give/heal requests bypass relationship-aware narration and keep the
existing game-action path. Recognized planner requests still execute their
existing bounded plan before narration; plan failure prevents the response, and
the existing one-consultation-per-turn limit remains in force. Relationship
stages are selected from the persisted Step 19 score, with scoped Step 17/18
history and learning as the compatibility fallback for relationships that do
not yet have a stored score. No relationship metadata is returned by `/chat` or
exposed through a new endpoint. Dialogue selection does not mutate health,
gold, inventory, game time, or other game state.

## Step 19: Relationship Evolution

`backend/app/services/npc_relationship_progression.py` remains the single
relationship evaluator and now reads/writes a bounded trust score in the
existing SQLite database. The additive `npc_player_relationships` table is
keyed by `(session_id, npc_id)` and cascades with the owning player session.
New relationships start at 20 points. `NEUTRAL` is 0–39, `FRIENDLY` is 40–79,
and `TRUSTING` is 80–100. Scores are clamped to that range, and negative
interactions can move a relationship back to an earlier stage.

Only fixed backend events change the score: `successful_help` adds 20,
`repeated_positive_interaction` adds 15, `successful_npc_communication` adds 10,
`gifted_item` adds 20, and a recognized `disrespectful_interaction` subtracts
25. Positive acknowledgments use the existing conservative message patterns;
disrespect uses a fixed allowlist. A successful existing `give_item` or
`heal_player` action and a completed help plan can contribute their configured
positive event. Failed actions/plans and purchases do not change relationship
state. No extra Gemini call is made for relationship calculations.

Updates occur only after the chat response and user message have been saved.
Each update is tied to that exact user-message row and NPC; the persisted last
processed message ID makes retries idempotent and rejects messages from another
session, NPC, role, or stale turn. The update changes only relationship state,
not health, gold, inventory, or game time. When no score row exists, the
existing scoped history and validated learning facts seed the legacy stage so
Step 18 dialogue behavior is retained. Scores and event metadata remain
internal; `/chat` and the frontend expose neither.

## Bounded NPC Plans

`backend/app/services/npc_planner.py` recognizes only `ASK_SWORD`,
`ASK_HERBS`, `ASK_HEALING`, `ASK_BLACKSMITH_HELP`, and `ASK_GENERAL_HELP` from
validated player text. Ravi can plan sword/blacksmith help directly; Meera can
plan herbs/healing help. Ravi's healing plan routes through the existing
`request_npc_message` relationship checks rather than executing Meera's
capabilities directly. Unclassified text creates no plan and follows the
existing chat path.

Only backend templates create plans. The executor accepts these steps:
`identify_player_need`, `inspect_supplies`, `offer_help`,
`provide_sword_information`, `provide_herb_information`, `request_npc_help`,
`speak_to_player`, and `stay_at_location`. Plans have one to three steps, and
the executor verifies NPC, source, goal, exact step ordering, and session before
execution. It stops at the first failed or rejected step; it does not retry or
regenerate plans. Player-request and event plans are internal, temporary, and
not persisted or exposed through a planning endpoint.

The executor only inspects read-only state/inventory or returns controlled
information/help results. It cannot update health, gold, inventory, location,
or game time. Event plans originate only from validated Step 13 decisions.
Gemini does not create or execute plans; it only writes the final narrative
after successful execution. Explicit peer replies are passed as untrusted
context, and plan-consumed consultations count against the existing one-call
limit. No plan steps, traces, or hidden reasoning are returned to the frontend
or stored in Chroma.

## Temporary Task Progress

Each executor call creates an in-memory task for the current request only.
Task statuses are `pending`, `running`, `completed`, `failed`, and `rejected`;
step statuses are `pending`, `running`, `completed`, `failed`, and `skipped`.
Each step can be attempted once, and the existing three-step plan limit applies.

The executor marks a step running before its fixed handler, then completed or
failed. Failure stops execution immediately; later steps remain pending and are
never retried. The final internal result contains only task status, completed
and total counts, and the failed step. Task IDs, per-step progress, and traces
are not persisted, sent to Gemini, or returned to the frontend. No task table or
new database is used.

## Dynamic NPC Movement

`backend/app/services/npc_movement.py` accepts only `village_square`, `forge`,
`healer_hut`, `market`, `forest_edge`, and the symbolic `player_location`.
Destinations come from fixed event rules and the existing schedule in
`backend/app/services/npc_schedules.py`, using only `player_state.game_time`.
Morning moves Ravi to the forge and Meera to the healer hut; afternoon follows
the schedule (market or work location); evening returns Ravi to the village
square and keeps Meera at the healer hut. Low-health Meera may move to the
symbolic player location.

Movement is an allowlisted `move_to_location` planner step and uses the
request-local task-progress transitions. Each NPC can move at most once per
event, with at most two NPC movements per event. Invalid locations, schedules,
sessions, events, or directed transitions fail the step and stop later steps;
there are no retries or background scheduler.

The current schema has no NPC location field (`player_state.location` belongs
to the player), so movement is a temporary, validated event result derived
from the prior schedule. It does not update player location or persist an NPC
position. Gemini may narrate only the validated location result; it never
chooses a destination. No movement history is written to SQLite or Chroma, and
the frontend receives no internal movement/task traces.

## Agentic Game Actions

Gemini can request only `buy_item`, `give_item`, and `heal_player`. The backend
validates each request and applies state changes in SQLite transactions; Gemini
never writes to SQLite directly. No other functions or game actions are exposed.

## Long-Term Semantic Memory

Only explicitly stated names, preferences, and selected item decisions are
considered for semantic memory. Ordinary messages and sensitive contact or
credential-like content are not stored. Relevant memories are retrieved for
the current session and selected NPC, then supplied as historical context to
that NPC. Ravi cannot retrieve Meera's memories or vice versa; full conversation
history remains in SQLite.