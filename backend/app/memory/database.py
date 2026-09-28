import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal, TypedDict

from app.npcs import get_npc

DATABASE_PATH = Path(__file__).resolve().parents[2] / "data" / "mystery_village.db"
MAX_HISTORY_MESSAGES = 20
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ConversationMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class StoredConversationMessage(TypedDict):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ConversationSummary(TypedDict):
    conversation_id: str
    npc_id: str
    npc_name: str
    title: str
    message_count: int
    started_at: str
    updated_at: str
    preview: str


class PlayerState(TypedDict):
    session_id: str
    location: str
    health: int
    gold: int
    game_time: str


class InventoryItem(TypedDict):
    item_name: str
    quantity: int


class NPCPlayerRelationshipState(TypedDict):
    session_id: str
    npc_id: str
    trust_score: int
    last_processed_message_id: int


def _validate_session_id(session_id: str) -> None:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("Invalid session ID.")


def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=5)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with closing(_connect()) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    npc_id TEXT NOT NULL DEFAULT 'ravi',
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            message_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(messages)")
            }
            if "npc_id" not in message_columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN npc_id TEXT NOT NULL DEFAULT 'ravi'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_npc_id "
                "ON messages (session_id, npc_id, id)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS player_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    location TEXT NOT NULL DEFAULT 'Mystery Village',
                    health INTEGER NOT NULL DEFAULT 100 CHECK (health >= 0),
                    gold INTEGER NOT NULL DEFAULT 50 CHECK (gold >= 0),
                    game_time TEXT NOT NULL DEFAULT '08:00',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    item_name TEXT NOT NULL CHECK (length(trim(item_name)) > 0),
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    UNIQUE (session_id, item_name),
                    FOREIGN KEY (session_id) REFERENCES player_state (session_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inventory_session_id "
                "ON inventory (session_id)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS npc_player_relationships (
                    session_id TEXT NOT NULL,
                    npc_id TEXT NOT NULL CHECK (npc_id IN ('ravi', 'meera')),
                    trust_score INTEGER NOT NULL DEFAULT 20
                        CHECK (trust_score BETWEEN 0 AND 100),
                    last_processed_message_id INTEGER NOT NULL DEFAULT 0
                        CHECK (last_processed_message_id >= 0),
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, npc_id),
                    FOREIGN KEY (session_id) REFERENCES player_state (session_id)
                        ON DELETE CASCADE
                )
                """
            )


def create_default_player_state(session_id: str) -> PlayerState:
    _validate_session_id(session_id)
    with closing(_connect()) as connection:
        with connection:
            connection.execute(
                "INSERT INTO player_state (session_id) VALUES (?) "
                "ON CONFLICT (session_id) DO NOTHING",
                (session_id,),
            )
            row = connection.execute(
                """
                SELECT session_id, location, health, gold, game_time
                FROM player_state
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

    return {
        "session_id": row[0],
        "location": row[1],
        "health": row[2],
        "gold": row[3],
        "game_time": row[4],
    }


def get_player_state(session_id: str) -> PlayerState:
    return create_default_player_state(session_id)


def update_player_state(
    session_id: str,
    *,
    location: str | None = None,
    health: int | None = None,
    gold: int | None = None,
    game_time: str | None = None,
) -> PlayerState:
    updates: dict[str, str | int] = {}
    if location is not None:
        if not location.strip():
            raise ValueError("Location must not be empty.")
        updates["location"] = location.strip()
    if health is not None:
        if isinstance(health, bool) or not isinstance(health, int) or not 0 <= health <= 100:
            raise ValueError("Health must be an integer from 0 to 100.")
        updates["health"] = health
    if gold is not None:
        if isinstance(gold, bool) or not isinstance(gold, int) or gold < 0:
            raise ValueError("Gold must be a non-negative integer.")
        updates["gold"] = gold
    if game_time is not None:
        if not game_time.strip():
            raise ValueError("Game time must not be empty.")
        updates["game_time"] = game_time.strip()

    _validate_session_id(session_id)
    create_default_player_state(session_id)
    if updates:
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = [*updates.values(), session_id]
        with closing(_connect()) as connection:
            with connection:
                connection.execute(
                    f"UPDATE player_state SET {assignments}, "
                    "updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    values,
                )
    return get_player_state(session_id)


def _validate_inventory_item(item_name: str, quantity: int) -> str:
    item_name = item_name.strip()
    if not item_name or len(item_name) > 100:
        raise ValueError("Item name must be between 1 and 100 characters.")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("Quantity must be a positive integer.")
    return item_name


def get_inventory(session_id: str) -> list[InventoryItem]:
    _validate_session_id(session_id)
    create_default_player_state(session_id)
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT item_name, quantity
            FROM inventory
            WHERE session_id = ?
            ORDER BY item_name COLLATE NOCASE
            """,
            (session_id,),
        ).fetchall()
    return [{"item_name": item_name, "quantity": quantity} for item_name, quantity in rows]


def add_inventory_item(session_id: str, item_name: str, quantity: int) -> None:
    _validate_session_id(session_id)
    item_name = _validate_inventory_item(item_name, quantity)
    create_default_player_state(session_id)
    with closing(_connect()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO inventory (session_id, item_name, quantity)
                VALUES (?, ?, ?)
                ON CONFLICT (session_id, item_name)
                DO UPDATE SET quantity = inventory.quantity + excluded.quantity
                """,
                (session_id, item_name, quantity),
            )


def remove_inventory_item_and_get_remaining(
    session_id: str,
    item_name: str,
    quantity: int,
) -> int | None:
    _validate_session_id(session_id)
    item_name = _validate_inventory_item(item_name, quantity)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT quantity FROM inventory WHERE session_id = ? AND item_name = ?",
            (session_id, item_name),
        ).fetchone()
        if row is None or row[0] < quantity:
            connection.rollback()
            return None

        remaining = row[0] - quantity
        if row[0] == quantity:
            connection.execute(
                "DELETE FROM inventory WHERE session_id = ? AND item_name = ?",
                (session_id, item_name),
            )
        else:
            connection.execute(
                "UPDATE inventory SET quantity = quantity - ? "
                "WHERE session_id = ? AND item_name = ?",
                (quantity, session_id, item_name),
            )
        connection.commit()
    return remaining


def remove_inventory_item(session_id: str, item_name: str, quantity: int) -> bool:
    return remove_inventory_item_and_get_remaining(session_id, item_name, quantity) is not None


def clear_inventory(session_id: str) -> None:
    _validate_session_id(session_id)
    with closing(_connect()) as connection:
        with connection:
            connection.execute(
                "DELETE FROM inventory WHERE session_id = ?",
                (session_id,),
            )


def purchase_inventory_item(
    session_id: str,
    item_name: str,
    quantity: int,
    total_price: int,
) -> tuple[int, int] | None:
    _validate_session_id(session_id)
    item_name = _validate_inventory_item(item_name, quantity)
    if isinstance(total_price, bool) or not isinstance(total_price, int) or total_price <= 0:
        raise ValueError("Total price must be a positive integer.")
    create_default_player_state(session_id)

    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        state = connection.execute(
            "SELECT gold FROM player_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if state is None or state[0] < total_price:
            connection.rollback()
            return None

        connection.execute(
            "UPDATE player_state SET gold = gold - ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (total_price, session_id),
        )
        connection.execute(
            """
            INSERT INTO inventory (session_id, item_name, quantity)
            VALUES (?, ?, ?)
            ON CONFLICT (session_id, item_name)
            DO UPDATE SET quantity = inventory.quantity + excluded.quantity
            """,
            (session_id, item_name, quantity),
        )
        new_gold = connection.execute(
            "SELECT gold FROM player_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        new_quantity = connection.execute(
            "SELECT quantity FROM inventory WHERE session_id = ? AND item_name = ?",
            (session_id, item_name),
        ).fetchone()[0]
        connection.commit()

    return new_gold, new_quantity


def apply_healing(session_id: str, amount: int) -> int:
    _validate_session_id(session_id)
    if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
        raise ValueError("Healing amount must be a positive integer.")
    create_default_player_state(session_id)

    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        state = connection.execute(
            "SELECT health FROM player_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if state is None:
            connection.rollback()
            raise ValueError("Player state does not exist.")

        health = min(100, state[0] + amount)
        connection.execute(
            "UPDATE player_state SET health = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE session_id = ?",
            (health, session_id),
        )
        connection.commit()
    return health


def save_message(
    session_id: str,
    role: Literal["user", "assistant"],
    content: str,
    npc_id: str = "ravi",
) -> int:
    _validate_session_id(session_id)
    get_npc(npc_id)
    if role not in ("user", "assistant"):
        raise ValueError("Invalid message role.")
    if not content.strip():
        raise ValueError("Message content must not be empty.")

    with closing(_connect()) as connection:
        with connection:
            cursor = connection.execute(
                "INSERT INTO messages (session_id, npc_id, role, content) "
                "VALUES (?, ?, ?, ?)",
                (session_id, npc_id, role, content.strip()),
            )
    return int(cursor.lastrowid)


def get_conversation_history(
    session_id: str,
    limit: int = MAX_HISTORY_MESSAGES,
    *,
    npc_id: str = "ravi",
) -> list[ConversationMessage]:
    _validate_session_id(session_id)
    get_npc(npc_id)
    if not 1 <= limit <= MAX_HISTORY_MESSAGES:
        raise ValueError("History limit must be between 1 and 20.")

    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ? AND npc_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, npc_id, limit),
        ).fetchall()

    return [
        {"role": role, "content": content}
        for role, content in reversed(rows)
    ]


def list_conversation_summaries(session_id: str) -> list[ConversationSummary]:
    _validate_session_id(session_id)
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT
                messages.npc_id,
                COUNT(*),
                MIN(messages.created_at),
                MAX(messages.created_at),
                COALESCE(
                    (
                        SELECT content FROM messages AS first_user
                        WHERE first_user.session_id = messages.session_id
                          AND first_user.npc_id = messages.npc_id
                          AND first_user.role = 'user'
                        ORDER BY first_user.id ASC LIMIT 1
                    ),
                    (
                        SELECT content FROM messages AS first_message
                        WHERE first_message.session_id = messages.session_id
                          AND first_message.npc_id = messages.npc_id
                        ORDER BY first_message.id ASC LIMIT 1
                    )
                )
            FROM messages
            WHERE messages.session_id = ?
            GROUP BY messages.session_id, messages.npc_id
            ORDER BY MAX(messages.id) DESC
            """,
            (session_id,),
        ).fetchall()

    summaries = []
    for npc_id, message_count, started_at, updated_at, preview in rows:
        npc = get_npc(npc_id)
        preview_text = preview or ""
        summaries.append(
            {
                "conversation_id": npc_id,
                "npc_id": npc_id,
                "npc_name": npc.name,
                "title": f"Conversation with {npc.name}",
                "message_count": message_count,
                "started_at": started_at,
                "updated_at": updated_at,
                "preview": (
                    preview_text if len(preview_text) <= 160 else f"{preview_text[:157]}..."
                ),
            }
        )
    return summaries


def get_stored_conversation_messages(
    session_id: str,
    npc_id: str,
) -> list[StoredConversationMessage]:
    _validate_session_id(session_id)
    get_npc(npc_id)
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE session_id = ? AND npc_id = ?
            ORDER BY id ASC
            """,
            (session_id, npc_id),
        ).fetchall()
    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "created_at": row[3],
        }
        for row in rows
    ]


def clear_conversation_history(session_id: str, *, npc_id: str = "ravi") -> None:
    _validate_session_id(session_id)
    get_npc(npc_id)
    with closing(_connect()) as connection:
        with connection:
            connection.execute(
                "DELETE FROM messages WHERE session_id = ? AND npc_id = ?",
                (session_id, npc_id),
            )


def get_npc_player_relationship_state(
    session_id: str,
    npc_id: str,
) -> NPCPlayerRelationshipState | None:
    _validate_session_id(session_id)
    get_npc(npc_id)
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT session_id, npc_id, trust_score, last_processed_message_id
            FROM npc_player_relationships
            WHERE session_id = ? AND npc_id = ?
            """,
            (session_id, npc_id),
        ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row[0],
        "npc_id": row[1],
        "trust_score": row[2],
        "last_processed_message_id": row[3],
    }


def apply_npc_player_relationship_delta(
    session_id: str,
    npc_id: str,
    delta: int,
    *,
    source_message_id: int,
    initial_score: int,
) -> int:
    _validate_session_id(session_id)
    get_npc(npc_id)
    if isinstance(delta, bool) or not isinstance(delta, int) or not -100 <= delta <= 100:
        raise ValueError("Relationship delta is not allowed.")
    if (
        isinstance(source_message_id, bool)
        or not isinstance(source_message_id, int)
        or source_message_id <= 0
    ):
        raise ValueError("Relationship source message is invalid.")
    if (
        isinstance(initial_score, bool)
        or not isinstance(initial_score, int)
        or not 0 <= initial_score <= 100
    ):
        raise ValueError("Initial relationship score is invalid.")

    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        player = connection.execute(
            "SELECT 1 FROM player_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        source_message = connection.execute(
            """
            SELECT role FROM messages
            WHERE id = ? AND session_id = ? AND npc_id = ?
            """,
            (source_message_id, session_id, npc_id),
        ).fetchone()
        if player is None or source_message is None or source_message[0] != "user":
            connection.rollback()
            raise ValueError("Relationship event is not tied to this player's NPC turn.")

        current = connection.execute(
            """
            SELECT trust_score, last_processed_message_id
            FROM npc_player_relationships
            WHERE session_id = ? AND npc_id = ?
            """,
            (session_id, npc_id),
        ).fetchone()
        current_score = current[0] if current is not None else initial_score
        last_message_id = current[1] if current is not None else 0
        if source_message_id <= last_message_id:
            connection.rollback()
            return current_score

        updated_score = max(0, min(100, current_score + delta))
        connection.execute(
            """
            INSERT INTO npc_player_relationships (
                session_id, npc_id, trust_score, last_processed_message_id
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT (session_id, npc_id)
            DO UPDATE SET trust_score = excluded.trust_score,
                          last_processed_message_id = excluded.last_processed_message_id,
                          updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, npc_id, updated_score, source_message_id),
        )
        connection.commit()
    return updated_score