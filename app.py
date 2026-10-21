"""
Live Database Intelligence Assistant (Showcase Challenge 2)
===========================================================
An autonomous Natural Language to SQL (NL2SQL) agent built with Streamlit,
Ollama, and SQLite.

Architectural Overview & Pipeline Stages
----------------------------------------
The system implements structured agentic workflows inspired by Anthropic's
*Routing* and *Evaluator-Optimizer* patterns and OpenAI's defense-in-depth
guardrail guidelines:

1. Intent Classification (Routing Workflow):
   Routes incoming prompts into 'Data Retrieval', 'Visualization',
   'General Knowledge', or 'Invalid', blocking out-of-scope requests before
   allocating database or reasoning resources.
2. Schema Contextualization:
   Dynamically introspects the SQLite DDL schema and injects foreign-key
   relationships and fuzzy-matching rules (`LOWER(...) LIKE '%...%'`) into the
   SQL generation system prompt.
3. Thought Streaming & Fast-Mode Early Cutoff:
   Streams reasoning tokens (`thinking`) live to the UI when deep reasoning is
   enabled, while a fast-mode heuristic halts generation immediately once a
   complete ```sql ... ``` block is isolated to eliminate post-query rambling.
4. Multi-Layer Security Guardrails:
   Treats all LLM output as untrusted input via four defensive layers:
   - Read-only statement enforcement (`SELECT` / `WITH` only, single statement).
   - Regex blacklist blocking mutating/administrative DDL & DML keywords.
   - System catalog protection and explicit table whitelisting.
   - Engine-level read-only connection enforcement (`file:analytics.db?mode=ro`).
5. Closed-Loop Self-Healing (Evaluator-Optimizer Workflow):
   Captures SQLite syntax/schema exceptions or table whitelist mismatches,
   appends the error context to the conversation history, and autonomously
   re-prompts the model to repair the SQL query (up to 3 attempts).
6. Anomaly Detection & Grounded Citation Synthesis:
   Scans retrieved records for business-rule anomalies (e.g., negative prices
   or quantities) and synthesizes a concise, factual answer with explicit
   table/column citations (`[Source: table (column)]`) and follow-up prompts.
"""

from contextlib import closing
from functools import lru_cache
import re
import sqlite3
import time
from typing import Any, Optional

import ollama
import pandas as pd
import streamlit as st


# =============================================================================
# 1. Configuration, Constants & Prompt Templates
# =============================================================================

DB_PATH: str = "analytics.db"

AVAILABLE_MODELS: list[str] = [
    "qwen2.5-coder:7b",
    "qwen3.5:9b",
]
MODEL_NAME: str = AVAILABLE_MODELS[0]

# Whitelisted domain tables allowed in user-generated queries.
ALLOWED_TABLES: tuple[str, ...] = (
    "customers",
    "products",
    "orders",
    "order_items",
)

ALLOWED_INTENTS: tuple[str, ...] = ("Data Retrieval", "Visualization")
ALL_INTENT_LABELS: tuple[str, ...] = (
    "Data Retrieval",
    "Visualization",
    "General Knowledge",
)

MAX_RETRIES: int = 3

# Throttle interval (seconds) for live Markdown streaming updates (~25 FPS)
# to keep the Streamlit frontend responsive during fast token generation.
STREAM_UPDATE_INTERVAL_SEC: float = 0.04

# Financial and inventory column substrings checked for negative-value anomalies.
ANOMALY_NUMERIC_KEYWORDS: tuple[str, ...] = (
    "price",
    "stock",
    "quantity",
    "total_amount",
    "unit_price",
    "amount",
)

# Pre-compiled security regexes for multi-layer SQL validation.
FORBIDDEN_SQL_PATTERN: re.Pattern[str] = re.compile(
    r"\b(UPDATE|DELETE|ALTER|DROP|INSERT|REPLACE|TRUNCATE|CREATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)
SYSTEM_TABLE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(sqlite_master|sqlite_schema|sqlite_temp_master|sqlite_temp_schema)\b",
    re.IGNORECASE,
)
TABLE_REF_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:FROM|JOIN)\s+[`\"'\[]?([a-zA-Z_][a-zA-Z0-9_]*)[`\"'\]]?",
    re.IGNORECASE,
)
CTE_ALIAS_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:WITH|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(",
    re.IGNORECASE,
)

EXAMPLE_QUERIES: list[str] = [
    "Who are the top 5 customers by total spending?",
    "Which product category generated the highest total revenue?",
    "How many orders were placed by customers in the US?",
    "What products did customer 'David 2' purchase?",
    "Show total sales by country in 2025.",
]


def build_sql_system_prompt(schema: str) -> str:
    """Constructs the schema-aware system prompt for SQLite query generation."""
    return f"""You are an SQL generator for SQLite.
Database Schema:
{schema}

Write ONLY a valid SQLite SELECT query to answer the user's request.
Guidelines:
1. Fuzzy String Matching: Always use `LIKE '%keyword%'` with `LOWER()` on text fields (e.g., `LOWER(name) LIKE '%novel%'` or `LOWER(country) LIKE '%us%'`).
2. Schema & Entity Relationships:
   - `customers` links to `orders` via `orders.customer_id = customers.id`.
   - `orders` links to `products` through `order_items` via `order_items.order_id = orders.id` and `order_items.product_id = products.id`.
   - For total sales, customer spending, or aggregations, join tables accordingly and use `SUM()`.
3. Reasoning Conciseness: Formulate the query directly. Once you write the query in ```sql ... ```, finish immediately without second-guessing or exploring alternate versions.
4. Wrap your query in ```sql ... ``` code blocks."""


def build_synthesis_prompt(
    user_prompt: str,
    executed_sql: str,
    retrieved_data: dict[str, Any],
) -> str:
    """Constructs the prompt for grounded natural-language answer synthesis."""
    return f"""Synthesize a natural language answer for the user based strictly on the retrieved database records.
User Question: {user_prompt}
Executed Query: {executed_sql}
Retrieved Data: {retrieved_data}

Rules:
1. Provide a direct, factual 1-2 sentence response.
2. Include explicit citations indicating the source table and columns (e.g., [Source: customers table (name), orders table (total_amount)]).
3. At the end, provide 2-3 logical follow-up questions the user might want to ask next, prefixed with '**Suggested Follow-ups:**'."""


# =============================================================================
# 2. Database Access & Schema Introspection Layer
# =============================================================================

def _connect_readonly() -> sqlite3.Connection:
    """
    Opens a SQLite connection strictly in read-only URI mode (`mode=ro`).

    Engine-level read-only enforcement guarantees that even if an unexpected
    query bypasses syntactic checks, SQLite itself rejects any write operation.
    """
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


@lru_cache(maxsize=1)
def get_db_schema() -> str:
    """Fetches the full DDL schema for all user tables in `analytics.db`."""
    with closing(_connect_readonly()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        return "\n".join(row[0] for row in cursor.fetchall() if row[0])


SCHEMA: str = get_db_schema()


@st.cache_data(show_spinner=False)
def get_table_previews(limit: int = 5) -> dict[str, pd.DataFrame]:
    """Fetches a top-N row preview of each table for the sidebar inspector."""
    previews: dict[str, pd.DataFrame] = {}
    with closing(_connect_readonly()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            if table not in ALLOWED_TABLES:
                continue
            cursor.execute(f"SELECT * FROM {table} LIMIT {int(limit)};")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            previews[table] = pd.DataFrame(rows, columns=columns)
    return previews


@st.cache_data(show_spinner=False)
def get_db_stats() -> dict[str, int]:
    """Fetches total record counts across all whitelisted primary tables."""
    stats: dict[str, int] = {}
    with closing(_connect_readonly()) as conn:
        cursor = conn.cursor()
        for table in ALLOWED_TABLES:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                stats[table] = cursor.fetchone()[0]
            except sqlite3.Error:
                stats[table] = 0
    return stats


def run_sql_query(query: str) -> dict[str, Any]:
    """
    Executes a validated SQL query against SQLite in read-only mode.

    Returns a dictionary containing `success` (bool), `columns` (list[str]),
    `data` (list[tuple]), or `error` (str) when execution fails so the
    self-healing loop can inspect the database error message.
    """
    with closing(_connect_readonly()) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return {"success": True, "columns": columns, "data": rows}
        except Exception as exc:
            return {"success": False, "error": str(exc)}


# =============================================================================
# 3. Multi-Layer Security Guardrails & Data Quality Checks
# =============================================================================

def extract_sql_query(content: str, thinking: str = "") -> str:
    """
    Extracts a clean SQL statement from the LLM response or reasoning trace.

    Prioritizes fenced ```sql ... ``` blocks, falls back to generic code blocks,
    and finally scans raw lines starting with `SELECT` or `WITH`.
    """
    target = content.strip() or thinking.strip()

    if "```sql" in target:
        sql = target.split("```sql", 1)[1].split("```", 1)[0].strip()
    elif "```" in target:
        sql = target.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        sql_lines: list[str] = []
        started = False
        for line in target.splitlines():
            if line.strip().upper().startswith(("SELECT", "WITH")):
                started = True
            if started:
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip() if sql_lines else target.strip()

    return sql.rstrip(";").strip()


def validate_sql_safety(sql_query: str) -> tuple[bool, Optional[str], bool]:
    """
    Multi-layer pre-execution security validator for LLM-generated SQL.

    Layers enforced:
    1. Keyword Blacklist: Blocks mutating/administrative commands (UPDATE,
       DELETE, ALTER, DROP, INSERT, etc.).
    2. System Catalog Protection: Blocks access to `sqlite_master` and internal
       schema metadata tables.
    3. Single-Statement & Read-Only Prefix Check: Prevents stacked query
       injection (`;`) and requires queries to start with `SELECT` or `WITH`.
    4. Table Whitelisting: Verifies that referenced tables belong to
       `ALLOWED_TABLES` (`customers`, `products`, `orders`, `order_items`).

    Returns:
        tuple[is_valid, error_message, is_recoverable]:
        - `is_recoverable=False` indicates a hard security violation that halts
          execution immediately.
        - `is_recoverable=True` indicates a schema/whitelist mismatch that can
          be fed back into the self-healing loop for correction.
    """
    if not sql_query or not sql_query.strip():
        return False, "Generated SQL query is empty.", True

    # Layer 1: Forbidden mutating/DDL keyword check (hard security violation)
    forbidden_match = FORBIDDEN_SQL_PATTERN.search(sql_query)
    if forbidden_match:
        forbidden_cmd = forbidden_match.group(0).upper()
        return (
            False,
            (
                f"Safety Violation: Query contains forbidden modifying command "
                f"'{forbidden_cmd}'. This assistant is strictly read-only and "
                f"cannot modify the database."
            ),
            False,
        )

    # Layer 2: System catalog disclosure protection (hard security violation)
    sys_match = SYSTEM_TABLE_PATTERN.search(sql_query)
    if sys_match:
        sys_table = sys_match.group(0)
        return (
            False,
            f"Safety Violation: Access to system catalog table '{sys_table}' is prohibited.",
            False,
        )

    # Layer 3: Single-statement & SELECT/WITH enforcement
    cleaned = re.sub(r"--.*?$", "", sql_query, flags=re.MULTILINE).strip()
    if ";" in cleaned:
        return (
            False,
            "Safety Violation: Multi-statement SQL execution is prohibited.",
            False,
        )
    if not cleaned.upper().startswith(("SELECT", "WITH")):
        return (
            False,
            "Query must begin with a read-only SELECT or WITH statement.",
            True,
        )

    # Layer 4: Table whitelist validation (excluding CTE aliases defined in WITH)
    cte_aliases = {alias.lower() for alias in CTE_ALIAS_PATTERN.findall(cleaned)}
    referenced_tables = {tbl.lower() for tbl in TABLE_REF_PATTERN.findall(cleaned)}
    allowed_set = {tbl.lower() for tbl in ALLOWED_TABLES}

    unapproved_tables = referenced_tables - allowed_set - cte_aliases
    if unapproved_tables:
        bad_list = ", ".join(sorted(unapproved_tables))
        allowed_list = ", ".join(ALLOWED_TABLES)
        return (
            False,
            (
                f"Table Whitelist Error: Unknown or unpermitted table(s): {bad_list}. "
                f"Only the following tables exist in the schema: {allowed_list}."
            ),
            True,
        )

    return True, None, False


def detect_anomalies(df: pd.DataFrame) -> list[str]:
    """
    Scans retrieved query results for domain-level data quality anomalies.

    Flags negative values in financial or inventory quantity columns (e.g.,
    `price`, `stock`, `quantity`, `total_amount`, `unit_price`, `amount`).
    """
    anomalies: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ANOMALY_NUMERIC_KEYWORDS):
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if (numeric_series < 0).any():
                anomalies.append(f"Negative value detected in column '{col}'")
    return anomalies


# =============================================================================
# 4. Agentic Pipeline & LLM Orchestration Layer
# =============================================================================

def model_supports_thinking(model_name: str) -> bool:
    """Returns True if the selected Ollama model supports native reasoning tokens."""
    lowered = model_name.lower()
    return "qwen3.5" in lowered or "r1" in lowered


def classify_intent(user_prompt: str, model_name: str = AVAILABLE_MODELS[0]) -> str:
    """
    Step 0 (Routing Workflow): Classifies user prompt intent before DB access.

    Settings rationale:
    - think=False: Suppresses reasoning monologue; single-label routing takes <0.5s
      instead of 30-40s of internal deliberation.
    - num_predict=20: Caps token generation strictly to the classification label.
    - temperature=0.0: Enforces greedy, deterministic classification.
    """
    prompt = (
        "Classify the user's intent into one of: 'Data Retrieval', "
        "'Visualization', 'General Knowledge', 'Invalid'. "
        f"User prompt: {user_prompt}\n"
        "Return ONLY the classification label."
    )
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={"num_predict": 20, "temperature": 0.0},
    )
    content = response["message"]["content"].strip()
    for label in ALL_INTENT_LABELS:
        if label.lower() in content.lower():
            return label
    return "Invalid"


def _has_complete_sql_fence(text: str) -> bool:
    """Returns True once a complete ```sql ... ``` fenced block has been streamed."""
    return "```sql" in text and "```" in text.split("```sql", 1)[1]


def _stream_sql_candidate(
    messages: list[dict[str, str]],
    model_name: str,
    should_think: bool,
    fast_mode: bool,
    debug_mode: bool,
    thought_box: Any = None,
) -> tuple[str, str, bool]:
    """
    Streams SQL generation from Ollama with live thought updates and early cutoff.

    Settings rationale:
    - stream=True: Enables real-time UI updates and immediate early cutoff once
      a complete ```sql ... ``` block is isolated.
    - think=should_think: Bypasses thinking tokens in Fast Mode or on direct
      coder models (~1-3s latency), or streams the full reasoning monologue
      when deep reasoning inspection is enabled.
    - temperature=0.0: Eliminates stochastic variance for deterministic SQL.
    """
    accumulated_thinking = ""
    accumulated_content = ""
    last_th_update = time.time()
    stopped_early = False

    chat_stream = ollama.chat(
        model=model_name,
        messages=messages,
        think=should_think,
        stream=True,
        options={"temperature": 0.0},
    )

    for chunk in chat_stream:
        msg = chunk.message
        th = getattr(msg, "thinking", None)
        ct = getattr(msg, "content", None)

        if th:
            accumulated_thinking += th
            now = time.time()
            if debug_mode and thought_box and (now - last_th_update >= STREAM_UPDATE_INTERVAL_SEC):
                thought_box.markdown(accumulated_thinking + "▌")
                last_th_update = now

            if fast_mode and _has_complete_sql_fence(accumulated_thinking):
                stopped_early = True
                break

        if ct:
            if not accumulated_content and debug_mode and thought_box and accumulated_thinking:
                thought_box.markdown(accumulated_thinking)
            accumulated_content += ct

            if fast_mode and _has_complete_sql_fence(accumulated_content):
                stopped_early = True
                break

    if debug_mode and thought_box and accumulated_thinking:
        thought_box.markdown(accumulated_thinking)

    return accumulated_content, accumulated_thinking, stopped_early


def _stream_synthesized_answer(
    synthesis_prompt: str,
    model_name: str,
    should_think: bool,
    debug_mode: bool,
    synth_thought_box: Any = None,
    response_placeholder: Any = None,
) -> str:
    """
    Streams the grounded natural-language answer and source citations.

    Settings rationale:
    - stream=True: Delivers typewriter-style token streaming to the answer card.
    - think=should_think: Streams synthesis reasoning only when requested.
    - temperature=0.1: Near-greedy decoding keeps citations and numbers strictly
      faithful to retrieved records while preserving natural prose.
    """
    accumulated_thinking = ""
    accumulated_content = ""
    last_th_update = time.time()
    last_ct_update = time.time()

    synth_stream = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": synthesis_prompt}],
        think=should_think,
        stream=True,
        options={"temperature": 0.1},
    )

    for chunk in synth_stream:
        msg = chunk.message
        th = getattr(msg, "thinking", None)
        ct = getattr(msg, "content", None)

        if th and should_think:
            accumulated_thinking += th
            now = time.time()
            if debug_mode and synth_thought_box and (now - last_th_update >= STREAM_UPDATE_INTERVAL_SEC):
                synth_thought_box.markdown(accumulated_thinking + "▌")
                last_th_update = now

        if ct:
            if not accumulated_content and debug_mode and synth_thought_box and accumulated_thinking:
                synth_thought_box.markdown(accumulated_thinking)
            accumulated_content += ct
            now = time.time()
            if response_placeholder and (now - last_ct_update >= STREAM_UPDATE_INTERVAL_SEC):
                response_placeholder.markdown(accumulated_content + "▌")
                last_ct_update = now

    if debug_mode and synth_thought_box and accumulated_thinking:
        synth_thought_box.markdown(accumulated_thinking)

    if response_placeholder and accumulated_content:
        response_placeholder.markdown(accumulated_content)

    return accumulated_content


def agent_pipeline(
    user_prompt: str,
    model_name: str = AVAILABLE_MODELS[0],
    debug_mode: bool = False,
    fast_mode: bool = True,
    status_container: Any = None,
    response_placeholder: Any = None,
) -> tuple[str, list[str], Optional[str], Optional[dict[str, Any]]]:
    """
    Executes the end-to-end multi-step NL2SQL agentic pipeline:
    Route Intent -> Generate SQL -> Validate Guardrails -> Execute Read-Only ->
    Self-Heal on Error (up to 3 attempts) -> Audit Anomalies -> Synthesize with Citations.
    """
    trace: list[str] = []
    should_think = (not fast_mode) and model_supports_thinking(model_name)

    # --- Step 0: Intent Classification (Routing Guardrail) ---
    intent = classify_intent(user_prompt, model_name=model_name)
    trace.append(f"**Model:** `{model_name}`")
    trace.append(f"**Intent Classification:** `{intent}`")
    if debug_mode and status_container:
        status_container.markdown(f"#### Step 0: Intent Classified as `{intent}`")

    if intent not in ALLOWED_INTENTS:
        msg = (
            f"Sorry, I can only help with data retrieval and database queries. "
            f"(Detected intent: {intent})"
        )
        if response_placeholder:
            response_placeholder.markdown(msg)
        return msg, trace, None, None

    # --- Step 1: Iterative SQL Generation, Validation & Self-Healing Loop ---
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_sql_system_prompt(SCHEMA)},
        {"role": "user", "content": user_prompt},
    ]

    final_data: Optional[dict[str, Any]] = None
    successful_sql: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_header = f"Attempt {attempt}"
        if attempt > 1:
            trace.append(
                f"**{attempt_header} Auto-Healing:** Re-prompting model with previous database error..."
            )
            if debug_mode and status_container:
                status_container.markdown(f"#### {attempt_header}: Autonomous Self-Correction")
        elif debug_mode and status_container:
            status_container.markdown(f"#### {attempt_header}: Formulating SQL Query")

        thought_box = None
        if debug_mode and status_container and should_think:
            with status_container.expander(
                f"Agent Internal Thought Process ({attempt_header})", expanded=True
            ):
                st.caption("Real-time LLM Internal Monologue & Reasoning Strategy")
                thought_box = st.empty()

        raw_content, raw_thinking, stopped_early = _stream_sql_candidate(
            messages=messages,
            model_name=model_name,
            should_think=should_think,
            fast_mode=fast_mode,
            debug_mode=debug_mode,
            thought_box=thought_box,
        )

        if stopped_early and debug_mode and status_container:
            status_container.caption(
                "*Early cutoff: Complete SQL identified. Halted subsequent rambling/second-guessing.*"
            )

        sql_query = extract_sql_query(raw_content, raw_thinking)
        trace.append(f"**{attempt_header} Generated SQL:** `{sql_query}`")

        if debug_mode and status_container:
            status_container.markdown(f"**Generated SQL Query ({attempt_header}):**")
            status_container.code(sql_query, language="sql")

        # Multi-layer SQL safety & table whitelist validation
        is_valid, validation_error, is_recoverable = validate_sql_safety(sql_query)
        if not is_valid and validation_error:
            if not is_recoverable:
                # Hard security violation (e.g., DROP/UPDATE/DELETE/system catalog): halt immediately
                trace.append(f"Execution blocked: `{validation_error}`")
                if debug_mode and status_container:
                    status_container.error(validation_error)
                if response_placeholder:
                    response_placeholder.error(validation_error)
                return validation_error, trace, None, None

            # Recoverable validation/whitelist error: route into self-healing loop
            db_result: dict[str, Any] = {"success": False, "error": validation_error}
        else:
            db_result = run_sql_query(sql_query)

        if db_result["success"]:
            successful_sql = sql_query
            final_data = db_result
            row_count = len(db_result["data"])
            trace.append(f"Database query executed successfully ({row_count} row(s) returned).")

            df_records = (
                pd.DataFrame(db_result["data"], columns=db_result["columns"])
                if row_count > 0
                else pd.DataFrame()
            )
            anomalies = detect_anomalies(df_records) if not df_records.empty else []
            if anomalies:
                anomaly_msg = ", ".join(anomalies)
                trace.append(f"**Anomaly Detected:** {anomaly_msg}")

            if debug_mode and status_container:
                status_container.success(
                    f"SQL Query executed successfully ({row_count} row(s) returned)"
                )
                if not df_records.empty:
                    status_container.dataframe(df_records, width="stretch")
                    if anomalies:
                        status_container.warning(f"Data Anomaly Detected: {anomaly_msg}")
                else:
                    status_container.info("Query returned 0 matching records.")
            break

        # Closed-loop self-healing feedback on SQLite or table-whitelist error
        error_msg = db_result["error"]
        trace.append(f"Execution failed: `{error_msg}`. Agent triggering auto-correction...")

        if debug_mode and status_container:
            status_container.error(f"SQL Execution failed: `{error_msg}`")
            if attempt < MAX_RETRIES:
                status_container.warning(
                    f"Triggering auto-healing feedback loop for Attempt {attempt + 1}..."
                )

        messages.append(
            {"role": "assistant", "content": raw_content or f"```sql\n{sql_query}\n```"}
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"The query failed with error: {error_msg}. "
                    "Please fix the SQL query according to the database schema."
                ),
            }
        )

    if not final_data or not final_data.get("success") or not successful_sql:
        fail_msg = "Agent could not resolve the database query after retries."
        if response_placeholder:
            response_placeholder.error(fail_msg)
        return fail_msg, trace, None, None

    # --- Step 2: Grounded Answer Synthesis with Source Citations ---
    synthesis_prompt = build_synthesis_prompt(user_prompt, successful_sql, final_data)

    synth_thought_box = None
    if debug_mode and status_container:
        status_container.markdown("#### Step 2: Synthesizing Answer & Formulating Citations")
        if should_think:
            with status_container.expander("Agent Synthesis Thought Process", expanded=True):
                st.caption("Real-time Reasoning over DB Records & Citation Formulation")
                synth_thought_box = st.empty()

    synthesized_answer = _stream_synthesized_answer(
        synthesis_prompt=synthesis_prompt,
        model_name=model_name,
        should_think=should_think,
        debug_mode=debug_mode,
        synth_thought_box=synth_thought_box,
        response_placeholder=response_placeholder,
    )

    return synthesized_answer, trace, successful_sql, final_data


# =============================================================================
# 5. Streamlit UI Presentation Layer
# =============================================================================

def inject_custom_styles() -> None:
    """Applies custom CSS styling for metric cards and the showcase header badge."""
    st.markdown(
        """
        <style>
            div[data-testid="stMetric"] {
                background-color: rgba(128, 128, 128, 0.05);
                border: 1px solid rgba(128, 128, 128, 0.15);
                border-radius: 8px;
                padding: 10px 14px;
            }
            .hero-badge {
                display: inline-block;
                padding: 3px 10px;
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                border-radius: 12px;
                background-color: rgba(59, 130, 246, 0.12);
                color: #3b82f6;
                border: 1px solid rgba(59, 130, 246, 0.25);
                margin-bottom: 6px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initializes required Streamlit session-state keys."""
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""
    if "auto_submit" not in st.session_state:
        st.session_state["auto_submit"] = False


def render_sidebar(db_stats: dict[str, int]) -> tuple[str, bool, bool, bool]:
    """
    Renders the assistant configuration sidebar, live schema inspector,
    and one-click example prompts.
    """
    with st.sidebar:
        st.header("Assistant Settings")
        selected_model = st.selectbox(
            "LLM Engine",
            options=AVAILABLE_MODELS,
            index=0,
            help=(
                "Switch between the fast SQL coder model (qwen2.5-coder:7b) "
                "and the deep reasoning model (qwen3.5:9b)."
            ),
        )

        is_reasoning = model_supports_thinking(selected_model)

        debug_mode = st.toggle(
            "Debug Trace Mode",
            value=True,
            help=(
                "Stream the agent's intent classification, internal monologue, "
                "generated SQL, and auto-healing feedback in real time."
            ),
        )

        if is_reasoning:
            fast_mode = st.toggle(
                "Fast Mode (Bypass Thinking)",
                value=True,
                help=(
                    "Bypasses deep internal reasoning monologue across classification, "
                    "SQL generation, and synthesis for ~1-3 second responses. "
                    "Turn OFF if you want to inspect full deep thinking."
                ),
            )
        else:
            fast_mode = True
            st.caption("*Direct Coder Engine: Operates natively without thinking latency.*")

        st.caption(f"Active Engine: `{selected_model}`")
        st.divider()

        st.subheader("Database Inspector")
        st.caption(f"Connected to `analytics.db` ({sum(db_stats.values()):,} total records)")
        with st.expander("Inspect Schema & Tables"):
            previews = get_table_previews()
            for table_name, df in previews.items():
                row_count = db_stats.get(table_name, len(df))
                st.markdown(f"**Table: `{table_name}`** (`{row_count}` rows)")
                st.dataframe(df, width="stretch")

        st.divider()
        st.subheader("Example Queries")
        st.caption("Click any query to execute immediately:")
        for ex in EXAMPLE_QUERIES:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state["user_query"] = ex
                st.session_state["auto_submit"] = True
                st.rerun()

    return selected_model, is_reasoning, debug_mode, fast_mode


def render_header_and_metrics(
    db_stats: dict[str, int],
    selected_model: str,
    debug_mode: bool,
    is_reasoning: bool,
    fast_mode: bool,
) -> None:
    """Renders the page hero banner, live database KPIs, and engine status alert."""
    st.markdown(
        '<div class="hero-badge">Turku AMK AI Showcase • Agentic SQL Assistant</div>',
        unsafe_allow_html=True,
    )
    st.title("Live Database Intelligence Assistant")
    st.caption(
        "Autonomous Text-to-SQL with Schema Verification, Self-Healing Error Recovery & Grounded Citations"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Database", "analytics.db", "SQLite 3 (Read-Only)")
    col2.metric("Customers", f"{db_stats.get('customers', 0)}", "Customer Profiles")
    col3.metric("Catalog Items", f"{db_stats.get('products', 0)}", "Product SKUs")
    col4.metric(
        "Transactions",
        f"{db_stats.get('orders', 0)}",
        f"{db_stats.get('order_items', 0)} Line Items",
    )

    if debug_mode:
        if is_reasoning and not fast_mode:
            st.info(
                f"**Deep Reasoning Mode Active:** Streaming live thought monologue "
                f"from `{selected_model}` before query synthesis."
            )
        elif is_reasoning:
            st.info(
                f"**Fast Demo Mode Active:** Reasoning model `{selected_model}` "
                f"running with suppressed thinking monologue for high-speed showcase."
            )
        else:
            st.info(
                f"**High-Speed Coder Mode Active:** Generating deterministic SQL "
                f"and answers directly with `{selected_model}` (zero thinking overhead)."
            )

    st.markdown("---")


def render_results_inspector(
    final_data: Optional[dict[str, Any]],
    sql_query: Optional[str],
    trace_log: list[str],
) -> None:
    """Renders the tabbed post-execution inspector (Records, SQL, and Trace)."""
    st.markdown("---")
    tab_records, tab_sql, tab_trace = st.tabs(
        ["Retrieved Records", "Generated SQL", "Execution Trace"]
    )

    with tab_records:
        if final_data and final_data.get("success") and final_data.get("data"):
            df_result = pd.DataFrame(final_data["data"], columns=final_data["columns"])
            st.dataframe(df_result, width="stretch")
            st.caption(f"Retrieved {len(df_result)} record(s) from `analytics.db`")
        elif final_data and final_data.get("success"):
            st.info("Query executed successfully, but returned 0 matching records.")
        else:
            st.caption("No database records retrieved.")

    with tab_sql:
        if sql_query:
            st.code(sql_query, language="sql")
        else:
            st.caption("No SQL query generated.")

    with tab_trace:
        for step in trace_log:
            st.markdown(f"- {step}")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Live Database Intelligence Assistant",
        layout="wide",
    )
    inject_custom_styles()
    init_session_state()

    db_stats = get_db_stats()
    selected_model, is_reasoning, debug_mode, fast_mode = render_sidebar(db_stats)
    render_header_and_metrics(
        db_stats=db_stats,
        selected_model=selected_model,
        debug_mode=debug_mode,
        is_reasoning=is_reasoning,
        fast_mode=fast_mode,
    )

    # Query Input Form
    with st.form(key="query_form", clear_on_submit=False):
        col_input, col_submit = st.columns([5, 1])
        user_input = col_input.text_input(
            "Ask a business question about customers, sales, orders, or products:",
            value=st.session_state.get("user_query", ""),
            placeholder="e.g. Who are the top 5 customers by total spending?",
            label_visibility="collapsed",
        )
        submit_clicked = col_submit.form_submit_button(
            "Run Query", type="primary", use_container_width=True
        )

    # Handle execution trigger from form submit or quick example button
    auto_submitted = st.session_state.get("auto_submit", False)
    if auto_submitted:
        st.session_state["auto_submit"] = False

    should_run = (submit_clicked or auto_submitted) and bool(user_input.strip())

    if should_run:
        query_text = user_input.strip()

        if debug_mode:
            status_box = st.status(
                f"**Agent Pipeline Trace** (`{selected_model}`)", expanded=True
            )
            st.subheader("Synthesized Answer")
            response_placeholder = st.empty()

            _, trace_log, sql_query, final_data = agent_pipeline(
                user_prompt=query_text,
                model_name=selected_model,
                debug_mode=True,
                fast_mode=fast_mode,
                status_container=status_box,
                response_placeholder=response_placeholder,
            )
            status_box.update(
                label=f"**Agent Finished Processing** (`{selected_model}`)",
                state="complete",
                expanded=False,
            )
        else:
            with st.spinner(f"Agent retrieving and verifying data with `{selected_model}`..."):
                st.subheader("Synthesized Answer")
                response_placeholder = st.empty()

                _, trace_log, sql_query, final_data = agent_pipeline(
                    user_prompt=query_text,
                    model_name=selected_model,
                    debug_mode=False,
                    fast_mode=fast_mode,
                    status_container=None,
                    response_placeholder=response_placeholder,
                )

        render_results_inspector(
            final_data=final_data,
            sql_query=sql_query,
            trace_log=trace_log,
        )


if __name__ == "__main__":
    main()
