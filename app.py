import sqlite3
import time

import ollama
import pandas as pd
import streamlit as st

DB_PATH = "company.db"
MODEL_NAME = "qwen3.5:9b"

# 1. Fetch DB Schema for Prompt Context
def get_db_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    schema = "\n".join([row[0] for row in cursor.fetchall() if row[0]])
    conn.close()
    return schema

SCHEMA = get_db_schema()

# Fetch table previews for sidebar inspection
def get_table_previews():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]
    previews = {}
    for table in tables:
        cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        previews[table] = pd.DataFrame(rows, columns=columns)
    conn.close()
    return previews

# 2. Database Tool Execution
def run_sql_query(query: str):
    """Executes a SQL query in read-only mode."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        return {"success": True, "columns": columns, "data": rows}
    except Exception as e:
        conn.close()
        return {"success": False, "error": str(e)}

def extract_sql_query(content: str, thinking: str = "") -> str:
    """Extracts clean SQL query from LLM response or thinking fallback."""
    target = content.strip()
    if not target and thinking:
        target = thinking

    if "```sql" in target:
        sql = target.split("```sql", 1)[1].split("```", 1)[0].strip()
    elif "```" in target:
        sql = target.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        lines = target.splitlines()
        sql_lines = []
        started = False
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith(("SELECT", "WITH")):
                started = True
            if started:
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip() if sql_lines else target.strip()

    return sql.rstrip(";").strip()

# 3. Multi-Step Agentic Loop (Generate -> Execute -> Auto-Heal -> Synthesize)
def agent_pipeline(
    user_prompt: str,
    debug_mode: bool = False,
    status_container = None,
    response_placeholder = None
):
    trace = []

    gen_system_prompt = f"""You are an SQL generator for SQLite.
Database Schema:
{SCHEMA}

Write ONLY a valid SQLite SELECT query to answer the user's request.
Guidelines:
1. Fuzzy String Matching: Users rarely provide exact product names. Always use `LIKE '%keyword%'` with `LOWER()` on text fields (e.g., `LOWER(name) LIKE '%industrial pump%'`) instead of strict `=` comparisons.
2. If multiple keywords are given, match against individual core tokens rather than demanding the entire phrase verbatim.
3. Wrap your query in ```sql ... ``` code blocks. Do not add conversational text."""

    messages = [
        {"role": "system", "content": gen_system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    max_retries = 3
    final_data = None
    successful_sql = None

    for attempt in range(1, max_retries + 1):
        attempt_header = f"Attempt {attempt}"
        if attempt > 1:
            trace.append(f"**{attempt_header} Auto-Healing:** Re-prompting model with previous database error...")
            if debug_mode and status_container:
                status_container.markdown(f"#### 🩹 {attempt_header}: Autonomous Self-Correction")
        else:
            if debug_mode and status_container:
                status_container.markdown(f"#### 💭 {attempt_header}: Formulating SQL Query")

        # Streamlit containers for streaming thoughts and SQL
        thought_box = None
        if debug_mode and status_container:
            with status_container.expander(f"Agent Internal Thought Process ({attempt_header})", expanded=True):
                st.caption("🧠 Real-time LLM Internal Monologue & Reasoning Strategy")
                thought_box = st.empty()

        accumulated_sql_thinking = ""
        accumulated_sql_content = ""
        last_th_update = time.time()

        # Stream SQL generation
        chat_stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            options={
                "temperature": 0.0,
            }
        )

        for chunk in chat_stream:
            msg = chunk.message
            th = getattr(msg, "thinking", None)
            ct = getattr(msg, "content", None)

            if th:
                accumulated_sql_thinking += th
                now = time.time()
                if debug_mode and thought_box and (now - last_th_update >= 0.04):
                    thought_box.markdown(accumulated_sql_thinking + "▌")
                    last_th_update = now

            if ct:
                if accumulated_sql_content == "" and debug_mode and thought_box and accumulated_sql_thinking:
                    thought_box.markdown(accumulated_sql_thinking)
                accumulated_sql_content += ct

        # Finalize thought box display
        if debug_mode and thought_box and accumulated_sql_thinking:
            thought_box.markdown(accumulated_sql_thinking)

        # Extract SQL query
        sql_query = extract_sql_query(accumulated_sql_content, accumulated_sql_thinking)
        trace.append(f"**{attempt_header} Generated SQL:** `{sql_query}`")

        if debug_mode and status_container:
            status_container.markdown(f"**Generated SQL Query ({attempt_header}):**")
            status_container.code(sql_query, language="sql")

        # Execute query against SQLite
        db_result = run_sql_query(sql_query)

        if db_result["success"]:
            successful_sql = sql_query
            final_data = db_result
            row_count = len(db_result["data"])
            trace.append(f"Database query executed successfully ({row_count} row(s) returned).")

            if debug_mode and status_container:
                status_container.success(f"✅ SQL Query executed successfully ({row_count} row(s) returned)")
                if row_count > 0:
                    df = pd.DataFrame(db_result["data"], columns=db_result["columns"])
                    status_container.dataframe(df, use_container_width=True)
                else:
                    status_container.info("Query returned 0 matching records.")
            break
        else:
            # Self-healing agentic feedback loop
            error_msg = db_result["error"]
            trace.append(f"Execution failed: `{error_msg}`. Agent triggering auto-correction...")

            if debug_mode and status_container:
                status_container.error(f"❌ SQL Execution failed: `{error_msg}`")
                if attempt < max_retries:
                    status_container.warning(f"🩹 Triggering auto-healing feedback loop for Attempt {attempt + 1}...")

            messages.append({"role": "assistant", "content": accumulated_sql_content or f"```sql\n{sql_query}\n```"})
            messages.append({"role": "user", "content": f"The query failed with error: {error_msg}. Please fix the SQL query according to the database schema."})

    if not final_data or not final_data.get("success"):
        fail_msg = "Agent could not resolve the database query after retries."
        if response_placeholder:
            response_placeholder.error(fail_msg)
        return fail_msg, trace

    # 4. Final Response Synthesis with Citations
    synthesis_prompt = f"""Synthesize the answer for the user based strictly on the retrieved database records.
User Question: {user_prompt}
Executed Query: {successful_sql}
Retrieved Data: {final_data}

Rules:
1. Provide a natural language answer.
2. Include explicit citations indicating which table, row ID, or columns provided the data (e.g., [Source: products table, product_id=1])."""

    synth_thought_box = None
    if debug_mode and status_container:
        status_container.markdown("#### 🧠 Step 2: Synthesizing Answer & Formulating Citations")
        with status_container.expander("Agent Synthesis Thought Process", expanded=True):
            st.caption("🧠 Real-time Reasoning over DB Records & Citation Formulation")
            synth_thought_box = st.empty()

    accumulated_synth_thinking = ""
    accumulated_synth_content = ""
    last_synth_th_update = time.time()
    last_synth_ct_update = time.time()

    synth_stream = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": synthesis_prompt}],
        stream=True,
        options={
            "temperature": 0.2
        }
    )

    for chunk in synth_stream:
        msg = chunk.message
        th = getattr(msg, "thinking", None)
        ct = getattr(msg, "content", None)

        if th:
            accumulated_synth_thinking += th
            now = time.time()
            if debug_mode and synth_thought_box and (now - last_synth_th_update >= 0.04):
                synth_thought_box.markdown(accumulated_synth_thinking + "▌")
                last_synth_th_update = now

        if ct:
            if accumulated_synth_content == "" and debug_mode and synth_thought_box and accumulated_synth_thinking:
                synth_thought_box.markdown(accumulated_synth_thinking)
            accumulated_synth_content += ct
            now = time.time()
            if response_placeholder and (now - last_synth_ct_update >= 0.04):
                response_placeholder.markdown(accumulated_synth_content + "▌")
                last_synth_ct_update = now

    if debug_mode and synth_thought_box and accumulated_synth_thinking:
        synth_thought_box.markdown(accumulated_synth_thinking)

    if response_placeholder and accumulated_synth_content:
        response_placeholder.markdown(accumulated_synth_content)

    return accumulated_synth_content, trace


# --- Streamlit UI ---
st.set_page_config(page_title="Live Database Assistant", page_icon="🤖", layout="centered")

# --- Initialize Session State ---
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Assistant Settings")
    debug_mode = st.toggle(
        "🐞 Debug Mode",
        value=True,
        help="Stream the agent's internal thought process, generated SQL, database query execution, and auto-healing in real time."
    )

    st.caption(f"LLM: `{MODEL_NAME}`")
    st.divider()

    st.subheader("📊 Database Preview")
    st.success("Connected to `company.db`")
    with st.expander("Inspect Schema & Tables"):
        previews = get_table_previews()
        for table_name, df in previews.items():
            st.markdown(f"**Table: `{table_name}`**")
            st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("💡 Example Queries")
    examples = [
        "What is the price of the Industrial Pump and who has active quotations for it?",
        "Which orders are currently in Processing status?",
        "List all products with stock below 10 units.",
        "What is the quoted price for Satakunta Tech Ab and what product is it for?"
    ]

    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["user_query"] = ex
            st.rerun()

# --- Main Interface ---
st.title("🤖 Live Database Assistant")
st.caption("Local Text-to-SQL with Autonomous Agentic Error Correction & Real-Time Thought Streaming")

if debug_mode:
    st.info("🐞 **Debug Mode Active:** The agent's step-by-step reasoning monologue, SQL queries, and DB results will stream live below.", icon="ℹ️")

# Query input using session state
user_input = st.text_input(
    "Ask a question about orders, products, or quotes:",
    key="user_query",
    placeholder="What is the price of the Industrial Pump and who has active quotations for it?"
)

col_btn, _ = st.columns([1, 4])
submit_clicked = col_btn.button("Submit", type="primary")

if submit_clicked and user_input.strip():
    st.markdown("---")
    trace_log = []

    if debug_mode:
        status_box = st.status("🤖 **Agent Reasoning & Execution Trace**", expanded=True)

        st.subheader("Response")
        response_placeholder = st.empty()

        answer, trace_log = agent_pipeline(
            user_prompt=user_input.strip(),
            debug_mode=True,
            status_container=status_box,
            response_placeholder=response_placeholder
        )
        status_box.update(label="✅ **Agent Finished Processing**", state="complete", expanded=True)
    else:
        with st.spinner("Agent retrieving and verifying data..."):
            st.subheader("Response")
            response_placeholder = st.empty()

            answer, trace_log = agent_pipeline(
                user_prompt=user_input.strip(),
                debug_mode=False,
                status_container=None,
                response_placeholder=response_placeholder
            )

    # Collapsible Trace for inspection
    with st.expander("Agent Reasoning & Citation Trace (Click to inspect)"):
        for step in trace_log:
            st.markdown(f"- {step}")
