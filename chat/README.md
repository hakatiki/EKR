# ProcureGraph Agent

An interactive chat application powered by FastAPI, LangChain, and LangGraph. The backend orchestrates an OpenAI Responses API agent that can switch between normal “ChatGPT-style” conversation and a structured procurement investigation workflow with automated planning, web search, and vector-store document lookup.

## Setup

1. Create `.env` inside `chat/` (the file is not committed). It should define at least:
   ```
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4.1-mini
   ```
   Optional:
   ```
   OPENAI_PROCUREMENT_VECTOR_STORE_ID=vs_...
   CORS_ALLOW_ORIGINS=http://localhost:5173,https://example.com
   ```
   The vector store identifier enables the built-in `file_search` tool for uploaded procurement documents.

2. Install dependencies and start the backend:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On PowerShell
   pip install -r chat/requirements.txt
   uvicorn chat.app.main:app --reload
   ```

3. Open the frontend in the browser at http://127.0.0.1:8000/ (served by FastAPI).

Environment variables:

- `OPENAI_API_KEY` – required. The backend now checks this variable on startup and
  exits with a clear error message if it is missing.
- `OPENAI_MODEL` – optional; defaults to `gpt-4.1-mini`.
- `OPENAI_PROCUREMENT_VECTOR_STORE_ID` – optional OpenAI vector store ID for ingesting procurement documents the agent can search via `file_search`.
- `CORS_ALLOW_ORIGINS` – comma-separated origins allowed for cross-origin requests, default `*`.

## Agent behavior

- **Normal mode:** behaves like ChatGPT with access to OpenAI web search preview and (optionally) file search.
- **Procurement investigations:** when prompted with requests such as “Investigate ACME Corp’s procurement,” the agent:
  1. Routes the request into investigation mode.
  2. Plans a TODO list of research steps.
  3. Executes the plan using web and document search.
  4. Returns a GitHub-flavored Markdown report with summary plus tables for company overview, ownership, procurement/tenders, competitors, and risk indicators.

## Frontend

- The chat UI lives in `chat/static/index.html` with styles in `chat/static/styles.css` and logic in `chat/static/app.js`.
- Messages stream in as text; once complete, any Markdown tables are rendered into tabbed “mini sheets” on the right panel.
- Tabs persist across the session so you can revisit previously generated tables without losing the chat flow.

