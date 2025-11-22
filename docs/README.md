# VoxBank Documentation

This directory contains architecture diagrams, data model visuals, and submission materials for VoxBank.

The intention is that a new reader can open this folder in GitHub (or locally) and quickly understand how the system fits together.

---

## 1. Files

- `architecture-diagram.png`  
  High‑level system architecture: frontend ↔ orchestrator ↔ MCP tools ↔ mock‑bank, plus supporting services.

- `sequence-diagram.png`  
  End‑to‑end sequence for a typical **voice interaction**, including STT, LLM decisions, MCP tool calls, and TTS.

- `data-model.png`  
  Database entity‑relationship diagram for mock‑bank:
  - `users`, `accounts`, `transactions`, `beneficiaries`, `cards`, `loans`, `reminders`.

- `mcp-flow.png`  
  MCP tool execution flow:
  - Orchestrator decision loop → MCP client → MCP HTTP server → mock‑bank REST → response back into LLM.

- `security-model.png`  
  High‑level security and compliance model:
  - Auth gating, tool whitelisting, separation of concerns, and data minimization principles.

- `round2-submission.pdf`  
  Final submission PDF for the competition/evaluation (contains a curated explanation of design decisions).

- `summary.txt`  
  Short textual summary of the project and its goals.

---

## 2. How to Use These Diagrams

### Architecture Diagram

Use `architecture-diagram.png` together with the root `README.md` to:

- Explain the four main services (frontend, orchestrator, mcp-tools, mock‑bank).
- Walk through data flows for:
  - Text chat
  - Voice chat (Whisper STT + multi‑engine TTS)
  - High‑risk operations like transfers

This is ideal for an overview slide or whiteboard session.

### Sequence Diagram

The sequence diagram is especially useful when describing:

- How audio leaves the browser as PCM chunks.
- Where Whisper STT runs.
- When the LLM is called and when MCP tools are invoked.
- How TTS audio comes back to the browser.

Pair it with `orchestrator/README.md` and `frontend/README.md` for a deep dive into voice.

### Data Model Diagram

`data-model.png` is the quickest way to grasp:

- How users, accounts, transactions, beneficiaries, cards, loans, and reminders relate.
- Where fields like `passphrase` and `audio_embedding` live.

Use it alongside `mock-bank/README.md` and `data/schema.sql` when reasoning about new endpoints or tools.

### MCP Flow & Security Model

`mcp-flow.png` shows the boundary between:

- The LLM agent (decision making) and
- The MCP tool layer (side‑effectful operations).

`security-model.png` documents:

- Why tools are whitelisted.
- Why auth is deterministic.
- Why the LLM never talks to the database directly.

These diagrams are useful for architecture reviews and security/compliance discussions.

---

## 3. Editing / Updating Diagrams

The current diagrams were created outside the repo, but you can maintain them with tools such as:

- [diagrams.net / draw.io](https://www.diagrams.net/)
- Lucidchart, Miro, Figma
- Mermaid (for code‑based diagrams embedded in Markdown)

When updating:

1. Keep the PNG filenames stable so existing links from `README.md` continue to work.
2. Prefer light backgrounds and clear labels for readability in GitHub’s dark/light themes.
3. If you add new diagrams (e.g., “Whisper STT pipeline”), reference them from the relevant README.

