# Customer Support Ticket Agent

A customer-support agent that answers policy questions from a supplied knowledge
base and raises support tickets through a mock ticket API. Built on FastAPI,
Streamlit, LangGraph, ChromaDB, and a locally served open-source model. Questions
can be typed or dictated, and any reply can be played back as speech.

The agent does two things and is careful about the boundary between them:

- **Answers policy questions** only from the four Markdown documents in
  `knowledge_base/`, and names the documents it used. When retrieval finds
  nothing on topic it says so instead of guessing.
- **Raises a ticket** when an issue is unresolved, collecting name, email, issue
  description, and category across as many turns as it takes, validating each
  value, and never inventing one.

A voice layer sits *around* that agent rather than inside it: speech is
transcribed to text, the customer confirms the transcript, and the text goes
through the same `POST /chat` request a typed message uses. See
[Voice](#voice-speech-in-speech-out).

## Architecture

```text
   Customer message  (typed, or spoken → STT → confirmed transcript)
          |
          v
   Streamlit UI  ──HTTP──▶  FastAPI  ──▶  SupportPipeline
  (streamlit_app.py)       (src/api/    (src/pipeline.py)
                            server.py)        |
                                              v
                                      LangGraph workflow
                                    (src/llm/workflow.py)
                                              |
                        ┌─────────────────────┴─────────────────────┐
                        v                                           v
                   [retrieve]  ──▶  [decide]  ──conditional edge──▶ [answer]
                        |               |                           [ticket]
                        v               v                               |
                 KnowledgeRetriever   ChatOpenAI                        v
                 (Chroma + MiniLM)    (structured output)      create_support_ticket
                        |                                      (src/tools/ticket_tool.py)
                        v                                               |
                 knowledge_base/*.md                            TicketRepository
                                                                        |
                        └───────────────────┬───────────────────────────┘
                                            v
                            ChatResponse {response, sources, ticket_id}
```

The voice layer is a separate branch off the same API, deliberately not wired
into the agent graph:

```text
   microphone ──▶ POST /voice/transcribe ──▶ VoicePipeline ──▶ WhisperSTTService
                                                                    |
                          editable transcript ◀─────────────────────┘
                                   |
                          (customer confirms)
                                   v
                            POST /chat  ← identical to typed input

   agent reply ──🔊 click──▶ POST /voice/synthesize ──▶ VoicePipeline ──▶ EdgeTTSService
                                                                    |
                          audio played under that reply ◀───────────┘
```

### How one turn flows

1. **`retrieve`** searches Chroma with the customer's message and returns
   `{content, source}` chunks. An empty result is meaningful, not an error.
2. **`decide`** asks the model for a structured `Decision`: a route (`answer` or
   `ticket`) plus any ticket fields the customer stated in this message.
3. A **conditional edge** dispatches to the node named by the route.
4. **`answer`** grounds a reply in the retrieved chunks, or refuses if there are
   none.
   **`ticket`** merges newly supplied fields into session state, asks for the
   first field still missing, and calls the ticket tool only once every field is
   present and valid.

### Adding a capability

Nodes are independent and the graph is assembled from a handler map, so a new
capability is three small changes and no rewiring:

1. Add the route name to `ROUTE_TARGETS` and to the `route` literal on `Decision`
   in `src/llm/workflow.py`.
2. Write the node function; return `response_text`, `sources`, and `ticket_id`.
3. Register it in the `handlers` map inside `build_support_workflow`.

`select_route` validates against `ROUTE_TARGETS`, and nodes and edges are added
by iterating that map, so nothing else needs to change.

## Technology choices

| Choice | What it is | Why |
| --- | --- | --- |
| **qwen2.5:3b** via **Ollama** | Open-source instruction-tuned LLM, served locally over an OpenAI-compatible API | Apache-2.0, runs on a laptop CPU/GPU with no API key or per-token cost, and — unlike most models this size — reliably supports the tool-calling that `with_structured_output` depends on for routing and field extraction. Any OpenAI-compatible endpoint works instead: change `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` and nothing else. |
| **all-MiniLM-L6-v2** | Sentence-transformer embeddings, 384 dimensions | Small and fast enough to embed the knowledge base at startup in a couple of seconds, downloads once, and runs locally with no external service. |
| **ChromaDB** | Persistent vector store | Embedded, no server to run, and persists to `VECTOR_DB_PATH` so restarts reuse the index. Configured with cosine distance so relevance scores are comparable across queries. |
| **LangGraph** | Agent orchestration | The workflow is genuinely branching — answer versus collect versus create — and LangGraph makes that routing explicit and testable as a graph, instead of hiding it in an opaque agent loop. |
| **FastAPI + Streamlit** | API and UI | Required by the assignment. |
| **faster-whisper** (STT) | Whisper inference via CTranslate2, running locally | No API key, no audio leaves the machine, and no system `ffmpeg` needed — its PyAV dependency decodes WAV, WebM, MP3, and OGG itself, which matters because browsers and desktop recorders disagree about container format. The `tiny` default transcribes a short sentence in about a second on CPU; raise `STT_MODEL` for accuracy. |
| **edge-tts** (TTS) | Microsoft Edge neural voices over their public endpoint | No API key and no local model to ship, and the voice quality is far better than an offline synthesizer like eSpeak. It returns MP3 bytes directly, which is exactly what `/voice/synthesize` serves. The tradeoff is that it needs outbound network access, so every failure path keeps the text reply on screen. |

Neither voice provider requires credentials, so there are no keys in this
submission. Both sit behind the supplied `STTService` / `TTSService` contracts,
so swapping in Vosk, Azure Speech, Coqui, or ElevenLabs means writing one class
and changing one line in `src/voice/service.py`.

No end-to-end conversational platform is used. Pipecat, LiveKit, Daily, Agora,
Twilio Voice, AssemblyAI Universal, Deepgram Aura, and the OpenAI Realtime API
are all absent; STT and TTS are integrated as individual components.

## Setup

Requires **Python 3.11 or newer**.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux / macOS

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### Serve the model

Install [Ollama](https://ollama.com/download) (Windows, macOS, and Linux
installers are all on that page), then pull the model:

```sh
ollama pull qwen2.5:3b
```

Ollama serves an OpenAI-compatible API on `http://localhost:11434/v1`, which is
the default in `.env.example`. Confirm it is running:

```sh
curl http://localhost:11434/api/tags
```

To use a hosted endpoint instead, point `LLM_BASE_URL` at it, set `LLM_API_KEY`,
and set `LLM_MODEL` to the served model name. No code changes are needed.

## Environment variables

Every variable has a working default; `.env.example` documents them all. Copy it
to `.env` and edit as needed. **`.env` is git-ignored — do not commit credentials.**

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible chat-completions endpoint. |
| `LLM_API_KEY` | `not-required` | Bearer token. Ollama ignores it; hosted endpoints need a real key. |
| `LLM_MODEL` | `qwen2.5:3b` | Model name as the endpoint knows it. |
| `API_BASE_URL` | `http://localhost:8000` | Where Streamlit looks for the FastAPI backend. |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | FastAPI bind address. |
| `STREAMLIT_HOST` / `STREAMLIT_PORT` | `127.0.0.1` / `8501` | Streamlit bind address. |
| `VECTOR_DB_PATH` | `.data/vector_db` | Chroma persistence directory. Git-ignored and rebuilt on demand. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model, downloaded on first run. |
| `RAG_COLLECTION` | `customer-support` | Chroma collection name. |
| `RAG_TOP_K` | `3` | Maximum chunks considered per query. |
| `VOICE_ENABLED` | `true` | Set `false` to run text-only. |
| `STT_MODEL` | `tiny` | faster-whisper size: `tiny`, `base`, `small`, `medium`, `large-v3`, or an `.en` variant. Downloaded on first use. |
| `STT_DEVICE` / `STT_COMPUTE_TYPE` | `cpu` / `int8` | Use `cuda` / `float16` on an NVIDIA GPU. |
| `TTS_VOICE` | `en-US-AriaNeural` | Any Edge neural voice, e.g. `en-GB-SoniaNeural`, `en-IN-NeerjaNeural`. |

## Run

Start the API. The first start downloads the embedding model and builds the
index, which takes roughly 15 seconds; later starts reuse it.

```sh
uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the UI:

```sh
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

- UI: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>

The UI sidebar shows backend and voice status, so a backend that is down, still
indexing, or running without speech is visible before you type anything.

On the very first run the Whisper model downloads (about 75 MB for `tiny`).
Allow the browser microphone permission when Streamlit asks for it.

### API

`GET /health` — component readiness.

```json
{
  "status": "ready",
  "components": {"retriever": true, "workflow": true, "knowledge_base": true},
  "model": "qwen2.5:3b"
}
```

Returns `503` with the same component map when any component is not ready. The
language model is reported as *configured*, never as *reachable*: verifying that
would mean calling the model on every health probe. A model outage surfaces as a
`502` from `/chat` instead.

`POST /chat` — process one customer message.

```sh
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-session-1","message":"How long does standard shipping take?"}'
```

```json
{
  "success": true,
  "session_id": "demo-session-1",
  "response": "Standard delivery usually takes three to five business days.",
  "sources": ["shipping.md"],
  "ticket_id": null
}
```

Status codes: `422` invalid request, `502` model or agent failure, `503`
pipeline not ready.

`GET /tickets/{ticket_id}` — retrieve a created ticket; `404` if unknown.

```json
{
  "ticket_id": "CST-2026-0001",
  "category": "payment",
  "summary": "Payment charged twice",
  "customer_name": "Asha Rao",
  "customer_email": "asha@example.com",
  "issue_description": "Payment was charged twice",
  "status": "open",
  "created_at": "2026-09-21T14:47:08.514253Z"
}
```

On Windows PowerShell, use `Invoke-RestMethod` or the `/docs` page instead of
`curl`.

## Voice: speech in, speech out

Voice is **additive**. The agent, its session state, RAG, and the ticket tool are
untouched by it: `POST /chat` has the same request and response it always had,
and `src/llm/workflow.py` contains no voice code at all. A dictated message
reaches the agent as an ordinary string, so the two inputs are indistinguishable
to it.

### Speaking to the agent

1. Open **🎙️ Speak instead of typing** below the chat and record with the
   browser microphone (`st.audio_input`).
2. The recording is posted to `POST /voice/transcribe` and a spinner shows while
   Whisper runs.
3. The transcript appears **in an editable text box**. Nothing has been sent yet.
4. Correct it if needed, then press **Send to agent** — or **Discard**.
5. Only on that explicit confirmation does the text go through `POST /chat`,
   exactly as if it had been typed.

Empty recordings, undecodable audio, and recordings with no intelligible speech
are each reported with a specific message, and typing always remains available.
Whisper runs with a voice-activity filter, so silence produces an empty
transcript and an honest error rather than invented words.

### Hearing a reply

Every agent response carries one 🔊 icon.

- Audio is generated **only when that icon is clicked** — nothing autoplays.
- The exact text displayed on screen is what gets synthesized.
- Audio is cached per message ID for the session, so a second click replays
  instead of re-synthesizing.
- While synthesis runs, every speaker icon is disabled and the message shows
  "Generating audio…".
- The API echoes the message ID back in an `X-Message-Id` header and the UI
  refuses audio that does not match, so a reply can never be voiced under the
  wrong message.
- If synthesis fails, a warning appears **under** the reply. The response text,
  its sources, and any ticket confirmation stay exactly where they were.

If the voice adapters cannot start — extras not installed, model unavailable,
`VOICE_ENABLED=false` — startup logs the reason, `/health` reports it, the
sidebar explains it, the microphone is hidden, and the text agent runs normally.
Voice failure is never allowed to take the agent down.

### Voice endpoints

`POST /voice/transcribe` — multipart upload, field name `audio`.

```sh
curl -X POST http://localhost:8000/voice/transcribe   -F "audio=@recording.wav;type=audio/wav"
```

```json
{"success": true, "transcript": "My payment was charged twice", "processing_time_ms": 820}
```

`POST /voice/synthesize` — JSON in, audio bytes out.

```sh
curl -X POST http://localhost:8000/voice/synthesize   -H 'Content-Type: application/json'   -d '{"message_id":"msg-1","text":"Ticket CST-2026-0001 has been created."}'   --output reply.mp3
```

Returns `audio/mpeg` with `X-Message-Id` and `X-Processing-Time-Ms` headers.

Both endpoints report failures as `{"success": false, "error": "..."}`:

| Status | Meaning |
| --- | --- |
| `400` | Empty audio, or no intelligible speech; blank synthesis text. |
| `415` | Audio format not supported, or the recording could not be decoded. |
| `422` | Request body failed validation. |
| `502` | The STT or TTS backend failed. |
| `503` | Voice support is unavailable or disabled. |

## Behavior worth knowing

**Grounding is enforced in code, not requested in a prompt.** Retrieval applies
two rules: an absolute cosine-relevance floor of `0.25`, and a relative margin
that drops any chunk scoring below 60% of the best hit for that query. The floor
decides whether the knowledge base has anything on topic at all; the margin keeps
source attribution precise, so a shipping question cites `shipping.md` alone
rather than every loosely related policy. Both live on `KnowledgeRetriever` in
`src/rag/retriever.py`.

When retrieval returns nothing, the `answer` node returns a fixed
"not in the knowledge base" reply **without calling the model at all**. This is
deliberate: asked "What is the capital of France?" with an empty context and a
system prompt forbidding ungrounded answers, qwen2.5:3b still replied "The capital
of France is Paris." A small instruction-tuned model cannot be trusted to refuse
on instruction alone, so the refusal is a code path with a test on it.

**Ticket fields are never invented.** Name, email, and issue description are used
only when the model reports the customer stated them. The category is accepted
only when the customer's own wording supports it — either a direct statement
("payment", "category: payment") or a category word in their message. A category
the model inferred from an issue that named none is discarded and the agent asks.
An address that fails validation is not stored, and the agent asks again rather
than writing a malformed email onto a ticket.

**One session, one ticket.** Duplicate protection holds at two levels:
`ConversationState.ticket_id` short-circuits a retried request with the original
ticket ID and a "already been created" reply, and `TicketRepository` maps each
session to one ticket so even a direct second tool call returns the first ticket.
If the ticket tool fails, no ID is written to the session, so nothing claims a
ticket exists when it does not.

## Tests

```sh
pytest -q
```

92 tests, roughly 18 seconds. **No language model, microphone, or network is
required** — `FakeChatModel` scripts the LLM and `ConfigurableFakeSTT` /
`ConfigurableFakeTTS` implement the supplied voice contracts, so routing,
grounding, ticket collection, transcription, synthesis, and every failure path
are deterministic and offline. Retrieval is *not* faked: tests run the real
embedding model against a temporary Chroma directory, which is what makes the
source-attribution assertions mean something.

| Requirement | Test |
| --- | --- |
| Policy question answered from RAG with sources | `test_workflow.py::test_policy_question_is_answered_from_the_knowledge_base`, `::test_each_topic_cites_its_own_document` |
| Unknown question handled without fabrication | `test_workflow.py::test_out_of_scope_question_is_refused_without_fabrication`, `::test_several_off_topic_questions_are_all_refused` |
| Multi-turn ticket creation | `test_workflow.py::test_multi_turn_conversation_creates_one_validated_ticket` |
| Retrieval of the created ticket | `test_api.py::test_a_ticket_created_in_chat_is_retrievable_by_id` |
| Missing-field validation | `test_workflow.py::test_missing_fields_produce_follow_up_questions_not_a_ticket`, `::test_a_category_the_customer_never_mentioned_is_not_accepted`, `::test_invalid_email_is_rejected_and_asked_for_again` |
| Duplicate ticket protection | `test_workflow.py::test_retrying_a_completed_request_returns_the_same_ticket`, `test_tickets.py::test_repository_prevents_duplicate_ticket_per_session` |
| Graceful model/backend failure | `test_workflow.py::test_model_failure_while_routing_is_reported_not_guessed`, `::test_model_failure_while_answering_is_reported_not_guessed`, `::test_a_failed_ticket_call_leaves_no_ticket_id_in_the_session`, `test_api.py::test_chat_returns_502_when_the_model_is_unavailable`, `::test_chat_returns_503_when_the_pipeline_is_not_ready` |

Voice requirements:

| Requirement | Test |
| --- | --- |
| Successful transcription with a fake STT adapter | `test_voice_api.py::test_transcription_returns_the_documented_payload`, `::test_transcription_passes_the_audio_and_media_type_through` |
| Empty-audio validation | `test_voice_api.py::test_empty_audio_is_rejected_with_a_clear_json_error`, `test_voice_pipeline.py::test_pipeline_rejects_empty_audio` |
| STT failure handling | `test_voice_api.py::test_stt_failure_returns_502_without_a_transcript`, `::test_unsupported_audio_format_is_reported_as_415`, `::test_unintelligible_audio_is_rejected_not_invented` |
| Successful synthesis with a fake TTS adapter | `test_voice_api.py::test_synthesis_returns_audio_with_the_correct_media_type`, `::test_synthesis_sends_the_exact_displayed_text` |
| TTS failure handling | `test_voice_api.py::test_tts_failure_returns_502_and_leaves_the_text_reply_alone`, `::test_empty_synthesis_audio_is_reported_not_served` |
| Voice-pipeline timing and media type | `test_voice_pipeline.py::test_pipeline_reports_timing_and_media_type`, `test_voice_api.py::test_synthesis_reports_processing_time` |
| Typed-chat regression after voice integration | `test_voice_api.py::test_typed_chat_is_unchanged_by_the_voice_integration`, `::test_voice_failure_does_not_affect_the_chat_endpoint`, `::test_a_transcript_reaches_the_agent_through_the_normal_chat_route` |
| Response-to-audio association | `test_voice_api.py::test_audio_is_associated_with_the_requesting_message` |

Useful subsets:

```sh
pytest tests/test_rag.py -q             # retrieval and relevance rules
pytest tests/test_workflow.py -q        # agent behavior end to end
pytest tests/test_api.py -q             # HTTP contract
pytest tests/test_voice_pipeline.py -q  # voice pipeline and adapters
pytest tests/test_voice_api.py -q       # voice endpoints and chat regression
```

The automated suite deliberately fakes the speech adapters. Real microphone
input and real playback are verified by hand during the demonstration, as the
assignment requires.

## Troubleshooting

**`/chat` returns 502.** The model endpoint is unreachable or the model name is
wrong. Check `curl http://localhost:11434/api/tags` and confirm `LLM_MODEL`
matches a name in that list.

**`/health` returns 503.** The index is still building, which takes about 15
seconds on a first start while the embedding model downloads. Watch the uvicorn
log and retry.

**Streamlit says the backend is unreachable.** The API is not running, or
`API_BASE_URL` does not match the port uvicorn bound. They must agree.

**The agent says a real policy question is not in the knowledge base.** The
question scored below the relevance floor. Ask using wording closer to the
documents, or lower `KnowledgeRetriever.RELEVANCE_THRESHOLD` in
`src/rag/retriever.py`. Lowering it too far lets off-topic questions retrieve
context, which is what the floor exists to prevent.

**Retrieval behaves oddly after changing `EMBEDDING_MODEL`.** The persisted index
holds vectors from the previous model. Delete `.data/vector_db` and restart; it
rebuilds automatically.

**`PydanticSerializationUnexpectedValue` warnings in the uvicorn log.** Emitted by
`langchain-openai` while serializing structured output. Harmless, and not raised
by this project's code.

**The microphone control is missing.** Voice did not start. The sidebar and
`GET /health` both name the reason: extras not installed, `VOICE_ENABLED=false`,
or the Whisper model failed to load. Typed chat is unaffected.

**The sidebar says the API is running an older build without voice endpoints.**
An API process started before the voice code was added is still serving. Confirm
with `curl http://localhost:8000/health` — a payload with no `voice` key is a
stale server — then restart uvicorn.

On Windows, stopping uvicorn does not always release the port: worker processes
it spawned can inherit the listening socket and keep answering on the old code.
If port 8000 stays busy after the server is gone, clear the orphans:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'multiprocessing.spawn' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

On Linux or macOS: `lsof -ti:8000 | xargs kill -9`.

**The browser never asks for microphone permission.** `st.audio_input` needs a
secure context. `localhost` counts as secure; a plain-HTTP LAN address does not.

**First transcription is slow.** The Whisper model downloads on first use
(about 75 MB for `tiny`). Later calls take roughly a second for a short sentence.

**Speaker icon shows "Could not reach the speech service".** `edge-tts` needs
outbound HTTPS. Check connectivity or a proxy. The text reply and ticket details
stay on screen either way — this is the designed failure mode.

**Transcripts are inaccurate.** `tiny` is the fastest model, not the best. Set
`STT_MODEL=base.en` or `small.en` for clearly better English accuracy at some
cost in speed.

**A model other than qwen2.5:3b routes badly or errors on every turn.** Routing
uses `with_structured_output`, which needs tool-calling support. Check the model
advertises it (`ollama show <model>` lists capabilities).

## Project layout

```text
src/
  api/server.py        FastAPI endpoints; thin transport boundary
  llm/workflow.py      LangGraph nodes, routing, and ticket collection
  llm/client.py        ChatOpenAI binding for the OpenAI-compatible endpoint
  llm/prompts.py       System prompt and answer template
  rag/retriever.py     Chroma init and relevance-aware search
  rag/document_loader.py, rag/embeddings.py
  sessions/store.py    Per-session conversation state
  tools/ticket_tool.py Mock ticket repository and validated LangChain tool
  voice/contracts.py   Supplied STTService / TTSService contracts
  voice/pipeline.py    Supplied VoicePipeline scaffold (timing and validation)
  voice/models.py      Supplied voice request/response models
  voice/stt_whisper.py faster-whisper adapter
  voice/tts_edge.py    edge-tts adapter
  voice/service.py     Optional voice layer; degrades without breaking the agent
  models.py            Pydantic contracts shared across every boundary
  pipeline.py          Wires model, retrieval, sessions, and tools together
  config.py            Environment-driven settings
streamlit_app.py       Chat UI
knowledge_base/*.md    Support policy documents
docs/                  Implementation notes and example voice payloads
tests/                 Unit and integration tests
```

The mid-session files (`contracts.py`, `pipeline.py`, `models.py`, the voice
test scaffold, and the example payloads) were merged into this project rather
than kept as a separate application: the contracts and scaffold live in
`src/voice/`, their test moved to `tests/test_voice_pipeline.py` with its
imports repointed, and `example_responses.json` is kept for reference at
`docs/voice_example_responses.json`. The supplied contract and pipeline code
itself is unmodified.
