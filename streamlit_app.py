"""Streamlit chat UI for the customer-support agent.

Typed chat and dictated chat share one path: speech is transcribed to text, the
customer confirms it, and the text is sent through the ordinary ``POST /chat``
request. Voice never bypasses the session, RAG, or ticket workflow, and the UI
stays fully usable when the voice service is unavailable.
"""

import os
import uuid

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

CHAT_TIMEOUT = httpx.Timeout(35.0, connect=5.0)
TRANSCRIBE_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
SYNTHESIZE_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


st.set_page_config(
    page_title="Customer Support",
    page_icon="🎧",
    layout="centered",
)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

DEFAULT_STATE = {
    "session_id": None,
    "messages": [],
    # message_id -> (audio bytes, media type). Reused for the rest of the
    # session so a second click on the same speaker icon replays instead of
    # re-synthesizing.
    "audio_cache": {},
    "audio_errors": {},
    # message_id currently being synthesized; disables every speaker icon.
    "synthesizing": None,
    # Transcript awaiting review. Voice input is never auto-sent.
    "pending_transcript": None,
    "last_recording": None,
    "chat_error": None,
    "voice_error": None,
}


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value() if callable(value) else value

    if st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())


def reset_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.audio_cache = {}
    st.session_state.audio_errors = {}
    st.session_state.synthesizing = None
    st.session_state.pending_transcript = None
    st.session_state.last_recording = None
    st.session_state.chat_error = None
    st.session_state.voice_error = None


init_state()


# --------------------------------------------------------------------------
# Backend calls
# --------------------------------------------------------------------------

def backend_status() -> tuple[str, str, dict]:
    """Probe the API so an unreachable backend is visible before the first message."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=httpx.Timeout(5.0, connect=2.0))
    except httpx.HTTPError:
        return "down", f"Backend unreachable at {API_BASE_URL}. Start FastAPI, then reload.", {}

    if response.status_code == 200:
        body = response.json()

        # A health payload with no "voice" key at all means the API process is
        # running a build from before the voice endpoints existed -- almost
        # always a server left running across the change. Say so precisely
        # rather than reporting a generic outage.
        voice = body.get("voice")

        if voice is None:
            voice = {
                "available": False,
                "detail": (
                    "The API is running an older build with no voice endpoints. "
                    "Restart FastAPI to enable speech."
                ),
            }

        return (
            "ready",
            f"Backend ready - model {body.get('model', 'unknown')}",
            voice,
        )

    if response.status_code == 503:
        return "starting", "Backend is starting: the knowledge index is still being built.", {}

    return "down", f"Backend returned HTTP {response.status_code}.", {}


def error_detail(response: httpx.Response, fallback: str) -> str:
    """Read an error message from either error shape the API uses."""
    try:
        body = response.json()
    except ValueError:
        return fallback

    if isinstance(body, dict):
        # Voice endpoints use {"success": false, "error": ...};
        # the rest of the API uses FastAPI's {"detail": ...}.
        for key in ("error", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"]

    return fallback


def send_message(text: str) -> None:
    """Send one customer turn through the existing chat workflow.

    Typed input and confirmed transcripts both arrive here, so the agent cannot
    tell them apart.
    """
    text = text.strip()
    if not text:
        return

    st.session_state.chat_error = None
    st.session_state.messages.append(
        {"id": str(uuid.uuid4()), "role": "user", "content": text}
    )

    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": text},
            timeout=CHAT_TIMEOUT,
        )
    except httpx.ConnectError:
        st.session_state.chat_error = (
            "The backend is unavailable. Start FastAPI and try again."
        )
        return
    except httpx.TimeoutException:
        st.session_state.chat_error = (
            "The backend took too long to respond. Please try again."
        )
        return
    except httpx.HTTPError:
        st.session_state.chat_error = (
            "A network error occurred while contacting the backend."
        )
        return

    if response.status_code >= 400:
        st.session_state.chat_error = error_detail(
            response, "The backend returned an error."
        )
        return

    try:
        data = response.json()
    except ValueError:
        st.session_state.chat_error = "The backend returned an invalid response."
        return

    answer = str(data.get("response", "")).strip()

    if not answer:
        st.session_state.chat_error = "The agent returned an empty response."
        return

    st.session_state.messages.append(
        {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer,
            "sources": data.get("sources") or [],
            "ticket_id": data.get("ticket_id"),
        }
    )


def transcribe(audio_bytes: bytes) -> None:
    """Turn a recording into a transcript awaiting the customer's confirmation."""
    st.session_state.voice_error = None

    try:
        response = httpx.post(
            f"{API_BASE_URL}/voice/transcribe",
            files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
            timeout=TRANSCRIBE_TIMEOUT,
        )
    except httpx.HTTPError:
        st.session_state.voice_error = (
            "Could not reach the transcription service. You can still type your message."
        )
        return

    if response.status_code >= 400:
        st.session_state.voice_error = error_detail(
            response, "Transcription failed. You can still type your message."
        )
        return

    try:
        transcript = str(response.json().get("transcript", "")).strip()
    except ValueError:
        st.session_state.voice_error = "The transcription service returned an invalid response."
        return

    if not transcript:
        st.session_state.voice_error = (
            "No understandable speech was detected. Try recording again."
        )
        return

    st.session_state.pending_transcript = transcript


def synthesize(message_id: str, text: str) -> None:
    """Fetch audio for one agent response and cache it against its message ID."""
    st.session_state.audio_errors.pop(message_id, None)

    try:
        response = httpx.post(
            f"{API_BASE_URL}/voice/synthesize",
            json={"message_id": message_id, "text": text},
            timeout=SYNTHESIZE_TIMEOUT,
        )
    except httpx.HTTPError:
        st.session_state.audio_errors[message_id] = (
            "Could not reach the speech service. The reply above is unchanged."
        )
        return

    if response.status_code >= 400:
        st.session_state.audio_errors[message_id] = error_detail(
            response, "Playback is unavailable right now. The reply above is unchanged."
        )
        return

    if not response.content:
        st.session_state.audio_errors[message_id] = "The speech service returned no audio."
        return

    # The API echoes the message ID back, so audio can only ever be attached to
    # the response whose speaker icon was clicked.
    returned_id = response.headers.get("X-Message-Id", message_id)

    if returned_id != message_id:
        st.session_state.audio_errors[message_id] = (
            "Received audio for a different message; playback was discarded."
        )
        return

    st.session_state.audio_cache[message_id] = (
        response.content,
        response.headers.get("content-type", "audio/mpeg"),
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_assistant_extras(message: dict, voice_ready: bool) -> None:
    message_id = message["id"]

    if message.get("sources"):
        st.caption("Sources: " + ", ".join(str(s) for s in message["sources"]))

    if message.get("ticket_id"):
        st.success(f"Ticket created: {message['ticket_id']}")

    busy = st.session_state.synthesizing
    cached = message_id in st.session_state.audio_cache

    controls = st.columns([1, 6])

    with controls[0]:
        clicked = st.button(
            "🔊",
            key=f"speak-{message_id}",
            help=(
                "Play this response"
                if voice_ready
                else "Voice playback is unavailable"
            ),
            # Repeated clicks are refused while any synthesis is in progress.
            disabled=(not voice_ready) or (busy is not None),
        )

    with controls[1]:
        if busy == message_id:
            st.caption("Generating audio…")

    if clicked and not cached:
        st.session_state.synthesizing = message_id
        st.rerun()

    if cached:
        audio_bytes, media_type = st.session_state.audio_cache[message_id]
        st.audio(audio_bytes, format=media_type)

    if message_id in st.session_state.audio_errors:
        # The text response and any ticket details stay on screen regardless.
        st.warning(st.session_state.audio_errors[message_id])


def render_history(voice_ready: bool) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                render_assistant_extras(message, voice_ready)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.title("🎧 Customer Support")
st.caption(
    "AI-powered support assistant with grounded knowledge retrieval and ticket creation."
)

state, detail, voice_status = backend_status()
voice_ready = bool(voice_status.get("available"))

with st.sidebar:
    st.subheader("Conversation")
    st.caption(f"Session: {st.session_state.session_id[:8]}…")

    if st.button("New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()
    st.subheader("Backend")

    if state == "ready":
        st.success(detail)
    elif state == "starting":
        st.warning(detail)
    else:
        st.error(detail)

    st.subheader("Voice")

    if voice_ready:
        st.success(
            f"Speech ready - {voice_status.get('stt_model', '?')} / "
            f"{voice_status.get('tts_voice', '?')}"
        )
    else:
        st.info(
            voice_status.get("detail")
            or "Voice is unavailable. Typed chat works normally."
        )

# Deferred synthesis: the click above set this and rerendered with every
# speaker icon disabled, so the work happens with the UI already in its busy
# state rather than blocking mid-click.
if st.session_state.synthesizing:
    pending_id = st.session_state.synthesizing
    target = next(
        (m for m in st.session_state.messages if m["id"] == pending_id),
        None,
    )

    if target is not None:
        synthesize(pending_id, target["content"])

    st.session_state.synthesizing = None
    st.rerun()

render_history(voice_ready)

if st.session_state.chat_error:
    st.error(st.session_state.chat_error)

if st.session_state.voice_error:
    st.warning(st.session_state.voice_error)

# --- Voice input -----------------------------------------------------------

if voice_ready:
    with st.expander("🎙️ Speak instead of typing", expanded=False):
        recording = st.audio_input(
            "Record your message",
            key="microphone",
            disabled=st.session_state.pending_transcript is not None,
        )

        if recording is not None and st.session_state.pending_transcript is None:
            audio_bytes = recording.getvalue()
            signature = (len(audio_bytes), hash(audio_bytes))

            # Streamlit reruns on every interaction; only transcribe a recording
            # the first time it is seen.
            if signature != st.session_state.last_recording:
                st.session_state.last_recording = signature

                if not audio_bytes:
                    st.session_state.voice_error = "The recording was empty."
                else:
                    with st.spinner("Transcribing…"):
                        transcribe(audio_bytes)

                st.rerun()

# Editable transcript: nothing is sent to the agent until the customer confirms.
if st.session_state.pending_transcript is not None:
    with st.form("transcript_review", clear_on_submit=False):
        st.caption("Review the transcript, edit if needed, then send.")

        edited = st.text_area(
            "Your message",
            value=st.session_state.pending_transcript,
            key="transcript_editor",
            height=100,
        )

        actions = st.columns(2)
        send_clicked = actions[0].form_submit_button(
            "Send to agent", use_container_width=True, type="primary"
        )
        discard_clicked = actions[1].form_submit_button(
            "Discard", use_container_width=True
        )

    if send_clicked:
        if edited.strip():
            st.session_state.pending_transcript = None
            with st.spinner("Thinking…"):
                send_message(edited)
            st.rerun()
        else:
            st.warning("The message is empty. Edit it or discard the recording.")

    if discard_clicked:
        st.session_state.pending_transcript = None
        st.session_state.voice_error = None
        st.rerun()

# --- Typed input -----------------------------------------------------------

if prompt := st.chat_input(
    "Ask about shipping, returns, payments, accounts, or report an issue"
):
    with st.spinner("Thinking…"):
        send_message(prompt)
    st.rerun()
