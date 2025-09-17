# frontend/voice_support.py
import os, asyncio, base64
from typing import List, Dict, Any, Optional

from google.cloud import speech_v2, texttospeech

# --- STT (Google Speech v2) ---
def stt_transcribe_bytes(raw: bytes, language: str = "en-US") -> str:
    """
    Transcribes arbitrary audio bytes using Speech-to-Text v2 with auto decoding.
    Requires ADC (gcloud auth application-default login) and project configured.
    """
    client = speech_v2.SpeechClient()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        # Speech v2 lib can infer project, but we prefer explicit
        project = client.project

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=[language],
        features=speech_v2.RecognitionFeatures(enable_automatic_punctuation=True),
        model="long",
    )
    req = speech_v2.RecognizeRequest(
        recognizer=f"projects/{project}/locations/global/recognizers/_",
        config=config,
        content=raw,
    )
    resp = client.recognize(request=req)
    if not resp.results:
        return ""
    return resp.results[0].alternatives[0].transcript.strip()

# --- TTS (Google Text-to-Speech) ---
def tts_mp3_bytes(text: str, voice_name: str = "en-US-Neural2-C") -> bytes:
    """
    Synthesizes MP3 audio from text.
    """
    tts = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    audio = tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_cfg)
    return audio.audio_content

# --- Agent invocation (HTTP or in-process) ---
def invoke_agent_http(messages: List[Dict[str, Any]], base_url: str = "http://127.0.0.1:8000", app_name: str = "app") -> str:
    """
    Calls ADK API server. Start it with:
      uv run adk api_server app --port 8000
    Returns the assistant's final text.
    """
    import requests
    r = requests.post(f"{base_url}/api/invoke", params={"app_name": app_name}, json={"messages": messages}, timeout=120)
    r.raise_for_status()
    data = r.json()
    # ADK responses differ by version; extract the last assistant content robustly:
    txt = ""
    if isinstance(data, dict):
        # common shape: {"messages":[...]}
        msgs = data.get("messages") or []
        for m in reversed(msgs):
            if m.get("role") == "assistant" and m.get("content"):
                txt = m["content"]
                break
        # fallback: event stream
        if not txt and "events" in data:
            for ev in reversed(data["events"]):
                if ev.get("role") == "assistant" and ev.get("content"):
                    txt = ev["content"]
                    break
    return txt or "Sorry, I couldn’t generate a response."

async def _invoke_agent_inproc_async(messages: List[Dict[str, Any]]) -> str:
    # In-process: import your ADK agent directly
    from app.agent import root_agent
    final = ""
    async for ev in root_agent.run_async({"messages": messages}):
        content = getattr(ev, "content", None)
        role = getattr(ev, "role", None)
        if role == "assistant" and content:
            final = content
    return final or "Sorry, I couldn’t generate a response."

def invoke_agent_inproc(messages: List[Dict[str, Any]]) -> str:
    return asyncio.run(_invoke_agent_inproc_async(messages))
